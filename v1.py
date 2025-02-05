import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import time
import math
from tqdm import tqdm  # for progress bar

# Ensure GPU usage
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------------
# Noisy Linear Layer for exploration
# -------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# -------------------------------
# Prioritized Replay Buffer (simple version)
# -------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0

    def add(self, experience):
        # experience is a tuple: (state, action, reward, next_state, done, priority)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array([exp[-1] for exp in self.buffer], dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights).to(DEVICE)

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            exp = self.buffer[idx]
            self.buffer[idx] = exp[:-1] + (priority,)


# -------------------------------
# Rainbow Network with LSTM and Dueling Architecture
# -------------------------------
class RainbowLSTMDQN(nn.Module):
    def __init__(self, input_size, lstm_hidden, action_size):
        """
        input_size: feature dimension per time step (e.g. 5)
        lstm_hidden: hidden size for the LSTM
        action_size: number of actions
        """
        super(RainbowLSTMDQN, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        # Dueling streams (using NoisyLinear layers)
        # Value stream
        self.value_fc = nn.Sequential(
            NoisyLinear(lstm_hidden, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )
        # Advantage stream
        self.advantage_fc = nn.Sequential(
            NoisyLinear(lstm_hidden, 64),
            nn.ReLU(),
            NoisyLinear(64, action_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # take the last time step
        value = self.value_fc(features)  # shape: (batch, 1)
        advantage = self.advantage_fc(features)  # shape: (batch, action_size)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# -------------------------------
# Agent with Rainbow Improvements
# -------------------------------
class Agent:
    def __init__(self, input_size, lstm_hidden, action_size, n_steps=3, capacity=100000):
        self.input_size = input_size  # feature dimension per time step
        self.action_size = action_size
        self.n_steps = n_steps
        self.gamma = 0.99
        self.epsilon = 1.0       # used early; later, noisy layers handle exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.batch_size = 512  # reduced batch size for speed
        self.beta_start = 0.4
        self.beta_frames = 100000

        # Online and target networks
        self.model = RainbowLSTMDQN(input_size, lstm_hidden, action_size).to(DEVICE)
        self.target_model = RainbowLSTMDQN(input_size, lstm_hidden, action_size).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = nn.HuberLoss()

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity)
        # For multi-step learning
        self.n_step_buffer = deque(maxlen=n_steps)
        # For tracking training frames for beta scheduling
        self.frame_idx = 0

        # For trading logic
        self.inventory = []

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store(self, state, action, reward, next_state, done):
        """Store transition using multi-step returns."""
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_steps:
            return
        # Compute multi-step return:
        R = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if d:
                break
        state0, action0, _, _, _ = self.n_step_buffer[0]
        last_next_state, _, _, _, last_done = self.n_step_buffer[-1]
        self.memory.add((state0.flatten(), action0, R, last_next_state.flatten(), last_done, 1.0))

    def finish_n_step(self):
        """
        Flush the remaining transitions in the n-step buffer without appending new ones.
        This prevents the buffer from never emptying.
        """
        while self.n_step_buffer:
            transition = self.n_step_buffer.popleft()
            # Combine this transition with any that remain
            transitions = [transition] + list(self.n_step_buffer)
            R = 0.0
            for idx, (_, _, r, _, d) in enumerate(transitions):
                R += (self.gamma ** idx) * r
                if d:
                    break
            state0, action0, _, _, _ = transition
            # Use the last transition in the list for next_state and done
            last_transition = transitions[-1]
            last_next_state, _, _, _, last_done = last_transition
            self.memory.add((state0.flatten(), action0, R, last_next_state.flatten(), last_done, 1.0))

    @torch.no_grad()
    def act(self, state_seq):
        # Ensure state_seq is 3D: (1, seq_len, input_size)
        if state_seq.ndim == 2:
            state_seq = state_seq[np.newaxis, ...]
        state = torch.FloatTensor(state_seq).to(DEVICE)
        q_values = self.model(state)
        return q_values.argmax().item()

    def expReplay(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0
        self.frame_idx += 1
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        samples, indices, weights = self.memory.sample(self.batch_size, beta=beta)
        states = np.vstack([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.vstack([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        seq_len = int(states.shape[1] / self.input_size)
        states = torch.FloatTensor(states.reshape(-1, seq_len, self.input_size)).to(DEVICE)
        next_states = torch.FloatTensor(next_states.reshape(-1, seq_len, self.input_size)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        next_q_online = self.model(next_states)
        next_actions = next_q_online.argmax(dim=1)
        with torch.no_grad():
            next_q_target = self.target_model(next_states)
        target_q = rewards + (1 - dones) * (self.gamma ** self.n_steps) * next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_error = target_q - current_q
        loss = (weights * self.criterion(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.reset_noise()

        new_priorities = (td_error.abs() + 1e-6).detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()


# -------------------------------
# Technical Indicator Helpers
# -------------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# -------------------------------
# Data Download and Processing
# -------------------------------
def download_sp500_data(start_date='2010-01-01', end_date='2023-12-31'):
    print("Downloading S&P 500 data...")
    df = yf.download('^GSPC', start=start_date, end=end_date)
    df['Returns'] = df['Adj Close'].pct_change()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['RSI'] = compute_RSI(df['Adj Close'], period=14)
    df = df.dropna()
    print(f"Downloaded {len(df)} days of data")
    return df


def download_and_split_data(train_end_date='2020-12-31'):
    df = download_sp500_data()
    train_df = df.loc[df.index <= train_end_date]
    test_df = df.loc[df.index > train_end_date]
    print(f"Train set: {len(train_df)} days, Test set: {len(test_df)} days")
    return train_df, test_df


# -------------------------------
# New State Representation using sequences
# -------------------------------
def get_state_seq(df, t, window_size):
    """
    Build a state sequence (of length window_size) for time t.
    Each time step includes:
      - Returns
      - Normalized difference from SMA50: (Adj Close - SMA_50) / SMA_50
      - Normalized difference from SMA200: (Adj Close - SMA_200) / SMA_200
      - RSI
      - Volatility
    """
    features = []
    start = t - window_size + 1 if t - window_size + 1 >= 0 else 0
    for i in range(start, t + 1):
        row = df.iloc[i]
        ret = row['Returns']
        norm_sma50 = (row['Adj Close'] - row['SMA_50']) / (row['SMA_50'] + 1e-8)
        norm_sma200 = (row['Adj Close'] - row['SMA_200']) / (row['SMA_200'] + 1e-8)
        rsi = row['RSI']
        vol = row['Volatility']
        features.append([ret, norm_sma50, norm_sma200, rsi, vol])
    if len(features) < window_size:
        pad = [[0, 0, 0, 50, 0]] * (window_size - len(features))
        features = pad + features
    return np.array(features)  # shape: (window_size, 5)


# -------------------------------
# Reward Function and Sharpe Ratio Computation
# -------------------------------
def compute_sharpe(trade_returns):
    if len(trade_returns) == 0:
        return 0.0
    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe


# -------------------------------
# Training and Testing Loops
# -------------------------------
def train(episodes=30, window_size=10):
    train_df, test_df = download_and_split_data()
    trade_returns = []  # For episode-level tracking

    feature_dim = 5  # as defined in get_state_seq
    action_size = 3  # hold, buy, sell
    lstm_hidden = 64
    agent = Agent(input_size=feature_dim, lstm_hidden=lstm_hidden, action_size=action_size, n_steps=3)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        start_time = time.time()
        total_profit = 0
        episode_trade_returns = []
        agent.inventory = []
        agent.n_step_buffer.clear()
        losses = []
        num_steps = len(train_df) - 1

        for t in tqdm(range(num_steps), desc=f"Episode {episode+1}", leave=False):
            state = get_state_seq(train_df, t, window_size)
            current_price = train_df.iloc[t]['Adj Close']
            action = agent.act(state)
            reward = 0

            if action == 1:  # Buy
                if not agent.inventory:
                    agent.inventory.append(current_price)
                    reward = 0.2
                else:
                    reward = -0.05
            elif action == 2:  # Sell
                if agent.inventory:
                    bought_price = agent.inventory.pop(0)
                    profit = current_price - bought_price
                    vol = train_df.iloc[t]['Volatility']
                    reward = profit * 10 - vol * 5
                    total_profit += profit
                    episode_trade_returns.append(profit)
                else:
                    reward = -0.05
            else:  # Hold
                if agent.inventory and t > 0:
                    reward = (current_price - train_df.iloc[t-1]['Adj Close']) * 10
                else:
                    reward = 0

            next_state = get_state_seq(train_df, t + 1, window_size)
            done = (t == num_steps - 1)
            agent.store(state, action, reward, next_state, done)

            # Only update replay every 5 steps to reduce overhead
            if t % 5 == 0:
                loss = agent.expReplay()
                if loss:
                    losses.append(loss)

        agent.finish_n_step()
        agent.scheduler.step()
        agent.update_target_network()
        episode_time = time.time() - start_time
        sharpe_ratio = compute_sharpe(episode_trade_returns)
        print(f"Episode {episode + 1} Total Profit: ${total_profit:.2f}")
        print(f"Episode Duration: {episode_time:.2f} seconds")
        if losses:
            print(f"Average Loss: {np.mean(losses):.4f}")
        print(f"Episode Sharpe Ratio: {sharpe_ratio:.4f}")
    return agent, test_df, window_size


def test_agent(agent, test_df, window_size):
    print("\nTesting agent on unseen data...")
    trade_returns = []
    total_profit = 0
    agent.inventory = []
    states_buy = []
    states_sell = []
    agent.epsilon = 0.0  # fully exploit the learned policy
    agent.is_eval = True

    for t in range(len(test_df) - 1):
        state = get_state_seq(test_df, t, window_size)
        current_price = test_df.iloc[t]['Adj Close']
        action = agent.act(state)
        if action == 1:  # Buy
            if not agent.inventory:
                agent.inventory.append(current_price)
                states_buy.append(t)
                print(f"Buy at ${current_price:.2f}")
        elif action == 2:  # Sell
            if agent.inventory:
                bought_price = agent.inventory.pop(0)
                profit = current_price - bought_price
                total_profit += profit
                trade_returns.append(profit)
                states_sell.append(t)
                print(f"Sell at ${current_price:.2f} | Profit: ${profit:.2f}")
    sharpe_ratio = compute_sharpe(trade_returns)
    print(f"Test Total Profit: ${total_profit:.2f}")
    print(f"Test Sharpe Ratio: {sharpe_ratio:.4f}")
    plt.figure(figsize=(15, 5))
    plt.plot(test_df['Adj Close'].values, label='S&P 500', alpha=0.6)
    plt.plot(states_buy, test_df['Adj Close'].values[states_buy], '^', color='g', markersize=8, label='Buy')
    plt.plot(states_sell, test_df['Adj Close'].values[states_sell], 'v', color='r', markersize=8, label='Sell')
    plt.title(f"Test Trading Behavior - Total Profit: ${total_profit:.2f}, Sharpe: {sharpe_ratio:.2f}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    agent, test_df, window_size = train(episodes=30, window_size=10)
    test_agent(agent, test_df, window_size)
