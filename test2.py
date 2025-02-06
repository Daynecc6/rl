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
import time  # Add this import at the top

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(DEVICE)
    
    def forward(self, x):
        return self.network(x)

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=None):
        self.state_size = state_size
        self.action_size = 3  # hold, buy, sell
        self.memory = deque(maxlen=100000)
        self.inventory = []
        self.is_eval = is_eval

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 1024
        
        self.model = DQN(state_size, self.action_size)
        if model_name:
            self.model.load_state_dict(torch.load(model_name))
        self.model.to(DEVICE)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = nn.HuberLoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            state.flatten(),
            action,
            reward,
            next_state.flatten(),
            done
        ))

    @torch.no_grad()
    def act(self, state):
        if not self.is_eval and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).to(DEVICE)
        q_values = self.model(state)
        return q_values.argmax().item()

    def expReplay(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        current_q_values = self.model(states)
        next_q_values = self.model(next_states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q = next_q_values.max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, expected_q)
        if torch.isnan(loss):
            print("NaN loss encountered; skipping update.")
            return 0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

def get_state(data, t, window_size):
    # Build state: differences and returns over the window.
    if t < window_size:
        padding = np.zeros(window_size - t - 1)
        block = np.concatenate((padding, data[0:t+1]))
    else:
        block = data[t-window_size+1:t+1]
    diffs = np.diff(block)
    # Avoid division by zero
    returns = diffs / (block[:-1] + 1e-8)
    state = np.concatenate([diffs, returns])
    return state.reshape(1, -1)

def download_sp500_data(start_date='2010-01-01', end_date='2023-12-31'):
    print("Downloading S&P 500 data...")
    df = yf.download('^GSPC', start=start_date, end=end_date)
    df['Returns'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    df = df.dropna()
    print(f"Downloaded {len(df)} days of data")
    return df

def download_and_split_data(train_end_date='2020-12-31'):
    df = download_sp500_data()
    train_df = df.loc[df.index <= train_end_date]
    test_df = df.loc[df.index > train_end_date]
    print(f"Train set: {len(train_df)} days, Test set: {len(test_df)} days")
    return train_df, test_df

def train(episodes=50, window_size=10):
    train_df, test_df = download_and_split_data()
    train_prices = train_df['Adj Close'].values
    # State size: (window_size - 1) diffs and (window_size - 1) returns
    agent = Agent(state_size=(window_size - 1) * 2)
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        start_time = time.time()  # Start timing
        total_profit = 0
        agent.inventory = []
        losses = []
        for t in range(len(train_prices) - 1):
            state = get_state(train_prices, t, window_size)
            action = agent.act(state)
            reward = 0
            if action == 1:  # Buy
                if not agent.inventory:
                    agent.inventory.append(train_prices[t])
                    # print(f"Buy at ${train_prices[t]:.2f}")
            elif action == 2 and agent.inventory:  # Sell
                bought_price = agent.inventory.pop(0)
                reward = train_prices[t] - bought_price
                total_profit += reward
                # print(f"Sell at ${train_prices[t]:.2f} | Profit: ${reward:.2f}")
            next_state = get_state(train_prices, t + 1, window_size)
            done = (t == len(train_prices) - 2)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.expReplay()
            if loss:
                losses.append(loss)
        agent.scheduler.step()
        episode_time = time.time() - start_time  # Calculate duration
        print(f"Episode {episode + 1} Total Profit: ${total_profit:.2f}")
        print(f"Episode Duration: {episode_time:.2f} seconds")
        if losses:
            print(f"Average Loss: {np.mean(losses):.4f}")
    return agent, test_df, window_size

def test_agent(agent, test_df, window_size):
    print("\nTesting agent on unseen data...")
    test_prices = test_df['Adj Close'].values
    total_profit = 0
    agent.inventory = []
    states_buy = []
    states_sell = []
    agent.is_eval = True
    agent.epsilon = 0.0
    for t in range(len(test_prices) - 1):
        state = get_state(test_prices, t, window_size)
        action = agent.act(state)
        if action == 1:  # Buy
            if not agent.inventory:
                agent.inventory.append(test_prices[t])
                states_buy.append(t)
                print(f"Buy at ${test_prices[t]:.2f}")
        elif action == 2 and agent.inventory:  # Sell
            bought_price = agent.inventory.pop(0)
            profit = test_prices[t] - bought_price
            total_profit += profit
            states_sell.append(t)
            # print(f"Sell at ${test_prices[t]:.2f} | Profit: ${profit:.2f}")
    print(f"Test Total Profit: ${total_profit:.2f}")
    plt.figure(figsize=(15, 5))
    plt.plot(test_prices, label='S&P 500', alpha=0.6)
    plt.plot(states_buy, test_prices[states_buy], '^', color='g', markersize=5, label='Buy')
    plt.plot(states_sell, test_prices[states_sell], 'v', color='r', markersize=5, label='Sell')
    plt.title(f"Test Trading Behavior - Total Profit: ${total_profit:.2f}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    agent, test_df, window_size = train(episodes=10, window_size=10)
    test_agent(agent, test_df, window_size)
