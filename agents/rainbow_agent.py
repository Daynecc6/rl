import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from models.networks import RainbowLSTMDQN
from memory.replay_buffer import PrioritizedReplayBuffer
from config import Config

class RainbowAgent:
    def __init__(self):
        self.device = Config.DEVICE
        self.input_size = Config.FEATURE_DIM
        self.action_size = Config.ACTION_SIZE
        self.n_steps = Config.N_STEPS
        
        # Network parameters
        self.model = RainbowLSTMDQN().to(self.device)
        self.target_model = RainbowLSTMDQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Training parameters
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.batch_size = Config.BATCH_SIZE
        self.beta_start = Config.BETA_START
        self.beta_frames = Config.BETA_FRAMES
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100, 
            gamma=0.5
        )
        self.criterion = nn.HuberLoss()
        
        # Memory
        self.memory = PrioritizedReplayBuffer(Config.MEMORY_CAPACITY)
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.frame_idx = 0
        
        # Trading state
        self.inventory = []  # Will now store tuples of (price, position_size, stop_loss, take_profit)
        self.position_size = 0.0  # Track total position size (0 to 1.0)
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD  # Minimum confidence for full position
        self.is_eval = False
        
        # Risk management
        self.stop_loss_pct = Config.STOP_LOSS_PCT
        self.take_profit_pct = Config.TAKE_PROFIT_PCT

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_steps:
            return
            
        R = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if d:
                break
                
        state0, action0, _, _, _ = self.n_step_buffer[0]
        last_next_state, _, _, _, last_done = self.n_step_buffer[-1]
        
        self.memory.add((
            state0.flatten(), 
            action0, 
            R, 
            last_next_state.flatten(), 
            last_done, 
            1.0
        ))

    def act(self, state_seq):
        import numpy as np
        # Epsilon-greedy exploration during training
        if np.random.rand() < self.epsilon and not self.is_eval:
            action = np.random.randint(self.action_size)
            confidence = 0.5  # Default confidence for random actions
            return action, confidence
            
        if state_seq.ndim == 2:
            state_seq = state_seq[np.newaxis, ...]
        state = torch.FloatTensor(state_seq).to(self.device)
        
        with torch.no_grad():
            q_values, confidence = self.model(state)
            
        # Get the best action and its confidence
        action = q_values.argmax(dim=1).item()
        action_confidence = confidence[0, action].item()
        
        return action, action_confidence
    
    def calculate_position_size(self, confidence, vol_factor=1.0):
        """
        Calculate position size based on confidence and market volatility.
        
        Args:
            confidence: Model's confidence in the action (0-1)
            vol_factor: Volatility scaling factor (higher volatility = smaller position)
            
        Returns:
            float: Position size (0-1)
        """
        # Base position size based on confidence
        position_size = confidence
        
        # Scale by volatility (inverse relationship)
        position_size = position_size / vol_factor
        
        # Apply minimum confidence threshold
        if confidence < self.confidence_threshold:
            position_size *= (confidence / self.confidence_threshold)
        
        # Ensure maximum position size limit
        position_size = min(position_size, Config.MAX_POSITION_SIZE)
        
        return position_size
    
    def calculate_dynamic_stops(self, price, vol_factor=1.0):
        """
        Calculate dynamic stop-loss and take-profit levels based on volatility.
        
        Args:
            price: Current price 
            vol_factor: Volatility scaling factor
            
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        # Scale stop loss and take profit by volatility
        stop_loss_pct = self.stop_loss_pct * vol_factor
        take_profit_pct = self.take_profit_pct * vol_factor
        
        # Calculate actual price levels
        stop_loss_price = price * (1 - stop_loss_pct)
        take_profit_price = price * (1 + take_profit_pct)
        
        return stop_loss_price, take_profit_price

    def train_step(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0
            
        self.frame_idx += 1
        beta = min(1.0, self.beta_start + self.frame_idx * 
                  (1.0 - self.beta_start) / self.beta_frames)

        samples, indices, weights = self.memory.sample(self.batch_size, beta=beta)
        
        # Prepare batch
        states = np.vstack([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.vstack([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        seq_len = int(states.shape[1] / self.input_size)
        
        # Convert to tensors
        states = torch.FloatTensor(
            states.reshape(-1, seq_len, self.input_size)
        ).to(self.device)
        next_states = torch.FloatTensor(
            next_states.reshape(-1, seq_len, self.input_size)
        ).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute current and target Q-values
        current_q, _ = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: get actions from online network
        next_q_online, _ = self.model(next_states)
        next_actions = next_q_online.argmax(dim=1)
        
        with torch.no_grad():
            next_q_target, _ = self.target_model(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # Compute target using n-step returns
        target_q = rewards + (1 - dones) * (self.gamma ** self.n_steps) * next_q
        
        # Compute loss and update priorities
        td_error = target_q - current_q
        loss = (weights * self.criterion(current_q, target_q)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Reset noise and update priorities
        self.model.reset_noise()
        new_priorities = (td_error.abs() + 1e-6).detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def finish_n_step(self):
        """Flush the remaining transitions in the n-step buffer."""
        while self.n_step_buffer:
            transition = self.n_step_buffer.popleft()
            transitions = [transition] + list(self.n_step_buffer)
            
            R = 0.0
            for idx, (_, _, r, _, d) in enumerate(transitions):
                R += (self.gamma ** idx) * r
                if d:
                    break
                    
            state0, action0, _, _, _ = transition
            last_transition = transitions[-1]
            last_next_state, _, _, _, last_done = last_transition
            
            self.memory.add((
                state0.flatten(),
                action0,
                R,
                last_next_state.flatten(),
                last_done,
                1.0
            ))
    
    def check_stop_loss_take_profit(self, current_price):
        """Check if any open positions have hit stop-loss or take-profit levels.
        
        Returns:
            tuple: (triggered_positions, remaining_positions, total_profit)
        """
        if not self.inventory:
            return [], [], 0.0
            
        triggered_positions = []
        remaining_positions = []
        total_profit = 0.0
        
        for entry_price, size, stop_loss, take_profit in self.inventory:
            # Check for take profit
            if current_price >= take_profit:
                profit = (current_price - entry_price) * size
                total_profit += profit
                triggered_positions.append((entry_price, size, stop_loss, take_profit, profit, "TAKE_PROFIT"))
                self.position_size -= size
            # Check for stop loss
            elif current_price <= stop_loss:
                loss = (current_price - entry_price) * size
                total_profit += loss
                triggered_positions.append((entry_price, size, stop_loss, take_profit, loss, "STOP_LOSS"))
                self.position_size -= size
            else:
                remaining_positions.append((entry_price, size, stop_loss, take_profit))
        
        # Update inventory to only include remaining positions
        self.inventory = remaining_positions
        
        return triggered_positions, remaining_positions, total_profit