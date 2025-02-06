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
        self.inventory = []
        self.is_eval = False

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
        if state_seq.ndim == 2:
            state_seq = state_seq[np.newaxis, ...]
            
        state = torch.FloatTensor(state_seq).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

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

        # Compute target Q-values
        next_q_online = self.model(next_states)
        next_actions = next_q_online.argmax(dim=1)
        
        with torch.no_grad():
            next_q_target = self.target_model(next_states)

        target_q = rewards + (1 - dones) * \
                  (self.gamma ** self.n_steps) * \
                  next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                  
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
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