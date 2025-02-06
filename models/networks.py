import torch.nn as nn
from models.layers import NoisyLinear
from config import Config

class RainbowLSTMDQN(nn.Module):
    def __init__(self, input_size=Config.FEATURE_DIM, 
                 lstm_hidden=Config.LSTM_HIDDEN, 
                 action_size=Config.ACTION_SIZE):
        super(RainbowLSTMDQN, self).__init__()
        
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        
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
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]
        
        value = self.value_fc(features)
        advantage = self.advantage_fc(features)
        
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()