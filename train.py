import numpy as np
import torch
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

print("Starting imports...")  # Debug print

try:
    from config import Config
    print("Config imported successfully")  # Debug print
    from agents.rainbow_agent import RainbowAgent
    print("RainbowAgent imported successfully")  # Debug print
    from data.data_loader import download_sp500_data, split_data
    print("Data loader imported successfully")  # Debug print
    from data.indicators import get_state_seq
    print("Indicators imported successfully")  # Debug print
    from utils.helpers import compute_sharpe
    print("Helpers imported successfully")  # Debug print
except Exception as e:
    print(f"Error during imports: {str(e)}")
    raise

def train():
    print("Starting training function...")
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load and split data
    df = download_sp500_data()
    train_df, test_df = split_data(df)
    
    # Initialize agent
    agent = RainbowAgent()
    
    for episode in range(Config.EPISODES):
        print(f"\nEpisode {episode + 1}/{Config.EPISODES}")
        start_time = time.time()
        
        total_profit = 0
        episode_trade_returns = []
        agent.inventory = []
        agent.n_step_buffer.clear()
        losses = []
        
        num_steps = len(train_df) - 1
        
        for t in tqdm(range(num_steps), desc=f"Episode {episode+1}", leave=False):
            state = get_state_seq(train_df, t, Config.WINDOW_SIZE)
            current_price = train_df.iloc[t]['Adj Close']
            
            # Get action and calculate reward
            action = agent.act(state)
            reward = 0
            
            # Execute action
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
                    
            # Get next state and store transition
            next_state = get_state_seq(train_df, t + 1, Config.WINDOW_SIZE)
            done = (t == num_steps - 1)
            agent.store(state, action, reward, next_state, done)
            
            # Train every 5 steps
            if t % 5 == 0:
                loss = agent.train_step()
                if loss:
                    losses.append(loss)
        
        # End of episode
        agent.finish_n_step()
        agent.scheduler.step()
        agent.update_target_network()
        
        # Print episode statistics
        episode_time = time.time() - start_time
        sharpe_ratio = compute_sharpe(episode_trade_returns)
        print(f"Episode {episode + 1} Total Profit: ${total_profit:.2f}")
        print(f"Episode Duration: {episode_time:.2f} seconds")
        if losses:
            print(f"Average Loss: {np.mean(losses):.4f}")
        print(f"Episode Sharpe Ratio: {sharpe_ratio:.4f}")

    # Save trained model
    torch.save(agent.model.state_dict(), "model_checkpoint.pth")
    print("Model saved to model_checkpoint.pth")
    
    return agent, test_df

if __name__ == "__main__":
    print("Script started...")  # Debug print
    try:
        agent, test_df = train()
        from test import test_agent  # Import here to avoid circular import
        test_agent(agent, test_df)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
