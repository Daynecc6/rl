import os
import time
import torch
import matplotlib.pyplot as plt
from data.indicators import get_state_seq
from utils.helpers import compute_sharpe
from config import Config
from agents.rainbow_agent import RainbowAgent
from data.data_loader import download_sp500_data, split_data

print(f"Current working directory: {os.getcwd()}")

def test_agent(agent, test_df):
    print("\nTesting agent on unseen data...")
    trade_returns = []
    total_profit = 0
    agent.inventory = []
    states_buy = []
    states_sell = []
    agent.epsilon = 0.0
    agent.is_eval = True

    for t in range(len(test_df) - 1):
        state = get_state_seq(test_df, t, Config.WINDOW_SIZE)
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

    # Create and verify save folder
    save_folder = os.path.join(os.getcwd(), "test_graphs")
    print(f"\nAttempting to save to folder: {save_folder}")
    
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
            print(f"Created folder: {save_folder}")
        except Exception as e:
            print(f"Error creating folder: {e}")
            return  # Exit if we can't create the folder

    # Generate timestamp and filename
    timestamp = int(time.time())
    filename = os.path.join(save_folder, f"test_graph_{timestamp}.png")
    print(f"Will save file as: {filename}")

    # Create the plot
    plt.figure(figsize=(15, 5))
    plt.plot(test_df['Adj Close'].values, label='S&P 500', alpha=0.6)
    
    if states_buy:  # Only plot buy points if there are any
        plt.scatter(states_buy, test_df['Adj Close'].values[states_buy], 
                   marker='^', color='g', s=100, label='Buy')
    
    if states_sell:  # Only plot sell points if there are any
        plt.scatter(states_sell, test_df['Adj Close'].values[states_sell], 
                   marker='v', color='r', s=100, label='Sell')
    
    plt.title(f"Test Trading Behavior - Total Profit: ${total_profit:.2f}, Sharpe: {sharpe_ratio:.2f}")
    plt.legend()
    plt.grid(True)

    # Save plot with extensive error handling
    try:
        print("Attempting to save plot...")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved plot to: {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()
        print("Closed plot")

if __name__ == "__main__":
    # Initialize agent
    print("\nInitializing agent...")
    agent = RainbowAgent()

    # Load saved model
    try:
        agent.model.load_state_dict(torch.load("model_checkpoint.pth", map_location=agent.device))
        agent.model.eval()
        print("Loaded trained model from model_checkpoint.pth")
    except FileNotFoundError:
        print("No saved model found. Run train.py first.")
        exit()

    # Load test data
    print("\nLoading test data...")
    df = download_sp500_data()
    _, test_df = split_data(df)
    print("Data loaded successfully")

    # Run test
    test_agent(agent, test_df)