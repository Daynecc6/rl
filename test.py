import matplotlib.pyplot as plt
from data.indicators import get_state_seq
from utils.helpers import compute_sharpe
from config import Config
from train import train

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

    # Calculate final metrics
    sharpe_ratio = compute_sharpe(trade_returns)
    print(f"Test Total Profit: ${total_profit:.2f}")
    print(f"Test Sharpe Ratio: {sharpe_ratio:.4f}")

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.plot(test_df['Adj Close'].values, label='S&P 500', alpha=0.6)
    plt.plot(states_buy, test_df['Adj Close'].values[states_buy], 
             '^', color='g', markersize=8, label='Buy')
    plt.plot(states_sell, test_df['Adj Close'].values[states_sell], 
             'v', color='r', markersize=8, label='Sell')
    plt.title(f"Test Trading Behavior - Total Profit: ${total_profit:.2f}, "
             f"Sharpe: {sharpe_ratio:.2f}")
    plt.legend()
    plt.show()
