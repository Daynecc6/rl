import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from data.indicators import get_state_seq, add_enhanced_features
from utils.helpers import compute_sharpe, compute_metrics, plot_equity_curve
from config import Config
from agents.rainbow_agent import RainbowAgent
from data.data_loader import download_sp500_data, split_data

# Suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(f"Current working directory: {os.getcwd()}")

def calculate_benchmark_returns(df):
    """Calculate buy-and-hold returns for the same period"""
    try:
        daily_returns = df['Returns'].values
        return daily_returns
    except Exception as e:
        print(f"Error calculating benchmark returns: {e}")
        return np.zeros(len(df))

def test_agent(agent, test_df):
    print("\nTesting agent on unseen data...")
    
    # Create a timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for results if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "test_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize testing variables
    trade_returns = []
    total_profit = 0
    agent.inventory = []
    agent.position_size = 0  # Track position size
    states_buy = []
    states_sell = []
    action_history = []
    confidence_history = []
    position_size_history = []
    price_history = []
    agent.epsilon = 0.0  # No exploration during testing
    agent.is_eval = True
    
    # Stop loss and take profit counters
    stop_loss_triggers = 0
    take_profit_triggers = 0
    
    # Calculate benchmark returns
    benchmark_returns = calculate_benchmark_returns(test_df)
    
    # Dictionary to track portfolio value over time
    portfolio = {'cash': 10000, 'position': 0, 'equity': [10000]}
    
    for t in range(len(test_df) - 1):
        state = get_state_seq(test_df, t, Config.WINDOW_SIZE)
        current_price = test_df.iloc[t]['Adj Close']
        price_history.append(current_price)
        
        # Get volatility factor for position sizing and dynamic stops
        vol_factor = max(1.0, test_df.iloc[t]['Volatility'] * 100)
        
        # Check for stop loss / take profit on existing positions
        if agent.inventory:
            triggered, remaining, tp_sl_profit = agent.check_stop_loss_take_profit(current_price)
            
            for pos in triggered:
                entry_price, size, _, _, profit, trigger_type = pos
                
                if trigger_type == "STOP_LOSS":
                    stop_loss_triggers += 1
                    print(f"STOP LOSS triggered at ${current_price:.2f}, entry: ${entry_price:.2f}, loss: ${profit:.2f}")
                else:  # TAKE_PROFIT
                    take_profit_triggers += 1
                    print(f"TAKE PROFIT triggered at ${current_price:.2f}, entry: ${entry_price:.2f}, profit: ${profit:.2f}")
                
                # Add to total profit and trade returns
                total_profit += profit
                trade_returns.append(profit)
                
                # Update portfolio
                portfolio['cash'] += profit
                portfolio['position'] -= size * entry_price
        
        # Get action and confidence from agent
        action, confidence = agent.act(state)
        action_history.append(action)
        confidence_history.append(confidence)
        
        # Execute action
        transaction_cost = current_price * Config.TRANSACTION_COST
        
        if action == 1:  # Buy
            if agent.position_size < Config.MAX_POSITION_SIZE:
                # Calculate buy size based on confidence and volatility
                buy_size = agent.calculate_position_size(confidence, vol_factor)
                buy_size = min(buy_size, Config.MAX_POSITION_SIZE - agent.position_size)
                
                # Apply transaction cost
                cost = transaction_cost * buy_size
                
                # Calculate position value in dollars
                position_value = current_price * buy_size
                
                # Check if we have enough cash
                max_affordable = portfolio['cash'] / (current_price * (1 + Config.TRANSACTION_COST))
                if max_affordable < buy_size:
                    buy_size = max_affordable
                    position_value = current_price * buy_size
                    cost = position_value * Config.TRANSACTION_COST
                
                if buy_size > 0:
                    # Calculate dynamic stop loss and take profit levels
                    stop_loss_price, take_profit_price = agent.calculate_dynamic_stops(current_price, vol_factor)
                    
                    # Add to inventory
                    agent.inventory.append((current_price, buy_size, stop_loss_price, take_profit_price))
                    agent.position_size += buy_size
                    states_buy.append(t)
                    
                    # Update portfolio
                    portfolio['cash'] -= (position_value + cost)
                    portfolio['position'] += position_value
                    
                    print(f"Buy {buy_size:.2f} units at ${current_price:.2f}, cost ${cost:.2f}, confidence: {confidence:.2f}")
                    print(f"  Stop loss: ${stop_loss_price:.2f}, Take profit: ${take_profit_price:.2f}")
        
        elif action == 2:  # Sell
            if agent.position_size > 0:
                # Calculate profit from selling entire position
                profit = 0
                cost = 0
                
                for price, size, _, _ in agent.inventory:
                    position_profit = (current_price - price) * size
                    profit += position_profit
                    cost += transaction_cost * size
                
                # Record sell point
                states_sell.append(t)
                
                # Update metrics
                net_profit = profit - cost
                total_profit += net_profit
                trade_returns.append(net_profit)
                
                # Update portfolio
                portfolio['cash'] += current_price * agent.position_size - cost
                portfolio['position'] = 0
                
                print(f"Sell at ${current_price:.2f} | Profit: ${profit:.2f}, Cost: ${cost:.2f}, Net: ${net_profit:.2f}")
                
                # Reset inventory and position size
                agent.inventory = []
                agent.position_size = 0
        
        # Track portfolio value history
        portfolio_value = portfolio['cash'] + portfolio['position'] * current_price
        portfolio['equity'].append(portfolio_value)
        position_size_history.append(agent.position_size)

    # Calculate final metrics
    metrics = compute_metrics(trade_returns, benchmark_returns[:len(trade_returns)] if trade_returns else None)
    sharpe_ratio = metrics['sharpe_ratio']
    sortino_ratio = metrics['sortino_ratio']
    win_rate = metrics['win_rate']
    max_drawdown = metrics['max_drawdown']
    
    # Print detailed results
    print(f"\nTest Results:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Final Portfolio Value: ${portfolio['equity'][-1]:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Stop Loss Triggers: {stop_loss_triggers}")
    print(f"Take Profit Triggers: {take_profit_triggers}")
    
    # Create and save detailed equity curve plot
    try:
        if trade_returns:
            fig = plot_equity_curve(
                trade_returns, 
                benchmark_returns[:len(trade_returns)],
                title=f"Trading Performance - Profit: ${total_profit:.2f}, Sharpe: {sharpe_ratio:.2f}"
            )
            equity_curve_path = os.path.join(results_dir, f"equity_curve_{timestamp}.png")
            fig.savefig(equity_curve_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Equity curve saved to: {equity_curve_path}")
    except Exception as e:
        print(f"Error creating equity curve: {e}")
        traceback.print_exc()

    # Create and save trade visualization with multiple subplots
    try:
        fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Price and trades
        axs[0].plot(test_df['Adj Close'].values, label='S&P 500', alpha=0.6)
        
        if states_buy:
            axs[0].scatter(states_buy, test_df['Adj Close'].values[states_buy], 
                       marker='^', color='g', s=100, label='Buy')
        
        if states_sell:
            axs[0].scatter(states_sell, test_df['Adj Close'].values[states_sell], 
                       marker='v', color='r', s=100, label='Sell')
        
        axs[0].set_title(f"Test Trading Behavior - Total Profit: ${total_profit:.2f}, Sharpe: {sharpe_ratio:.2f}")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Actions and confidence
        actions_array = np.array(action_history)
        confidence_array = np.array(confidence_history)
        ax2 = axs[1]
        
        # Plot confidence as line
        ax2_twin = ax2.twinx()
        ax2_twin.plot(confidence_array, 'b-', alpha=0.7, label='Confidence')
        ax2_twin.set_ylabel('Confidence', color='b')
        ax2_twin.tick_params(axis='y', labelcolor='b')
        ax2_twin.set_ylim(0, 1)
        
        # Plot actions as scatter
        for action_val in range(3):
            # Create a mask for each action type
            mask = actions_array == action_val
            label = "Hold" if action_val == 0 else "Buy" if action_val == 1 else "Sell"
            color = "blue" if action_val == 0 else "green" if action_val == 1 else "red"
            
            # Plot points only where action matches
            ax2.scatter(
                np.where(mask)[0], 
                [action_val] * np.sum(mask),
                marker='o',
                color=color,
                alpha=0.6,
                label=label
            )
        
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Hold', 'Buy', 'Sell'])
        ax2.set_title("Agent Actions and Confidence Over Time")
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2_twin.legend(loc='upper right')
        
        # Plot 3: Portfolio value and position size
        ax3 = axs[2]
        ax3.plot(portfolio['equity'], 'g-', label='Portfolio Value')
        ax3.set_title("Portfolio Value and Position Size")
        ax3.set_xlabel("Trading Day")
        ax3.set_ylabel("Portfolio Value ($)")
        ax3.grid(True)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(position_size_history, 'r-', alpha=0.7, label='Position Size')
        ax3_twin.set_ylabel('Position Size', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        ax3_twin.set_ylim(0, 1)
        
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save trade visualization
        trade_vis_path = os.path.join(results_dir, f"test_trades_{timestamp}.png")
        plt.savefig(trade_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trade visualization saved to: {trade_vis_path}")
    except Exception as e:
        print(f"Error creating trade visualization: {e}")
        traceback.print_exc()
    
    # Save detailed test metrics to CSV
    try:
        test_results = {
            'total_profit': [total_profit],
            'final_portfolio_value': [portfolio['equity'][-1]],
            'total_trades': [metrics['total_trades']],
            'win_rate': [win_rate],
            'sharpe_ratio': [sharpe_ratio],
            'sortino_ratio': [sortino_ratio],
            'max_drawdown': [max_drawdown],
            'avg_return': [metrics['avg_return']],
            'volatility': [metrics['volatility']],
            'alpha': [metrics['alpha']],
            'beta': [metrics['beta']],
            'stop_loss_triggers': [stop_loss_triggers],
            'take_profit_triggers': [take_profit_triggers]
        }
        
        results_df = pd.DataFrame(test_results)
        results_csv_path = os.path.join(results_dir, f"test_metrics_{timestamp}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Test metrics saved to: {results_csv_path}")
    except Exception as e:
        print(f"Error saving test metrics: {e}")
        traceback.print_exc()
    
    return total_profit, metrics

if __name__ == "__main__":
    # Initialize agent
    print("\nInitializing agent...")
    agent = RainbowAgent()

    try:
        # Load saved model
        model_files = [f for f in os.listdir() if f.endswith('.pth')]
        if not model_files:
            # Look in training_results directory
            results_dir = os.path.join(os.getcwd(), "training_results")
            if os.path.exists(results_dir):
                model_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.pth')]
        
        if model_files:
            model_path = model_files[-1]  # Use the last model (likely most recent)
            print(f"Loading model from {model_path}")
            agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.model.eval()
            print(f"Loaded trained model from {model_path}")
        else:
            print("No saved model found. Run train.py first.")
            exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        exit()

    # Load test data with enhanced features
    print("\nLoading test data...")
    try:
        df = download_sp500_data()
        df = add_enhanced_features(df)
        _, test_df = split_data(df)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading test data: {e}")
        traceback.print_exc()
        exit()

    # Run test
    test_agent(agent, test_df)