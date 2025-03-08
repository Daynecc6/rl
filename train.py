import numpy as np
import torch
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import traceback  # For detailed error tracking

# Add this to suppress the OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Starting imports...")  # Debug print

try:
    from config import Config
    print("Config imported successfully")  # Debug print
    from agents.rainbow_agent import RainbowAgent
    print("RainbowAgent imported successfully")  # Debug print
    from data.data_loader import download_sp500_data, split_data
    print("Data loader imported successfully")  # Debug print
    from data.indicators import get_state_seq, add_enhanced_features
    print("Indicators imported successfully")  # Debug print
    from utils.helpers import compute_sharpe, compute_metrics, plot_equity_curve
    print("Helpers imported successfully")  # Debug print
except Exception as e:
    print(f"Error during imports: {str(e)}")
    traceback.print_exc()  # Print full traceback
    raise

# Create directory for saving results
RESULTS_DIR = os.path.join(os.getcwd(), "training_results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory: {RESULTS_DIR}")

def calculate_benchmark_returns(df):
    """Calculate buy-and-hold returns for the same period"""
    print("Calculating benchmark returns...")
    try:
        daily_returns = df['Returns'].values
        return daily_returns
    except Exception as e:
        print(f"Error calculating benchmark returns: {e}")
        return np.zeros(len(df))

def train():
    print("Starting training function...")
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load and split data with enhanced features
    print("Loading data and adding enhanced features...")
    df = download_sp500_data()
    
    # Add enhanced features
    df = add_enhanced_features(df)
    
    train_df, test_df = split_data(df)
    print(f"Data loaded: {len(train_df)} training samples, {len(test_df)} testing samples")
    
    # Calculate benchmark returns for comparison
    benchmark_returns = calculate_benchmark_returns(train_df)
    
    # Initialize agent
    print("Initializing agent...")
    agent = RainbowAgent()
    print(f"Agent initialized with epsilon={agent.epsilon}")
    
    # Initialize training history
    training_history = {
        'episode': [],
        'total_profit': [],
        'sharpe_ratio': [],
        'sortino_ratio': [],
        'max_drawdown': [],
        'win_rate': [],
        'avg_loss': [],
        'num_trades': [],
        'stop_loss_triggers': [],
        'take_profit_triggers': []
    }
    
    # Make a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for episode in range(Config.EPISODES):
        try:
            print(f"\nEpisode {episode + 1}/{Config.EPISODES}")
            start_time = time.time()
            
            # Initialize episode metrics
            total_profit = 0
            episode_trade_returns = []
            agent.inventory = []
            agent.position_size = 0  # Track position size (0 to 1.0)
            agent.n_step_buffer.clear()
            losses = []
            actions_taken = {0: 0, 1: 0, 2: 0}  # Count of each action type
            
            # Track stop loss and take profit triggers
            stop_loss_triggers = 0
            take_profit_triggers = 0
            
            num_steps = len(train_df) - 1
            print(f"Starting episode with {num_steps} steps...")
            
            for t in tqdm(range(num_steps), desc=f"Episode {episode+1}", leave=False):
                state = get_state_seq(train_df, t, Config.WINDOW_SIZE)
                current_price = train_df.iloc[t]['Adj Close']
                prev_price = train_df.iloc[t-1]['Adj Close'] if t > 0 else current_price
                
                # Get volatility factor for dynamic position sizing and stops
                vol_factor = max(1.0, train_df.iloc[t]['Volatility'] * 100)  # Scale volatility
                
                # Check for stop loss / take profit on existing positions
                if agent.inventory:
                    triggered, remaining, tp_sl_profit = agent.check_stop_loss_take_profit(current_price)
                    
                    for pos in triggered:
                        _, _, _, _, profit, trigger_type = pos
                        if trigger_type == "STOP_LOSS":
                            stop_loss_triggers += 1
                        else:  # TAKE_PROFIT
                            take_profit_triggers += 1
                            
                        # Add to total profit and track trade returns
                        total_profit += profit
                        episode_trade_returns.append(profit)
                
                # Get action and confidence
                action, confidence = agent.act(state)
                actions_taken[action] += 1
                
                # Execute action and get reward
                reward = 0
                
                # Transaction cost (applied to buys and sells)
                transaction_cost = current_price * Config.TRANSACTION_COST
                
                if action == 1:  # Buy
                    # Only buy if we have room in our position
                    if agent.position_size < Config.MAX_POSITION_SIZE:
                        # Calculate buy size based on confidence and volatility
                        buy_size = agent.calculate_position_size(confidence, vol_factor)
                        buy_size = min(buy_size, Config.MAX_POSITION_SIZE - agent.position_size)
                        
                        # Apply transaction cost
                        cost = transaction_cost * buy_size
                        
                        # Calculate dynamic stop loss and take profit levels
                        stop_loss_price, take_profit_price = agent.calculate_dynamic_stops(current_price, vol_factor)
                        
                        # Add to inventory with position size info and stops
                        agent.inventory.append((current_price, buy_size, stop_loss_price, take_profit_price))
                        agent.position_size += buy_size
                        
                        # Small positive reward for taking a position, minus transaction cost
                        reward = 0.2 * confidence - (cost / current_price * 10)
                    else:
                        # Penalize for trying to exceed position limits
                        reward = -0.1
                
                elif action == 2:  # Sell
                    if agent.position_size > 0:
                        # Calculate profits and apply transaction cost
                        position_value = 0
                        cost = 0
                        profit = 0
                        
                        # Sell entire position
                        for price, size, _, _ in agent.inventory:
                            # Calculate profit for this part of position
                            position_profit = (current_price - price) * size
                            profit += position_profit
                            
                            # Add transaction cost
                            cost += transaction_cost * size
                        
                        # Clear inventory and reset position size
                        agent.inventory = []
                        agent.position_size = 0
                        
                        # Calculate total profit (minus costs)
                        total_profit += profit - cost
                        
                        # Add to trade returns for metrics
                        episode_trade_returns.append(profit - cost)
                        
                        # Reward based on profit, adjusted for volatility
                        vol = train_df.iloc[t]['Volatility']
                        # Higher reward for profitable trades with high confidence
                        if profit > 0:
                            reward = (profit / current_price) * 100 * confidence - (cost / current_price * 10)
                        else:
                            reward = (profit / current_price) * 100 - vol * 5 - (cost / current_price * 10)
                    else:
                        # Penalize for trying to sell nothing
                        reward = -0.1
                
                else:  # Hold (action == 0)
                    if agent.position_size > 0 and t > 0:
                        # Reward/penalize based on daily price change
                        daily_return = (current_price - prev_price) / prev_price
                        reward = daily_return * 50 * agent.position_size
                    else:
                        # Check market regime to determine reward for staying out
                        market_regime = train_df.iloc[t]['Market_Regime']
                        vol_regime = train_df.iloc[t]['Vol_Regime']
                        
                        # More reward for staying out in downtrend or high volatility
                        if market_regime < 0 or vol_regime > 1:
                            reward = 0.05
                        else:
                            # Small positive reward for staying out of risky market
                            reward = 0.01
                
                # Get next state
                next_state = get_state_seq(train_df, t + 1, Config.WINDOW_SIZE)
                done = (t == num_steps - 1)
                
                # Store transition
                agent.store(state, action, reward, next_state, done)
                
                # Train more frequently (every 5 steps)
                if t % 5 == 0:
                    loss = agent.train_step()
                    if loss:
                        losses.append(loss)
            
            print("Episode complete. Finishing n-step returns...")
            # End of episode - finish any n-step returns
            agent.finish_n_step()
            
            print("Updating learning rate scheduler...")
            agent.scheduler.step()
            
            print("Updating target network...")
            agent.update_target_network()
            
            # Calculate episode metrics
            episode_time = time.time() - start_time
            print("Calculating episode metrics...")
            
            # Print trade information for debugging
            print(f"Number of trades in this episode: {len(episode_trade_returns)}")
            print(f"Stop loss triggers: {stop_loss_triggers}, Take profit triggers: {take_profit_triggers}")
            
            if len(episode_trade_returns) > 0:
                print(f"First few trade returns: {episode_trade_returns[:5]}")
            
            try:
                metrics = compute_metrics(episode_trade_returns, benchmark_returns[:len(episode_trade_returns)] if len(episode_trade_returns) > 0 else None)
                print(f"Metrics calculated successfully")
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                traceback.print_exc()
                metrics = {
                    "total_trades": len(episode_trade_returns),
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "avg_return": 0.0,
                    "volatility": 0.0,
                    "alpha": 0.0,
                    "beta": 0.0
                }
            
            # Update training history
            print("Updating training history...")
            training_history['episode'].append(episode + 1)
            training_history['total_profit'].append(total_profit)
            training_history['sharpe_ratio'].append(metrics['sharpe_ratio'])
            training_history['sortino_ratio'].append(metrics['sortino_ratio'])
            training_history['max_drawdown'].append(metrics['max_drawdown'])
            training_history['win_rate'].append(metrics['win_rate'])
            training_history['avg_loss'].append(np.mean(losses) if losses else 0)
            training_history['num_trades'].append(metrics['total_trades'])
            training_history['stop_loss_triggers'].append(stop_loss_triggers)
            training_history['take_profit_triggers'].append(take_profit_triggers)
            
            # Print episode statistics
            print(f"Episode {episode + 1} Total Profit: ${total_profit:.2f}")
            print(f"Episode Duration: {episode_time:.2f} seconds")
            print(f"Actions Taken - Hold: {actions_taken[0]}, Buy: {actions_taken[1]}, Sell: {actions_taken[2]}")
            if losses:
                print(f"Average Loss: {np.mean(losses):.4f}")
            print(f"Metrics - Sharpe: {metrics['sharpe_ratio']:.4f}, Sortino: {metrics['sortino_ratio']:.4f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.4f}, Win Rate: {metrics['win_rate']:.2%}")
            print(f"Total Trades: {metrics['total_trades']}")
            
            # Save intermediate model every 10 episodes or on first episode
            if (episode + 1) % 10 == 0 or episode == 0 or episode == Config.EPISODES - 1:
                checkpoint_path = os.path.join(RESULTS_DIR, f"model_ep{episode+1}_{timestamp}.pth")
                print(f"Saving checkpoint to {checkpoint_path}...")
                torch.save(agent.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                
                # Also save training history so far
                history_df = pd.DataFrame(training_history)
                history_csv_path = os.path.join(RESULTS_DIR, f"training_history_partial_{timestamp}.csv")
                history_df.to_csv(history_csv_path, index=False)
                print(f"Partial training history saved to {history_csv_path}")
                
        except Exception as e:
            print(f"Error during episode {episode + 1}: {str(e)}")
            traceback.print_exc()  # Print full error traceback
            
            # Save the model even if there was an error
            error_model_path = os.path.join(RESULTS_DIR, f"model_error_ep{episode+1}_{timestamp}.pth")
            try:
                torch.save(agent.model.state_dict(), error_model_path)
                print(f"Model saved after error to {error_model_path}")
            except:
                print("Could not save model after error")
            
            # Continue with the next episode
            continue
    
    # Save final model
    try:
        print("Training complete. Saving final model...")
        final_model_path = os.path.join(RESULTS_DIR, f"model_final_{timestamp}.pth")
        torch.save(agent.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Create and save final training history as CSV
        history_df = pd.DataFrame(training_history)
        history_csv_path = os.path.join(RESULTS_DIR, f"training_history_final_{timestamp}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"Training history saved to {history_csv_path}")
        
        # Save training history plots if we have data
        if training_history['episode']:
            print("Creating training history plots...")
            try:
                plt.figure(figsize=(15, 20))
                
                # Plot 1: Profit over episodes
                plt.subplot(5, 1, 1)
                plt.plot(training_history['episode'], training_history['total_profit'])
                plt.title('Total Profit per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Profit ($)')
                plt.grid(True)
                
                # Plot 2: Sharpe & Sortino ratios
                plt.subplot(5, 1, 2)
                plt.plot(training_history['episode'], training_history['sharpe_ratio'], label='Sharpe')
                plt.plot(training_history['episode'], training_history['sortino_ratio'], label='Sortino')
                plt.title('Risk-Adjusted Returns per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Ratio')
                plt.legend()
                plt.grid(True)
                
                # Plot 3: Win rate and drawdown
                plt.subplot(5, 1, 3)
                plt.plot(training_history['episode'], training_history['win_rate'], label='Win Rate')
                plt.plot(training_history['episode'], training_history['max_drawdown'], label='Max Drawdown')
                plt.title('Win Rate and Max Drawdown per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                # Plot 4: Average loss and number of trades
                plt.subplot(5, 1, 4)
                ax1 = plt.gca()
                ax1.plot(training_history['episode'], training_history['avg_loss'], 'b-', label='Avg Loss')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Average Loss', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                
                ax2 = ax1.twinx()
                ax2.plot(training_history['episode'], training_history['num_trades'], 'r-', label='Num Trades')
                ax2.set_ylabel('Number of Trades', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                plt.title('Training Loss and Number of Trades per Episode')
                plt.grid(True)
                
                # Plot 5: Stop loss and take profit triggers
                plt.subplot(5, 1, 5)
                plt.plot(training_history['episode'], training_history['stop_loss_triggers'], 'r-', label='Stop Loss')
                plt.plot(training_history['episode'], training_history['take_profit_triggers'], 'g-', label='Take Profit')
                plt.title('Stop Loss and Take Profit Triggers per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                
                # Save the figure
                history_plot_path = os.path.join(RESULTS_DIR, f"training_history_{timestamp}.png")
                plt.savefig(history_plot_path)
                plt.close()
                print(f"Training history plot saved to {history_plot_path}")
            except Exception as e:
                print(f"Error creating training plots: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
        traceback.print_exc()
    
    return agent, test_df

if __name__ == "__main__":
    print("Script started...")  # Debug print
    try:
        agent, test_df = train()
        print("Training complete. Starting testing phase...")
        
        # Import test function here to avoid circular import
        try:
            from test import test_agent
            test_agent(agent, test_df)
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()