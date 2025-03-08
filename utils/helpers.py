import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_sharpe(trade_returns):
    """
    Compute Sharpe ratio (included for backward compatibility)
    """
    if len(trade_returns) == 0:
        return 0.0
    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe

def compute_metrics(trade_returns, benchmark_returns=None):
    """
    Compute comprehensive trading performance metrics
    
    Args:
        trade_returns: List of trade returns
        benchmark_returns: Optional benchmark returns (e.g., S&P 500)
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Handle empty trade returns safely
    if not trade_returns or len(trade_returns) == 0:
        print("Warning: No trades to calculate metrics")
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0,
            "alpha": 0.0,
            "beta": 0.0
        }
    
    try:
        # Convert to numpy array if not already
        returns = np.array(trade_returns)
        
        # Basic statistics
        total_trades = len(returns)
        win_rate = np.sum(returns > 0) / total_trades if total_trades > 0 else 0
        avg_return = float(np.mean(returns))
        volatility = float(np.std(returns))
        
        # Risk-adjusted returns
        sharpe = float((avg_return / (volatility + 1e-8)) * np.sqrt(252))  # Annualized
        
        # Sortino ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-8
        sortino = float((avg_return / (downside_deviation + 1e-8)) * np.sqrt(252))  # Annualized
        
        # Maximum drawdown
        if len(returns) > 1:
            cum_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = running_max - cum_returns
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0
        
        # Benchmark comparison if provided
        alpha = 0.0
        beta = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Make sure benchmark returns match length of trade returns
            benchmark_returns = benchmark_returns[:len(returns)]
            
            if len(benchmark_returns) > 1:
                # Calculate beta (market correlation)
                try:
                    cov_matrix = np.cov(returns, benchmark_returns)
                    if cov_matrix.shape == (2, 2):  # Make sure we have valid covariance
                        covariance = cov_matrix[0, 1]
                        benchmark_variance = np.var(benchmark_returns)
                        beta = float(covariance / (benchmark_variance + 1e-8))
                    
                        # Calculate alpha (excess return)
                        benchmark_avg_return = np.mean(benchmark_returns)
                        alpha = float(avg_return - (beta * benchmark_avg_return))
                except:
                    print("Warning: Error calculating beta/alpha")
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return safe default values
        return {
            "total_trades": len(trade_returns) if trade_returns else 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0,
            "alpha": 0.0,
            "beta": 0.0
        }

def plot_equity_curve(trade_returns, benchmark_returns=None, title="Trading Performance"):
    """
    Plot equity curve and drawdown chart
    
    Args:
        trade_returns: List of trade returns
        benchmark_returns: Optional benchmark returns for comparison
        title: Plot title
    
    Returns:
        matplotlib.figure.Figure: Figure object containing the plots
    """
    try:
        if not trade_returns or len(trade_returns) == 0:
            # Create an empty figure if no trades
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No trades to display", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            return fig
            
        cum_returns = np.cumsum(trade_returns)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(cum_returns, label='Strategy')
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_returns = benchmark_returns[:len(trade_returns)]
            cum_benchmark = np.cumsum(benchmark_returns)
            ax1.plot(cum_benchmark, label='Benchmark', alpha=0.7)
        
        ax1.set_title(title)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        if len(trade_returns) > 1:
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (running_max - cum_returns) / (running_max + 1e-8) * 100  # In percentage
            ax2.fill_between(range(len(drawdown)), 0, -drawdown, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Trade')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Not enough trades for drawdown calculation", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting equity curve: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return fig