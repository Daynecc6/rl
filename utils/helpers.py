import numpy as np

def compute_sharpe(trade_returns):
    if len(trade_returns) == 0:
        return 0.0
    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe