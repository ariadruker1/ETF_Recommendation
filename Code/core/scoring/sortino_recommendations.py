import pandas as pd
import numpy as np

def sortino_score(etf_df, time_horizon, target_return, target_std, amount_recommend=5):
    """
    Calculates the Sortino Ratio for each ETF and returns the top recommendations.

    The Sortino Ratio measures an ETF's risk-adjusted return considering only
    downside volatility relative to a target return. A higher Sortino Ratio
    indicates a better return for the risk of falling below the target.

    Args:
        etf_df (pd.DataFrame): DataFrame with ETF metrics, including annual growth.
        time_horizon (int): The period in years for which the metrics were calculated.
        target_return (float): The investor's desired annual return (in %).
        target_std (float): Optional: User's acceptable standard deviation (for reference).
        amount_recommend (int): Number of top ETFs to return.

    Returns:
        pd.DataFrame: The ETF DataFrame with 'DownsideDeviation' and 'Sortino' columns,
                      sorted descending by 'Sortino' and limited to top recommendations.
    """
    growth_col = f'Annual_Growth_{time_horizon}Y'

    df = etf_df.dropna(subset=[growth_col]).copy()

    # Calculate downside deviation: only consider returns below target_return
    df['DownsideDeviation'] = df[growth_col].apply(
        lambda x: max(target_return - x, 0)
    )

    # If you have just a single value per ETF, downside deviation is same as above
    # To annualize or scale by time_horizon, could divide by sqrt(time_horizon)
    df['Sortino'] = (df[growth_col] - target_return) / df['DownsideDeviation'].replace(0, np.nan)

    # Handle divide by zero (if ETF never falls below target, Sortino -> inf)
    df['Sortino'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Sort descending and return top recommendations
    return df.sort_values('Sortino', ascending=False).head(amount_recommend)
