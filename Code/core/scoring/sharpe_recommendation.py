import pandas as pd

def sharpe_score(etf_df, time_horizon, risk_free_df):
    """
    Calculates the Sharpe Ratio for each ETF and sorts them by the score.

    The Sharpe Ratio measures an ETF's performance by adjusting for its risk.
    It is calculated by taking the ETF's excess return (annual growth minus
    the risk-free rate) and dividing it by the ETF's standard deviation.
    A higher Sharpe Ratio indicates a better risk-adjusted return.

    Args:
        etf_df (pd.DataFrame): A DataFrame containing ETF metrics, including
                               annual growth and standard deviation.
        time_horizon (int): The time period in years for which the metrics were
                            calculated.
        risk_free_df (pd.DataFrame): A DataFrame containing historical risk-free
                                     rates, used to find the average risk-free
                                     rate over the specified time horizon.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns, 'ExcessReturn'
                      and 'Sharpe', sorted in descending order by the 'Sharpe'
                      ratio.
    """
    growth_col = f'Annual_Growth_{time_horizon}Y'
    std_col = f'Standard_Deviation_{time_horizon}Y'

    df = etf_df.dropna(subset=[growth_col, std_col]).copy()

    end = risk_free_df.index.max()
    start = end - pd.DateOffset(years=time_horizon)
    rf_period = risk_free_df.loc[start:end]
    avg_rf = rf_period['yield_pct'].mean()

    df['ExcessReturn'] = df[growth_col] - avg_rf
    df['Sharpe'] = df['ExcessReturn'] / df[std_col]

    return df.sort_values('Sharpe', ascending=False)
