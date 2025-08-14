import pandas as pd
import numpy as np

def get_etf_data(etf_list, time_horizon, price_data, end_date, min_etf_age=0):
    """
    Returns a DataFrame with ETF metrics: annual growth and std deviation
    for the specified time horizon. Filters out ETFs that do not have
    sufficient history or missing data.
    """
    df_list = []

    start_date = end_date - pd.DateOffset(years=time_horizon)

    for etf in etf_list:
        # Attempt to get adjusted close prices
        try:
            prices = price_data[(etf, 'Adj Close')].dropna()
        except (KeyError, TypeError):
            try:
                prices = price_data[etf].dropna()
            except (KeyError, TypeError):
                continue

        period_prices = prices.loc[start_date:end_date]
        if len(period_prices) < 2:  # not enough data to compute metrics
            continue

        # Annualized return
        total_return = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
        annual_return = ((1 + total_return) ** (1 / time_horizon) - 1) * 100

        # Annualized standard deviation
        daily_returns = period_prices.pct_change().dropna()
        annual_std = daily_returns.std() * np.sqrt(252) * 100  # percent

        df_list.append({
            'Ticker': etf,
            f'Annual_Growth_{time_horizon}Y': annual_return,
            f'Standard_Deviation_{time_horizon}Y': annual_std
        })

    if not df_list:
        return pd.DataFrame()  # return empty if nothing valid

    df = pd.DataFrame(df_list)
    # Filter ETFs that have missing values
    df = df.dropna()
    return df
