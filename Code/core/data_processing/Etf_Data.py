import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def get_etf_data(tickers, time_horizon, all_data, end_date):
    """
    Calculates key financial metrics for a list of ETFs over a specified time horizon.

    This function iterates through a list of ETF tickers, calculating the
    compound annual growth rate (CAGR) and the annualized standard deviation
    for each one over the given `time_horizon`.

    Args:
        tickers (list): A list of ETF ticker symbols to analyze.
        time_horizon (int): The number of years to use for the metric calculation.
        all_data (pd.DataFrame): A multi-level indexed DataFrame containing historical
                                 price data for all ETFs, with tickers as the top level.
        end_date (pd.Timestamp): The final date for the analysis period.

    Returns:
        pd.DataFrame: A DataFrame where each row represents an ETF and includes
                      its ticker, calculated annual growth, and standard deviation.
    """

    start_date = end_date - pd.DateOffset(years=time_horizon)
    results = []

    # Filter time period
    period_data = all_data[all_data.index >= start_date]

    for i, ticker in enumerate(tickers):
        row = {'Ticker': ticker}
        try:
            # Check if ticker exists in the data
            if ticker not in all_data.columns.get_level_values(0):
                annual_growth, annual_std = None, None
            else:
                # Get data for this specific ticker
                ticker_data = period_data[(ticker, 'Adj Close')].dropna()

                if len(ticker_data) < 2:
                    annual_growth, annual_std = None, None
                else:
                    # Find the start and end price to calculate compound annual growth rate
                    start_price = ticker_data.iloc[0]
                    end_price = ticker_data.iloc[-1]
                    actual_days = (
                        ticker_data.index[-1] - ticker_data.index[0]).days
                    actual_years = actual_days / 365.25

                    annual_growth = ((end_price / start_price)
                                     ** (1 / actual_years) - 1) * 100

                    # calculate annual standard deviation
                    daily_returns = ticker_data.pct_change().dropna()
                    annual_std_1y = daily_returns.std() * np.sqrt(252) * 100
                    annual_std = annual_std_1y / np.sqrt(actual_years)
                    annual_std = round(annual_std, 2)
        except:
            annual_growth, annual_std = None, None

        row[f'Annual_Growth_{time_horizon}Y'] = annual_growth
        row[f'Standard_Deviation_{time_horizon}Y'] = annual_std
        results.append(row)

    return pd.DataFrame(results)


def filter_etf_data(data, user_return, user_risk, user_time_horizon):
    """
    Filter ETFs based on user-defined return and risk criteria.
    Args:
    data (pd.DataFrame): DataFrame containing ETF data with growth and standard deviation
    user_return (float): User's expected return
    user_risk (float): User's expected risk (standard deviation)
    Returns:
    pd.DataFrame: Filtered DataFrame with ETFs that fall into the second quadrant, if less than
    5 etfs, include first and third quadrants as well. If still less than 5, return all etfs.
    """
    filtered_data = data[
        (data[f'Annual_Growth_{user_time_horizon}Y'] >= user_return) &
        (data[f'Standard_Deviation_{user_time_horizon}Y'] <= user_risk)
    ]

    if len(filtered_data) < 5:
        filtered_data = data[
            (data[f'Annual_Growth_{user_time_horizon}Y'] >= user_return) |
            (data[f'Standard_Deviation_{user_time_horizon}Y'] <= user_risk)
        ]

    if len(filtered_data) < 5:
        # return all etfs
        return data

    return filtered_data.reset_index(drop=True)
