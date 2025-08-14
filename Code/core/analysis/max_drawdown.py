import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from core.data_processing.ishares_ETF_list import download_valid_data
from datetime import datetime
import pandas as pd


def calculate_max_drawdown(user_max_drawdown, user_minimum_efs_age, valid_tickers, data, end_date):
    """
    Filters a list of ETF tickers based on the user's maximum drawdown tolerance
    and the minimum required age of the ETF.

    The maximum drawdown is calculated as a weighted average of the ETF's full
    history (30%) and the last 10 years of data (70%) to prioritize recent performance.
    ETFs that are younger than the user's specified minimum age are also excluded.

    Args:
        user_max_drawdown (float): The maximum percentage drawdown the user can tolerate.
        user_minimum_efs_age (int): The minimum age in years an ETF must be to be considered.
        valid_tickers (list): A list of valid ETF ticker symbols.
        data (pd.DataFrame): A DataFrame containing the historical 'Adj Close'
                             price data for all valid ETFs.
        end_date (pd.Timestamp): The final date for the analysis period.

    Returns:
        list: A filtered list of ticker symbols for ETFs that meet both the
              maximum drawdown and minimum age criteria.
    """
    
    tickers_within_user_drawdown_tolerance = []

    for ticker in valid_tickers:
        if 'Adj Close' not in data[ticker]:
            continue

        prices = data[ticker]['Adj Close'].dropna()
        past_10_year_date = end_date - pd.DateOffset(years=10)

        prices_origin = prices[prices.index <= end_date]
        prices_10_year = prices[(prices.index >= past_10_year_date) & (
            prices.index <= end_date)]

        if not prices_origin.empty:
            running_max = prices_origin.cummax()
            drawdown = (prices_origin - running_max) / running_max
            max_drawdown_origin = drawdown.min() * 100
        else:
            max_drawdown_origin = None

        if not prices_10_year.empty:
            running_max_10yr = prices_10_year.cummax()
            drawdown_10yr = (prices_10_year -
                             running_max_10yr) / running_max_10yr
            max_drawdown_10yr = drawdown_10yr.min() * 100
        else:
            max_drawdown_10yr = None

        if max_drawdown_origin is not None and max_drawdown_10yr is not None:
            max_drawdown = 0.3 * max_drawdown_origin + 0.7 * max_drawdown_10yr
        elif max_drawdown_origin is not None:
            max_drawdown = max_drawdown_origin
        elif max_drawdown_10yr is not None:
            max_drawdown = max_drawdown_10yr
        else:
            continue

        minimum_age_etf = datetime.now() - pd.DateOffset(years=user_minimum_efs_age)
        if max_drawdown >= -user_max_drawdown and prices.index.min() < minimum_age_etf:
            tickers_within_user_drawdown_tolerance.append(ticker)

    return tickers_within_user_drawdown_tolerance
