import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import (
    RECOMMENDATION_COUNT
)
from core.scoring.sharpe_recommendation import sharpe_score
from core.scoring.custom_score import utility_score
from core.scoring.etf_recommendation_evaluation import top_recommend
from core.data_processing.risk_free_rates import fetch_risk_free_boc
from visualization.visualizing_etf_metrics import plot_risk_return_user
from core.data_processing.Etf_Data import get_etf_data, filter_etf_data
from core.analysis.max_drawdown import calculate_max_drawdown
from datetime import datetime
import pandas as pd


def recommendation_test(
    time_horizon, desired_growth, std_deviation, max_drawdown,
    minimum_etf_age, risk_preference, valid_tickers, data, test_period
):
    """
    Generates ETF recommendations based on a training period and returns
    lists of top ETFs from a custom utility score and Sharpe ratio.

    This function simulates the recommendation process by using a specified
    `test_period` to define a historical training window. It calculates various
    ETF metrics and scores (utility and Sharpe), and then returns the top
    recommended tickers from each method for that period.

    Args:
        time_horizon (int): The user's time horizon for investment, in years.
        desired_growth (float): The user's desired annual growth rate.
        std_deviation (float): The user's acceptable annual standard deviation.
        max_drawdown (float): The maximum tolerated drawdown, as a percentage.
        minimum_etf_age (int): The minimum age of an ETF to be considered, in years.
        risk_preference (list): A list containing the risk and return
                                preference weights, e.g., [risk_weight, return_weight].
        valid_tickers (list): A list of all available ETF tickers.
        data (pd.DataFrame): A DataFrame containing the historical price data for all ETFs.
        test_period (int): The length of the back-testing period, in years.

    Returns:
        tuple: A tuple containing two lists of strings:
               - `custom_recommended_list` (list of str): Tickers of ETFs recommended
                 by the custom utility score.
               - `sharpe_recommended_list` (list of str): Tickers of ETFs recommended
                 by the Sharpe ratio.
    """
    today = pd.Timestamp(datetime.now())
    train_end = today - pd.DateOffset(years=test_period)

    md_tolerable_list = calculate_max_drawdown(max_drawdown, minimum_etf_age, valid_tickers, data, train_end)
    etf_metrics = get_etf_data(md_tolerable_list, time_horizon, data, train_end)
    risk_free_data = fetch_risk_free_boc("1995-01-01")

    etf_utility_calculation = utility_score(
        etf_metrics, time_horizon, risk_free_data, risk_preference)

    if 'Utility_Score' in etf_utility_calculation.columns:
        custom_clean = etf_utility_calculation.dropna(subset=['Utility_Score'])
        custom_recommended_list = top_recommend(custom_clean, 'Utility_Score', RECOMMENDATION_COUNT)[
            'Ticker'].tolist() if not custom_clean.empty else []
    else:
        custom_recommended_list = []

    sharpe_scoring_calculation = sharpe_score(
        etf_metrics, time_horizon, risk_free_data)

    if 'Sharpe' in sharpe_scoring_calculation.columns:
        sharpe_clean = sharpe_scoring_calculation.dropna(subset=['Sharpe'])
        sharpe_recommended_list = top_recommend(sharpe_clean, 'Sharpe', RECOMMENDATION_COUNT)[
            'Ticker'].tolist() if not sharpe_clean.empty else []
    else:
        sharpe_recommended_list = []

    return custom_recommended_list, sharpe_recommended_list
