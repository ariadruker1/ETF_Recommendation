'''
Aria Druker and Linghe Zhou
'''
"""
ETF Recommendation Engine

This script serves as the main entry point for a Streamlit application that
recommends ETFs based on a user's financial profile. It processes a predefined
list of ETFs, analyzes their historical performance, and provides recommendations
using both a custom utility score and the standard Sharpe ratio. The script
also includes a quantitative back-testing feature to compare the performance
of the recommended ETF baskets over a specific period. Additionally there are 
options for graphing: user and ETFs risk reward profiles, and the post-training
performance comparison of each recommendation engine.
"""
import pandas as pd
from datetime import datetime
from config.constants import (
    USER_TIME_HORIZON, USER_DESIRED_GROWTH, USER_FLUCTUATION,
    USER_WORST_CASE, USER_MINIMUM_ETF_AGE, USER_RISK_PREFERENCE, TESTING_PERIOD, RECOMMENDATION_COUNT
)
from core.data_processing.ishares_ETF_list import download_valid_data
from core.user.user_profile import getUserProfile
from core.analysis.max_drawdown import calculate_max_drawdown
from core.data_processing.Etf_Data import get_etf_data
from visualization.visualizing_etf_metrics import plot_risk_return_user
from core.data_processing.risk_free_rates import fetch_risk_free_boc
from core.scoring.etf_recommendation_evaluation import top_recommend
from core.scoring.custom_score import utility_score
from testing.recommendation_test import recommendation_test
from visualization.graph_performance import graph_annual_growth_rate
from testing.compare_custom_Sharpe_test_results import quantitative_etf_basket_comparison
from core.scoring.sharpe_recommendation import sharpe_score

def main():
    valid_tickers, data = download_valid_data()
    user = getUserProfile()
    end_date = pd.Timestamp(datetime.now())
    md_tolerable_list = calculate_max_drawdown(user[USER_WORST_CASE], user[USER_MINIMUM_ETF_AGE], valid_tickers, data, end_date)
    etf_metrics = get_etf_data(md_tolerable_list, user[USER_TIME_HORIZON], data, end_date)
    risk_free_data = fetch_risk_free_boc("1995-01-01")
    etf_utility_calculation = utility_score(etf_metrics, user[USER_TIME_HORIZON], risk_free_data, user[USER_RISK_PREFERENCE])
    etf_utility_recommend = top_recommend(etf_utility_calculation, 'Utility_Score', RECOMMENDATION_COUNT)
    etf_sharpe_recommend = top_recommend(sharpe_score(etf_metrics, user[USER_TIME_HORIZON], risk_free_data), 'Sharpe', RECOMMENDATION_COUNT)
    print("Full time recommendations:")
    print("Custom Recommendations:")
    print(etf_utility_recommend)
    print("Sharpe Recommendations:")
    print(etf_sharpe_recommend)
    plot_risk_return_user(
        etf_metrics,
        user[USER_DESIRED_GROWTH],
        user[USER_FLUCTUATION],
        user[USER_TIME_HORIZON],
        f'ETF Risk-Return Space with User Profile (Time Horizon = {user[USER_TIME_HORIZON]}Y)',
        set(etf_sharpe_recommend['Ticker']),
        set(etf_utility_recommend['Ticker']),
        user[USER_RISK_PREFERENCE], user[USER_MINIMUM_ETF_AGE], user[USER_RISK_PREFERENCE]
    )
    custom_recommended_list, sharpe_recommended_list = recommendation_test(
        user[USER_TIME_HORIZON], user[USER_DESIRED_GROWTH], user[USER_FLUCTUATION],
        user[USER_WORST_CASE], user[USER_MINIMUM_ETF_AGE], user[USER_RISK_PREFERENCE],
        valid_tickers, data, TESTING_PERIOD
    )
    print("Test period recommendations:")
    print("Custom Recommendations:")
    print(custom_recommended_list)
    print("Sharpe Recommendations:")
    print(sharpe_recommended_list)
    test_start = end_date - pd.DateOffset(years=TESTING_PERIOD)
    results = quantitative_etf_basket_comparison(
        data, custom_recommended_list, sharpe_recommended_list, user[USER_DESIRED_GROWTH], 
        user[USER_FLUCTUATION], test_start, end_date, risk_free_data['yield_pct'].mean())
    print(results) 
    graph_annual_growth_rate(
        data,
        custom_recommended_list,
        sharpe_recommended_list,
        TESTING_PERIOD,
        user[USER_TIME_HORIZON],
        user[USER_DESIRED_GROWTH],
        user[USER_FLUCTUATION],
        user[USER_WORST_CASE],      # max drawdown
        user[USER_MINIMUM_ETF_AGE],
        user[USER_RISK_PREFERENCE]
    )
    print(f'Time_Horizon: {user[USER_TIME_HORIZON]}\nGrowth: {user[USER_DESIRED_GROWTH]}\nSTD: {user[USER_FLUCTUATION]}\nMax_Drawdown:'
          + f'{user[USER_WORST_CASE]}\nMin_ETF_Age: {user[USER_MINIMUM_ETF_AGE]}\nRisk_Return_Ratio: {user[USER_RISK_PREFERENCE]}\n')

if __name__ == "__main__":
    main()
