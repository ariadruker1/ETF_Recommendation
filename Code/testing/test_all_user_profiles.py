import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.constants import (
    TESTING_PERIOD, RECOMMENDATION_COUNT, TOP_RANGE_RECOMMENDATIONS
)
import itertools
import pandas as pd
from datetime import datetime
from core.data_processing.ishares_ETF_list import download_valid_data
from core.analysis.max_drawdown import calculate_max_drawdown
from core.data_processing.Etf_Data import get_etf_data
from core.data_processing.risk_free_rates import fetch_risk_free_boc
from core.scoring.etf_recommendation_evaluation import top_recommend
from core.scoring.custom_score import utility_score
from testing.recommendation_test import recommendation_test
from core.scoring.sharpe_recommendation import sharpe_score

# Constants for index access
USER_TIME_HORIZON = 0
USER_DESIRED_GROWTH = 1
USER_FLUCTUATION = 2
USER_WORST_CASE = 3
USER_MINIMUM_ETF_AGE = 4
USER_RISK_PREFERENCE = 5


def generate_all_user_tests():
    """
    Automates the generation and analysis of ETF recommendations for a full
    range of hypothetical user profiles.

    This function systematically tests the recommendation engine by creating
    a matrix of all possible user preference combinations from a predefined set
    of parameters. For each combination, it generates recommendations based on
    the full dataset and a back-testing period. It then calculates and
    records metrics such as the overlap between custom and Sharpe recommendations,
    as well as the consistency between full-time and test-period recommendations.
    The final results are compiled into a pandas DataFrame and saved to an
    Excel file for further analysis.

    Args:
        None: This function does not take any arguments. All test parameters
              are hard-coded within the function for controlled testing.

    Returns:
        pd.DataFrame: A DataFrame containing the results of all user profile
                      tests, including the profile parameters, various overlap
                      metrics, and the recommended tickers for each scenario.
                      This DataFrame is also saved to an Excel file.
    """


    valid_tickers, data = download_valid_data()
    end_date = pd.Timestamp(datetime.now())

    time_horizons = [1, 8, 25]
    growths = [2, 21]
    stds = [5, 35]
    max_drawdowns = [15, 25, 35, 45, 100]
    min_etf_ages = [0, 3, 10]
    risk_preferences = [[3, 1], [1, 1], [1, 3]]

    rows = []

    for combo in itertools.product(time_horizons, growths, stds, max_drawdowns, min_etf_ages, risk_preferences):
        user = list(combo)
        print(f"Processing combo: {combo}")

        try:
            # Full-time recommendations - direct calc method
            md_tolerable_list = calculate_max_drawdown(
                user[USER_WORST_CASE], user[USER_MINIMUM_ETF_AGE] + TESTING_PERIOD, valid_tickers, data, end_date
            )
            etf_metrics_full_time = get_etf_data(md_tolerable_list, user[USER_TIME_HORIZON] + TESTING_PERIOD, data, end_date)
            risk_free_data = fetch_risk_free_boc("1995-01-01")

            utility_scores = utility_score(etf_metrics_full_time, user[USER_TIME_HORIZON] + TESTING_PERIOD, risk_free_data, user[USER_RISK_PREFERENCE])
            sharpe_scores = sharpe_score(etf_metrics_full_time, user[USER_TIME_HORIZON] + TESTING_PERIOD, risk_free_data)

            # Get top RECOMMENDATION_COUNT (e.g., 5) recommendations for full-time
            full_time_custom_df = top_recommend(utility_scores, 'Utility_Score', RECOMMENDATION_COUNT)
            full_time_sharpe_df = top_recommend(sharpe_scores, 'Sharpe', RECOMMENDATION_COUNT)
            full_time_custom_list = full_time_custom_df['Ticker'].tolist()
            full_time_sharpe_list = full_time_sharpe_df['Ticker'].tolist()

            # Get TOP_RANGE_RECOMMENDATIONS (e.g., 15) recommendations for full-time for comparison
            full_time_custom_top_range_df = top_recommend(utility_scores, 'Utility_Score', TOP_RANGE_RECOMMENDATIONS)
            full_time_sharpe_top_range_df = top_recommend(sharpe_scores, 'Sharpe', TOP_RANGE_RECOMMENDATIONS)
            full_time_custom_top_range_list = full_time_custom_top_range_df['Ticker'].tolist()
            full_time_sharpe_top_range_list = full_time_sharpe_top_range_df['Ticker'].tolist()

            if not full_time_custom_list or not full_time_sharpe_list:
                print(f"Skipping combo {combo} due to empty full-time recommendations.")
                continue

            # Test period recommendations via recommendation_test
            custom_list, sharpe_list = recommendation_test(
                user[USER_TIME_HORIZON], user[USER_DESIRED_GROWTH], user[USER_FLUCTUATION],
                user[USER_WORST_CASE], user[USER_MINIMUM_ETF_AGE], user[USER_RISK_PREFERENCE],
                valid_tickers, data, TESTING_PERIOD
            )

            if not custom_list or not sharpe_list:
                print(f"Skipping combo {combo} due to empty test period recommendations.")
                continue

            # Sets for overlap calculations
            set_full_custom = set(full_time_custom_list)
            set_full_sharpe = set(full_time_sharpe_list)
            set_full_custom_top_range = set(full_time_custom_top_range_list)
            set_full_sharpe_top_range = set(full_time_sharpe_top_range_list)
            set_test_custom = set(custom_list)
            set_test_sharpe = set(sharpe_list)

            # Counts of overlaps
            overlap_full_custom_sharpe = len(set_full_custom & set_full_sharpe)
            overlap_test_custom_sharpe = len(set_test_custom & set_test_sharpe)
            overlap_full_test_custom = len(set_full_custom & set_test_custom)
            overlap_full_test_sharpe = len(set_full_sharpe & set_test_sharpe)
            
            # New columns: Test period recommendations vs. Full-time top 15
            custom_test_in_custom_full_top_range = len(set_test_custom & set_full_custom_top_range)
            sharpe_test_in_sharpe_full_top_range = len(set_test_sharpe & set_full_sharpe_top_range)

            # Intersections as comma-separated strings (sorted)
            intersection_custom_test_sharpe_test = ', '.join(sorted(set_test_custom & set_test_sharpe))
            intersection_custom_full_test = ', '.join(sorted(set_full_custom & set_test_custom))
            intersection_sharpe_full_test = ', '.join(sorted(set_full_sharpe & set_test_sharpe))

            # Convert ticker lists to comma-separated strings for output
            custom_full_time_tickers = ', '.join(full_time_custom_list)
            custom_test_time_tickers = ', '.join(custom_list)
            sharpe_full_time_tickers = ', '.join(full_time_sharpe_list)
            sharpe_test_time_tickers = ', '.join(sharpe_list)
            
            # New columns for the top 15 lists
            custom_full_time_top_15 = ', '.join(full_time_custom_top_range_list)
            sharpe_full_time_top_15 = ', '.join(full_time_sharpe_top_range_list)

            # Build final row
            row = {
                "time_horizon": user[USER_TIME_HORIZON],
                "growth": user[USER_DESIRED_GROWTH],
                "fluctuation": user[USER_FLUCTUATION],
                "max_drawdown": user[USER_WORST_CASE],
                "min_etf_age": user[USER_MINIMUM_ETF_AGE],
                "risk_preference": user[USER_RISK_PREFERENCE],
                "overlap_full_custom_sharpe": overlap_full_custom_sharpe,
                "overlap_test_custom_sharpe": overlap_test_custom_sharpe,
                "overlap_full_test_custom": overlap_full_test_custom,
                "overlap_full_test_sharpe": overlap_full_test_sharpe,
                "custom_test_in_custom_full_top_range": custom_test_in_custom_full_top_range,
                "sharpe_test_in_sharpe_full_top_range": sharpe_test_in_sharpe_full_top_range,
                "custom_full_time_tickers": custom_full_time_tickers,
                "custom_test_time_tickers": custom_test_time_tickers,
                "sharpe_full_time_tickers": sharpe_full_time_tickers,
                "sharpe_test_time_tickers": sharpe_test_time_tickers,
                "custom_full_time_top_15": custom_full_time_top_15,
                "sharpe_full_time_top_15": sharpe_full_time_top_15,
                "intersection_custom_test_sharpe_test": intersection_custom_test_sharpe_test,
                "intersection_custom_full_test": intersection_custom_full_test,
                "intersection_sharpe_full_test": intersection_sharpe_full_test,
            }

            rows.append(row)

        except Exception as e:
            print(f"Failed on combo {combo}: {e}")
            continue

    if not rows:
        print("No valid data rows collected to write to Excel.")
        return None

    df = pd.DataFrame(rows)
    df.to_excel('~/Desktop/all_users_etf_overlap_and_tickers_1yr_test_with_top_15.xlsx', index=False)
    print(f"Saved results with {len(df)} rows to Excel.")
    return df


if __name__ == "__main__":
    generate_all_user_tests()