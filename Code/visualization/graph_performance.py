import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

def graph_annual_growth_rate(
    data, 
    custom_recommend_list, 
    sharpe_recommend_list,
    test_period,
    user_time_horizon,
    user_growth_pct,
    user_std_pct,
    user_max_drawdown,
    user_etf_age,
    user_risk_pref
):
    """
    Graphs the rolling annual growth rate of recommended ETFs and compares them to user goals.

    This function creates a plot that visualizes the rolling annual growth rate for
    a set of recommended ETFs over a specific test period. It color-codes the ETFs
    based on whether they were recommended by the custom utility score, the Sharpe
    ratio, or both. The plot also includes a shaded region representing the user's
    ideal growth rate, providing a clear visual benchmark for evaluating performance.
    A summary of the user's profile is included on the plot for context.

    Args:
        data (pd.DataFrame): DataFrame containing historical ETF price data, typically
            with a MultiIndex for tickers and price types (e.g., 'Adj Close').
        custom_recommend_list (list): A list of ETF ticker symbols recommended by the
            custom utility scoring algorithm.
        sharpe_recommend_list (list): A list of ETF ticker symbols recommended by the
            Sharpe ratio algorithm.
        test_period (int): The duration of the test period in years.
        user_time_horizon (int): The user's investment time horizon in years, used
            for the plot title.
        user_growth_pct (int): The user's desired annual growth rate in percentage.
        user_std_pct (int): The user's acceptable annual standard deviation in percentage.
        user_max_drawdown (int): The user's maximum tolerated drawdown in percentage.
        user_etf_age (int): The minimum age of an ETF to be considered.
        user_risk_pref (list): The risk-return preference weights of the user.

    Returns:
        None: This function displays a plot and saves it as 'etf_risk_return.png',
            but it does not return a value.
    """
    today = pd.Timestamp.now().normalize()
    start_date = today - pd.DateOffset(years=test_period)
    
    # Generate business day dates for x axis
    dates = pd.date_range(start=start_date, end=today, freq='B')
    
    plt.figure(figsize=(14, 7))

    # Plot shaded std deviation area
    plt.fill_between(dates, user_growth_pct - 2, user_growth_pct + 2,
                     color='green', alpha=0.15, label="User Ideal Growth")

    # Combine and determine colors for ETFs
    all_etfs = set(custom_recommend_list) | set(sharpe_recommend_list)
    used_labels = set()
    for etf in all_etfs:
        if etf in custom_recommend_list and etf in sharpe_recommend_list:
            color = 'purple'
            label = 'In Both (Custom & Sharpe)'
        elif etf in custom_recommend_list:
            color = 'orange'
            label = 'Custom Only'
        else:
            color = 'blue'
            label = 'Sharpe Only'

        # Avoid duplicate labels in legend
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)

        # Get price series for ETF, adjust close
        try:
            price_series = data[(etf, 'Adj Close')].dropna()
        except Exception:
            try:
                price_series = data[etf].dropna()
            except Exception:
                print(f"[⚠] Missing price data for {etf}, skipping.")
                continue

        # Restrict to date range and reindex to business days, forward fill missing
        price_series = price_series.loc[start_date:today].reindex(dates).ffill()

        if len(price_series) < 252:  # less than approx 1 year trading days
            print(f"[!] Not enough data for {etf}, skipping.")
            continue

        # Calculate rolling annual growth rate (year-over-year slope)
        # Using 252 trading days = 1 year approx
        window = 252
        log_prices = np.log(price_series)
        rolling_slope = (
            log_prices.rolling(window=window)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] * 252, raw=True)
        )

        # Convert slope of log price to annual growth % (exp(slope) - 1)*100
        annual_growth_rate = (np.exp(rolling_slope) - 1) * 100
        if not annual_growth_rate.empty:
            latest_growth = annual_growth_rate.iloc[-1]
            print(f"{etf}: {latest_growth:.2f}% annual growth")
        
        # Plot annual growth rate curve with label only once per category
        plt.plot(annual_growth_rate.index, annual_growth_rate.values, label=label if label else "", color=color, alpha=0.8)

        # Label ETF at the end of their line (avoid overlapping by vertical offset)
        try:
            last_date = annual_growth_rate.index[-1]
            last_value = annual_growth_rate.iloc[-1]
            # offset label by small amount vertically depending on index to reduce overlap
            plt.text(last_date, last_value + (used_labels.__len__() * 0.3), etf,
                     fontsize=8, color=color, va='center', ha='left')
        except Exception:
            continue

    plt.title(f"Annual Growth Rate Results:\nAfter {user_time_horizon} years training the recommended ETFs performance from each method.\n")
    plt.xlabel("Date")
    plt.ylabel("Annual Growth Rate (%)")
    plt.grid(True)

    # Place legend outside on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title="ETF Group")
    
    # Place profile inside top-left corner with some padding
    profile_text = (
        f"User Profile:\n"
        f"• Horizon: {user_time_horizon}y\n"
        f"• Growth Goal: {user_growth_pct}%\n"
        f"• Std Dev: ±{user_std_pct}%\n"
        f"• Max Drawdown: {user_max_drawdown}%\n"
        f"• Min ETF Age: {user_etf_age}y\n"
        f"• Risk (Return:Risk): {user_risk_pref[1]}:{user_risk_pref[0]}"
    )
    plt.gca().text(
        0.02, 0.98, profile_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        va='top',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="gray", alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
    plt.savefig('etf_risk_return.png')
    plt.close()
