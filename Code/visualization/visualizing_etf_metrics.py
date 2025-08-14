import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_risk_return_user(
    etf_metrics_df,
    user_growth,
    user_std,
    time_horizon,
    title,
    sharpe_list,
    utility_list,
    user_risk_ratio,
    user_min_etf_age,
    user_max_drawdown
):
    """
    Creates a scatter plot visualizing the risk and return of ETFs, benchmarked against a user's profile.

    This function generates a scatter plot to compare the historical performance of various ETFs
    in a risk-return framework. Each ETF is plotted based on its standard deviation (risk)
    and its annual growth (return). The plot highlights specific ETFs recommended by two
    different scoring methods (Sharpe and a custom utility score) using distinct colors.
    A key feature is the inclusion of a marker representing the user's desired risk-return
    profile and an inset box with a detailed summary of the user's investment preferences.

    Args:
        etf_metrics_df (pd.DataFrame): A DataFrame containing calculated metrics for ETFs,
            including columns for standard deviation and annual growth over a specified time horizon.
        user_growth (float): The user's desired annual growth rate in percentage.
        user_std (float): The user's acceptable annual standard deviation (risk) in percentage.
        time_horizon (int): The investment time horizon in years. This is used to
            dynamically select the correct columns from `etf_metrics_df`.
        title (str): The title for the plot.
        sharpe_list (list): A list of ETF ticker symbols recommended by the Sharpe ratio.
        utility_list (list): A list of ETF ticker symbols recommended by the custom
            utility scoring algorithm.
        user_risk_ratio (list): A list of two integers representing the user's
            weighted preference for return versus risk.
        user_min_etf_age (int): The minimum age in years for an ETF to be considered.
        user_max_drawdown (float): The user's maximum tolerated drawdown in percentage.

    Returns:
        None: This function displays a plot and saves it as 'etf_risk_return.png',
            but it does not return a value.
    """

    std_col = f'Standard_Deviation_{time_horizon}Y'
    growth_col = f'Annual_Growth_{time_horizon}Y'

    df_clean = etf_metrics_df.dropna(subset=[std_col, growth_col]).copy()
    
    labels_for_legend = {
        'Sharpe Only': 'blue',
        'Utility Only': 'orange',
        'Both': 'purple',
        'Not Selected': 'grey'
    }

    colors = []
    for ticker in df_clean['Ticker']:
        in_sharpe = ticker in sharpe_list
        in_util = ticker in utility_list

        if in_sharpe and in_util:
            colors.append('purple')
        elif in_util:
            colors.append('orange')
        elif in_sharpe:
            colors.append('blue')
        else:
            colors.append('grey')

    plt.figure(figsize=(14, 8))  # wide for space on right

    scatter = plt.scatter(df_clean[std_col], df_clean[growth_col], c=colors, alpha=0.7)

    plt.xlabel('Standard Deviation (Risk %)')
    plt.ylabel('Annual Growth (%)')
    plt.title(title)
    plt.grid(True)

    # Plot user profile point
    plt.scatter(user_std, user_growth, color='red', s=100, edgecolor='black', label='User Profile')

    # Create legend handles
    patches = [mpatches.Patch(color=color, label=label) for label, color in labels_for_legend.items()]
    user_patch = plt.Line2D([0], [0], marker='o', color='w', label='User Profile',
                           markerfacecolor='red', markersize=10, markeredgecolor='black')
    patches.append(user_patch)

    # Put legend on the right (leave enough room)
    plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Label selected ETFs (Sharpe, Utility, or Both)
    selected_mask = df_clean['Ticker'].isin(set(sharpe_list) | set(utility_list))
    selected_df = df_clean[selected_mask].sort_values(by=std_col).reset_index()

    # Smaller vertical offsets, or no offsets
    y_offsets = np.linspace(-0.15, 0.15, len(selected_df))  # tighter offsets for clarity

    for i, row in selected_df.iterrows():
        x = row[std_col]
        y = row[growth_col]
        ticker = row['Ticker']
        plt.text(x, y + y_offsets[i], ticker, fontsize=8, ha='left', va='center')

    # Place the user profile text directly in a box at the top-left of the chart
    profile_text = (
        f"User Profile:\n"
        f"Horizon: {time_horizon}y\n"
        f"Growth Goal: {user_growth}%\n"
        f"Std Dev: ±{user_std}%\n"
        f"Max Drawdown: {user_max_drawdown}%\n"
        f"Min ETF Age: {user_min_etf_age}y\n"
        f"Risk (Return:Risk): {user_risk_ratio[1]}:{user_risk_ratio[0]}"
    )
    plt.text(0.01, 0.98, profile_text, transform=plt.gca().transAxes,
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))

    # Adjust layout to fit legend on right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('user_etf_risk_return.png')
    plt.close()