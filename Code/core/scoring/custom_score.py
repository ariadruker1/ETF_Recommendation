import pandas as pd

def utility_score(etf_df, time_horizon, risk_free_df, risk_pref):
    """
    Calculates a custom utility score for each ETF based on user preferences.

    The utility score is a weighted average of an ETF's normalized excess return
    and its normalized standard deviation. The weights are derived from the
    user's risk preference. A higher utility score indicates a more favorable
    risk-return profile according to the user's criteria.

    Args:
        etf_df (pd.DataFrame): A DataFrame containing ETF metrics, including
                               'Annual_Growth' and 'Standard_Deviation'.
        time_horizon (int): The user's investment time horizon in years,
                            used to select the correct growth and standard
                            deviation columns.
        risk_free_df (pd.DataFrame): A DataFrame containing historical risk-free
                                     rates, used to calculate excess returns.
        risk_pref (tuple): A tuple (risk_weight, return_weight) that defines
                           the user's preference for risk versus return.

    Returns:
        pd.DataFrame: A DataFrame with the added 'Utility_Score' column,
                      sorted in descending order by the score.
    """

    growth_col = f'Annual_Growth_{time_horizon}Y'
    std_col = f'Standard_Deviation_{time_horizon}Y'
    risk_w, return_w = risk_pref
    W_return = return_w / (return_w + risk_w)
    W_risk = risk_w / (return_w + risk_w)

    end = risk_free_df.index.max()
    start = end - pd.DateOffset(years=time_horizon)
    avg_rf = risk_free_df.loc[start:end, 'yield_pct'].mean()

    df = etf_df.dropna(subset=[growth_col, std_col]).copy()
    df['ExcessReturn'] = df[growth_col] - avg_rf  # in percent

    # Instead of z-score, use raw scaled values
    # Normalize excess return and std dev by their max (or mean)
    df['norm_excess'] = df['ExcessReturn'] / df['ExcessReturn'].max()
    df['norm_std'] = df[std_col] / df[std_col].max()

    df['Utility_Score'] = W_return * df['norm_excess'] - W_risk * df['norm_std']

    return df.sort_values('Utility_Score', ascending=False)
