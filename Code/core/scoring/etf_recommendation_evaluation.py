def top_recommend(df, column_title, amount):
    """
    Identifies and returns the top performing ETFs based on a specified metric.

    This function takes a DataFrame, a column title representing a metric
    (e.g., 'Sharpe_Ratio' or 'Utility_Score'), and an amount, and returns the
    ETFs with the highest values for that metric. The function first handles
    any missing values and then sorts the DataFrame in descending order before
    returning the top entries.

    Args:
        df (pd.DataFrame): The DataFrame containing ETF data and metrics.
        column_title (str): The name of the column to use for sorting and ranking
                            the ETFs.
        amount (int): The number of top ETFs to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top `amount` ETFs, sorted
                      by the specified `column_title` in descending order.

    Raises:
        ValueError: If the `column_title` is not found in the DataFrame.
    """
 
    if column_title not in df.columns:
        raise ValueError(f"Column '{column_title}' not found in DataFrame.")

    # Drop rows with NaN Sharpe ratios to avoid errors in sorting
    df_clean = df.dropna(subset=[column_title])

    # Sort descending by Sharpe ratio
    df_sorted = df_clean.sort_values(by=column_title, ascending=False)

    result = df_sorted.head(amount)

    return result
