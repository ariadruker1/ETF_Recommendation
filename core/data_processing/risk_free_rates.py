import requests
import pandas as pd
import streamlit as st


@st.cache_data(ttl=604800, show_spinner=False)
def fetch_risk_free_boc(start_date="1995-01-01"):
    """
    Downloads historical 3-month Treasury Bill secondary-market average yield from the Bank of Canada (BoC).

    This function fetches data from the BoC's Valet API for a specified date range,
    parses the JSON response, and returns a pandas DataFrame. The data is
    business-day interpolated to provide a daily risk-free rate.

    Args:
        start_date (str, optional): The start date for the data retrieval in
                                    'YYYY-MM-DD' format. Defaults to '1995-01-01'.

    Returns:
        pd.DataFrame: A DataFrame with the daily risk-free rates, indexed by date.
                      The DataFrame has a single column 'yield_pct' containing the
                      annualized percentage yield.

    Raises:
        RuntimeError: If there are issues fetching the data from the API,
                      the response is not valid JSON, or no observations are found.
    """
    
    url = f"https://www.bankofcanada.ca/valet/observations/V39079/json?start_date={start_date}"
    response = requests.get(url)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        # surface the HTTP error with context
        raise RuntimeError(
            f"Failed to fetch BoC data: {e}. Response text: {response.text[:500]}") from e

    try:
        payload = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"Response not valid JSON. Raw content starts with: {response.text[:500]}") from e

    observations = payload.get("observations")
    if not observations:
        raise RuntimeError(
            f"No observations in API response. Full payload: {payload}")

    # Attempt to auto-detect the series key (should be 'V39079')
    sample = observations[0]
    series_keys = [k for k in sample.keys() if k != "d"]
    if not series_keys:
        raise RuntimeError(f"No series key found in observation: {sample}")
    if len(series_keys) > 1:
        # unexpected, but pick the one that matches V39079 if present, else first
        if "V39079" in series_keys:
            series_key = "V39079"
        else:
            series_key = series_keys[0]
    else:
        series_key = series_keys[0]

    rows = []
    for obs in observations:
        date_str = obs.get("d")
        if date_str is None:
            continue
        try:
            rate_dict = obs.get(series_key, {})
            yield_pct = float(rate_dict.get("v"))
        except (TypeError, ValueError):
            # skip malformed/missing value
            continue
        date = pd.to_datetime(date_str)
        rows.append({"date": date, "yield_pct": yield_pct})

    if not rows:
        raise RuntimeError(
            f"Parsed zero rows from observations. Sample obs: {observations[:3]}. "
            f"Detected series key: {series_key}"
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df_daily = df.copy()
    return df_daily
