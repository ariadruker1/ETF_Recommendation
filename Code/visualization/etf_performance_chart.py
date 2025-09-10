
def create_etf_performance_chart(etf_recommend_df, data, chart_title):
    import pandas as pd
    import plotly.graph_objects as go

    fig = go.Figure()
    etf_tickers = etf_recommend_df['Ticker'].tolist()
    end_date = pd.Timestamp(datetime.now())

    # Step 1: Find first available date for each ETF
    first_dates = []
    for ticker in etf_tickers:
        if (ticker, 'Adj Close') in data.columns:
            series = data[(ticker, 'Adj Close')].dropna()
            if not series.empty:
                first_dates.append(series.index.min())

    if not first_dates:
        return fig  # No data to plot

    # Step 2: Youngest ETF determines common start date
    start_date = max(first_dates)

    # Step 3: Plot each ETF starting from the common start date
    for ticker in etf_tickers:
        if (ticker, 'Adj Close') not in data.columns:
            continue

        series = data[(ticker, 'Adj Close')].dropna()
        # Slice to common range
        series = series.loc[series.index >= start_date]
        if series.empty:
            continue

        # Normalize
        normalized = 100 * series / series.iloc[0]

        hover_text = [
            f"<b>{ticker}</b><br>Date: {idx.strftime('%Y-%m-%d')}<br>"
            f"Normalized Price: {val:.2f}"
            for idx, val in zip(normalized.index, normalized.values)
        ]

        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode='lines',
            name=ticker,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode='closest',
        height=400
    )
    return fig
