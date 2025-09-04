import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from core.data_processing.ishares_ETF_list import download_valid_data
from core.data_processing.etf_data import get_etf_data
from core.analysis.max_drawdown import calculate_max_drawdown
from core.data_processing.risk_free_rates import fetch_risk_free_boc
from config.constants import (
    USER_TIME_HORIZON, USER_DESIRED_GROWTH, USER_FLUCTUATION,
    USER_WORST_CASE, USER_MINIMUM_ETF_AGE, USER_RISK_PREFERENCE
)
from core.scoring.sharpe_recommendation import sharpe_score

# Global styles
st.markdown("""
<style>
body, div, label, span, p, small { font-size:16px !important; }
h1, h2, h3, h4, h5 { font-size:1.5em !important; }
.center-button {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.green-button button {
    background-color: #28a745 !important;
    color: white !important;
    font-size: 1.2em !important;
    padding: 0.5rem 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = [None] * 6

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

# ---------- STEP 0: Intro ----------
if st.session_state.step == 0:
    st.title("ETF Recommendations") 
    st.markdown("""
        <span style='color:gray;'>
        ℹ️ An ETF (Exchange-Traded Fund) is an investment that holds a mix of assets such as stocks, bonds, or other investments, 
        and trades on the stock market like a single stock. Because it combines many assets, it spreads out risk and is generally 
        less risky than investing in a single stock.
        </span>
        <hr>
        """, unsafe_allow_html=True)

    st.write("\nPlease answer the following questions to get personalized ETF recommendations based on your investment goals and risk tolerance:")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Find Investments!"):
            st.session_state.step = 1
            st.rerun()

# Options for all questions
time_horizon_options = [1, 4, 8, 15, 25]
desired_growth_options = [2, 5, 10, 16, 21]
fluctuation_options = [5, 10, 15, 20, 35]
worse_case_options = [15, 25, 35, 45, 100]
minimum_etf_age_options = [10, 5, 3, 1, 0]
risk_preference_options = [[3,1],[2,1],[1,1],[1,2],[1,3]]

# ---------- QUESTION STEPS ----------
if 1 <= st.session_state.step <= 5:
    progress = (st.session_state.step - 1) / 5
    st.progress(progress)
    st.write(f"Question {st.session_state.step} of 5")

# Step 1: Time Horizon
if st.session_state.step == 1:
    st.subheader("Time Horizon")
    st.markdown("""
    <p style='color:gray;'>
    ℹ️ How long do you plan to keep your money invested. Longer investments have more time to grow and recover. 
    Learn about <a href='https://www.investopedia.com/terms/t/timehorizon.asp' target='_blank'>Time Horizons (Investopedia)</a>.
    </p>
    """, unsafe_allow_html=True)

    options = ["Select a Time Horizon", 1, 2, 3, 4, 5]
    choice = st.selectbox("Investment time horizon?", options, format_func=lambda x: {
        "Select a Time Horizon": "Select a Time Horizon",
        1: "0-2 years", 2: "3-5 years", 3: "6-10 years", 4: "11-20 years", 5: "20+ years"
    }[x], key="q1")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Back", key="back_to_intro"):
            st.session_state.step = 0
            st.rerun()
    with col3:
        if st.button("Next", key="next1"):
            if choice == "Select a Time Horizon":
                st.warning("Please select a time horizon.")
            else:
                st.session_state.user_profile[USER_TIME_HORIZON] = time_horizon_options[choice-1]
                st.session_state.step = 2
                st.rerun()

# Step 2: Growth Goals
elif st.session_state.step == 2:
    st.subheader("Growth Goals")
    st.markdown("""
    <p style='color:gray;'>
    ℹ️ Investing with higher growth over time can help your money grow faster through compounding, but it also comes with increased risk. 
    Learn more about <a href='https://www.investopedia.com/terms/c/cagr.asp' target='_blank'>Compound Annual Growth Rates (Investopedia)</a>
    and <a href='https://www.investopedia.com/terms/i/inflation.asp' target='_blank'> Inflation (Investopedia)</a>.
    </p>
    """, unsafe_allow_html=True)

    options = ["Select Growth Goal", 1,2,3,4,5]
    choice = st.selectbox("Annual growth goals for how fast your money grows each year (before inflation)", options, format_func=lambda x:{
        "Select Growth Goal": "Select Growth Goal",
        1: "<3%: Slow (>24 years to double)",
        2: "3-7%: Modest (10-24 years to double)",
        3: "8-12%: Steady (6-9 years to double)",
        4: "13-20%: Fast (4-6 years to double)",
        5: ">20%: Very Fast (<4 years to double)"
    }[x], key="q2")
    
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Back", key="back2"):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Next", key="next2"):
            if choice == "Select Growth Goal":
                st.warning("Please select a growth goal.")
            else:
                st.session_state.user_profile[USER_DESIRED_GROWTH] = desired_growth_options[choice-1]
                st.session_state.step = 3
                st.rerun()

# Step 3: Risk Tolerance
elif st.session_state.step == 3:
    st.subheader("Risk Tolerance")
    st.markdown("""
    <p style='color:gray;'>
    ℹ️ Fluctuations show how much an investment can go up or down in a year. Bigger swings mean higher risk, but keeping money invested longer usually smooths these ups and downs.
    Learn more about <a href='https://www.investopedia.com/terms/r/risktolerance.asp' target='_blank'>Risk Tolerance (Investopedia)</a>.
    </p>
    """, unsafe_allow_html=True)

    options = ["Select Fluctuation Tolerance", 1,2,3,4,5]
    choice = st.selectbox("Annual fluctuation tolerance?", options, format_func=lambda x:{
        "Select Fluctuation Tolerance": "Select Fluctuation Tolerance",
        1: "Not much (<5%)",
        2: "Small ups/downs (<10%)",
        3: "Regular swings (<15%)",
        4: "Large moves are okay (<20%)",
        5: "Volatility doesn't bother me (>20%)"
    }[x], key="q3")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Back", key="back3"):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("Next", key="next3"):
            if choice == "Select Fluctuation Tolerance":
                st.warning("Please select fluctuation tolerance.")
            else:
                st.session_state.user_profile[USER_FLUCTUATION] = fluctuation_options[choice-1]
                st.session_state.step = 4
                st.rerun()

# Step 4: Maximum Loss Tolerance
elif st.session_state.step == 4:
    st.subheader("Maximum Loss Tolerance")
    st.markdown("""
    <p style='color:gray;'>
    ℹ️ Maximum drawdown is the largest drop your investment could experience, like buying at the highest point and selling at the lowest. 
    It helps the app only suggest ETFs that have historically stayed within your comfort zone.
    Learn more about <a href='https://www.investopedia.com/terms/m/maximumdrawdown.asp' target='_blank'>Maximum Drawdown (Investopedia)</a>.
    </p>
    """, unsafe_allow_html=True)

    options = ["Select Max Loss Tolerance", 1,2,3,4,5]
    choice = st.selectbox("Greatest loss you could tolerate?", options, format_func=lambda x:{
        "Select Max Loss Tolerance":"Select Max Loss Tolerance",
        1:"Low (<15%)",2:"Minor (<25%)",3:"Moderate (<35%)",4:"High (<45%)",5:"Very high (>45%)"
    }[x], key="q4")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Back", key="back4"):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Next", key="next4"):
            if choice=="Select Max Loss Tolerance":
                st.warning("Please select a maximum loss tolerance.")
            else:
                st.session_state.user_profile[USER_WORST_CASE] = worse_case_options[choice-1]
                st.session_state.step = 5
                st.rerun()

# Step 5: ETF Age / History
elif st.session_state.step == 5:
    st.subheader("ETF Performance History")
    st.markdown("""
    <p style='color:gray;'>
    ℹ️ ETF age shows how long it has been trading. Older ETFs have been through more market ups and downs, like the 2020 crash. Newer ETFs haven’t experienced these events, so we have less reliable data to base suggestions on.
    Learn about <a href='https://www.investopedia.com/terms/e/etf.asp' target='_blank'>ETFs (Investopedia)</a>
    and <a href='https://www.investopedia.com/financial-edge/0512/low-vs.-high-risk-investments-for-beginners.aspx' target='_blank'>Risk Determination (Investopedia)</a>.
    </p>
    """, unsafe_allow_html=True)

    options = ["Select ETF Age Minimum", 1,2,3,4,5]
    choice = st.selectbox("Minimum ETF age?", options, format_func=lambda x:{
        "Select ETF Age Minimum":"Select ETF Age Minimum",
        1:"Very Established (>10y)",2:"Moderately Established (>5y)",
        3:"Relatively New (>3y)",4:"New (>1y)",5:"No Min"
    }[x], key="q5")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Back", key="back5"):
            st.session_state.step = 4
            st.rerun()
    with col3:
        if st.button("Get My Recommendations", key="final", type="primary"):
            if choice=="Select ETF Age Minimum":
                st.warning("Please select an ETF age minimum.")
            else:
                st.session_state.user_profile[USER_MINIMUM_ETF_AGE] = minimum_etf_age_options[choice-1]
                st.session_state.user_profile[USER_RISK_PREFERENCE] = risk_preference_options[choice-1]
                st.session_state.step = 6
                st.rerun()

# Step 6: Recommendations
elif st.session_state.step == 6:
    st.subheader("🎯 Your Personalized ETF Recommendations")
    with st.spinner("Generating recommendations..."):
        try:
            user = st.session_state.user_profile
            valid_tickers, data = download_valid_data()
            end_date = pd.Timestamp(datetime.now())
            md_tolerable_list = calculate_max_drawdown(
                user[USER_WORST_CASE], user[USER_MINIMUM_ETF_AGE], valid_tickers, data, end_date
            )
            etf_metrics = get_etf_data(md_tolerable_list, user[USER_TIME_HORIZON], data, end_date, min_etf_age=user[USER_MINIMUM_ETF_AGE])
            risk_free_data = fetch_risk_free_boc("1995-01-01")
            etf_sharpe = sharpe_score(etf_metrics, user[USER_TIME_HORIZON], risk_free_data)

            st.success("✅ Analysis complete!")

            st.subheader("📈 Recommendations")
            if not etf_sharpe.empty:
                st.plotly_chart(create_etf_performance_chart(etf_sharpe, data,
                                                            f"Top 5 ETFs:"), width='stretch')
            else:
                st.warning("No Sharpe-based ETFs found.")

            st.subheader("📊 Detailed Metrics")
            growth_col = f'Annual_Growth_{user[USER_TIME_HORIZON]}Y'
            std_col = f'Standard_Deviation_{user[USER_TIME_HORIZON]}Y'

            if not etf_sharpe.empty:
                sharpe_simple = etf_sharpe[['Ticker', growth_col, std_col]].reset_index(drop=True)
                sharpe_simple.columns = ['Ticker','Annual Growth (%)','Standard Deviation (%)']

                # Create HTML table with links
                table_html = "<table>"
                table_html += "<tr><th>Ticker</th><th>Annual Growth (%)</th><th>Standard Deviation (%)</th></tr>"
                for _, row in sharpe_simple.iterrows():
                    ticker = row['Ticker']
                    growth = row['Annual Growth (%)']
                    std = row['Standard Deviation (%)']
                    link = f"https://ca.finance.yahoo.com/quote/{ticker}/"
                    table_html += f"<tr><td><a href='{link}' target='_blank'>{ticker}</a></td><td>{growth}</td><td>{std}</td></tr>"
                table_html += "</table>"

                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.write("No data available")

            if st.button("Start Over", key="restart"):
                st.session_state.step = 0
                st.session_state.user_profile = [None]*6
                st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if st.button("Try Again", key="retry"):
                st.session_state.step = 0
                st.rerun()

st.markdown("<small style='color:gray; margin-top:2rem;'>*For educational use only. Not financial advice.</small>", unsafe_allow_html=True)
