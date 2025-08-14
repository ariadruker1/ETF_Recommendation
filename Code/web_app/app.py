# web_app/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ---------- Helper: Chart ----------
def create_etf_performance_chart(etf_recommend_df, data, time_horizon, chart_title):
    fig = go.Figure()
    etf_tickers = etf_recommend_df['Ticker'].tolist()
    end_date = pd.Timestamp(datetime.now())
    start_date = end_date - pd.DateOffset(years=time_horizon)

    for ticker in etf_tickers:
        try:
            price_series = None
            try:
                price_series = data[(ticker, 'Adj Close')].dropna()
            except (KeyError, TypeError):
                try:
                    price_series = data[ticker].dropna()
                except (KeyError, TypeError):
                    continue

            period_prices = price_series.loc[start_date:end_date]
            if period_prices.empty:
                continue

            normalized_prices = 100 * period_prices / period_prices.iloc[0]

            etf_metrics = etf_recommend_df[etf_recommend_df['Ticker'] == ticker].iloc[0]
            growth_col = f'Annual_Growth_{time_horizon}Y'
            std_col = f'Standard_Deviation_{time_horizon}Y'

            hover_text = [
                f"<b>{ticker}</b><br>Date: {date.strftime('%Y-%m-%d')}<br>"
                f"Annual Growth: {etf_metrics[growth_col]:.2f}%<br>"
                f"Standard Deviation: {etf_metrics[std_col]:.2f}%"
                for date, price in zip(normalized_prices.index, normalized_prices.values)
            ]

            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices.values,
                mode='lines',
                name=ticker,
                line=dict(width=2),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            ))
        except Exception:
            continue

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode='closest',
        height=400
    )
    return fig

# ---------- Session State ----------
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = [None] * 6

# ---------- Main ----------
st.title("ETF Recommendations")
st.write("Answer a few questions to get personalized ETF recommendations.*")

# ---------- Options ----------
time_horizon_options = [1, 4, 8, 15, 25]
desired_growth_options = [2, 5, 10, 16, 21]
fluctuation_options = [5, 10, 15, 20, 35]
worse_case_options = [15, 25, 35, 45, 100]
minimum_etf_age_options = [10, 5, 3, 1, 0]
risk_preference_options = [[3,1],[2,1],[1,1],[1,2],[1,3]]

# ---------- Progress Bar ----------
if st.session_state.step <= 5:
    progress = (st.session_state.step - 1) / 5
    st.progress(progress)
    st.write(f"Question {st.session_state.step} of 5")

# ---------- Step 1 ----------
if st.session_state.step == 1:
    st.subheader("Time Horizon")
    st.markdown("<small style='color:gray;'>How long you plan to keep your money invested.</small>", unsafe_allow_html=True)
    options = ["Select a Time Horizon", 1, 2, 3, 4, 5]
    choice = st.selectbox("Investment time horizon?", options, format_func=lambda x: {
        "Select a Time Horizon": "Select a Time Horizon",
        1: "0-2 years", 2: "3-5 years", 3: "6-10 years", 4: "11-20 years", 5: "20+ years"
    }[x], key="q1")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Next", key="next1"):
            if choice == "Select a Time Horizon":
                st.warning("Please select a time horizon.")
            else:
                st.session_state.user_profile[USER_TIME_HORIZON] = time_horizon_options[choice-1]
                st.session_state.step = 2
                st.rerun()

# ---------- Step 2 ----------
elif st.session_state.step == 2:
    st.subheader("Growth Goals")
    st.markdown("<small style='color:gray;'>Desired annual growth.</small>", unsafe_allow_html=True)
    options = ["Select Growth Goal", 1,2,3,4,5]
    choice = st.selectbox("Annual growth goals?", options, format_func=lambda x:{
        "Select Growth Goal": "Select Growth Goal",
        1: "Beat inflation (<3%)",
        2: "Modest and reliable (3-7%)",
        3: "Steady longterm (8-12%)",
        4: "Strong returns with moderate risk (13-20%)",
        5: "High growth (>20%)"
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

# ---------- Step 3 ----------
elif st.session_state.step == 3:
    st.subheader("Risk Tolerance")
    st.markdown("<small style='color:gray;'>How much ups and downs you can tolerate.</small>", unsafe_allow_html=True)
    options = ["Select Fluctuation Tolerance", 1,2,3,4,5]
    choice = st.selectbox("Annual fluctuation tolerance?", options, format_func=lambda x:{
        "Select Fluctuation Tolerance": "Select Fluctuation Tolerance",
        1: "Not much (<5%)",
        2: "Small ups/downs (<10%)",
        3: "Regular swings (<15%)",
        4: "Large moves okay (<20%)",
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

# ---------- Step 4 ----------
elif st.session_state.step == 4:
    st.subheader("Maximum Loss Tolerance")
    st.markdown("<small style='color:gray;'>Largest loss you could tolerate.</small>", unsafe_allow_html=True)
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

# ---------- Step 5 ----------
elif st.session_state.step == 5:
    st.subheader("ETF Track Record")
    st.markdown("<small style='color:gray;'>Older ETFs have a known performance history.</small>", unsafe_allow_html=True)
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

# ---------- Step 6: Recommendations ----------
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

            # Chart
            st.subheader("📈 Sharpe Recommendations")
            if not etf_sharpe.empty:
                st.plotly_chart(create_etf_performance_chart(etf_sharpe, data, user[USER_TIME_HORIZON],
                                                            f"Top 5 ETFs by Sharpe ({user[USER_TIME_HORIZON]}y)"), use_container_width=True)
            else:
                st.warning("No Sharpe-based ETFs found.")

            # Metrics table
            st.subheader("📊 Detailed Metrics")
            growth_col = f'Annual_Growth_{user[USER_TIME_HORIZON]}Y'
            std_col = f'Standard_Deviation_{user[USER_TIME_HORIZON]}Y'

            if not etf_sharpe.empty:
                sharpe_simple = etf_sharpe[['Ticker', growth_col, std_col]].reset_index(drop=True)
                sharpe_simple.columns = ['Ticker','Annual Growth (%)','Standard Deviation (%)']
                st.dataframe(sharpe_simple, use_container_width=True, hide_index=True)
            else:
                st.write("No data available")

            if st.button("Start Over", key="restart"):
                st.session_state.step = 1
                st.session_state.user_profile = [None]*6
                st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if st.button("Try Again", key="retry"):
                st.session_state.step = 1
                st.rerun()

# ---------- Footer ----------
st.markdown("<small style='color:gray; margin-top:2rem;'>*Educational only. Not financial advice.*</small>", unsafe_allow_html=True)
