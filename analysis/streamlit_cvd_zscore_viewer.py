"""
CVD Z-Score Viewer
Visualize CVD z-score behavior across earnings events.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="CVD Z-Score Viewer",
    page_icon="📈",
    layout="wide"
)

# Default data path
DEFAULT_DATA_PATH = "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv"

# Consistent color palette for tickers
TICKER_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1


# ========== Helper Functions for Early Event Meta Features ==========

def calculate_delta_efficiency(event_slice_5m, event_price):
    """Delta Efficiency: Price change per unit of net order flow (0-5m)."""
    if event_slice_5m.empty or pd.isna(event_price) or event_price == 0:
        return np.nan
    
    # Get price change over first 5m
    if 'close' not in event_slice_5m.columns:
        return np.nan
    
    price_start = event_slice_5m['close'].iloc[0] if len(event_slice_5m) > 0 else np.nan
    price_end = event_slice_5m['close'].iloc[-1] if len(event_slice_5m) > 0 else np.nan
    
    if pd.isna(price_start) or pd.isna(price_end):
        return np.nan
    
    price_change_pct = (price_end - price_start) / event_price * 100
    
    # Get cumulative delta (cvd) change
    if 'cvd_since_event' not in event_slice_5m.columns:
        return np.nan
    
    cvd_start = event_slice_5m['cvd_since_event'].iloc[0] if len(event_slice_5m) > 0 else 0
    cvd_end = event_slice_5m['cvd_since_event'].iloc[-1] if len(event_slice_5m) > 0 else 0
    
    if pd.isna(cvd_start):
        cvd_start = 0
    if pd.isna(cvd_end):
        return np.nan
    
    net_delta = cvd_end - cvd_start
    
    if net_delta == 0:
        return np.nan
    
    # Normalize by total volume for comparability
    if 'volume' in event_slice_5m.columns:
        total_vol = event_slice_5m['volume'].sum()
        if total_vol > 0:
            net_delta_normalized = net_delta / total_vol
            if net_delta_normalized != 0:
                return price_change_pct / net_delta_normalized
    
    return price_change_pct / net_delta if net_delta != 0 else np.nan


def calculate_cvd_monotonicity_sign(event_slice_5m):
    """CVD Monotonicity (Sign): % of bars where CVD moves in same direction."""
    if 'cvd_since_event' not in event_slice_5m.columns or len(event_slice_5m) < 2:
        return np.nan
    
    cvd = event_slice_5m['cvd_since_event'].dropna()
    if len(cvd) < 2:
        return np.nan
    
    diffs = cvd.diff().dropna()
    if len(diffs) == 0:
        return np.nan
    
    # Count positive and negative moves
    pos_moves = (diffs > 0).sum()
    neg_moves = (diffs < 0).sum()
    total_moves = pos_moves + neg_moves
    
    if total_moves == 0:
        return np.nan
    
    # Return the dominant direction proportion
    return max(pos_moves, neg_moves) / total_moves


def calculate_cvd_monotonicity_net_gross(event_slice_5m):
    """CVD Monotonicity (Net/Gross): |Net CVD change| / Sum of |changes|."""
    if 'cvd_since_event' not in event_slice_5m.columns or len(event_slice_5m) < 2:
        return np.nan
    
    cvd = event_slice_5m['cvd_since_event'].dropna()
    if len(cvd) < 2:
        return np.nan
    
    diffs = cvd.diff().dropna()
    if len(diffs) == 0:
        return np.nan
    
    gross = diffs.abs().sum()
    if gross == 0:
        return np.nan
    
    net = abs(diffs.sum())
    return net / gross


def calculate_price_cvd_ratio(range_pct, z_max, z_min):
    """Price Range vs CVD Range: price range / cvd zscore range."""
    if pd.isna(range_pct) or pd.isna(z_max) or pd.isna(z_min):
        return np.nan
    
    z_range = z_max - z_min
    if z_range == 0:
        return np.nan
    
    return range_pct / z_range


def calculate_vwap_deviation_5m(event_slice):
    """VWAP Deviation at 5m: |price - vwap| / price at 5m mark."""
    if event_slice.empty:
        return np.nan
    
    # Find the row closest to 5m (300s)
    slice_5m = event_slice[event_slice['seconds_since_event'] <= 300]
    if slice_5m.empty:
        return np.nan
    
    last_row = slice_5m.iloc[-1]
    
    if 'close' not in slice_5m.columns or 'vwap_since_event' not in slice_5m.columns:
        return np.nan
    
    price = last_row['close']
    vwap = last_row['vwap_since_event']
    
    if pd.isna(price) or pd.isna(vwap) or price == 0:
        return np.nan
    
    return abs(price - vwap) / price * 100


def calculate_vol_front_loading(event_slice_5m):
    """Vol Front-Loading: Volume in first 2m / Volume in 2-5m."""
    if event_slice_5m.empty or 'volume' not in event_slice_5m.columns:
        return np.nan
    
    vol_0_2 = event_slice_5m[event_slice_5m['seconds_since_event'] <= 120]['volume'].sum()
    vol_2_5 = event_slice_5m[(event_slice_5m['seconds_since_event'] > 120) & 
                             (event_slice_5m['seconds_since_event'] <= 300)]['volume'].sum()
    
    if vol_2_5 == 0:
        return np.nan
    
    return vol_0_2 / vol_2_5


@st.cache_data
def load_data(data_path: str):
    """Load the consolidated earnings data."""
    try:
        df = pd.read_csv(data_path)
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'])
        if 'year' not in df.columns:
            df['year'] = df['acceptance_datetime_utc'].dt.year
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    st.title("📈 CVD Z-Score Viewer")
    st.markdown("Explore CVD z-score behavior across earnings events")
    
    # ========== SIDEBAR: Data Source ==========
    st.sidebar.header("📂 Data Source")
    data_path = st.sidebar.text_input(
        "Data File Path",
        value=DEFAULT_DATA_PATH,
        help="Path to the consolidated earnings CSV file",
        key="data_path"
    )
    
    # Show required columns info
    with st.sidebar.expander("ℹ️ Required Columns"):
        st.markdown("""
        **Critical columns:**
        - `acceptance_datetime_utc`
        - `ticker`, `year`
        - `seconds_since_event`
        - `close`, `cvd_zscore`
        - `cvd_since_event`
        - `target_ret_600s`
        
        **Optional columns:**
        - `event_price`, `vwap_since_event`
        - `vw`, `first_5m_range_pct`
        - `Surprise(%)`, `volume`
        - `target_ret_120s`, `target_ret_240s`
        """)
    
    # Load data
    df = load_data(data_path)
    if df.empty:
        st.error("No data available. Please check the data path.")
        return

    # ========== SIDEBAR: Filters ==========
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Filters")
    
    # Year filter
    available_years = sorted(df['year'].dropna().unique().astype(int))
    selected_years = st.sidebar.multiselect(
        "Years",
        options=available_years,
        default=[2025] if 2025 in available_years else available_years[-1:],
        key="year_filter"
    )
    
    # Time range filter
    st.sidebar.markdown("**Time Range (seconds since event)**")
    time_min = st.sidebar.number_input("From (sec)", min_value=0, max_value=7200, value=300, step=60, key="time_min")
    time_max = st.sidebar.number_input("To (sec)", min_value=0, max_value=7200, value=1800, step=60, key="time_max")
    st.sidebar.caption(f"{time_min/60:.1f} min to {time_max/60:.1f} min")
    
    # Quick presets
    preset = st.sidebar.selectbox(
        "Quick Presets",
        ["Custom", "First 5 min", "First 30 min", "First 60 min", "30-60 min", "60-120 min"],
        index=0,
        key="time_preset"
    )
    
    if preset == "First 5 min":
        time_min, time_max = 0, 300
    elif preset == "First 30 min":
        time_min, time_max = 0, 1800
    elif preset == "First 60 min":
        time_min, time_max = 0, 3600
    elif preset == "30-60 min":
        time_min, time_max = 1800, 3600
    elif preset == "60-120 min":
        time_min, time_max = 3600, 7200
    
    # Filter data
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['seconds_since_event'] >= time_min) &
        (df['seconds_since_event'] <= time_max)
    ].copy()
    
    # Show filter stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Data Stats**")
    st.sidebar.caption(f"Rows: {len(filtered_df):,}")
    st.sidebar.caption(f"Tickers: {filtered_df['ticker'].nunique()}")
    st.sidebar.caption(f"Events: {filtered_df.groupby(['ticker', 'acceptance_datetime_utc']).ngroups}")

    # ========== MAIN CONTENT ==========
    tab1, tab8, tab2, tab3, tab4, tab5, tab6, tab7, tab9 = st.tabs([
        "📊 CVD Z-Score Trajectories & Distributions",
        "🔄 0-5m vs 5-30m Comparison",
        "📉 Ticker Correlations",
        "🔍 Event Deep Dive",
        "🎯 Correlation Group Analysis",
        "📈 Event Correlation Deep Dive",
        "📐 VWAP Deviation Explorer",
        "🔬 First 5m Signals Explorer",
        "🎲 Logistic Regression"
    ])
    
    # ========== TAB 1: Events by Ticker ==========
    with tab1:
        st.info("📊 **Overview**: Visualize CVD Z-Score trajectories over time for multiple events, and compare distributions across tickers.")
        st.markdown("### CVD Z-Score vs Time Since Event")
        st.markdown("Each line represents one earnings event for the selected ticker(s)")
        
        # Ticker selector
        available_tickers = sorted(filtered_df['ticker'].unique())
        default_tickers = available_tickers[:5] if len(available_tickers) >= 5 else available_tickers
        selected_tickers = st.multiselect(
            "Select Tickers",
            options=available_tickers,
            default=default_tickers,
            key="tab1_tickers"
        )
        
        if not selected_tickers:
            st.info("Please select at least one ticker.")
        else:
            ticker_df = filtered_df[filtered_df['ticker'].isin(selected_tickers)]
            
            # Create event identifier
            ticker_df = ticker_df.copy()
            ticker_df['event_id'] = ticker_df['ticker'] + ' | ' + ticker_df['acceptance_datetime_utc'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Trajectory chart
            fig = px.line(
                ticker_df,
                x='seconds_since_event',
                y='cvd_zscore',
                color='ticker',
                line_group='event_id',
                color_discrete_sequence=TICKER_COLORS,
                labels={
                    'seconds_since_event': 'Seconds Since Event',
                    'cvd_zscore': 'CVD Z-Score',
                    'ticker': 'Ticker'
                },
                title="CVD Z-Score Trajectories"
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(
                height=500,
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True, key="tab1_trajectory")
            
            # Distribution plot
            st.markdown("### CVD Z-Score Distribution by Ticker")
            fig_dist = px.box(
                ticker_df,
                x='ticker',
                y='cvd_zscore',
                color='ticker',
                color_discrete_sequence=TICKER_COLORS,
                title="CVD Z-Score Distribution"
            )
            fig_dist.update_layout(
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="tab1_distribution")
            
            # Summary stats
            st.markdown("### Summary Statistics")
            stats = ticker_df.groupby('ticker')['cvd_zscore'].agg(['count', 'mean', 'std', 'min', 'max'])
            stats = stats.round(3)
            st.dataframe(stats, use_container_width=True)
        
        # Overall distribution histogram (all tickers, year filter only)
        st.markdown("---")
        st.markdown("### Overall CVD Z-Score Distribution (All Tickers)")
        
        n_bins = st.slider(
            "Number of bins",
            min_value=10, max_value=200, value=50, step=10,
            key="tab1_nbins"
        )
        
        # Filter: year only, all tickers
        dist_df_0_5 = df[
            (df['year'].isin(selected_years)) &
            (df['seconds_since_event'] >= 0) &
            (df['seconds_since_event'] <= 300)
        ].copy()
        
        dist_df_5_30 = df[
            (df['year'].isin(selected_years)) &
            (df['seconds_since_event'] > 300) &
            (df['seconds_since_event'] <= 1800)
        ].copy()
        
        col_dist1, col_dist2 = st.columns(2)
        
        # 0-5m Distribution
        with col_dist1:
            if dist_df_0_5.empty:
                st.warning("No data for 0-5m window.")
            else:
                n_tickers_0_5 = dist_df_0_5['ticker'].nunique()
                n_events_0_5 = dist_df_0_5.groupby(['ticker', 'acceptance_datetime_utc']).ngroups
                
                fig_hist_0_5 = px.histogram(
                    dist_df_0_5,
                    x='cvd_zscore',
                    nbins=n_bins,
                    title=f"0-5m | {n_tickers_0_5} tickers, {n_events_0_5} events",
                    labels={'cvd_zscore': 'CVD Z-Score', 'count': 'Count'}
                )
                fig_hist_0_5.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
                fig_hist_0_5.add_vline(
                    x=dist_df_0_5['cvd_zscore'].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"μ={dist_df_0_5['cvd_zscore'].mean():.2f}"
                )
                fig_hist_0_5.update_layout(height=400, template='plotly_white')
                fig_hist_0_5.update_traces(marker_color='#f97316')  # Orange
                st.plotly_chart(fig_hist_0_5, use_container_width=True, key="tab1_hist_0_5")
                
                # Stats for 0-5m
                st.caption(f"n={len(dist_df_0_5):,} | σ={dist_df_0_5['cvd_zscore'].std():.2f} | skew={dist_df_0_5['cvd_zscore'].skew():.2f}")
        
        # 5-30m Distribution
        with col_dist2:
            if dist_df_5_30.empty:
                st.warning("No data for 5-30m window.")
            else:
                n_tickers_5_30 = dist_df_5_30['ticker'].nunique()
                n_events_5_30 = dist_df_5_30.groupby(['ticker', 'acceptance_datetime_utc']).ngroups
                
                fig_hist_5_30 = px.histogram(
                    dist_df_5_30,
                    x='cvd_zscore',
                    nbins=n_bins,
                    title=f"5-30m | {n_tickers_5_30} tickers, {n_events_5_30} events",
                    labels={'cvd_zscore': 'CVD Z-Score', 'count': 'Count'}
                )
                fig_hist_5_30.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
                fig_hist_5_30.add_vline(
                    x=dist_df_5_30['cvd_zscore'].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"μ={dist_df_5_30['cvd_zscore'].mean():.2f}"
                )
                fig_hist_5_30.update_layout(height=400, template='plotly_white')
                fig_hist_5_30.update_traces(marker_color='#3b82f6')  # Blue
                st.plotly_chart(fig_hist_5_30, use_container_width=True, key="tab1_hist_5_30")
                
                # Stats for 5-30m
                st.caption(f"n={len(dist_df_5_30):,} | σ={dist_df_5_30['cvd_zscore'].std():.2f} | skew={dist_df_5_30['cvd_zscore'].skew():.2f}")

    # ========== TAB 3: Single Event ==========
    with tab3:
        st.info("🔍 **Overview**: Deep dive into a single earnings event. Compare CVD Z-Score with price return and forward 10min return over time.")
        st.markdown("### Single Event Analysis")
        
        # Event selector
        col1, col2 = st.columns(2)
        with col1:
            available_tickers_t2 = sorted(filtered_df['ticker'].unique())
            selected_ticker_t2 = st.selectbox(
                "Select Ticker",
                options=available_tickers_t2,
                key="tab2_ticker"
            )
        
        if selected_ticker_t2:
            ticker_events = filtered_df[filtered_df['ticker'] == selected_ticker_t2]['acceptance_datetime_utc'].unique()
            ticker_events = sorted(ticker_events)
            
            with col2:
                selected_event = st.selectbox(
                    "Select Event",
                    options=ticker_events,
                    format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'),
                    key="tab2_event"
                )
            
            if selected_event:
                event_df = filtered_df[
                    (filtered_df['ticker'] == selected_ticker_t2) &
                    (filtered_df['acceptance_datetime_utc'] == selected_event)
                ].sort_values('seconds_since_event')
                
                if event_df.empty:
                    st.warning("No data for selected event in current time range.")
                else:
                    # Calculate price return since event
                    event_price = event_df['event_price'].iloc[0] if 'event_price' in event_df.columns else np.nan
                    if pd.notna(event_price) and event_price > 0:
                        event_df['price_return_pct'] = ((event_df['close'] - event_price) / event_price) * 100
                    else:
                        event_df['price_return_pct'] = np.nan
                    
                    # Forward 10min return
                    if 'target_ret_600s' in event_df.columns:
                        event_df['fwd_10m_return_pct'] = event_df['target_ret_600s'] * 100
                    else:
                        event_df['fwd_10m_return_pct'] = np.nan
                    
                    # Chart 1: CVD Z-Score vs Return Since Event
                    st.markdown("#### CVD Z-Score vs Return Since Event")
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig1.add_trace(
                        go.Scatter(
                            x=event_df['seconds_since_event'],
                            y=event_df['cvd_zscore'],
                            mode='lines',
                            name='CVD Z-Score',
                            line=dict(color='#3b82f6', width=2)
                        ),
                        secondary_y=False
                    )
                    
                    fig1.add_trace(
                        go.Scatter(
                            x=event_df['seconds_since_event'],
                            y=event_df['price_return_pct'],
                            mode='lines',
                            name='Return Since Event %',
                            line=dict(color='#22c55e', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    fig1.update_layout(
                        title=f"{selected_ticker_t2} - CVD Z-Score vs Price Return",
                        height=400,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    fig1.update_xaxes(title_text="Seconds Since Event")
                    fig1.update_yaxes(title_text="CVD Z-Score", secondary_y=False)
                    fig1.update_yaxes(title_text="Return Since Event (%)", secondary_y=True)
                    
                    st.plotly_chart(fig1, use_container_width=True, key="tab2_chart1")
                    
                    # Chart 2: CVD Z-Score vs Forward 10min Return
                    st.markdown("#### CVD Z-Score vs Forward 10min Return")
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig2.add_trace(
                        go.Scatter(
                            x=event_df['seconds_since_event'],
                            y=event_df['cvd_zscore'],
                            mode='lines',
                            name='CVD Z-Score',
                            line=dict(color='#3b82f6', width=2)
                        ),
                        secondary_y=False
                    )
                    
                    fig2.add_trace(
                        go.Scatter(
                            x=event_df['seconds_since_event'],
                            y=event_df['fwd_10m_return_pct'],
                            mode='lines',
                            name='Fwd 10min Return %',
                            line=dict(color='#f97316', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    fig2.update_layout(
                        title=f"{selected_ticker_t2} - CVD Z-Score vs Forward 10min Return",
                        height=400,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    fig2.update_xaxes(title_text="Seconds Since Event")
                    fig2.update_yaxes(title_text="CVD Z-Score", secondary_y=False)
                    fig2.update_yaxes(title_text="Fwd 10min Return (%)", secondary_y=True)
                    
                    st.plotly_chart(fig2, use_container_width=True, key="tab2_chart2")
                    
                    # Chart 3: VW Price vs CVD Z-Score
                    st.markdown("#### Price Action (VW) vs CVD Z-Score")
                    if 'vw' in event_df.columns and event_df['vw'].notna().any():
                        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig3.add_trace(
                            go.Scatter(
                                x=event_df['seconds_since_event'],
                                y=event_df['vw'],
                                mode='lines',
                                name='VW Price',
                                line=dict(color='#8b5cf6', width=2)
                            ),
                            secondary_y=False
                        )
                        
                        fig3.add_trace(
                            go.Scatter(
                                x=event_df['seconds_since_event'],
                                y=event_df['cvd_zscore'],
                                mode='lines',
                                name='CVD Z-Score',
                                line=dict(color='#3b82f6', width=2)
                            ),
                            secondary_y=True
                        )
                        
                        fig3.update_layout(
                            title=f"{selected_ticker_t2} - VW Price vs CVD Z-Score",
                            height=400,
                            template='plotly_white',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                        )
                        fig3.update_xaxes(title_text="Seconds Since Event")
                        fig3.update_yaxes(title_text="VW Price ($)", secondary_y=False)
                        fig3.update_yaxes(title_text="CVD Z-Score", secondary_y=True)
                        
                        st.plotly_chart(fig3, use_container_width=True, key="tab2_chart3")
                    else:
                        st.info("VW (volume-weighted price) column not available in data.")
                    
                    # Summary stats
                    st.markdown("#### Event Summary")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("CVD Z-Score Range", f"{event_df['cvd_zscore'].min():.2f} to {event_df['cvd_zscore'].max():.2f}")
                    with col_s2:
                        ret_range = event_df['price_return_pct'].dropna()
                        if not ret_range.empty:
                            st.metric("Return Since Event Range", f"{ret_range.min():.2f}% to {ret_range.max():.2f}%")
                    with col_s3:
                        fwd_range = event_df['fwd_10m_return_pct'].dropna()
                        if not fwd_range.empty:
                            st.metric("Fwd 10min Return Range", f"{fwd_range.min():.2f}% to {fwd_range.max():.2f}%")
                    
                    # ========== 5-30m Distribution Estimator ==========
                    st.markdown("---")
                    st.markdown("#### 📐 5-30m CVD Z-Score Distribution Estimator")
                    st.markdown("*Estimate the 5-30m distribution based on 0-5m observations using percentile-specific regression.*")
                    
                    # Get full event data (not filtered by sidebar time range)
                    full_event_df = df[
                        (df['ticker'] == selected_ticker_t2) &
                        (df['acceptance_datetime_utc'] == selected_event)
                    ].sort_values('seconds_since_event')
                    
                    # Get 0-5m and 5-30m data for this event
                    event_0_5 = full_event_df[(full_event_df['seconds_since_event'] >= 0) & 
                                               (full_event_df['seconds_since_event'] <= 300)]['cvd_zscore'].dropna()
                    event_5_30 = full_event_df[(full_event_df['seconds_since_event'] > 300) & 
                                                (full_event_df['seconds_since_event'] <= 1800)]['cvd_zscore'].dropna()
                    
                    if len(event_0_5) < 5:
                        st.warning("Insufficient 0-5m data for this event.")
                    else:
                        # Calculate global regression parameters from all events in selected years
                        df_global = df[df['year'].isin(selected_years)].copy()
                        global_stats = []
                        min_obs_global = 5
                        
                        for (t, e), grp in df_global.groupby(['ticker', 'acceptance_datetime_utc']):
                            grp = grp.sort_values('seconds_since_event')
                            w0_5 = grp[(grp['seconds_since_event'] >= 0) & (grp['seconds_since_event'] <= 300)]['cvd_zscore'].dropna()
                            w5_30 = grp[(grp['seconds_since_event'] > 300) & (grp['seconds_since_event'] <= 1800)]['cvd_zscore'].dropna()
                            if len(w0_5) >= min_obs_global and len(w5_30) >= min_obs_global:
                                global_stats.append({
                                    'mean_0_5': w0_5.mean(),
                                    'mean_5_30': w5_30.mean(),
                                    'p5_0_5': np.percentile(w0_5, 5),
                                    'p5_5_30': np.percentile(w5_30, 5),
                                    'p10_0_5': np.percentile(w0_5, 10),
                                    'p10_5_30': np.percentile(w5_30, 10),
                                    'p90_0_5': np.percentile(w0_5, 90),
                                    'p90_5_30': np.percentile(w5_30, 90),
                                    'p95_0_5': np.percentile(w0_5, 95),
                                    'p95_5_30': np.percentile(w5_30, 95),
                                })
                        
                        if len(global_stats) < 10:
                            st.warning("Insufficient events to calculate global parameters.")
                        else:
                            global_df = pd.DataFrame(global_stats)
                            
                            # Calculate regression for each metric
                            metrics = [
                                ('Mean', 'mean_0_5', 'mean_5_30'),
                                ('P5', 'p5_0_5', 'p5_5_30'),
                                ('P10', 'p10_0_5', 'p10_5_30'),
                                ('P90', 'p90_0_5', 'p90_5_30'),
                                ('P95', 'p95_0_5', 'p95_5_30'),
                            ]
                            
                            reg_params = {}
                            for name, col_0_5, col_5_30 in metrics:
                                beta_m, alpha_m = np.polyfit(global_df[col_0_5], global_df[col_5_30], 1)
                                reg_params[name] = {'alpha': alpha_m, 'beta': beta_m}
                            
                            # Display regression parameters table
                            st.markdown("**Regression Parameters** (from all events in selected years)")
                            st.latex(r"Y_{5\text{-}30} = \alpha + \beta \times X_{0\text{-}5}")
                            
                            param_data = []
                            for name in ['Mean', 'P5', 'P10', 'P90', 'P95']:
                                param_data.append({
                                    'Metric': name,
                                    'α': reg_params[name]['alpha'],
                                    'β': reg_params[name]['beta']
                                })
                            param_df = pd.DataFrame(param_data)
                            st.dataframe(
                                param_df.style.format({'α': '{:.4f}', 'β': '{:.4f}'}).background_gradient(
                                    subset=['β'], cmap='RdYlGn', vmin=0, vmax=1.5
                                ),
                                use_container_width=True
                            )
                            
                            # Calculate this event's 0-5m values
                            event_values_0_5 = {
                                'Mean': event_0_5.mean(),
                                'P5': np.percentile(event_0_5, 5),
                                'P10': np.percentile(event_0_5, 10),
                                'P90': np.percentile(event_0_5, 90),
                                'P95': np.percentile(event_0_5, 95),
                            }
                            
                            # Calculate estimated 5-30m values
                            est_values_5_30 = {}
                            for name in ['Mean', 'P5', 'P10', 'P90', 'P95']:
                                alpha_m = reg_params[name]['alpha']
                                beta_m = reg_params[name]['beta']
                                est_values_5_30[name] = alpha_m + beta_m * event_values_0_5[name]
                            
                            # Display this event's values
                            st.markdown("**This Event's Statistics**")
                            
                            # Create comparison table
                            comparison_data = []
                            has_actual = len(event_5_30) >= 5
                            
                            if has_actual:
                                actual_values_5_30 = {
                                    'Mean': event_5_30.mean(),
                                    'P5': np.percentile(event_5_30, 5),
                                    'P10': np.percentile(event_5_30, 10),
                                    'P90': np.percentile(event_5_30, 90),
                                    'P95': np.percentile(event_5_30, 95),
                                }
                            
                            for name in ['Mean', 'P5', 'P10', 'P90', 'P95']:
                                row = {
                                    'Metric': name,
                                    '0-5m (Actual)': event_values_0_5[name],
                                    '5-30m (Estimated)': est_values_5_30[name],
                                }
                                if has_actual:
                                    row['5-30m (Actual)'] = actual_values_5_30[name]
                                    row['Error'] = actual_values_5_30[name] - est_values_5_30[name]
                                comparison_data.append(row)
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            if has_actual:
                                st.dataframe(
                                    comparison_df.style.format({
                                        '0-5m (Actual)': '{:.4f}',
                                        '5-30m (Estimated)': '{:.4f}',
                                        '5-30m (Actual)': '{:.4f}',
                                        'Error': '{:+.4f}'
                                    }).background_gradient(
                                        subset=['Error'], cmap='RdYlGn_r', vmin=-0.5, vmax=0.5
                                    ),
                                    use_container_width=True
                                )
                            else:
                                st.dataframe(
                                    comparison_df.style.format({
                                        '0-5m (Actual)': '{:.4f}',
                                        '5-30m (Estimated)': '{:.4f}',
                                    }),
                                    use_container_width=True
                                )
                                st.info("Insufficient 5-30m data to compare with actual values.")
                            
                            # Visual summary
                            st.markdown("**Estimated 5-30m Range**")
                            col_v1, col_v2, col_v3 = st.columns(3)
                            col_v1.metric("Est. P5 → P95 Range", 
                                         f"{est_values_5_30['P5']:.2f} to {est_values_5_30['P95']:.2f}")
                            col_v2.metric("Est. P10 → P90 Range", 
                                         f"{est_values_5_30['P10']:.2f} to {est_values_5_30['P90']:.2f}")
                            col_v3.metric("Est. Mean", f"{est_values_5_30['Mean']:.4f}")
                    
                    # Raw data
                    with st.expander("📋 View Raw Data"):
                        display_cols = ['seconds_since_event', 'cvd_zscore', 'close', 'volume', 
                                        'price_return_pct', 'fwd_10m_return_pct']
                        display_cols = [c for c in display_cols if c in event_df.columns]
                        st.dataframe(event_df[display_cols], use_container_width=True, height=400)

    # ========== TAB 5: Event Correlation Explorer ==========
    with tab5:
        st.info("📈 **Overview**: Explore event-level factors (surprise %, quarter, early features) that might predict CVD-return correlation strength. Compare 10min vs short-term correlations.")
        st.markdown("### Event-Level Correlations")
        
        # Only use 2024/2025 data for this analysis
        df_events = df.copy()
        df_events = df_events[df_events['year'].isin([2024, 2025])]
        
        if 'minutes_since_event' not in df_events.columns:
            df_events['minutes_since_event'] = df_events['seconds_since_event'] / 60.0
        
        # Correlation window selector
        corr_windows = {
            "5m-30m": (300, 1800),
            "5m-60m": (300, 3600),
            "30m-60m": (1800, 3600)
        }
        
        selected_window = st.selectbox(
            "CVD window for correlation calculation",
            options=list(corr_windows.keys()),
            index=0,
            key="corr_window"
        )
        
        # X-axis feature selector
        x_axis_options = [
            "First 5m Range %",
            "CVD Z-Score Max (2-5m)",
            "CVD Z-Score Min (2-5m)",
            "Quarter (event time)",
            "Earnings Surprise (%)",
            "Delta Efficiency (0-5m)",
            "CVD Monotonicity Sign (0-5m)",
            "CVD Monotonicity Net/Gross (0-5m)",
            "Price/CVD Ratio (0-5m)",
            "VWAP Deviation at 5m (%)",
            "Vol Front-Loading (0-5m)"
        ]
        
        selected_x = st.selectbox(
            "X-axis feature",
            options=x_axis_options,
            index=3,  # Default to "Quarter (event time)"
            key="corr_x_axis"
        )
        
        # Compute per-event stats
        stats = []
        win_start, win_end = corr_windows[selected_window]
        min_samples = 10
        
        for (ticker, evt), grp in df_events.groupby(['ticker', 'acceptance_datetime_utc']):
            grp = grp.sort_values('seconds_since_event')
            if 'target_ret_600s' not in grp.columns:
                continue
            
            # Correlation window data
            corr_slice = grp[(grp['seconds_since_event'] >= win_start) & (grp['seconds_since_event'] <= win_end)]
            corr_val = np.nan
            if len(corr_slice) >= min_samples:
                corr_df = corr_slice[['cvd_zscore', 'target_ret_600s']].dropna()
                if len(corr_df) >= min_samples:
                    corr_val = corr_df.corr().iloc[0, 1]
            
            # First 5m range pct
            range_val = grp['first_5m_range_pct'].iloc[0] if 'first_5m_range_pct' in grp.columns else np.nan
            
            # CVD zscore max/min 2-5m
            win_2_5 = grp[(grp['seconds_since_event'] >= 120) & (grp['seconds_since_event'] <= 300)]
            if not win_2_5.empty:
                cvd_max = win_2_5['cvd_zscore'].max(skipna=True)
                cvd_min = win_2_5['cvd_zscore'].min(skipna=True)
            else:
                cvd_max = np.nan
                cvd_min = np.nan
            
            # Quarter
            event_ts = pd.to_datetime(evt)
            quarter = f"Q{(event_ts.month - 1) // 3 + 1}"
            
            # Surprise
            surprise = grp['Surprise(%)'].iloc[0] if 'Surprise(%)' in grp.columns else np.nan
            
            # Early Event Meta Features (0-5m)
            slice_5m = grp[grp['seconds_since_event'] <= 300]
            event_price = grp['event_price'].iloc[0] if 'event_price' in grp.columns else np.nan
            
            delta_eff = calculate_delta_efficiency(slice_5m, event_price) if pd.notna(event_price) else np.nan
            cvd_mono_sign = calculate_cvd_monotonicity_sign(slice_5m)
            cvd_mono_ng = calculate_cvd_monotonicity_net_gross(slice_5m)
            price_cvd = calculate_price_cvd_ratio(range_val, cvd_max, cvd_min)
            vwap_dev = calculate_vwap_deviation_5m(grp)
            vol_fl = calculate_vol_front_loading(slice_5m)
            
            # Short return correlations (for comparison chart)
            corr_2m = np.nan
            corr_4m = np.nan
            if 'target_ret_120s' in grp.columns and len(corr_slice) >= min_samples:
                corr_df_2m = corr_slice[['cvd_zscore', 'target_ret_120s']].dropna()
                if len(corr_df_2m) >= min_samples:
                    corr_2m = corr_df_2m.corr().iloc[0, 1]
            if 'target_ret_240s' in grp.columns and len(corr_slice) >= min_samples:
                corr_df_4m = corr_slice[['cvd_zscore', 'target_ret_240s']].dropna()
                if len(corr_df_4m) >= min_samples:
                    corr_4m = corr_df_4m.corr().iloc[0, 1]
            
            stats.append({
                'ticker': ticker,
                'acceptance_datetime_utc': evt,
                'corr_cvd_vs_fwd10m': corr_val,
                'corr_cvd_vs_2m': corr_2m,
                'corr_cvd_vs_4m': corr_4m,
                'first_5m_range_pct': range_val,
                'cvd_zscore_max_2_5m': cvd_max,
                'cvd_zscore_min_2_5m': cvd_min,
                'quarter': quarter,
                'Surprise(%)': surprise,
                'delta_efficiency_5m': delta_eff,
                'cvd_monotonicity_sign_5m': cvd_mono_sign,
                'cvd_monotonicity_net_gross_5m': cvd_mono_ng,
                'price_cvd_ratio_5m': price_cvd,
                'vwap_deviation_5m': vwap_dev,
                'vol_front_loading_5m': vol_fl
            })
        
        stats_df = pd.DataFrame(stats)
        valid_base = stats_df.dropna(subset=['corr_cvd_vs_fwd10m'])
        
        if valid_base.empty:
            st.warning("No sufficient data for correlation analysis.")
        else:
            # Map X-axis option to column
            x_col_map = {
                "First 5m Range %": 'first_5m_range_pct',
                "CVD Z-Score Max (2-5m)": 'cvd_zscore_max_2_5m',
                "CVD Z-Score Min (2-5m)": 'cvd_zscore_min_2_5m',
                "Quarter (event time)": 'quarter',
                "Earnings Surprise (%)": 'Surprise(%)',
                "Delta Efficiency (0-5m)": 'delta_efficiency_5m',
                "CVD Monotonicity Sign (0-5m)": 'cvd_monotonicity_sign_5m',
                "CVD Monotonicity Net/Gross (0-5m)": 'cvd_monotonicity_net_gross_5m',
                "Price/CVD Ratio (0-5m)": 'price_cvd_ratio_5m',
                "VWAP Deviation at 5m (%)": 'vwap_deviation_5m',
                "Vol Front-Loading (0-5m)": 'vol_front_loading_5m'
            }
            
            feat_col = x_col_map[selected_x]
            
            # Special handling for Quarter (boxplot)
            if selected_x == "Quarter (event time)":
                st.markdown("#### Correlation by Quarter")
                fig_q = px.box(
                    valid_base,
                    x='quarter',
                    y='corr_cvd_vs_fwd10m',
                    color='quarter',
                    title=f"Corr(CVD, 10min return) by Quarter [{selected_window}]",
                    labels={'quarter': 'Quarter', 'corr_cvd_vs_fwd10m': 'Correlation'}
                )
                fig_q.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig_q.update_layout(height=450, template='plotly_white', showlegend=False)
                st.plotly_chart(fig_q, use_container_width=True, key="quarter_boxplot")
            
            # Special handling for Surprise (log scale)
            elif selected_x == "Earnings Surprise (%)":
                st.markdown("#### Correlation vs Earnings Surprise")
                plot_df = valid_base.dropna(subset=['Surprise(%)'])
                if plot_df.empty:
                    st.warning("No earnings surprise data available.")
                else:
                    # Signed log transformation
                    plot_df = plot_df.copy()
                    plot_df['surprise_log'] = np.sign(plot_df['Surprise(%)']) * np.log1p(np.abs(plot_df['Surprise(%)']))
                    
                    fig_s = px.scatter(
                        plot_df,
                        x='surprise_log',
                        y='corr_cvd_vs_fwd10m',
                        color='ticker',
                        hover_data=['ticker', 'acceptance_datetime_utc', 'Surprise(%)'],
                        color_discrete_sequence=TICKER_COLORS,
                        title=f"Corr(CVD, 10min return) vs Earnings Surprise [{selected_window}]",
                        labels={'surprise_log': 'Signed log(1 + |Surprise %|)', 'corr_cvd_vs_fwd10m': 'Correlation'}
                    )
                    fig_s.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_s.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_s.update_layout(
                        height=450,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_s, use_container_width=True, key="surprise_scatter")
            
            # Log scale for delta efficiency, price/cvd ratio, vol front loading
            elif selected_x in ["Delta Efficiency (0-5m)", "Price/CVD Ratio (0-5m)", "Vol Front-Loading (0-5m)"]:
                plot_df = valid_base.dropna(subset=[feat_col])
                if plot_df.empty:
                    st.warning(f"No data for {selected_x}")
                else:
                    plot_df = plot_df.copy()
                    plot_df['x_log'] = np.sign(plot_df[feat_col]) * np.log1p(np.abs(plot_df[feat_col]))
                    
                    fig_feat = px.scatter(
                        plot_df,
                        x='x_log',
                        y='corr_cvd_vs_fwd10m',
                        color='ticker',
                        hover_data=['ticker', 'acceptance_datetime_utc', feat_col],
                        color_discrete_sequence=TICKER_COLORS,
                        title=f"Corr(CVD, 10min return) vs {selected_x} [{selected_window}]",
                        labels={'x_log': f'Signed log(1 + |{selected_x}|)', 'corr_cvd_vs_fwd10m': 'Correlation'}
                    )
                    fig_feat.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_feat.update_layout(
                        height=450,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_feat, use_container_width=True, key=f"event_corr_scatter_{feat_col}")
            
            # Regular scatter for other features
            else:
                plot_df = valid_base.dropna(subset=[feat_col])
                if plot_df.empty:
                    st.warning(f"No data for {selected_x}")
                else:
                    fig_feat = px.scatter(
                        plot_df,
                        x=feat_col,
                        y='corr_cvd_vs_fwd10m',
                        color='ticker',
                        hover_data=['ticker', 'acceptance_datetime_utc'],
                        color_discrete_sequence=TICKER_COLORS,
                        title=f"Corr(CVD, 10min return) vs {selected_x} [{selected_window}]",
                        labels={feat_col: selected_x, 'corr_cvd_vs_fwd10m': 'Correlation'}
                    )
                    fig_feat.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_feat.update_layout(
                        height=450,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_feat, use_container_width=True, key=f"event_corr_scatter_{feat_col}")
            
            # --- Corr(CVD, 10m) vs Corr(CVD, 2m/4m) scatter ---
            st.markdown("---")
            st.markdown("#### Correlation Comparison: 10min vs Short-term Returns")
            
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                short_ret = st.selectbox(
                    "Short-term return",
                    options=["2min return", "4min return"],
                    index=0,
                    key="short_ret_select"
                )
            with cmp_col2:
                cmp_window = st.selectbox(
                    "Window",
                    options=list(corr_windows.keys()),
                    index=0,
                    key="cmp_window"
                )
            
            # Recalculate for comparison window if different
            cmp_stats = []
            cmp_win_start, cmp_win_end = corr_windows[cmp_window]
            
            for (ticker, evt), grp in df_events.groupby(['ticker', 'acceptance_datetime_utc']):
                grp = grp.sort_values('seconds_since_event')
                corr_slice = grp[(grp['seconds_since_event'] >= cmp_win_start) & (grp['seconds_since_event'] <= cmp_win_end)]
                
                if len(corr_slice) < min_samples:
                    continue
                
                # 10min correlation
                corr_10m = np.nan
                if 'target_ret_600s' in grp.columns:
                    cdf = corr_slice[['cvd_zscore', 'target_ret_600s']].dropna()
                    if len(cdf) >= min_samples:
                        corr_10m = cdf.corr().iloc[0, 1]
                
                # Short-term correlation
                short_col = 'target_ret_120s' if '2min' in short_ret else 'target_ret_240s'
                corr_short = np.nan
                if short_col in grp.columns:
                    cdf_s = corr_slice[['cvd_zscore', short_col]].dropna()
                    if len(cdf_s) >= min_samples:
                        corr_short = cdf_s.corr().iloc[0, 1]
                
                cmp_stats.append({
                    'ticker': ticker,
                    'acceptance_datetime_utc': evt,
                    'corr_cvd_10m': corr_10m,
                    'corr_cvd_short': corr_short
                })
            
            cmp_df = pd.DataFrame(cmp_stats)
            cmp_df = cmp_df.dropna(subset=['corr_cvd_10m', 'corr_cvd_short'])
            
            if cmp_df.empty:
                st.warning("No sufficient data for correlation comparison.")
            else:
                short_label = "2min" if '2min' in short_ret else "4min"
                fig_cmp = px.scatter(
                    cmp_df,
                    x='corr_cvd_short',
                    y='corr_cvd_10m',
                    color='ticker',
                    hover_data=['ticker', 'acceptance_datetime_utc'],
                    color_discrete_sequence=TICKER_COLORS,
                    labels={
                        'corr_cvd_short': f"Corr(CVD, {short_label} return) [{cmp_window}]",
                        'corr_cvd_10m': f"Corr(CVD, 10min return) [{cmp_window}]"
                    },
                    title=f"Corr(CVD, 10min) vs Corr(CVD, {short_label}) in {cmp_window} window"
                )
                
                # Add diagonal reference line
                min_val = min(cmp_df['corr_cvd_short'].min(), cmp_df['corr_cvd_10m'].min())
                max_val = max(cmp_df['corr_cvd_short'].max(), cmp_df['corr_cvd_10m'].max())
                fig_cmp.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    name='y=x',
                    showlegend=False
                ))
                
                fig_cmp.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
                fig_cmp.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)
                
                fig_cmp.update_layout(
                    height=600,
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                )
                
                st.plotly_chart(fig_cmp, use_container_width=True, key="event_corr_cmp_scatter")
                st.caption(f"Events: {len(cmp_df)} | Diagonal line = equal correlation strength")
                
                with st.expander("📋 View Comparison Data"):
                    st.dataframe(
                        cmp_df[['ticker', 'acceptance_datetime_utc', 'corr_cvd_short', 'corr_cvd_10m']].sort_values('corr_cvd_10m', ascending=False),
                        use_container_width=True,
                        height=400
                    )

    # ========== TAB 4: Strategy Group Analysis ==========
    with tab4:
        st.info("🎯 **Overview**: Group events into strategy regimes (Strong Mean Reversion, Moderate Mean Reversion, No Signal, Momentum) based on CVD-return correlation. Analyze feature distributions across groups.")
        st.markdown("### Correlation Group Analysis")
        
        # Define correlation windows
        corr_windows_t4 = {
            "5m-30m": (300, 1800),
            "5m-60m": (300, 3600),
            "30m-60m": (1800, 3600)
        }
        
        # Window selector
        selected_window_t4 = st.selectbox(
            "CVD window for correlation calculation",
            options=list(corr_windows_t4.keys()),
            index=0,
            key="strategy_corr_window"
        )
        
        # Adjustable thresholds
        st.markdown("**Adjust Group Thresholds:**")
        thresh_cols = st.columns(3)
        with thresh_cols[0]:
            thresh_strong_mr = st.slider(
                "Strong MR threshold",
                min_value=-0.9, max_value=0.0, value=-0.5, step=0.05,
                help="Below this = Strong Mean Reversion",
                key="thresh_strong_mr"
            )
        with thresh_cols[1]:
            thresh_moderate_mr = st.slider(
                "Moderate MR threshold",
                min_value=-0.5, max_value=0.5, value=-0.25, step=0.05,
                help="Between Strong MR and this = Moderate Mean Reversion",
                key="thresh_moderate_mr"
            )
        with thresh_cols[2]:
            thresh_momentum = st.slider(
                "Momentum threshold",
                min_value=-0.25, max_value=0.9, value=0.25, step=0.05,
                help="Above this = Momentum",
                key="thresh_momentum"
            )
        
        # Compute per-event stats
        df_events_t4 = df.copy()
        df_events_t4 = df_events_t4[df_events_t4['year'].isin([2024, 2025])]
        
        if 'minutes_since_event' not in df_events_t4.columns:
            df_events_t4['minutes_since_event'] = df_events_t4['seconds_since_event'] / 60.0
        
        stats_t4 = []
        win_start_t4, win_end_t4 = corr_windows_t4[selected_window_t4]
        min_samples_t4 = 10
        
        for (ticker, evt), grp in df_events_t4.groupby(['ticker', 'acceptance_datetime_utc']):
            grp = grp.sort_values('seconds_since_event')
            if 'target_ret_600s' not in grp.columns:
                continue
            
            # Correlation window data
            corr_slice = grp[(grp['seconds_since_event'] >= win_start_t4) & (grp['seconds_since_event'] <= win_end_t4)]
            corr_val = np.nan
            if len(corr_slice) >= min_samples_t4:
                corr_df = corr_slice[['cvd_zscore', 'target_ret_600s']].dropna()
                if len(corr_df) >= min_samples_t4:
                    corr_val = corr_df.corr().iloc[0, 1]
            
            # First 5m range pct
            range_val = grp['first_5m_range_pct'].iloc[0] if 'first_5m_range_pct' in grp.columns else np.nan
            
            # CVD zscore max/min 2-5m
            win_2_5 = grp[(grp['seconds_since_event'] >= 120) & (grp['seconds_since_event'] <= 300)]
            if not win_2_5.empty:
                cvd_max = win_2_5['cvd_zscore'].max(skipna=True)
                cvd_min = win_2_5['cvd_zscore'].min(skipna=True)
            else:
                cvd_max = np.nan
                cvd_min = np.nan
            
            # Early Event Meta Features (0-5m)
            slice_5m = grp[grp['seconds_since_event'] <= 300]
            event_price = grp['event_price'].iloc[0] if 'event_price' in grp.columns else np.nan
            
            delta_eff = calculate_delta_efficiency(slice_5m, event_price) if pd.notna(event_price) else np.nan
            cvd_mono_sign = calculate_cvd_monotonicity_sign(slice_5m)
            cvd_mono_ng = calculate_cvd_monotonicity_net_gross(slice_5m)
            price_cvd = calculate_price_cvd_ratio(range_val, cvd_max, cvd_min)
            vwap_dev = calculate_vwap_deviation_5m(grp)
            vol_fl = calculate_vol_front_loading(slice_5m)
            
            stats_t4.append({
                'ticker': ticker,
                'acceptance_datetime_utc': evt,
                'corr_cvd_vs_fwd10m': corr_val,
                'first_5m_range_pct': range_val,
                'cvd_zscore_max_2_5m': cvd_max,
                'cvd_zscore_min_2_5m': cvd_min,
                'delta_efficiency_5m': delta_eff,
                'cvd_monotonicity_sign_5m': cvd_mono_sign,
                'cvd_monotonicity_net_gross_5m': cvd_mono_ng,
                'price_cvd_ratio_5m': price_cvd,
                'vwap_deviation_5m': vwap_dev,
                'vol_front_loading_5m': vol_fl,
                'Surprise(%)': grp['Surprise(%)'].iloc[0] if 'Surprise(%)' in grp.columns else np.nan
            })
        
        stats_df_t4 = pd.DataFrame(stats_t4)
        valid_base_t4 = stats_df_t4.dropna(subset=['corr_cvd_vs_fwd10m'])
        
        if valid_base_t4.empty:
            st.warning("No sufficient data for strategy group analysis.")
        else:
            # Classify events into groups
            def classify_strategy(corr, t1, t2, t3):
                if corr < t1:
                    return "1. Strong MR"
                elif corr < t2:
                    return "2. Moderate MR"
                elif corr < t3:
                    return "3. No Signal"
                else:
                    return "4. Momentum"
            
            group_df = valid_base_t4.copy()
            group_df['strategy_group'] = group_df['corr_cvd_vs_fwd10m'].apply(
                lambda x: classify_strategy(x, thresh_strong_mr, thresh_moderate_mr, thresh_momentum)
            )
            
            # Summary counts
            group_counts = group_df['strategy_group'].value_counts().sort_index()
            st.markdown("**Event Counts by Group:**")
            count_cols = st.columns(4)
            group_names = ["1. Strong MR", "2. Moderate MR", "3. No Signal", "4. Momentum"]
            group_colors = ["#ef4444", "#f97316", "#6b7280", "#22c55e"]
            for i, (grp_name, color) in enumerate(zip(group_names, group_colors)):
                count = group_counts.get(grp_name, 0)
                pct = count / len(group_df) * 100 if len(group_df) > 0 else 0
                count_cols[i].markdown(f"**<span style='color:{color}'>{grp_name}</span>**", unsafe_allow_html=True)
                count_cols[i].metric("", f"{count} ({pct:.0f}%)")
            
            st.markdown("---")
            
            # Feature comparison by group - Strip plots
            st.markdown("**Feature Distributions by Strategy Group:**")
            
            # All early meta features to compare
            compare_features = [
                ('first_5m_range_pct', 'First 5m Range %'),
                ('cvd_zscore_max_2_5m', 'CVD Z Max (2-5m)'),
                ('cvd_zscore_min_2_5m', 'CVD Z Min (2-5m)'),
                ('cvd_monotonicity_sign_5m', 'CVD Monotonicity Sign'),
                ('cvd_monotonicity_net_gross_5m', 'CVD Monotonicity Net/Gross'),
                ('vwap_deviation_5m', 'VWAP Deviation 5m'),
                ('delta_efficiency_5m', 'Delta Efficiency'),
                ('price_cvd_ratio_5m', 'Price/CVD Ratio'),
                ('vol_front_loading_5m', 'Vol Front-Loading'),
                ('Surprise(%)', 'Earnings Surprise %')
            ]
            
            # Create strip/box plots for each feature
            for feat_col, feat_label in compare_features:
                if feat_col not in group_df.columns:
                    continue
                
                plot_df = group_df.dropna(subset=[feat_col])
                if plot_df.empty:
                    continue
                
                fig_strip = px.strip(
                    plot_df,
                    x='strategy_group',
                    y=feat_col,
                    color='strategy_group',
                    color_discrete_map={
                        "1. Strong MR": "#ef4444",
                        "2. Moderate MR": "#f97316",
                        "3. No Signal": "#6b7280",
                        "4. Momentum": "#22c55e"
                    },
                    hover_data=['ticker', 'acceptance_datetime_utc', 'corr_cvd_vs_fwd10m'],
                    title=f"{feat_label} by Strategy Group",
                    labels={feat_col: feat_label, 'strategy_group': 'Strategy Group'}
                )
                
                # Add box overlay for summary stats
                fig_strip.add_trace(
                    go.Box(
                        x=plot_df['strategy_group'],
                        y=plot_df[feat_col],
                        name='Distribution',
                        marker_color='rgba(0,0,0,0.3)',
                        line_color='rgba(0,0,0,0.5)',
                        boxpoints=False,
                        showlegend=False
                    )
                )
                
                fig_strip.update_layout(
                    height=400,
                    template='plotly_white',
                    showlegend=False,
                    xaxis_categoryorder='array',
                    xaxis_categoryarray=group_names
                )
                
                st.plotly_chart(fig_strip, use_container_width=True, key=f"strategy_strip_{feat_col}")

    # ========== TAB 2: Ticker Correlations ==========
    with tab2:
        st.info("📉 **Overview**: Compare CVD-return correlation strength across different tickers. Identify which stocks show stronger mean reversion or momentum signals.")
        st.markdown("### Ticker-Level CVD Z-Score Correlations")
        
        # Use filtered_df from sidebar
        if filtered_df.empty:
            st.warning("No data available with current filters.")
        elif 'target_ret_600s' not in filtered_df.columns:
            st.warning("Target return column not available.")
        else:
            # Calculate correlation per ticker
            ticker_corrs = []
            min_samples_t5 = 30
            
            for ticker in filtered_df['ticker'].unique():
                ticker_data = filtered_df[filtered_df['ticker'] == ticker][['cvd_zscore', 'target_ret_600s']].dropna()
                if len(ticker_data) >= min_samples_t5:
                    corr = ticker_data.corr().iloc[0, 1]
                    if not np.isnan(corr):
                        n_events = filtered_df[filtered_df['ticker'] == ticker].groupby('acceptance_datetime_utc').ngroups
                        ticker_corrs.append({
                            'Ticker': ticker,
                            'Correlation': corr,
                            'N_Rows': len(ticker_data),
                            'N_Events': n_events
                        })
            
            if not ticker_corrs:
                st.warning(f"No tickers have sufficient data (min {min_samples_t5} samples).")
            else:
                corr_df = pd.DataFrame(ticker_corrs).sort_values('Correlation')
                
                # Summary stats
                st.markdown("#### Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tickers", len(corr_df))
                col2.metric("Mean Corr", f"{corr_df['Correlation'].mean():.3f}")
                col3.metric("Median Corr", f"{corr_df['Correlation'].median():.3f}")
                col4.metric("Std", f"{corr_df['Correlation'].std():.3f}")
                
                # Bar chart
                st.markdown("#### Correlation by Ticker")
                fig_bar = px.bar(
                    corr_df,
                    x='Ticker',
                    y='Correlation',
                    color='Correlation',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    hover_data=['N_Rows', 'N_Events'],
                    title=f"CVD Z-Score vs 10min Return Correlation (Time: {time_min/60:.0f}-{time_max/60:.0f} min)"
                )
                fig_bar.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_bar.add_hline(y=corr_df['Correlation'].mean(), line_dash="dash", line_color="blue", 
                                 annotation_text=f"Mean: {corr_df['Correlation'].mean():.3f}")
                fig_bar.update_layout(
                    height=500,
                    template='plotly_white',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="ticker_corr_bar")
                
                # Distribution
                st.markdown("#### Correlation Distribution")
                fig_hist = px.histogram(
                    corr_df,
                    x='Correlation',
                    nbins=20,
                    title="Distribution of Ticker Correlations"
                )
                fig_hist.add_vline(x=0, line_dash="dot", line_color="gray")
                fig_hist.add_vline(x=corr_df['Correlation'].mean(), line_dash="dash", line_color="blue",
                                  annotation_text=f"Mean: {corr_df['Correlation'].mean():.3f}")
                fig_hist.update_layout(height=350, template='plotly_white')
                st.plotly_chart(fig_hist, use_container_width=True, key="ticker_corr_hist")
                
                # Data table
                st.markdown("#### Correlation Data")
                display_df = corr_df.sort_values('Correlation', ascending=True)
                
                # Color code
                def color_corr(val):
                    if val < -0.3:
                        return 'background-color: #fca5a5'  # red
                    elif val < -0.1:
                        return 'background-color: #fcd34d'  # yellow
                    elif val > 0.1:
                        return 'background-color: #86efac'  # green
                    return ''
                
                styled_df = display_df.style.applymap(color_corr, subset=['Correlation']).format({
                    'Correlation': '{:.4f}'
                })
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Top/Bottom tickers
                st.markdown("#### Strongest Correlations")
                col_top, col_bot = st.columns(2)
                
                with col_top:
                    st.markdown("**Most Negative (Mean Reversion)**")
                    top_neg = corr_df.nsmallest(5, 'Correlation')[['Ticker', 'Correlation', 'N_Events']]
                    st.dataframe(top_neg.style.format({'Correlation': '{:.4f}'}), use_container_width=True)
                
                with col_bot:
                    st.markdown("**Most Positive (Momentum)**")
                    top_pos = corr_df.nlargest(5, 'Correlation')[['Ticker', 'Correlation', 'N_Events']]
                    st.dataframe(top_pos.style.format({'Correlation': '{:.4f}'}), use_container_width=True)

    # ========== TAB 6: VWAP Deviation Analysis ==========
    with tab6:
        st.info("📐 **Overview**: Track VWAP deviation (|price - vwap| / price) from 1m to 15m. See how VWAP deviation patterns differ across strategy groups.")
        st.markdown("### VWAP Deviation Analysis")
        
        # Use 2024/2025 data
        df_vwap = df.copy()
        df_vwap = df_vwap[df_vwap['year'].isin([2024, 2025])]
        
        if 'vwap_since_event' not in df_vwap.columns:
            st.error("VWAP data not available in dataset.")
        else:
            # CVD correlation window selector
            vwap_corr_windows = {
                "5m-30m": (300, 1800),
                "5m-60m": (300, 3600),
                "30m-60m": (1800, 3600)
            }
            
            selected_vwap_window = st.selectbox(
                "CVD correlation window",
                options=list(vwap_corr_windows.keys()),
                index=0,
                key="vwap_corr_window"
            )
            
            # Calculate VWAP deviation at each minute (1-15m) and CVD correlation for each event
            vwap_time_points = list(range(1, 16))  # 1m to 15m
            min_samples_vwap = 10
            
            event_data = []
            win_start_v, win_end_v = vwap_corr_windows[selected_vwap_window]
            
            for (ticker, evt), grp in df_vwap.groupby(['ticker', 'acceptance_datetime_utc']):
                grp = grp.sort_values('seconds_since_event')
                
                # Calculate CVD vs 10min return correlation
                corr_slice = grp[(grp['seconds_since_event'] >= win_start_v) & (grp['seconds_since_event'] <= win_end_v)]
                cvd_corr = np.nan
                if len(corr_slice) >= min_samples_vwap and 'target_ret_600s' in grp.columns:
                    corr_df_v = corr_slice[['cvd_zscore', 'target_ret_600s']].dropna()
                    if len(corr_df_v) >= min_samples_vwap:
                        cvd_corr = corr_df_v.corr().iloc[0, 1]
                
                if np.isnan(cvd_corr):
                    continue
                
                # Calculate VWAP deviation at each time point
                vwap_devs = {}
                for t_min in vwap_time_points:
                    t_sec = t_min * 60
                    slice_t = grp[grp['seconds_since_event'] <= t_sec]
                    if slice_t.empty:
                        vwap_devs[f'vwap_dev_{t_min}m'] = np.nan
                        continue
                    
                    last_row = slice_t.iloc[-1]
                    price = last_row.get('close', np.nan)
                    vwap = last_row.get('vwap_since_event', np.nan)
                    
                    if pd.notna(price) and pd.notna(vwap) and price > 0:
                        vwap_devs[f'vwap_dev_{t_min}m'] = abs(price - vwap) / price * 100
                    else:
                        vwap_devs[f'vwap_dev_{t_min}m'] = np.nan
                
                event_data.append({
                    'ticker': ticker,
                    'event': evt,
                    'cvd_corr': cvd_corr,
                    **vwap_devs
                })
            
            if not event_data:
                st.warning("No sufficient data for VWAP deviation analysis.")
            else:
                vwap_df = pd.DataFrame(event_data)
                
                # Classify events by correlation strength
                def classify_corr(c):
                    if c < -0.4:
                        return "Strong MR"
                    elif c < -0.15:
                        return "Moderate MR"
                    elif c < 0.15:
                        return "No Signal"
                    else:
                        return "Momentum"
                
                vwap_df['corr_group'] = vwap_df['cvd_corr'].apply(classify_corr)
                
                st.markdown(f"**Events analyzed:** {len(vwap_df)}")
                
                # ---- Chart 1: Heatmap - Correlation between VWAP dev @ each minute and CVD correlation ----
                st.markdown("---")
                st.markdown("#### Correlation: VWAP Deviation vs CVD-Return Correlation")
                st.markdown("Shows how VWAP deviation at each time point correlates with the CVD-10min return correlation")
                
                heatmap_corrs = []
                for t_min in vwap_time_points:
                    col_name = f'vwap_dev_{t_min}m'
                    valid = vwap_df[[col_name, 'cvd_corr']].dropna()
                    if len(valid) >= 20:
                        corr = valid.corr().iloc[0, 1]
                    else:
                        corr = np.nan
                    heatmap_corrs.append({'Time': f'{t_min}m', 'Correlation': corr})
                
                heatmap_df = pd.DataFrame(heatmap_corrs)
                
                fig_bar = px.bar(
                    heatmap_df,
                    x='Time',
                    y='Correlation',
                    color='Correlation',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title=f"Corr(VWAP_dev@Xm, CVD_corr[{selected_vwap_window}])",
                    labels={'Correlation': 'Correlation'}
                )
                fig_bar.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_bar.update_layout(height=350, template='plotly_white')
                st.plotly_chart(fig_bar, use_container_width=True, key="vwap_corr_bar")
                
                # ---- Chart 2: Trajectory by correlation group ----
                st.markdown("---")
                st.markdown("#### VWAP Deviation Trajectories by Strategy Group")
                st.markdown("Mean VWAP deviation over time, grouped by CVD correlation strength")
                
                # Reshape for plotting
                trajectory_data = []
                for _, row in vwap_df.iterrows():
                    for t_min in vwap_time_points:
                        val = row.get(f'vwap_dev_{t_min}m', np.nan)
                        if not np.isnan(val):
                            trajectory_data.append({
                                'Time (min)': t_min,
                                'VWAP Deviation %': val,
                                'Group': row['corr_group'],
                                'ticker': row['ticker']
                            })
                
                traj_df = pd.DataFrame(trajectory_data)
                
                if not traj_df.empty:
                    # Calculate mean and std per group per time
                    traj_summary = traj_df.groupby(['Time (min)', 'Group'])['VWAP Deviation %'].agg(['mean', 'std', 'count']).reset_index()
                    
                    group_order = ["Strong MR", "Moderate MR", "No Signal", "Momentum"]
                    group_colors = {"Strong MR": "#ef4444", "Moderate MR": "#f97316", "No Signal": "#6b7280", "Momentum": "#22c55e"}
                    
                    fig_traj = px.line(
                        traj_summary,
                        x='Time (min)',
                        y='mean',
                        color='Group',
                        color_discrete_map=group_colors,
                        category_orders={'Group': group_order},
                        markers=True,
                        title="Mean VWAP Deviation Over Time by Strategy Group",
                        labels={'mean': 'Mean VWAP Deviation %'}
                    )
                    fig_traj.update_layout(
                        height=450,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_traj, use_container_width=True, key="vwap_trajectory")
                    
                    # Group counts
                    group_counts = vwap_df['corr_group'].value_counts()
                    st.caption(f"Group sizes: {', '.join([f'{g}: {group_counts.get(g, 0)}' for g in group_order])}")
                
                # ---- Chart 3: Scatter plots at key time points ----
                st.markdown("---")
                st.markdown("#### Scatter: VWAP Deviation vs CVD Correlation")
                
                scatter_times = st.multiselect(
                    "Select time points to visualize",
                    options=[f'{t}m' for t in vwap_time_points],
                    default=['3m', '5m', '10m'],
                    key="vwap_scatter_times"
                )
                
                if scatter_times:
                    n_plots = len(scatter_times)
                    cols = st.columns(min(n_plots, 3))
                    
                    for i, t_str in enumerate(scatter_times):
                        col_name = f'vwap_dev_{t_str}'
                        plot_data = vwap_df[[col_name, 'cvd_corr', 'ticker', 'corr_group']].dropna()
                        
                        if not plot_data.empty:
                            with cols[i % 3]:
                                fig_scatter = px.scatter(
                                    plot_data,
                                    x=col_name,
                                    y='cvd_corr',
                                    color='corr_group',
                                    color_discrete_map=group_colors,
                                    category_orders={'corr_group': group_order},
                                    hover_data=['ticker'],
                                    title=f"VWAP Dev @ {t_str}",
                                    labels={col_name: f'VWAP Dev % @ {t_str}', 'cvd_corr': 'CVD Correlation'}
                                )
                                fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                                fig_scatter.update_layout(
                                    height=350,
                                    template='plotly_white',
                                    showlegend=False
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True, key=f"vwap_scatter_{t_str}")
                                
                                # Show correlation
                                corr_val = plot_data[[col_name, 'cvd_corr']].corr().iloc[0, 1]
                                st.caption(f"Corr: {corr_val:.3f} (n={len(plot_data)})")
                
                # ---- Summary stats table ----
                st.markdown("---")
                st.markdown("#### Summary Statistics by Group")
                
                summary_rows = []
                for group in group_order:
                    group_data = vwap_df[vwap_df['corr_group'] == group]
                    if len(group_data) == 0:
                        continue
                    row = {'Group': group, 'N Events': len(group_data)}
                    for t_min in [3, 5, 10, 15]:
                        col = f'vwap_dev_{t_min}m'
                        if col in group_data.columns:
                            row[f'VWAP Dev @{t_min}m'] = group_data[col].mean()
                    row['Mean CVD Corr'] = group_data['cvd_corr'].mean()
                    summary_rows.append(row)
                
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(
                        summary_df.style.format({
                            'VWAP Dev @3m': '{:.3f}%',
                            'VWAP Dev @5m': '{:.3f}%',
                            'VWAP Dev @10m': '{:.3f}%',
                            'VWAP Dev @15m': '{:.3f}%',
                            'Mean CVD Corr': '{:.3f}'
                        }),
                        use_container_width=True
                    )

    # ========== TAB 7: Pattern Analysis ==========
    with tab7:
        st.info("🔬 **Overview**: Analyze first 5 minutes patterns: CVD velocity/acceleration, price-CVD divergence, return momentum, and VW price autocorrelation. Find early signals that predict later CVD-return behavior.")
        st.markdown("### First 5m Signals Analysis")
        
        # Use 2024/2025 data
        df_pattern = df.copy()
        df_pattern = df_pattern[df_pattern['year'].isin([2024, 2025])]
        
        # CVD correlation window selector
        pattern_corr_windows = {
            "5m-30m": (300, 1800),
            "5m-60m": (300, 3600),
            "30m-60m": (1800, 3600)
        }
        
        selected_pattern_window = st.selectbox(
            "CVD correlation window for comparison",
            options=list(pattern_corr_windows.keys()),
            index=0,
            key="pattern_corr_window"
        )
        
        min_samples_pattern = 10
        win_start_p, win_end_p = pattern_corr_windows[selected_pattern_window]
        
        # Calculate pattern features for each event
        pattern_data = []
        
        for (ticker, evt), grp in df_pattern.groupby(['ticker', 'acceptance_datetime_utc']):
            grp = grp.sort_values('seconds_since_event')
            
            # Calculate CVD vs 10min return correlation
            corr_slice = grp[(grp['seconds_since_event'] >= win_start_p) & (grp['seconds_since_event'] <= win_end_p)]
            cvd_corr = np.nan
            if len(corr_slice) >= min_samples_pattern and 'target_ret_600s' in grp.columns:
                corr_df_p = corr_slice[['cvd_zscore', 'target_ret_600s']].dropna()
                if len(corr_df_p) >= min_samples_pattern:
                    cvd_corr = corr_df_p.corr().iloc[0, 1]
            
            if np.isnan(cvd_corr):
                continue
            
            # Get first 5m data
            slice_5m = grp[grp['seconds_since_event'] <= 300]
            if len(slice_5m) < 5:
                continue
            
            # ============ FEATURE 1: CVD Velocity & Acceleration ============
            # CVD at different time points
            cvd_vals = slice_5m['cvd_since_event'].dropna()
            if len(cvd_vals) < 3:
                cvd_velocity = np.nan
                cvd_acceleration = np.nan
                cvd_smoothness = np.nan
            else:
                # Velocity: overall slope (final - initial) / time
                cvd_start = cvd_vals.iloc[0] if len(cvd_vals) > 0 else 0
                cvd_end = cvd_vals.iloc[-1] if len(cvd_vals) > 0 else 0
                cvd_velocity = cvd_end - cvd_start  # Net CVD change in 5m
                
                # Acceleration: compare first half vs second half velocity
                mid_idx = len(cvd_vals) // 2
                first_half = cvd_vals.iloc[:mid_idx]
                second_half = cvd_vals.iloc[mid_idx:]
                
                if len(first_half) > 1 and len(second_half) > 1:
                    vel_1st = first_half.iloc[-1] - first_half.iloc[0]
                    vel_2nd = second_half.iloc[-1] - second_half.iloc[0]
                    cvd_acceleration = vel_2nd - vel_1st  # Positive = speeding up
                else:
                    cvd_acceleration = np.nan
                
                # Smoothness: ratio of |net change| to sum of |changes| (already have this as monotonicity)
                diffs = cvd_vals.diff().dropna()
                if len(diffs) > 0 and diffs.abs().sum() > 0:
                    cvd_smoothness = abs(diffs.sum()) / diffs.abs().sum()
                else:
                    cvd_smoothness = np.nan
            
            # CVD reversal count
            if len(cvd_vals) >= 3:
                diffs = cvd_vals.diff().dropna()
                signs = np.sign(diffs)
                sign_changes = (signs.diff().abs() > 0).sum()
                cvd_reversals = sign_changes
            else:
                cvd_reversals = np.nan
            
            # ============ FEATURE 2: Price-CVD Divergence ============
            # Price return in first 5m
            event_price = grp['event_price'].iloc[0] if 'event_price' in grp.columns else np.nan
            if pd.notna(event_price) and event_price > 0 and len(slice_5m) > 0:
                price_5m = slice_5m['close'].iloc[-1]
                price_return_5m = (price_5m - event_price) / event_price * 100
            else:
                price_return_5m = np.nan
            
            # CVD value at 5m
            cvd_5m = cvd_vals.iloc[-1] if len(cvd_vals) > 0 else np.nan
            
            # Divergence: sign disagreement
            if pd.notna(price_return_5m) and pd.notna(cvd_5m):
                price_cvd_sign_match = np.sign(price_return_5m) == np.sign(cvd_5m)
                # Divergence score: price moved one way, CVD moved the other
                divergence_score = price_return_5m * (-np.sign(cvd_5m)) if cvd_5m != 0 else 0
            else:
                price_cvd_sign_match = np.nan
                divergence_score = np.nan
            
            # ============ FEATURE 3: Return Autocorrelation ============
            # Compare early return direction with later return
            # Get return at 2m and return at 5m mark
            slice_2m = grp[grp['seconds_since_event'] <= 120]
            
            if len(slice_2m) > 0 and pd.notna(event_price) and event_price > 0:
                price_2m = slice_2m['close'].iloc[-1]
                return_0_2m = (price_2m - event_price) / event_price * 100
            else:
                return_0_2m = np.nan
            
            if pd.notna(price_return_5m) and pd.notna(return_0_2m):
                return_2m_5m = price_return_5m - return_0_2m  # Return from 2m to 5m
            else:
                return_2m_5m = np.nan
            
            # Early momentum persistence: does 0-2m return predict 2-5m return?
            if pd.notna(return_0_2m) and pd.notna(return_2m_5m) and return_0_2m != 0:
                momentum_persistence = return_2m_5m / abs(return_0_2m)  # Positive = continuation
            else:
                momentum_persistence = np.nan
            
            # Return sign match (0-2m vs 2-5m)
            if pd.notna(return_0_2m) and pd.notna(return_2m_5m):
                early_return_continues = np.sign(return_0_2m) == np.sign(return_2m_5m)
            else:
                early_return_continues = np.nan
            
            # ============ FEATURE 4: Price Autocorrelation using per-bar VW (first 5m) ============
            # Use 'vw' column - per-bar volume-weighted price (not cumulative)
            vw_series = slice_5m['vw'].dropna() if 'vw' in slice_5m.columns else pd.Series()
            
            # 1. VW autocorr at lag=10, 20, 30
            if len(vw_series) > 35:
                vwap_autocorr_10 = vw_series.autocorr(lag=10)
                vwap_autocorr_20 = vw_series.autocorr(lag=20)
                vwap_autocorr_30 = vw_series.autocorr(lag=30)
            else:
                vwap_autocorr_10 = np.nan
                vwap_autocorr_20 = np.nan
                vwap_autocorr_30 = np.nan
            
            # 2. VW autocorr half-life (lag where autocorr drops below 0.5)
            vwap_half_life = np.nan
            if len(vw_series) > 60:
                for lag in range(1, min(60, len(vw_series) - 5)):
                    ac = vw_series.autocorr(lag=lag)
                    if pd.notna(ac) and ac < 0.5:
                        vwap_half_life = lag
                        break
                if np.isnan(vwap_half_life):
                    vwap_half_life = 60  # Cap at 60 if never drops below 0.5
            
            pattern_data.append({
                'ticker': ticker,
                'event': evt,
                'cvd_corr': cvd_corr,
                # CVD Velocity/Acceleration
                'cvd_velocity_5m': cvd_velocity,
                'cvd_acceleration': cvd_acceleration,
                'cvd_smoothness': cvd_smoothness,
                'cvd_reversals': cvd_reversals,
                # Price-CVD Divergence
                'price_return_5m': price_return_5m,
                'cvd_5m': cvd_5m,
                'price_cvd_sign_match': price_cvd_sign_match,
                'divergence_score': divergence_score,
                # Return Autocorrelation
                'return_0_2m': return_0_2m,
                'return_2m_5m': return_2m_5m,
                'momentum_persistence': momentum_persistence,
                'early_return_continues': early_return_continues,
                # VWAP Autocorrelation (first 5m)
                'vwap_autocorr_10': vwap_autocorr_10,
                'vwap_autocorr_20': vwap_autocorr_20,
                'vwap_autocorr_30': vwap_autocorr_30,
                'vwap_half_life': vwap_half_life
            })
        
        if not pattern_data:
            st.warning("No sufficient data for pattern analysis.")
        else:
            pattern_df = pd.DataFrame(pattern_data)
            
            # Classify by correlation strength
            def classify_corr_p(c):
                if c < -0.4:
                    return "Strong MR"
                elif c < -0.15:
                    return "Moderate MR"
                elif c < 0.15:
                    return "No Signal"
                else:
                    return "Momentum"
            
            pattern_df['corr_group'] = pattern_df['cvd_corr'].apply(classify_corr_p)
            group_order_p = ["Strong MR", "Moderate MR", "No Signal", "Momentum"]
            group_colors_p = {"Strong MR": "#ef4444", "Moderate MR": "#f97316", "No Signal": "#6b7280", "Momentum": "#22c55e"}
            
            st.markdown(f"**Events analyzed:** {len(pattern_df)}")
            
            # ============ SECTION 1: CVD Velocity & Acceleration ============
            st.markdown("---")
            st.markdown("### 1️⃣ CVD Velocity & Acceleration (0-5m)")
            st.markdown("*Does the speed/direction of early CVD predict later behavior?*")
            
            col1a, col1b = st.columns(2)
            
            with col1a:
                # CVD Velocity vs Correlation
                plot_vel = pattern_df[['cvd_velocity_5m', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_vel.empty:
                    fig_vel = px.scatter(
                        plot_vel,
                        x='cvd_velocity_5m',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="CVD Velocity (net change in 5m) vs CVD Correlation",
                        labels={'cvd_velocity_5m': 'CVD Velocity (0-5m)', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_vel.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_vel.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_vel.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_vel, use_container_width=True, key="pattern_cvd_velocity")
                    corr_v = plot_vel[['cvd_velocity_5m', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_v:.3f}")
            
            with col1b:
                # CVD Smoothness vs Correlation
                plot_smooth = pattern_df[['cvd_smoothness', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_smooth.empty:
                    fig_smooth = px.scatter(
                        plot_smooth,
                        x='cvd_smoothness',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="CVD Smoothness (|net|/Σ|changes|) vs CVD Correlation",
                        labels={'cvd_smoothness': 'CVD Smoothness', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_smooth.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_smooth.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_smooth, use_container_width=True, key="pattern_cvd_smoothness")
                    corr_s = plot_smooth[['cvd_smoothness', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_s:.3f}")
            
            col1c, col1d = st.columns(2)
            
            with col1c:
                # CVD Acceleration vs Correlation
                plot_acc = pattern_df[['cvd_acceleration', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_acc.empty:
                    fig_acc = px.scatter(
                        plot_acc,
                        x='cvd_acceleration',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="CVD Acceleration (2nd half - 1st half velocity)",
                        labels={'cvd_acceleration': 'CVD Acceleration', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_acc.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_acc.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_acc.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_acc, use_container_width=True, key="pattern_cvd_acceleration")
                    corr_a = plot_acc[['cvd_acceleration', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_a:.3f}")
            
            with col1d:
                # CVD Reversals vs Correlation
                plot_rev = pattern_df[['cvd_reversals', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_rev.empty:
                    fig_rev = px.scatter(
                        plot_rev,
                        x='cvd_reversals',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="CVD Reversals (direction changes in 5m)",
                        labels={'cvd_reversals': 'CVD Reversal Count', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_rev.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_rev.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_rev, use_container_width=True, key="pattern_cvd_reversals")
                    corr_r = plot_rev[['cvd_reversals', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_r:.3f}")
            
            # ============ SECTION 2: Price-CVD Divergence ============
            st.markdown("---")
            st.markdown("### 2️⃣ Price-CVD Divergence (0-5m)")
            st.markdown("*When price and order flow disagree, what happens next?*")
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                # Divergence Score vs Correlation
                plot_div = pattern_df[['divergence_score', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_div.empty:
                    fig_div = px.scatter(
                        plot_div,
                        x='divergence_score',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="Price-CVD Divergence Score vs CVD Correlation",
                        labels={'divergence_score': 'Divergence Score', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_div.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_div.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_div.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_div, use_container_width=True, key="pattern_divergence")
                    corr_d = plot_div[['divergence_score', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_d:.3f}")
            
            with col2b:
                # Sign Match distribution by group
                plot_sign = pattern_df[['price_cvd_sign_match', 'corr_group']].dropna()
                if not plot_sign.empty:
                    sign_summary = plot_sign.groupby('corr_group')['price_cvd_sign_match'].mean().reset_index()
                    sign_summary.columns = ['Group', 'Sign Match Rate']
                    sign_summary['Sign Match Rate'] *= 100
                    
                    fig_sign = px.bar(
                        sign_summary,
                        x='Group',
                        y='Sign Match Rate',
                        color='Group',
                        color_discrete_map=group_colors_p,
                        category_orders={'Group': group_order_p},
                        title="Price-CVD Sign Agreement Rate by Group",
                        labels={'Sign Match Rate': 'Sign Match %'}
                    )
                    fig_sign.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="50%")
                    fig_sign.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_sign, use_container_width=True, key="pattern_sign_match")
            
            # ============ SECTION 3: Return Autocorrelation ============
            st.markdown("---")
            st.markdown("### 3️⃣ Return Autocorrelation (Early vs Later)")
            st.markdown("*Does early price momentum persist or reverse?*")
            
            col3a, col3b = st.columns(2)
            
            with col3a:
                # 0-2m return vs 2-5m return
                plot_ret = pattern_df[['return_0_2m', 'return_2m_5m', 'corr_group', 'ticker']].dropna()
                if not plot_ret.empty:
                    fig_ret = px.scatter(
                        plot_ret,
                        x='return_0_2m',
                        y='return_2m_5m',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="Return 0-2m vs Return 2-5m",
                        labels={'return_0_2m': 'Return 0-2m (%)', 'return_2m_5m': 'Return 2-5m (%)'}
                    )
                    fig_ret.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ret.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ret.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_ret, use_container_width=True, key="pattern_return_autocorr")
                    corr_ret = plot_ret[['return_0_2m', 'return_2m_5m']].corr().iloc[0, 1]
                    st.caption(f"Return autocorrelation: {corr_ret:.3f}")
            
            with col3b:
                # Momentum Persistence vs CVD Correlation
                plot_mom = pattern_df[['momentum_persistence', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                # Clip extreme values
                plot_mom = plot_mom[plot_mom['momentum_persistence'].abs() < 10]
                if not plot_mom.empty:
                    fig_mom = px.scatter(
                        plot_mom,
                        x='momentum_persistence',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="Momentum Persistence vs CVD Correlation",
                        labels={'momentum_persistence': 'Momentum Persistence', 'cvd_corr': 'CVD-Return Corr'}
                    )
                    fig_mom.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_mom.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_mom.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_mom, use_container_width=True, key="pattern_momentum")
                    corr_m = plot_mom[['momentum_persistence', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_m:.3f}")
            
            # Early return continuation rate by group
            plot_cont = pattern_df[['early_return_continues', 'corr_group']].dropna()
            if not plot_cont.empty:
                cont_summary = plot_cont.groupby('corr_group')['early_return_continues'].mean().reset_index()
                cont_summary.columns = ['Group', 'Continuation Rate']
                cont_summary['Continuation Rate'] *= 100
                
                fig_cont = px.bar(
                    cont_summary,
                    x='Group',
                    y='Continuation Rate',
                    color='Group',
                    color_discrete_map=group_colors_p,
                    category_orders={'Group': group_order_p},
                    title="Early Return Continuation Rate by Group (0-2m same direction as 2-5m)",
                    labels={'Continuation Rate': 'Continuation Rate %'}
                )
                fig_cont.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="50% (random)")
                fig_cont.update_layout(height=350, template='plotly_white', showlegend=False)
                st.plotly_chart(fig_cont, use_container_width=True, key="pattern_continuation")
            
            # ============ SECTION 4: VW Price Autocorrelation (first 5m) ============
            st.markdown("---")
            st.markdown("### 4️⃣ VW Price Autocorrelation (first 5m)")
            st.markdown("*How persistent/choppy is per-bar volume-weighted price in the first 5 minutes?*")
            
            col4a, col4b = st.columns(2)
            
            with col4a:
                # VWAP Autocorr at lag=10 vs CVD Correlation
                plot_ac10 = pattern_df[['vwap_autocorr_10', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_ac10.empty:
                    fig_ac10 = px.scatter(
                        plot_ac10,
                        x='vwap_autocorr_10',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="VWAP Autocorr (lag=10) vs CVD-Return Correlation",
                        labels={'vwap_autocorr_10': 'VWAP Autocorr (lag=10)', 'cvd_corr': 'CVD-Return Corr [5-30m]'}
                    )
                    fig_ac10.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ac10.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_ac10, use_container_width=True, key="pattern_vwap_autocorr_10")
                    corr_ac10 = plot_ac10[['vwap_autocorr_10', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_ac10:.3f}")
                else:
                    st.info("No VWAP autocorr data available")
            
            with col4b:
                # VWAP Autocorr at lag=20 vs CVD Correlation
                plot_ac20 = pattern_df[['vwap_autocorr_20', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_ac20.empty:
                    fig_ac20 = px.scatter(
                        plot_ac20,
                        x='vwap_autocorr_20',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="VWAP Autocorr (lag=20) vs CVD-Return Correlation",
                        labels={'vwap_autocorr_20': 'VWAP Autocorr (lag=20)', 'cvd_corr': 'CVD-Return Corr [5-30m]'}
                    )
                    fig_ac20.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ac20.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_ac20, use_container_width=True, key="pattern_vwap_autocorr_20")
                    corr_ac20 = plot_ac20[['vwap_autocorr_20', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_ac20:.3f}")
                else:
                    st.info("No VWAP autocorr data available")
            
            col4c, col4d = st.columns(2)
            
            with col4c:
                # VWAP Autocorr at lag=30 vs CVD Correlation
                plot_ac30 = pattern_df[['vwap_autocorr_30', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_ac30.empty:
                    fig_ac30 = px.scatter(
                        plot_ac30,
                        x='vwap_autocorr_30',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="VWAP Autocorr (lag=30) vs CVD-Return Correlation",
                        labels={'vwap_autocorr_30': 'VWAP Autocorr (lag=30)', 'cvd_corr': 'CVD-Return Corr [5-30m]'}
                    )
                    fig_ac30.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_ac30.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_ac30, use_container_width=True, key="pattern_vwap_autocorr_30")
                    corr_ac30 = plot_ac30[['vwap_autocorr_30', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_ac30:.3f}")
                else:
                    st.info("No VWAP autocorr data available")
            
            with col4d:
                # VWAP Half-life vs CVD Correlation
                plot_hl = pattern_df[['vwap_half_life', 'cvd_corr', 'ticker', 'corr_group']].dropna()
                if not plot_hl.empty:
                    fig_hl = px.scatter(
                        plot_hl,
                        x='vwap_half_life',
                        y='cvd_corr',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        hover_data=['ticker'],
                        title="VWAP Autocorr Half-Life vs CVD-Return Correlation",
                        labels={'vwap_half_life': 'VWAP Half-Life (bars)', 'cvd_corr': 'CVD-Return Corr [5-30m]'}
                    )
                    fig_hl.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_hl.update_layout(height=400, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_hl, use_container_width=True, key="pattern_vwap_half_life")
                    corr_hl = plot_hl[['vwap_half_life', 'cvd_corr']].corr().iloc[0, 1]
                    st.caption(f"Correlation: {corr_hl:.3f}")
                else:
                    st.info("No half-life data available")
            
            # Distribution of autocorr metrics by group
            st.markdown("#### VWAP Autocorr Metrics by Strategy Group")
            col4e, col4f = st.columns(2)
            
            with col4e:
                plot_ac_box = pattern_df[['vwap_autocorr_10', 'corr_group']].dropna()
                if not plot_ac_box.empty:
                    fig_ac_box = px.box(
                        plot_ac_box,
                        x='corr_group',
                        y='vwap_autocorr_10',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        title="VWAP Autocorr (lag=10) by Group"
                    )
                    fig_ac_box.update_layout(height=350, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_ac_box, use_container_width=True, key="pattern_vwap_ac_box")
            
            with col4f:
                plot_hl_box = pattern_df[['vwap_half_life', 'corr_group']].dropna()
                if not plot_hl_box.empty:
                    fig_hl_box = px.box(
                        plot_hl_box,
                        x='corr_group',
                        y='vwap_half_life',
                        color='corr_group',
                        color_discrete_map=group_colors_p,
                        category_orders={'corr_group': group_order_p},
                        title="VWAP Half-Life by Group"
                    )
                    fig_hl_box.update_layout(height=350, template='plotly_white', showlegend=False)
                    st.plotly_chart(fig_hl_box, use_container_width=True, key="pattern_vwap_hl_box")
            
            # ============ Summary Table ============
            st.markdown("---")
            st.markdown("### Summary: Feature Correlations with CVD-Return Correlation")
            
            features_to_summarize = [
                ('cvd_velocity_5m', 'CVD Velocity'),
                ('cvd_acceleration', 'CVD Acceleration'),
                ('cvd_smoothness', 'CVD Smoothness'),
                ('cvd_reversals', 'CVD Reversals'),
                ('divergence_score', 'Price-CVD Divergence'),
                ('vwap_autocorr_10', 'VWAP Autocorr (lag=10)'),
                ('vwap_autocorr_20', 'VWAP Autocorr (lag=20)'),
                ('vwap_autocorr_30', 'VWAP Autocorr (lag=30)'),
                ('vwap_half_life', 'VWAP Half-Life'),
                ('momentum_persistence', 'Momentum Persistence')
            ]
            
            summary_corrs = []
            for col, label in features_to_summarize:
                if col in pattern_df.columns:
                    valid = pattern_df[[col, 'cvd_corr']].dropna()
                    if col == 'momentum_persistence':
                        valid = valid[valid[col].abs() < 10]
                    if len(valid) >= 20:
                        corr = valid.corr().iloc[0, 1]
                        summary_corrs.append({'Feature': label, 'Correlation': corr, 'N': len(valid)})
            
            if summary_corrs:
                summary_corr_df = pd.DataFrame(summary_corrs).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(
                    summary_corr_df.style.format({'Correlation': '{:.4f}'}).background_gradient(
                        subset=['Correlation'], cmap='RdBu_r', vmin=-0.3, vmax=0.3
                    ),
                    use_container_width=True
                )

    # ========== TAB 8: 0-5m vs 5-30m Comparison ==========
    with tab8:
        st.info("🔄 **Overview**: Compare CVD Z-Score behavior between early (0-5m) and later (5-30m) periods at the event level. Test directional persistence, trend strength, regime stability, and volatility decay.")
        st.markdown("### 0-5m vs 5-30m CVD Z-Score Comparison")
        
        # Use data filtered by year only
        df_compare = df[df['year'].isin(selected_years)].copy()
        
        if df_compare.empty:
            st.warning("No data available for selected years.")
        else:
            # Calculate per-event stats for both windows
            event_stats = []
            min_obs = 5  # Minimum observations per window
            
            for (ticker, evt), grp in df_compare.groupby(['ticker', 'acceptance_datetime_utc']):
                grp = grp.sort_values('seconds_since_event')
                
                # 0-5m window
                win_0_5 = grp[(grp['seconds_since_event'] >= 0) & (grp['seconds_since_event'] <= 300)]['cvd_zscore'].dropna()
                # 5-30m window
                win_5_30 = grp[(grp['seconds_since_event'] > 300) & (grp['seconds_since_event'] <= 1800)]['cvd_zscore'].dropna()
                
                if len(win_0_5) < min_obs or len(win_5_30) < min_obs:
                    continue
                
                # Calculate stats
                event_stats.append({
                    'ticker': ticker,
                    'event': evt,
                    # Means
                    'mean_0_5': win_0_5.mean(),
                    'mean_5_30': win_5_30.mean(),
                    # Percentiles
                    'p5_0_5': np.percentile(win_0_5, 5),
                    'p5_5_30': np.percentile(win_5_30, 5),
                    'p10_0_5': np.percentile(win_0_5, 10),
                    'p10_5_30': np.percentile(win_5_30, 10),
                    'p90_0_5': np.percentile(win_0_5, 90),
                    'p90_5_30': np.percentile(win_5_30, 90),
                    'p95_0_5': np.percentile(win_0_5, 95),
                    'p95_5_30': np.percentile(win_5_30, 95),
                    # Skewness
                    'skew_0_5': win_0_5.skew(),
                    'skew_5_30': win_5_30.skew(),
                    # % Positive observations
                    'pct_pos_0_5': (win_0_5 > 0).mean() * 100,
                    'pct_pos_5_30': (win_5_30 > 0).mean() * 100,
                    # Std Dev
                    'std_0_5': win_0_5.std(),
                    'std_5_30': win_5_30.std(),
                    # Counts
                    'n_0_5': len(win_0_5),
                    'n_5_30': len(win_5_30)
                })
            
            if not event_stats:
                st.warning("No events with sufficient data in both windows.")
            else:
                stats_df = pd.DataFrame(event_stats)
                
                st.markdown(f"**Events analyzed:** {len(stats_df)} | **Tickers:** {stats_df['ticker'].nunique()}")
                
                # ========== Chart 1: Directional Persistence ==========
                st.markdown("---")
                st.markdown("### 1️⃣ Directional Persistence (Mean & Percentiles)")
                st.markdown("*Compare 0-5m vs 5-30m for Mean and key percentiles (5th, 10th, 90th, 95th)*")
                
                # Define metrics to analyze
                persistence_metrics = [
                    ('mean', 'mean_0_5', 'mean_5_30', 'Mean'),
                    ('p5', 'p5_0_5', 'p5_5_30', '5th Percentile'),
                    ('p10', 'p10_0_5', 'p10_5_30', '10th Percentile'),
                    ('p90', 'p90_0_5', 'p90_5_30', '90th Percentile'),
                    ('p95', 'p95_0_5', 'p95_5_30', '95th Percentile'),
                ]
                
                # Calculate regression for all metrics
                regression_results = []
                for metric_key, col_0_5, col_5_30, label in persistence_metrics:
                    reg_data = stats_df[[col_0_5, col_5_30]].dropna()
                    if len(reg_data) >= 10:
                        beta_m, alpha_m = np.polyfit(reg_data[col_0_5], reg_data[col_5_30], 1)
                        y_pred_m = alpha_m + beta_m * reg_data[col_0_5]
                        ss_res_m = ((reg_data[col_5_30] - y_pred_m) ** 2).sum()
                        ss_tot_m = ((reg_data[col_5_30] - reg_data[col_5_30].mean()) ** 2).sum()
                        r2_m = 1 - (ss_res_m / ss_tot_m) if ss_tot_m != 0 else 0
                        corr_m = reg_data.corr().iloc[0, 1]
                    else:
                        alpha_m, beta_m, r2_m, corr_m = np.nan, np.nan, np.nan, np.nan
                    regression_results.append({
                        'metric': label,
                        'col_0_5': col_0_5,
                        'col_5_30': col_5_30,
                        'alpha': alpha_m,
                        'beta': beta_m,
                        'r2': r2_m,
                        'corr': corr_m
                    })
                
                # Regression Summary Table
                reg_summary = pd.DataFrame(regression_results)
                st.markdown("**Regression Summary: Y(5-30m) = α + β × X(0-5m)**")
                st.dataframe(
                    reg_summary[['metric', 'alpha', 'beta', 'r2', 'corr']].style.format({
                        'alpha': '{:.4f}', 'beta': '{:.4f}', 'r2': '{:.4f}', 'corr': '{:.4f}'
                    }).background_gradient(subset=['beta'], cmap='RdYlGn', vmin=0, vmax=1.5),
                    use_container_width=True
                )
                st.caption("β < 1 → mean reversion; β ≈ 1 → persistence; β > 1 → momentum amplification")
                
                # Create tabs for each metric
                metric_tabs = st.tabs([r['metric'] for r in regression_results])
                
                for i, (tab, reg_info) in enumerate(zip(metric_tabs, regression_results)):
                    with tab:
                        col_0_5 = reg_info['col_0_5']
                        col_5_30 = reg_info['col_5_30']
                        alpha_m = reg_info['alpha']
                        beta_m = reg_info['beta']
                        r2_m = reg_info['r2']
                        corr_m = reg_info['corr']
                        
                        col_chart, col_stats = st.columns([3, 1])
                        
                        with col_chart:
                            fig_m = px.scatter(
                                stats_df,
                                x=col_0_5,
                                y=col_5_30,
                                color='ticker',
                                color_discrete_sequence=TICKER_COLORS,
                                hover_data=['ticker', 'event'],
                                title=f"{reg_info['metric']}: 0-5m vs 5-30m",
                                labels={col_0_5: f"{reg_info['metric']} (0-5m)", col_5_30: f"{reg_info['metric']} (5-30m)"}
                            )
                            # Add diagonal line (y=x)
                            min_val = min(stats_df[col_0_5].min(), stats_df[col_5_30].min())
                            max_val = max(stats_df[col_0_5].max(), stats_df[col_5_30].max())
                            fig_m.add_trace(go.Scatter(
                                x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', line=dict(color='gray', dash='dash', width=1),
                                name='y=x', showlegend=False
                            ))
                            # Add regression line if valid
                            if not np.isnan(alpha_m):
                                x_range = np.array([stats_df[col_0_5].min(), stats_df[col_0_5].max()])
                                y_reg = alpha_m + beta_m * x_range
                                fig_m.add_trace(go.Scatter(
                                    x=x_range, y=y_reg,
                                    mode='lines', line=dict(color='red', width=2),
                                    name=f'Fit: β={beta_m:.3f}', showlegend=True
                                ))
                            fig_m.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                            fig_m.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                            fig_m.update_layout(
                                height=400, template='plotly_white',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                            )
                            st.plotly_chart(fig_m, use_container_width=True, key=f"compare_{col_0_5}")
                        
                        with col_stats:
                            if not np.isnan(corr_m):
                                st.metric("Correlation", f"{corr_m:.3f}")
                                same_sign = ((stats_df[col_0_5] > 0) == (stats_df[col_5_30] > 0)).mean() * 100
                                st.metric("Same Sign %", f"{same_sign:.1f}%")
                                st.markdown("**Regression**")
                                st.metric("α", f"{alpha_m:.4f}")
                                st.metric("β", f"{beta_m:.4f}")
                                st.metric("R²", f"{r2_m:.4f}")
                            else:
                                st.warning("Insufficient data")
                
                # ========== Chart 2: Trend Strength (Skewness) ==========
                st.markdown("---")
                st.markdown("### 2️⃣ Trend Strength (Skewness Check)")
                st.markdown("*If 0-5m had a fat tail (aggressive buying/selling), does it persist?*")
                
                col2a, col2b = st.columns([3, 1])
                
                with col2a:
                    # Filter extreme skewness values
                    skew_df = stats_df[(stats_df['skew_0_5'].abs() < 5) & (stats_df['skew_5_30'].abs() < 5)]
                    
                    fig_skew = px.scatter(
                        skew_df,
                        x='skew_0_5',
                        y='skew_5_30',
                        color='ticker',
                        color_discrete_sequence=TICKER_COLORS,
                        hover_data=['ticker', 'event'],
                        title="Skewness: 0-5m vs 5-30m",
                        labels={'skew_0_5': 'Skewness (0-5m)', 'skew_5_30': 'Skewness (5-30m)'}
                    )
                    # Add diagonal line
                    min_s = min(skew_df['skew_0_5'].min(), skew_df['skew_5_30'].min())
                    max_s = max(skew_df['skew_0_5'].max(), skew_df['skew_5_30'].max())
                    fig_skew.add_trace(go.Scatter(
                        x=[min_s, max_s], y=[min_s, max_s],
                        mode='lines', line=dict(color='gray', dash='dash', width=1),
                        name='y=x', showlegend=False
                    ))
                    fig_skew.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_skew.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_skew.update_layout(
                        height=450, template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_skew, use_container_width=True, key="compare_skew")
                
                with col2b:
                    corr_skew = skew_df[['skew_0_5', 'skew_5_30']].corr().iloc[0, 1]
                    same_sign_skew = ((skew_df['skew_0_5'] > 0) == (skew_df['skew_5_30'] > 0)).mean() * 100
                    st.metric("Correlation", f"{corr_skew:.3f}")
                    st.metric("Same Sign %", f"{same_sign_skew:.1f}%")
                    st.caption("Positive corr → tail direction persists")
                
                # ========== Chart 3: Regime Stability (% Positive) ==========
                st.markdown("---")
                st.markdown("### 3️⃣ Regime Stability (Sign Ratio)")
                st.markdown("*If 80% of ticks were positive in 0-5m, what % are positive in 5-30m?*")
                
                col3a, col3b = st.columns([3, 1])
                
                with col3a:
                    fig_pct = px.scatter(
                        stats_df,
                        x='pct_pos_0_5',
                        y='pct_pos_5_30',
                        color='ticker',
                        color_discrete_sequence=TICKER_COLORS,
                        hover_data=['ticker', 'event'],
                        title="% Positive Observations: 0-5m vs 5-30m",
                        labels={'pct_pos_0_5': '% Positive (0-5m)', 'pct_pos_5_30': '% Positive (5-30m)'}
                    )
                    # Add diagonal line
                    fig_pct.add_trace(go.Scatter(
                        x=[0, 100], y=[0, 100],
                        mode='lines', line=dict(color='gray', dash='dash', width=1),
                        name='y=x', showlegend=False
                    ))
                    fig_pct.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_pct.add_vline(x=50, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_pct.update_layout(
                        height=450, template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_pct, use_container_width=True, key="compare_pct_pos")
                
                with col3b:
                    corr_pct = stats_df[['pct_pos_0_5', 'pct_pos_5_30']].corr().iloc[0, 1]
                    # % of events where both windows are >50% positive or both <50%
                    same_regime = (((stats_df['pct_pos_0_5'] > 50) == (stats_df['pct_pos_5_30'] > 50))).mean() * 100
                    st.metric("Correlation", f"{corr_pct:.3f}")
                    st.metric("Same Regime %", f"{same_regime:.1f}%")
                    st.caption("High values → regime persists")
                
                # ========== Chart 4: Volatility Decay (Std Dev) ==========
                st.markdown("---")
                st.markdown("### 4️⃣ Volatility Decay (Sigma Ratio)")
                st.markdown("*Does high volatility in 0-5m predict high volatility in 5-30m, or mean-revert?*")
                
                col4a, col4b = st.columns([3, 1])
                
                with col4a:
                    fig_std = px.scatter(
                        stats_df,
                        x='std_0_5',
                        y='std_5_30',
                        color='ticker',
                        color_discrete_sequence=TICKER_COLORS,
                        hover_data=['ticker', 'event'],
                        title="Std Dev: 0-5m vs 5-30m",
                        labels={'std_0_5': 'Std Dev (0-5m)', 'std_5_30': 'Std Dev (5-30m)'}
                    )
                    # Add diagonal line
                    max_std = max(stats_df['std_0_5'].max(), stats_df['std_5_30'].max())
                    fig_std.add_trace(go.Scatter(
                        x=[0, max_std], y=[0, max_std],
                        mode='lines', line=dict(color='gray', dash='dash', width=1),
                        name='y=x', showlegend=False
                    ))
                    fig_std.update_layout(
                        height=450, template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_std, use_container_width=True, key="compare_std")
                
                with col4b:
                    corr_std = stats_df[['std_0_5', 'std_5_30']].corr().iloc[0, 1]
                    # % where 5-30m volatility is lower
                    vol_decay_pct = (stats_df['std_5_30'] < stats_df['std_0_5']).mean() * 100
                    # Calculate k = Median(σ_5-30 / σ_0-5)
                    sigma_ratio = stats_df['std_5_30'] / stats_df['std_0_5']
                    k_median = sigma_ratio.median()
                    st.metric("Correlation", f"{corr_std:.3f}")
                    st.metric("Vol Decay %", f"{vol_decay_pct:.1f}%")
                    st.markdown("**Median Sigma Ratio**")
                    st.latex(r"k = \text{Median}\left( \frac{\sigma_{5\text{-}30}}{\sigma_{0\text{-}5}} \right)")
                    st.metric("k", f"{k_median:.4f}")
                    st.caption("k < 1 → volatility decays; k > 1 → volatility persists")
                
                # ========== Summary Table ==========
                st.markdown("---")
                st.markdown("### Summary Statistics")
                
                summary_data = {
                    'Metric': ['Mean', 'Skewness', '% Positive', 'Std Dev'],
                    'Correlation': [
                        stats_df[['mean_0_5', 'mean_5_30']].corr().iloc[0, 1],
                        skew_df[['skew_0_5', 'skew_5_30']].corr().iloc[0, 1],
                        stats_df[['pct_pos_0_5', 'pct_pos_5_30']].corr().iloc[0, 1],
                        stats_df[['std_0_5', 'std_5_30']].corr().iloc[0, 1]
                    ],
                    'Interpretation': [
                        'Directional persistence',
                        'Tail direction persistence',
                        'Regime stability',
                        'Volatility clustering'
                    ]
                }
                summary_table = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_table.style.format({'Correlation': '{:.3f}'}).background_gradient(
                        subset=['Correlation'], cmap='RdYlGn', vmin=-0.5, vmax=1.0
                    ),
                    use_container_width=True
                )
                
                # Raw data expander
                with st.expander("📋 View Event-Level Data"):
                    display_cols = ['ticker', 'event', 'mean_0_5', 'mean_5_30', 'skew_0_5', 'skew_5_30',
                                    'pct_pos_0_5', 'pct_pos_5_30', 'std_0_5', 'std_5_30']
                    st.dataframe(
                        stats_df[display_cols].style.format({
                            'mean_0_5': '{:.3f}', 'mean_5_30': '{:.3f}',
                            'skew_0_5': '{:.2f}', 'skew_5_30': '{:.2f}',
                            'pct_pos_0_5': '{:.1f}%', 'pct_pos_5_30': '{:.1f}%',
                            'std_0_5': '{:.3f}', 'std_5_30': '{:.3f}'
                        }),
                        use_container_width=True,
                        height=400
                    )


    # ========== TAB 9: Logistic Regression ==========
    with tab9:
        st.info("🎲 **Overview**: Logistic regression model predicting positive 10-minute forward returns using CVD Z-Score and VWAP Distance % as predictors. Data from 5m to 30m after event.")
        st.markdown("### Logistic Regression: Predicting Return Direction")
        
        # Artifacts directory
        artifacts_dir = os.path.join(os.path.dirname(__file__), "logistic_regression_artifacts")
        
        # Check if artifacts exist
        required_files = [
            "model_summary.csv",
            "confusion_metrics.json",
            "roc_curve.csv",
            "probability_distribution.csv",
            "metadata.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(artifacts_dir, f))]
        
        if missing_files:
            st.warning(f"""
            **Artifacts not found.** Please run the training script first:
            
            ```bash
            source venv/bin/activate && python analysis/fit_logistic_regression.py
            ```
            
            Missing files: {', '.join(missing_files)}
            """)
        else:
            try:
                # Load artifacts
                model_summary = pd.read_csv(os.path.join(artifacts_dir, "model_summary.csv"))
                
                with open(os.path.join(artifacts_dir, "confusion_metrics.json"), 'r') as f:
                    confusion_data = json.load(f)
                
                roc_data = pd.read_csv(os.path.join(artifacts_dir, "roc_curve.csv"))
                prob_dist = pd.read_csv(os.path.join(artifacts_dir, "probability_distribution.csv"))
                
                with open(os.path.join(artifacts_dir, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                
                # ========== Model Summary ==========
                st.markdown("#### Model Summary")
                
                # Format and display
                st.dataframe(
                    model_summary.style.format({
                        'Coefficient': '{:.4f}',
                        'Std_Error': '{:.4f}',
                        'Z_Score': '{:.3f}',
                        'P_Value': '{:.4f}',
                        'Odds_Ratio': '{:.4f}',
                        'CI_Lower_95': '{:.4f}',
                        'CI_Upper_95': '{:.4f}'
                    }).apply(
                        lambda x: ['background-color: #d4edda' if v < 0.05 else '' 
                                   for v in x] if x.name == 'P_Value' else [''] * len(x),
                        axis=0
                    ),
                    use_container_width=True
                )
                
                # Interpretation
                st.markdown("**Interpretation:**")
                interp_cols = st.columns(2)
                
                cvd_row = model_summary[model_summary['Variable'] == 'CVD Z-Score'].iloc[0]
                vwap_row = model_summary[model_summary['Variable'] == 'VWAP Distance %'].iloc[0]
                
                with interp_cols[0]:
                    cvd_or = cvd_row['Odds_Ratio']
                    cvd_pval = cvd_row['P_Value']
                    sig_cvd = "✅ Significant" if cvd_pval < 0.05 else "❌ Not significant"
                    st.metric(
                        "CVD Z-Score Odds Ratio",
                        f"{cvd_or:.4f}",
                        delta=f"p={cvd_pval:.4f} {sig_cvd}"
                    )
                    if cvd_or > 1:
                        st.caption("Higher CVD → Higher odds of positive return")
                    else:
                        st.caption("Higher CVD → Lower odds of positive return")
                
                with interp_cols[1]:
                    vwap_or = vwap_row['Odds_Ratio']
                    vwap_pval = vwap_row['P_Value']
                    sig_vwap = "✅ Significant" if vwap_pval < 0.05 else "❌ Not significant"
                    st.metric(
                        "VWAP Distance % Odds Ratio",
                        f"{vwap_or:.4f}",
                        delta=f"p={vwap_pval:.4f} {sig_vwap}"
                    )
                    if vwap_or > 1:
                        st.caption("Further from VWAP → Higher odds of positive return")
                    else:
                        st.caption("Further from VWAP → Lower odds of positive return")
                
                # ========== Confusion Matrix ==========
                st.markdown("---")
                st.markdown("#### Confusion Matrix")
                
                cm = np.array(confusion_data['confusion_matrix'])
                
                # Create annotated heatmap
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    text=cm,
                    texttemplate='%{text}',
                    textfont={'size': 20},
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig_cm.update_layout(
                    title='Confusion Matrix',
                    height=350,
                    template='plotly_white',
                    xaxis_title='Predicted',
                    yaxis_title='Actual'
                )
                
                cm_col1, cm_col2 = st.columns([2, 1])
                with cm_col1:
                    st.plotly_chart(fig_cm, use_container_width=True, key="logreg_cm")
                
                with cm_col2:
                    st.markdown("**Classification Metrics**")
                    st.metric("Accuracy", f"{confusion_data['accuracy']:.3f}")
                    st.metric("Precision", f"{confusion_data['precision']:.3f}")
                    st.metric("Recall", f"{confusion_data['recall']:.3f}")
                    st.metric("F1 Score", f"{confusion_data['f1_score']:.3f}")
                
                # ========== ROC Curve ==========
                st.markdown("---")
                st.markdown("#### ROC Curve")
                
                roc_auc = metadata['roc_auc']
                
                fig_roc = go.Figure()
                
                # ROC curve
                fig_roc.add_trace(go.Scatter(
                    x=roc_data['fpr'], y=roc_data['tpr'],
                    mode='lines',
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                # Diagonal reference
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random (AUC = 0.5)',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig_roc.update_layout(
                    title=f'ROC Curve (AUC = {roc_auc:.3f})',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400,
                    template='plotly_white',
                    legend=dict(x=0.6, y=0.1)
                )
                
                roc_col1, roc_col2 = st.columns([3, 1])
                with roc_col1:
                    st.plotly_chart(fig_roc, use_container_width=True, key="logreg_roc")
                
                with roc_col2:
                    st.markdown("**AUC Interpretation:**")
                    if roc_auc >= 0.7:
                        st.success(f"AUC = {roc_auc:.3f}\nGood discrimination")
                    elif roc_auc >= 0.6:
                        st.warning(f"AUC = {roc_auc:.3f}\nModerate discrimination")
                    else:
                        st.error(f"AUC = {roc_auc:.3f}\nPoor discrimination")
                    
                    st.caption("""
                    - 0.5 = No discrimination (random)
                    - 0.6-0.7 = Poor
                    - 0.7-0.8 = Acceptable
                    - 0.8-0.9 = Excellent
                    - >0.9 = Outstanding
                    """)
                
                # ========== Predicted Probability Distribution ==========
                st.markdown("---")
                st.markdown("#### Predicted Probability Distribution")
                
                prob_dist['Actual Outcome'] = prob_dist['actual'].apply(
                    lambda x: 'Positive' if x == 1 else 'Negative'
                )
                
                # Two-column layout for better visualizations
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    # Box plot - shows distribution separation clearly
                    fig_box = go.Figure()
                    
                    for outcome, color in [('Negative', '#ef4444'), ('Positive', '#22c55e')]:
                        outcome_data = prob_dist[prob_dist['Actual Outcome'] == outcome]['predicted_prob']
                        fig_box.add_trace(go.Box(
                            y=outcome_data,
                            name=outcome,
                            marker_color=color,
                            boxpoints='outliers',
                            jitter=0.3
                        ))
                    
                    fig_box.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                      annotation_text="Decision Boundary")
                    fig_box.update_layout(
                        title='P(Positive) by Actual Outcome',
                        yaxis_title='Predicted Probability',
                        height=400,
                        template='plotly_white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_box, use_container_width=True, key="logreg_box")
                
                with prob_col2:
                    # Calibration plot - actual positive rate in each probability bin
                    n_bins = 10
                    prob_dist['prob_bin'] = pd.cut(prob_dist['predicted_prob'], bins=n_bins, labels=False)
                    
                    calibration_data = prob_dist.groupby('prob_bin').agg(
                        bin_center=('predicted_prob', 'mean'),
                        actual_positive_rate=('actual', 'mean'),
                        count=('actual', 'count')
                    ).reset_index()
                    
                    fig_cal = go.Figure()
                    
                    # Ideal calibration line
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Perfect Calibration',
                        line=dict(color='gray', dash='dash', width=1)
                    ))
                    
                    # Actual calibration
                    fig_cal.add_trace(go.Bar(
                        x=calibration_data['bin_center'],
                        y=calibration_data['actual_positive_rate'],
                        name='Actual Positive Rate',
                        marker_color='#3b82f6',
                        opacity=0.8,
                        width=0.08
                    ))
                    
                    fig_cal.update_layout(
                        title='Calibration: Predicted vs Actual',
                        xaxis_title='Mean Predicted Probability (bin)',
                        yaxis_title='Actual Positive Rate',
                        height=400,
                        template='plotly_white',
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        legend=dict(x=0.02, y=0.98)
                    )
                    st.plotly_chart(fig_cal, use_container_width=True, key="logreg_calibration")
                
                # Violin plot below - full width for detailed view
                st.markdown("**Detailed Distribution (Violin Plot):**")
                fig_violin = go.Figure()
                
                for outcome, color in [('Negative', '#ef4444'), ('Positive', '#22c55e')]:
                    outcome_data = prob_dist[prob_dist['Actual Outcome'] == outcome]['predicted_prob']
                    fig_violin.add_trace(go.Violin(
                        x=[outcome] * len(outcome_data),
                        y=outcome_data,
                        name=outcome,
                        fillcolor=color,
                        opacity=0.6,
                        line_color=color,
                        meanline_visible=True,
                        box_visible=True
                    ))
                
                fig_violin.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig_violin.update_layout(
                    title='Probability Distribution by Actual Outcome',
                    xaxis_title='Actual Outcome',
                    yaxis_title='Predicted Probability',
                    height=350,
                    template='plotly_white',
                    showlegend=False
                )
                st.plotly_chart(fig_violin, use_container_width=True, key="logreg_violin")
                
                # Summary stats
                st.markdown("**Probability Statistics by Outcome:**")
                stats_col1, stats_col2 = st.columns(2)
                
                neg_probs = prob_dist[prob_dist['actual'] == 0]['predicted_prob']
                pos_probs = prob_dist[prob_dist['actual'] == 1]['predicted_prob']
                
                with stats_col1:
                    st.markdown("**Actual Negative:**")
                    st.caption(f"Mean P: {neg_probs.mean():.3f} | Median: {neg_probs.median():.3f}")
                    st.caption(f"% correctly predicted <0.5: {(neg_probs < 0.5).mean()*100:.1f}%")
                
                with stats_col2:
                    st.markdown("**Actual Positive:**")
                    st.caption(f"Mean P: {pos_probs.mean():.3f} | Median: {pos_probs.median():.3f}")
                    st.caption(f"% correctly predicted ≥0.5: {(pos_probs >= 0.5).mean()*100:.1f}%")
                
                # ========== Data Summary ==========
                st.markdown("---")
                st.markdown("#### Data Summary")
                
                data_cols = st.columns(4)
                with data_cols[0]:
                    st.metric("Total Observations", f"{metadata['total_observations']:,}")
                with data_cols[1]:
                    st.metric("Positive Returns", f"{metadata['positive_returns']:,} ({metadata['positive_pct']:.1f}%)")
                with data_cols[2]:
                    st.metric("Negative Returns", f"{metadata['negative_returns']:,} ({100-metadata['positive_pct']:.1f}%)")
                with data_cols[3]:
                    st.metric("Events Included", f"{metadata['n_events']}")
                
                # Data filter info
                with st.expander("📋 Model Details"):
                    st.markdown(f"""
                    **Data Filters:**
                    - Years: {metadata['data_filter']['years']}
                    - Time Window: {metadata['data_filter']['time_window_minutes']}
                    
                    **Predictors:** {', '.join(metadata['predictors'])}
                    
                    **Target:** {metadata['target']}
                    
                    **To retrain the model:**
                    ```bash
                    source venv/bin/activate && python analysis/fit_logistic_regression.py
                    ```
                    """)
            
            except Exception as e:
                st.error(f"Error loading artifacts: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
