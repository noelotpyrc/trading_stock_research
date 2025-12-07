"""
CVD Z-Score Viewer
Visualize CVD z-score behavior across earnings events.
"""
import streamlit as st
import pandas as pd
import numpy as np
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 CVD Z-Score Trajectories & Distributions",
        "📉 Ticker Correlations",
        "🔍 Event Deep Dive",
        "🎯 Correlation Group Analysis",
        "📈 Event Correlation Deep Dive",
        "📐 VWAP Deviation Explorer",
        "🔬 First 5m Signals Explorer"
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


if __name__ == "__main__":
    main()
