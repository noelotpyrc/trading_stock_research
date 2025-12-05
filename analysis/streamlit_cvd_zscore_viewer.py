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

# Data path
DATA_PATH = "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv"

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
def load_data():
    """Load the consolidated earnings data."""
    try:
        df = pd.read_csv(DATA_PATH)
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
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available. Please check the data path.")
        return

    # ========== SIDEBAR ==========
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 All Events (by Ticker)",
        "🔍 Single Event",
        "📈 Event Correlation Explorer",
        "🎯 Strategy Group Analysis",
        "📉 Ticker Correlations"
    ])
    
    # ========== TAB 1: Events by Ticker ==========
    with tab1:
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

    # ========== TAB 2: Single Event ==========
    with tab2:
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

    # ========== TAB 3: Event Correlation Explorer ==========
    with tab3:
        st.markdown("### Event-Level Correlations")
        st.markdown("Explore what predicts CVD z-score correlation with forward returns.")
        
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
            index=0,
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
        st.markdown("### Strategy Group Analysis")
        st.markdown("Group events by correlation strength to identify what predicts each strategy regime.")
        
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

    # ========== TAB 5: Ticker Correlations ==========
    with tab5:
        st.markdown("### Ticker-Level CVD Z-Score Correlations")
        st.markdown("Correlation between CVD Z-Score and 10min forward return by ticker (using sidebar filters)")
        
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


if __name__ == "__main__":
    main()
