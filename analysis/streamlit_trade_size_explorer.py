#!/usr/bin/env python3
"""
Interactive Streamlit app to explore average trade size distributions.

Features:
- Select ticker and earnings event
- Toggle between 1-min and 5-min time groupings
- Visualize avg_trade_size over time, colored by trading hours
- Earnings event marked on chart
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Paths
DATA_FILE = Path('data/avg_trade_size_exploration/all_events.csv.gz')

# Color mapping for trading hours
TRADING_HOURS_COLORS = {
    'Pre': '#FFA500',      # Orange
    'Regular': '#2E86DE',  # Blue
    'Post': '#9B59B6',     # Purple
    'Overnight': '#95A5A6' # Gray
}


@st.cache_data
def load_trade_data():
    """Load pre-processed trade size data."""
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.info("Please run: `python analysis/prepare_trade_size_data.py`")
        st.stop()
    
    df = pd.read_csv(DATA_FILE, compression='gzip', parse_dates=['event_datetime', 'timestamp'])
    
    # Convert to naive UTC (strip timezone) to force Plotly to display UTC time
    # otherwise it converts to local time (e.g. 20:30 UTC -> 16:30 ET)
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    if df['event_datetime'].dt.tz is not None:
        df['event_datetime'] = df['event_datetime'].dt.tz_localize(None)
        
    return df


def create_trade_size_chart(df, bucket_col, event_datetime, log_scale=False):
    """
    Create interactive chart showing avg_trade_size vs time buckets.
    
    Plots ALL individual second bars (not aggregated), using the bucket column
    for x-axis positioning. This shows the distribution within each bucket.
    
    Args:
        df: DataFrame with second bar data (NOT aggregated)
        bucket_col: 'time_bucket_1min' or 'time_bucket_5min' for x-axis
        event_datetime: Earnings event datetime
        log_scale: Whether to use log scale for y-axis
        
    Returns:
        Plotly figure
    """
    # Create string format for Plotly to prevent timezone conversion
    # This forces "Wall Time" display (e.g. 20:30 stays 20:30)
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    fig = go.Figure()
    
    # Plot each trading hour category separately for color coding
    # Each point is an individual second bar
    for trading_hour in ['Pre', 'Regular', 'Post', 'Overnight']:
        hour_df = df[df['trading_hours'] == trading_hour]
        
        if len(hour_df) > 0:
            fig.add_trace(go.Scatter(
                x=hour_df['timestamp_str'],  # Use string to force wall time
                y=hour_df['avg_trade_size'],
                mode='markers',
                name=trading_hour,
                marker=dict(
                    color=TRADING_HOURS_COLORS[trading_hour],
                    size=4,
                    opacity=0.5
                ),
                hovertemplate=(
                    f'<b>{trading_hour}</b><br>' +
                    'Time: %{x}<br>' +
                    'Avg Trade Size: $%{y:,.0f}<br>' +
                    '<extra></extra>'
                )
            ))
    
    # Add vertical line at earnings event
    event_str = event_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.add_vline(
        x=event_str, 
        line_dash="dash", 
        line_color="red", 
        line_width=2
    )
    
    # Add annotation separately to avoid TypeError with string x-axis
    fig.add_annotation(
        x=event_str,
        y=1,
        yref="paper",
        text="Earnings Event",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )
    
    # Update layout
    bucket_size = '1-min' if '1min' in bucket_col else '5-min'
    
    fig.update_layout(
        title=f"Average Trade Size Distribution ({bucket_size} buckets, individual second bars)",
        xaxis_title="Time (UTC)",
        yaxis_title="Average Trade Size ($)",
        yaxis_type="log" if log_scale else "linear",
        xaxis_type="date",  # Force date axis to handle string timestamps continuously
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title="Trading Hours",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        template="plotly_white"
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_candlestick_chart(df, bucket_col, event_datetime):
    """
    Create candlestick chart with round-lot trades overlaid.
    
    Args:
        df: DataFrame with second bar data
        bucket_col: 'time_bucket_1min' or 'time_bucket_5min'
        event_datetime: Earnings event datetime
        
    Returns:
        Plotly figure
    """
    # Identify round lots (num=1 and vol is multiple of 100 up to 500)
    df = df.copy()
    df['is_round_lot'] = (
        (df['num'] == 1) & 
        (df['vol'].isin([100, 200, 300, 400, 500]))
    )
    
    # Create string timestamp for grouping
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create candles
    candles = df.groupby(bucket_col).agg({
        'timestamp_str': 'first',  # Use string timestamp
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'vol': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=candles['timestamp_str'],
        open=candles['open'],
        high=candles['high'],
        low=candles['low'],
        close=candles['close'],
        name='Price'
    ))
    
    # Round lot overlay
    round_lots = df[df['is_round_lot']]
    
    # Size map for different lot sizes (larger volume = larger dot)
    lot_sizes = {
        100: 4,
        200: 6,
        300: 8,
        400: 10,
        500: 12
    }
    
    for vol_size in [100, 200, 300, 400, 500]:
        subset = round_lots[round_lots['vol'] == vol_size]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['timestamp_str'],  # Use string timestamp
                y=subset['vw'],
                mode='markers',
                marker=dict(
                    size=lot_sizes.get(vol_size, 4), 
                    color='rgba(0, 0, 255, 0.5)',  # Semi-transparent blue
                    line=dict(color='blue', width=1), # Solid outline
                    symbol='circle'
                ),
                name=f'{vol_size}sh lots',
                hovertemplate=(
                    f'<b>{vol_size} shares</b><br>' +
                    'Time: %{x}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    '<extra></extra>'
                )
            ))
            
    # Add vertical line at earnings event
    event_str = event_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    fig.add_vline(
        x=event_str, 
        line_dash="dash", 
        line_color="blue", 
        line_width=2
    )
    
    # Add annotation separately
    fig.add_annotation(
        x=event_str,
        y=1,
        yref="paper",
        text="Earnings Event",
        showarrow=False,
        yshift=10,
        font=dict(color="blue")
    )
    
    bucket_size = '1-min' if '1min' in bucket_col else '5-min'
    
    fig.update_layout(
        title=f"Price Action with Round Lot Trades ({bucket_size} buckets)",
        xaxis_title="Time (UTC)",
        yaxis_title="Price ($)",
        xaxis_type="date",  # Force date axis to handle string timestamps continuously
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig


def create_combined_round_lot_flow_chart(df, event_datetime):
    """
    Create combined chart showing both round lot CVD metrics.
    
    Args:
        df: DataFrame with second bar data
        event_datetime: Earnings event datetime
        
    Returns:
        Plotly figure with dual y-axes
    """
    # Identify round lots
    df = df.copy()
    df['is_round_lot'] = (
        (df['num'] == 1) & 
        (df['vol'].isin([100, 200, 300, 400, 500]))
    )
    
    # Filter to round lots only
    round_lots = df[df['is_round_lot']].copy()
    
    if len(round_lots) == 0:
        # Return empty chart
        fig = go.Figure()
        fig.update_layout(title="Round Lot Flow (No Data)")
        return fig
    
    # Sort by timestamp
    round_lots = round_lots.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate previous price for tick rule
    round_lots['prev_close'] = round_lots['close'].shift(1)
    
    # Calculate signed volume (for CVD 1)
    round_lots['signed_volume'] = np.where(
        round_lots['vw'] > round_lots['prev_close'], 
        round_lots['vol'],
        np.where(
            round_lots['vw'] < round_lots['prev_close'],
            -round_lots['vol'],
            0
        )
    )
    
    # Calculate percentage price impact (for CVD 2)
    round_lots['price_impact_pct'] = np.where(
        round_lots['prev_close'] > 0,
        ((round_lots['vw'] - round_lots['prev_close']) / round_lots['prev_close']) * 100,
        0
    )
    round_lots['weighted_impact'] = round_lots['price_impact_pct'] * round_lots['vol']
    
    # Filter to data after earnings event
    round_lots = round_lots[round_lots['timestamp'] >= event_datetime].copy()
    
    # Calculate cumulative sums
    round_lots['cvd'] = round_lots['signed_volume'].cumsum()
    round_lots['impact_cvd'] = round_lots['weighted_impact'].cumsum()
    
    # Create string timestamp
    round_lots['timestamp_str'] = round_lots['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sign-based CVD (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=round_lots['timestamp_str'],
            y=round_lots['cvd'],
            mode='lines',
            name='Sign-Based CVD',
            line=dict(color='purple', width=2),
            hovertemplate='Time: %{x}<br>CVD: %{y:,.0f} shares<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add price-impact CVD (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=round_lots['timestamp_str'],
            y=round_lots['impact_cvd'],
            mode='lines',
            name='Impact-Weighted CVD',
            line=dict(color='darkgreen', width=2, dash='dot'),
            hovertemplate='Time: %{x}<br>Impact CVD: %{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, secondary_y=False)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, secondary_y=True)
    
    # Update layout
    fig.update_xaxes(
        title_text="Time (UTC)",
        type="date"
    )
    
    fig.update_yaxes(
        title_text="Sign-Based CVD (shares)",
        secondary_y=False,
        title_font=dict(color='purple')
    )
    
    fig.update_yaxes(
        title_text="Impact-Weighted CVD (% × shares)",
        secondary_y=True,
        title_font=dict(color='darkgreen')
    )
    
    fig.update_layout(
        title="Round Lot Flow - Combined CVD Metrics",
        height=400,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_second_bar_analysis_chart(df, event_datetime):
    """
    Create combined chart with candlestick + round lot overlay (top) and 
    round lot flow metrics (bottom) sharing the same x-axis for synchronized zooming.
    
    Args:
        df: DataFrame with second bar data
        event_datetime: Earnings event datetime
        
    Returns:
        Plotly figure with subplots
    """
    # Filter to data after earnings event
    df = df[df['timestamp'] >= event_datetime].copy()
    
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(title="Second Bar Analysis (No Data After Earnings)")
        return fig
    
    # Identify round lots
    df['is_round_lot'] = (
        (df['num'] == 1) & 
        (df['vol'].isin([100, 200, 300, 400, 500]))
    )
    
    # Create string timestamp
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare round lot data
    round_lots = df[df['is_round_lot']].copy()
    
    if len(round_lots) > 0:
        # Sort and calculate metrics
        round_lots = round_lots.sort_values('timestamp').reset_index(drop=True)
        round_lots['prev_close'] = round_lots['close'].shift(1)
        
        # Sign-based CVD
        round_lots['signed_volume'] = np.where(
            round_lots['vw'] > round_lots['prev_close'], 
            round_lots['vol'],
            np.where(
                round_lots['vw'] < round_lots['prev_close'],
                -round_lots['vol'],
                0
            )
        )
        round_lots['cvd'] = round_lots['signed_volume'].cumsum()
        
        # Price-impact CVD
        round_lots['price_impact_pct'] = np.where(
            round_lots['prev_close'] > 0,
            ((round_lots['vw'] - round_lots['prev_close']) / round_lots['prev_close']) * 100,
            0
        )
        round_lots['weighted_impact'] = round_lots['price_impact_pct'] * round_lots['vol']
        round_lots['impact_cvd'] = round_lots['weighted_impact'].cumsum()
        
        round_lots['timestamp_str'] = round_lots['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create subplots: row 1 = candlestick, row 2 = flow metrics
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ],
        subplot_titles=("Price Action with Round Lots", "Round Lot Flow Metrics")
    )
    
    # ===== ROW 1: Candlestick with Round Lot Overlay =====
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp_str'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Round lot overlay
    if len(round_lots) > 0:
        lot_sizes = {100: 4, 200: 6, 300: 8, 400: 10, 500: 12}
        
        for vol_size in [100, 200, 300, 400, 500]:
            subset = round_lots[round_lots['vol'] == vol_size]
            if len(subset) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=subset['timestamp_str'],
                        y=subset['vw'],
                        mode='markers',
                        marker=dict(
                            size=lot_sizes.get(vol_size, 4), 
                            color='rgba(0, 0, 255, 0.5)',
                            line=dict(color='blue', width=1),
                            symbol='circle'
                        ),
                        name=f'{vol_size}sh',
                        legendgroup='lots',
                        hovertemplate=(
                            f'<b>{vol_size} shares</b><br>' +
                            'Time: %{x}<br>' +
                            'Price: $%{y:.2f}<br>' +
                            '<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
    
    # ===== ROW 2: Round Lot Flow Metrics =====
    if len(round_lots) > 0:
        # Sign-based CVD (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=round_lots['timestamp_str'],
                y=round_lots['cvd'],
                mode='lines',
                name='Sign CVD',
                line=dict(color='purple', width=2),
                legendgroup='flow',
                hovertemplate='CVD: %{y:,.0f} shares<extra></extra>'
            ),
            row=2, col=1,
            secondary_y=False
        )
        
        # Price-impact CVD (right y-axis)
        fig.add_trace(
            go.Scatter(
                x=round_lots['timestamp_str'],
                y=round_lots['impact_cvd'],
                mode='lines',
                name='Impact CVD',
                line=dict(color='darkgreen', width=2, dash='dot'),
                legendgroup='flow',
                hovertemplate='Impact: %{y:,.2f}<extra></extra>'
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # Zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1, secondary_y=False)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1, secondary_y=True)
    
    # Add earnings event marker (both rows)
    event_str = event_datetime.strftime('%Y-%m-%d %H:%M:%S')
    for row in [1, 2]:
        fig.add_vline(
            x=event_str, 
            line_dash="dash", 
            line_color="blue", 
            line_width=2,
            row=row, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Time (UTC)", type="date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sign CVD (shares)", title_font=dict(color='purple'), row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Impact CVD (% × shares)", title_font=dict(color='darkgreen'), row=2, col=1, secondary_y=True)
    
    # Layout
    fig.update_layout(
        height=800,
        template="plotly_white",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():


    st.set_page_config(
        page_title="Trade Size Explorer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Average Trade Size Explorer")
    st.markdown("Explore average dollar amount per trade across different earnings events and time periods")
    
    # Load data
    # Force clear cache to ensure we get the new naive timestamps
    # st.cache_data.clear()  # Uncomment if needed, but changing the function body usually invalidates cache
    
    with st.spinner("Loading data..."):
        df = load_trade_data()
        
    # Debug: Check timezone status
    # st.write(f"Debug - Event Time Raw: {df['event_datetime'].iloc[0]}")
    # st.write(f"Debug - Timezone info: {df['event_datetime'].dt.tz}")
    
    # Sidebar controls
    st.sidebar.header("📊 Filters")
    
    # Ticker selection
    tickers = sorted(df['ticker'].unique())
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        tickers,
        index=0
    )
    
    # Filter to selected ticker
    ticker_df = df[df['ticker'] == selected_ticker].copy()
    
    # Event selection
    events = ticker_df[['event_date', 'eps_estimate', 'reported_eps', 'surprise_pct']].drop_duplicates()
    events = events.sort_values('event_date')
    
    event_options = []
    for _, row in events.iterrows():
        eps_str = f"EPS: {row['reported_eps']:.2f}" if pd.notna(row['reported_eps']) else "EPS: N/A"
        surp_str = f"({row['surprise_pct']:+.1f}%)" if pd.notna(row['surprise_pct']) else ""
        event_options.append(f"{row['event_date']} - {eps_str} {surp_str}")
    
    selected_event_idx = st.sidebar.selectbox(
        "Select Earnings Event",
        range(len(event_options)),
        format_func=lambda i: event_options[i]
    )
    
    selected_event_date = events.iloc[selected_event_idx]['event_date']
    
    # Time grouping
    time_grouping = st.sidebar.radio(
        "⏰ Time Grouping",
        ["1-min", "5-min"],
        index=1  # Default to 5-min
    )
    
    bucket_col = 'time_bucket_1min' if time_grouping == '1-min' else 'time_bucket_5min'
    
    st.sidebar.markdown("---")
    
    # Display options
    st.sidebar.header("🎨 Display Options")
    
    window_hours = st.sidebar.slider(
        "Time Window (hours from event)",
        min_value=1,
        max_value=72,
        value=72,
        step=1
    )
    
    log_scale = st.sidebar.checkbox("Log scale for trade size", value=False)
    
    # Outlier filtering
    st.sidebar.markdown("**Outlier Filtering**")
    filter_outliers = st.sidebar.checkbox("Remove outliers", value=True)
    
    if filter_outliers:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lower_pct = st.number_input("Lower %", min_value=0, max_value=50, value=1, step=1,
                                       help="Remove values below this percentile")
        with col2:
            upper_pct = st.number_input("Upper %", min_value=50, max_value=100, value=99, step=1,
                                       help="Remove values above this percentile")
    else:
        lower_pct, upper_pct = 0, 100
    
    # Filter to selected event
    event_df = ticker_df[ticker_df['event_date'] == selected_event_date].copy()
    
    if len(event_df) == 0:
        st.error(f"No data found for {selected_ticker} - {selected_event_date}")
        return
    
    # Apply time window filter
    event_df = event_df[
        event_df['minutes_from_event'].between(-window_hours*60, window_hours*60)
    ]
    
    # Apply outlier filter if enabled
    if filter_outliers:
        lower_bound = event_df['avg_trade_size'].quantile(lower_pct / 100)
        upper_bound = event_df['avg_trade_size'].quantile(upper_pct / 100)
        filtered_df = event_df[
            (event_df['avg_trade_size'] >= lower_bound) & 
            (event_df['avg_trade_size'] <= upper_bound)
        ]
        
        # Check if filter removed all data
        if len(filtered_df) == 0:
            st.warning("⚠️ Outlier filter removed all data. Showing unfiltered data instead.")
            st.session_state['filter_applied'] = False
        else:
            event_df = filtered_df
            # Store bounds for display
            st.session_state['lower_bound'] = lower_bound
            st.session_state['upper_bound'] = upper_bound
            st.session_state['filter_applied'] = True
    else:
        st.session_state['filter_applied'] = False
    
    # Final check for empty dataframe
    if len(event_df) == 0:
        st.error(f"No data available for {selected_ticker} - {selected_event_date} in the selected time window")
        return
    
    # Show event metadata
    event_meta = events.iloc[selected_event_idx]
    
    # Outlier filter banner
    if st.session_state.get('filter_applied', False):
        lower = st.session_state.get('lower_bound', 0)
        upper = st.session_state.get('upper_bound', 0)
        st.info(f"🔍 Outlier filter active: showing trades between ${lower:,.0f} and ${upper:,.0f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", selected_ticker)
    
    with col2:
        st.metric("Event Date", selected_event_date)
    
    with col3:
        if pd.notna(event_meta['reported_eps']):
            delta = None
            if pd.notna(event_meta['eps_estimate']):
                delta = f"{event_meta['reported_eps'] - event_meta['eps_estimate']:.2f}"
            st.metric("Reported EPS", f"${event_meta['reported_eps']:.2f}", delta=delta)
        else:
            st.metric("Reported EPS", "N/A")
    
    with col4:
        if pd.notna(event_meta['surprise_pct']):
            st.metric("Surprise", f"{event_meta['surprise_pct']:+.1f}%")
        else:
            st.metric("Surprise", "N/A")
    
    st.markdown("---")
    
    # Create visualizations
    event_datetime = event_df['event_datetime'].iloc[0]
    
    st.subheader("1. Trade Size Distribution")
    fig1 = create_trade_size_chart(event_df, bucket_col, event_datetime, log_scale)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("2. Price Action & Round Lot Trades (Aggregated)")
    fig2 = create_candlestick_chart(event_df, bucket_col, event_datetime)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Post-Earnings Analysis (Second-Bar Resolution)")
    st.markdown("---")
    
    st.subheader("3. Combined Price Action & Round Lot Flow")
    st.caption("Top: Second-bar candlesticks with round lot overlay | Bottom: Flow metrics (Sign CVD & Impact CVD) | Synchronized zoom")
    fig3 = create_second_bar_analysis_chart(event_df, event_datetime)
    st.plotly_chart(fig3, use_container_width=True)

    
    # Summary statistics (on raw second bars)
    st.markdown("---")
    st.subheader("📋 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Overall**")
        st.write(f"Mean: ${event_df['avg_trade_size'].mean():,.0f}")
        st.write(f"Median: ${event_df['avg_trade_size'].median():,.0f}")
        st.write(f"Std Dev: ${event_df['avg_trade_size'].std():,.0f}")
    
    with col2:
        st.markdown("**Range**")
        st.write(f"Min: ${event_df['avg_trade_size'].min():,.0f}")
        st.write(f"Max: ${event_df['avg_trade_size'].max():,.0f}")
        st.write(f"P95: ${event_df['avg_trade_size'].quantile(0.95):,.0f}")
    
    with col3:
        st.markdown("**Data**")
        bucket_count = event_df[bucket_col].nunique()
        st.write(f"Unique time buckets: {bucket_count}")
        st.write(f"Second bars: {len(event_df):,}")
        st.write(f"Avg bars per bucket: {len(event_df) / bucket_count:.1f}")
    
    # Trading hours breakdown
    st.markdown("---")
    st.subheader("⏰ Trading Hours Breakdown")
    
    hours_stats = event_df.groupby('trading_hours')['avg_trade_size'].agg(['mean', 'median', 'count'])
    hours_stats = hours_stats.reindex(['Pre', 'Regular', 'Post', 'Overnight'])
    hours_stats = hours_stats.dropna()
    
    hours_stats.columns = ['Mean Trade Size', 'Median Trade Size', 'Num Bars']
    hours_stats['Mean Trade Size'] = hours_stats['Mean Trade Size'].apply(lambda x: f"${x:,.0f}")
    hours_stats['Median Trade Size'] = hours_stats['Median Trade Size'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(hours_stats, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Data window: {DAYS_BEFORE} days before to {DAYS_AFTER} day after earnings event")


# Import numpy after streamlit to avoid issues
import numpy as np

# Constants from data preparation
DAYS_BEFORE = 3
DAYS_AFTER = 1


if __name__ == '__main__':
    main()
