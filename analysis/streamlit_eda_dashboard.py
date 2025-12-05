"""EDA Dashboard for Earnings Consolidated Data.

Focus on correlation analysis between features and forward return targets
at different granularities (overall, by ticker, by year).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from eda_utils import (
    load_earnings_data,
    get_column_groups,
    get_target_columns,
    get_feature_columns,
    calculate_correlation_with_targets,
    get_top_correlations,
    format_large_numbers
)

# Page config
st.set_page_config(
    page_title="Earnings Data EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data path - Extended post-event dataset (0-120 min)
DATA_PATH = "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv"


@st.cache_data
def load_data():
    """Load and cache the data."""
    return load_earnings_data(DATA_PATH)


def main():
    st.markdown('<div class="main-header">📊 Earnings Data EDA</div>', unsafe_allow_html=True)
    st.markdown("**Explore correlations between features and forward return targets**")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar - Filters
    st.sidebar.header("🎛️ Filters")
    
    # Ticker filter
    all_tickers = sorted(df['ticker'].unique())
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=all_tickers,
        default=all_tickers,
        help="Filter data by ticker symbols"
    )
    
    # Year filter
    all_years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=all_years,
        default=all_years,
        help="Filter data by year"
    )
    
    # Time window filter
    time_window_options = {
        'All Data': None,
        'Event Window (±30m)': 'event_plus_minus_30m',
        '5 Days Before': '5_days_before',
        '5 Days After': '5_days_after',
        'Event +30-120m': 'event_plus_30_to_120m',
        'Next Open ±30m': 'next_open_plus_minus_30m'
    }
    selected_window = st.sidebar.selectbox(
        "Time Window",
        options=list(time_window_options.keys()),
        help="Filter to specific time windows"
    )
    
    # Trading session filter
    session_options = ['All', 'Regular', 'PreMarket', 'PostMarket']
    selected_session = st.sidebar.selectbox(
        "Trading Session",
        options=session_options,
        help="Filter by trading session"
    )
    
    # Apply filters
    filtered_df = df[
        (df['ticker'].isin(selected_tickers)) &
        (df['year'].isin(selected_years))
    ].copy()
    
    if time_window_options[selected_window] is not None:
        filtered_df = filtered_df[filtered_df[time_window_options[selected_window]] == True]
    
    if selected_session != 'All':
        filtered_df = filtered_df[filtered_df['trading_session'] == selected_session]
    
    # Display summary stats
    st.markdown('<div class="sub-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", format_large_numbers(len(filtered_df)))
    with col2:
        st.metric("Tickers", filtered_df['ticker'].nunique())
    with col3:
        st.metric("Events", filtered_df.groupby(['ticker', 'acceptance_datetime_utc']).ngroups)
    with col4:
        st.metric("Columns", df.shape[1])
    with col5:
        date_range = f"{filtered_df['acceptance_datetime_utc'].min().date()} to {filtered_df['acceptance_datetime_utc'].max().date()}"
        st.metric("Date Range", "")
        st.caption(date_range)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overall Correlations",
        "🎯 By Ticker",
        "📅 By Year",
        "⏱️ Time Evolution"
    ])
    
    # Get features and targets
    targets = get_target_columns()
    features = get_feature_columns(df, exclude_targets=True)
    
    # Sidebar - Feature selection
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Feature Selection")
    
    column_groups = get_column_groups()
    
    # Only show analysis-ready feature groups (exclude those marked as "Excluded")
    feature_group_options = [
        k for k in column_groups.keys() 
        if k not in ['Forward Returns (Targets)', 'Metadata'] 
        and 'Excluded' not in k
    ]
    
    selected_groups = st.sidebar.multiselect(
        "Feature Groups (Analysis-Ready)",
        options=feature_group_options,
        default=feature_group_options[:3],  # Default to first 3 groups
        help="Only includes normalized features suitable for correlation analysis (excludes raw OHLCV, raw prices, etc.)"
    )
    
    # Get features from selected groups
    selected_features = []
    for group in selected_groups:
        group_cols = column_groups[group]
        # Only include columns that exist in features list
        selected_features.extend([col for col in group_cols if col in features])
    
    if not selected_features:
        st.warning("⚠️ No features selected. Please select at least one feature group.")
        return
    
    # Correlation method
    corr_method = st.sidebar.selectbox(
        "Correlation Method",
        options=['pearson', 'spearman', 'kendall'],
        help="Pearson: linear, Spearman: monotonic, Kendall: rank-based"
    )
    
    # Minimum samples
    min_samples = st.sidebar.number_input(
        "Min Samples for Correlation",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Minimum number of valid samples required to calculate correlation"
    )
    
    # Tab 1: Overall Correlations
    with tab1:
        st.markdown('<div class="sub-header">Overall Feature-Target Correlations</div>', unsafe_allow_html=True)
        st.markdown(f"Analyzing **{len(selected_features)} features** vs **{len(targets)} targets** using **{corr_method}** correlation")
        
        with st.spinner("Calculating correlations..."):
            overall_corr = calculate_correlation_with_targets(
                filtered_df,
                selected_features,
                targets,
                method=corr_method,
                min_samples=min_samples
            )
        
        # Display heatmap
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Correlation Heatmap**")
            fig = px.imshow(
                overall_corr,
                labels=dict(x="Target", y="Feature", color="Correlation"),
                x=targets,
                y=selected_features,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(
                height=max(400, len(selected_features) * 15),
                xaxis_title="Forward Return Targets",
                yaxis_title="Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 20 Correlations**")
            top_corr = get_top_correlations(overall_corr, n=20, abs_value=True)
            
            if len(top_corr) > 0:
                # Color code the correlation values
                def color_correlation(val):
                    if pd.isna(val):
                        return ''
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                
                styled_df = top_corr.style.applymap(
                    color_correlation,
                    subset=['correlation']
                ).format({'correlation': '{:.4f}'})
                
                st.dataframe(styled_df, height=600, use_container_width=True)
            else:
                st.info("No correlations found with minimum sample requirement")
        
        # Download correlations
        st.markdown("---")
        csv = overall_corr.to_csv()
        st.download_button(
            label="📥 Download Correlation Matrix (CSV)",
            data=csv,
            file_name="overall_correlations.csv",
            mime="text/csv"
        )
    
    # Tab 2: By Ticker
    with tab2:
        st.markdown('<div class="sub-header">Correlations by Ticker</div>', unsafe_allow_html=True)
        
        # Select ticker to analyze
        ticker_to_analyze = st.selectbox(
            "Select Ticker for Detailed Analysis",
            options=selected_tickers,
            help="View correlations specific to this ticker"
        )
        
        ticker_df = filtered_df[filtered_df['ticker'] == ticker_to_analyze]
        
        st.info(f"**{ticker_to_analyze}**: {len(ticker_df):,} rows | "
                f"{ticker_df.groupby('acceptance_datetime_utc').ngroups} events")
        
        with st.spinner(f"Calculating correlations for {ticker_to_analyze}..."):
            ticker_corr = calculate_correlation_with_targets(
                ticker_df,
                selected_features,
                targets,
                method=corr_method,
                min_samples=min_samples
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{ticker_to_analyze} Correlation Heatmap**")
            fig = px.imshow(
                ticker_corr,
                labels=dict(x="Target", y="Feature", color="Correlation"),
                x=targets,
                y=selected_features,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(
                height=max(400, len(selected_features) * 15),
                xaxis_title="Forward Return Targets",
                yaxis_title="Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 20 Correlations**")
            top_ticker_corr = get_top_correlations(ticker_corr, n=20, abs_value=True)
            
            if len(top_ticker_corr) > 0:
                styled_df = top_ticker_corr.style.applymap(
                    lambda val: f'color: {"green" if val > 0 else "red"}' if pd.notna(val) else '',
                    subset=['correlation']
                ).format({'correlation': '{:.4f}'})
                
                st.dataframe(styled_df, height=600, use_container_width=True)
            else:
                st.info("No correlations found with minimum sample requirement")
        
        # Compare with overall
        st.markdown("---")
        st.markdown("**Compare Ticker vs Overall Correlations**")
        
        comparison_target = st.selectbox(
            "Select target to compare",
            options=targets,
            format_func=lambda x: x.replace('target_ret_', '').replace('s', ' sec')
        )
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Feature': selected_features,
            f'{ticker_to_analyze}': ticker_corr[comparison_target],
            'Overall': overall_corr[comparison_target]
        })
        comparison_df['Difference'] = comparison_df[f'{ticker_to_analyze}'] - comparison_df['Overall']
        comparison_df = comparison_df.dropna().sort_values('Difference', key=abs, ascending=False)
        
        # Plot comparison
        if len(comparison_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison_df['Feature'][:20],
                y=comparison_df[f'{ticker_to_analyze}'][:20],
                name=ticker_to_analyze,
                marker_color='steelblue'
            ))
            fig.add_trace(go.Bar(
                x=comparison_df['Feature'][:20],
                y=comparison_df['Overall'][:20],
                name='Overall',
                marker_color='lightcoral'
            ))
            fig.update_layout(
                title=f"Top 20 Features: {ticker_to_analyze} vs Overall ({comparison_target})",
                xaxis_title="Feature",
                yaxis_title="Correlation",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: By Year
    with tab3:
        st.markdown('<div class="sub-header">Correlations by Year</div>', unsafe_allow_html=True)
        
        # Year selector
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_year = st.selectbox(
                "Select Year",
                options=sorted(selected_years),
                help="View correlations for a specific year"
            )
        
        # Filter data for selected year
        year_df = filtered_df[filtered_df['year'] == selected_year]
        
        st.info(f"**{selected_year}**: {len(year_df):,} rows | "
                f"{year_df['ticker'].nunique()} tickers | "
                f"{year_df.groupby(['ticker', 'acceptance_datetime_utc']).ngroups} events")
        
        if len(year_df) < min_samples:
            st.warning(f"⚠️ Not enough data for {selected_year} (only {len(year_df)} rows, need {min_samples})")
        else:
            with st.spinner(f"Calculating correlations for {selected_year}..."):
                year_corr = calculate_correlation_with_targets(
                    year_df,
                    selected_features,
                    targets,
                    method=corr_method,
                    min_samples=min_samples
                )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{selected_year} Correlation Heatmap**")
                fig = px.imshow(
                    year_corr,
                    labels=dict(x="Target", y="Feature", color="Correlation"),
                    x=targets,
                    y=selected_features,
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    aspect="auto",
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(
                    height=max(400, len(selected_features) * 15),
                    xaxis_title="Forward Return Targets",
                    yaxis_title="Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Top 20 Correlations**")
                top_year_corr = get_top_correlations(year_corr, n=20, abs_value=True)
                
                if len(top_year_corr) > 0:
                    styled_df = top_year_corr.style.applymap(
                        lambda val: f'color: {"green" if val > 0 else "red"}' if pd.notna(val) else '',
                        subset=['correlation']
                    ).format({'correlation': '{:.4f}'})
                    
                    st.dataframe(styled_df, height=600, use_container_width=True)
                else:
                    st.info("No correlations found with minimum sample requirement")
            
            # Compare with overall
            st.markdown("---")
            st.markdown(f"**Compare {selected_year} vs Overall Correlations**")
            
            comparison_target = st.selectbox(
                "Select target to compare",
                options=targets,
                format_func=lambda x: x.replace('target_ret_', '').replace('s', ' sec'),
                key='year_comparison_target'
            )
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Feature': selected_features,
                f'{selected_year}': year_corr[comparison_target],
                'Overall': overall_corr[comparison_target]
            })
            comparison_df['Difference'] = comparison_df[f'{selected_year}'] - comparison_df['Overall']
            comparison_df = comparison_df.dropna().sort_values('Difference', key=abs, ascending=False)
            
            # Plot comparison
            if len(comparison_df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison_df['Feature'][:20],
                    y=comparison_df[f'{selected_year}'][:20],
                    name=str(selected_year),
                    marker_color='steelblue'
                ))
                fig.add_trace(go.Bar(
                    x=comparison_df['Feature'][:20],
                    y=comparison_df['Overall'][:20],
                    name='Overall',
                    marker_color='lightcoral'
                ))
                fig.update_layout(
                    title=f"Top 20 Features: {selected_year} vs Overall ({comparison_target})",
                    xaxis_title="Feature",
                    yaxis_title="Correlation",
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show biggest differences
                st.markdown("**Biggest Differences from Overall**")
                biggest_diff = comparison_df.nlargest(10, 'Difference', keep='all')[['Feature', f'{selected_year}', 'Overall', 'Difference']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"*Stronger in {selected_year}*")
                    st.dataframe(
                        biggest_diff.head(5).style.format({
                            f'{selected_year}': '{:.4f}',
                            'Overall': '{:.4f}',
                            'Difference': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown(f"*Weaker in {selected_year}*")
                    weaker = comparison_df.nsmallest(5, 'Difference', keep='all')[['Feature', f'{selected_year}', 'Overall', 'Difference']]
                    st.dataframe(
                        weaker.style.format({
                            f'{selected_year}': '{:.4f}',
                            'Overall': '{:.4f}',
                            'Difference': '{:.4f}'
                        }),
                        use_container_width=True
                    )

    # Tab 4: Time Evolution
    with tab4:
        st.markdown('<div class="sub-header">⏱️ Feature Correlations Over Time</div>', unsafe_allow_html=True)
        st.markdown("Analyze how feature correlations evolve through 5-minute time buckets since earnings event")
        
        # Check if we have the right columns
        if 'seconds_since_event' not in filtered_df.columns:
            st.error("This dataset doesn't have 'seconds_since_event' column. Please use the extended dataset.")
        else:
            # Controls
            te_col1, te_col2, te_col3, te_col4 = st.columns(4)
            
            with te_col1:
                te_year = st.selectbox(
                    "Year",
                    options=sorted(selected_years),
                    key="te_year"
                )
            
            with te_col2:
                te_target = st.selectbox(
                    "Target Return",
                    options=targets,
                    index=len(targets) - 1 if targets else 0,  # Default to longest horizon
                    format_func=lambda x: x.replace('target_ret_', '').replace('s', ' sec'),
                    key="te_target"
                )
            
            with te_col3:
                te_top_n = st.slider(
                    "Top N Features per Bucket",
                    min_value=3, max_value=15, value=6,
                    key="te_top_n"
                )
            
            with te_col4:
                te_min_corr = st.slider(
                    "Min |Correlation| Threshold",
                    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                    key="te_min_corr"
                )
            
            # Filter to selected year
            te_df = filtered_df[filtered_df['year'] == te_year].copy()
            
            if len(te_df) == 0:
                st.warning(f"No data for {te_year}")
            else:
                # Create 5-minute buckets (0-120 min = 24 buckets)
                bucket_labels = [f"{i}-{i+5}m" for i in range(0, 120, 5)]
                te_df['time_bucket'] = pd.cut(
                    te_df['seconds_since_event'],
                    bins=list(range(0, 7260, 300)),  # 0, 300, 600, ..., 7200
                    labels=bucket_labels,
                    include_lowest=True
                )
                
                # Calculate correlations for each bucket
                bucket_corrs = {}
                bucket_rl_stats = {}
                
                for bucket in bucket_labels:
                    bucket_data = te_df[te_df['time_bucket'] == bucket]
                    if len(bucket_data) < min_samples:
                        continue
                    
                    # Calculate correlations for all features
                    corrs = {}
                    for feat in selected_features:
                        if feat in bucket_data.columns and te_target in bucket_data.columns:
                            valid_data = bucket_data[[feat, te_target]].dropna()
                            if len(valid_data) >= min_samples:
                                corr = valid_data.corr(method=corr_method).iloc[0, 1]
                                if not np.isnan(corr):
                                    corrs[feat] = corr
                    
                    if corrs:
                        bucket_corrs[bucket] = corrs
                    
                    # Calculate Round Lot stats for this bucket
                    if 'rl_volume' in bucket_data.columns and 'volume' in bucket_data.columns:
                        rl_vol_sum = bucket_data['rl_volume'].sum()
                        total_vol = bucket_data['volume'].sum()
                        rl_vol_pct = (rl_vol_sum / total_vol * 100) if total_vol > 0 else 0
                    else:
                        rl_vol_pct = np.nan
                    
                    if 'rl_buy_volume' in bucket_data.columns and 'rl_volume' in bucket_data.columns:
                        rl_buy_sum = bucket_data['rl_buy_volume'].sum()
                        rl_vol_sum = bucket_data['rl_volume'].sum()
                        rl_buy_pct = (rl_buy_sum / rl_vol_sum * 100) if rl_vol_sum > 0 else 50
                        rl_net_dir = rl_buy_pct - 50  # Centered around 0
                    else:
                        rl_net_dir = np.nan
                    
                    bucket_rl_stats[bucket] = {'rl_vol_pct': rl_vol_pct, 'rl_net_dir': rl_net_dir}
                
                if not bucket_corrs:
                    st.warning("Not enough data in any time bucket for correlation analysis.")
                else:
                    # Collect top features from each bucket (union)
                    all_top_features = set()
                    for bucket, corrs in bucket_corrs.items():
                        # Filter by threshold
                        filtered_corrs = {k: v for k, v in corrs.items() if abs(v) >= te_min_corr}
                        # Get top N
                        sorted_corrs = sorted(filtered_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:te_top_n]
                        all_top_features.update([f for f, _ in sorted_corrs])
                    
                    if not all_top_features:
                        st.warning(f"No features meet the |correlation| >= {te_min_corr} threshold.")
                    else:
                        all_top_features = sorted(list(all_top_features))
                        
                        # Build heatmap data
                        heatmap_data = []
                        for feat in all_top_features:
                            row = {'Feature': feat}
                            for bucket in bucket_labels:
                                if bucket in bucket_corrs and feat in bucket_corrs[bucket]:
                                    row[bucket] = bucket_corrs[bucket][feat]
                                else:
                                    row[bucket] = np.nan
                            heatmap_data.append(row)
                        
                        heatmap_df = pd.DataFrame(heatmap_data)
                        heatmap_df = heatmap_df.set_index('Feature')
                        
                        # Heatmap
                        st.markdown("### Correlation Heatmap by Time Bucket")
                        
                        fig_heatmap = px.imshow(
                            heatmap_df,
                            labels=dict(x="Time Bucket", y="Feature", color="Correlation"),
                            color_continuous_scale="RdBu_r",
                            color_continuous_midpoint=0,
                            aspect="auto",
                            zmin=-0.5,
                            zmax=0.5
                        )
                        fig_heatmap.update_layout(
                            height=max(400, len(all_top_features) * 25),
                            xaxis_title="Time Since Event",
                            yaxis_title="Feature",
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True, key="time_evolution_heatmap")
                        
                        # Key insights
                        st.markdown("### Key Insights")
                        
                        # Find strongest features in early vs late periods
                        early_buckets = [b for b in bucket_labels[:3] if b in bucket_corrs]  # 0-15 min
                        late_buckets = [b for b in bucket_labels[12:] if b in bucket_corrs]  # 60-120 min
                        
                        col_i1, col_i2 = st.columns(2)
                        
                        with col_i1:
                            st.markdown("**Early Period (0-15 min)**")
                            if early_buckets:
                                early_avg = {}
                                for feat in all_top_features:
                                    vals = [bucket_corrs[b].get(feat, np.nan) for b in early_buckets]
                                    vals = [v for v in vals if not np.isnan(v)]
                                    if vals:
                                        early_avg[feat] = np.mean([abs(v) for v in vals])
                                
                                if early_avg:
                                    sorted_early = sorted(early_avg.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for feat, avg in sorted_early:
                                        st.write(f"• {feat}: |corr| = {avg:.3f}")
                            else:
                                st.write("No data for early period")
                        
                        with col_i2:
                            st.markdown("**Late Period (60-120 min)**")
                            if late_buckets:
                                late_avg = {}
                                for feat in all_top_features:
                                    vals = [bucket_corrs[b].get(feat, np.nan) for b in late_buckets]
                                    vals = [v for v in vals if not np.isnan(v)]
                                    if vals:
                                        late_avg[feat] = np.mean([abs(v) for v in vals])
                                
                                if late_avg:
                                    sorted_late = sorted(late_avg.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for feat, avg in sorted_late:
                                        st.write(f"• {feat}: |corr| = {avg:.3f}")
                            else:
                                st.write("No data for late period")


if __name__ == "__main__":
    main()
