"""
Logistic Regression Viewer
Visualize logistic regression model results and explore entry signals.
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
    page_title="LR Entry Signal Explorer",
    page_icon="🎲",
    layout="wide"
)

# Default data path
DEFAULT_DATA_PATH = "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "logistic_regression_artifacts")


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


def load_model_artifacts():
    """Load logistic regression artifacts."""
    required_files = [
        "model_summary.csv",
        "confusion_metrics.json",
        "roc_curve.csv", 
        "probability_distribution.csv",
        "metadata.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))]
    
    if missing_files:
        return None, missing_files
    
    try:
        artifacts = {
            'model_summary': pd.read_csv(os.path.join(ARTIFACTS_DIR, "model_summary.csv")),
            'roc_data': pd.read_csv(os.path.join(ARTIFACTS_DIR, "roc_curve.csv")),
            'prob_dist': pd.read_csv(os.path.join(ARTIFACTS_DIR, "probability_distribution.csv")),
        }
        
        with open(os.path.join(ARTIFACTS_DIR, "confusion_metrics.json"), 'r') as f:
            artifacts['confusion_data'] = json.load(f)
        
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), 'r') as f:
            artifacts['metadata'] = json.load(f)
        
        return artifacts, []
    except Exception as e:
        return None, [str(e)]


def compute_predicted_probability(df, model_summary):
    """Compute predicted probability using logistic function."""
    intercept = model_summary[model_summary['Variable'] == 'Intercept']['Coefficient'].values[0]
    cvd_coef = model_summary[model_summary['Variable'] == 'CVD Z-Score']['Coefficient'].values[0]
    vwap_coef = model_summary[model_summary['Variable'] == 'VWAP Distance %']['Coefficient'].values[0]
    
    if 'vwap_distance_pct' in df.columns:
        logit = intercept + cvd_coef * df['cvd_zscore'] + vwap_coef * df['vwap_distance_pct']
    else:
        logit = intercept + cvd_coef * df['cvd_zscore']
    
    return 1 / (1 + np.exp(-logit))


def main():
    st.title("🎲 Logistic Regression Entry Signal Explorer")
    st.markdown("Explore model predictions and find entry signals combining CVD Z-Score and VWAP Distance")
    
    # Load artifacts
    artifacts, missing = load_model_artifacts()
    
    if artifacts is None:
        st.error(f"""
        **Model artifacts not found.** Please run the training script first:
        
        ```bash
        ./venv/bin/python3.13 analysis/fit_logistic_regression.py
        ```
        
        Missing: {', '.join(missing)}
        """)
        return
    
    # ========== SIDEBAR ==========
    st.sidebar.header("📂 Data Source")
    data_path = st.sidebar.text_input(
        "Data File Path",
        value=DEFAULT_DATA_PATH,
        key="data_path"
    )
    
    # Load data
    df = load_data(data_path)
    if df.empty:
        st.error("No data available.")
        return
    
    # Year filter
    st.sidebar.header("🔧 Filters")
    available_years = sorted(df['year'].dropna().unique().astype(int))
    selected_years = st.sidebar.multiselect(
        "Years",
        options=available_years,
        default=[2024, 2025] if 2024 in available_years else available_years[-2:],
        key="year_filter"
    )
    
    # Filter data
    filtered_df = df[df['year'].isin(selected_years)].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Rows:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Tickers:** {filtered_df['ticker'].nunique()}")
    
    # ========== TABS ==========
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Summary",
        "🎯 Entry Signal Explorer",
        "📈 Signal Statistics"
    ])
    
    # ========== TAB 1: Model Summary ==========
    with tab1:
        st.markdown("### Model Performance")
        
        model_summary = artifacts['model_summary']
        confusion_data = artifacts['confusion_data']
        roc_data = artifacts['roc_data']
        prob_dist = artifacts['prob_dist']
        metadata = artifacts['metadata']
        
        # Model coefficients
        st.markdown("#### Coefficients")
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
            st.metric("CVD Z-Score Odds Ratio", f"{cvd_or:.4f}", delta=f"p={cvd_pval:.4f} {sig_cvd}")
            st.caption("Higher CVD → Lower odds of positive return" if cvd_or < 1 else "Higher CVD → Higher odds of positive return")
        
        with interp_cols[1]:
            vwap_or = vwap_row['Odds_Ratio']
            vwap_pval = vwap_row['P_Value']
            sig_vwap = "✅ Significant" if vwap_pval < 0.05 else "❌ Not significant"
            st.metric("VWAP Distance % Odds Ratio", f"{vwap_or:.4f}", delta=f"p={vwap_pval:.4f} {sig_vwap}")
            st.caption("Further from VWAP → Higher odds of positive return" if vwap_or > 1 else "Further from VWAP → Lower odds of positive return")
        
        # Metrics row
        st.markdown("---")
        st.markdown("#### Classification Metrics")
        
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("AUC", f"{metadata['roc_auc']:.3f}")
        with metric_cols[1]:
            st.metric("Accuracy", f"{confusion_data['accuracy']:.3f}")
        with metric_cols[2]:
            st.metric("Precision", f"{confusion_data['precision']:.3f}")
        with metric_cols[3]:
            st.metric("Recall", f"{confusion_data['recall']:.3f}")
        with metric_cols[4]:
            st.metric("F1 Score", f"{confusion_data['f1_score']:.3f}")
        
        # ROC and Confusion Matrix side by side
        st.markdown("---")
        roc_col, cm_col = st.columns(2)
        
        with roc_col:
            st.markdown("#### ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'], mode='lines',
                                         name=f'ROC (AUC = {metadata["roc_auc"]:.3f})',
                                         line=dict(color='#3b82f6', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                         line=dict(color='gray', dash='dash')))
            fig_roc.update_layout(height=350, template='plotly_white', 
                                  xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True, key="roc")
        
        with cm_col:
            st.markdown("#### Confusion Matrix")
            cm = np.array(confusion_data['confusion_matrix'])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Pred Neg', 'Pred Pos'], y=['Actual Neg', 'Actual Pos'],
                text=cm, texttemplate='%{text}', textfont={'size': 18},
                colorscale='Blues', showscale=False
            ))
            fig_cm.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig_cm, use_container_width=True, key="cm")
        
        # Probability distribution
        st.markdown("---")
        st.markdown("#### Probability Distribution by Outcome")
        
        prob_dist['Actual Outcome'] = prob_dist['actual'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            fig_box = go.Figure()
            for outcome, color in [('Negative', '#ef4444'), ('Positive', '#22c55e')]:
                outcome_data = prob_dist[prob_dist['Actual Outcome'] == outcome]['predicted_prob']
                fig_box.add_trace(go.Box(y=outcome_data, name=outcome, marker_color=color))
            fig_box.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig_box.update_layout(title='P(Positive) by Actual Outcome', height=350, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, key="box")
        
        with dist_col2:
            # Calibration
            n_bins = 10
            prob_dist['prob_bin'] = pd.cut(prob_dist['predicted_prob'], bins=n_bins, labels=False)
            cal_data = prob_dist.groupby('prob_bin').agg(
                bin_center=('predicted_prob', 'mean'),
                actual_rate=('actual', 'mean')
            ).reset_index()
            
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect',
                                         line=dict(color='gray', dash='dash')))
            fig_cal.add_trace(go.Bar(x=cal_data['bin_center'], y=cal_data['actual_rate'],
                                     name='Actual', marker_color='#3b82f6', width=0.08))
            fig_cal.update_layout(title='Calibration Plot', height=350, template='plotly_white',
                                  xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_cal, use_container_width=True, key="cal")
        
        # Data summary
        st.markdown("---")
        st.markdown("#### Training Data Summary")
        data_cols = st.columns(4)
        with data_cols[0]:
            st.metric("Observations", f"{metadata['total_observations']:,}")
        with data_cols[1]:
            st.metric("Positive", f"{metadata['positive_returns']:,} ({metadata['positive_pct']:.1f}%)")
        with data_cols[2]:
            st.metric("Negative", f"{metadata['negative_returns']:,}")
        with data_cols[3]:
            st.metric("Events", f"{metadata['n_events']}")
    
    # ========== TAB 2: Entry Signal Explorer ==========
    with tab2:
        st.markdown("### Entry Signal Explorer")
        st.caption("Visualize price, CVD Z-Score, and predicted probability for individual events.")
        
        # Event selector
        col1, col2 = st.columns(2)
        with col1:
            available_tickers = sorted(filtered_df['ticker'].unique())
            selected_ticker = st.selectbox("Select Ticker", options=available_tickers, key="ticker")
        
        if selected_ticker:
            ticker_events = filtered_df[filtered_df['ticker'] == selected_ticker]['acceptance_datetime_utc'].unique()
            ticker_events = sorted(ticker_events)
            
            with col2:
                selected_event = st.selectbox(
                    "Select Event",
                    options=ticker_events,
                    format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'),
                    key="event"
                )
            
            if selected_event:
                # Time range
                time_col1, time_col2 = st.columns(2)
                with time_col1:
                    time_min = st.number_input("From (seconds)", min_value=0, max_value=7200, value=0, step=60)
                with time_col2:
                    time_max = st.number_input("To (seconds)", min_value=0, max_value=7200, value=1800, step=60)
                
                # Get event data
                event_df = filtered_df[
                    (filtered_df['ticker'] == selected_ticker) &
                    (filtered_df['acceptance_datetime_utc'] == selected_event) &
                    (filtered_df['seconds_since_event'] >= time_min) &
                    (filtered_df['seconds_since_event'] <= time_max)
                ].sort_values('seconds_since_event').copy()
                
                if event_df.empty:
                    st.warning("No data for selected event/time range.")
                else:
                    # Compute predicted probability
                    event_df['pred_prob'] = compute_predicted_probability(event_df, artifacts['model_summary'])
                    
                    # Calculate price return
                    event_price = event_df['event_price'].iloc[0] if 'event_price' in event_df.columns else np.nan
                    if pd.notna(event_price) and event_price > 0:
                        event_df['price_return_pct'] = ((event_df['close'] - event_price) / event_price) * 100
                    
                    # Create stacked subplot
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.35, 0.35, 0.30],
                        subplot_titles=('Price', 'CVD Z-Score', 'Predicted P(Positive)')
                    )
                    
                    # Row 1: Price
                    price_col = 'vw' if 'vw' in event_df.columns else 'close'
                    fig.add_trace(
                        go.Scatter(x=event_df['seconds_since_event'], y=event_df[price_col],
                                   mode='lines', name='Price', line=dict(color='#8b5cf6', width=2)),
                        row=1, col=1
                    )
                    
                    # Row 2: CVD Z-Score with background colors
                    for i in range(len(event_df) - 1):
                        p = event_df['pred_prob'].iloc[i]
                        if pd.isna(p):
                            color = 'rgba(128,128,128,0.1)'
                        elif p >= 0.6:
                            color = 'rgba(34,197,94,0.3)'
                        elif p >= 0.5:
                            color = 'rgba(34,197,94,0.15)'
                        elif p >= 0.4:
                            color = 'rgba(239,68,68,0.15)'
                        else:
                            color = 'rgba(239,68,68,0.3)'
                        
                        fig.add_vrect(
                            x0=event_df['seconds_since_event'].iloc[i],
                            x1=event_df['seconds_since_event'].iloc[i+1],
                            fillcolor=color, layer='below', line_width=0,
                            row=2, col=1
                        )
                    
                    fig.add_trace(
                        go.Scatter(x=event_df['seconds_since_event'], y=event_df['cvd_zscore'],
                                   mode='lines', name='CVD Z-Score', line=dict(color='#3b82f6', width=2)),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
                    
                    # Row 3: Predicted Probability
                    fig.add_trace(
                        go.Scatter(x=event_df['seconds_since_event'], y=event_df['pred_prob'],
                                   mode='lines', name='P(Positive)', line=dict(color='#f97316', width=2),
                                   fill='tozeroy', fillcolor='rgba(249,115,22,0.2)'),
                        row=3, col=1
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7, row=3, col=1)
                    fig.add_hline(y=0.6, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)
                    fig.add_hline(y=0.4, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)
                    
                    fig.update_layout(
                        height=700, template='plotly_white', showlegend=False,
                        title_text=f"{selected_ticker} - Entry Signal Analysis"
                    )
                    fig.update_xaxes(title_text="Seconds Since Event", row=3, col=1)
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="CVD Z-Score", row=2, col=1)
                    fig.update_yaxes(title_text="P(Positive)", range=[0, 1], row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True, key="entry_chart")
                    
                    # Legend
                    st.caption("""
                    **Background Colors:** 🟢 Dark=P≥0.6 | 🟢 Light=P≥0.5 | 🔴 Light=P<0.5 | 🔴 Dark=P<0.4
                    """)
                    
                    # Signal summary
                    st.markdown("**Signal Summary:**")
                    sig_cols = st.columns(4)
                    valid_probs = event_df['pred_prob'].dropna()
                    with sig_cols[0]:
                        st.metric("Mean P", f"{valid_probs.mean():.3f}")
                    with sig_cols[1]:
                        st.metric("Max P", f"{valid_probs.max():.3f}")
                    with sig_cols[2]:
                        st.metric("% P≥0.6", f"{(valid_probs >= 0.6).mean()*100:.1f}%")
                    with sig_cols[3]:
                        st.metric("% P<0.4", f"{(valid_probs < 0.4).mean()*100:.1f}%")
    
    # ========== TAB 3: Signal Statistics ==========
    with tab3:
        st.markdown("### Signal Statistics Across Events")
        st.caption("Aggregate statistics on predicted probabilities across multiple events.")
        
        # Filter to 5m-30m window
        stats_df = filtered_df[
            (filtered_df['seconds_since_event'] >= 300) &
            (filtered_df['seconds_since_event'] <= 1800)
        ].copy()
        
        if stats_df.empty:
            st.warning("No data in 5m-30m window.")
        else:
            # Compute probabilities
            stats_df['pred_prob'] = compute_predicted_probability(stats_df, artifacts['model_summary'])
            
            # Aggregate by event
            event_stats = []
            for (ticker, evt), grp in stats_df.groupby(['ticker', 'acceptance_datetime_utc']):
                probs = grp['pred_prob'].dropna()
                if len(probs) < 10:
                    continue
                
                # Check if forward return is available
                fwd_ret = grp['target_ret_600s'].dropna()
                
                event_stats.append({
                    'ticker': ticker,
                    'event': pd.to_datetime(evt).strftime('%Y-%m-%d %H:%M'),
                    'mean_prob': probs.mean(),
                    'max_prob': probs.max(),
                    'min_prob': probs.min(),
                    'pct_above_60': (probs >= 0.6).mean() * 100,
                    'pct_below_40': (probs < 0.4).mean() * 100,
                    'actual_positive': (fwd_ret > 0).mean() if len(fwd_ret) > 0 else np.nan
                })
            
            if not event_stats:
                st.warning("No events with sufficient data.")
            else:
                event_df_stats = pd.DataFrame(event_stats)
                
                # Summary metrics
                st.markdown("#### Overall Statistics")
                overall_cols = st.columns(4)
                with overall_cols[0]:
                    st.metric("Events Analyzed", len(event_df_stats))
                with overall_cols[1]:
                    st.metric("Avg Mean P", f"{event_df_stats['mean_prob'].mean():.3f}")
                with overall_cols[2]:
                    st.metric("Avg % P≥0.6", f"{event_df_stats['pct_above_60'].mean():.1f}%")
                with overall_cols[3]:
                    st.metric("Avg % P<0.4", f"{event_df_stats['pct_below_40'].mean():.1f}%")
                
                # Distribution of mean probabilities
                st.markdown("---")
                st.markdown("#### Distribution of Mean P(Positive) Across Events")
                
                fig_hist = px.histogram(
                    event_df_stats, x='mean_prob', nbins=30,
                    title='Distribution of Mean Predicted Probability per Event',
                    labels={'mean_prob': 'Mean P(Positive)'}
                )
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray")
                fig_hist.update_layout(height=350, template='plotly_white')
                st.plotly_chart(fig_hist, use_container_width=True, key="prob_hist")
                
                # Scatter: Mean P vs Actual Positive Rate
                st.markdown("#### Mean P vs Actual Positive Rate")
                valid_scatter = event_df_stats.dropna(subset=['actual_positive'])
                
                if len(valid_scatter) > 10:
                    fig_scatter = px.scatter(
                        valid_scatter, x='mean_prob', y='actual_positive',
                        color='ticker', hover_data=['event'],
                        title='Mean Predicted P vs Actual Positive Rate',
                        labels={'mean_prob': 'Mean P(Positive)', 'actual_positive': 'Actual Positive Rate'}
                    )
                    fig_scatter.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                     line=dict(color='gray', dash='dash'), name='y=x'))
                    fig_scatter.update_layout(height=400, template='plotly_white',
                                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")
                else:
                    st.info("Not enough events with actual return data for scatter plot.")
                
                # Event table
                st.markdown("---")
                st.markdown("#### Event-Level Data")
                st.dataframe(
                    event_df_stats.style.format({
                        'mean_prob': '{:.3f}',
                        'max_prob': '{:.3f}',
                        'min_prob': '{:.3f}',
                        'pct_above_60': '{:.1f}%',
                        'pct_below_40': '{:.1f}%',
                        'actual_positive': '{:.3f}'
                    }).background_gradient(subset=['mean_prob'], cmap='RdYlGn', vmin=0.3, vmax=0.7),
                    use_container_width=True,
                    height=400
                )


if __name__ == "__main__":
    main()
