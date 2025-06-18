"""
FACS Data Analysis - Main Application
Streamlit-based web application for flow cytometry data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.fcs_processor import FCSProcessor, load_and_process_fcs
from utils.plotting import create_histogram, create_scatter_plot
from utils.gating import GatingTool
import config

# Page configuration
st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title
    st.markdown('<div class="main-header">🔬 FACS Data Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'fcs_metadata' not in st.session_state:
        st.session_state.fcs_metadata = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "FCSファイルを選択してください",
            type=['fcs'],
            help="FCS 2.0、3.0、3.1形式に対応しています"
        )
        
        if uploaded_file is not None:
            st.success("✅ ファイルがアップロードされました")
            
            # Processing options
            st.header("⚙️ 処理オプション")
            
            transformation = st.selectbox(
                "変換方法",
                ["none", "log", "asinh", "biexp"],
                help="データ変換方法を選択してください"
            )
            
            max_events = st.number_input(
                "最大イベント数",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="表示する最大イベント数"
            )
            
            # Process file button
            if st.button("📊 ファイルを処理", type="primary"):
                with st.spinner("FCSファイルを処理中..."):
                    processor, data, metadata = load_and_process_fcs(
                        uploaded_file, transformation, max_events
                    )
                    
                    if processor is not None:
                        st.session_state.processor = processor
                        st.session_state.fcs_data = data
                        st.session_state.fcs_metadata = metadata
                        st.success("✅ 処理が完了しました")
                        st.rerun()
                    else:
                        st.error("❌ ファイルの処理に失敗しました")
    
    # Main content area
    if st.session_state.fcs_data is not None:
        display_analysis_interface()
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen when no file is loaded"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 FACS Data Analysis へようこそ</h3>
            <p>このアプリケーションでは、フローサイトメトリー（FACS）データの解析が可能です。</p>
            
            <h4>📋 主な機能：</h4>
            <ul>
                <li>🔍 FCSファイルの読み込みと解析</li>
                <li>📊 ヒストグラムと散布図の作成</li>
                <li>🎯 インタラクティブなゲーティング</li>
                <li>📈 統計解析と結果エクスポート</li>
            </ul>
            
            <h4>🔧 使用方法：</h4>
            <ol>
                <li>左サイドバーからFCSファイルをアップロード</li>
                <li>処理オプションを設定</li>
                <li>「ファイルを処理」ボタンをクリック</li>
                <li>解析結果を確認・可視化</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def display_analysis_interface():
    """Display main analysis interface"""
    
    data = st.session_state.fcs_data
    metadata = st.session_state.fcs_metadata
    processor = st.session_state.processor
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 基本情報", "📈 可視化", "🎯 ゲーティング", "📋 統計解析"
    ])
    
    with tab1:
        display_file_info(data, metadata, processor)
    
    with tab2:
        display_visualization(data)
    
    with tab3:
        display_gating_interface(data)
    
    with tab4:
        display_statistics(data, processor)

def display_file_info(data, metadata, processor):
    """Display file information and basic statistics"""
    
    st.markdown('<div class="section-header">📁 ファイル情報</div>', 
                unsafe_allow_html=True)
    
    # File information
    file_info = processor.get_file_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基本情報**")
        info_df = pd.DataFrame([
            ["総イベント数", f"{file_info.get('total_events', 'N/A'):,}"],
            ["パラメータ数", file_info.get('total_parameters', 'N/A')],
            ["取得日", file_info.get('acquisition_date', 'N/A')],
            ["取得時刻", file_info.get('acquisition_time', 'N/A')],
            ["サイトメーター", file_info.get('cytometer', 'N/A')]
        ], columns=["項目", "値"])
        st.dataframe(info_df, hide_index=True)
    
    with col2:
        st.markdown("**実験情報**")
        exp_df = pd.DataFrame([
            ["実験名", file_info.get('experiment_name', 'N/A')],
            ["サンプルID", file_info.get('sample_id', 'N/A')],
            ["オペレーター", file_info.get('operator', 'N/A')],
            ["ソフトウェア", file_info.get('software', 'N/A')]
        ], columns=["項目", "値"])
        st.dataframe(exp_df, hide_index=True)
    
    # Channel information
    st.markdown('<div class="section-header">📋 チャンネル情報</div>', 
                unsafe_allow_html=True)
    
    channel_info = processor.get_channel_info()
    if channel_info:
        channel_df = pd.DataFrame.from_dict(channel_info, orient='index')
        st.dataframe(channel_df)
    else:
        st.info("チャンネル情報が利用できません")
    
    # Data preview
    st.markdown('<div class="section-header">📊 データプレビュー</div>', 
                unsafe_allow_html=True)
    
    st.dataframe(data.head(100), height=300)

def display_visualization(data):
    """Display visualization options"""
    
    st.markdown('<div class="section-header">📈 データ可視化</div>', 
                unsafe_allow_html=True)
    
    channels = list(data.columns)
    
    # Visualization options
    viz_type = st.selectbox(
        "可視化タイプ",
        ["ヒストグラム", "散布図", "等高線プロット"]
    )
    
    if viz_type == "ヒストグラム":
        display_histogram(data, channels)
    elif viz_type == "散布図":
        display_scatter_plot(data, channels)
    elif viz_type == "等高線プロット":
        display_contour_plot(data, channels)

def display_histogram(data, channels):
    """Display histogram visualization"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_channel = st.selectbox("チャンネル選択", channels)
        bins = st.slider("ビン数", 20, 200, 50)
        log_scale = st.checkbox("対数スケール")
    
    with col2:
        fig = px.histogram(
            data, 
            x=selected_channel, 
            nbins=bins,
            title=f"{selected_channel} ヒストグラム"
        )
        
        if log_scale:
            fig.update_layout(yaxis_type="log")
        
        st.plotly_chart(fig, use_container_width=True)

def display_scatter_plot(data, channels):
    """Display scatter plot visualization"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_channel = st.selectbox("X軸", channels, index=0)
        y_channel = st.selectbox("Y軸", channels, index=1 if len(channels) > 1 else 0)
        alpha = st.slider("透明度", 0.1, 1.0, 0.6)
        sample_size = st.slider("サンプルサイズ", 1000, len(data), min(10000, len(data)))
    
    with col2:
        # Subsample for better performance
        plot_data = data.sample(n=min(sample_size, len(data)))
        
        fig = px.scatter(
            plot_data,
            x=x_channel,
            y=y_channel,
            title=f"{x_channel} vs {y_channel}",
            opacity=alpha
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_contour_plot(data, channels):
    """Display contour plot visualization"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_channel = st.selectbox("X軸", channels, index=0, key="contour_x")
        y_channel = st.selectbox("Y軸", channels, index=1 if len(channels) > 1 else 0, key="contour_y")
        nbins = st.slider("密度計算ビン数", 20, 100, 50)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram2dContour(
            x=data[x_channel],
            y=data[y_channel],
            nbinsx=nbins,
            nbinsy=nbins,
            contours_coloring='fill',
            contours_showlabels=True
        ))
        
        fig.update_layout(
            title=f"{x_channel} vs {y_channel} 密度プロット",
            xaxis_title=x_channel,
            yaxis_title=y_channel
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_gating_interface(data):
    """Display gating interface"""
    
    st.markdown('<div class="section-header">🎯 ゲーティング解析</div>', 
                unsafe_allow_html=True)
    
    st.info("ゲーティング機能は開発中です。高度なゲーティング機能については、advanced_gating.pyページをご確認ください。")
    
    # Simple threshold gating example
    channels = list(data.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gate_channel = st.selectbox("ゲートチャンネル", channels)
        threshold = st.number_input(
            "閾値",
            value=float(data[gate_channel].median()),
            step=float(data[gate_channel].std() / 10)
        )
    
    with col2:
        # Apply simple threshold gate
        gated_data = data[data[gate_channel] > threshold]
        
        st.metric("元データ", f"{len(data):,} events")
        st.metric("ゲート後", f"{len(gated_data):,} events")
        st.metric("ゲート率", f"{len(gated_data)/len(data)*100:.1f}%")

def display_statistics(data, processor):
    """Display statistical analysis"""
    
    st.markdown('<div class="section-header">📋 統計解析</div>', 
                unsafe_allow_html=True)
    
    # Basic statistics for all channels
    stats = processor.get_basic_stats()
    
    if stats:
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df = stats_df.round(2)
        
        st.markdown("**全チャンネル統計情報**")
        st.dataframe(stats_df)
        
        # Export options
        st.markdown('<div class="section-header">💾 データエクスポート</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 統計データをダウンロード"):
                csv_stats = stats_df.to_csv()
                st.download_button(
                    label="CSV形式でダウンロード",
                    data=csv_stats,
                    file_name="facs_statistics.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 生データをダウンロード"):
                csv_data = processor.export_data(data)
                st.download_button(
                    label="CSV形式でダウンロード",
                    data=csv_data,
                    file_name="facs_raw_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
