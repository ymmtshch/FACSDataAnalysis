"""
FACS Data Analysis - Main Application
Streamlit-based web application for flow cytometry data analysis
Simplified version according to README specifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
try:
    from utils.fcs_processor import FCSProcessor, load_and_process_fcs
    from utils.plotting import PlottingUtils
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    st.error(f"❌ 必須のutilsモジュールが見つかりません: {e}")
    st.stop()

# Streamlit configuration
st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with orange theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF6B35;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title
    st.markdown('<div class="main-header">🔬 FACS Data Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("*StreamlitベースのFACS（フローサイトメトリー）データ解析Webアプリケーション*")
    st.markdown("---")
    
    # Initialize session state
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'plotting_utils' not in st.session_state:
        st.session_state.plotting_utils = None
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    if st.session_state.fcs_data is not None:
        display_analysis_tabs()
    else:
        display_welcome_screen()

def setup_sidebar():
    """Setup sidebar with file upload and options"""
    
    with st.sidebar:
        st.header("📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "FCSファイルを選択してください",
            type=['fcs'],
            help="FCS 2.0、3.0、3.1形式対応"
        )
        
        if uploaded_file is not None:
            st.success("✅ ファイルがアップロードされました")
            
            # Processing options
            st.header("⚙️ 処理オプション")
            
            transformation = st.selectbox(
                "データ変換",
                ["なし", "Log10", "Asinh", "Biexponential"],
                help="データ変換方法を選択"
            )
            
            max_events = st.number_input(
                "最大イベント数",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="パフォーマンス最適化（1,000～100,000）"
            )
            
            # Process button
            if st.button("📊 ファイルを処理", type="primary"):
                process_file(uploaded_file, transformation, max_events)

def process_file(uploaded_file, transformation, max_events):
    """Process FCS file"""
    try:
        with st.spinner("処理中..."):
            processor, data, metadata, error_message = load_and_process_fcs(
                uploaded_file, transformation, max_events
            )
            
            if data is not None:
                st.session_state.processor = processor
                st.session_state.fcs_data = data
                st.session_state.plotting_utils = PlottingUtils()
                
                library_used = getattr(processor, 'used_library', '不明')
                st.sidebar.success(f"✅ 処理完了 (使用ライブラリ: {library_used})")
                st.rerun()
            else:
                st.sidebar.error(f"❌ 処理失敗: {error_message}")
                
    except Exception as e:
        st.sidebar.error(f"❌ エラー: {str(e)}")

def display_welcome_screen():
    """Display welcome screen"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 FACS Data Analysis へようこそ</h3>
            <p>フローサイトメトリー（FACS）データの解析アプリケーションです。</p>
            
            <h4>📋 主な機能：</h4>
            <ul>
                <li>🔍 FCSファイルの読み込みと解析</li>
                <li>📊 ヒストグラムと散布図の作成</li>
                <li>🎯 基本的なゲーティング</li>
                <li>📈 統計解析とエクスポート</li>
            </ul>
            
            <h4>🔧 使用方法：</h4>
            <ol>
                <li>サイドバーからFCSファイルをアップロード</li>
                <li>処理オプションを設定</li>
                <li>「ファイルを処理」ボタンをクリック</li>
                <li>解析結果を確認</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def display_analysis_tabs():
    """Display main analysis interface with 4 tabs"""
    
    data = st.session_state.fcs_data
    processor = st.session_state.processor
    plotting_utils = st.session_state.plotting_utils
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総イベント数", f"{len(data):,}")
    with col2:
        st.metric("パラメータ数", len(data.columns))
    with col3:
        st.metric("使用ライブラリ", getattr(processor, 'used_library', '不明'))
    with col4:
        st.metric("データサイズ", f"{data.memory_usage().sum() / 1024 / 1024:.1f} MB")
    
    # 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 基本情報", "📈 可視化", "🎯 ゲーティング", "📋 統計解析"
    ])
    
    with tab1:
        display_basic_info(data, processor)
    
    with tab2:
        display_visualization(data, plotting_utils)
    
    with tab3:
        display_gating(data)
    
    with tab4:
        display_statistics(data, processor)

def display_basic_info(data, processor):
    """Basic information tab"""
    
    st.markdown('<div class="section-header">📁 ファイル情報</div>', 
                unsafe_allow_html=True)
    
    # File info
    file_info = processor.get_file_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基本情報**")
        info_data = [
            ["総イベント数", f"{file_info.get('total_events', 'N/A'):,}"],
            ["パラメータ数", file_info.get('parameters', 'N/A')],
            ["取得日時", file_info.get('date', 'N/A')],
            ["使用機器", file_info.get('cytometer', 'N/A')]
        ]
        st.dataframe(pd.DataFrame(info_data, columns=["項目", "値"]), 
                    hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**実験情報**")
        exp_data = [
            ["実験名", file_info.get('experiment_name', 'N/A')],
            ["サンプルID", file_info.get('sample_id', 'N/A')],
            ["オペレーター", file_info.get('operator', 'N/A')],
            ["ソフトウェア", file_info.get('software', 'N/A')]
        ]
        st.dataframe(pd.DataFrame(exp_data, columns=["項目", "値"]), 
                    hide_index=True, use_container_width=True)
    
    # Data preview
    st.markdown('<div class="section-header">📊 データプレビュー</div>', 
                unsafe_allow_html=True)
    st.dataframe(data.head(100), height=300, use_container_width=True)

def display_visualization(data, plotting_utils):
    """Visualization tab"""
    
    st.markdown('<div class="section-header">📈 データ可視化</div>', 
                unsafe_allow_html=True)
    
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
    # Plot type selection
    viz_type = st.selectbox(
        "可視化タイプ",
        ["ヒストグラム", "散布図", "密度プロット（等高線）"]
    )
    
    if viz_type == "ヒストグラム":
        col1, col2 = st.columns([1, 3])
        
        with col1:
            channel = st.selectbox("チャンネル", channels)
            bins = st.slider("ビン数", 20, 200, 50)
            log_scale = st.checkbox("対数スケール")
        
        with col2:
            fig = plotting_utils.create_histogram(data, channel, bins=bins)
            if log_scale:
                fig.update_layout(yaxis_type="log")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "散布図":
        col1, col2 = st.columns([1, 3])
        
        with col1:
            x_channel = st.selectbox("X軸", channels, index=0)
            y_channel = st.selectbox("Y軸", channels, index=1 if len(channels) > 1 else 0)
            alpha = st.slider("透明度", 0.1, 1.0, 0.6)
        
        with col2:
            fig = plotting_utils.create_scatter_plot(data, x_channel, y_channel, alpha=alpha)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # 密度プロット
        col1, col2 = st.columns([1, 3])
        
        with col1:
            x_channel = st.selectbox("X軸", channels, index=0, key="density_x")
            y_channel = st.selectbox("Y軸", channels, index=1 if len(channels) > 1 else 0, key="density_y")
        
        with col2:
            fig = plotting_utils.create_density_plot(data, x_channel, y_channel)
            st.plotly_chart(fig, use_container_width=True)

def display_gating(data):
    """Gating tab"""
    
    st.markdown('<div class="section-header">🎯 ゲーティング解析</div>', 
                unsafe_allow_html=True)
    
    st.info("💡 基本閾値ゲーティング機能を提供します。高度なゲーティング機能は`pages/advanced_gating.py`をご利用ください。")
    
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ゲート設定**")
        gate_channel = st.selectbox("ゲートチャンネル", channels)
        
        default_threshold = float(data[gate_channel].median())
        threshold = st.number_input(
            "閾値（デフォルト：中央値）",
            value=default_threshold,
            step=float(data[gate_channel].std() / 10) if data[gate_channel].std() > 0 else 1.0
        )
        
        gate_direction = st.selectbox(
            "ゲート方向",
            ["以上 (≥)", "より大きい (>)", "以下 (≤)", "より小さい (<)"]
        )
    
    with col2:
        st.markdown("**ゲート結果**")
        
        # Apply gate
        if gate_direction == "以上 (≥)":
            gated_data = data[data[gate_channel] >= threshold]
        elif gate_direction == "より大きい (>)":
            gated_data = data[data[gate_channel] > threshold]
        elif gate_direction == "以下 (≤)":
            gated_data = data[data[gate_channel] <= threshold]
        else:
            gated_data = data[data[gate_channel] < threshold]
        
        # Display results
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("ゲート前", f"{len(data):,}")
            st.metric("ゲート後", f"{len(gated_data):,}")
        with col2b:
            gate_rate = (len(gated_data) / len(data)) * 100 if len(data) > 0 else 0
            st.metric("ゲート率", f"{gate_rate:.1f}%")
            st.metric("除外数", f"{len(data) - len(gated_data):,}")
    
    # Visualization
    st.markdown("### ゲート可視化")
    
    fig = px.histogram(data, x=gate_channel, nbins=50, title=f"{gate_channel} ゲート表示")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"閾値: {threshold:.2f}")
    
    # Highlight gated region
    if gate_direction in ["以上 (≥)", "より大きい (>)"]:
        fig.add_vrect(x0=threshold, x1=data[gate_channel].max(),
                      fillcolor="green", opacity=0.2, annotation_text="ゲート領域")
    else:
        fig.add_vrect(x0=data[gate_channel].min(), x1=threshold,
                      fillcolor="green", opacity=0.2, annotation_text="ゲート領域")
    
    st.plotly_chart(fig, use_container_width=True)

def display_statistics(data, processor):
    """Statistics tab"""
    
    st.markdown('<div class="section-header">📋 統計解析</div>', 
                unsafe_allow_html=True)
    
    # Get statistics
    stats = processor.get_basic_stats()
    
    if stats:
        # Display statistics
        st.markdown("### 全チャンネル統計")
        st.markdown("**平均値、中央値、標準偏差、最小値、最大値**")
        
        stats_df = pd.DataFrame.from_dict(stats, orient='index').round(3)
        st.dataframe(stats_df, use_container_width=True)
        
        # Summary
        st.markdown("### 統計サマリー")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("解析チャンネル数", len(numeric_cols))
        with col2:
            st.metric("総イベント数", f"{len(data):,}")
        with col3:
            range_min = data[numeric_cols].min().min()
            range_max = data[numeric_cols].max().max()
            st.metric("データ範囲", f"{range_min:.1f} - {range_max:.1f}")
        with col4:
            st.metric("データサイズ", f"{data.memory_usage().sum() / 1024 / 1024:.1f} MB")
        
        # Export section
        st.markdown('<div class="section-header">💾 データエクスポート</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export processed data
            csv_data = processor.export_data(data, data_type="data")
            filename = getattr(processor, 'filename', 'fcs_data').rsplit('.', 1)[0]
            
            st.download_button(
                label="📁 処理済みデータをCSVでダウンロード",
                data=csv_data,
                file_name=f"{filename}_data.csv",
                mime="text/csv",
                help="処理済みデータをCSV形式でダウンロード"
            )
        
        with col2:
            # Export statistics
            stats_csv = stats_df.to_csv(index=True, encoding='utf-8')
            
            st.download_button(
                label="📈 統計データをCSVでダウンロード",
                data=stats_csv,
                file_name=f"{filename}_stats.csv",
                mime="text/csv",
                help="統計情報をCSV形式でダウンロード"
            )
    
    else:
        st.info("統計情報が利用できません")

if __name__ == "__main__":
    main()
