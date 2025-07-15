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
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import utility modules with better error handling
try:
    from utils.fcs_processor import FCSProcessor, load_and_process_fcs
    from utils.plotting import PlottingUtils
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    st.error(f"❌ 必須のutilsモジュールが見つかりません: {e}")
    st.error("utils/フォルダ内のfcs_processor.pyとplotting.pyが必要です")
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
    .metric-container {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E0E0E0;
        margin: 0.5rem 0;
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
    
    # Initialize session state with defaults
    initialize_session_state()
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    if st.session_state.fcs_data is not None:
        display_analysis_tabs()
    else:
        display_welcome_screen()

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'fcs_data': None,
        'processor': None,
        'plotting_utils': None,
        'file_processed': False,
        'error_message': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_sidebar():
    """Setup sidebar with file upload and options"""
    
    with st.sidebar:
        st.header("📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "FCSファイルを選択してください",
            type=['fcs'],
            help="FCS 2.0、3.0、3.1形式対応 (最大100MB)",
            key="fcs_uploader"
        )
        
        if uploaded_file is not None:
            # File info
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.success(f"✅ ファイル: {uploaded_file.name}")
            st.info(f"📊 サイズ: {file_size:.1f} MB")
            
            # Processing options
            st.header("⚙️ 処理オプション")
            
            transformation = st.selectbox(
                "データ変換",
                ["なし", "Log10", "Asinh"],  # Removed Biexponential for simplicity
                help="データ変換方法を選択",
                key="transformation_select"
            )
            
            max_events = st.number_input(
                "最大イベント数",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="パフォーマンス最適化（推奨: 10,000）",
                key="max_events_input"
            )
            
            # Process button
            if st.button("📊 ファイルを処理", type="primary", key="process_button"):
                if file_size > 100:
                    st.error("❌ ファイルサイズが100MBを超えています")
                else:
                    process_file(uploaded_file, transformation, max_events)
        
        # Additional info
        if st.session_state.fcs_data is not None:
            st.header("📊 現在のデータ")
            st.info(f"イベント数: {len(st.session_state.fcs_data):,}")
            st.info(f"パラメータ数: {len(st.session_state.fcs_data.columns)}")
            
            # Reset button
            if st.button("🔄 データをリセット", key="reset_button"):
                reset_session_state()
                st.rerun()

def reset_session_state():
    """Reset session state"""
    st.session_state.fcs_data = None
    st.session_state.processor = None
    st.session_state.plotting_utils = None
    st.session_state.file_processed = False
    st.session_state.error_message = None

def process_file(uploaded_file, transformation, max_events):
    """Process FCS file with better error handling"""
    try:
        with st.spinner("📊 FCSファイルを処理中..."):
            # Process file
            processor, data, metadata, error_message = load_and_process_fcs(
                uploaded_file, transformation, max_events
            )
            
            if data is not None and len(data) > 0:
                # Success
                st.session_state.processor = processor
                st.session_state.fcs_data = data
                st.session_state.plotting_utils = PlottingUtils()
                st.session_state.file_processed = True
                st.session_state.error_message = None
                
                # Show success message
                library_used = getattr(processor, 'used_library', 'fcsparser')
                st.sidebar.success(f"✅ 処理完了")
                st.sidebar.info(f"📚 使用ライブラリ: {library_used}")
                st.sidebar.info(f"📊 読み込みイベント数: {len(data):,}")
                
                st.rerun()
            else:
                # Error
                error_msg = error_message or "不明なエラーが発生しました"
                st.session_state.error_message = error_msg
                st.sidebar.error(f"❌ 処理失敗: {error_msg}")
                
    except Exception as e:
        error_msg = f"ファイル処理中にエラーが発生しました: {str(e)}"
        st.session_state.error_message = error_msg
        st.sidebar.error(f"❌ {error_msg}")

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
                <li>📊 ヒストグラム、散布図、密度プロットの作成</li>
                <li>🎯 基本的な閾値ゲーティング</li>
                <li>📈 統計解析とCSVエクスポート</li>
            </ul>
            
            <h4>🔧 使用方法：</h4>
            <ol>
                <li>サイドバーからFCSファイルをアップロード</li>
                <li>処理オプションを設定（変換方法、最大イベント数）</li>
                <li>「ファイルを処理」ボタンをクリック</li>
                <li>4つのタブで解析結果を確認</li>
            </ol>
            
            <h4>💡 ヒント：</h4>
            <ul>
                <li>大きなファイルは最大イベント数を調整してください</li>
                <li>エラーが発生した場合は変換を「なし」に変更してください</li>
                <li>詳細な解析は「基本解析」および「高度ゲーティング」ページをご利用ください</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Error display
        if st.session_state.error_message:
            st.error(f"❌ 最後のエラー: {st.session_state.error_message}")

def display_analysis_tabs():
    """Display main analysis interface with 4 tabs"""
    
    data = st.session_state.fcs_data
    processor = st.session_state.processor
    plotting_utils = st.session_state.plotting_utils
    
    if data is None or len(data) == 0:
        st.error("❌ データが利用できません")
        return
    
    # Display metrics
    display_metrics(data, processor)
    
    # 4 tabs as specified in README
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

def display_metrics(data, processor):
    """Display key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総イベント数", f"{len(data):,}")
    
    with col2:
        st.metric("パラメータ数", len(data.columns))
    
    with col3:
        library_used = getattr(processor, 'used_library', '不明')
        st.metric("使用ライブラリ", library_used)
    
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("データサイズ", f"{memory_mb:.1f} MB")

def display_basic_info(data, processor):
    """Basic information tab"""
    
    st.markdown('<div class="section-header">📁 ファイル情報</div>', 
                unsafe_allow_html=True)
    
    try:
        # File info
        file_info = processor.get_file_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**基本情報**")
            info_data = [
                ["総イベント数", f"{file_info.get('total_events', len(data)):,}"],
                ["パラメータ数", file_info.get('parameters', len(data.columns))],
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
    
    except Exception as e:
        st.warning(f"ファイル情報の取得中にエラーが発生しました: {str(e)}")
    
    # Channel information
    st.markdown('<div class="section-header">📊 チャンネル情報</div>', 
                unsafe_allow_html=True)
    
    channels = data.columns.tolist()
    numeric_channels = data.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**全チャンネル**")
        st.write(f"総数: {len(channels)}")
        st.write(", ".join(channels))
    
    with col2:
        st.markdown("**数値チャンネル**")
        st.write(f"総数: {len(numeric_channels)}")
        st.write(", ".join(numeric_channels))
    
    # Data preview
    st.markdown('<div class="section-header">📊 データプレビュー</div>', 
                unsafe_allow_html=True)
    
    # Show first 100 rows
    preview_data = data.head(100)
    st.dataframe(preview_data, height=300, use_container_width=True)
    
    st.info(f"💡 表示: 最初の100行 / 総計 {len(data):,} 行")

def display_visualization(data, plotting_utils):
    """Visualization tab"""
    
    st.markdown('<div class="section-header">📈 データ可視化</div>', 
                unsafe_allow_html=True)
    
    # Get numeric channels
    numeric_channels = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_channels:
        st.error("❌ 数値チャンネルが見つかりません")
        return
    
    # Plot type selection
    viz_type = st.selectbox(
        "可視化タイプ",
        ["ヒストグラム", "散布図", "密度プロット（等高線）"],
        key="viz_type_select"
    )
    
    try:
        if viz_type == "ヒストグラム":
            display_histogram(data, numeric_channels, plotting_utils)
        elif viz_type == "散布図":
            display_scatter_plot(data, numeric_channels, plotting_utils)
        else:  # 密度プロット
            display_density_plot(data, numeric_channels, plotting_utils)
    
    except Exception as e:
        st.error(f"❌ プロット作成中にエラーが発生しました: {str(e)}")

def display_histogram(data, channels, plotting_utils):
    """Display histogram"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        channel = st.selectbox("チャンネル", channels, key="hist_channel")
        bins = st.slider("ビン数", 20, 200, 50, key="hist_bins")
        log_scale = st.checkbox("対数スケール", key="hist_log")
        
        # Show basic stats
        st.markdown("**統計情報**")
        channel_data = data[channel].dropna()
        stats_data = [
            ["最小値", f"{channel_data.min():.2f}"],
            ["最大値", f"{channel_data.max():.2f}"],
            ["平均値", f"{channel_data.mean():.2f}"],
            ["中央値", f"{channel_data.median():.2f}"]
        ]
        st.dataframe(pd.DataFrame(stats_data, columns=["統計", "値"]), 
                    hide_index=True, use_container_width=True)
    
    with col2:
        fig = plotting_utils.create_histogram(data, channel, bins=bins)
        if log_scale:
            fig.update_layout(yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

def display_scatter_plot(data, channels, plotting_utils):
    """Display scatter plot"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_channel = st.selectbox("X軸", channels, index=0, key="scatter_x")
        y_channel = st.selectbox("Y軸", channels, 
                                index=1 if len(channels) > 1 else 0, 
                                key="scatter_y")
        alpha = st.slider("透明度", 0.1, 1.0, 0.6, key="scatter_alpha")
        
        # Sample size for large datasets
        sample_size = min(len(data), 10000)
        if len(data) > 10000:
            st.info(f"💡 パフォーマンスのため {sample_size:,} 点をサンプリング表示")
    
    with col2:
        # Sample data if too large
        plot_data = data.sample(n=min(len(data), 10000))
        fig = plotting_utils.create_scatter_plot(plot_data, x_channel, y_channel, alpha=alpha)
        st.plotly_chart(fig, use_container_width=True)

def display_density_plot(data, channels, plotting_utils):
    """Display density plot"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_channel = st.selectbox("X軸", channels, index=0, key="density_x")
        y_channel = st.selectbox("Y軸", channels, 
                                index=1 if len(channels) > 1 else 0, 
                                key="density_y")
        
        # Sample size for large datasets
        sample_size = min(len(data), 10000)
        if len(data) > 10000:
            st.info(f"💡 パフォーマンスのため {sample_size:,} 点をサンプリング表示")
    
    with col2:
        # Sample data if too large
        plot_data = data.sample(n=min(len(data), 10000))
        fig = plotting_utils.create_density_plot(plot_data, x_channel, y_channel)
        st.plotly_chart(fig, use_container_width=True)

def display_gating(data):
    """Gating tab with basic threshold gating"""
    
    st.markdown('<div class="section-header">🎯 基本ゲーティング</div>', 
                unsafe_allow_html=True)
    
    st.info("💡 基本的な閾値ゲーティング機能です。複雑なゲーティングは「高度ゲーティングページ」をご利用ください。")
    
    # Get numeric channels
    numeric_channels = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_channels:
        st.error("❌ 数値チャンネルが見つかりません")
        return
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ゲート設定**")
            gate_channel = st.selectbox("ゲートチャンネル", numeric_channels, key="gate_channel")
            
            # Calculate reasonable default threshold
            channel_data = data[gate_channel].dropna()
            default_threshold = float(channel_data.median())
            
            threshold = st.number_input(
                "閾値",
                value=default_threshold,
                step=float(channel_data.std() / 10) if channel_data.std() > 0 else 1.0,
                format="%.3f",
                key="gate_threshold"
            )
            
            gate_direction = st.selectbox(
                "ゲート方向",
                ["以上 (≥)", "より大きい (>)", "以下 (≤)", "より小さい (<)"],
                key="gate_direction"
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
            else:  # より小さい (<)
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
        
        # Create histogram with gate line
        fig = px.histogram(data, x=gate_channel, nbins=50, 
                          title=f"{gate_channel} ゲート表示")
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                      annotation_text=f"閾値: {threshold:.3f}")
        
        # Highlight gated region
        if gate_direction in ["以上 (≥)", "より大きい (>)"]:
            fig.add_vrect(x0=threshold, x1=channel_data.max(),
                          fillcolor="green", opacity=0.2, 
                          annotation_text="ゲート領域")
        else:
            fig.add_vrect(x0=channel_data.min(), x1=threshold,
                          fillcolor="green", opacity=0.2, 
                          annotation_text="ゲート領域")
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ ゲーティング処理中にエラーが発生しました: {str(e)}")

def display_statistics(data, processor):
    """Statistics tab"""
    
    st.markdown('<div class="section-header">📋 統計解析</div>', 
                unsafe_allow_html=True)
    
    try:
        # Get statistics
        stats = processor.get_basic_stats()
        
        if stats and len(stats) > 0:
            # Display statistics
            st.markdown("### 全チャンネル統計")
            
            stats_df = pd.DataFrame.from_dict(stats, orient='index')
            stats_df = stats_df.round(3)
            
            # Reorder columns for better display
            desired_columns = ['mean', 'median', 'std', 'min', 'max']
            available_columns = [col for col in desired_columns if col in stats_df.columns]
            if available_columns:
                stats_df = stats_df[available_columns]
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Summary metrics
            st.markdown("### 統計サマリー")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("解析チャンネル数", len(numeric_cols))
            with col2:
                st.metric("総イベント数", f"{len(data):,}")
            with col3:
                if len(numeric_cols) > 0:
                    range_min = data[numeric_cols].min().min()
                    range_max = data[numeric_cols].max().max()
                    st.metric("データ範囲", f"{range_min:.1f} - {range_max:.1f}")
                else:
                    st.metric("データ範囲", "N/A")
            with col4:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("データサイズ", f"{memory_mb:.1f} MB")
        
        else:
            st.warning("⚠️ 統計情報を取得できませんでした")
            
            # Manual calculation
            st.markdown("### 基本統計（手動計算）")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                manual_stats = data[numeric_cols].describe()
                st.dataframe(manual_stats, use_container_width=True)
            else:
                st.error("❌ 数値データがありません")
        
        # Export section
        st.markdown('<div class="section-header">💾 データエクスポート</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export processed data
            try:
                csv_data = processor.export_data(data, data_type="data")
                filename = getattr(processor, 'filename', 'fcs_data')
                if filename.endswith('.fcs'):
                    filename = filename[:-4]  # Remove .fcs extension
                
                st.download_button(
                    label="📁 処理済みデータをCSVでダウンロード",
                    data=csv_data,
                    file_name=f"{filename}_data.csv",
                    mime="text/csv",
                    help="処理済みデータをCSV形式でダウンロード"
                )
            except Exception as e:
                st.error(f"❌ データエクスポートエラー: {str(e)}")
        
        with col2:
            # Export statistics
            try:
                if stats and len(stats) > 0:
                    stats_csv = pd.DataFrame.from_dict(stats, orient='index').to_csv(
                        index=True, encoding='utf-8'
                    )
                else:
                    # Use manual stats
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        stats_csv = data[numeric_cols].describe().to_csv(
                            index=True, encoding='utf-8'
                        )
                    else:
                        stats_csv = "No numeric data available"
                
                filename = getattr(processor, 'filename', 'fcs_data')
                if filename.endswith('.fcs'):
                    filename = filename[:-4]
                
                st.download_button(
                    label="📈 統計データをCSVでダウンロード",
                    data=stats_csv,
                    file_name=f"{filename}_stats.csv",
                    mime="text/csv",
                    help="統計情報をCSV形式でダウンロード"
                )
            except Exception as e:
                st.error(f"❌ 統計エクスポートエラー: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ 統計解析中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
