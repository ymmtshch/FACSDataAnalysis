"""
FACS Data Analysis - Main Application
Streamlit-based web application for flow cytometry data analysis
Updated according to README specifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

# Import utility modules as specified in README
try:
    from utils.fcs_processor import FCSProcessor, load_and_process_fcs
    from utils.plotting import PlottingUtils
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.error("❌ 必須のutilsモジュールが見つかりません。utils/fcs_processor.py、utils/plotting.pyが必要です。")

# FCS file reading libraries (in order of preference as per README)
try:
    import fcsparser
    FCSPARSER_AVAILABLE = True
except ImportError:
    FCSPARSER_AVAILABLE = False

try:
    import flowio
    FLOWIO_AVAILABLE = True
except ImportError:
    FLOWIO_AVAILABLE = False

try:
    import flowkit
    FLOWKIT_AVAILABLE = True
except ImportError:
    FLOWKIT_AVAILABLE = False

# Check if any FCS library is available
if not FCSPARSER_AVAILABLE and not FLOWIO_AVAILABLE and not FLOWKIT_AVAILABLE:
    st.error("❌ FCSファイル読み込みライブラリが見つかりません。fcsparser、flowio、またはflowkitをインストールしてください。")
    st.stop()

# Check for utils availability
if not UTILS_AVAILABLE:
    st.warning("⚠️ utilsモジュールが利用できません。基本機能のみ提供されます。")
    st.stop()

# Page configuration - as specified in README
st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Updated with README specifications (Orange theme #FF6B35)
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
    .library-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .library-available {
        background-color: #E8F5E8;
        color: #2E7D32;
    }
    .library-unavailable {
        background-color: #FFEBEE;
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)

class SimpleFCSProcessor:
    """Simplified FCS processor without FlowKit dependency"""
    
    def __init__(self):
        self.data = None
        self.metadata = None
        self.channels = None
        self.library_used = None
    
    def load_fcs_file(self, file, max_events=10000):
        """Load FCS file using available libraries"""
        try:
            # Try flowio first (recommended)
            if FLOWIO_AVAILABLE:
                return self._load_with_flowio(file, max_events)
            elif FCSPARSER_AVAILABLE:
                return self._load_with_fcsparser(file, max_events)
            else:
                raise ImportError("No FCS reading library available")
        except Exception as e:
            st.error(f"❌ FCSファイルの読み込みに失敗しました: {str(e)}")
            return None, None, None
    
    def _load_with_flowio(self, file, max_events):
        """Load FCS file using flowio"""
        try:
            fcs = flowio.FlowData(file.getvalue())
            self.library_used = "flowio"
            
            # Get event data
            events = fcs.events
            if isinstance(events, np.ndarray):
                data_array = events
            else:
                # Handle array.array case
                data_array = np.array(events)
            
            # Reshape if needed
            if len(data_array.shape) == 1:
                n_params = int(fcs.text.get('$PAR', 0))
                if n_params > 0:
                    data_array = data_array.reshape(-1, n_params)
            
            # Limit events
            if len(data_array) > max_events:
                indices = np.random.choice(len(data_array), max_events, replace=False)
                data_array = data_array[indices]
            
            # Get channel names
            channels = []
            n_params = int(fcs.text.get('$PAR', 0))
            for i in range(1, n_params + 1):
                channel_name = fcs.text.get(f'$P{i}N', f'Channel_{i}')
                if not channel_name or channel_name == '':
                    channel_name = fcs.text.get(f'$P{i}S', f'Channel_{i}')
                channels.append(channel_name)
            
            # Create DataFrame
            df = pd.DataFrame(data_array, columns=channels)
            
            # Store metadata
            self.metadata = dict(fcs.text)
            self.channels = channels
            self.data = df
            
            return df, self.metadata, channels
            
        except Exception as e:
            st.error(f"FlowIO読み込みエラー: {str(e)}")
            return None, None, None
    
    def _load_with_fcsparser(self, file, max_events):
        """Load FCS file using fcsparser"""
        try:
            # Reset file pointer
            file.seek(0)
            
            # Read with fcsparser
            metadata, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=True)
            self.library_used = "fcsparser"
            
            # Limit events
            if len(data) > max_events:
                data = data.sample(n=max_events, random_state=42)
            
            # Get channel names
            channels = list(data.columns)
            
            # Store data
            self.metadata = metadata
            self.channels = channels
            self.data = data
            
            return data, metadata, channels
            
        except Exception as e:
            st.error(f"FCSParser読み込みエラー: {str(e)}")
            return None, None, None
    
    def apply_transformation(self, data, transformation="none"):
        """Apply data transformation"""
        if transformation == "none":
            return data
        
        transformed_data = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            if transformation == "log":
                # Log10 transformation (add small value to avoid log(0))
                transformed_data[column] = np.log10(data[column] + 1)
            elif transformation == "asinh":
                # Asinh transformation
                transformed_data[column] = np.arcsinh(data[column] / 150)
            elif transformation == "biexp":
                # Simple biexponential approximation
                transformed_data[column] = np.sign(data[column]) * np.log10(np.abs(data[column]) + 1)
        
        return transformed_data
    
    def get_file_info(self):
        """Get basic file information"""
        if self.metadata is None:
            return {}
        
        info = {}
        
        # Basic information
        info['total_events'] = self.metadata.get('$TOT', 'N/A')
        info['total_parameters'] = self.metadata.get('$PAR', 'N/A')
        info['acquisition_date'] = self.metadata.get('$DATE', 'N/A')
        info['acquisition_time'] = self.metadata.get('$BTIM', 'N/A')
        info['cytometer'] = self.metadata.get('$CYT', 'N/A')
        
        # Experiment information
        info['experiment_name'] = self.metadata.get('$EXP', 'N/A')
        info['sample_id'] = self.metadata.get('$SMNO', 'N/A')
        info['operator'] = self.metadata.get('$OP', 'N/A')
        info['software'] = self.metadata.get('$SRC', 'N/A')
        
        return info
    
    def get_channel_info(self):
        """Get channel information"""
        if self.metadata is None or self.channels is None:
            return {}
        
        channel_info = {}
        n_params = int(self.metadata.get('$PAR', 0))
        
        for i in range(1, n_params + 1):
            channel_name = self.channels[i-1] if i-1 < len(self.channels) else f'Channel_{i}'
            
            channel_info[channel_name] = {
                'Range': self.metadata.get(f'$P{i}R', 'N/A'),
                'Bits': self.metadata.get(f'$P{i}B', 'N/A'),
                'Gain': self.metadata.get(f'$P{i}G', 'N/A'),
                'Voltage': self.metadata.get(f'$P{i}V', 'N/A')
            }
        
        return channel_info
    
    def get_basic_stats(self):
        """Get basic statistics for all channels"""
        if self.data is None:
            return {}
        
        stats = {}
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            stats[column] = {
                'Mean': self.data[column].mean(),
                'Median': self.data[column].median(),
                'Std': self.data[column].std(),
                'Min': self.data[column].min(),
                'Max': self.data[column].max(),
                'Count': self.data[column].count()
            }
        
        return stats
    
    def export_data(self, data):
        """Export data to CSV"""
        return data.to_csv(index=False)

def main():
    """Main application function - Updated according to README specifications"""
    
    # Title with orange theme
    st.markdown('<div class="main-header">🔬 FACS Data Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("*StreamlitベースのFACS（フローサイトメトリー）データ解析Webアプリケーション*")
    st.markdown("---")
    
    # Display available libraries status
    display_library_status()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content area
    if st.session_state.fcs_data is not None:
        display_analysis_interface()
    else:
        display_welcome_screen()

def display_library_status():
    """Display available FCS libraries status"""
    st.sidebar.markdown("### 📚 利用可能ライブラリ")
    
    libraries = [
        ("FCSParser", FCSPARSER_AVAILABLE, "推奨・第一優先"),
        ("FlowIO", FLOWIO_AVAILABLE, "第二優先"),
        ("FlowKit", FLOWKIT_AVAILABLE, "フォールバック")
    ]
    
    for lib_name, available, priority in libraries:
        status_class = "library-available" if available else "library-unavailable"
        status_icon = "✅" if available else "❌"
        st.sidebar.markdown(
            f'<div class="library-status {status_class}">'
            f'{status_icon} {lib_name} ({priority})'
            f'</div>',
            unsafe_allow_html=True
        )

def initialize_session_state():
    """Initialize session state variables"""
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'fcs_metadata' not in st.session_state:
        st.session_state.fcs_metadata = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'plotting_utils' not in st.session_state:
        st.session_state.plotting_utils = None
    if 'transformation_applied' not in st.session_state:
        st.session_state.transformation_applied = "none"
    if 'max_events_used' not in st.session_state:
        st.session_state.max_events_used = 10000

def setup_sidebar():
    """Setup sidebar with file upload and processing options"""
    
    with st.sidebar:
        st.header("📁 ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "FCSファイルを選択してください",
            type=['fcs'],
            help="FCS 2.0、3.0、3.1形式の標準的なフローサイトメトリーファイルに対応"
        )
        
        if uploaded_file is not None:
            st.success("✅ ファイルがアップロードされました")
            
            # Processing options as specified in README
            st.header("⚙️ 処理オプション")
            
            transformation = st.selectbox(
                "データ変換",
                ["none", "log10", "asinh", "biexponential"],
                index=0,
                help="なし、Log10、Asinh、Biexponential変換に対応"
            )
            
            max_events = st.number_input(
                "最大イベント数",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="パフォーマンス最適化のための最大イベント数設定（1,000～100,000）"
            )
            
            # Process file button
            if st.button("📊 ファイルを処理", type="primary"):
                with st.spinner("FCSファイルを統合処理パイプライン（utils/fcs_processor経由）で処理中..."):
                    process_fcs_file(uploaded_file, transformation, max_events)

def process_fcs_file(uploaded_file, transformation, max_events):
    """Process FCS file using utils.fcs_processor"""
    try:
        # Use the load_and_process_fcs function from utils
        processor, data, metadata = load_and_process_fcs(
            uploaded_file, 
            transformation, 
            max_events
        )
        
        if data is not None and processor is not None:
            st.session_state.processor = processor
            st.session_state.fcs_data = data
            st.session_state.fcs_metadata = metadata
            st.session_state.plotting_utils = PlottingUtils()
            st.session_state.transformation_applied = transformation
            st.session_state.max_events_used = max_events
            
            # Get library used info from processor
            library_used = getattr(processor, 'library_used', 'Unknown')
            st.sidebar.success(f"✅ 処理が完了しました")
            st.sidebar.info(f"使用ライブラリ: {library_used}")
            st.rerun()
        else:
            st.sidebar.error("❌ ファイルの処理に失敗しました")
            
    except Exception as e:
        st.sidebar.error(f"❌ 処理エラー: {str(e)}")
        with st.sidebar.expander("エラー詳細"):
            st.text(str(e))

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
            
            <h4>📚 対応ライブラリ：</h4>
            <ul>
                <li>FlowIO（推奨）</li>
                <li>FCSParser（フォールバック）</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_analysis_interface():
    """Display main analysis interface with 4-tab structure as per README"""
    
    data = st.session_state.fcs_data
    metadata = st.session_state.fcs_metadata
    processor = st.session_state.processor
    plotting_utils = st.session_state.plotting_utils
    
    # Display current processing status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総イベント数", f"{len(data):,}")
    with col2:
        st.metric("パラメータ数", len(data.columns))
    with col3:
        st.metric("変換方法", st.session_state.transformation_applied)
    with col4:
        st.metric("最大イベント数", f"{st.session_state.max_events_used:,}")
    
    # 4 tabs as specified in README
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 基本情報", "📈 可視化", "🎯 ゲーティング", "📋 統計解析"
    ])
    
    with tab1:
        display_basic_info_tab(data, metadata, processor)
    
    with tab2:
        display_visualization_tab(data, plotting_utils)
    
    with tab3:
        display_gating_tab(data)
    
    with tab4:
        display_statistics_tab(data, processor)

def display_basic_info_tab(data, metadata, processor):
    """📊 基本情報タブ - File info, experiment info, channel info, data preview"""
    
    st.markdown('<div class="section-header">📁 ファイル情報</div>', 
                unsafe_allow_html=True)
    
    # Get comprehensive file information using FCSProcessor
    file_info = processor.get_file_info()
    
    # Basic Information section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基本情報**")
        basic_info = [
            ["総イベント数", format_number(file_info.get('total_events', 'N/A'))],
            ["パラメータ数", file_info.get('total_parameters', 'N/A')],
            ["取得日時", file_info.get('acquisition_date', 'N/A')],
            ["使用機器情報", file_info.get('cytometer', 'N/A')],
            ["処理ライブラリ", getattr(processor, 'library_used', 'N/A')]
        ]
        basic_df = pd.DataFrame(basic_info, columns=["項目", "値"])
        st.dataframe(basic_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**実験情報**")
        exp_info = [
            ["実験名", file_info.get('experiment_name', 'N/A')],
            ["サンプルID", file_info.get('sample_id', 'N/A')],
            ["オペレーター", file_info.get('operator', 'N/A')],
            ["ソフトウェア", file_info.get('software', 'N/A')]
        ]
        exp_df = pd.DataFrame(exp_info, columns=["項目", "値"])
        st.dataframe(exp_df, hide_index=True, use_container_width=True)
    
    # Channel Information section
    st.markdown('<div class="section-header">📋 チャンネル情報</div>', 
                unsafe_allow_html=True)
    
    channel_info = processor.get_channel_info()
    if channel_info:
        # Display channel details
        channel_df = pd.DataFrame.from_dict(channel_info, orient='index')
        st.dataframe(channel_df, use_container_width=True)
    else:
        st.info("チャンネル詳細情報が利用できません")
    
    # Data Preview section  
    st.markdown('<div class="section-header">📊 データプレビュー</div>', 
                unsafe_allow_html=True)
    
    st.markdown("**最初の100行のデータ表示**")
    st.dataframe(data.head(100), height=300, use_container_width=True)

def format_number(value):
    """Format number with commas if numeric"""
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value

def display_visualization_tab(data, plotting_utils):
    """📈 可視化タブ - Multiple plot types with enhanced options"""
    
    st.markdown('<div class="section-header">📈 データ可視化</div>', 
                unsafe_allow_html=True)
    
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
    # Visualization type selection
    viz_type = st.selectbox(
        "可視化タイプ",
        ["ヒストグラム", "散布図", "密度プロット（等高線）"],
        help="多様なプロットタイプから選択"
    )
    
    if viz_type == "ヒストグラム":
        display_enhanced_histogram(data, channels, plotting_utils)
    elif viz_type == "散布図":
        display_enhanced_scatter_plot(data, channels, plotting_utils)
    elif viz_type == "密度プロット（等高線）":
        display_enhanced_density_plot(data, channels, plotting_utils)

def display_enhanced_histogram(data, channels, plotting_utils):
    """Enhanced histogram with README specifications"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**ヒストグラム設定**")
        selected_channel = st.selectbox("チャンネル選択", channels)
        bins = st.slider("カスタマイズ可能なビン数", 20, 200, 50)
        log_scale = st.checkbox("対数スケール表示オプション")
        
        # Display channel statistics
        st.markdown("**選択チャンネル統計**")
        channel_stats = {
            "平均値": data[selected_channel].mean(),
            "中央値": data[selected_channel].median(),
            "標準偏差": data[selected_channel].std(),
            "最小値": data[selected_channel].min(),
            "最大値": data[selected_channel].max()
        }
        for stat, value in channel_stats.items():
            st.metric(stat, f"{value:.2f}")
    
    with col2:
        # Use PlottingUtils if available
        if plotting_utils:
            fig = plotting_utils.create_histogram(
                data, 
                selected_channel, 
                title=f"{selected_channel} - 単一チャンネルの分布表示"
            )
        else:
            # Fallback to basic plotly
            fig = px.histogram(
                data, 
                x=selected_channel, 
                nbins=bins,
                title=f"{selected_channel} ヒストグラム"
            )
        
        if log_scale:
            fig.update_layout(yaxis_type="log")
        
        st.plotly_chart(fig, use_container_width=True)

def display_enhanced_scatter_plot(data, channels, plotting_utils):
    """Enhanced scatter plot with README specifications"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**散布図設定**")
        x_channel = st.selectbox("X軸チャンネル選択", channels, index=0)
        y_channel = st.selectbox("Y軸チャンネル選択", channels, index=1 if len(channels) > 1 else 0)
        alpha = st.slider("透明度調整", 0.1, 1.0, 0.6)
        sample_size = st.slider(
            "サンプリング（パフォーマンス最適化）", 
            1000, 
            min(100000, len(data)), 
            min(10000, len(data))
        )
    
    with col2:
        # Subsample for better performance
        plot_data = data.sample(n=min(sample_size, len(data)), random_state=42)
        
        if plotting_utils:
            fig = plotting_utils.create_scatter_plot(
                plot_data,
                x_channel,
                y_channel,
                title=f"{x_channel} vs {y_channel} - 2Dプロットでの相関解析"
            )
        else:
            # Fallback
            fig = px.scatter(
                plot_data,
                x=x_channel,
                y=y_channel,
                title=f"{x_channel} vs {y_channel}",
                opacity=alpha
            )
        
        st.plotly_chart(fig, use_container_width=True)

def display_enhanced_density_plot(data, channels, plotting_utils):
    """Enhanced density plot with README specifications"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**密度プロット設定**")
        x_channel = st.selectbox("X軸", channels, index=0, key="density_x")
        y_channel = st.selectbox("Y軸", channels, index=1 if len(channels) > 1 else 0, key="density_y")
        nbins = st.slider("密度計算ビン数", 20, 100, 50)
    
    with col2:
        if plotting_utils:
            fig = plotting_utils.create_density_plot(
                data,
                x_channel,
                y_channel,
                title=f"{x_channel} vs {y_channel} - 2Dヒストグラムベースの密度可視化"
            )
        else:
            # Fallback
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
        
def display_gating_tab(data):
    """🎯 ゲーティングタブ - Basic threshold gating + link to advanced"""
    
    st.markdown('<div class="section-header">🎯 ゲーティング解析</div>', 
                unsafe_allow_html=True)
    
    # Information about advanced gating
    st.info("💡 **基本閾値ゲーティング機能**を提供します。高度なゲーティング機能（矩形、ポリゴン、楕円ゲート等）については、advanced_gating.pyページをご利用ください。")
    
    # Link to advanced gating page
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔗 高度ゲーティングページへ", type="secondary", use_container_width=True):
            st.markdown("**高度ゲーティング機能（advanced_gating.py）で利用可能：**")
            st.markdown("- 矩形ゲート: 2次元での矩形領域選択")
            st.markdown("- ポリゴンゲート: 任意の多角形領域での選択") 
            st.markdown("- 楕円ゲート: 楕円形領域での選択")
            st.markdown("- 閾値ゲート: 単一パラメータでの閾値設定")
            st.markdown("- インタラクティブ可視化: 密度プロットでのリアルタイムゲート表示")
            st.markdown("- 詳細統計解析: ゲート内データの包括的統計情報")
    
    st.markdown("---")
    
    # Basic threshold gating implementation
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
    st.markdown("### 基本閾値ゲーティング")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**ゲート設定**")
        gate_channel = st.selectbox("ゲートチャンネル選択", channels)
        
        # Default threshold to median as specified in README
        default_threshold = float(data[gate_channel].median())
        threshold = st.number_input(
            "閾値設定（デフォルト：中央値）",
            value=default_threshold,
            step=float(data[gate_channel].std() / 10),
            format="%.2f"
        )
        
        gate_direction = st.selectbox(
            "ゲート方向",
            ["以上 (≥)", "より大きい (>)", "以下 (≤)", "より小さい (<)"]
        )
    
    with col2:
        st.markdown("**ゲート結果**")
        
        # Apply basic threshold gate based on direction
        if gate_direction == "以上 (≥)":
            gated_data = data[data[gate_channel] >= threshold]
        elif gate_direction == "より大きい (>)":
            gated_data = data[data[gate_channel] > threshold]
        elif gate_direction == "以下 (≤)":
            gated_data = data[data[gate_channel] <= threshold]
        else:  # より小さい (<)
            gated_data = data[data[gate_channel] < threshold]
        
        # Display gate statistics as specified in README
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("ゲート前イベント数", f"{len(data):,}")
            st.metric("ゲート後イベント数", f"{len(gated_data):,}")
        with col2b:
            gate_rate = (len(gated_data) / len(data)) * 100 if len(data) > 0 else 0
            st.metric("ゲート率", f"{gate_rate:.1f}%")
            st.metric("除外イベント数", f"{len(data) - len(gated_data):,}")
    
    # Visualization of gating
    st.markdown("### ゲート可視化")
    
    # Create histogram with gate line
    fig = px.histogram(data, x=gate_channel, nbins=50, 
                      title=f"{gate_channel} ヒストグラム with ゲート")
    
    # Add threshold line
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"閾値: {threshold:.2f}")
    
    # Color the gated region
    if gate_direction in ["以上 (≥)", "より大きい (>)"]:
        fig.add_vrect(x0=threshold, x1=data[gate_channel].max(),
                     fillcolor="green", opacity=0.2, annotation_text="ゲート領域")
    else:
        fig.add_vrect(x0=data[gate_channel].min(), x1=threshold,
                     fillcolor="green", opacity=0.2, annotation_text="ゲート領域")
    
    st.plotly_chart(fig, use_container_width=True)

def display_statistics_tab(data, processor):
    """📋 統計解析タブ - Comprehensive statistics and export"""
    
    st.markdown('<div class="section-header">📋 統計解析</div>', 
                unsafe_allow_html=True)
    
    # Get comprehensive statistics using FCSProcessor
    stats = processor.get_basic_stats()
    
    if stats:
        # Display all-channel statistics as specified in README
        st.markdown("### 全チャンネル統計")
        st.markdown("**平均値、中央値、標準偏差、最小値、最大値**")
        
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df = stats_df.round(3)
        
        # Format the statistics dataframe for better display
        formatted_stats_df = stats_df.copy()
        for col in formatted_stats_df.columns:
            if col in ['Mean', 'Median', 'Std', 'Min', 'Max']:
                formatted_stats_df[col] = formatted_stats_df[col].apply(lambda x: f"{x:,.3f}")
            elif col == 'Count':
                formatted_stats_df[col] = formatted_stats_df[col].apply(lambda x: f"{int(x):,}")
        
        st.dataframe(formatted_stats_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 統計サマリー")
        col1, col2, col3, col4 = st.columns(4)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        with col1:
            st.metric("解析チャンネル数", len(numeric_columns))
        with col2:
            total_events = len(data)
            st.metric("総イベント数", f"{total_events:,}")
        with col3:
            # Calculate overall data range
            overall_min = data[numeric_columns].min().min()
            overall_max = data[numeric_columns].max().max()
            st.metric("データ範囲", f"{overall_min:.1f} - {overall_max:.1f}")
        with col4:
            # Applied transformation
            st.metric("適用変換", st.session_state.transformation_applied)
        
        # Data Export section as specified in README
        st.markdown('<div class="section-header">💾 データエクスポート</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
if __name__ == "__main__":
    main()
