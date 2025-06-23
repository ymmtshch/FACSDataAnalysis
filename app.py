"""
FACS Data Analysis - Main Application
Streamlit-based web application for flow cytometry data analysis
Modified to exclude FlowKit dependency
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

# FCS file reading libraries (in order of preference)
try:
    import flowio
    FLOWIO_AVAILABLE = True
except ImportError:
    FLOWIO_AVAILABLE = False

try:
    import fcsparser
    FCSPARSER_AVAILABLE = True
except ImportError:
    FCSPARSER_AVAILABLE = False

# Check if any FCS library is available
if not FLOWIO_AVAILABLE and not FCSPARSER_AVAILABLE:
    st.error("❌ FCSファイル読み込みライブラリが見つかりません。flowioまたはfcsparserをインストールしてください。")
    st.stop()

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
    """Main application function"""
    
    # Title
    st.markdown('<div class="main-header">🔬 FACS Data Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Display available libraries
    library_status = []
    if FLOWIO_AVAILABLE:
        library_status.append("✅ FlowIO")
    if FCSPARSER_AVAILABLE:
        library_status.append("✅ FCSParser")
    
    st.sidebar.markdown(f"**利用可能ライブラリ:** {', '.join(library_status)}")
    
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
                    processor = SimpleFCSProcessor()
                    data, metadata, channels = processor.load_fcs_file(uploaded_file, max_events)
                    
                    if data is not None:
                        # Apply transformation
                        if transformation != "none":
                            data = processor.apply_transformation(data, transformation)
                        
                        st.session_state.processor = processor
                        st.session_state.fcs_data = data
                        st.session_state.fcs_metadata = metadata
                        
                        st.success(f"✅ 処理が完了しました（使用ライブラリ: {processor.library_used}）")
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
            
            <h4>📚 対応ライブラリ：</h4>
            <ul>
                <li>FlowIO（推奨）</li>
                <li>FCSParser（フォールバック）</li>
            </ul>
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
            ["総イベント数", f"{file_info.get('total_events', 'N/A'):,}" if isinstance(file_info.get('total_events'), (int, float)) else file_info.get('total_events', 'N/A')],
            ["パラメータ数", file_info.get('total_parameters', 'N/A')],
            ["取得日", file_info.get('acquisition_date', 'N/A')],
            ["取得時刻", file_info.get('acquisition_time', 'N/A')],
            ["サイトメーター", file_info.get('cytometer', 'N/A')],
            ["使用ライブラリ", processor.library_used]
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
    
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
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
    
    st.info("基本的なゲーティング機能です。高度なゲーティング機能については、advanced_gating.pyページをご確認ください。")
    
    # Simple threshold gating
    channels = list(data.select_dtypes(include=[np.number]).columns)
    
    if not channels:
        st.error("数値チャンネルが見つかりません")
        return
    
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
