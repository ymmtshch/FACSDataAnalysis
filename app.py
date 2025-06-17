import streamlit as st
import pandas as pd
import numpy as np
import flowkit as fk
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial import ConvexHull
import json
import base64

# ページ設定
st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
def init_session_state():
    """セッション状態を初期化"""
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'fcs_meta' not in st.session_state:
        st.session_state.fcs_meta = None
    if 'current_gates' not in st.session_state:
        st.session_state.current_gates = []
    if 'gate_stats' not in st.session_state:
        st.session_state.gate_stats = {}
    if 'selected_channels' not in st.session_state:
        st.session_state.selected_channels = []

# FCSファイル処理関数
@st.cache_data
def load_fcs_file(uploaded_file):
    """FCSファイルを読み込み、データフレームとメタデータを返す"""
    try:
        # FCSファイルを読み込み
        sample = fk.Sample(uploaded_file.getvalue())
        
        # データフレームに変換
        df = sample.as_dataframe(source='xform')
        
        # メタデータを取得
        metadata = {
            'filename': uploaded_file.name,
            'n_events': len(df),
            'channels': list(df.columns),
            'acquisition_date': sample.acquisition_date if hasattr(sample, 'acquisition_date') else 'Unknown',
            'compensation_matrix': sample.compensation_matrix if hasattr(sample, 'compensation_matrix') else None
        }
        
        return df, metadata
    except Exception as e:
        st.error(f"FCSファイルの読み込みに失敗しました: {str(e)}")
        return None, None

def create_histogram(data, channel, bins=50, show_stats=True):
    """ヒストグラムを作成"""
    fig = go.Figure()
    
    # ヒストグラム
    fig.add_trace(go.Histogram(
        x=data[channel],
        nbinsx=bins,
        name=channel,
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    if show_stats:
        # 統計値の計算
        mean_val = data[channel].mean()
        median_val = data[channel].median()
        std_val = data[channel].std()
        
        # 統計線を追加
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {median_val:.2f}")
        fig.add_vline(x=mean_val-std_val, line_dash="dot", line_color="orange", 
                     annotation_text=f"-1σ: {mean_val-std_val:.2f}")
        fig.add_vline(x=mean_val+std_val, line_dash="dot", line_color="orange", 
                     annotation_text=f"+1σ: {mean_val+std_val:.2f}")
    
    fig.update_layout(
        title=f"Histogram: {channel}",
        xaxis_title=channel,
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    return fig

def create_scatter_plot(data, x_channel, y_channel, color_channel=None, sample_size=None):
    """散布図を作成"""
    # データのサンプリング
    if sample_size and len(data) > sample_size:
        plot_data = data.sample(n=sample_size, random_state=42)
    else:
        plot_data = data
    
    fig = go.Figure()
    
    if color_channel and color_channel in plot_data.columns:
        # カラーマップ付き散布図
        fig.add_trace(go.Scatter(
            x=plot_data[x_channel],
            y=plot_data[y_channel],
            mode='markers',
            marker=dict(
                color=plot_data[color_channel],
                colorscale='Viridis',
                size=3,
                opacity=0.6,
                colorbar=dict(title=color_channel)
            ),
            name='Data points'
        ))
    else:
        # 通常の散布図
        fig.add_trace(go.Scatter(
            x=plot_data[x_channel],
            y=plot_data[y_channel],
            mode='markers',
            marker=dict(
                color='lightblue',
                size=3,
                opacity=0.6
            ),
            name='Data points'
        ))
    
    fig.update_layout(
        title=f"Scatter Plot: {x_channel} vs {y_channel}",
        xaxis_title=x_channel,
        yaxis_title=y_channel,
        height=500,
        showlegend=False
    )
    
    return fig

def create_contour_plot(data, x_channel, y_channel, bins=50):
    """等高線プロットを作成"""
    # 2Dヒストグラムの計算
    x_data = data[x_channel]
    y_data = data[y_channel]
    
    # 範囲の設定
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    # ヒストグラムの計算
    hist, x_edges, y_edges = np.histogram2d(
        x_data, y_data, bins=bins,
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    # ビンの中心点を計算
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    fig = go.Figure()
    
    # 等高線プロット
    fig.add_trace(go.Contour(
        x=x_centers,
        y=y_centers,
        z=hist.T,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Density")
    ))
    
    fig.update_layout(
        title=f"Contour Plot: {x_channel} vs {y_channel}",
        xaxis_title=x_channel,
        yaxis_title=y_channel,
        height=500
    )
    
    return fig

def calculate_statistics(data, channels):
    """選択されたチャンネルの統計情報を計算"""
    stats_data = []
    for channel in channels:
        if channel in data.columns:
            stats_data.append({
                'Channel': channel,
                'Count': len(data[channel]),
                'Mean': data[channel].mean(),
                'Median': data[channel].median(),
                'Std': data[channel].std(),
                'Min': data[channel].min(),
                'Max': data[channel].max(),
                'Q25': data[channel].quantile(0.25),
                'Q75': data[channel].quantile(0.75)
            })
    
    return pd.DataFrame(stats_data)

def export_to_csv(data, filename="facs_data"):
    """データをCSVとしてエクスポート"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    return href

# メイン関数
def main():
    init_session_state()
    
    # サイドバー
    st.sidebar.title("🔬 FACS Data Analysis")
    st.sidebar.markdown("---")
    
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "Upload FCS File",
        type=['fcs'],
        help="Select a Flow Cytometry Standard (.fcs) file"
    )
    
    if uploaded_file is not None:
        # ファイルを読み込み
        with st.spinner("Loading FCS file..."):
            fcs_data, fcs_meta = load_fcs_file(uploaded_file)
        
        if fcs_data is not None:
            st.session_state.fcs_data = fcs_data
            st.session_state.fcs_meta = fcs_meta
            
            # ファイル情報の表示
            st.sidebar.success(f"✅ File loaded: {fcs_meta['filename']}")
            st.sidebar.info(f"Events: {fcs_meta['n_events']:,}")
            st.sidebar.info(f"Channels: {len(fcs_meta['channels'])}")
    
    # メインコンテンツ
    if st.session_state.fcs_data is not None:
        # タブの作成
        tab1, tab2, tab3 = st.tabs(["📊 Basic Analysis", "🎯 Advanced Gating", "📈 Statistics"])
        
        with tab1:
            basic_analysis_page()
        
        with tab2:
            advanced_gating_page()
        
        with tab3:
            statistics_page()
    
    else:
        # ウェルカムページ
        st.title("🔬 FACS Data Analysis Platform")
        st.markdown("""
        ### Welcome to the FACS Data Analysis Platform
        
        This application provides comprehensive flow cytometry data analysis capabilities:
        
        **Features:**
        - 📁 FCS file loading and processing
        - 📊 Interactive data visualization
        - 🔍 Advanced gating capabilities
        - 📈 Statistical analysis and reporting
        - 💾 Data export functionality
        
        **Getting Started:**
        1. Upload your FCS file using the sidebar
        2. Explore your data in the Basic Analysis tab
        3. Apply gates in the Advanced Gating tab
        4. Review statistics in the Statistics tab
        
        ---
        *Please upload an FCS file to begin analysis.*
        """)

def basic_analysis_page():
    """基本解析ページ"""
    st.header("📊 Basic Analysis")
    
    data = st.session_state.fcs_data
    meta = st.session_state.fcs_meta
    
    # チャンネル選択
    channels = meta['channels']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Overview")
        
        # メタデータ表示
        st.markdown(f"""
        <div class="metric-card">
            <h4>File Information</h4>
            <p><strong>Filename:</strong> {meta['filename']}</p>
            <p><strong>Events:</strong> {meta['n_events']:,}</p>
            <p><strong>Channels:</strong> {len(channels)}</p>
            <p><strong>Date:</strong> {meta['acquisition_date']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # チャンネル選択
        st.subheader("Channel Selection")
        selected_channels = st.multiselect(
            "Select channels for analysis:",
            channels,
            default=channels[:3] if len(channels) >= 3 else channels
        )
        
        # プロット設定
        st.subheader("Plot Settings")
        sample_size = st.number_input(
            "Sample size for plots:",
            min_value=1000,
            max_value=len(data),
            value=min(10000, len(data)),
            step=1000
        )
    
    with col2:
        if selected_channels:
            # ヒストグラム
            st.subheader("Histograms")
            hist_channel = st.selectbox("Select channel for histogram:", selected_channels)
            
            if hist_channel:
                bins = st.slider("Number of bins:", 20, 100, 50)
                show_stats = st.checkbox("Show statistics", value=True)
                
                fig = create_histogram(data, hist_channel, bins, show_stats)
                st.plotly_chart(fig, use_container_width=True)
            
            # 散布図
            if len(selected_channels) >= 2:
                st.subheader("Scatter Plot")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    x_channel = st.selectbox("X-axis:", selected_channels, index=0)
                with col_y:
                    y_channel = st.selectbox("Y-axis:", selected_channels, index=1)
                
                color_channel = st.selectbox(
                    "Color by (optional):",
                    ["None"] + selected_channels,
                    index=0
                )
                
                color_ch = color_channel if color_channel != "None" else None
                
                if x_channel and y_channel:
                    fig = create_scatter_plot(data, x_channel, y_channel, color_ch, sample_size)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 等高線プロット
            if len(selected_channels) >= 2:
                st.subheader("Contour Plot")
                
                col_x2, col_y2 = st.columns(2)
                with col_x2:
                    x_contour = st.selectbox("X-axis (contour):", selected_channels, index=0, key="contour_x")
                with col_y2:
                    y_contour = st.selectbox("Y-axis (contour):", selected_channels, index=1, key="contour_y")
                
                if x_contour and y_contour:
                    contour_bins = st.slider("Contour resolution:", 20, 100, 50)
                    fig = create_contour_plot(data, x_contour, y_contour, contour_bins)
                    st.plotly_chart(fig, use_container_width=True)
        
        # データプレビュー
        st.subheader("Data Preview")
        if selected_channels:
            st.dataframe(data[selected_channels].head(100), use_container_width=True)
        else:
            st.dataframe(data.head(100), use_container_width=True)

def advanced_gating_page():
    """詳細ゲーティングページ"""
    st.header("🎯 Advanced Gating")
    st.info("🚧 Advanced gating functionality is under development. This will include interactive polygon gating, ellipse gating, and Boolean operations.")
    
    data = st.session_state.fcs_data
    channels = st.session_state.fcs_meta['channels']
    
    # 現在のゲート表示
    if st.session_state.current_gates:
        st.subheader("Current Gates")
        for i, gate in enumerate(st.session_state.current_gates):
            st.write(f"Gate {i+1}: {gate}")
    
    # 簡単なレンジゲーティング
    st.subheader("Range Gating")
    
    gate_channel = st.selectbox("Select channel for range gating:", channels)
    
    if gate_channel:
        col1, col2 = st.columns(2)
        
        with col1:
            min_val = st.number_input(
                f"Minimum {gate_channel}:",
                value=float(data[gate_channel].min()),
                min_value=float(data[gate_channel].min()),
                max_value=float(data[gate_channel].max())
            )
        
        with col2:
            max_val = st.number_input(
                f"Maximum {gate_channel}:",
                value=float(data[gate_channel].max()),
                min_value=float(data[gate_channel].min()),
                max_value=float(data[gate_channel].max())
            )
        
        if st.button("Apply Range Gate"):
            # ゲートの適用
            gated_data = data[(data[gate_channel] >= min_val) & (data[gate_channel] <= max_val)]
            
            # ゲート情報の保存
            gate_info = {
                'gate_id': f"range_gate_{len(st.session_state.current_gates)}",
                'gate_type': 'range',
                'channel': gate_channel,
                'min_value': min_val,
                'max_value': max_val,
                'n_events': len(gated_data),
                'percentage': len(gated_data) / len(data) * 100
            }
            
            st.session_state.current_gates.append(gate_info)
            st.success(f"Gate applied! {len(gated_data):,} events ({gate_info['percentage']:.1f}%) selected.")
            
            # ゲート結果の可視化
            fig = create_histogram(data, gate_channel, show_stats=False)
            
            # ゲート範囲をハイライト
            fig.add_vrect(
                x0=min_val, x1=max_val,
                fillcolor="red", opacity=0.2,
                annotation_text="Gated Region"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def statistics_page():
    """統計ページ"""
    st.header("📈 Statistics")
    
    data = st.session_state.fcs_data
    channels = st.session_state.fcs_meta['channels']
    
    # チャンネル選択
    selected_channels = st.multiselect(
        "Select channels for statistics:",
        channels,
        default=channels[:5] if len(channels) >= 5 else channels
    )
    
    if selected_channels:
        # 統計テーブル
        st.subheader("Descriptive Statistics")
        stats_df = calculate_statistics(data, selected_channels)
        st.dataframe(stats_df, use_container_width=True)
        
        # 相関行列
        if len(selected_channels) >= 2:
            st.subheader("Correlation Matrix")
            corr_matrix = data[selected_channels].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Channel Correlation Matrix",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # データエクスポート
        st.subheader("Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Selected Channels"):
                export_data = data[selected_channels]
                csv_link = export_to_csv(export_data, "selected_channels")
                st.markdown(csv_link, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export All Data"):
                csv_link = export_to_csv(data, "all_channels")
                st.markdown(csv_link, unsafe_allow_html=True)
        
        # ゲート統計
        if st.session_state.current_gates:
            st.subheader("Gate Statistics")
            gate_stats = []
            for gate in st.session_state.current_gates:
                gate_stats.append({
                    'Gate ID': gate['gate_id'],
                    'Type': gate['gate_type'],
                    'Channel': gate.get('channel', 'N/A'),
                    'Events': gate['n_events'],
                    'Percentage': f"{gate['percentage']:.1f}%"
                })
            
            gate_df = pd.DataFrame(gate_stats)
            st.dataframe(gate_df, use_container_width=True)

if __name__ == "__main__":
    main()
