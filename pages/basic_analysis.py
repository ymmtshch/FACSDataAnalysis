import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, Range1d, BasicTicker, PrintfTickFormatter
from bokeh.palettes import Viridis256, Blues9
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.transform import linear_cmap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_basic_analysis():
    """基本解析ページのメイン関数"""
    st.title("📊 基本解析")
    
    # セッション状態からデータを取得
    if 'fcs_data' not in st.session_state or st.session_state.fcs_data is None:
        st.warning("先にFCSファイルをアップロードしてください。")
        if st.button("ホームに戻る"):
            st.switch_page("app.py")
        return
    
    df = st.session_state.fcs_data
    meta = st.session_state.get('fcs_meta', {})
    
    # サイドバーで解析設定
    st.sidebar.header("解析設定")
    
    # パラメータ選択
    available_params = [col for col in df.columns if col not in ['Time', 'Event_length']]
    
    # ヒストグラム設定
    st.sidebar.subheader("📈 ヒストグラム")
    hist_param = st.sidebar.selectbox(
        "パラメータ選択",
        available_params,
        key="hist_param"
    )
    
    hist_bins = st.sidebar.slider(
        "ビン数",
        min_value=20,
        max_value=200,
        value=50,
        key="hist_bins"
    )
    
    show_stats = st.sidebar.checkbox("統計線表示", value=True, key="show_stats")
    
    # 散布図設定
    st.sidebar.subheader("🔍 散布図")
    x_param = st.sidebar.selectbox(
        "X軸パラメータ",
        available_params,
        key="scatter_x"
    )
    
    y_param = st.sidebar.selectbox(
        "Y軸パラメータ",
        available_params,
        index=1 if len(available_params) > 1 else 0,
        key="scatter_y"
    )
    
    # サンプリング設定
    max_points = st.sidebar.number_input(
        "最大表示点数",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        key="max_points"
    )
    
    # 等高線設定
    st.sidebar.subheader("🌄 等高線プロット")
    contour_levels = st.sidebar.slider(
        "等高線レベル数",
        min_value=5,
        max_value=20,
        value=10,
        key="contour_levels"
    )
    
    show_contour = st.sidebar.checkbox("等高線表示", value=True, key="show_contour")
    
    # メインエリア
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📋 データ情報")
        st.write(f"**総イベント数**: {len(df):,}")
        st.write(f"**パラメータ数**: {len(available_params)}")
        
        if meta:
            st.write("**メタデータ**")
            for key, value in meta.items():
                if isinstance(value, (str, int, float)):
                    st.write(f"- {key}: {value}")
    
    with col1:
        # タブで表示切り替え
        tab1, tab2, tab3 = st.tabs(["ヒストグラム", "散布図", "等高線プロット"])
        
        with tab1:
            create_histogram(df, hist_param, hist_bins, show_stats)
        
        with tab2:
            create_scatter_plot(df, x_param, y_param, max_points)
        
        with tab3:
            if show_contour:
                create_contour_plot(df, x_param, y_param, contour_levels, max_points)
            else:
                st.info("等高線表示がオフになっています。")
    
    # 統計情報
    st.subheader("📊 統計情報")
    show_statistics(df, [hist_param, x_param, y_param])
    
    # エクスポート機能
    st.subheader("💾 データエクスポート")
    export_data(df)

def create_histogram(df, param, bins, show_stats):
    """ヒストグラム作成"""
    try:
        data = df[param].dropna()
        
        if len(data) == 0:
            st.warning(f"パラメータ '{param}' にデータがありません。")
            return
        
        # Plotlyでヒストグラム作成
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name="データ",
            opacity=0.7,
            marker_color='lightblue',
            marker_line_color='black',
            marker_line_width=1
        ))
        
        # 統計線追加
        if show_stats:
            mean_val = data.mean()
            median_val = data.median()
            
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"平均: {mean_val:.2f}",
                annotation_position="top"
            )
            
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"中央値: {median_val:.2f}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=f"{param} ヒストグラム",
            xaxis_title=param,
            yaxis_title="頻度",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"ヒストグラム作成エラー: {str(e)}")

def create_scatter_plot(df, x_param, y_param, max_points):
    """散布図作成"""
    try:
        # データサンプリング
        if len(df) > max_points:
            sample_df = df.sample(n=max_points, random_state=42)
            st.info(f"データが多いため、{max_points:,}点をランダムサンプリングして表示します。")
        else:
            sample_df = df
        
        # NaN値を除去
        plot_data = sample_df[[x_param, y_param]].dropna()
        
        if len(plot_data) == 0:
            st.warning("選択したパラメータにデータがありません。")
            return
        
        # Plotlyで散布図作成
        fig = px.scatter(
            plot_data,
            x=x_param,
            y=y_param,
            title=f"{x_param} vs {y_param}",
            opacity=0.6,
            height=500
        )
        
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            xaxis_title=x_param,
            yaxis_title=y_param
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"散布図作成エラー: {str(e)}")

def create_contour_plot(df, x_param, y_param, levels, max_points):
    """等高線プロット作成"""
    try:
        # データサンプリング
        if len(df) > max_points:
            sample_df = df.sample(n=max_points, random_state=42)
        else:
            sample_df = df
        
        # NaN値を除去
        plot_data = sample_df[[x_param, y_param]].dropna()
        
        if len(plot_data) < 100:
            st.warning("等高線プロットには最低100点のデータが必要です。")
            return
        
        x_data = plot_data[x_param].values
        y_data = plot_data[y_param].values
        
        # 2D密度計算
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        # グリッド作成
        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        
        # 2Dヒストグラム作成
        H, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=[x_grid, y_grid])
        
        # グリッドの中心座標
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # Plotlyで等高線プロット作成
        fig = go.Figure()
        
        # 散布図（背景）
        fig.add_trace(go.Scatter(
            x=x_data[::max(1, len(x_data)//1000)],  # さらにサンプリング
            y=y_data[::max(1, len(y_data)//1000)],
            mode='markers',
            marker=dict(size=2, opacity=0.3, color='lightgray'),
            name='データ点',
            showlegend=False
        ))
        
        # 等高線
        fig.add_trace(go.Contour(
            z=H.T,
            x=x_centers,
            y=y_centers,
            contours=dict(
                start=H.min(),
                end=H.max(),
                size=(H.max() - H.min()) / levels,
                coloring='lines'
            ),
            line=dict(width=2),
            name='密度等高線',
            showscale=True,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f"{x_param} vs {y_param} 密度等高線",
            xaxis_title=x_param,
            yaxis_title=y_param,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"等高線プロット作成エラー: {str(e)}")

def show_statistics(df, params):
    """統計情報表示"""
    try:
        stats_data = []
        
        for param in set(params):  # 重複を除去
            if param in df.columns:
                data = df[param].dropna()
                if len(data) > 0:
                    stats_data.append({
                        'パラメータ': param,
                        '平均': f"{data.mean():.2f}",
                        '中央値': f"{data.median():.2f}",
                        '標準偏差': f"{data.std():.2f}",
                        '最小値': f"{data.min():.2f}",
                        '最大値': f"{data.max():.2f}",
                        'データ数': f"{len(data):,}"
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("統計情報を計算できませんでした。")
            
    except Exception as e:
        st.error(f"統計情報計算エラー: {str(e)}")

def export_data(df):
    """データエクスポート機能"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV エクスポート
        if st.button("📄 CSV でエクスポート"):
            try:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="CSV ダウンロード",
                    data=csv_data,
                    file_name="facs_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"CSV エクスポートエラー: {str(e)}")
    
    with col2:
        # 統計情報エクスポート
        if st.button("📊 統計情報エクスポート"):
            try:
                stats = df.describe()
                stats_csv = stats.to_csv()
                st.download_button(
                    label="統計情報ダウンロード",
                    data=stats_csv,
                    file_name="facs_statistics.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"統計情報エクスポートエラー: {str(e)}")
    
    with col3:
        # データプレビュー
        if st.button("👀 データプレビュー"):
            st.subheader("データプレビュー（最初の100行）")
            st.dataframe(df.head(100))

# ページ実行
if __name__ == "__main__":
    show_basic_analysis()
