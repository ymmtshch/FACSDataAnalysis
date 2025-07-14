import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import fcsparser
import plotly.express as px
import plotly.graph_objects as go
from utils.fcs_processor import FCSProcessor
from utils.plotting import PlottingUtils

def read_fcs_file(file_path):
    """FCSファイルを読み込む（fcsparserを使用）"""
    try:
        meta, data = fcsparser.parse(file_path, meta_data_only=False, reformat_meta=True)
        return meta, data
    except Exception as e:
        st.error(f"FCSファイルの読み込みエラー: {str(e)}")
        raise e

def main():
    st.title("基本解析")
    st.write("FCSファイルの詳細解析と可視化を行います。")
    
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "FCSファイルを選択してください",
        type=['fcs'],
        help="FCS 2.0/3.0/3.1形式のファイルをサポート"
    )
    
    if uploaded_file is None:
        st.info("👈 サイドバーからFCSファイルをアップロードしてください")
        return
    
    try:
        # FCSファイルの読み込み
        with st.spinner("FCSファイルを読み込み中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                meta, data = read_fcs_file(tmp_file_path)
                
                # FCSProcessorでデータ処理
                uploaded_file.seek(0)
                file_data = uploaded_file.read()
                processor = FCSProcessor(file_data, uploaded_file.name)
                df_processed = processor.preprocess_data(data, meta)
                
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # ファイル情報表示
        st.subheader("📊 ファイル情報")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("イベント数", f"{len(df_processed):,}")
        with col2:
            st.metric("パラメータ数", len(df_processed.columns))
        with col3:
            acquisition_date = meta.get('$DATE', meta.get('date', 'N/A'))
            st.metric("取得日", acquisition_date)
        
        # 詳細メタデータ表示
        if st.expander("📋 詳細メタデータ"):
            important_keys = ['$TOT', '$PAR', '$DATE', '$BTIM', '$ETIM', '$CYT', '$CYTNUM']
            meta_display = {key: meta[key] for key in important_keys if key in meta}
            
            if meta_display:
                st.json(meta_display)
            else:
                st.write("主要なメタデータが見つかりません")
        
        # チャンネル選択
        st.subheader("🎯 チャンネル選択")
        channels = list(df_processed.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            x_channel = st.selectbox("X軸チャンネル", channels, index=0)
        with col2:
            y_channel = st.selectbox("Y軸チャンネル", channels, index=1 if len(channels) > 1 else 0)
        
        # 個別変換設定
        st.subheader("🔧 データ変換")
        transform_options = ["なし", "Log10", "Asinh", "Biexponential"]
        
        col1, col2 = st.columns(2)
        with col1:
            x_transform = st.selectbox("X軸変換", transform_options)
        with col2:
            y_transform = st.selectbox("Y軸変換", transform_options)
        
        # データ変換適用
        df_plot = df_processed.copy()
        
        if x_transform != "なし":
            df_plot[x_channel] = processor.apply_transform(
                df_plot[x_channel], x_transform.lower()
            )
        
        if y_transform != "なし":
            df_plot[y_channel] = processor.apply_transform(
                df_plot[y_channel], y_transform.lower()
            )
        
        # 表示設定
        st.subheader("⚡ 表示設定")
        max_events = min(100000, len(df_processed))
        max_points = st.slider("表示ポイント数", 1000, max_events, min(10000, max_events))
        
        plot_type = st.selectbox("プロットタイプ", ["散布図", "密度プロット", "ヒストグラム"])
        
        # データサンプリング
        if len(df_plot) > max_points:
            df_plot_sampled = df_plot.sample(n=max_points, random_state=42)
            st.info(f"📊 {max_points:,}ポイントをサンプリング表示（全{len(df_plot):,}イベント中）")
        else:
            df_plot_sampled = df_plot
        
        # 可視化
        st.subheader("📈 データ可視化")
        plotting_utils = PlottingUtils()
        
        try:
            if plot_type == "散布図":
                fig = plotting_utils.create_scatter_plot(df_plot_sampled, x_channel, y_channel)
                
            elif plot_type == "密度プロット":
                fig = plotting_utils.create_density_plot(df_plot_sampled, x_channel, y_channel)
                
            elif plot_type == "ヒストグラム":
                col1, col2 = st.columns(2)
                with col1:
                    fig_x = plotting_utils.create_histogram(df_plot_sampled, x_channel)
                    fig_x.update_layout(title=f"{x_channel} ヒストグラム")
                    st.plotly_chart(fig_x, use_container_width=True)
                
                with col2:
                    fig_y = plotting_utils.create_histogram(df_plot_sampled, y_channel)
                    fig_y.update_layout(title=f"{y_channel} ヒストグラム")
                    st.plotly_chart(fig_y, use_container_width=True)
                
                # ヒストグラムの場合は統計情報のみ表示
                show_stats_only = True
            
            if plot_type != "ヒストグラム":
                # タイトル設定
                x_title = f"{x_channel}" + (f" ({x_transform})" if x_transform != "なし" else "")
                y_title = f"{y_channel}" + (f" ({y_transform})" if y_transform != "なし" else "")
                
                fig.update_layout(
                    title=f"{x_channel} vs {y_channel}",
                    xaxis_title=x_title,
                    yaxis_title=y_title
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"プロット作成エラー: {str(e)}")
        
        # リアルタイム統計解析
        st.subheader("📊 統計情報")
        
        stats_data = {
            'チャンネル': [
                f"{x_channel}" + (f" ({x_transform})" if x_transform != "なし" else ""),
                f"{y_channel}" + (f" ({y_transform})" if y_transform != "なし" else "")
            ],
            '平均': [
                f"{df_plot_sampled[x_channel].mean():.2f}",
                f"{df_plot_sampled[y_channel].mean():.2f}"
            ],
            '中央値': [
                f"{df_plot_sampled[x_channel].median():.2f}",
                f"{df_plot_sampled[y_channel].median():.2f}"
            ],
            '標準偏差': [
                f"{df_plot_sampled[x_channel].std():.2f}",
                f"{df_plot_sampled[y_channel].std():.2f}"
            ],
            '最小値': [
                f"{df_plot_sampled[x_channel].min():.2f}",
                f"{df_plot_sampled[y_channel].min():.2f}"
            ],
            '最大値': [
                f"{df_plot_sampled[x_channel].max():.2f}",
                f"{df_plot_sampled[y_channel].max():.2f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # 自動ファイル命名によるデータエクスポート
        st.subheader("💾 データエクスポート")
        base_filename = uploaded_file.name.replace('.fcs', '')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**統計情報エクスポート**")
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📊 統計情報CSVダウンロード",
                data=csv_stats,
                file_name=f"{base_filename}_stats.csv",
                mime="text/csv"
            )
        
        with col2:
            st.write("**表示データエクスポート**")
            csv_data = df_plot_sampled.to_csv(index=False)
            st.download_button(
                label="📈 表示データCSVダウンロード",
                data=csv_data,
                file_name=f"{base_filename}_data.csv",
                mime="text/csv"
            )
        
        # データプレビュー
        if st.expander("🔍 データプレビュー"):
            st.dataframe(df_plot_sampled.head(1000), use_container_width=True)

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
        if st.expander("🔧 エラー詳細"):
            st.exception(e)
            st.write("**解決方法:**")
            st.write("1. 別のFCSファイルでテストしてください")
            st.write("2. FCSファイルが破損していないか確認してください")
            st.write("3. アプリケーションを再起動してください")

if __name__ == "__main__":
    main()
