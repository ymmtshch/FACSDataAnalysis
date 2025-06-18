import streamlit as st
import pandas as pd
import numpy as np
import fcsparser
from utils.fcs_processor import FCSProcessor
from utils.plotting import PlottingUtils
import plotly.express as px
import plotly.graph_objects as go
from config import Config

def main():
    st.title("基本解析")
    st.write("FCSファイルの基本的な解析と可視化を行います。")
    
    # サイドバーでファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "FCSファイルを選択してください",
        type=['fcs'],
        help="標準的なFCS 2.0/3.0/3.1形式のファイルをサポートしています"
    )
    
    if uploaded_file is None:
        st.info("👈 サイドバーからFCSファイルをアップロードしてください")
        return
    
    try:
        # FCSファイルの読み込み (fcsparserを使用)
        with st.spinner("FCSファイルを読み込み中..."):
            # fcsparserでファイルを読み込み
            meta, data = fcsparser.parse(uploaded_file, meta_data_only=False, reformat_meta=True)
            
            # FCSProcessorインスタンスを作成
            processor = FCSProcessor()
            
            # データの前処理
            df_processed = processor.preprocess_data(data, meta)
            
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # ファイル情報の表示
        st.subheader("📊 ファイル情報")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("イベント数", f"{len(df_processed):,}")
        with col2:
            st.metric("パラメータ数", len(df_processed.columns))
        with col3:
            try:
                acquisition_date = meta.get('$DATE', 'N/A')
                st.metric("取得日", acquisition_date)
            except:
                st.metric("取得日", "N/A")
        
        # メタデータ情報
        if st.expander("📋 詳細メタデータ"):
            # 主要なメタデータを表示
            important_keys = ['$TOT', '$PAR', '$DATE', '$BTIM', '$ETIM', '$CYT', '$CYTNUM']
            meta_display = {}
            for key in important_keys:
                if key in meta:
                    meta_display[key] = meta[key]
            
            if meta_display:
                st.json(meta_display)
            else:
                st.write("メタデータが見つかりません")
        
        # チャンネル選択
        st.subheader("🎯 チャンネル選択")
        channels = list(df_processed.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            x_channel = st.selectbox(
                "X軸チャンネル",
                channels,
                index=0 if channels else 0
            )
        with col2:
            y_channel = st.selectbox(
                "Y軸チャンネル", 
                channels,
                index=1 if len(channels) > 1 else 0
            )
        
        # データ変換オプション
        st.subheader("🔧 データ変換")
        transform_options = ["なし", "Log10", "Asinh", "Biexponential"]
        
        col1, col2 = st.columns(2)
        with col1:
            x_transform = st.selectbox("X軸変換", transform_options)
        with col2:
            y_transform = st.selectbox("Y軸変換", transform_options)
        
        # データ変換の適用
        df_plot = df_processed.copy()
        
        if x_transform != "なし":
            df_plot[x_channel] = processor.apply_transform(
                df_plot[x_channel], x_transform.lower()
            )
        
        if y_transform != "なし":
            df_plot[y_channel] = processor.apply_transform(
                df_plot[y_channel], y_transform.lower()
            )
        
        # サンプリング設定
        st.subheader("⚡ 表示設定")
        col1, col2 = st.columns(2)
        
        with col1:
            max_points = st.slider(
                "表示ポイント数", 
                1000, 
                min(100000, len(df_processed)),
                10000,
                help="大量データの場合、表示速度向上のためサンプリングします"
            )
        
        with col2:
            plot_type = st.selectbox(
                "プロットタイプ",
                ["散布図", "密度プロット", "ヒストグラム"]
            )
        
        # データのサンプリング
        if len(df_plot) > max_points:
            df_plot_sampled = df_plot.sample(n=max_points, random_state=42)
            st.info(f"表示速度向上のため、{max_points:,}ポイントをサンプリング表示しています")
        else:
            df_plot_sampled = df_plot
        
        # プロット作成
        st.subheader("📈 データ可視化")
        
        plotting_utils = PlottingUtils()
        
        if plot_type == "散布図":
            fig = plotting_utils.create_scatter_plot(
                df_plot_sampled, 
                x_channel, 
                y_channel,
                title=f"{x_channel} vs {y_channel}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "密度プロット":
            fig = plotting_utils.create_density_plot(
                df_plot_sampled,
                x_channel,
                y_channel,
                title=f"{x_channel} vs {y_channel} 密度プロット"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "ヒストグラム":
            col1, col2 = st.columns(2)
            
            with col1:
                fig_x = plotting_utils.create_histogram(
                    df_plot_sampled,
                    x_channel,
                    title=f"{x_channel} ヒストグラム"
                )
                st.plotly_chart(fig_x, use_container_width=True)
            
            with col2:
                fig_y = plotting_utils.create_histogram(
                    df_plot_sampled,
                    y_channel,
                    title=f"{y_channel} ヒストグラム"
                )
                st.plotly_chart(fig_y, use_container_width=True)
        
        # 統計情報
        st.subheader("📊 統計情報")
        
        # 選択チャンネルの統計
        stats_data = {
            'チャンネル': [x_channel, y_channel],
            '平均': [
                df_plot_sampled[x_channel].mean(),
                df_plot_sampled[y_channel].mean()
            ],
            '中央値': [
                df_plot_sampled[x_channel].median(),
                df_plot_sampled[y_channel].median()
            ],
            '標準偏差': [
                df_plot_sampled[x_channel].std(),
                df_plot_sampled[y_channel].std()
            ],
            '最小値': [
                df_plot_sampled[x_channel].min(),
                df_plot_sampled[y_channel].min()
            ],
            '最大値': [
                df_plot_sampled[x_channel].max(),
                df_plot_sampled[y_channel].max()
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # データプレビュー
        if st.expander("🔍 データプレビュー"):
            st.dataframe(df_plot_sampled.head(1000), use_container_width=True)
        
        # データエクスポート
        st.subheader("💾 データエクスポート")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("統計情報をCSV出力"):
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="統計情報CSVダウンロード",
                    data=csv,
                    file_name=f"{uploaded_file.name.replace('.fcs', '')}_stats.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("表示データをCSV出力"):
                csv = df_plot_sampled.to_csv(index=False)
                st.download_button(
                    label="表示データCSVダウンロード",
                    data=csv,
                    file_name=f"{uploaded_file.name.replace('.fcs', '')}_data.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("ファイルが正しいFCS形式であることを確認してください。")
        
        if st.expander("エラー詳細"):
            st.exception(e)

if __name__ == "__main__":
    main()
