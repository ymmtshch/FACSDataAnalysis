import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os

# FlowIOまたはfcsparserが利用可能かチェック（FlowKitは除外）
try:
    import flowio
    FCS_LIBRARY = "flowio"
except ImportError:
    try:
        import fcsparser
        FCS_LIBRARY = "fcsparser"
    except ImportError:
        st.error("FCS読み込みライブラリが見つかりません。flowio または fcsparser をインストールしてください。")
        st.stop()

from utils.fcs_processor import FCSProcessor
from utils.plotting import PlottingUtils
import plotly.express as px
import plotly.graph_objects as go
from config import Config

def read_fcs_file(file_path):
    """FCSファイルを読み込む関数（flowioまたはfcsparserを使用）"""
    try:
        if FCS_LIBRARY == "flowio":
            # FlowIOを使用
            fcs_data = flowio.FlowData(file_path)
            
            # メタデータの取得
            meta = {}
            for key, value in fcs_data.text.items():
                meta[key] = value
            
            # データの取得とデバッグ情報
            events = fcs_data.events
            
            # デバッグ情報の表示
            st.sidebar.write(f"デバッグ情報:")
            st.sidebar.write(f"- イベントタイプ: {type(events)}")
            st.sidebar.write(f"- チャンネル数: {fcs_data.channel_count}")
            
            # array.array を NumPy配列に変換
            try:
                if hasattr(events, 'dtype'):
                    # 既にNumPy配列の場合
                    events_array = events
                    st.sidebar.write(f"- データ形状: {events_array.shape}")
                else:
                    # array.array の場合、NumPy配列に変換
                    events_array = np.array(events)
                    st.sidebar.write(f"- 変換後データ形状: {events_array.shape}")
                
                # 2次元配列に変換（必要に応じて）
                if events_array.ndim == 1:
                    # 1次元配列の場合、チャンネル数で分割
                    total_events = len(events_array) // fcs_data.channel_count
                    events_array = events_array.reshape(total_events, fcs_data.channel_count)
                    st.sidebar.write(f"- 再整形後: {events_array.shape}")
                
            except Exception as conversion_error:
                st.error(f"データ変換エラー: {str(conversion_error)}")
                st.write("FlowIOの代替方法を試します...")
                
                # 代替方法: データを手動で処理
                try:
                    # eventsがlistの場合
                    if isinstance(events, (list, tuple)):
                        events_array = np.array(events)
                    else:
                        # array.arrayの場合、tolist()してからnumpy配列に変換
                        events_list = events.tolist() if hasattr(events, 'tolist') else list(events)
                        events_array = np.array(events_list)
                    
                    # 2次元配列に変換
                    if events_array.ndim == 1:
                        total_events = len(events_array) // fcs_data.channel_count
                        events_array = events_array.reshape(total_events, fcs_data.channel_count)
                    
                    st.sidebar.write(f"- 代替方法成功: {events_array.shape}")
                    
                except Exception as alt_error:
                    raise Exception(f"データ変換に失敗しました: {str(alt_error)}")
            
            # チャンネル名の取得（FlowIOの正しい方法）
            channel_names = []
            for i in range(fcs_data.channel_count):
                # まずPnN (チャンネル名)を試す
                channel_name_key = f'$P{i+1}N'
                channel_short_key = f'$P{i+1}S'  # ショート名
                
                if channel_name_key in fcs_data.text and fcs_data.text[channel_name_key].strip():
                    channel_names.append(fcs_data.text[channel_name_key].strip())
                elif channel_short_key in fcs_data.text and fcs_data.text[channel_short_key].strip():
                    channel_names.append(fcs_data.text[channel_short_key].strip())
                else:
                    # どちらもない場合はデフォルト名を使用
                    channel_names.append(f'Channel_{i+1}')
            
            # チャンネル名が重複している場合の処理
            seen = set()
            unique_channel_names = []
            for name in channel_names:
                if name in seen:
                    counter = 2
                    new_name = f"{name}_{counter}"
                    while new_name in seen:
                        counter += 1
                        new_name = f"{name}_{counter}"
                    unique_channel_names.append(new_name)
                    seen.add(new_name)
                else:
                    unique_channel_names.append(name)
                    seen.add(name)
            
            # DataFrameに変換
            data = pd.DataFrame(events_array, columns=unique_channel_names)
            
            return meta, data
                    
        elif FCS_LIBRARY == "fcsparser":
            # fcsparserを使用（NumPy 2.0互換性の問題がある可能性）
            try:
                # NumPy 1.xに一時的にダウングレードが必要かもしれません
                meta, data = fcsparser.parse(file_path, meta_data_only=False, reformat_meta=True)
                return meta, data
            except AttributeError as e:
                if "newbyteorder" in str(e):
                    st.error("fcsparserがNumPy 2.0と互換性がありません。以下のコマンドでFlowIOをインストールしてください：")
                    st.code("pip install flowio")
                    st.stop()
                else:
                    raise e
                    
    except Exception as e:
        st.error(f"FCSファイルの読み込みエラー: {str(e)}")
        raise e

def main():
    st.title("基本解析")
    st.write("FCSファイルの基本的な解析と可視化を行います。")
    
    # 使用中のライブラリを表示
    st.sidebar.info(f"FCS読み込みライブラリ: {FCS_LIBRARY}")
    
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
        # FCSファイルの読み込み
        with st.spinner("FCSファイルを読み込み中..."):
            # UploadedFileを一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # FCSファイルを読み込み
                meta, data = read_fcs_file(tmp_file_path)
                
                # デバッグ情報
                st.sidebar.write(f"チャンネル数: {len(data.columns)}")
                if st.sidebar.checkbox("チャンネル名を表示"):
                    st.sidebar.write("チャンネル名:")
                    for i, col in enumerate(data.columns):
                        st.sidebar.write(f"{i+1}: {col}")
                
                # DataFrameに変換（必要に応じて）
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                
                # ファイルデータを再読み込みしてFCSProcessorに渡す
                uploaded_file.seek(0)  # ファイルポインタをリセット
                file_data = uploaded_file.read()
                
                # FCSProcessorインスタンスを作成（正しい引数で）
                processor = FCSProcessor(file_data, uploaded_file.name)
                
                # データの前処理
                df_processed = processor.preprocess_data(data, meta)
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
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
                acquisition_date = meta.get('$DATE', meta.get('date', 'N/A'))
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
            
            # FlowKit互換（小文字キー）も確認
            if not meta_display:
                flowkit_keys = ['tot', 'par', 'date', 'btim', 'etim', 'cyt', 'cytnum']
                for key in flowkit_keys:
                    if key in meta:
                        meta_display[key] = meta[key]
            
            if meta_display:
                st.json(meta_display)
            else:
                st.write("主要なメタデータが見つかりません")
                # 全メタデータを表示
                if st.checkbox("全メタデータを表示"):
                    st.json(dict(list(meta.items())[:20]))  # 最初の20項目のみ表示
        
        # チャンネル選択
        st.subheader("🎯 チャンネル選択")
        channels = list(df_processed.columns)
        
        if not channels:
            st.error("データにチャンネルが見つかりません")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            x_channel = st.selectbox(
                "X軸チャンネル",
                channels,
                index=0
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
        
        # ライブラリの推奨事項を表示
        if FCS_LIBRARY == "fcsparser":
            st.info("💡 より安定したFCS読み込みのため、以下のコマンドでFlowIOをインストールすることをお勧めします：")
            st.code("pip install flowio")
        
        if st.expander("エラー詳細"):
            st.exception(e)

if __name__ == "__main__":
    main()
