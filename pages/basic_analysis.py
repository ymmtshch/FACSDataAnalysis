import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os

# 自動ライブラリ選択（推奨順序：fcsparser → flowio → flowkit）
FCS_LIBRARY = None
try:
    import fcsparser
    FCS_LIBRARY = "fcsparser"
except ImportError:
    try:
        import flowio
        FCS_LIBRARY = "flowio"
    except ImportError:
        try:
            import flowkit
            FCS_LIBRARY = "flowkit"
        except ImportError:
            st.error("FCS読み込みライブラリが見つかりません。fcsparser、flowio、または flowkit をインストールしてください。")
            st.stop()

from utils.fcs_processor import FCSProcessor
from utils.plotting import PlottingUtils
import plotly.express as px
import plotly.graph_objects as go
from config import Config

def read_fcs_file(file_path):
    """FCSファイルを読み込む関数（自動ライブラリ選択）"""
    try:
        if FCS_LIBRARY == "fcsparser":
            # fcsparserを使用（第一優先）
            try:
                meta, data = fcsparser.parse(file_path, meta_data_only=False, reformat_meta=True)
                
                # デバッグ情報の表示
                st.sidebar.write(f"デバッグ情報 (fcsparser):")
                st.sidebar.write(f"- データ型: {type(data)}")
                st.sidebar.write(f"- データ形状: {data.shape}")
                st.sidebar.write(f"- チャンネル数: {len(data.columns)}")
                
                return meta, data
                
            except AttributeError as e:
                if "newbyteorder" in str(e):
                    st.warning("fcsparserがNumPy 2.0と互換性がありません。flowioを使用します。")
                    # flowioにフォールバック
                    return read_fcs_file_fallback(file_path, "flowio")
                else:
                    raise e
                    
        elif FCS_LIBRARY == "flowio":
            return read_fcs_file_fallback(file_path, "flowio")
            
        elif FCS_LIBRARY == "flowkit":
            return read_fcs_file_fallback(file_path, "flowkit")
            
    except Exception as e:
        st.error(f"FCSファイルの読み込みエラー: {str(e)}")
        raise e

def read_fcs_file_fallback(file_path, library):
    """フォールバック用のFCSファイル読み込み"""
    if library == "flowio":
        import flowio
        
        # FlowIOを使用
        fcs_data = flowio.FlowData(file_path)
        
        # メタデータの取得
        meta = {}
        for key, value in fcs_data.text.items():
            meta[key] = value
        
        # データの取得とデバッグ情報
        events = fcs_data.events
        
        # デバッグ情報の表示
        st.sidebar.write(f"デバッグ情報 (flowio):")
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
        
    elif library == "flowkit":
        import flowkit
        
        # FlowKitを使用
        sample = flowkit.Sample(file_path)
        
        # メタデータの取得
        meta = sample.metadata
        
        # データの取得（自動的にDataFrameに変換）
        data = sample.as_dataframe()
        
        # デバッグ情報の表示
        st.sidebar.write(f"デバッグ情報 (flowkit):")
        st.sidebar.write(f"- データ型: {type(data)}")
        st.sidebar.write(f"- データ形状: {data.shape}")
        st.sidebar.write(f"- チャンネル数: {len(data.columns)}")
        
        return meta, data

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
                
                # FCSProcessorインスタンスを作成
                processor = FCSProcessor(file_data, uploaded_file.name)
                
                # データの前処理
                df_processed = processor.preprocess_data(data, meta)
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # ファイル情報の表示（3列メトリクス）
        st.subheader("📊 ファイル情報")  
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("イベント数", f"{len(df_processed):,}")
        with col2:
            st.metric("パラメータ数", len(df_processed.columns))
        with col3:
            try:
                # 各種メタデータキーを試す
                acquisition_date = (
                    meta.get('$DATE') or 
                    meta.get('date') or 
                    meta.get('$BTIM') or 
                    meta.get('btim') or 
                    'N/A'
                )
                st.metric("取得日", acquisition_date)
            except:
                st.metric("取得日", "N/A")
        
        # 詳細メタデータ表示（展開可能）
        if st.expander("📋 詳細メタデータ"):
            # 標準メタデータを表示
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
            
            # 全メタデータを表示（オプション）
            if st.checkbox("全メタデータを表示"):
                if isinstance(meta, dict):
                    st.json(dict(list(meta.items())[:20]))  # 最初の20項目のみ表示
                else:
                    st.write("メタデータの形式が不正です")
        
        # 高精度チャンネル選択
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
                index=0,
                help="X軸に表示するチャンネルを選択"
            )
        with col2:
            y_channel = st.selectbox(
                "Y軸チャンネル", 
                channels,
                index=1 if len(channels) > 1 else 0,
                help="Y軸に表示するチャンネルを選択"
            )
        
        # 個別変換設定
        st.subheader("🔧 データ変換")
        transform_options = ["なし", "Log10", "Asinh", "Biexponential"]
        
        col1, col2 = st.columns(2)
        with col1:
            x_transform = st.selectbox(
                "X軸変換", 
                transform_options,
                help="X軸データに適用する変換を選択"
            )
        with col2:
            y_transform = st.selectbox(
                "Y軸変換", 
                transform_options,
                help="Y軸データに適用する変換を選択"
            )
        
        # データ変換の適用
        df_plot = df_processed.copy()
        
        if x_transform != "なし":
            try:
                df_plot[x_channel] = processor.apply_transform(
                    df_plot[x_channel], x_transform.lower()
                )
            except Exception as e:
                st.warning(f"X軸変換エラー: {str(e)}。変換をスキップします。")
        
        if y_transform != "なし":
            try:
                df_plot[y_channel] = processor.apply_transform(
                    df_plot[y_channel], y_transform.lower()
                )
            except Exception as e:
                st.warning(f"Y軸変換エラー: {str(e)}。変換をスキップします。")
        
        # 表示設定・パフォーマンス最適化
        st.subheader("⚡ 表示設定")
        col1, col2 = st.columns(2)
        
        with col1:
            max_events = min(100000, len(df_processed))
            max_points = st.slider(
                "表示ポイント数", 
                1000, 
                max_events,
                min(10000, max_events),
                help="大量データの場合、表示速度向上のためサンプリングします"
            )
        
        with col2:
            plot_type = st.selectbox(
                "プロットタイプ",
                ["散布図", "密度プロット", "ヒストグラム"],
                help="データの可視化方法を選択"
            )
        
        # スマートサンプリング
        if len(df_plot) > max_points:
            df_plot_sampled = df_plot.sample(n=max_points, random_state=42)
            st.info(f"📊 表示速度向上のため、{max_points:,}ポイントをサンプリング表示しています（全{len(df_plot):,}イベント中）")
        else:
            df_plot_sampled = df_plot
        
        # 多様な可視化オプション
        st.subheader("📈 データ可視化")
        
        plotting_utils = PlottingUtils()
        
        try:
            if plot_type == "散布図":
                fig = plotting_utils.create_scatter_plot(
                    df_plot_sampled, 
                    x_channel, 
                    y_channel
                )
                fig.update_layout(
                    title=f"{x_channel} vs {y_channel}",
                    xaxis_title=f"{x_channel}" + (f" ({x_transform})" if x_transform != "なし" else ""),
                    yaxis_title=f"{y_channel}" + (f" ({y_transform})" if y_transform != "なし" else "")
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "密度プロット":
                fig = plotting_utils.create_density_plot(
                    df_plot_sampled,
                    x_channel,
                    y_channel
                )
                fig.update_layout(
                    title=f"{x_channel} vs {y_channel} 密度プロット",
                    xaxis_title=f"{x_channel}" + (f" ({x_transform})" if x_transform != "なし" else ""),
                    yaxis_title=f"{y_channel}" + (f" ({y_transform})" if y_transform != "なし" else "")
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "ヒストグラム":
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_x = plotting_utils.create_histogram(
                        df_plot_sampled,
                        x_channel
                    )
                    fig_x.update_layout(
                        title=f"{x_channel} ヒストグラム" + (f" ({x_transform})" if x_transform != "なし" else ""),
                        xaxis_title=f"{x_channel}",
                        yaxis_title="頻度"
                    )
                    st.plotly_chart(fig_x, use_container_width=True)
                
                with col2:
                    fig_y = plotting_utils.create_histogram(
                        df_plot_sampled,
                        y_channel
                    )
                    fig_y.update_layout(
                        title=f"{y_channel} ヒストグラム" + (f" ({y_transform})" if y_transform != "なし" else ""),
                        xaxis_title=f"{y_channel}",
                        yaxis_title="頻度"
                    )
                    st.plotly_chart(fig_y, use_container_width=True)
                    
        except Exception as plot_error:
            st.error(f"プロット作成エラー: {str(plot_error)}")
            if st.expander("プロットエラー詳細"):
                st.exception(plot_error)
        
        # リアルタイム統計解析
        st.subheader("📊 統計情報")
        
        # 選択チャンネルの統計（変換後）
        try:
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
            
        except Exception as stats_error:
            st.error(f"統計計算エラー: {str(stats_error)}")
        
        # データプレビュー
        if st.expander("🔍 データプレビュー"):
            st.write(f"表示中のデータサンプル（最大1000行）:")
            st.dataframe(df_plot_sampled.head(1000), use_container_width=True)
        
        # 改良されたデータエクスポート
        st.subheader("💾 データエクスポート")
        
        # 自動ファイル命名
        base_filename = uploaded_file.name.replace('.fcs', '')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**統計情報エクスポート**")
            if st.button("統計情報をCSV準備", key="stats_prepare"):
                try:
                    csv_stats = stats_df.to_csv(index=False)
                    st.download_button(
                        label="📊 統計情報CSVダウンロード",
                        data=csv_stats,
                        file_name=f"{base_filename}_stats.csv",
                        mime="text/csv",
                        key="stats_download"
                    )
                    st.success("統計情報CSV準備完了！")
                except Exception as e:
                    st.error(f"統計情報CSV準備エラー: {str(e)}")
        
        with col2:
            st.write("**表示データエクスポート**")
            if st.button("表示データをCSV準備", key="data_prepare"):
                try:
                    csv_data = df_plot_sampled.to_csv(index=False)
                    st.download_button(
                        label="📈 表示データCSVダウンロード",
                        data=csv_data,
                        file_name=f"{base_filename}_data.csv",
                        mime="text/csv",
                        key="data_download"
                    )
                    st.success("表示データCSV準備完了！")
                except Exception as e:
                    st.error(f"表示データCSV準備エラー: {str(e)}")

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
        # ライブラリ固有エラーハンドリング
        if FCS_LIBRARY == "fcsparser" and "newbyteorder" in str(e):
            st.error("**NumPy 2.0互換性問題**")
            st.info("💡 以下のいずれかの解決方法を試してください：")
            st.code("pip install flowio  # FlowIOを使用")
            st.code("pip install numpy==1.24.3  # NumPyダウングレード")
        elif "array.array" in str(e):
            st.error("**データ変換エラー**")
            st.info("💡 代替ライブラリの使用を推奨します：")
            st.code("pip install fcsparser  # fcsparserを使用")
        
        # 包括的エラーハンドリング
        if st.expander("🔧 エラー詳細とトラブルシューティング"):
            st.write("**エラー詳細:**")
            st.exception(e)
            
            st.write("**推奨解決方法:**")
            st.write("1. 別のFCSファイルでテストしてください")
            st.write("2. FCSファイルが破損していないか確認してください")
            st.write("3. 他のFCS読み込みライブラリをインストールしてください")
            if FCS_LIBRARY == "fcsparser":
                st.code("pip install flowio")
            elif FCS_LIBRARY == "flowio":
                st.code("pip install fcsparser")
            st.write("4. アプリケーションを再起動してください")

if __name__ == "__main__":
    main()
