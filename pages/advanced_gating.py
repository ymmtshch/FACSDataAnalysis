import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
from config import Config

# 自動ライブラリ選択による最適化された読み込み
def load_fcs_file(uploaded_file):
    """FCSファイルを自動ライブラリ選択で読み込む"""
    try:
        # 第一優先: fcsparser
        try:
            import fcsparser
            meta, data = fcsparser.parse(uploaded_file, meta_data_only=False, reformat_meta=True)
            return meta, data, "fcsparser"
        except Exception as e:
            st.warning(f"fcsparser での読み込みに失敗: {str(e)}")
            
        # 第二優先: flowio
        try:
            import flowio
            fcs_data = flowio.FlowData(uploaded_file)
            
            # メタデータの取得
            meta = {}
            for key, value in fcs_data.text.items():
                meta[key] = value
            
            # イベントデータの取得
            events = fcs_data.events
            
            # チャンネル名の取得
            channels = []
            for i in range(1, fcs_data.channel_count + 1):
                channel_name = fcs_data.text.get(f'$P{i}N', f'Channel_{i}')
                channels.append(channel_name)
            
            # DataFrameに変換
            data = pd.DataFrame(events, columns=channels)
            
            return meta, data, "flowio"
        except Exception as e:
            st.warning(f"flowio での読み込みに失敗: {str(e)}")
            
        # 第三優先: flowkit
        try:
            import flowkit
            fcs_data = flowkit.Sample(uploaded_file)
            
            # メタデータの取得
            meta = fcs_data.metadata
            
            # イベントデータの取得
            data = fcs_data.as_dataframe()
            
            return meta, data, "flowkit"
        except Exception as e:
            st.error(f"flowkit での読み込みにも失敗: {str(e)}")
            raise e
            
    except Exception as e:
        st.error(f"すべてのFCS読み込みライブラリでの読み込みに失敗しました: {str(e)}")
        raise e

def main():
    st.title("高度ゲーティング解析")
    st.write("インタラクティブなゲーティングによる詳細解析を行います。")
    
    # セッション状態の初期化
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'gates' not in st.session_state:
        st.session_state.gates = []
    if 'gated_data' not in st.session_state:
        st.session_state.gated_data = None
    if 'used_library' not in st.session_state:
        st.session_state.used_library = None
    
    # サイドバーでファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "FCSファイルを選択してください",
        type=['fcs'],
        help="標準的なFCS 2.0/3.0/3.1形式のファイルをサポートしています"
    )
    
    # サイドバーにライブラリ情報を表示
    if st.session_state.used_library:
        st.sidebar.info(f"📚 使用ライブラリ: {st.session_state.used_library}")
    
    if uploaded_file is None:
        st.info("👈 サイドバーからFCSファイルをアップロードしてください")
        return
    
    try:
        # FCSファイルの読み込み
        if st.session_state.fcs_data is None or st.session_state.get('current_file') != uploaded_file.name:
            with st.spinner("FCSファイルを読み込み中..."):
                # 自動ライブラリ選択による読み込み
                meta, data, used_library = load_fcs_file(uploaded_file)
                
                # utils.fcs_processor を使用
                from utils.fcs_processor import FCSProcessor
                processor = FCSProcessor()
                
                # データの前処理
                df_processed = processor.preprocess_data(data, meta)
                
                st.session_state.fcs_data = df_processed
                st.session_state.meta_data = meta
                st.session_state.current_file = uploaded_file.name
                st.session_state.used_library = used_library
                st.session_state.gates = []  # 新しいファイルの場合ゲートをリセット
        
        df = st.session_state.fcs_data
        meta = st.session_state.meta_data
        
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # 3列メトリクス表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("イベント数", f"{len(df):,}")
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("アクティブゲート数", len(st.session_state.gates))
        
        # 詳細メタデータ表示（展開可能）
        if st.expander("📋 詳細メタデータ"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("基本情報")
                # 標準メタデータとFlowKit互換の小文字キー両方をチェック
                total_events = meta.get('$TOT', meta.get('tot', 'N/A'))
                parameters = meta.get('$PAR', meta.get('par', 'N/A'))
                date_info = meta.get('$DATE', meta.get('date', 'N/A'))
                begin_time = meta.get('$BTIM', meta.get('btim', 'N/A'))
                end_time = meta.get('$ETIM', meta.get('etim', 'N/A'))
                cytometer = meta.get('$CYT', meta.get('cyt', 'N/A'))
                cytometer_num = meta.get('$CYTNUM', meta.get('cytnum', 'N/A'))
                
                st.write(f"**総イベント数**: {total_events}")
                st.write(f"**パラメータ数**: {parameters}")
                st.write(f"**取得日**: {date_info}")
                st.write(f"**開始時刻**: {begin_time}")
                st.write(f"**終了時刻**: {end_time}")
                st.write(f"**使用機器**: {cytometer}")
                st.write(f"**機器番号**: {cytometer_num}")
            
            with col2:
                st.subheader("その他のメタデータ")
                # 全メタデータ項目の表示（最初の20項目）
                meta_items = list(meta.items())[:20]
                for key, value in meta_items:
                    if key not in ['$TOT', '$PAR', '$DATE', '$BTIM', '$ETIM', '$CYT', '$CYTNUM', 
                                   'tot', 'par', 'date', 'btim', 'etim', 'cyt', 'cytnum']:
                        st.write(f"**{key}**: {value}")
        
        # チャンネル選択
        st.subheader("🎯 チャンネル選択")
        channels = list(df.columns)
        
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
        
        # FCSProcessorインスタンス
        from utils.fcs_processor import FCSProcessor
        processor = FCSProcessor()
        
        # データ変換の適用
        df_plot = df.copy()
        if x_transform != "なし":
            df_plot[x_channel] = processor.apply_transform(df_plot[x_channel], x_transform.lower())
        if y_transform != "なし":
            df_plot[y_channel] = processor.apply_transform(df_plot[y_channel], y_transform.lower())
        
        # 表示設定・パフォーマンス最適化
        max_points = st.slider("表示ポイント数", 1000, min(100000, len(df)), 15000)
        
        if len(df_plot) > max_points:
            df_plot_sampled = df_plot.sample(n=max_points, random_state=42)
            st.info(f"表示速度向上のため、{max_points:,}ポイントをサンプリング表示しています")
        else:
            df_plot_sampled = df_plot
        
        # ゲーティング設定
        st.subheader("🎯 ゲーティング設定")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gate_type = st.selectbox(
                "ゲートタイプ",
                ["矩形ゲート", "ポリゴンゲート", "楕円ゲート", "閾値ゲート"]
            )
        
        with col2:
            gate_name = st.text_input("ゲート名", value=f"Gate_{len(st.session_state.gates)+1}")
        
        with col3:
            if st.button("ゲートをクリア"):
                st.session_state.gates = []
                st.rerun()
        
        # ゲーティングユーティリティ
        from utils.gating import GateManager
        from utils.plotting import PlottingUtils
        
        gating_utils = GateManager()
        plotting_utils = PlottingUtils()
        
        # インタラクティブ解析
        st.subheader("📈 インタラクティブプロット")
        
        # 密度プロットの作成
        fig = plotting_utils.create_density_plot(
            df_plot_sampled,
            x_channel,
            y_channel,
            title=f"{x_channel} vs {y_channel}",
            show_colorbar=True
        )
        
        # 既存のゲートを表示
        for i, gate in enumerate(st.session_state.gates):
            if gate['x_channel'] == x_channel and gate['y_channel'] == y_channel:
                fig = gating_utils.add_gate_to_plot(fig, i)
        
        # プロットを表示
        plot_container = st.container()
        with plot_container:
            event = st.plotly_chart(fig, use_container_width=True, key="main_plot")
        
        # ゲート作成フォーム
        if gate_type == "矩形ゲート":
            st.subheader("📦 矩形ゲート設定")
            col1, col2 = st.columns(2)
            
            with col1:
                x_min = st.number_input(f"{x_channel} 最小値", value=float(df_plot_sampled[x_channel].min()))
                x_max = st.number_input(f"{x_channel} 最大値", value=float(df_plot_sampled[x_channel].max()))
            
            with col2:
                y_min = st.number_input(f"{y_channel} 最小値", value=float(df_plot_sampled[y_channel].min()))
                y_max = st.number_input(f"{y_channel} 最大値", value=float(df_plot_sampled[y_channel].max()))
            
            if st.button("矩形ゲートを追加"):
                gate = gating_utils.create_rectangular_gate(
                    gate_name, x_channel, y_channel, x_min, x_max, y_min, y_max
                )
                st.session_state.gates.append(gate)
                st.success(f"ゲート '{gate_name}' を追加しました")
                st.rerun()
        
        elif gate_type == "ポリゴンゲート":
            st.subheader("🔷 ポリゴンゲート設定")
            st.write("座標をカンマ区切りで入力してください（例：x1,y1;x2,y2;x3,y3）")
            
            coordinates_input = st.text_area(
                "ポリゴン座標",
                value="",
                help="例：100,200;300,400;500,300;400,100"
            )
            
            if st.button("ポリゴンゲートを追加") and coordinates_input:
                try:
                    coordinates = gating_utils.parse_polygon_coordinates(coordinates_input)
                    gate = gating_utils.create_polygon_gate(
                        gate_name, x_channel, y_channel, coordinates
                    )
                    st.session_state.gates.append(gate)
                    st.success(f"ゲート '{gate_name}' を追加しました")
                    st.rerun()
                except Exception as e:
                    st.error(f"座標の解析に失敗しました: {str(e)}")
        
        elif gate_type == "楕円ゲート":
            st.subheader("⭕ 楕円ゲート設定")
            col1, col2 = st.columns(2)
            
            with col1:
                center_x = st.number_input(f"中心X ({x_channel})", value=float(df_plot_sampled[x_channel].mean()))
                center_y = st.number_input(f"中心Y ({y_channel})", value=float(df_plot_sampled[y_channel].mean()))
            
            with col2:
                width = st.number_input("幅", value=float(df_plot_sampled[x_channel].std() * 2), min_value=0.1)
                height = st.number_input("高さ", value=float(df_plot_sampled[y_channel].std() * 2), min_value=0.1)
            
            if st.button("楕円ゲートを追加"):
                gate = gating_utils.create_ellipse_gate(
                    gate_name, x_channel, y_channel, center_x, center_y, width, height
                )
                st.session_state.gates.append(gate)
                st.success(f"ゲート '{gate_name}' を追加しました")
                st.rerun()
        
        elif gate_type == "閾値ゲート":
            st.subheader("📏 閾値ゲート設定")
            
            threshold_channel = st.selectbox("閾値チャンネル", channels)
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_value = st.number_input(
                    "閾値",
                    value=float(df_plot[threshold_channel].median())
                )
            
            with col2:
                threshold_direction = st.selectbox("方向", ["以上", "以下", "より大きい", "より小さい"])
            
            if st.button("閾値ゲートを追加"):
                gate = gating_utils.create_threshold_gate(
                    gate_name, threshold_channel, threshold_value, threshold_direction
                )
                st.session_state.gates.append(gate)
                st.success(f"ゲート '{gate_name}' を追加しました")
                st.rerun()
        
        # 複数ゲート管理
        if st.session_state.gates:
            st.subheader("🎯 アクティブゲート")
            
            gate_data = []
            for i, gate in enumerate(st.session_state.gates):
                # ゲートの適用
                gated_indices = gating_utils.apply_gate(df_plot, gate)
                gate_info = {
                    "ゲート名": gate['name'],
                    "タイプ": gate['type'],
                    "チャンネル": f"{gate.get('x_channel', gate.get('channel', ''))} / {gate.get('y_channel', '')}",
                    "イベント数": len(gated_indices),
                    "割合 (%)": f"{len(gated_indices)/len(df_plot)*100:.2f}"
                }
                gate_data.append(gate_info)
            
            gate_df = pd.DataFrame(gate_data)
            st.dataframe(gate_df, use_container_width=True)
            
            # 個別ゲート削除
            gate_to_delete = st.selectbox(
                "削除するゲート",
                ["選択してください"] + [gate['name'] for gate in st.session_state.gates]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("選択ゲートを削除") and gate_to_delete != "選択してください":
                    st.session_state.gates = [
                        gate for gate in st.session_state.gates 
                        if gate['name'] != gate_to_delete
                    ]
                    st.success(f"ゲート '{gate_to_delete}' を削除しました")
                    st.rerun()
            
            with col2:
                if st.button("全ゲートクリア"):
                    st.session_state.gates = []
                    st.success("全てのゲートをクリアしました")
                    st.rerun()
        
        # 詳細統計解析
        if st.session_state.gates:
            st.subheader("📊 詳細統計解析")
            
            selected_gate = st.selectbox(
                "解析するゲート",
                [gate['name'] for gate in st.session_state.gates]
            )
            
            if selected_gate:
                # 選択されたゲートを適用
                gate = next(g for g in st.session_state.gates if g['name'] == selected_gate)
                gated_indices = gating_utils.apply_gate(df_plot, gate)
                
                if len(gated_indices) > 0:
                    gated_data = df_plot.loc[gated_indices]
                    
                    # 統計情報
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("ゲート内イベント数", f"{len(gated_data):,}")
                        st.metric("全体に対する割合", f"{len(gated_data)/len(df_plot)*100:.2f}%")
                    
                    with col2:
                        if len(gated_data) > 0:
                            st.metric(f"{x_channel} 平均", f"{gated_data[x_channel].mean():.2f}")
                            st.metric(f"{y_channel} 平均", f"{gated_data[y_channel].mean():.2f}")
                    
                    # 詳細統計テーブル
                    if st.expander("詳細統計情報"):
                        stats_channels = st.multiselect(
                            "統計を表示するチャンネル",
                            channels,
                            default=[x_channel, y_channel]
                        )
                        
                        if stats_channels:
                            stats_data = gated_data[stats_channels].describe()
                            st.dataframe(stats_data, use_container_width=True)
                    
                    # 改良されたデータエクスポート
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("ゲート内データCSV準備"):
                            st.session_state.gated_csv = gated_data.to_csv(index=False)
                            st.success("CSVデータを準備しました")
                    
                    with col2:
                        if 'gated_csv' in st.session_state:
                            st.download_button(
                                label="ゲート内データCSVダウンロード",
                                data=st.session_state.gated_csv,
                                file_name=f"{uploaded_file.name.replace('.fcs', '')}_{selected_gate}_gated.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("選択されたゲート内にイベントが見つかりません")
        
        # リアルタイム統計解析
        if st.expander("📊 リアルタイム統計解析"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"X軸 ({x_channel}) 統計")
                x_stats = {
                    "平均": df_plot_sampled[x_channel].mean(),
                    "中央値": df_plot_sampled[x_channel].median(),
                    "標準偏差": df_plot_sampled[x_channel].std(),
                    "最小値": df_plot_sampled[x_channel].min(),
                    "最大値": df_plot_sampled[x_channel].max()
                }
                for key, value in x_stats.items():
                    st.metric(key, f"{value:.2f}")
            
            with col2:
                st.subheader(f"Y軸 ({y_channel}) 統計")
                y_stats = {
                    "平均": df_plot_sampled[y_channel].mean(),
                    "中央値": df_plot_sampled[y_channel].median(),
                    "標準偏差": df_plot_sampled[y_channel].std(),
                    "最小値": df_plot_sampled[y_channel].min(),
                    "最大値": df_plot_sampled[y_channel].max()
                }
                for key, value in y_stats.items():
                    st.metric(key, f"{value:.2f}")

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
        # 包括的エラーハンドリング
        if st.expander("エラー詳細"):
            st.exception(e)
            
            # NumPy 2.0互換性問題の検出
            if "newbyteorder" in str(e) or "numpy" in str(e).lower():
                st.info("""
                **NumPy 2.0互換性問題の可能性があります**
                
                対処法：
                1. flowioライブラリを使用してください
                2. NumPy 1.x系にダウングレードしてください
                3. 別のFCS読み込みライブラリを試してください
                """)

if __name__ == "__main__":
    main()
