import streamlit as st
import pandas as pd
import numpy as np
import fcsparser
from utils.fcs_processor import FCSProcessor
from utils.plotting import PlottingUtils  # 新しく追加したクラスを使用
from utils.gating import GateManager
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
from config import Config

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
        if st.session_state.fcs_data is None or st.session_state.get('current_file') != uploaded_file.name:
            with st.spinner("FCSファイルを読み込み中..."):
                # fcsparserでファイルを読み込み
                meta, data = fcsparser.parse(uploaded_file, meta_data_only=False, reformat_meta=True)
                
                # FCSProcessorインスタンスを作成
                processor = FCSProcessor()
                
                # データの前処理
                df_processed = processor.preprocess_data(data, meta)
                
                st.session_state.fcs_data = df_processed
                st.session_state.meta_data = meta
                st.session_state.current_file = uploaded_file.name
                st.session_state.gates = []  # 新しいファイルの場合ゲートをリセット
        
        df = st.session_state.fcs_data
        meta = st.session_state.meta_data
        
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # ファイル情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("イベント数", f"{len(df):,}")
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("アクティブゲート数", len(st.session_state.gates))
        
        # チャンネル選択
        st.subheader("🎯 チャンネル選択")
        channels = list(df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            x_channel = st.selectbox("X軸チャンネル", channels, index=0)
        with col2:
            y_channel = st.selectbox("Y軸チャンネル", channels, index=1 if len(channels) > 1 else 0)
        
        # データ変換
        st.subheader("🔧 データ変換")
        transform_options = ["なし", "Log10", "Asinh", "Biexponential"]
        
        col1, col2 = st.columns(2)
        with col1:
            x_transform = st.selectbox("X軸変換", transform_options)
        with col2:
            y_transform = st.selectbox("Y軸変換", transform_options)
        
        # FCSProcessorインスタンス
        processor = FCSProcessor()
        
        # データ変換の適用
        df_plot = df.copy()
        if x_transform != "なし":
            df_plot[x_channel] = processor.apply_transform(df_plot[x_channel], x_transform.lower())
        if y_transform != "なし":
            df_plot[y_channel] = processor.apply_transform(df_plot[y_channel], y_transform.lower())
        
        # サンプリング設定
        max_points = st.slider("表示ポイント数", 1000, min(50000, len(df)), 15000)
        
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
        gating_utils = GateManager()
        plotting_utils = PlottingUtils()
        
        # インタラクティブプロット
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
                fig = gating_utils.add_gate_to_plot(fig, gate, i)
        
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
        
        # アクティブゲート一覧
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
            
            # ゲート削除
            gate_to_delete = st.selectbox(
                "削除するゲート",
                ["選択してください"] + [gate['name'] for gate in st.session_state.gates]
            )
            
            if st.button("選択ゲートを削除") and gate_to_delete != "選択してください":
                st.session_state.gates = [
                    gate for gate in st.session_state.gates 
                    if gate['name'] != gate_to_delete
                ]
                st.success(f"ゲート '{gate_to_delete}' を削除しました")
                st.rerun()
        
        # ゲート統計解析
        if st.session_state.gates:
            st.subheader("📊 ゲート統計解析")
            
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
                    
                    # ゲート内データのエクスポート
                    if st.button("ゲート内データをCSV出力"):
                        csv = gated_data.to_csv(index=False)
                        st.download_button(
                            label="ゲート内データCSVダウンロード",
                            data=csv,
                            file_name=f"{uploaded_file.name.replace('.fcs', '')}_{selected_gate}_gated.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("選択されたゲート内にイベントが見つかりません")

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
        if st.expander("エラー詳細"):
            st.exception(e)

if __name__ == "__main__":
    main()
