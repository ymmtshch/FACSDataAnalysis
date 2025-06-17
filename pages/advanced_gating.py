import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    BoxSelectTool, LassoSelectTool, PolySelectTool, 
    HoverTool, ResetTool, PanTool, WheelZoomTool,
    ColumnDataSource, CustomJS, Button,
    ColorBar, LinearColorMapper, Div
)
from bokeh.layouts import column, row
from bokeh.palettes import Viridis256
from bokeh.transform import transform
import json
from typing import Dict, List, Tuple, Optional
import uuid

def create_contour_data(x: np.ndarray, y: np.ndarray, bins: int = 50) -> Dict:
    """
    等高線データを作成
    """
    # ヒストグラム2Dでデータ密度を計算
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    
    # メッシュグリッドを作成
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    return {
        'x': X.flatten(),
        'y': Y.flatten(),
        'density': hist.T.flatten(),
        'x_edges': x_edges,
        'y_edges': y_edges
    }

def create_gating_plot(df: pd.DataFrame, x_param: str, y_param: str, 
                      show_contour: bool = True, bins: int = 50) -> figure:
    """
    ゲーティング用の散布図+等高線プロットを作成
    """
    # データの範囲を取得
    x_data = df[x_param].values
    y_data = df[y_param].values
    
    x_min, x_max = np.percentile(x_data, [1, 99])
    y_min, y_max = np.percentile(y_data, [1, 99])
    
    # プロット作成
    p = figure(
        width=700,
        height=600,
        title=f"Gating Plot: {x_param} vs {y_param}",
        x_axis_label=x_param,
        y_axis_label=y_param,
        toolbar_location="above"
    )
    
    # 等高線表示
    if show_contour:
        contour_data = create_contour_data(x_data, y_data, bins)
        
        # カラーマッパー
        color_mapper = LinearColorMapper(
            palette=Viridis256, 
            low=0, 
            high=np.max(contour_data['density'])
        )
        
        # 等高線データソース
        contour_source = ColumnDataSource(data=contour_data)
        
        # 等高線プロット
        p.rect(
            x='x', y='y',
            width=(contour_data['x_edges'][1] - contour_data['x_edges'][0]),
            height=(contour_data['y_edges'][1] - contour_data['y_edges'][0]),
            source=contour_source,
            fill_color=transform('density', color_mapper),
            fill_alpha=0.6,
            line_color=None
        )
        
        # カラーバー
        color_bar = ColorBar(
            color_mapper=color_mapper,
            width=8,
            location=(0, 0),
            title="Density"
        )
        p.add_layout(color_bar, 'right')
    
    # サンプリングしたデータポイントを表示（パフォーマンス向上のため）
    if len(df) > 10000:
        sample_indices = np.random.choice(len(df), 10000, replace=False)
        plot_df = df.iloc[sample_indices]
    else:
        plot_df = df
    
    # データソース
    source = ColumnDataSource(data={
        'x': plot_df[x_param].values,
        'y': plot_df[y_param].values,
        'index': plot_df.index.values
    })
    
    # 散布図
    scatter = p.circle(
        'x', 'y',
        source=source,
        size=2,
        alpha=0.3,
        color='navy',
        selection_color='red',
        nonselection_alpha=0.1
    )
    
    # ツール設定
    tools = [
        PanTool(),
        WheelZoomTool(),
        BoxSelectTool(),
        LassoSelectTool(),
        PolySelectTool(),
        ResetTool(),
        HoverTool(tooltips=[
            (x_param, '@x{0.00}'),
            (y_param, '@y{0.00}'),
            ('Index', '@index')
        ])
    ]
    
    p.tools = tools
    p.toolbar.active_drag = tools[0]  # Pan tool as default
    
    return p, source

def calculate_gate_statistics(df: pd.DataFrame, selected_indices: List[int]) -> Dict:
    """
    ゲート内データの統計情報を計算
    """
    if not selected_indices:
        return {}
    
    gated_data = df.iloc[selected_indices]
    total_events = len(df)
    gated_events = len(gated_data)
    
    stats = {
        'total_events': total_events,
        'gated_events': gated_events,
        'percentage': (gated_events / total_events) * 100,
        'statistics': {}
    }
    
    # 各パラメータの統計
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = gated_data[col]
        stats['statistics'][col] = {
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max())
        }
    
    return stats

def save_gate_to_session(gate_name: str, gate_data: Dict):
    """
    ゲート情報をセッション状態に保存
    """
    if 'current_gates' not in st.session_state:
        st.session_state.current_gates = []
    
    if 'gate_stats' not in st.session_state:
        st.session_state.gate_stats = {}
    
    gate_info = {
        'gate_id': str(uuid.uuid4()),
        'gate_name': gate_name,
        'gate_type': 'selection',
        'selected_indices': gate_data.get('selected_indices', []),
        'axes': gate_data.get('axes', ('', '')),
        'timestamp': pd.Timestamp.now()
    }
    
    st.session_state.current_gates.append(gate_info)
    st.session_state.gate_stats[gate_name] = gate_data.get('statistics', {})

def display_gate_statistics(stats: Dict):
    """
    ゲート統計情報を表示
    """
    if not stats:
        st.info("ゲートを作成して統計情報を表示してください。")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総イベント数", f"{stats['total_events']:,}")
    
    with col2:
        st.metric("ゲート内イベント数", f"{stats['gated_events']:,}")
    
    with col3:
        st.metric("ゲート率", f"{stats['percentage']:.2f}%")
    
    # パラメータ別統計
    if stats.get('statistics'):
        st.subheader("パラメータ別統計")
        
        stats_df = pd.DataFrame(stats['statistics']).T
        stats_df = stats_df.round(2)
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            column_config={
                'mean': st.column_config.NumberColumn('平均', format="%.2f"),
                'median': st.column_config.NumberColumn('中央値', format="%.2f"),
                'std': st.column_config.NumberColumn('標準偏差', format="%.2f"),
                'min': st.column_config.NumberColumn('最小値', format="%.2f"),
                'max': st.column_config.NumberColumn('最大値', format="%.2f")
            }
        )

def display_saved_gates():
    """
    保存されたゲート一覧を表示
    """
    if 'current_gates' not in st.session_state or not st.session_state.current_gates:
        st.info("保存されたゲートはありません。")
        return
    
    st.subheader("保存されたゲート")
    
    for i, gate in enumerate(st.session_state.current_gates):
        with st.expander(f"🎯 {gate['gate_name']} ({gate['timestamp'].strftime('%H:%M:%S')})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**ゲートID:** {gate['gate_id'][:8]}...")
                st.write(f"**軸:** {gate['axes'][0]} vs {gate['axes'][1]}")
                st.write(f"**選択イベント数:** {len(gate['selected_indices']):,}")
            
            with col2:
                if st.button("削除", key=f"delete_gate_{i}"):
                    st.session_state.current_gates.pop(i)
                    if gate['gate_name'] in st.session_state.gate_stats:
                        del st.session_state.gate_stats[gate['gate_name']]
                    st.rerun()

def main():
    """
    メイン関数
    """
    st.title("🎯 Advanced Gating Analysis")
    st.markdown("高度なゲーティング解析を行います。マウス操作でゲート領域を設定し、詳細な統計情報を取得できます。")
    
    # データ確認
    if 'fcs_data' not in st.session_state or st.session_state.fcs_data is None:
        st.error("FCSデータが読み込まれていません。基本解析ページでデータをアップロードしてください。")
        return
    
    df = st.session_state.fcs_data
    
    # サイドバー設定
    st.sidebar.header("プロット設定")
    
    # パラメータ選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    x_param = st.sidebar.selectbox(
        "X軸パラメータ",
        numeric_cols,
        index=0 if len(numeric_cols) > 0 else None
    )
    
    y_param = st.sidebar.selectbox(
        "Y軸パラメータ",
        numeric_cols,
        index=1 if len(numeric_cols) > 1 else 0
    )
    
    # 表示オプション
    show_contour = st.sidebar.checkbox("等高線表示", value=True)
    contour_bins = st.sidebar.slider("等高線解像度", 20, 100, 50)
    
    # メインエリア
    tab1, tab2, tab3 = st.tabs(["📊 ゲーティングプロット", "📈 統計情報", "💾 保存されたゲート"])
    
    with tab1:
        st.subheader(f"ゲーティングプロット: {x_param} vs {y_param}")
        
        # プロット作成
        plot, source = create_gating_plot(
            df, x_param, y_param, 
            show_contour=show_contour, 
            bins=contour_bins
        )
        
        # プロット表示
        plot_container = st.empty()
        plot_container.bokeh_chart(plot, use_container_width=True)
        
        # ゲート操作パネル
        st.subheader("ゲート操作")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            gate_name = st.text_input(
                "ゲート名",
                value=f"Gate_{len(st.session_state.get('current_gates', []))+1}"
            )
        
        with col2:
            if st.button("📊 統計計算", type="primary"):
                # JavaScriptで選択データを取得（実際の実装では別途JSコールバックが必要）
                st.info("プロット上でデータを選択してから統計計算を実行してください。")
        
        with col3:
            if st.button("💾 ゲート保存"):
                # 実際の実装では選択されたデータのインデックスを取得
                # ここではデモ用の仮データ
                selected_indices = []  # JS callbackから取得
                if selected_indices:
                    stats = calculate_gate_statistics(df, selected_indices)
                    gate_data = {
                        'selected_indices': selected_indices,
                        'axes': (x_param, y_param),
                        'statistics': stats
                    }
                    save_gate_to_session(gate_name, gate_data)
                    st.success(f"ゲート '{gate_name}' を保存しました！")
                else:
                    st.warning("データを選択してからゲートを保存してください。")
        
        # 使用方法の説明
        with st.expander("🔧 使用方法"):
            st.markdown("""
            **ゲーティングの手順:**
            1. **ツール選択**: プロット上部のツールバーから選択ツールを選択
               - 🔲 Box Select: 矩形選択
               - 🎯 Lasso Select: 自由描画選択
               - 📐 Poly Select: ポリゴン選択
            2. **データ選択**: マウスでドラッグして領域を選択
            3. **統計計算**: "統計計算"ボタンをクリック
            4. **ゲート保存**: ゲート名を入力して"ゲート保存"をクリック
            
            **ツールバー操作:**
            - 🖱️ Pan: プロットの移動
            - 🔍 Wheel Zoom: マウスホイールでズーム
            - 🔄 Reset: 表示範囲をリセット
            """)
    
    with tab2:
        st.subheader("ゲート統計情報")
        
        # 最新の統計情報を表示
        if 'gate_stats' in st.session_state and st.session_state.gate_stats:
            latest_gate = list(st.session_state.gate_stats.keys())[-1]
            latest_stats = st.session_state.gate_stats[latest_gate]
            
            st.write(f"**最新ゲート:** {latest_gate}")
            display_gate_statistics(latest_stats)
        else:
            display_gate_statistics({})
        
        # データ概要
        st.subheader("データ概要")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("総イベント数", f"{len(df):,}")
            st.metric("パラメータ数", len(numeric_cols))
        
        with col2:
            st.metric("ゲート数", len(st.session_state.get('current_gates', [])))
            if st.session_state.get('current_gates'):
                total_gated = sum(len(gate['selected_indices']) for gate in st.session_state.current_gates)
                st.metric("総ゲートイベント数", f"{total_gated:,}")
    
    with tab3:
        display_saved_gates()
        
        # ゲートエクスポート機能
        st.subheader("📤 ゲートエクスポート")
        
        if st.session_state.get('current_gates'):
            export_format = st.radio(
                "エクスポート形式",
                ["JSON", "CSV (統計のみ)"],
                horizontal=True
            )
            
            if st.button("エクスポート"):
                if export_format == "JSON":
                    # ゲート情報をJSONでエクスポート
                    export_data = {
                        'gates': st.session_state.current_gates,
                        'statistics': st.session_state.gate_stats,
                        'export_timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    # JSON文字列に変換（NumPy配列などの変換処理）
                    json_str = json.dumps(export_data, default=str, indent=2)
                    
                    st.download_button(
                        label="📁 JSONファイルをダウンロード",
                        data=json_str,
                        file_name=f"gates_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                elif export_format == "CSV (統計のみ)":
                    # 統計情報をCSVでエクスポート
                    if st.session_state.gate_stats:
                        stats_list = []
                        for gate_name, stats in st.session_state.gate_stats.items():
                            if 'statistics' in stats:
                                for param, param_stats in stats['statistics'].items():
                                    row = {
                                        'gate_name': gate_name,
                                        'parameter': param,
                                        **param_stats
                                    }
                                    stats_list.append(row)
                        
                        if stats_list:
                            stats_df = pd.DataFrame(stats_list)
                            csv_str = stats_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📄 CSVファイルをダウンロード",
                                data=csv_str,
                                file_name=f"gate_statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
        else:
            st.info("エクスポートするゲートがありません。")

if __name__ == "__main__":
    main()
