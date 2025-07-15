import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import Config

# Simplified FCS file loading
def load_fcs_file(uploaded_file):
    """Simple FCS file loading with error handling"""
    try:
        # Primary: fcsparser
        import fcsparser
        meta, data = fcsparser.parse(uploaded_file, meta_data_only=False, reformat_meta=True)
        return meta, data, "fcsparser"
    except Exception as e:
        st.error(f"FCSファイルの読み込みに失敗しました: {str(e)}")
        st.info("対処法: ファイル形式がFCS 2.0/3.0/3.1であることを確認してください")
        raise e

# Simple data transformation
def apply_transform(data, transform_type):
    """Apply simple data transformation"""
    if transform_type == "log10":
        # Add small epsilon to avoid log(0)
        return np.log10(data + 1e-10)
    elif transform_type == "asinh":
        return np.arcsinh(data / 150)  # Common cofactor for flow cytometry
    else:
        return data

# Simple rectangular gate
def create_rectangular_gate(name, x_channel, y_channel, x_min, x_max, y_min, y_max):
    """Create simple rectangular gate"""
    return {
        'name': name,
        'type': 'rectangular',
        'x_channel': x_channel,
        'y_channel': y_channel,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }

# Simple threshold gate
def create_threshold_gate(name, channel, threshold, direction):
    """Create simple threshold gate"""
    return {
        'name': name,
        'type': 'threshold',
        'channel': channel,
        'threshold': threshold,
        'direction': direction
    }

# Apply gate to data
def apply_gate(data, gate):
    """Apply gate and return indices of gated events"""
    try:
        if gate['type'] == 'rectangular':
            mask = (
                (data[gate['x_channel']] >= gate['x_min']) &
                (data[gate['x_channel']] <= gate['x_max']) &
                (data[gate['y_channel']] >= gate['y_min']) &
                (data[gate['y_channel']] <= gate['y_max'])
            )
            return data[mask].index
        
        elif gate['type'] == 'threshold':
            if gate['direction'] == '以上':
                mask = data[gate['channel']] >= gate['threshold']
            elif gate['direction'] == '以下':
                mask = data[gate['channel']] <= gate['threshold']
            elif gate['direction'] == 'より大きい':
                mask = data[gate['channel']] > gate['threshold']
            else:  # より小さい
                mask = data[gate['channel']] < gate['threshold']
            
            return data[mask].index
        
        else:
            return pd.Index([])
    
    except Exception as e:
        st.error(f"ゲート適用エラー: {str(e)}")
        return pd.Index([])

def main():
    st.title("高度ゲーティング解析")
    st.write("シンプルなゲーティング解析を行います。")
    
    # Initialize session state
    if 'fcs_data' not in st.session_state:
        st.session_state.fcs_data = None
    if 'gates' not in st.session_state:
        st.session_state.gates = []
    if 'meta_data' not in st.session_state:
        st.session_state.meta_data = None
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "FCSファイルを選択してください",
        type=['fcs'],
        help="FCS 2.0/3.0/3.1形式のファイルをサポートしています"
    )
    
    if uploaded_file is None:
        st.info("👈 サイドバーからFCSファイルをアップロードしてください")
        return
    
    try:
        # Load FCS file
        if (st.session_state.fcs_data is None or 
            st.session_state.get('current_file') != uploaded_file.name):
            
            with st.spinner("FCSファイルを読み込み中..."):
                meta, data, used_library = load_fcs_file(uploaded_file)
                
                # Simple data preprocessing
                df = pd.DataFrame(data)
                # Remove any completely empty columns
                df = df.dropna(axis=1, how='all')
                
                st.session_state.fcs_data = df
                st.session_state.meta_data = meta
                st.session_state.current_file = uploaded_file.name
                st.session_state.gates = []  # Reset gates for new file
        
        df = st.session_state.fcs_data
        meta = st.session_state.meta_data
        
        st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("イベント数", f"{len(df):,}")
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("アクティブゲート数", len(st.session_state.gates))
        
        # Channel selection
        st.subheader("🎯 チャンネル選択")
        channels = list(df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            x_channel = st.selectbox("X軸チャンネル", channels, index=0)
        with col2:
            y_channel = st.selectbox("Y軸チャンネル", channels, 
                                   index=1 if len(channels) > 1 else 0)
        
        # Data transformation
        st.subheader("🔧 データ変換")
        transform_options = ["なし", "log10", "asinh"]
        
        col1, col2 = st.columns(2)
        with col1:
            x_transform = st.selectbox("X軸変換", transform_options)
        with col2:
            y_transform = st.selectbox("Y軸変換", transform_options)
        
        # Apply transformations
        df_plot = df.copy()
        if x_transform != "なし":
            df_plot[x_channel] = apply_transform(df_plot[x_channel], x_transform)
        if y_transform != "なし":
            df_plot[y_channel] = apply_transform(df_plot[y_channel], y_transform)
        
        # Sampling for display performance
        max_points = st.slider("表示ポイント数", 1000, min(50000, len(df)), 10000)
        
        if len(df_plot) > max_points:
            df_plot_sampled = df_plot.sample(n=max_points, random_state=42)
            st.info(f"表示速度向上のため、{max_points:,}ポイントをサンプリング表示しています")
        else:
            df_plot_sampled = df_plot
        
        # Gating controls
        st.subheader("🎯 ゲーティング設定")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gate_type = st.selectbox("ゲートタイプ", ["矩形ゲート", "閾値ゲート"])
        
        with col2:
            gate_name = st.text_input("ゲート名", 
                                    value=f"Gate_{len(st.session_state.gates)+1}")
        
        with col3:
            if st.button("全ゲートクリア"):
                st.session_state.gates = []
                st.rerun()
        
        # Create scatter plot
        st.subheader("📈 散布図")
        
        fig = px.scatter(
            df_plot_sampled,
            x=x_channel,
            y=y_channel,
            opacity=0.6,
            title=f"{x_channel} vs {y_channel}",
            width=700,
            height=500
        )
        
        fig.update_layout(
            xaxis_title=f"{x_channel} ({x_transform})",
            yaxis_title=f"{y_channel} ({y_transform})",
            showlegend=False
        )
        
        # Add existing gates to plot
        for gate in st.session_state.gates:
            if gate['type'] == 'rectangular' and gate['x_channel'] == x_channel and gate['y_channel'] == y_channel:
                fig.add_shape(
                    type="rect",
                    x0=gate['x_min'], y0=gate['y_min'],
                    x1=gate['x_max'], y1=gate['y_max'],
                    line=dict(color="red", width=2),
                    fillcolor="red",
                    opacity=0.2
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gate creation forms
        if gate_type == "矩形ゲート":
            st.subheader("📦 矩形ゲート設定")
            
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input(
                    f"{x_channel} 最小値", 
                    value=float(df_plot_sampled[x_channel].quantile(0.1))
                )
                x_max = st.number_input(
                    f"{x_channel} 最大値", 
                    value=float(df_plot_sampled[x_channel].quantile(0.9))
                )
            
            with col2:
                y_min = st.number_input(
                    f"{y_channel} 最小値", 
                    value=float(df_plot_sampled[y_channel].quantile(0.1))
                )
                y_max = st.number_input(
                    f"{y_channel} 最大値", 
                    value=float(df_plot_sampled[y_channel].quantile(0.9))
                )
            
            if st.button("矩形ゲートを追加"):
                if gate_name and x_min < x_max and y_min < y_max:
                    gate = create_rectangular_gate(
                        gate_name, x_channel, y_channel, x_min, x_max, y_min, y_max
                    )
                    st.session_state.gates.append(gate)
                    st.success(f"ゲート '{gate_name}' を追加しました")
                    st.rerun()
                else:
                    st.error("ゲート名を入力し、最小値 < 最大値になるように設定してください")
        
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
                threshold_direction = st.selectbox(
                    "方向", 
                    ["以上", "以下", "より大きい", "より小さい"]
                )
            
            if st.button("閾値ゲートを追加"):
                if gate_name:
                    gate = create_threshold_gate(
                        gate_name, threshold_channel, threshold_value, threshold_direction
                    )
                    st.session_state.gates.append(gate)
                    st.success(f"ゲート '{gate_name}' を追加しました")
                    st.rerun()
                else:
                    st.error("ゲート名を入力してください")
        
        # Gate management
        if st.session_state.gates:
            st.subheader("🎯 アクティブゲート")
            
            gate_data = []
            for gate in st.session_state.gates:
                gated_indices = apply_gate(df_plot, gate)
                gate_info = {
                    "ゲート名": gate['name'],
                    "タイプ": gate['type'],
                    "チャンネル": gate.get('x_channel', gate.get('channel', '')),
                    "イベント数": len(gated_indices),
                    "割合 (%)": f"{len(gated_indices)/len(df_plot)*100:.2f}"
                }
                gate_data.append(gate_info)
            
            gate_df = pd.DataFrame(gate_data)
            st.dataframe(gate_df, use_container_width=True)
            
            # Gate deletion
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
        
        # Statistical analysis
        if st.session_state.gates:
            st.subheader("📊 統計解析")
            
            selected_gate = st.selectbox(
                "解析するゲート",
                [gate['name'] for gate in st.session_state.gates]
            )
            
            if selected_gate:
                gate = next(g for g in st.session_state.gates if g['name'] == selected_gate)
                gated_indices = apply_gate(df_plot, gate)
                
                if len(gated_indices) > 0:
                    gated_data = df_plot.loc[gated_indices]
                    
                    # Basic statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("ゲート内イベント数", f"{len(gated_data):,}")
                        st.metric("全体に対する割合", f"{len(gated_data)/len(df_plot)*100:.2f}%")
                    
                    with col2:
                        if len(gated_data) > 0:
                            st.metric(f"{x_channel} 平均", f"{gated_data[x_channel].mean():.2f}")
                            st.metric(f"{y_channel} 平均", f"{gated_data[y_channel].mean():.2f}")
                    
                    # Detailed statistics
                    if st.expander("詳細統計情報"):
                        stats_channels = st.multiselect(
                            "統計を表示するチャンネル",
                            channels,
                            default=[x_channel, y_channel]
                        )
                        
                        if stats_channels:
                            stats_data = gated_data[stats_channels].describe()
                            st.dataframe(stats_data, use_container_width=True)
                    
                    # Data export
                    if st.button("ゲート内データCSV準備"):
                        try:
                            csv_data = gated_data.to_csv(index=False)
                            st.download_button(
                                label="ゲート内データCSVダウンロード",
                                data=csv_data,
                                file_name=f"{uploaded_file.name.replace('.fcs', '')}_{selected_gate}_gated.csv",
                                mime="text/csv"
                            )
                            st.success("CSVダウンロードボタンが表示されました")
                        except Exception as e:
                            st.error(f"CSV準備エラー: {str(e)}")
                else:
                    st.warning("選択されたゲート内にイベントが見つかりません")
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
        # Error details
        if st.expander("エラー詳細"):
            st.exception(e)
            
            # Common troubleshooting
            st.info("""
            **よくある問題と対処法**
            
            1. **ファイル読み込みエラー**: FCS形式とファイルサイズ（100MB以下）を確認
            2. **メモリ不足**: 表示ポイント数を削減
            3. **変換エラー**: 変換方法を"なし"に変更
            4. **ゲートエラー**: ゲート範囲を再確認
            """)

if __name__ == "__main__":
    main()
