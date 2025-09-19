
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import Config
from utils.fcs_loader import process_file

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
        st.info("👉 サイドバーからFCSファイルをアップロードしてください")
        return

    # Transformation and max events
    transformation = st.sidebar.selectbox("データ変換", ["なし", "log10", "asinh"], index=0)
    max_events = st.sidebar.number_input("最大イベント数", min_value=1000, max_value=100000, value=50000, step=1000)

    # Load and process file using shared utility
    if (st.session_state.fcs_data is None or
        st.session_state.get('current_file') != uploaded_file.name):

        with st.spinner("FCSファイルを読み込み中..."):
            processor, data, metadata, error_message = process_file(uploaded_file, transformation, max_events)

        if data is not None and len(data) > 0:
            st.session_state.fcs_data = data
            st.session_state.meta_data = metadata
            st.session_state.processor = processor
            st.session_state.current_file = uploaded_file.name
            st.session_state.gates = []
            st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
        else:
            st.error(f"❌ 読み込み失敗: {error_message}")
            return

    # Continue with gating logic (unchanged)
    df = st.session_state.fcs_data
    st.metric("イベント数", f"{len(df):,}")
    st.metric("パラメータ数", len(df.columns))
    st.metric("アクティブゲート数", len(st.session_state.gates))

    # ... (rest of gating UI and logic remains unchanged)

if __name__ == "__main__":
    main()
