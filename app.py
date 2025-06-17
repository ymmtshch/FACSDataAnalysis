import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# fcsparserを使用（flowkitの軽量代替）
try:
    import fcsparser
    FCS_AVAILABLE = True
except ImportError:
    FCS_AVAILABLE = False
    st.error("FCSパーサーが利用できません。requirements.txtを確認してください。")

st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 FACS Data Analysis")
st.markdown("**フローサイトメトリーデータの解析ツール**")

# サイドバー設定
st.sidebar.header("📁 ファイルアップロード")

if FCS_AVAILABLE:
    uploaded_file = st.sidebar.file_uploader(
        "FCSファイルを選択してください",
        type=['fcs'],
        help="標準的なFCS 2.0/3.0/3.1ファイルに対応"
    )
    
    if uploaded_file is not None:
        try:
            # FCSファイルの読み込み
            with st.spinner("FCSファイルを読み込み中..."):
                # バイトデータを一時ファイルとして処理
                file_content = uploaded_file.read()
                
                # fcsparserを使用してFCSファイルを解析
                meta, data = fcsparser.parse(file_content, meta_data_only=False, reformat_meta=True)
                
                # データフレームに変換
                df = pd.DataFrame(data)
                
            st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
            
            # ファイル情報表示
            st.sidebar.subheader("📊 ファイル情報")
            st.sidebar.info(f"""
            - **イベント数**: {len(df):,}
            - **パラメータ数**: {len(df.columns)}
            - **ファイルサイズ**: {len(file_content):,} bytes
            """)
            
            # メインコンテンツ
            tab1, tab2, tab3 = st.tabs(["📈 基本解析", "🎯 散布図解析", "📊 統計情報"])
            
            with tab1:
                st.subheader("ヒストグラム解析")
                
                # パラメータ選択
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    param_options = list(df.columns)
                    selected_param = st.selectbox(
                        "解析パラメータを選択:",
                        param_options,
                        index=0 if param_options else 0
                    )
                
                with col2:
                    log_scale = st.checkbox("対数スケール", value=False)
                    bins = st.slider("ビン数", min_value=50, max_value=500, value=100)
                
                if selected_param:
                    # ヒストグラム作成
                    fig = px.histogram(
                        df, 
                        x=selected_param,
                        nbins=bins,
                        title=f"{selected_param} のヒストグラム",
                        labels={selected_param: selected_param}
                    )
                    
                    if log_scale and (df[selected_param] > 0).all():
                        fig.update_xaxes(type="log")
                        fig.update_layout(title=f"{selected_param} のヒストグラム (対数スケール)")
                    
                    fig.update_layout(
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("散布図解析")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    x_param = st.selectbox(
                        "X軸パラメータ:",
                        param_options,
                        index=0 if len(param_options) > 0 else 0,
                        key="x_param"
                    )
                
                with col2:
                    y_param = st.selectbox(
                        "Y軸パラメータ:",
                        param_options,
                        index=1 if len(param_options) > 1 else 0,
                        key="y_param"
                    )
                
                col3, col4 = st.columns([1, 1])
                
                with col3:
                    sample_size = st.slider(
                        "表示イベント数 (サンプリング)",
                        min_value=1000,
                        max_value=min(50000, len(df)),
                        value=min(10000, len(df))
                    )
                
                with col4:
                    opacity = st.slider("透明度", min_value=0.1, max_value=1.0, value=0.6)
                
                if x_param and y_param:
                    # サンプリング
                    df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
                    
                    # 散布図作成
                    fig = px.scatter(
                        df_sample,
                        x=x_param,
                        y=y_param,
                        title=f"{x_param} vs {y_param}",
                        opacity=opacity
                    )
                    
                    fig.update_layout(height=600)
                    fig.update_traces(marker=dict(size=2))
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("統計情報")
                
                # 基本統計
                st.write("**基本統計量:**")
                st.dataframe(df.describe(), use_container_width=True)
                
                # データプレビュー
                st.write("**データプレビュー (最初の100行):**")
                st.dataframe(df.head(100), use_container_width=True)
                
                # CSV出力
                st.subheader("📥 データエクスポート")
                
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📁 CSVファイルとしてダウンロード",
                    data=csv_data,
                    file_name=f"{uploaded_file.name.replace('.fcs', '')}_analyzed.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ ファイル処理エラー: {str(e)}")
            st.info("ファイル形式を確認してください。標準的なFCS 2.0/3.0/3.1ファイルが必要です。")
    
    else:
        st.info("👆 左のサイドバーからFCSファイルをアップロードしてください。")
        
        # デモデータの説明
        st.markdown("""
        ## 🔬 FACS Data Analysis について
        
        このアプリケーションは、フローサイトメトリー（FACS）データの基本的な解析を行うツールです。
        
        ### 📋 主な機能:
        - **FCSファイル読み込み**: 標準的なFCS 2.0/3.0/3.1ファイルに対応
        - **ヒストグラム解析**: 各パラメータの分布を可視化
        - **散布図解析**: 2つのパラメータの相関を解析
        - **統計情報**: 基本統計量とデータプレビュー
        - **データエクスポート**: 解析結果をCSV形式で出力
        
        ### 🚀 使用方法:
        1. 左のサイドバーからFCSファイルをアップロード
        2. 各タブで解析を実行
        3. 必要に応じて結果をダウンロード
        """)

else:
    st.error("""
    ❌ **FCSパーサーライブラリが利用できません**
    
    requirements.txtに以下を追加してください:
    ```
    fcsparser==0.2.8
    ```
    """)

# フッター
st.markdown("---")
st.markdown("**FACS Data Analysis** - Streamlit Cloud対応版")
