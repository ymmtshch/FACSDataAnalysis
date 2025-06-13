import streamlit as st
import pandas as pd
import fcsparser
import tempfile
import os

st.title("FACS Data Analysis")
st.write("FlowCytometry Standard（.fcs）ファイルからイベントデータを抽出し、CSV形式でダウンロードできるアプリケーションです。")

# ファイルアップロード
uploaded_file = st.file_uploader("FCS ファイルをアップロードしてください", type=['fcs'])

if uploaded_file is not None:
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # FCSファイルを読み込み
        with st.spinner('FCSファイルを処理中...'):
            meta, data = fcsparser.parse(tmp_file_path, reformat_meta=True)
        
        # データフレームに変換
        df = pd.DataFrame(data)
        
        # 一時ファイルを削除
        os.unlink(tmp_file_path)
        
        st.success(f"FCSファイルが正常に読み込まれました。{len(df)} イベントが検出されました。")
        
        # データのプレビュー表示
        st.subheader("データプレビュー（上位10行）")
        st.dataframe(df.head(10))
        
        # 統計情報
        st.subheader("データ統計")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総イベント数", len(df))
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("ファイルサイズ", f"{len(uploaded_file.getvalue())/1024/1024:.2f} MB")
        
        # メタデータ表示
        if st.checkbox("メタデータを表示"):
            st.subheader("FCSファイル メタデータ")
            meta_df = pd.DataFrame(list(meta.items()), columns=['Key', 'Value'])
            st.dataframe(meta_df)
        
        # CSVダウンロード
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="CSV をダウンロード",
            data=csv_data,
            file_name="events_output.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"FCSファイルの処理中にエラーが発生しました: {str(e)}")
        st.write("エラーの詳細:")
        st.code(str(e))
        
        # 一時ファイルが残っている場合は削除
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

else:
    st.info("FCSファイルをアップロードしてください。")
    
    # 使用方法の説明
    st.subheader("使用方法")
    st.write("""
    1. 「FCS ファイルをアップロードしてください」ボタンをクリックして、FCSファイルを選択します
    2. ファイルが正常に読み込まれると、イベントデータの上位10行が表示されます
    3. 「CSV をダウンロード」ボタンをクリックして、全イベントデータをCSV形式でダウンロードできます
    """)
    
    st.subheader("対応ファイル形式")
    st.write("- .fcs（Flow Cytometry Standard）ファイル")
    
    st.subheader("注意事項")
    st.write("""
    - 大きなFCSファイルの場合、処理に時間がかかる場合があります
    - イベントデータは生データ（raw）として抽出されます
    - メタデータも確認できます
    """)
