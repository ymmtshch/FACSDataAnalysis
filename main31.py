import streamlit as st
import flowkit as fk
import pandas as pd
import io

st.title("FCS イベントデータ抽出 & CSV ダウンロードアプリ")

uploaded_file = st.file_uploader("FCS ファイルをアップロードしてください", type=["fcs"])

if uploaded_file is not None:
    try:
        # 一時的にファイルを読み込み
        with open("temp.fcs", "wb") as f:
            f.write(uploaded_file.read())

        # FlowKitで読み込み
        sample = fk.Sample("temp.fcs")

        # DataFrameとして取得
        df_events = sample.as_dataframe(source='raw')

        st.success("FCS ファイルを読み込みました！")
        st.subheader("イベントデータ（上位10行）")
        st.dataframe(df_events.head(10))

        # CSVへ変換してダウンロードリンクを生成
        csv = df_events.to_csv(index=False)
        b64 = csv.encode('utf-8')
        st.download_button(
            label="CSV をダウンロード",
            data=b64,
            file_name="events_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
else:
    st.info("FCS ファイルをアップロードすると、イベント情報を表示・ダウンロードできます。")
