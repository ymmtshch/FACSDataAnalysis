
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(page_title="📊 高度解析", page_icon="📊", layout="wide")

st.title("📊 高度解析機能")
st.write("FACSデータに対して主成分分析（PCA）やクラスタリング（KMeans）を実行します。")

# Initialize session state safely
if 'fcs_data' not in st.session_state or st.session_state.fcs_data is None:
    st.warning("⚠️ FCSデータが読み込まれていません。サイドバーからファイルをアップロードしてください。")
    st.stop()

# Get numeric data
data = st.session_state.fcs_data
try:
    numeric_data = data.select_dtypes(include=[np.number])
except Exception as e:
    st.error(f"❌ 数値データの抽出中にエラーが発生しました: {str(e)}")
    st.stop()

if numeric_data.empty:
    st.error("❌ 数値チャンネルが見つかりません。")
    st.stop()

# PCA Section
st.subheader("🔍 主成分分析（PCA）")

n_components = st.slider("主成分数", min_value=2, max_value=min(5, len(numeric_data.columns)), value=2)

try:
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(numeric_data.fillna(0))
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

    st.markdown("**PCA 分散比率**")
    explained_var = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(n_components)],
        "分散比率": pca.explained_variance_ratio_
    })
    st.dataframe(explained_var, use_container_width=True)

    if n_components >= 2:
        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Scatter Plot", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ PCA解析中にエラーが発生しました: {str(e)}")

# Clustering Section
st.subheader("🔗 クラスタリング（KMeans）")

n_clusters = st.slider("クラスタ数", min_value=2, max_value=10, value=3)

try:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(numeric_data.fillna(0))
    pca_df["Cluster"] = cluster_labels.astype(str)

    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA + KMeans クラスタリング", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**クラスタごとのイベント数**")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_df = pd.DataFrame({
        "クラスタ": cluster_counts.index,
        "イベント数": cluster_counts.values
    })
    st.dataframe(cluster_df, use_container_width=True)

except Exception as e:
    st.error(f"❌ クラスタリング中にエラーが発生しました: {str(e)}")
