import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ==============================
# Page Config
# ==============================

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🛍️ Customer Segmentation & Marketing Analysis")
st.write("Interactive Machine Learning Dashboard")

# ==============================
# Sidebar Navigation
# ==============================

st.sidebar.title("📌 Navigation")

section = st.sidebar.radio(
    "Go to Section:",
    [
        "Upload Data",
        "Data Overview",
        "Outlier Treatment",
        "PCA Analysis",
        "Clustering Analysis",
        "Hierarchical Clustering",
        "Marketing Insights",
        "About"
    ]
)

# ==============================
# Session State
# ==============================

if "df" not in st.session_state:
    st.session_state.df = None

# ==============================
# 1️⃣ Upload Data
# ==============================

if section == "Upload Data":

    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully!")

# ==============================
# 2️⃣ Data Overview
# ==============================

elif section == "Data Overview":

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Data Summary")
        st.write(df.describe())

    else:
        st.warning("Please upload data first.")

# ==============================
# 3️⃣ Outlier Treatment
# ==============================

elif section == "Outlier Treatment":

    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns

        col = st.selectbox("Select Column", numeric_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Please upload data first.")

# ==============================
# 4️⃣ PCA Analysis
# ==============================

elif section == "PCA Analysis":

    if st.session_state.df is not None:

        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        pca.fit(X_scaled)

        explained = np.cumsum(pca.explained_variance_ratio_)

        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained)+1), explained, marker="o")
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Components")
        ax.set_ylabel("Variance")
        st.pyplot(fig)

    else:
        st.warning("Please upload data first.")

# ==============================
# 5️⃣ Clustering Analysis
# ==============================

elif section == "Clustering Analysis":

    if st.session_state.df is not None:

        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow
        inertia = []
        K_range = range(2, 11)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker="o")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Select k
        k_selected = st.slider("Select k", 2, 10, 3)

        kmeans = KMeans(n_clusters=k_selected, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        st.subheader("Cluster Summary")
        st.dataframe(df.groupby("Cluster").mean())

    else:
        st.warning("Please upload data first.")

# ==============================
# 6️⃣ Hierarchical Clustering
# ==============================

elif section == "Hierarchical Clustering":

    if st.session_state.df is not None:

        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        sample_data = X_scaled[:500]
        linked = linkage(sample_data, method="ward")

        fig = plt.figure(figsize=(10,5))
        dendrogram(linked)
        plt.title("Dendrogram")
        st.pyplot(fig)

    else:
        st.warning("Please upload data first.")

# ==============================
# 7️⃣ Marketing Insights
# ==============================

elif section == "Marketing Insights":

    if st.session_state.df is not None and "Cluster" in st.session_state.df.columns:

        df = st.session_state.df
        summary = df.groupby("Cluster").mean()

        for cluster in summary.index:
            st.subheader(f"Cluster {cluster}")

            if summary.iloc[cluster].mean() > df.mean().mean():
                st.write("💎 High Value Customers → Target Premium Products")
            else:
                st.write("💰 Budget Customers → Offer Discounts & Deals")

    else:
        st.warning("Run clustering first.")

# ==============================
# 8️⃣ About
# ==============================

elif section == "About":

    st.write("""
    ### Customer Segmentation & Marketing Analysis App
    
    This project demonstrates:
    - Data preprocessing
    - PCA dimensionality reduction
    - KMeans clustering
    - Hierarchical clustering
    - Marketing recommendations
    
    Developed using Streamlit & Scikit-Learn.
    """)
