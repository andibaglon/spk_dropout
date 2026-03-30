import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="AI Dropout Dashboard",
    layout="wide",
    page_icon="🎓"
)

st.title("🎓 AI Dashboard: Prediksi Risiko Dropout Mahasiswa")

# =============================
# UPLOAD DATA
# =============================
file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # =============================
    # PREPROCESSING
    # =============================
    df_model = df.copy()
    le = LabelEncoder()

    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop(columns=['Dropout'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================
    # SIDEBAR CONTROL
    # =============================
    st.sidebar.header("⚙️ Pengaturan Model")
    k = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 3)

    # =============================
    # MODEL
    # =============================
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # =============================
    # SILHOUETTE SCORE
    # =============================
    score = silhouette_score(X_scaled, df['Cluster'])

    # =============================
    # AUTO LABEL RISK
    # =============================
    def label_risk(row):
        if 'IPK' in df.columns and 'Kehadiran' in df.columns:
            if row['IPK'] < 2.5 and row['Kehadiran'] < 75:
                return "High Risk"
            elif row['IPK'] < 3:
                return "Medium Risk"
            else:
                return "Low Risk"
        return "Unknown"

    df['Risk'] = df.apply(label_risk, axis=1)

    # =============================
    # KPI CARDS
    # =============================
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    high_risk = (df['Risk'] == "High Risk").sum()
    medium_risk = (df['Risk'] == "Medium Risk").sum()
    low_risk = (df['Risk'] == "Low Risk").sum()

    col1.metric("Total Mahasiswa", total)
    col2.metric("🔴 High Risk", high_risk)
    col3.metric("🟡 Medium Risk", medium_risk)
    col4.metric("🟢 Low Risk", low_risk)

    st.metric("Silhouette Score", round(score, 3))

    # =============================
    # CLUSTER VISUAL (PLOTLY)
    # =============================
    st.subheader("📍 Visualisasi Cluster (Interaktif)")

    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=df.columns,
        title="Cluster Visualization (PCA)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # DISTRIBUSI RISK
    # =============================
    st.subheader("📊 Distribusi Risiko")

    risk_fig = px.pie(
        df,
        names="Risk",
        title="Distribusi Risiko Dropout"
    )

    st.plotly_chart(risk_fig, use_container_width=True)

    # =============================
    # FILTER INTERAKTIF
    # =============================
    st.subheader("🔎 Filter Data")

    selected_cluster = st.selectbox("Pilih Cluster", sorted(df['Cluster'].unique()))
    filtered = df[df['Cluster'] == selected_cluster]

    st.dataframe(filtered.head())

    # =============================
    # PROFIL CLUSTER
    # =============================
    st.subheader("📈 Profil Cluster")

    profile = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(profile)

    # =============================
    # AUTO INSIGHT
    # =============================
    st.subheader("🧠 AI Insight")

    insight_text = ""

    for i in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == i]

        ipk = subset['IPK'].mean() if 'IPK' in df.columns else 0
        hadir = subset['Kehadiran'].mean() if 'Kehadiran' in df.columns else 0

        if ipk < 2.5 and hadir < 75:
            insight_text += f"Cluster {i} → Risiko tinggi (IPK & kehadiran rendah)\n"
        elif ipk < 3:
            insight_text += f"Cluster {i} → Risiko sedang\n"
        else:
            insight_text += f"Cluster {i} → Risiko rendah\n"

    st.text(insight_text)

    # =============================
    # ELBOW METHOD
    # =============================
    st.subheader("📉 Elbow Method")

    inertia = []
    K_range = range(2, 8)

    for k_ in K_range:
        km = KMeans(n_clusters=k_, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    elbow_fig = px.line(
        x=list(K_range),
        y=inertia,
        markers=True,
        title="Elbow Method"
    )

    st.plotly_chart(elbow_fig, use_container_width=True)

    # =============================
    # DOWNLOAD
    # =============================
    st.subheader("⬇️ Export Data")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Hasil", csv, "hasil_dropout_cluster.csv")

else:
    st.info("Upload dataset untuk memulai dashboard.")