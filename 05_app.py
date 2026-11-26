import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Spotify Hit & Mood Engine", page_icon="🎧", layout="wide")

# Load saved models & processed dataset
@st.cache_resource
def load_models():
    hit_bundle = joblib.load("models/hit_classifier.joblib")
    mood_bundle = joblib.load("models/mood_clusterer.joblib")
    return hit_bundle, mood_bundle

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/spotify_with_moods.csv")
    df["language"] = df.apply(detect_language, axis=1)
    return df

MOOD_NAMES = {
    0: "Happy & Energetic",
    1: "Moody & Energetic",
    2: "Chill & Acoustic",
    3: "Ambient & Instrumental"
}

MOOD_DESCRIPTIONS = {
    0: "Happy & Energetic (High danceability)",
    1: "Moody & Energetic (Fast & intense)",
    2: "Chill & Acoustic (Laid-back vibe)",
    3: "Ambient & Instrumental (Calm & peaceful)"
}

COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

# Helper to detect Indian/regional language tags from track/album/genre text
def detect_language(row):
    text = f"{row.get('track_name', '')} {row.get('album_name', '')} {row.get('track_genre', '')}".lower()
    if "telugu" in text:
        return "Telugu"
    elif "tamil" in text:
        return "Tamil"
    elif "hindi" in text or "bollywood" in text:
        return "Hindi"
    elif "malayalam" in text:
        return "Malayalam"
    elif "punjabi" in text:
        return "Punjabi"
    elif "kannada" in text:
        return "Kannada"
    elif "spanish" in text or "latin" in text:
        return "Spanish"
    elif "korean" in text or "k-pop" in text:
        return "Korean"
    return "Other / not specified"

hit_bundle, mood_bundle = load_models()
hit_model = hit_bundle["pipeline"]
hit_features = hit_bundle["features"]

mood_scaler = mood_bundle["prep"]
pca = mood_bundle["pca"]
kmeans = mood_bundle["kmeans"]
mood_features = mood_bundle["mood_cols"]

df = load_data()

st.title("🎧 Spotify Hit & Mood Engine")
st.write("Predict hit potential, classify track mood, and explore PCA mood clusters across languages.")

# --- SIDEBAR: Audio Feature Controls ---
st.sidebar.header("🎛️ Track Audio Controls")
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.65, 0.01)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.72, 0.01)
valence = st.sidebar.slider("Valence (Positivity)", 0.0, 1.0, 0.58, 0.01)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.15, 0.01)
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -6.5, 0.5)
tempo = st.sidebar.slider("Tempo (BPM)", 40.0, 220.0, 120.0, 1.0)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.08, 0.01)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.01, 0.01)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
duration_min = st.sidebar.slider("Duration (minutes)", 0.5, 10.0, 3.5, 0.1)

genres = sorted(df["track_genre"].dropna().unique().tolist()) if "track_genre" in df.columns else ["pop"]
track_genre = st.sidebar.selectbox("Genre", genres)
explicit = st.sidebar.selectbox("Explicit", ["False", "True"])

# Build input row matching training features
sample_row = {
    "duration_ms": duration_min * 60_000,
    "danceability": danceability,
    "energy": energy,
    "key": 5,
    "loudness": loudness,
    "mode": 1,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "time_signature": 4,
    "explicit": explicit,
    "track_genre": track_genre,
}

# 1. Hit Prediction
hit_input = pd.DataFrame([{c: sample_row.get(c) for c in hit_features}])
hit_prob = float(hit_model.predict_proba(hit_input)[0][1])
hit_label = "🔥 Hit Potential" if hit_prob >= 0.5 else "📉 Not a Hit"

# 2. Mood Prediction & PCA Coordinates
mood_input = pd.DataFrame([{c: sample_row[c] for c in mood_features}])
mood_scaled = mood_scaler.transform(mood_input)
cluster_id = int(kmeans.predict(mood_scaled)[0])
user_pca = pca.transform(mood_scaled)[0]

# --- TOP METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Hit Prediction", hit_label, f"{hit_prob * 100:.1f}% probability")
col2.metric("Mood Cluster", MOOD_NAMES[cluster_id])
col3.info(f"**Vibe:** {MOOD_DESCRIPTIONS[cluster_id]}")

st.divider()

# --- MAIN SECTION: Matplotlib PCA Cluster Graph + Language Filter ---
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.subheader("📊 Mood Clusters (PCA 2D Projection)")
    fig, ax = plt.subplots(figsize=(8, 5))

    for cid in sorted(MOOD_NAMES.keys()):
        subset = df[df["mood_cluster"] == cid]
        sample = subset.sample(min(len(subset), 1200), random_state=42)
        ax.scatter(
            sample["pca_1"],
            sample["pca_2"],
            c=COLORS[cid],
            label=MOOD_NAMES[cid],
            alpha=0.45,
            s=14
        )

    # Plot the user's current slider track on the cluster graph
    ax.scatter(
        user_pca[0],
        user_pca[1],
        c="black",
        marker="*",
        s=260,
        edgecolors="white",
        linewidths=1.5,
        label="Your Custom Track",
        zorder=5
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Spotify Mood Clusters & Your Track Position")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

with right_col:
    st.subheader("🌏 Browse Tracks by Mood & Language")
    selected_mood_id = st.selectbox(
        "Filter by Mood",
        options=list(MOOD_NAMES.keys()),
        format_func=lambda x: MOOD_DESCRIPTIONS[x],
        index=cluster_id
    )

    available_langs = ["All"] + sorted(df["language"].unique().tolist())
    selected_lang = st.selectbox("Filter by Language", available_langs)

    filtered = df[df["mood_cluster"] == selected_mood_id]
    if selected_lang != "All":
        filtered = filtered[filtered["language"] == selected_lang]

    st.caption(f"Showing {min(len(filtered), 30)} of {len(filtered)} matching tracks")
    show_cols = [c for c in ["track_name", "artists", "language", "popularity", "energy", "valence"] if c in filtered.columns]
    st.dataframe(filtered[show_cols].head(30), use_container_width=True, height=360)