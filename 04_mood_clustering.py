from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load cleaned dataset
df = pd.read_csv(Path("data/processed/spotify_clean.csv"))

# 2. Select audio features that define musical mood
mood_candidates = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
mood_cols = [c for c in mood_candidates if c in df.columns]

# 3. Preprocess and fit K-Means (4 mood clusters)
prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_scaled = prep.fit_transform(df[mood_cols])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["mood_cluster"] = kmeans.fit_predict(X_scaled)

# 4. Inspect cluster averages so you can verify mood labels
cluster_summary = df.groupby("mood_cluster")[mood_cols].mean().round(2)
print("Cluster Feature Averages:")
print(cluster_summary[["energy", "valence", "danceability", "acousticness", "tempo"]])

# Automatically assign readable mood names based on energy and valence
mood_names = {}
med_energy = cluster_summary["energy"].median()
med_valence = cluster_summary["valence"].median()

for cid, row in cluster_summary.iterrows():
    high_e = row["energy"] >= med_energy
    high_v = row["valence"] >= med_valence
    if high_e and high_v:
        mood_names[cid] = "Energetic & Happy (Party/Upbeat)"
    elif high_e and not high_v:
        mood_names[cid] = "Intense & Moody (Workout/Dark)"
    elif not high_e and high_v:
        mood_names[cid] = "Chill & Acoustic (Feel-Good)"
    else:
        mood_names[cid] = "Calm & Melancholic (Sad/Ambient)"

df["mood_label"] = df["mood_cluster"].map(mood_names)
print("\nTrack counts per mood:")
print(df["mood_label"].value_counts())

# 5. PCA 2D projection for visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca_1"] = coords[:, 0]
df["pca_2"] = coords[:, 1]

Path("outputs").mkdir(exist_ok=True)
plt.figure(figsize=(10, 6))
for label, group in df.groupby("mood_label"):
    sample = group.sample(min(len(group), 1500), random_state=42)
    plt.scatter(sample["pca_1"], sample["pca_2"], label=label, alpha=0.5, s=12)

plt.title("Spotify Track Mood Clusters (PCA 2D Projection)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/mood_clusters_pca.png", dpi=150)
print("\nSaved PCA plot to outputs/mood_clusters_pca.png")

# 6. Save clustered dataset and mood pipeline for the UI
df.to_csv("data/processed/spotify_with_moods.csv", index=False)
Path("models").mkdir(exist_ok=True)
joblib.dump({
    "prep": prep,
    "kmeans": kmeans,
    "pca": pca,
    "mood_cols": mood_cols,
    "mood_names": mood_names
}, "models/mood_clusterer.joblib")
print("Saved mood model to models/mood_clusterer.joblib")