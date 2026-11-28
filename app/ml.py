"""Training and inference for the Spotify Hit & Mood Engine."""
from __future__ import annotations
import os, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("SPOTIFY_DATA_PATH", BASE / "data" / "spotify_tracks.csv"))
ARTIFACT_DIR = Path("/tmp/spotify-hit-mood-engine-artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
NUMERIC = ["danceability","energy","key","loudness","mode","speechiness","acousticness",
           "instrumentalness","liveness","valence","tempo","time_signature","duration_ms"]
AUDIO = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
TARGET_THRESHOLD = 60
_STATE = None

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    for c in NUMERIC + ["popularity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["track_genre"] = df["track_genre"].fillna("unknown").astype(str)
    df["artists"] = df["artists"].fillna("Unknown artist")
    df["track_name"] = df["track_name"].fillna("Untitled")
    df = df.dropna(subset=["popularity"])
    return df

def _metrics(y, pred, proba):
    return {
        "accuracy": round(float(accuracy_score(y,pred)),4),
        "precision": round(float(precision_score(y,pred,zero_division=0)),4),
        "recall": round(float(recall_score(y,pred,zero_division=0)),4),
        "f1": round(float(f1_score(y,pred,zero_division=0)),4),
        "roc_auc": round(float(roc_auc_score(y,proba)),4)
    }

def train_models(force=False):
    global _STATE
    if _STATE is not None and not force:
        return _STATE
    df = load_data()
    # Use a reproducible, stratified training sample to keep first-run setup practical.
    work = df.sample(n=min(24000, len(df)), random_state=42)
    X = work[NUMERIC + ["track_genre"]].copy()
    y = (work["popularity"] >= TARGET_THRESHOLD).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.22, random_state=42, stratify=y
    )
    numeric_pipe = Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    prep = ColumnTransformer([
        ("num",numeric_pipe,NUMERIC),
        ("genre",OneHotEncoder(handle_unknown="ignore"),["track_genre"])
    ])
    models = {
        "Logistic Regression": Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1200,class_weight="balanced"))]),
        "KNN": Pipeline([("prep",prep),("model",KNeighborsClassifier(n_neighbors=13,weights="distance",n_jobs=-1))])
    }
    for model in models.values():
        model.fit(X_train,y_train)
    # Boosting uses compact numeric + ordinal-genre features for efficient tuning.
    boost_prep = ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),NUMERIC),
        ("genre",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                           ("ordinal",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),["track_genre"])
    ])
    boost_pipe = Pipeline([("prep",boost_prep),("model",HistGradientBoostingClassifier(random_state=42))])
    search = RandomizedSearchCV(
        boost_pipe,
        {"model__learning_rate":[.04,.07,.1], "model__max_iter":[100,160,220],
         "model__max_leaf_nodes":[15,31], "model__l2_regularization":[0,.5,1.0]},
        n_iter=5, cv=3, scoring="f1", random_state=42, n_jobs=-1, refit=True
    )
    search.fit(X_train,y_train)
    models["Tuned Boosting"] = search.best_estimator_
    metrics = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:,1]
        metrics[name] = _metrics(y_test,pred,proba)
    # Mood clustering and PCA on a reproducible sample.
    mood_df = df.dropna(subset=AUDIO).sample(n=min(7000,len(df.dropna(subset=AUDIO))),random_state=17)
    scaler = StandardScaler()
    Z = scaler.fit_transform(mood_df[AUDIO])
    kmeans = KMeans(n_clusters=5,random_state=42,n_init=10)
    labels = kmeans.fit_predict(Z)
    pca = PCA(n_components=2,random_state=42)
    coords = pca.fit_transform(Z)
    centers = mood_df.assign(cluster=labels).groupby("cluster")[["energy","valence","danceability","acousticness"]].mean()
    def mood_name(row):
        if row["energy"] >= .68 and row["valence"] >= .55: return "Euphoric"
        if row["energy"] >= .68: return "High Energy"
        if row["valence"] >= .58: return "Feel Good"
        if row["acousticness"] >= .55: return "Chill & Acoustic"
        return "Melancholic"
    mood_names = {int(i):mood_name(centers.loc[i]) for i in centers.index}
    mood_points = []
    for i, (_, row) in enumerate(mood_df.iterrows()):
        mood_points.append({
            "x": round(float(coords[i,0]),3), "y":round(float(coords[i,1]),3),
            "cluster":int(labels[i]), "mood":mood_names[int(labels[i])],
            "track":str(row["track_name"]), "artist":str(row["artists"]),
            "popularity":int(row["popularity"]), "genre":str(row["track_genre"])
        })
    # Persist fitted estimators and the clustering/projection objects for reproducibility.
    joblib.dump({
        "models": models, "metrics": metrics, "boost_params": search.best_params_,
        "scaler": scaler, "kmeans": kmeans, "pca": pca,
        "mood_names": mood_names, "explained": pca.explained_variance_ratio_.tolist(),
        "target_threshold": TARGET_THRESHOLD, "feature_columns": NUMERIC + ["track_genre"]
    }, ARTIFACT_DIR / "trained_models.joblib", compress=3)
    _STATE = {"df":df,"models":models,"metrics":metrics,"boost_params":search.best_params_,
             "scaler":scaler,"kmeans":kmeans,"pca":pca,"mood_names":mood_names,
             "mood_points":mood_points,"explained":pca.explained_variance_ratio_.tolist()}
    return _STATE

def feature_row(payload):
    row = {c: payload.get(c) for c in NUMERIC}
    row["track_genre"] = payload.get("track_genre","pop")
    for c in NUMERIC:
        if row[c] is None: row[c] = float(load_data()[c].median())
    return pd.DataFrame([row],columns=NUMERIC+["track_genre"])

def predict(payload):
    state=train_models()
    X=feature_row(payload)
    result={}
    for name,model in state["models"].items():
        p=float(model.predict_proba(X)[0,1])
        result[name]={"hit_probability":round(p,4),"prediction":"Hit" if p>=.5 else "Not a hit"}
    return {"threshold":TARGET_THRESHOLD,"models":result,
            "consensus_probability":round(float(np.mean([v["hit_probability"] for v in result.values()])),4)}

def get_state():
    return train_models()
