from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load cleaned dataset
df = pd.read_csv(Path("data/processed/spotify_clean.csv"))

# 2. Define target label (hit vs not-hit)
if "hit" in df.columns:
    y = df["hit"].astype(int)
elif "popularity" in df.columns:
    # Standard threshold: popularity >= 70 counts as a hit (adjust if needed)
    y = (df["popularity"] >= 70).astype(int)
else:
    raise ValueError(f"No 'hit' or 'popularity' column found. Columns: {df.columns.tolist()}")

print("Class balance (0 = Not Hit, 1 = Hit):")
print(y.value_counts(normalize=True).round(3))

# 3. Select audio & metadata features (exclude target and ID/text columns)
numeric_candidates = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature"
]
categorical_candidates = ["explicit", "track_genre"]

num_cols = [c for c in numeric_candidates if c in df.columns]
cat_cols = [c for c in categorical_candidates if c in df.columns]

X = df[num_cols + cat_cols].copy()
if "explicit" in cat_cols:
    X["explicit"] = X["explicit"].astype(str)

# 4. Stratified train/test split (fits preprocessing on training set only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

# 6. Train and compare models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

best_model = None
best_f1 = -1.0

for name, clf in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds, digits=3))
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipe

# 7. Save best classifier for your UI later
Path("models").mkdir(exist_ok=True)
joblib.dump({"pipeline": best_model, "features": num_cols + cat_cols}, "models/hit_classifier.joblib")
print("Saved best hit classifier to models/hit_classifier.joblib")