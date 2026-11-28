# Spotify Hit & Mood Engine

A complete full-stack ML dashboard built around the supplied Spotify tracks dataset (114,000 rows).

## Features
- Hit classification: Logistic Regression, K-Nearest Neighbors, and hyperparameter-tuned HistGradientBoostingClassifier.
- Hit definition: Spotify popularity **>= 60**. The predictor reports each model's probability and their mean.
- Mood discovery: standardized audio features -> PCA 2D projection -> K-Means (5 clusters), with descriptive mood labels.
- Interactive dashboard: dataset KPIs, popularity/genre charts, Plotly mood map, hit predictor, searchable track explorer, and model evaluation metrics.
- FastAPI backend and responsive HTML/CSS/JavaScript frontend.

## Quick start

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the website
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Interactive API docs: http://127.0.0.1:8000/docs.

## Jupyter notebook

Open `spotify_hit_mood_analysis.ipynb` in VS Code, select the Python environment where the project requirements are installed, and run the cells from top to bottom. The notebook explores the dataset, trains and evaluates the hit classifiers, and visualizes the mood clusters. Model training can take a few minutes.

The first request that needs ML results trains the models and saves fitted estimators plus PCA/K-Means objects to `artifacts/trained_models.joblib`. The training sample is stratified and capped at 24,000 rows to keep first-run setup practical; the full dataset is used for dashboard statistics and track exploration. The PCA/K-Means projection uses up to 7,000 tracks.

## API
- `GET /api/health` – service health
- `GET /api/overview` – dataset KPIs and chart aggregates
- `GET /api/genres` – genre options
- `GET /api/models` – holdout metrics and best boosting parameters
- `POST /api/predict` – hit probabilities from audio features
- `GET /api/moods` – PCA coordinates and K-Means mood labels
- `GET /api/tracks?q=&genre=&sort=&limit=` – searchable track data
- `GET /api/recommend?mood=Feel%20Good&limit=12` – popular tracks from a mood cluster

## Model methodology
- Target: `1` when popularity >= 60, otherwise `0`.
- Train/test: reproducible 78/22 stratified split from a 24k sample.
- Logistic Regression and KNN use median imputation, standard scaling, and one-hot genre encoding.
- Tuned Boosting uses HistGradientBoosting with ordinal genre encoding and randomized 3-fold CV over learning rate, iterations, leaf nodes, and L2 regularization.
- Metrics: accuracy, precision, recall, F1, ROC-AUC.
- Mood features: danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo. PCA is for visualization; K-Means uses standardized feature vectors.

## Project structure
```
spotify-hit-mood-engine/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── ml.py
├── artifacts/                 # trained models can be exported here
├── data/spotify_tracks.csv    # uploaded dataset
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── spotify_hit_mood_analysis.ipynb
├── tests/
├── requirements.txt
└── README.md
```

## Notes
- Charts load Chart.js and Plotly.js from public CDNs; an internet connection is needed for those chart libraries and Google Fonts. Core API/model functionality runs locally.
- The dataset's `track_id` values are identifiers only; this project does not call the Spotify API or stream copyrighted audio.
- For a production deployment, restrict CORS, add authentication/rate limiting, persist fitted artifacts, and pin dependency versions.
