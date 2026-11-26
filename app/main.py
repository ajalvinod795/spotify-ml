from __future__ import annotations
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
from .ml import get_state, predict, NUMERIC, TARGET_THRESHOLD

BASE=Path(__file__).resolve().parents[1]
app=FastAPI(title="Spotify Hit & Mood Engine",version="1.0.0",
            description="Predict track hit potential and explore audio-feature mood clusters.")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
app.mount("/static",StaticFiles(directory=BASE/"frontend"),name="static")

class TrackInput(BaseModel):
    danceability:float=Field(.65,ge=0,le=1)
    energy:float=Field(.65,ge=0,le=1)
    key:int=Field(5,ge=0,le=11)
    loudness:float=-7
    mode:int=Field(1,ge=0,le=1)
    speechiness:float=Field(.05,ge=0,le=1)
    acousticness:float=Field(.2,ge=0,le=1)
    instrumentalness:float=Field(.0,ge=0,le=1)
    liveness:float=Field(.15,ge=0,le=1)
    valence:float=Field(.6,ge=0,le=1)
    tempo:float=Field(120,ge=30,le=250)
    time_signature:int=Field(4,ge=1,le=7)
    duration_ms:int=Field(210000,ge=30000,le=900000)
    track_genre:str="pop"

@app.get("/")
def home(): return FileResponse(BASE/"frontend"/"index.html")

@app.get("/api/health")
def health(): return {"status":"ok","service":"Spotify Hit & Mood Engine"}

@app.get("/api/overview")
def overview():
    s=get_state(); df=s["df"]
    return {"tracks":int(len(df)),"genres":int(df.track_genre.nunique()),
        "artists":int(df.artists.nunique()),"mean_popularity":round(float(df.popularity.mean()),1),
        "hit_rate":round(float((df.popularity>=TARGET_THRESHOLD).mean())*100,1),
        "mean_energy":round(float(df.energy.mean()),3),"mean_valence":round(float(df.valence.mean()),3),
        "mean_danceability":round(float(df.danceability.mean()),3),
        "genres_top":df.groupby("track_genre").size().sort_values(ascending=False).head(12).rename_axis("genre").reset_index(name="tracks").to_dict("records"),
        "popularity_bins":pd.cut(df.popularity,bins=[-1,20,40,60,80,100],labels=["0–20","21–40","41–60","61–80","81–100"]).value_counts().sort_index().rename_axis("range").reset_index(name="tracks").to_dict("records")}

@app.get("/api/models")
def model_metrics():
    s=get_state()
    return {"metrics":s["metrics"],"best_boosting_params":s["boost_params"],"target":"Popularity ≥ 60"}

@app.post("/api/predict")
def predict_hit(payload:TrackInput): return predict(payload.model_dump())

@app.get("/api/moods")
def moods():
    s=get_state()
    return {"points":s["mood_points"],"mood_names":s["mood_names"],
            "explained_variance":[round(float(x),4) for x in s["explained"]],
            "clusters":5}

@app.get("/api/tracks")
def tracks(q:Optional[str]=None,genre:Optional[str]=None,mood:Optional[str]=None,
           limit:int=Query(30,ge=1,le=100),sort:str="popularity"):
    df=get_state()["df"].copy()
    if q:
        mask=df.track_name.str.contains(q,case=False,na=False) | df.artists.str.contains(q,case=False,na=False)
        df=df[mask]
    if genre and genre!="all": df=df[df.track_genre==genre]
    allowed={"popularity","energy","danceability","valence","tempo"}
    if sort not in allowed: sort="popularity"
    df=df.sort_values(sort,ascending=False).head(limit)
    cols=["track_id","track_name","artists","album_name","track_genre","popularity","energy","danceability","valence","tempo","explicit"]
    return {"tracks":df[cols].fillna("").to_dict("records"),"total":int(len(df))}

@app.get("/api/genres")
def genres():
    df=get_state()["df"]
    return {"genres":sorted(df.track_genre.dropna().unique().tolist())}

@app.get("/api/recommend")
def recommend(mood:str="Feel Good",limit:int=12):
    s=get_state()
    pts=[p for p in s["mood_points"] if p["mood"].lower()==mood.lower()]
    pts=sorted(pts,key=lambda x:x["popularity"],reverse=True)[:max(1,min(limit,30))]
    return {"mood":mood,"tracks":pts}

@app.get("/api/feature-summary")
def feature_summary():
    df=get_state()["df"]
    features=["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]
    return {"features":[{"feature":c,"mean":round(float(df[c].mean()),3),
                         "median":round(float(df[c].median()),3)} for c in features]}
