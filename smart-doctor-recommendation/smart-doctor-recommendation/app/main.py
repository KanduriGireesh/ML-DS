from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.utils import load_data, compute_sentiment, compute_scores, recommend
from app.models import RecommendRequest, DoctorOut
from typing import List
import pandas as pd

app = FastAPI(title="Smart Doctor Recommendation API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = load_data()
df = compute_sentiment(df)
df = compute_scores(df)

@app.get("/", tags=["health"])
def home():
    return {"message": "Smart Doctor Recommendation API is running.", "records": len(df)}

@app.post("/recommend", response_model=List[DoctorOut], tags=["recommend"])
def recommend_post(req: RecommendRequest):
    results = recommend(df, req.disease, req.city, req.budget, top_n=req.top_n)
    if results.empty:
        raise HTTPException(status_code=404, detail="No matching doctors found for given criteria.")
    out = results.to_dict(orient="records")
    return out

@app.get("/recommend", response_model=List[DoctorOut], tags=["recommend"])
def recommend_get(
    disease: str = Query(...),
    city: str = Query(...),
    budget: float = Query(...),
    top_n: int = Query(3)
):
    results = recommend(df, disease, city, budget, top_n=top_n)
    if results.empty:
        raise HTTPException(status_code=404, detail="No matching doctors found for given criteria.")
    return results.to_dict(orient="records")

@app.get("/doctors", tags=["data"])
def list_doctors(city: str = None, specialty: str = None, limit: int = 50):
    tmp = df.copy()
    if city:
        tmp = tmp[tmp["City"].str.contains(city, case=False, na=False)]
    if specialty:
        tmp = tmp[tmp["Specialty"].str.contains(specialty, case=False, na=False)]
    return tmp.head(limit).to_dict(orient="records")

@app.post("/add_review", tags=["data"])
def add_review(doctor_name: str = Query(...), review: str = Query(...)):
    global df
    idx = df[ df["Doctor"].str.contains(doctor_name, case=False, na=False) ].index
    if idx.empty:
        raise HTTPException(status_code=404, detail="Doctor not found")

    i = idx[0]
    existing = str(df.at[i, "Reviews"]) if pd.notna(df.at[i, "Reviews"]) else ""
    df.at[i, "Reviews"] = existing + " \n" + review

    df = compute_sentiment(df)
    df = compute_scores(df)

    return {"message": "Review added and scores recomputed for dataset."}
