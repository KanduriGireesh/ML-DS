import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

analyzer = SentimentIntensityAnalyzer()

DEFAULT_WEIGHTS = {
    "sentiment": 0.4,
    "experience": 0.3,
    "rating": 0.2,
    "cost": 0.1
}

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "doctors.csv"

def load_data(path: str = None) -> pd.DataFrame:
    p = DATA_PATH if path is None else Path(path)
    df = pd.read_csv(p)
    return df

def compute_sentiment(df: pd.DataFrame, review_col: str = "Reviews") -> pd.DataFrame:
    if "sentiment" in df.columns:
        return df
    df = df.copy()
    df["sentiment"] = df[review_col].fillna("").apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    return df

def compute_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    df = df.copy()
    df["exp_norm"] = df["Experience"] / (df["Experience"].max() if df["Experience"].max() > 0 else 1)
    df["rating_norm"] = df["Rating"] / 5.0
    df["cost_norm"] = df["Cost"] / (df["Cost"].max() if df["Cost"].max() > 0 else 1)

    df["final_score"] = (
        weights["sentiment"] * df["sentiment"] +
        weights["experience"] * df["exp_norm"] +
        weights["rating"] * df["rating_norm"] -
        weights["cost"] * df["cost_norm"]
    )

    minv, maxv = df["final_score"].min(), df["final_score"].max()
    if maxv - minv > 0:
        df["final_score_norm"] = (df["final_score"] - minv) / (maxv - minv)
    else:
        df["final_score_norm"] = df["final_score"].apply(lambda x: 0.5)

    return df

def recommend(df: pd.DataFrame, disease: str, city: str, budget: float, top_n: int = 3) -> pd.DataFrame:
    df2 = df.copy()
    mask = (
        df2["Specialty"].str.contains(disease, case=False, na=False) &
        df2["City"].str.contains(city, case=False, na=False) &
        (df2["Cost"] <= budget)
    )
    filtered = df2[mask]
    if filtered.empty:
        return filtered
    ranked = filtered.sort_values(by="final_score_norm", ascending=False).head(top_n)
    return ranked
