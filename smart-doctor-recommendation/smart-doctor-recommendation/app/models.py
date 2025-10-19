from pydantic import BaseModel
from typing import Optional

class RecommendRequest(BaseModel):
    disease: str
    city: str
    budget: float
    top_n: Optional[int] = 3

class DoctorOut(BaseModel):
    Doctor: str
    Hospital: str
    Specialty: str
    Experience: int
    Rating: float
    Cost: int
    Reviews: str
    sentiment: float
    final_score_norm: float
