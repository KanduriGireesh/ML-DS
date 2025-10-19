import pandas as pd
import random
from pathlib import Path

CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Delhi"]
SPECIALTIES = ["Cardiology", "Neurology", "Orthopedics", "Dermatology", "Pediatrics"]

POS_REVIEWS = [
    "Very patient and skilled doctor.",
    "Explained everything well and helped a lot.",
    "Friendly staff and effective treatment.",
    "Great experience; highly recommended.",
    "Good follow-up and clear communication."
]
NEG_REVIEWS = [
    "Very long waiting time.",
    "Rude staff and not helpful.",
    "Treatment was expensive and results were average.",
    "Did not explain the procedure clearly.",
    "Unprofessional behaviour from the clinic."
]
MIXED_REVIEWS = [
    "Doctor is skilled but the clinic is expensive.",
    "Good treatment but long wait times.",
    "Helpful doctor but appointment scheduling is hard.",
]

NUM_RECORDS = 200
OUTFILE = Path(__file__).resolve().parents[0] / "doctors.csv"

rows = []
for i in range(NUM_RECORDS):
    city = random.choice(CITIES)
    spec = random.choice(SPECIALTIES)
    doc_name = f"Dr. {random.choice(['Rao','Mehta','Kumar','Sharma','Singh','Iyer','Patel','Reddy','Varma','Nair'])} {random.randint(1,999)}"
    hospital = random.choice(["Apollo","Yashoda","Fortis","Max","Manipal","Aster","Medanta"]) + " Hospital"
    exp = random.randint(1, 40)
    rating = round(random.uniform(3.0, 5.0), 1)
    cost = random.choice([300, 400, 500, 600, 700, 800, 1000, 1200, 1500])

    if rating >= 4.3:
        review = random.choice(POS_REVIEWS + MIXED_REVIEWS)
    elif rating >= 3.7:
        review = random.choice(MIXED_REVIEWS + POS_REVIEWS + NEG_REVIEWS)
    else:
        review = random.choice(NEG_REVIEWS + MIXED_REVIEWS)

    rows.append({
        "Doctor": doc_name,
        "Hospital": hospital,
        "City": city,
        "Specialty": spec,
        "Experience": exp,
        "Rating": rating,
        "Cost": cost,
        "Reviews": review
    })

df = pd.DataFrame(rows)
df.to_csv(OUTFILE, index=False)
print(f"Saved synthetic dataset to: {OUTFILE}")
