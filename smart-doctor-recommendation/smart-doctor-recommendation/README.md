# Smart Doctor & Hospital Recommendation (FastAPI)
Full demo project that recommends doctors/hospitals by combining sentiment (from reviews), experience, rating and cost.

## Run locally
1. Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. (Optional) regenerate dataset:
   ```bash
   python data/generate_synthetic_data.py
   ```
3. Start server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
4. Open Swagger UI: http://127.0.0.1:8000/docs

## Endpoints
- `GET /` - health
- `GET /recommend?disease=&city=&budget=&top_n=` - recommend doctors (query params)
- `POST /recommend` - JSON body recommendation
- `GET /doctors` - list doctors
- `POST /add_review` - add review to a doctor (in-memory update)

## Notes
Uses `vaderSentiment` for quick sentiment scoring. For production, persist data to DB and use background reindexing.
