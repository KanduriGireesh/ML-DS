#!/usr/bin/env bash
BASE="http://127.0.0.1:8000"

echo "Health check:"
curl -s "$BASE/" | jq

echo "Recommend GET:"
curl -s "$BASE/recommend?disease=Cardiology&city=Hyderabad&budget=1000&top_n=2" | jq

echo "Recommend POST:"
curl -s -X POST "$BASE/recommend" -H 'Content-Type: application/json' -d '{"disease":"Cardiology","city":"Hyderabad","budget":1000,"top_n":2}' | jq
