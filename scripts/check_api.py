"""Smoke-тест запущенного API: /health и /predict на примере из sample_request.json."""

import argparse
import json
import sys
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()

payload = json.loads((Path(__file__).parent / "sample_request.json").read_text(encoding="utf-8"))
health = requests.get(f"{args.url}/health", timeout=10)
print("GET /health ->", health.status_code, health.json())
resp = requests.post(f"{args.url}/predict", json=payload, timeout=10)
print("POST /predict ->", resp.status_code, resp.json())
ok = health.ok and resp.ok and {"default_prediction", "default_probability"} <= resp.json().keys()
sys.exit(0 if ok else 1)
