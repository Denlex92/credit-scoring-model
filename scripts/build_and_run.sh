#!/usr/bin/env bash
# Сборка Docker-образа с API и локальный запуск контейнера.
# Перед сборкой модель должна быть обучена: dvc repro (или python -m src.models.train).
set -euo pipefail

IMAGE="${IMAGE:-credit-default-api}"
TAG="${TAG:-latest}"
PORT="${PORT:-8000}"

if [ ! -f models/credit_default_model.pkl ]; then
  echo "Модель не найдена. Сначала выполните: dvc repro" >&2
  exit 1
fi

docker build -t "${IMAGE}:${TAG}" .
docker rm -f credit-default-api >/dev/null 2>&1 || true
docker run -d --name credit-default-api -p "${PORT}:8000" "${IMAGE}:${TAG}"

echo "Ожидание запуска API..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
python scripts/check_api.py --url "http://localhost:${PORT}"
echo "Swagger UI: http://localhost:${PORT}/docs  |  остановить: docker rm -f credit-default-api"
