FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4
WORKDIR /app

COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-api.txt

COPY scripts/recommender_api.py scripts/recommender_inference.py scripts/discovery.py ./scripts/
COPY docker/check_models.py ./docker/check_models.py

EXPOSE 8000
CMD ["sh", "-c", "python docker/check_models.py && exec python -X utf8 scripts/recommender_api.py --host 0.0.0.0 --port 8000"]

