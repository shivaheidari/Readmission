# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

COPY ./services/prediction/requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY ./services/prediction/api.py .
COPY ./services/prediction/Model ./model_artifacts/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
