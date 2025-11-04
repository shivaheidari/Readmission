# Stage 1: Install dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

# Copy and install dependencies to separate directory to reduce final image size
COPY ./services/explanation/requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: Final image with copy of dependencies and app code
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY ./services/explanation/api.py .

# Copy model artifacts
COPY ./model_artifacts/best_model ./model_artifacts/best_model

# Expose port
EXPOSE 8000

# Entrypoint command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
