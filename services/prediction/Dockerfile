FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies for this specific service
COPY /services/prediction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service's API script
COPY ./services/prediction/api.py .

COPY ./model_artifacts/best_model ./model_artifacts/best_model

# Expose the port the API will run on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]