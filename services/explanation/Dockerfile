# Start from a lean Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies for this specific service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service's API script
COPY api.py .

# Copy the shared model artifacts into a 'model' subfolder inside the container
COPY ../../models/best_model ./model/best_model

# Expose the port the API will run on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]