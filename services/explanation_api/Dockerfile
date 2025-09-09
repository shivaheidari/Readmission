# 1. Start from a lean Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code and the trained model artifacts
COPY api.py .
# This copies the trained model into a 'model' folder inside this container
COPY ../models/best_model ./model

# 5. Expose the port the API will run on
EXPOSE 8000

# 6. Define the command to run your application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]