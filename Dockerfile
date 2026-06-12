# Start from a Python base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your app code
COPY src/ ./src/

# Start the FastAPI server when the container runs
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
