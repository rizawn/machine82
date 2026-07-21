FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY ./api_gateway/requirements.txt ./api_gateway_requirements.txt
COPY ./MLRL01/requirements.txt ./MLRL01_requirements.txt
COPY ./MLRL02/requirements.txt ./MLRL02_requirements.txt

# Install all python packages
RUN pip install --no-cache-dir -r ./api_gateway_requirements.txt
RUN pip install --no-cache-dir -r ./MLRL01_requirements.txt
RUN pip install --no-cache-dir -r ./MLRL02_requirements.txt

# Copy source directories
COPY ./api_gateway ./api_gateway
COPY ./MLRL01 ./MLRL01
COPY ./MLRL02 ./MLRL02

# Environment variables setup
ENV PYTHONPATH=/app:/app/api_gateway:/app/MLRL01:/app/MLRL02

WORKDIR /app/api_gateway

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
