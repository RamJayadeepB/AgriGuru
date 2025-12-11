FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for wget/curl if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Make start script executable
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8501

# Streamlit config
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["bash", "-lc", "/app/start.sh"]

