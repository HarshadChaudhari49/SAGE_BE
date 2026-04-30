FROM python:3.11-slim

# Runtime configuration for hosted Docker platforms.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Install minimal OS build tools needed by Python ML packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, and saved models.
COPY . .

EXPOSE 7860

# Start only the deployable Flask dashboard server.
CMD ["python", "simulation_server.py"]
