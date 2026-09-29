FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV & YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cloud_server.py .
COPY yolov8n.pt .

EXPOSE 3000

CMD ["uvicorn", "cloud_server:app", "--host", "0.0.0.0", "--port", "3000"]
