FROM python:3.10-slim

# Install system packages required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cloud_server.py .
COPY yolov8n.pt .

EXPOSE 3000

CMD ["uvicorn", "cloud_server:app", "--host", "0.0.0.0", "--port", "3000"]
