#!/usr/bin/env python3
"""
cloud_server.py

FastAPI backend for ESP32-CAM Dog Detection.
Deploy this to Render / Railway / VPS / EC2.

Features:
- Live YOLOv8 Dog Detection on received JPEG frames
- Sends instant alert notifications via:
  1. Ntfy (Free instant push notification on phone app)
  2. Telegram Bot (Free instant Telegram msg with photo)
  3. Twilio SMS / WhatsApp (Direct alert SMS to mobile number)
"""

import os
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from ultralytics import YOLO

app = FastAPI(title="ESP32 Dog Detector Cloud API")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
SECRET_KEY = os.getenv("ALERT_SECRET", "umesh-dog-secret")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.15"))  # Lower threshold for high sensitivity
DOG_CLASS_ID = 16  # COCO class 16 = dog

# Notification Settings
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "umesh-dog-car-alert")

# Telegram Bot (Optional - set env variables to enable)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Twilio SMS / WhatsApp (Optional - set env variables to enable)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")  # e.g., '+1234567890' or 'whatsapp:+14155238886'
TWILIO_TO = os.getenv("TWILIO_TO", "")      # e.g., '+919876543210' or 'whatsapp:+919876543210'

# Load YOLO model on startup
model = YOLO(MODEL_PATH)


def send_alerts(accuracy_percent: int, image_path: str):
    alert_msg = f"⚠️ DOG DETECTED ON CAR/PREMISES! Confidence: {accuracy_percent}%"

    # 1. Ntfy Push Notification (Free, High Priority sound alert)
    if NTFY_TOPIC:
        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=alert_msg,
                headers={
                    "Title": "🚨 Dog Alert!",
                    "Priority": "high",
                    "Tags": "dog,warning,alert"
                },
                timeout=5
            )
            print(f"[ALERT] Ntfy push sent to topic: {NTFY_TOPIC}")
        except Exception as e:
            print(f"[ERROR] Ntfy Push Failed: {e}")

    # 2. Telegram Bot Alert (Free msg + photo to Telegram chat)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": alert_msg},
                    files={"photo": photo},
                    timeout=5
                )
            print("[ALERT] Telegram notification sent!")
        except Exception as e:
            print(f"[ERROR] Telegram Alert Failed: {e}")

    # 3. Twilio SMS / WhatsApp Alert
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM and TWILIO_TO:
        try:
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            requests.post(
                twilio_url,
                data={
                    "From": TWILIO_FROM,
                    "To": TWILIO_TO,
                    "Body": alert_msg
                },
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=5
            )
            print(f"[ALERT] Twilio SMS/WhatsApp sent to {TWILIO_TO}")
        except Exception as e:
            print(f"[ERROR] Twilio SMS Failed: {e}")


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "ESP32 Standalone Cloud Dog Detection Server Running"
    }


@app.post("/detect")
async def detect(request: Request, x_alert_secret: str = Header(None)):
    # 1. Security Check
    if x_alert_secret != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid Secret Key")

    # 2. Read JPEG frame bytes from ESP32
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame received")

    # 3. Save temporary frame
    temp_filename = "temp_frame.jpg"
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)

    # 4. Predict using YOLOv8 (conf threshold 0.10 for high sensitivity)
    results = model.predict(
        source=temp_filename,
        conf=0.10,
        verbose=False
    )[0]

    dog_found = False
    best_score = 0.0
    all_seen = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            all_seen.append(f"{cls_name}:{round(score, 2)}")
            
            if cls_id == DOG_CLASS_ID:
                dog_found = True
                if score > best_score:
                    best_score = score

    # 5. Send Notification if dog is detected
    if dog_found:
        accuracy_percent = int(best_score * 100)
        send_alerts(accuracy_percent, temp_filename)

    return {
        "dog": dog_found,
        "score": round(best_score, 2),
        "seen": all_seen,
        "message": "Dog detected & Mobile alert sent!" if dog_found else "No dog detected"
    }

