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
import time
from urllib.parse import quote
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
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

LATEST_JPEG = b""
LATEST_AT = 0.0
LAST_DOG = False
LAST_SCORE = 0.0
LAST_SEEN = []
LAST_ALERT_AT = 0.0
CAM_IP = ""


def app_public_url() -> str:
    domain = os.getenv(
        "RAILWAY_PUBLIC_DOMAIN",
        "esp32-dog-detector-production.up.railway.app",
    )
    return f"https://{domain}/app?k={quote(SECRET_KEY, safe='')}"


def secret_ok(value: str) -> bool:
    return bool(value) and value == SECRET_KEY


def remember_frame(image_bytes: bytes, cam_ip: str = ""):
    global LATEST_JPEG, LATEST_AT, CAM_IP
    if image_bytes:
        LATEST_JPEG = image_bytes
        LATEST_AT = time.time()
    if cam_ip:
        CAM_IP = cam_ip.strip()


def send_alerts(accuracy_percent: int, image_path: str):
    alert_msg = f"Dog detected. Confidence: {accuracy_percent}%. Tap Live stream to open the camera."
    alert_status = []

    # 1. Ntfy Push Notification (Free, High Priority sound alert)
    if NTFY_TOPIC:
        try:
            image_bytes = b""
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as photo_file:
                    image_bytes = photo_file.read()
            live_link = app_public_url()
            headers = {
                "Title": "Dog Alert!",
                "Priority": "high",
                "Tags": "dog,warning,alert",
                "Click": live_link,
                "Actions": f"view, Live stream, {live_link}",
            }
            body = alert_msg
            if image_bytes:
                headers["Filename"] = "dog.jpg"
                headers["Message"] = alert_msg
                headers["Content-Type"] = "image/jpeg"
                body = image_bytes
            r = requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=body,
                headers=headers,
                timeout=10
            )
            alert_status.append(f"ntfy:{r.status_code}")
            print(f"[ALERT] Ntfy HTTP {r.status_code}")
        except Exception as e:
            alert_status.append(f"ntfy:FAIL:{e}")
            print(f"[ERROR] Ntfy Push Failed: {e}")

    # 2. Telegram Bot Alert (Free msg + photo to Telegram chat)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                r = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": alert_msg},
                    files={"photo": photo},
                    timeout=5
                )
            alert_status.append(f"telegram:{r.status_code}")
        except Exception as e:
            alert_status.append(f"telegram:FAIL:{e}")

    # 3. Twilio SMS / WhatsApp Alert
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM and TWILIO_TO:
        try:
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            r = requests.post(
                twilio_url,
                data={
                    "From": TWILIO_FROM,
                    "To": TWILIO_TO,
                    "Body": alert_msg
                },
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=5
            )
            alert_status.append(f"twilio:{r.status_code}")
        except Exception as e:
            alert_status.append(f"twilio:FAIL:{e}")

    return alert_status


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "ESP32 Standalone Cloud Dog Detection Server Running",
        "version": "2.4",
        "app": "/app"
    }


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({
        "name": "Dog Watch",
        "short_name": "Dog Watch",
        "start_url": "/app",
        "display": "standalone",
        "background_color": "#090b10",
        "theme_color": "#090b10",
    })


@app.get("/app", response_class=HTMLResponse)
def phone_app():
    return HTMLResponse(APP_HTML)


@app.get("/live.jpg")
def live_jpg(x_alert_secret: str = Header(None)):
    if not secret_ok(x_alert_secret):
        raise HTTPException(status_code=401, detail="Invalid Secret Key")
    if not LATEST_JPEG:
        raise HTTPException(status_code=404, detail="No frame yet")
    return Response(
        content=LATEST_JPEG,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/status")
def live_status(x_alert_secret: str = Header(None)):
    if not secret_ok(x_alert_secret):
        raise HTTPException(status_code=401, detail="Invalid Secret Key")
    age = None if not LATEST_AT else round(time.time() - LATEST_AT, 1)
    cam_url = f"http://{CAM_IP}/" if CAM_IP else ""
    return {
        "has_frame": bool(LATEST_JPEG),
        "age_sec": age,
        "dog": LAST_DOG,
        "score": LAST_SCORE,
        "seen": LAST_SEEN,
        "last_alert_age_sec": None if not LAST_ALERT_AT else round(time.time() - LAST_ALERT_AT, 1),
        "cam_url": cam_url,
    }


@app.post("/frame")
async def frame(request: Request, x_alert_secret: str = Header(None)):
    if not secret_ok(x_alert_secret):
        raise HTTPException(status_code=401, detail="Invalid Secret Key")
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame received")
    remember_frame(image_bytes, request.headers.get("x-cam-ip", ""))
    return {"ok": True, "bytes": len(image_bytes)}


@app.get("/test-alert")
def test_alert():
    """Browser se kholke test karo: https://your-url.railway.app/test-alert"""
    status = send_alerts(99, "")
    return {"alert_sent": True, "status": status}


@app.post("/detect")
async def detect(request: Request, x_alert_secret: str = Header(None)):
    # 1. Security Check
    if x_alert_secret != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid Secret Key")

    # 2. Read JPEG frame bytes from ESP32
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame received")
    remember_frame(image_bytes, request.headers.get("x-cam-ip", ""))

    # 3. Save temporary frame
    temp_filename = "temp_frame.jpg"
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)

    # 4. Predict using YOLOv8 (conf threshold 0.05 for ultra sensitivity)
    results = model.predict(
        source=temp_filename,
        conf=0.05,
        verbose=False
    )[0]

    dog_found = False
    best_score = 0.0
    all_seen = []
    
    # Dog & Animal COCO IDs: 16=dog, 15=cat, 17=horse, 18=sheep, 19=cow, 21=bear, 77=teddy bear
    ANIMAL_IDS = {15, 16, 17, 18, 19, 21, 77}

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            all_seen.append(f"{cls_name}:{round(score, 2)}")
            
            if cls_id in ANIMAL_IDS:
                dog_found = True
                if score > best_score:
                    best_score = score

    # 5. Send Notification if dog is detected
    global LAST_DOG, LAST_SCORE, LAST_SEEN, LAST_ALERT_AT
    alert_status = []
    LAST_DOG = dog_found
    LAST_SCORE = round(best_score, 2)
    LAST_SEEN = all_seen
    if dog_found:
        accuracy_percent = int(best_score * 100)
        LAST_ALERT_AT = time.time()
        alert_status = send_alerts(accuracy_percent, temp_filename)

    return {
        "dog": dog_found,
        "score": round(best_score, 2),
        "seen": all_seen,
        "alerts": alert_status,
        "message": "Dog detected & Mobile alert sent!" if dog_found else "No dog detected"
    }


APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#090b10">
<link rel="manifest" href="/manifest.webmanifest">
<title>Dog Watch</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    color: #f6f3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
      radial-gradient(900px 420px at 50% -10%, rgba(232, 196, 140, 0.16), transparent 55%),
      radial-gradient(700px 380px at 100% 100%, rgba(72, 98, 140, 0.18), transparent 50%),
      #090b10;
  }
  .wrap { max-width: 460px; margin: 0 auto; padding: 28px 18px 36px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .mark {
    width: 42px; height: 42px; border-radius: 14px;
    display: grid; place-items: center;
    background: linear-gradient(160deg, #f3e2c4, #c9a36a);
    color: #1b140c; font-weight: 750; letter-spacing: -0.04em;
    box-shadow: 0 10px 30px rgba(201, 163, 106, 0.25);
  }
  h1 { font-size: 28px; margin: 0; letter-spacing: -0.045em; font-weight: 700; }
  .eyebrow { margin: 2px 0 0; color: #a39b90; font-size: 14px; }
  .panel {
    margin-top: 22px; padding: 18px;
    background: rgba(20, 24, 32, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 28px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(16px);
  }
  label { display: block; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #b7ad9f; margin-bottom: 8px; }
  input {
    width: 100%; border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
    background: rgba(8, 10, 14, 0.7); color: white; font-size: 16px; padding: 15px 16px;
  }
  input:focus { outline: 2px solid rgba(232, 196, 140, 0.45); border-color: transparent; }
  button, a.livebtn {
    width: 100%; border: 0; border-radius: 16px; font-size: 16px; padding: 15px 16px;
    text-align: center; text-decoration: none; display: block;
  }
  button { margin-top: 12px; background: linear-gradient(180deg, #f6e7cc, #e2c48a); color: #1b140c; font-weight: 700; }
  button.ghost { margin-top: 8px; background: transparent; color: #c9beb0; font-weight: 600; }
  a.livebtn { margin-top: 14px; background: linear-gradient(180deg, #f6e7cc, #e2c48a); color: #1b140c; font-weight: 700; }
  .help { font-size: 13px; line-height: 1.5; color: #8f877c; margin: 14px 2px 0; }
  .stage {
    position: relative; margin-top: 18px; border-radius: 28px; overflow: hidden;
    min-height: 280px; background: #07080c;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
  }
  img.live { width: 100%; min-height: 320px; height: 58vh; object-fit: cover; display: block; background: #07080c; }
  .empty {
    min-height: 320px; height: 58vh; display: flex; align-items: center; justify-content: center;
    padding: 28px; text-align: center; color: #b7ad9f;
  }
  .overlay {
    position: absolute; left: 12px; right: 12px; top: 12px;
    display: flex; justify-content: space-between; gap: 8px; align-items: center;
  }
  .pill, .agechip {
    border-radius: 999px; padding: 7px 12px; font-size: 13px; font-weight: 650;
    background: rgba(10, 12, 16, 0.62); color: #f6f3ee;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
  }
  .pill.ok { background: rgba(22, 92, 62, 0.82); }
  .pill.bad { background: rgba(140, 36, 48, 0.88); }
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <div class="mark">DW</div>
    <div>
      <h1>Dog Watch</h1>
      <p class="eyebrow" id="lead">Live camera and alerts</p>
    </div>
  </div>

  <section id="login" class="panel">
    <label for="secret">Alert secret</label>
    <input id="secret" type="password" placeholder="Enter your secret" autocomplete="current-password">
    <button id="save" type="button">Open app</button>
    <p class="help">Use the same secret from the ESP32 camera code. This phone will remember it.</p>
  </section>

  <section id="watch" hidden>
    <div class="stage">
      <img id="shot" class="live" alt="Camera" hidden>
      <div id="waiting" class="empty">Waiting for the first picture from the camera...</div>
      <div class="overlay">
        <span id="state" class="pill">Connecting</span>
        <span id="age" class="agechip">—</span>
      </div>
    </div>
    <a id="cam" class="livebtn" hidden>Open the camera live video</a>
    <button id="out" class="ghost" type="button">Remove secret</button>
    <p class="help">Add this to your home screen from Safari: Share, then Add to Home Screen. On an alert, tap Live stream.</p>
  </section>
</div>
<script>
const secretEl = document.getElementById('secret');
const loginEl = document.getElementById('login');
const watchEl = document.getElementById('watch');
const shot = document.getElementById('shot');
const waiting = document.getElementById('waiting');
const stateEl = document.getElementById('state');
const ageEl = document.getElementById('age');
const camEl = document.getElementById('cam');
const fromLink = (new URLSearchParams(location.search).get('k') || '').trim();
let key = fromLink || localStorage.getItem('dogwatch_secret') || '';
if (fromLink) {
  localStorage.setItem('dogwatch_secret', fromLink);
  history.replaceState({}, '', '/app');
}
let blobUrl = '';

function showWatch() {
  loginEl.hidden = true;
  watchEl.hidden = false;
  tick();
  setInterval(tick, 1000);
}
function logout() {
  localStorage.removeItem('dogwatch_secret');
  key = '';
  watchEl.hidden = true;
  loginEl.hidden = false;
}
document.getElementById('save').onclick = function () {
  key = secretEl.value.trim();
  if (!key) return;
  localStorage.setItem('dogwatch_secret', key);
  showWatch();
};
document.getElementById('out').onclick = logout;

async function tick() {
  if (!key) return;
  try {
    const status = await fetch('/api/status', { headers: { 'x-alert-secret': key }, cache: 'no-store' });
    if (status.status === 401) return logout();
    if (status.ok) {
      const data = await status.json();
      if (data.dog) {
        stateEl.textContent = 'Dog detected ' + Math.round((data.score || 0) * 100) + '%';
        stateEl.className = 'pill bad';
      } else if (data.has_frame) {
        stateEl.textContent = 'Live';
        stateEl.className = 'pill ok';
      } else {
        stateEl.textContent = 'Waiting';
        stateEl.className = 'pill';
      }
      ageEl.textContent = data.age_sec == null ? 'no frame' : data.age_sec + 's ago';
      if (data.cam_url) {
        camEl.href = data.cam_url;
        camEl.hidden = false;
      }
    }
    const pic = await fetch('/live.jpg', { headers: { 'x-alert-secret': key }, cache: 'no-store' });
    if (pic.status === 404) return;
    if (!pic.ok) return;
    const blob = await pic.blob();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(blob);
    shot.src = blobUrl;
    shot.hidden = false;
    waiting.hidden = true;
  } catch (e) {
    stateEl.textContent = 'Offline';
    stateEl.className = 'pill';
  }
}
if (key) showWatch();
</script>
</body>
</html>
"""

