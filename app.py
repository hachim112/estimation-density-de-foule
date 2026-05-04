import os
import uuid
import base64
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("Loading DETR model...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# =========================
# RISK FUNCTION
# =========================
def get_risk(count):
    if count < 10:
        return "LOW", (0, 255, 0)
    elif count < 40:
        return "MEDIUM", (0, 255, 255)
    else:
        return "HIGH", (0, 0, 255)

def get_risk_data(count):
    if count < 10:
        return "LOW", "#00ff88", 0.2
    elif count < 40:
        return "MEDIUM", "#ffcc00", 0.6
    else:
        return "HIGH", "#ff3355", 1.0

# =========================
# PROCESS FRAME
# =========================
def process(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.6
    )[0]
    count = 0
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:  # person
            count += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = round(score.item(), 3)
            detections.append({"box": [x1, y1, x2, y2], "confidence": conf})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 60, y1), (0, 255, 136), -1)
            cv2.putText(frame, f"{conf:.2f}", (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    risk, color = get_risk(count)
    # Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 110), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"CROWD DETECTION", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)
    cv2.putText(frame, f"People: {count}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Risk:   {risk}", (15, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame, count, risk, detections

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Resize if too large
    h, w = frame.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    frame, count, risk, detections = process(frame)
    risk_label, risk_color, risk_level = get_risk_data(count)

    # Encode result image to base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'success': True,
        'image': img_b64,
        'count': count,
        'risk': risk_label,
        'risk_color': risk_color,
        'risk_level': risk_level,
        'detections': detections,
        'device': str(device)
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_stats')
def camera_stats():
    return Response(gen_stats(), mimetype='text/event-stream')

# Camera state
camera_data = {"count": 0, "risk": "LOW", "risk_color": "#00ff88"}

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, count, risk, _ = process(frame)
        risk_label, risk_color, _ = get_risk_data(count)
        camera_data['count'] = count
        camera_data['risk'] = risk_label
        camera_data['risk_color'] = risk_color
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

def gen_stats():
    import time
    while True:
        data = f"data: {camera_data['count']},{camera_data['risk']},{camera_data['risk_color']}\n\n"
        yield data
        time.sleep(0.5)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
