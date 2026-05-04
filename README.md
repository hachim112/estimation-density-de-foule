# CrowdSense — AI Crowd Detection Web App

A Flask web application for real-time crowd detection and risk assessment using Facebook's DETR (Detection Transformer) model.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```bash
   python app.py
   ```

3. **Open your browser**
   ```
   http://localhost:5000
   ```

## Features

- **Image Analysis** — Upload any image and detect people with bounding boxes + confidence scores
- **Live Camera** — Stream your webcam through the model in real-time
- **Risk Assessment** — LOW / MEDIUM / HIGH crowd risk levels
  - LOW: < 10 people
  - MEDIUM: 10–39 people
  - HIGH: 40+ people
- **GPU Support** — Automatically uses CUDA if available

## Project Structure

```
crowd-app/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Temporary upload storage
└── results/            # Processed image storage
```

## Notes

- First run downloads the DETR model (~160MB) from HuggingFace
- Max upload size: 16MB
- Images larger than 1200px are auto-resized for faster inference
- Camera streaming uses MJPEG over HTTP
