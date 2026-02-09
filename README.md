---
title: ASL Translation API
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ASL Translation API

FastAPI backend for ASL sign language recognition using EfficientNet-B3.

## Endpoints

- `GET /` - Health check
- `POST /predict` - Upload an image to get ASL prediction

## Usage
```bash
curl -X POST "https://YOUR_USERNAME-asl-api.hf.space/predict" \
  -F "file=@image.jpg"
```