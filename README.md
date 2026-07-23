# NHAI Offline Face Authentication System

> Biometric attendance for field personnel that works with zero internet connectivity.

[![Live Demo](https://img.shields.io/badge/Live-Demo-00d4aa?style=flat-square)](https://nhai-faceauth.vercel.app)
[![Download APK](https://img.shields.io/badge/Download-APK-3b82f6?style=flat-square)](https://github.com/BhumikaB1/NHAI_project/releases)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## What this is

A fully offline mobile biometric authentication system built for NHAI (National Highways Authority of India). Face recognition and liveness detection run entirely on-device using TensorFlow Lite — no server, no internet, no compromise.

**Live demo result:** Similarity 0.925 · LIVENESS PASS · AUTHENTICATED · <1 second

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile | React Native 0.85 · TypeScript |
| Camera | React Native Vision Camera v4 |
| ML Model | MobileFaceNet (TFLite) · 128-d embeddings |
| Face Detection | MediaPipe |
| Native Layer | Kotlin · Android Native Modules |
| Storage | AsyncStorage · Local file storage |
| Sync | NetInfo · AWS (when online) |

## Authentication Pipeline

Camera Frame
↓ EXIF orientation correction
↓ MediaPipe face detection + quality gate
↓ Face alignment + 112×112 crop
↓ Normalize to [-1, 1]
↓ MobileFaceNet TFLite inference
↓ 128-d L2-normalized embedding
↓ Cosine similarity matching
↓ threshold 0.72 + margin 0.04
→ AUTHENTICATED / REJECTED
→ Attendance logged offline


## Key Features

- **100% offline** — TFLite runs inside Kotlin native module, zero network calls
- **3-image enrollment** — averages embeddings for pose robustness  
- **Liveness detection** — EAR blink + head pose, prevents photo spoofing
- **Duplicate prevention** — blocks same face registering under two names
- **Face quality gate** — rejects blur, low light, extreme angles before inference
- **Offline attendance queue** — syncs to AWS when connectivity restored

## Results

| Metric | Value |
|---|---|
| Similarity score (live test) | 0.925 |
| Auth threshold | 0.72 + 0.04 margin |
| Total model size | ~11 MB |
| Auth latency | < 1 second |
| Tests passing | 12 / 12 |

## Setup

```bash
# Install dependencies
npm install

# Run on Android (connected device)
npx react-native run-android

# ML backend (Python, for development only)
cd ml
pip install -r requirements.txt
python models/download_models.py
python server.py
```

## Project Structure

NHAI_project/
├── android/ # Native Android (Kotlin TFLite module)
├── ios/ # iOS support
├── src/services/ # React Native API layer
├── ml/ # Python ML pipeline (dev/testing)
│ ├── detection/ # MediaPipe face detector
│ ├── recognition/ # MobileFaceNet embeddings
│ ├── liveness/ # EYE blink + head pose
│ ├── utils/ # CLAHE, alignment, normalization
│ ├── main.py # FaceAuthSystem integration API
│ └── server.py # Flask HTTP bridge
├── App.tsx # Main React Native app
└── models/ # TFLite model files (not in repo — see setup)
