# Hackathon 7.0 — ML Pipeline (Person 1)

Offline facial recognition + liveness detection engine.
Integrates with React Native via a clean JSON API (see `main.py`).

---

## Project structure

```
ml/
├── models/
│   ├── download_models.py      ← run this FIRST
│   ├── mobilefacenet.tflite    ← downloaded by script
│   └── embeddings.json         ← auto-created, stores enrolled faces
│
├── detection/
│   └── face_detector.py        ← Phase 2: MediaPipe face detection
│
├── recognition/
│   ├── face_recognizer.py      ← Phase 3: MobileFaceNet TFLite embeddings
│   └── matcher.py              ← Phase 4: cosine similarity + embedding store
│
├── liveness/
│   └── liveness_detector.py    ← Phase 5: EAR blink + head pose
│
├── utils/
│   └── preprocessing.py        ← CLAHE, alignment, normalization, blur check
│
├── tests/
│   └── test_pipeline.py        ← Phase 7: full test suite
│
├── main.py                     ← Phase 6: integration API for Person 2
├── requirements.txt
└── README.md
```

---

## Setup (run once)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MobileFaceNet model (~4 MB)
cd ml
python models/download_models.py

# 3. Run tests
python -m pytest tests/ -v
```

---

## Corrections vs original roadmap spec

| Spec says | Correct value | Why |
|---|---|---|
| 128-d embedding | **512-d** | MobileFaceNet outputs 512-d. 128-d is FaceNet-Lite (different model). |
| Threshold 0.75 | **0.60** | 0.75 causes false rejects. 0.60 is the proven MobileFaceNet threshold. |
| No preprocessing step | **Required** | CLAHE + eye alignment improves accuracy ~5%. Must be explicit in pipeline. |

---

## Integration API for Person 2

Person 2 **only** imports `FaceAuthSystem` from `main.py`.
They never touch detection/, recognition/, or liveness/ internals.

```python
from main import FaceAuthSystem

system = FaceAuthSystem()

# Enroll a user (once, during onboarding)
system.register_user("user_001", "Rahul Sharma", frame)

# Per authentication session:
system.start_session()

while True:
    frame  = get_camera_frame()
    result = system.authenticate(frame)

    # result schema:
    # {
    #   "faceDetected":  bool,
    #   "liveness":      "PENDING" | "PASS" | "FAIL",
    #   "similarity":    float,      # 0.0 - 1.0
    #   "authenticated": bool,
    #   "matchedUserId": str | None,
    #   "instruction":   str,        # show this to the user on-screen
    # }

    if result["liveness"] != "PENDING":
        break
```

---

## Model size budget

| Component | Size |
|---|---|
| BlazeFace / MediaPipe detection | ~1.0 MB |
| MobileFaceNet INT8 (recognition) | ~4.0 MB |
| Silent-Face liveness (optional) | ~2.5 MB |
| MediaPipe Face Mesh (liveness)  | ~3.5 MB |
| **Total** | **~11 MB ✅** |

9 MB under the 20 MB limit.

---

## Threshold tuning guide

After collecting ~50 positive test pairs from your target demographic:

```python
from recognition.matcher import cosine_similarity
import numpy as np

# Collect similarity scores for same-person pairs
same_scores = [cosine_similarity(emb_a, emb_b) for emb_a, emb_b in same_pairs]

# Set threshold at: mean - 2 * std
threshold = np.mean(same_scores) - 2 * np.std(same_scores)
print(f"Recommended threshold: {threshold:.3f}")
```

Start at 0.60. Tighten toward 0.65 if you see false positives in testing.

---

## Deliverables checklist (Person 1)

- [x] Face Detection Module (`detection/face_detector.py`)
- [x] Face Recognition Module (`recognition/face_recognizer.py`)
- [x] Cosine Similarity Matching (`recognition/matcher.py`)
- [x] Liveness Detection (`liveness/liveness_detector.py`)
- [x] TFLite Model Package (downloaded via `models/download_models.py`)
- [x] Integration API for React Native (`main.py`)
- [x] Test Suite (`tests/test_pipeline.py`)
- [ ] Accuracy metrics on Indian face test set (run after dataset collection)
- [ ] Inference latency benchmarks on physical Android device
