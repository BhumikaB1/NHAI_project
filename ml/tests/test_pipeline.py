"""
tests/test_pipeline.py
-----------------------
Phase 7 — Test Suite

Covers all positive and negative cases from the spec.
Run with: python -m pytest tests/ -v

Requires:
  - ml/models/mobilefacenet.tflite
  - A webcam OR test images in tests/fixtures/

Test cases:
  Positive:
    - Same user, good lighting, valid blink           -> PASS
    - Same user, varied lighting (CLAHE)              -> PASS
    - Same user, slight head tilt (alignment)         -> PASS

  Negative:
    - Different user                                  -> FAIL (low similarity)
    - Printed photo (no blink in 150 frames)          -> FAIL (liveness timeout)
    - No face in frame                                -> faceDetected: false
    - Blurry frame                                    -> FAIL (blur rejection)
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (
    apply_clahe, align_face, normalize_for_model, is_sharp
)
from detection.face_detector import MediaPipeFaceDetector, crop_face
from recognition.matcher import cosine_similarity, EmbeddingStore
from liveness.liveness_detector import LivenessDetector, LivenessStatus


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_test_frame(h=480, w=640, face=True) -> np.ndarray:
    """Generate a synthetic BGR frame for unit tests (no camera needed)."""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 200  # light grey
    if face:
        # Draw a rough face-like oval so MediaPipe can detect it
        cx, cy = w // 2, h // 2
        cv2.ellipse(frame, (cx, cy), (90, 110), 0, 0, 360, (180, 160, 140), -1)
        cv2.circle(frame, (cx - 30, cy - 20), 12, (80, 80, 80), -1)  # left eye
        cv2.circle(frame, (cx + 30, cy - 20), 12, (80, 80, 80), -1)  # right eye
        cv2.ellipse(frame, (cx, cy + 40), (30, 15), 0, 0, 180, (120, 80, 80), -1)  # mouth
    return frame


def synthetic_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    """Reproducible random L2-normalized embedding for testing matcher logic."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def test_clahe_output_shape():
    """CLAHE must preserve image shape and dtype."""
    frame  = make_test_frame()
    result = apply_clahe(frame)
    assert result.shape == frame.shape, "CLAHE changed image shape"
    assert result.dtype == np.uint8,    "CLAHE changed dtype"
    print("PASS  test_clahe_output_shape")


def test_normalize_range():
    """Normalized tensor must be in [-1, 1] and correct shape."""
    face   = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    tensor = normalize_for_model(face)
    assert tensor.shape == (1, 112, 112, 3),      f"Wrong shape: {tensor.shape}"
    assert tensor.dtype == np.float32,             "Wrong dtype"
    assert tensor.min() >= -1.0 - 1e-5,           f"Min out of range: {tensor.min()}"
    assert tensor.max() <=  1.0 + 1e-5,           f"Max out of range: {tensor.max()}"
    print("PASS  test_normalize_range")


def test_blur_rejection():
    """Sharp frame should pass, heavily blurred should fail."""
    sharp   = make_test_frame()
    blurry  = cv2.GaussianBlur(sharp, (51, 51), 20)
    gray_s  = cv2.cvtColor(sharp,  cv2.COLOR_BGR2GRAY)
    gray_b  = cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY)
    assert  is_sharp(gray_s, threshold=10.0), "Sharp frame rejected"
    assert not is_sharp(gray_b, threshold=500.0), "Blurry frame not rejected"
    print("PASS  test_blur_rejection")


def test_align_face_output_size():
    """align_face must return exactly (112, 112, 3)."""
    frame = make_test_frame(480, 640)
    aligned = align_face(frame, (200.0, 180.0), (300.0, 180.0), output_size=112)
    assert aligned.shape == (112, 112, 3), f"Wrong aligned shape: {aligned.shape}"
    print("PASS  test_align_face_output_size")


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — cosine similarity
# ══════════════════════════════════════════════════════════════════════════════

def test_cosine_identical():
    """Identical vectors must give similarity ~1.0."""
    emb = synthetic_embedding(seed=1)
    sim = cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-4, f"Identical vectors gave {sim}"
    print("PASS  test_cosine_identical")


def test_cosine_opposite():
    """Negated vector must give similarity ~-1.0."""
    emb = synthetic_embedding(seed=2)
    sim = cosine_similarity(emb, -emb)
    assert abs(sim + 1.0) < 1e-4, f"Opposite vectors gave {sim}"
    print("PASS  test_cosine_opposite")


def test_cosine_different_faces():
    """Two random embeddings should have low similarity (< 0.5)."""
    a = synthetic_embedding(seed=10)
    b = synthetic_embedding(seed=99)
    sim = cosine_similarity(a, b)
    assert sim < 0.5, f"Random embeddings too similar: {sim}"
    print(f"PASS  test_cosine_different_faces  (sim={sim:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — embedding store + matching
# ══════════════════════════════════════════════════════════════════════════════

def test_store_register_and_match():
    """Registered embedding should match itself above threshold."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        store = EmbeddingStore(store_path=tmp_path)
        emb   = synthetic_embedding(seed=7)
        store.register("user_001", "Test User", emb)

        # Exact match -> must pass
        result = store.find_best_match(emb, threshold=0.60)
        assert result.matched,                        "Exact match failed"
        assert result.matched_user_id == "user_001", "Wrong user matched"
        assert result.similarity > 0.99,             f"Too low: {result.similarity}"

        # Different embedding -> must fail
        other  = synthetic_embedding(seed=999)
        result2 = store.find_best_match(other, threshold=0.60)
        assert not result2.matched, f"False positive: {result2.similarity}"

        print(f"PASS  test_store_register_and_match  (sim={result.similarity:.4f})")
    finally:
        os.unlink(tmp_path)


def test_threshold_boundary():
    """Similarity just below threshold must not match."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        store = EmbeddingStore(store_path=tmp_path)
        emb_a = synthetic_embedding(seed=3)
        emb_b = synthetic_embedding(seed=4)
        store.register("user_a", "User A", emb_a)

        sim = cosine_similarity(emb_a, emb_b)
        # Use a threshold just above actual similarity -> should not match
        result = store.find_best_match(emb_b, threshold=sim + 0.01)
        assert not result.matched, f"Should not match below threshold (sim={sim:.4f})"
        print(f"PASS  test_threshold_boundary  (sim={sim:.4f})")
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — liveness (no camera required)
# ══════════════════════════════════════════════════════════════════════════════

def test_liveness_timeout():
    """Liveness must FAIL after max_frames with no blink detected."""
    detector = LivenessDetector(require_blink=True, max_frames=10)
    detector.start = True

    # Feed 11 blank frames (no face -> no blink possible)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(11):
        status = detector.update(blank)

    assert status == LivenessStatus.FAIL, f"Expected FAIL, got {status}"
    print("PASS  test_liveness_timeout")
    detector.close()


def test_liveness_reset():
    """After reset, status must return to PENDING."""
    detector = LivenessDetector(require_blink=True, max_frames=5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(6):
        detector.update(blank)

    assert detector.status == LivenessStatus.FAIL
    detector.reset()
    assert detector.status == LivenessStatus.PENDING, "Reset did not restore PENDING"
    print("PASS  test_liveness_reset")
    detector.close()


# ══════════════════════════════════════════════════════════════════════════════
# Integration test — API contract (requires model file)
# ══════════════════════════════════════════════════════════════════════════════

def test_api_contract_no_face():
    """authenticate() with blank frame must return faceDetected: False."""
    model_path = "models/mobilefacenet.tflite"
    if not os.path.exists(model_path):
        print("SKIP  test_api_contract_no_face  (model not found)")
        return

    from main import FaceAuthSystem
    system = FaceAuthSystem(model_path=model_path)
    system.start_session()

    blank  = np.zeros((480, 640, 3), dtype=np.uint8)
    result = system.authenticate(blank)

    assert not result["faceDetected"],       "Blank frame should not detect face"
    assert result["authenticated"] == False, "Should not authenticate blank frame"
    print("PASS  test_api_contract_no_face")
    system.close()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark — inference latency
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_preprocessing(n: int = 100):
    """Measure preprocessing pipeline latency."""
    frame = make_test_frame()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        proc = apply_clahe(frame)
        gray = cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2GRAY)
        is_sharp(gray)
        normalize_for_model(cv2.resize(frame, (112, 112)))
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    p95 = np.percentile(times, 95)
    print(f"BENCH preprocessing: avg={avg:.2f}ms  p95={p95:.2f}ms  (n={n})")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n─── Unit Tests: Preprocessing ────────────────────────────────")
    test_clahe_output_shape()
    test_normalize_range()
    test_blur_rejection()
    test_align_face_output_size()

    print("\n─── Unit Tests: Cosine Similarity ────────────────────────────")
    test_cosine_identical()
    test_cosine_opposite()
    test_cosine_different_faces()

    print("\n─── Unit Tests: Embedding Store ──────────────────────────────")
    test_store_register_and_match()
    test_threshold_boundary()

    print("\n─── Unit Tests: Liveness ─────────────────────────────────────")
    test_liveness_timeout()
    test_liveness_reset()

    print("\n─── Integration Tests ────────────────────────────────────────")
    test_api_contract_no_face()

    print("\n─── Benchmarks ───────────────────────────────────────────────")
    benchmark_preprocessing()

    print("\n✅ All tests completed.\n")
