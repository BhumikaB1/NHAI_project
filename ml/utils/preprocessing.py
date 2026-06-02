"""
utils/preprocessing.py
-----------------------
Image preprocessing pipeline:
  1. CLAHE lighting normalization (LAB space)
  2. Geometric alignment via eye keypoints
  3. Pixel normalization to [-1, 1] float32
  4. Motion blur rejection (Laplacian variance)

All functions operate on BGR numpy arrays (OpenCV default).
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ── Canonical eye positions for 112×112 aligned face ──────────────────────────
# These match the ArcFace / MobileFaceNet training alignment standard.
LEFT_EYE_CANONICAL  = np.array([38.2946, 51.6963], dtype=np.float32)
RIGHT_EYE_CANONICAL = np.array([73.5318, 51.5014], dtype=np.float32)
FACE_SIZE = 112


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE only to the L channel in LAB space.
    Fixes harsh sunlight, low light, and mixed shadows
    without shifting skin tone colours.

    Args:
        bgr: BGR image (any size)
    Returns:
        BGR image with corrected contrast
    """
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)


def align_face(
    bgr: np.ndarray,
    left_eye: Tuple[float, float],
    right_eye: Tuple[float, float],
    output_size: int = FACE_SIZE,
) -> np.ndarray:
    """
    Apply a similarity transform (rotation + scale) so both eyes map to
    canonical positions, then crop to output_size × output_size.

    This is the single biggest accuracy improvement step — models trained
    on aligned faces degrade significantly on tilted inputs.

    Args:
        bgr:        Full frame or loose face crop (BGR)
        left_eye:   (x, y) of left eye centre in bgr coordinates
        right_eye:  (x, y) of right eye centre in bgr coordinates
        output_size: Target square size (default 112 for MobileFaceNet)
    Returns:
        Aligned, cropped BGR face of shape (output_size, output_size, 3)
    """
    src = np.array([left_eye, right_eye], dtype=np.float32)
    dst = np.array([LEFT_EYE_CANONICAL, RIGHT_EYE_CANONICAL], dtype=np.float32)

    # estimateAffinePartial2D gives rotation + uniform scale (no shear)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        # Fallback: just resize without alignment
        return cv2.resize(bgr, (output_size, output_size))

    aligned = cv2.warpAffine(bgr, M, (output_size, output_size),
                             flags=cv2.INTER_LINEAR)
    return aligned


def normalize_for_model(bgr_face: np.ndarray) -> np.ndarray:
    """
    Convert 112×112 BGR face crop to float32 tensor in [-1, 1].
    Layout: HWC (height × width × channels) — what TFLite expects.

    Args:
        bgr_face: uint8 BGR image of shape (112, 112, 3)
    Returns:
        float32 array of shape (1, 112, 112, 3) — batched for TFLite
    """
    rgb   = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
    img   = rgb.astype(np.float32)
    img   = (img / 255.0 - 0.5) / 0.5          # → [-1, 1]
    return np.expand_dims(img, axis=0)           # → (1, 112, 112, 3)


def is_sharp(gray: np.ndarray, threshold: float = 20.0) -> bool:
    """
    Reject blurry frames before running inference.
    Uses Laplacian variance — fast and reliable.

    A blurry face produces a shifted embedding that causes false rejects,
    so it's better to skip the frame and wait for a sharp one.

    Args:
        gray:      Grayscale face crop (any size)
        threshold: Variance floor. 80 works well for 112×112 crops.
                   Lower → more permissive. Higher → stricter.
    Returns:
        True if frame is sharp enough to process
    """
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold


def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a full camera frame before detection.
    Call this once per frame, before passing to face detector.
    """
    return apply_clahe(bgr_frame)
