"""
detection/face_detector.py
---------------------------
Phase 2 — Face Detection

Primary:  MediaPipe Face Detection  (fast, reliable, no model file needed)
Fallback: BlazeFace via TFLite      (fully offline, no MediaPipe dependency)

Output contract:
    {
        "faceDetected": bool,
        "bbox": [x, y, w, h],          # pixel coords in original frame
        "keypoints": {                  # normalised 0-1 coords
            "left_eye":  [x, y],
            "right_eye": [x, y],
            "nose":      [x, y],
            "mouth":     [x, y],
        },
        "confidence": float
    }
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import mediapipe as mp


@dataclass
class FaceDetectionResult:
    face_detected: bool
    bbox: Optional[List[int]] = None          # [x, y, w, h] pixel space
    keypoints: Optional[Dict[str, List[float]]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faceDetected": self.face_detected,
            "bbox":         self.bbox,
            "keypoints":    self.keypoints,
            "confidence":   round(self.confidence, 4),
        }


class MediaPipeFaceDetector:
    """
    Wraps MediaPipe Face Detection.
    model_selection=0  short range <= 2 m  (selfie / attendance)
    model_selection=1  full range  <= 5 m
    """

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        self._mp_det = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection,
        )

    def detect(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        try:
            res = self._mp_det.process(rgb)
        except Exception:
            return FaceDetectionResult(face_detected=False)

        if res is None or not res.detections:
            return FaceDetectionResult(face_detected=False)

        det = max(res.detections, key=lambda d: d.score[0])
        bb  = det.location_data.relative_bounding_box

        x  = max(0, int(bb.xmin * w))
        y  = max(0, int(bb.ymin * h))
        bw = min(int(bb.width  * w), w - x)
        bh = min(int(bb.height * h), h - y)

        # MediaPipe keypoint order:
        # 0=right eye, 1=left eye, 2=nose, 3=mouth, 4=right ear, 5=left ear
        kp = det.location_data.relative_keypoints
        keypoints = {
            "left_eye":  [kp[1].x, kp[1].y],
            "right_eye": [kp[0].x, kp[0].y],
            "nose":      [kp[2].x, kp[2].y],
            "mouth":     [kp[3].x, kp[3].y],
        }

        return FaceDetectionResult(
            face_detected=True,
            bbox=[x, y, bw, bh],
            keypoints=keypoints,
            confidence=float(det.score[0]),
        )

    def close(self):
        self._mp_det.close()


def crop_face(
    bgr_frame: np.ndarray,
    bbox: List[int],
    padding: float = 0.2,
) -> np.ndarray:
    """Crop face ROI with percentage padding on all sides."""
    h, w = bgr_frame.shape[:2]
    x, y, bw, bh = bbox
    px = int(bw * padding)
    py = int(bh * padding)
    x1 = max(0, x - px);  y1 = max(0, y - py)
    x2 = min(w, x + bw + px); y2 = min(h, y + bh + py)
    return bgr_frame[y1:y2, x1:x2].copy()


def get_eye_coords_pixels(
    bgr_frame: np.ndarray,
    keypoints: Dict[str, List[float]],
) -> tuple:
    """Normalised keypoint coords -> pixel coords."""
    h, w = bgr_frame.shape[:2]
    le = keypoints["left_eye"]
    re = keypoints["right_eye"]
    return (le[0] * w, le[1] * h), (re[0] * w, re[1] * h)
