"""
main.py
--------
Phase 6 — Integration API for Person 2 (React Native bridge)

Person 2 uses ONLY this file. They never touch detection/, recognition/,
liveness/ internals. This is the single entry point.

Two modes:

  1. REGISTRATION — enroll a new user
     Input:  { "mode": "register", "user_id": "...", "name": "...", "image": frame }
     Output: { "registered": true, "user_id": "..." }

  2. AUTHENTICATION — verify identity + liveness
     Input:  { "mode": "auth", "image": frame }
     Output: {
                 "faceDetected":  true,
                 "liveness":      "PASS",
                 "similarity":    0.91,
                 "authenticated": true,
                 "matchedUserId": "user_001",
                 "instruction":   "Liveness verified."
             }

Usage:
    from main import FaceAuthSystem

    system = FaceAuthSystem()

    # Register once per user
    system.register_user("user_001", "Rahul Sharma", frame)

    # Start a new auth session (one per person walking up to camera)
    system.start_session()

    # Call on each camera frame until result is not PENDING
    for frame in camera_frames:
        result = system.authenticate(frame)
        if result["liveness"] != "PENDING":
            break
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

from detection.face_detector   import MediaPipeFaceDetector, crop_face, get_eye_coords_pixels
from recognition.face_recognizer import FaceRecognizer
from recognition.matcher         import EmbeddingStore, MatchResult, SIMILARITY_THRESHOLD
from liveness.liveness_detector  import LivenessDetector, LivenessStatus
from utils.preprocessing         import (
    preprocess_frame,
    align_face,
    normalize_for_model,
    is_sharp,
)


class FaceAuthSystem:
    """
    High-level facade. One instance per app lifecycle.
    Call start_session() before each new authentication attempt.
    """

    def __init__(
        self,
        model_path:       str   = "models/mobilefacenet.tflite",
        embeddings_path:  str   = "models/embeddings.json",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        require_blink:    bool  = True,
        require_head_turn: bool = False,
    ):
        print("[FaceAuthSystem] Initializing...")
        self._detector   = MediaPipeFaceDetector(min_confidence=0.6)
        self._recognizer = FaceRecognizer(model_path=model_path)
        self._store      = EmbeddingStore(store_path=embeddings_path)
        self._threshold  = similarity_threshold
        self._require_blink     = require_blink
        self._require_head_turn = require_head_turn

        # Liveness detector is session-scoped (reset per auth attempt)
        self._liveness: Optional[LivenessDetector] = None
        print(f"[FaceAuthSystem] Ready. {len(self._store)} enrolled users.")

    # ── Session management ─────────────────────────────────────────────────────

    def start_session(self):
        """
        Call this before each new authentication attempt.
        Resets liveness state so each user gets a clean challenge.
        """
        if self._liveness:
            self._liveness.reset()
        else:
            self._liveness = LivenessDetector(
                require_blink=self._require_blink,
                require_head_turn=self._require_head_turn,
            )

    # ── Registration ───────────────────────────────────────────────────────────

    def register_user(
        self,
        user_id: str,
        name:    str,
        frame:   np.ndarray,
    ) -> Dict[str, Any]:
        """
        Enroll a new user from a single clear face image.
        Call this from an admin/onboarding screen.

        Args:
            user_id: Unique identifier (e.g. employee ID)
            name:    Display name
            frame:   BGR numpy array from camera
        Returns:
            {"registered": bool, "user_id": str, "error": str|None}
        """
        embedding, error = self._extract_embedding(frame)
        if error:
            return {"registered": False, "user_id": user_id, "error": error}

        self._store.register(user_id, name, embedding)
        return {"registered": True, "user_id": user_id, "error": None}

    # ── Authentication (main loop) ─────────────────────────────────────────────

    def authenticate(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process one camera frame. Call repeatedly until liveness != "PENDING".

        Args:
            frame: BGR numpy array (current camera frame)
        Returns:
            Full result dict (see module docstring for schema)
        """
        if self._liveness is None:
            self.start_session()

        base_result = {
            "faceDetected":  False,
            "liveness":      "PENDING",
            "similarity":    0.0,
            "authenticated": False,
            "matchedUserId": None,
            "instruction":   "Position your face in the frame",
        }

        # ── Step 1: lighting normalization ────────────────────────────────────
        proc_frame = preprocess_frame(frame)

        # ── Step 2: face detection ────────────────────────────────────────────
        det = self._detector.detect(proc_frame)
        if not det.face_detected:
            base_result["instruction"] = "No face detected. Please look at the camera."
            return base_result

        base_result["faceDetected"] = True

        # ── Step 3: liveness check (runs every frame) ─────────────────────────
        liveness_status = self._liveness.update(proc_frame)
        base_result["liveness"]    = liveness_status.value
        base_result["instruction"] = self._liveness.instruction

        # ── Step 4: recognition (only after liveness PASS) ───────────────────
        if liveness_status == LivenessStatus.PASS:
            embedding, error = self._extract_embedding(proc_frame, det)
            if error:
                base_result["liveness"] = "FAIL"
                base_result["instruction"] = f"Recognition error: {error}"
                return base_result

            match = self._store.find_best_match(embedding, self._threshold)
            base_result["similarity"]    = round(match.similarity, 4)
            base_result["authenticated"] = match.matched
            base_result["matchedUserId"] = match.matched_user_id

        elif liveness_status == LivenessStatus.FAIL:
            base_result["instruction"] = "Liveness failed. Please retry."

        return base_result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_embedding(
        self,
        frame: np.ndarray,
        det=None,
    ):
        """
        Full preprocessing -> embedding pipeline.
        Returns (embedding, None) on success, (None, error_str) on failure.
        """
        if det is None:
            det = self._detector.detect(frame)

        if not det.face_detected:
            return None, "No face detected"

        # Crop with padding
        crop = crop_face(frame, det.bbox, padding=0.2)

        # Blur check — skip garbage frames
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if not is_sharp(gray, threshold=20.0):
            return None, "Frame too blurry"

        # Geometric alignment using eye keypoints
        if det.keypoints:
            left_eye_px, right_eye_px = get_eye_coords_pixels(frame, det.keypoints)
            # Transform eye coords from full frame to crop space
            x, y = det.bbox[0], det.bbox[1]
            pad_x = int(det.bbox[2] * 0.2)
            pad_y = int(det.bbox[3] * 0.2)
            left_eye_crop  = (left_eye_px[0]  - (x - pad_x), left_eye_px[1]  - (y - pad_y))
            right_eye_crop = (right_eye_px[0] - (x - pad_x), right_eye_px[1] - (y - pad_y))
            aligned = align_face(crop, left_eye_crop, right_eye_crop)
        else:
            aligned = cv2.resize(crop, (112, 112))

        # Pixel normalization
        tensor = normalize_for_model(aligned)

        # Inference
        embedding = self._recognizer.get_embedding(tensor)
        return embedding, None

    def close(self):
        self._detector.close()
        if self._liveness:
            self._liveness.close()
