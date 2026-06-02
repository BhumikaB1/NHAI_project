"""
liveness/liveness_detector.py
------------------------------
Phase 5 — Liveness Detection

Uses MediaPipe Face Mesh (468 landmarks) — no separate model file needed.

Method 1 (Required): Blink detection via Eye Aspect Ratio (EAR)
Method 2 (Optional): Head turn detection via 3D pose estimation

EAR formula (Soukupova & Cech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Where p1-p6 are the 6 eye landmark points.
    EAR ~ 0.3 when open, drops to ~0.0 when closed.
    Blink threshold: EAR < 0.22 for at least 2 consecutive frames.

Head pose uses solvePnP with 6 canonical 3D face points.
    Yaw  > 15 deg  -> turned left/right
    Pitch > 10 deg -> nodding

Output:
    {"liveness": "PASS"} or {"liveness": "FAIL"}
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional
from collections import deque

import mediapipe as mp


# ── MediaPipe landmark indices ─────────────────────────────────────────────────
# Right eye (from wearer's perspective)
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Left eye
LEFT_EYE  = [33,  160, 158, 133, 153, 144]

# 6 points used for head pose estimation
# Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
HEAD_POSE_LANDMARKS = [1, 152, 226, 446, 57, 287]

# Canonical 3D face model coordinates (in mm, arbitrary scale)
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,   -63.6,  -12.5 ],   # Chin
    [-43.3,  32.7,  -26.0 ],   # Left eye corner
    [43.3,   32.7,  -26.0 ],   # Right eye corner
    [-28.9, -28.9,  -24.1 ],   # Left mouth corner
    [28.9,  -28.9,  -24.1 ],   # Right mouth corner
], dtype=np.float64)

# ── Thresholds ─────────────────────────────────────────────────────────────────
EAR_BLINK_THRESHOLD   = 0.25   # EAR below this = eye closed
EAR_CONSEC_FRAMES     = 1      # Min frames below threshold to count as blink
BLINKS_REQUIRED       = 1      # Number of blinks needed to pass
YAW_THRESHOLD_DEG     = 15.0   # Degrees of head turn to count as turn
PITCH_THRESHOLD_DEG   = 10.0
HEAD_TURN_REQUIRED    = 1      # Number of head turns needed


class LivenessStatus(Enum):
    PENDING = "PENDING"
    PASS    = "PASS"
    FAIL    = "FAIL"


class LivenessDetector:
    """
    Stateful liveness detector. Create one instance per auth session.
    Call update() on each frame. Read status to check result.
    Call reset() to start a new session.
    """

    def __init__(
        self,
        require_blink: bool = True,
        require_head_turn: bool = False,
        max_frames: int = 300,          # ~5 sec at 30fps before FAIL
    ):
        self._require_blink     = require_blink
        self._require_head_turn = require_head_turn
        self._max_frames        = max_frames

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.reset()

    def reset(self):
        """Start a fresh liveness session."""
        self._frame_count       = 0
        self._ear_consec        = 0
        self._blink_count       = 0
        self._head_turns        = 0
        self._prev_yaw          = None
        self._head_turned_left  = False
        self._head_turned_right = False
        self.status             = LivenessStatus.PENDING
        self.instruction        = "Please blink naturally"

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, bgr_frame: np.ndarray) -> LivenessStatus:
        """
        Process one camera frame. Updates internal state.

        Args:
            bgr_frame: Current camera frame (BGR)
        Returns:
            Current LivenessStatus (PENDING / PASS / FAIL)
        """
        if self.status != LivenessStatus.PENDING:
            return self.status

        self._frame_count += 1
        if self._frame_count % 2 != 0:
          return self.status

    # Timeout check continues below...
        if self._frame_count > self._max_frames:

        # Timeout — no valid liveness after max_frames
         if self._frame_count > self._max_frames:
            self.status      = LivenessStatus.FAIL
            self.instruction = "Liveness timeout. Please retry."
            return self.status

        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return self.status  # no face detected this frame, keep waiting

        landmarks = result.multi_face_landmarks[0].landmark
        h, w      = bgr_frame.shape[:2]

        # -- Blink detection --
        if self._require_blink:
            self._process_blink(landmarks, w, h)

        # -- Head turn detection --
        if self._require_head_turn:
            self._process_head_turn(landmarks, w, h)

        # -- Check pass conditions --
        blink_ok = (not self._require_blink) or (self._blink_count >= BLINKS_REQUIRED)
        turn_ok  = (not self._require_head_turn) or (self._head_turns >= HEAD_TURN_REQUIRED)

        if blink_ok and turn_ok:
            self.status      = LivenessStatus.PASS
            self.instruction = "Liveness verified."
        elif self._require_blink and self._blink_count < BLINKS_REQUIRED:
            self.instruction = "Please blink once"
        elif self._require_head_turn and self._head_turns < HEAD_TURN_REQUIRED:
            self.instruction = "Please turn your head slightly left or right"

        return self.status

    def to_dict(self) -> dict:
        return {
            "liveness":    self.status.value,
            "instruction": self.instruction,
            "blinks":      self._blink_count,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_eye_points(self, landmarks, indices, w, h):
        return np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
            dtype=np.float64
        )

    def _ear(self, eye: np.ndarray) -> float:
        """
        Eye Aspect Ratio.
        eye: (6, 2) array of landmark pixel coords in order
             [outer, upper1, upper2, inner, lower1, lower2]
        """
        # Vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def _process_blink(self, landmarks, w, h):
        left_pts  = self._get_eye_points(landmarks, LEFT_EYE,  w, h)
        right_pts = self._get_eye_points(landmarks, RIGHT_EYE, w, h)

        ear = (self._ear(left_pts) + self._ear(right_pts)) / 2.0

        if ear < EAR_BLINK_THRESHOLD:
            self._ear_consec += 1
        else:
            # Eye just reopened after being closed long enough
            if self._ear_consec >= EAR_CONSEC_FRAMES:
                self._blink_count += 1
            self._ear_consec = 0

    def _process_head_turn(self, landmarks, w, h):
        face_2d = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h]
             for i in HEAD_POSE_LANDMARKS],
            dtype=np.float64
        )
        focal_length = w
        cam_matrix   = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw   = angles[1] * 360   # convert to degrees
        pitch = angles[0] * 360

        if abs(yaw) > YAW_THRESHOLD_DEG:
            if yaw < 0 and not self._head_turned_left:
                self._head_turned_left = True
                self._head_turns += 1
            elif yaw > 0 and not self._head_turned_right:
                self._head_turned_right = True
                self._head_turns += 1

    def close(self):
        self._mesh.close()
