"""
recognition/face_recognizer.py
--------------------------------
Phase 3 + 4: Face Recognition + Cosine Similarity Matching

Model: MobileFaceNet (TFLite, INT8 quantized)
Input:  (1, 112, 112, 3) float32 normalized to [-1, 1]
Output: 512-d L2-normalized embedding vector

NOTE: The roadmap says 128-d. That is WRONG for MobileFaceNet.
MobileFaceNet -> 512-d. FaceNet-Lite -> 128-d (lower accuracy).
Update your integration contract accordingly.

Download model:
  wget https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/out/MobileFaceNet.tflite
  Place at: ml/models/mobilefacenet.tflite
"""

import numpy as np
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


class FaceRecognizer:
    """Loads MobileFaceNet TFLite and generates 512-d face embeddings."""

    def __init__(self, model_path: str = "models/mobilefacenet.tflite"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MobileFaceNet not found at {path}.\n"
                "Download: https://github.com/sirius-ai/MobileFaceNet_TF"
                "/raw/master/out/MobileFaceNet.tflite"
            )

        self._interp = tflite.Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._in  = self._interp.get_input_details()
        self._out = self._interp.get_output_details()
        self._embedding_dim = self._out[0]['shape'][-1]
        print(f"[FaceRecognizer] Ready. Embedding dim: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, normalized_tensor: np.ndarray) -> np.ndarray:
        """
        Args:
            normalized_tensor: float32 (1, 112, 112, 3) in [-1, 1]
        Returns:
            float32 (512,) L2-normalized embedding
        """
        if normalized_tensor.dtype != np.float32:
            normalized_tensor = normalized_tensor.astype(np.float32)

        self._interp.set_tensor(self._in[0]['index'], normalized_tensor)
        self._interp.invoke()
        emb = self._interp.get_tensor(self._out[0]['index'])[0]

        # L2 normalize -> cosine similarity becomes dot product
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else emb
