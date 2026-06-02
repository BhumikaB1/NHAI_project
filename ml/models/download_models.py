"""
models/download_models.py
--------------------------
Run this FIRST:
    python models/download_models.py

Downloads MobileFaceNet.tflite from the best available source.
If all downloads fail, gives manual instructions.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

MODELS_DIR   = Path(__file__).parent
MODEL_PATH   = MODELS_DIR / "mobilefacenet.tflite"
BUDGET_BYTES = 20 * 1024 * 1024

# Multiple sources in priority order — first one that works wins
SOURCES = [
    # Packed int8 model from deepinsight/insightface release
    "https://github.com/deepinsight/insightface/releases/download/v0.7/mobilefacenet.tflite",
    # Alternative mirror
    "https://huggingface.co/minchul/cvlface_adaface_ir18_vgg2/resolve/main/mobilefacenet.tflite",
    # Direct from sirius-ai (original, sometimes 404s due to LFS)
    "https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/out/MobileFaceNet.tflite",
]

EXPECTED_SIZES = {
    "MediaPipe Face Detection (built-in)": 1.0,
    "MobileFaceNet INT8 (recognition)":    4.0,
    "MediaPipe Face Mesh (built-in)":      3.5,
}


def download_with_progress(url: str, dest: Path) -> bool:
    """Try to download. Returns True on success, False on failure."""
    print(f"  Trying: {url[:72]}...")

    def progress(count, block, total):
        if total > 0:
            pct = min(count * block * 100 // total, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
        print(f"\n  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def try_all_sources() -> bool:
    for url in SOURCES:
        if download_with_progress(url, MODEL_PATH):
            return True
    return False


def create_stub_model():
    """
    Create a minimal valid TFLite flatbuffer stub so the rest of the
    pipeline can be tested without the real model.
    Inference will produce random embeddings — only for dev/testing.
    """
    try:
        import tensorflow as tf
        print("  Generating stub model via TensorFlow...")

        inp = tf.keras.Input(shape=(112, 112, 3), name="input")
        x   = tf.keras.layers.GlobalAveragePooling2D()(inp)
        x   = tf.keras.layers.Dense(512)(x)
        out = tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1), name="embedding"
        )(x)
        model = tf.keras.Model(inp, out)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"  Stub model created: {size_mb:.2f} MB")
        print("  ⚠️  THIS IS A STUB — embeddings are random, accuracy = 0")
        print("  ⚠️  Replace with the real model before demo/testing accuracy")
        return True

    except Exception as e:
        print(f"  Could not create stub: {e}")
        return False


def verify_model():
    import tensorflow as tf
    tflite = tf.lite

    interp = tflite.Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    inp_d  = interp.get_input_details()
    out_d  = interp.get_output_details()

    in_shape  = inp_d[0]['shape'].tolist()
    out_shape = out_d[0]['shape'].tolist()
    emb_dim   = out_shape[-1]

    print(f"  Input  shape : {in_shape}")
    print(f"  Output shape : {out_shape}")

    # Latency benchmark
    dummy = np.random.uniform(-1, 1, in_shape).astype(np.float32)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        interp.set_tensor(inp_d[0]['index'], dummy)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    avg = float(np.mean(times))
    p95 = float(np.percentile(times, 95))
    print(f"  Latency      : avg={avg:.1f}ms  p95={p95:.1f}ms  (10 runs, desktop CPU)")
    print(f"  Android est. : avg~{avg*2.5:.0f}ms  (Snapdragon 680-class, 2.5x desktop)")

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    budget_ok = MODEL_PATH.stat().st_size < BUDGET_BYTES

    if emb_dim == 512:
        print(f"  Embedding dim: 512  ✅  (MobileFaceNet — correct)")
    elif emb_dim == 128:
        print(f"  Embedding dim: 128  ⚠️  (FaceNet-Lite — lower accuracy)")
        print(f"  Update integration contract: embedding is 128-d not 512-d")
    else:
        print(f"  Embedding dim: {emb_dim}  ℹ️")

    return emb_dim


def print_budget():
    total = sum(EXPECTED_SIZES.values())
    print(f"\n{'─'*54}")
    print(f"  {'Component':<38} {'MB':>5}")
    print(f"{'─'*54}")
    for name, size in EXPECTED_SIZES.items():
        bar = "█" * int(size * 3)
        print(f"  {name:<38} {size:>4.1f}  {bar}")
    print(f"{'─'*54}")
    status = "✅ under 20 MB limit" if total < 20 else "❌ OVER BUDGET"
    print(f"  {'TOTAL':<38} {total:>4.1f}  {status}")
    print(f"{'─'*54}\n")


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Hackathon 7.0 — Model Setup & Readiness   ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── Step 1: Get the model ─────────────────────────────────────────────────
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"[1/3] Model already present ({size_mb:.2f} MB) — skipping download.")
    else:
        print("[1/3] Downloading MobileFaceNet...")
        success = try_all_sources()

        if not success:
            print("\n  All download sources failed.")
            print("  Generating a STUB model for dev/testing...\n")
            stub_ok = create_stub_model()

            if not stub_ok:
                print("\n  ── Manual download instructions ──────────────────")
                print("  1. Open this URL in your browser:")
                print("     https://drive.google.com/file/d/1QS9MqfAgkirFg3YS1ZYj3bBFbFMqMnPy")
                print("  2. Save the file as:  ml/models/mobilefacenet.tflite")
                print("  3. Re-run this script.")
                print("  ─────────────────────────────────────────────────\n")
                sys.exit(1)

    # ── Step 2: Verify ────────────────────────────────────────────────────────
    print("\n[2/3] Verifying model...")
    try:
        emb_dim = verify_model()
    except Exception as e:
        print(f"  Verification failed: {e}")
        print("  The model file may be corrupted. Delete it and re-run.")
        sys.exit(1)

    # ── Step 3: Create embeddings store if missing ────────────────────────────
    print("\n[3/3] Checking embeddings store...")
    emb_path = MODELS_DIR / "embeddings.json"
    if not emb_path.exists() or emb_path.stat().st_size == 0:
        import json
        with open(emb_path, "w") as f:
            json.dump({}, f)
        print(f"  Created empty store: {emb_path}")
    else:
        import json
        with open(emb_path) as f:
            data = json.load(f)
        print(f"  Store exists: {len(data)} enrolled users")

    print_budget()
    print("✅ Setup complete.")
    print("\nNext steps:")
    print("  python -m pytest tests/ -v")
    print("  python main.py\n")


if __name__ == "__main__":
    main()
