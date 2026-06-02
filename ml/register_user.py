import cv2, sys, time, argparse, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import FaceAuthSystem

MODEL_PATH = "models/mobilefacenet.tflite"
EMBEDDINGS_PATH = "models/embeddings.json"
CAPTURE_FRAMES = 5

def register(user_id, name):
    print(f"\n[Register] Enrolling: {name}. Press SPACE x5, Q to cancel.")
    system = FaceAuthSystem(model_path=MODEL_PATH, embeddings_path=EMBEDDINGS_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    captured = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        display = frame.copy()
        n = len(captured)
        bw = int(640 * n / CAPTURE_FRAMES)
        cv2.rectangle(display, (0,460),(640,480),(40,40,40),-1)
        cv2.rectangle(display, (0,460),(bw,480),(0,200,80),-1)
        cv2.putText(display, f"SPACE to capture ({n}/{CAPTURE_FRAMES})  Q=quit",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
        cv2.putText(display, f"Enrolling: {name}",
                    (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.putText(display, "Vary angle slightly each capture",
                    (20,455), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        cv2.imshow(f"Register: {name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:
            print("[Register] Cancelled.")
            break

        elif key == ord(' ') and n < CAPTURE_FRAMES:
            res = system.register_user(f"__tmp_{n}", "tmp", frame)
            if res["registered"]:
                emb = system._store._store.get(f"__tmp_{n}", {}).get("embedding")
                if emb is not None:
                    captured.append(emb.copy())
                    print(f"  Captured {n+1}/{CAPTURE_FRAMES}")
                    flash = frame.copy()
                    cv2.rectangle(flash,(0,0),(640,480),(0,200,80),8)
                    cv2.imshow(f"Register: {name}", flash)
                    cv2.waitKey(150)
                else:
                    print(f"  No face detected — try again")
            else:
                print(f"  Failed: {res['error']} — try again")

        if len(captured) == CAPTURE_FRAMES:
            for i in range(CAPTURE_FRAMES):
                system._store._store.pop(f"__tmp_{i}", None)
            avg = np.mean(captured, axis=0)
            norm = np.linalg.norm(avg)
            avg = avg / norm if norm > 0 else avg
            system._store.register(user_id, name, avg)
            print(f"\n[Register] SUCCESS: {name} enrolled as {user_id}")
            print(f"[Register] Total users: {len(system._store)}")
            time.sleep(1.5)
            break

    cap.release()
    cv2.destroyAllWindows()
    system.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    register(a.id, a.name)