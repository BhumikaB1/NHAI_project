"""
ml/server.py
-------------
HTTP bridge between React Native (Person 2) and the ML pipeline (Person 1).
Person 2 calls these endpoints with a base64 image frame.

Run: python ml/server.py
     pip install flask first

Endpoints:
  POST /authenticate  - verify face + liveness
  POST /register      - enroll a new user
  POST /new_session   - reset liveness state
  GET  /health        - check server is running
"""

from flask import Flask, request, jsonify
import cv2, numpy as np, base64, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import FaceAuthSystem

app    = Flask(__name__)
system = FaceAuthSystem()
system.start_session()


def decode_image(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    arr       = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "enrolled_users": len(system._store)})


@app.route('/new_session', methods=['POST'])
def new_session():
    system.start_session()
    return jsonify({"status": "ok"})


@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Input:  { "image": "<base64 jpg string>" }
    Output: {
        "faceDetected":  true,
        "liveness":      "PASS" | "FAIL" | "PENDING",
        "similarity":    0.925,
        "authenticated": true,
        "matchedUserId": "emp_001",
        "instruction":   "Please blink"
    }
    """
    try:
        data  = request.get_json()
        frame = decode_image(data['image'])
        result = system.authenticate(frame)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/register', methods=['POST'])
def register():
    """
    Input:  { "userId": "emp_001", "name": "Bhumika", "image": "<base64>" }
    Output: { "registered": true, "userId": "emp_001", "error": null }
    """
    try:
        data   = request.get_json()
        frame  = decode_image(data['image'])
        result = system.register_user(data['userId'], data['name'], frame)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    print("\n[Server] ML pipeline HTTP bridge running on http://localhost:5000")
    print("[Server] Endpoints: /health  /authenticate  /register  /new_session\n")
    app.run(host='0.0.0.0', port=5000, debug=False)