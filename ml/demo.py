import cv2, sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import FaceAuthSystem

MODEL_PATH = "models/mobilefacenet.tflite"
EMBEDDINGS_PATH = "models/embeddings.json"

def run():
    print("\n[Demo] Starting... Controls: R=Register  SPACE=New session  Q=Quit")
    system = FaceAuthSystem(model_path=MODEL_PATH, embeddings_path=EMBEDDINGS_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    system.start_session()

    result = {"faceDetected":False,"liveness":"PENDING","similarity":0.0,
              "authenticated":False,"instruction":"Show your face","matchedUserId":None}
    registering = False
    reg_count = 0
    fps, fc, ft = 0.0, 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if time.time()-ft >= 0.5:
            fps = fc/(time.time()-ft); fc=0; ft=time.time()
        result = system.authenticate(frame)
        # Keep full frame for display

        if not registering:
            result = system.authenticate(frame)

        d = frame.copy()

        # Top bar
        cv2.rectangle(d,(0,0),(640,50),(20,20,20),-1)
        cv2.putText(d,"Hackathon 7.0 | Offline Face Auth",
                    (10,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
        cv2.putText(d,f"FPS:{fps:.0f}  R=Register  SPACE=New session  Q=Quit",
                    (10,38),cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,140,140),1)

        # Liveness colour
        lv = result.get("liveness","PENDING")
        col = (0,220,80) if lv=="PASS" else (0,60,220) if lv=="FAIL" else (0,210,255)

        # Bottom panel
        cv2.rectangle(d,(0,370),(640,480),(20,20,20),-1)
        cv2.putText(d,f"LIVENESS: {lv}",
                    (14,398),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)

        sim = result.get("similarity",0.0)
        cv2.rectangle(d,(14,408),(294,420),(60,60,60),-1)
        fill = int(280*min(sim,1.0))
        bc = (0,220,80) if sim>=0.6 else (0,210,255) if sim>=0.4 else (0,60,220)
        if fill>0: cv2.rectangle(d,(14,408),(14+fill,420),bc,-1)
        cv2.putText(d,f"Similarity: {sim:.3f} / threshold 0.60",
                    (14,438),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

        auth = result.get("authenticated",False)
        uid  = result.get("matchedUserId","")
        if auth:
            cv2.putText(d,f"AUTHENTICATED  {uid}",
                        (14,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,80),2)
        elif lv=="PASS":
            cv2.putText(d,"NOT RECOGNISED — Press R to register",
                        (14,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,60,220),2)
        else:
            cv2.putText(d,result.get("instruction",""),
                        (14,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

        instr2 = "REGISTERING — press R to capture" if registering else ""
        if instr2:
            cv2.putText(d,instr2,(160,240),cv2.FONT_HERSHEY_SIMPLEX,0.65,(220,140,0),2)

        cv2.imshow("Hackathon 7.0 — Face Auth",d)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'),27]: break
        elif key == ord(' '):
            system.start_session(); registering=False
            result["instruction"]="New session — show your face"
            print("[Demo] New session.")
        elif key == ord('r'):
            if not registering:
                registering=True
            else:
                reg_count+=1
                uid2=f"user_{reg_count:03d}"
                r=system.register_user(uid2,uid2,frame)
                print(f"[Demo] {'Registered '+uid2 if r['registered'] else 'Failed: '+str(r['error'])}")
                registering=False; system.start_session()

    cap.release(); cv2.destroyAllWindows(); system.close()
    print("[Demo] Exited.")

if __name__=="__main__":
    run()