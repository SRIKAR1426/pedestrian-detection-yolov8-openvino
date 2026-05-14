import cv2
import numpy as np
import openvino as ov
import os
import time
from pathlib import Path

# ── Config ──────────────────────────────────────
OV_XML    = r'G:\My Drive\pedestrian_detection\models\openvino_int8\best_int8_openvino_model\best.xml'
VAL_IMGS  = r'G:\My Drive\pedestrian_detection\dataset\images\val'
OUTPUT    = r'G:\My Drive\pedestrian_detection\results\demo_output'
CONF      = 0.25
IOU       = 0.45
IMG_SIZE  = 640
# ────────────────────────────────────────────────

os.makedirs(OUTPUT, exist_ok=True)

# Load OpenVINO model
print("Loading OpenVINO INT8 model...")
core     = ov.Core()
model    = core.read_model(OV_XML)
compiled = core.compile_model(
    model       = model,
    device_name = 'CPU',
    config      = {
        'PERFORMANCE_HINT'      : 'LATENCY',
        'NUM_STREAMS'           : '1',
        'INFERENCE_NUM_THREADS' : '4',
    }
)
req = compiled.create_infer_request()
print("Model loaded!\n")

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size/h, size/w)
    nh, nw = int(h*r), int(w*r)
    img_r = cv2.resize(img, (nw, nh))
    top  = (size-nh)//2
    left = (size-nw)//2
    padded = cv2.copyMakeBorder(img_r, top, size-nh-top,
                                 left, size-nw-left,
                                 cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, r, left, top

def detect(img):
    orig_h, orig_w = img.shape[:2]
    padded, r, pl, pt = letterbox(img, IMG_SIZE)
    inp = padded[:,:,::-1].transpose(2,0,1)[np.newaxis,:].astype(np.float32)/255.0
    out = req.infer({0: inp})[compiled.output(0)][0].T
    boxes = []
    for p in out:
        if p[4] < CONF:
            continue
        cx,cy,w,h = p[:4]
        x1 = (cx-w/2-pl)/r; y1 = (cy-h/2-pt)/r
        x2 = (cx+w/2-pl)/r; y2 = (cy+h/2-pt)/r
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(orig_w,x2), min(orig_h,y2)
        if (y2-y1) > (x2-x1):  # taller than wide
            boxes.append([x1,y1,x2,y2,p[4]])
    # NMS
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if 
                 (min(best[2],b[2])-max(best[0],b[0])) *
                 (min(best[3],b[3])-max(best[1],b[1])) /
                 ((best[2]-best[0])*(best[3]-best[1]) +
                  (b[2]-b[0])*(b[3]-b[1]) + 1e-6) < IOU]
    return keep

def draw(img, boxes):
    out = img.copy()
    for box in boxes:
        x1,y1,x2,y2,conf = box
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f'Pedestrian {conf:.2f}'
        (lw,lh),_ = cv2.getTextSize(label,
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1,y1-25), (x1+lw+8,y1), (0,255,0), -1)
        cv2.putText(out, label, (x1+4,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    # Info overlay
    cv2.putText(out, f'Pedestrians: {len(boxes)}',
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(out, 'OpenVINO INT8 | YOLOv8s',
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    return out

# Get all validation images
images = [
    '006395.png',  # 18 pedestrians
    '005086.png',  # 16 pedestrians
    '006613.png',  # 11 pedestrians
    '000109.png',  # 10 pedestrians
    '000200.png',  # 10 pedestrians
    '000487.png',  # 10 pedestrians
    '003784.png',  # 9 pedestrians
    '001366.png',  # 9 pedestrians
    '005430.png',  # 9 pedestrians
    '003133.png',  # 9 pedestrians
]

print(f"Found {len(images)} validation images")
print("Controls:")
print("  SPACE or N = next image")
print("  B          = previous image")
print("  S          = save current image")
print("  Q          = quit\n")

idx = 0
while True:
    img_name = images[idx]
    img_path = os.path.join(VAL_IMGS, img_name)
    
    img = cv2.imread(img_path)
    if img is None:
        idx = (idx+1) % len(images)
        continue
    
    # Detect
    start = time.time()
    boxes = detect(img)
    elapsed = (time.time()-start)*1000
    
    # Draw
    result = draw(img, boxes)
    
    # Add image info
    cv2.putText(result, f'{img_name} ({idx+1}/{len(images)})',
                (10, result.shape[0]-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(result, f'Inference: {elapsed:.1f}ms',
                (10, result.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    print(f"[{idx+1}/{len(images)}] {img_name} → {len(boxes)} pedestrians | {elapsed:.1f}ms")
    
    cv2.imshow('Pedestrian Detection — OpenVINO INT8', result)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_path = os.path.join(OUTPUT, f'detected_{img_name}')
        cv2.imwrite(save_path, result)
        print(f"Saved: {save_path}")
    elif key == ord('b'):
        idx = (idx-1) % len(images)
    else:
        idx = (idx+1) % len(images)

cv2.destroyAllWindows()
print("\nDemo finished!")