from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import openvino as ov
import os
import time
import base64
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Config ──────────────────────────────────────
OV_XML     = r'G:\My Drive\pedestrian_detection\models\openvino_int8\best_int8_openvino_model\best.xml'
VAL_IMGS   = r'G:\My Drive\pedestrian_detection\dataset\images\val'
UPLOAD_DIR = r'G:\My Drive\pedestrian_detection\webapp\uploads'
RESULTS_DIR= r'G:\My Drive\pedestrian_detection\webapp\results'
CONF_THRES = 0.25
IOU_THRES  = 0.45
IMG_SIZE   = 640

ALLOWED_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_VID = {'mp4', 'avi', 'mov'}
# ────────────────────────────────────────────────

# Load OpenVINO model once at startup
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
print("Model loaded!")

def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r    = min(size/h, size/w)
    nh, nw = int(h*r), int(w*r)
    img_r  = cv2.resize(img, (nw, nh))
    top    = (size-nh)//2
    left   = (size-nw)//2
    padded = cv2.copyMakeBorder(img_r, top, size-nh-top,
                                 left, size-nw-left,
                                 cv2.BORDER_CONSTANT,
                                 value=(114,114,114))
    return padded, r, left, top

def detect(img):
    orig_h, orig_w = img.shape[:2]
    padded, r, pl, pt = letterbox(img, IMG_SIZE)
    inp = padded[:,:,::-1].transpose(2,0,1)[np.newaxis,:].astype(np.float32)/255.0
    out = req.infer({0: inp})[compiled.output(0)][0].T
    boxes = []
    for p in out:
        if p[4] < CONF_THRES:
            continue
        cx,cy,w,h = p[:4]
        x1 = (cx-w/2-pl)/r; y1 = (cy-h/2-pt)/r
        x2 = (cx+w/2-pl)/r; y2 = (cy+h/2-pt)/r
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(orig_w,x2), min(orig_h,y2)
        if (y2-y1) > (x2-x1):
            boxes.append([x1,y1,x2,y2,float(p[4])])
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if
                 (min(best[2],b[2])-max(best[0],b[0])) *
                 (min(best[3],b[3])-max(best[1],b[1])) /
                 ((best[2]-best[0])*(best[3]-best[1]) +
                  (b[2]-b[0])*(b[3]-b[1]) + 1e-6) < IOU_THRES]
    return keep

def draw_boxes(img, boxes):
    out = img.copy()
    for box in boxes:
        x1,y1,x2,y2,conf = box
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f'Pedestrian {conf:.2f}'
        (lw,lh),_ = cv2.getTextSize(label,
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1,y1-25),(x1+lw+8,y1),(0,255,0),-1)
        cv2.putText(out, label, (x1+4,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return out

def img_to_base64(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

# ── Routes ──────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset_images')
def dataset_images():
    images = sorted([f for f in os.listdir(VAL_IMGS)
                     if f.endswith('.png') or f.endswith('.jpg')])
    return jsonify({'images': images})  # all images

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files and 'filename' not in request.json:
        return jsonify({'error': 'No file provided'}), 400

    if 'file' in request.files:
        file = request.files['file']
        if not allowed_file(file.filename, ALLOWED_IMG):
            return jsonify({'error': 'Invalid file type'}), 400
        fname = secure_filename(file.filename)
        path  = os.path.join(UPLOAD_DIR, fname)
        file.save(path)
        img = cv2.imread(path)
    else:
        fname = request.json['filename']
        img   = cv2.imread(os.path.join(VAL_IMGS, fname))

    if img is None:
        return jsonify({'error': 'Cannot read image'}), 400

    start  = time.time()
    boxes  = detect(img)
    elapsed= (time.time()-start)*1000

    result = draw_boxes(img, boxes)

    confs = [round(b[4]*100, 1) for b in boxes]
    avg_conf = round(sum(confs)/len(confs), 1) if confs else 0

    return jsonify({
        'original'   : img_to_base64(img),
        'detected'   : img_to_base64(result),
        'count'      : len(boxes),
        'inference_ms': round(elapsed, 1),
        'avg_conf'   : avg_conf,
        'confidences': confs,
    })

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not allowed_file(file.filename, ALLOWED_VID):
        return jsonify({'error': 'Invalid file type'}), 400

    fname = secure_filename(file.filename)
    input_path  = os.path.join(UPLOAD_DIR, fname)
    output_path = os.path.join(RESULTS_DIR, 'detected_' + fname)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count  = 0
    total_dets   = 0
    total_time   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start  = time.time()
        boxes  = detect(frame)
        elapsed= (time.time()-start)*1000
        total_time += elapsed

        result = draw_boxes(frame, boxes)

        # Add overlay
        cv2.putText(result, f'Pedestrians: {len(boxes)}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(result, f'OpenVINO INT8 | YOLOv8s',
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        out.write(result)
        total_dets  += len(boxes)
        frame_count += 1

    cap.release()
    out.release()

    avg_fps = round(1000 / (total_time/frame_count), 1) if frame_count else 0

    return jsonify({
        'output_video' : 'detected_' + fname,
        'total_frames' : frame_count,
        'total_dets'   : total_dets,
        'avg_dets'     : round(total_dets/frame_count, 2) if frame_count else 0,
        'avg_fps'      : avg_fps,
    })

@app.route('/download_video/<filename>')
def download_video(filename):
    path = os.path.join(RESULTS_DIR, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)