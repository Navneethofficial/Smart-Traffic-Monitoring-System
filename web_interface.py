from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os
import base64
from io import BytesIO
from PIL import Image
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables
model = None
tracker = None
total_count = []
processing_video = False
current_frame = None
frame_lock = threading.Lock()

# YOLO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def initialize_model(model_name='yolov8n.pt'):
    """Initialize YOLO model and tracker"""
    global model, tracker, total_count
    try:
        model = YOLO(model_name)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        total_count = []
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def process_frame(img, conf_threshold=0.3, counting_line=[370, 297, 750, 297]):
    """Process a single frame for car detection and counting"""
    global tracker, total_count
    
    if model is None or tracker is None:
        return img, len(total_count)
    
    # YOLO detection
    results = model(img, stream=True, verbose=False)
    detections = np.empty((0, 5))
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > conf_threshold:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
    
    # Tracking
    resultsTracker = tracker.update(detections)
    
    # Draw counting line
    cv2.line(img, (counting_line[0], counting_line[1]), 
             (counting_line[2], counting_line[3]), (0, 0, 255), 3)
    
    # Draw tracked objects and count
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Draw corners
        corner_length = min(w, h) // 5
        cv2.line(img, (x1, y1), (x1 + corner_length, y1), (255, 0, 255), 3)
        cv2.line(img, (x1, y1), (x1, y1 + corner_length), (255, 0, 255), 3)
        
        # Draw ID
        cv2.putText(img, f'ID: {id}', (x1, max(35, y1 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # Counting logic
        if counting_line[0] < cx < counting_line[2] and counting_line[1] - 15 < cy < counting_line[3] + 15:
            if id not in total_count:
                total_count.append(id)
                # Flash green when counted
                cv2.line(img, (counting_line[0], counting_line[1]),
                        (counting_line[2], counting_line[3]), (0, 255, 0), 5)
    
    # Draw counter box
    cv2.rectangle(img, (10, 10), (250, 90), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (250, 90), (255, 255, 255), 2)
    cv2.putText(img, 'VEHICLE COUNT', (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, str(len(total_count)), (85, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    return img, len(total_count)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    global processing_video, total_count, tracker
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    conf_threshold = float(request.form.get('confidence', 0.3))
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    file.save(filename)
    
    # Initialize model and reset counter
    initialize_model()
    processing_video = True
    
    # Process video - use H264 codec for better browser compatibility
    output_filename = f'output_{int(time.time())}.mp4'
    output_path = os.path.join('static', 'results', output_filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.join('static', 'results'), exist_ok=True)
    
    cap = cv2.VideoCapture(filename)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(cap.get(cv2.CAP_PROP_FPS), 20.0)  # Ensure minimum fps
    
    # Try different codecs for better compatibility
    fourcc_list = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    
    for fourcc_code in fourcc_list:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                print(f"Using codec: {fourcc_code}")
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        # Fallback to mp4v
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, (frame_width, frame_height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, count = process_frame(frame, conf_threshold)
        out.write(processed_frame)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    processing_video = False
    
    print(f"Video processing complete: {frame_count} frames, {len(total_count)} vehicles counted")
    print(f"Output saved to: {output_path}")
    
    # Verify file exists
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output video file was not created'}), 500
    
    return jsonify({
        'success': True,
        'output_video': f'/static/results/{output_filename}',
        'total_count': len(total_count),
        'frames_processed': frame_count,
        'output_path': output_path
    })

@app.route('/webcam_feed')
def webcam_feed():
    """Generate webcam frames for live detection"""
    return Response(generate_webcam_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_webcam_frames():
    """Generate frames from webcam"""
    global total_count, tracker
    initialize_model()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, count = process_frame(frame)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/get_count')
def get_count():
    """Get current count"""
    return jsonify({'count': len(total_count)})

@app.route('/reset_count', methods=['POST'])
def reset_count():
    """Reset the counter"""
    global total_count, tracker
    total_count = []
    initialize_model()
    return jsonify({'success': True})

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)