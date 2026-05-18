
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_file
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Dict, Tuple
from collections import defaultdict
from werkzeug.utils import secure_filename
from datetime import datetime
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Dashboard global stats
detections_summary = {
    "congestion_level": "Low", 
    "total_unique": 0,
    "current_count": 0,
    "vehicle_counts": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
}

# 2. History List (Initially empty for real data)
video_history = []

class CongestionDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.counted_ids = set()
        self.unique_objects = defaultdict(set)

    def process_frame(self, frame):
        global detections_summary
        results = self.model.track(frame, persist=True, conf=0.45, iou=0.5, imgsz=640)
        
        current_frame_count = 0
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for i, obj_id in enumerate(ids):
                class_name = self.model.names[clss[i]].lower()
                if class_name == 'person': continue

                if obj_id not in self.counted_ids:
                    self.counted_ids.add(obj_id)
                    if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        self.unique_objects[class_name].add(obj_id)

                current_frame_count += 1
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"ID:{obj_id} {class_name}", (x1, y1-10), 0, 0.5, (0, 255, 255), 2)

        # Update stats
        detections_summary.update({
            "total_unique": len(self.counted_ids),
            "current_count": current_frame_count,
            "congestion_level": "High" if current_frame_count > 10 else "Medium" if current_frame_count > 4 else "Low",
            "vehicle_counts": {
                "car": len(self.unique_objects['car']),
                "truck": len(self.unique_objects['truck']),
                "bus": len(self.unique_objects['bus']),
                "motorcycle": len(self.unique_objects['motorcycle'])
            }
        })
        return frame

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/stats')
def stats(): return jsonify(detections_summary) 

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        global detections_summary
        detections_summary = {
            "congestion_level": "Low", "total_unique": 0, "current_count": 0,
            "vehicle_counts": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        }
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return render_template('index.html', filename=filename)

def gen(path):
    global detections_summary, video_history
    # Reset stats for new video
    detections_summary.update({
        "congestion_level": "Low", "total_unique": 0, "current_count": 0,
        "vehicle_counts": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    })
    
    detector = CongestionDetector('yolov8n.pt') 
    cap = cv2.VideoCapture(path)
    filename = os.path.basename(path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = detector.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    # 3. Dynamic History Update: Save after video ends
    new_entry = {
        "id": len(video_history) + 1,
        "filename": filename,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cars": detections_summary["total_unique"],
        "congestion_level": detections_summary["congestion_level"]
    }
    video_history.append(new_entry)
    
    detections_summary.update({"current_count": 0, "congestion_level": "Low"})
    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(gen(os.path.join(app.config['UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history(): return render_template('history.html', history=video_history)

@app.route('/download_report/<int:session_id>')
def download_report(session_id):
    item = next((x for x in video_history if x['id'] == session_id), None)
    if item:
        # 4. Aligned Professional Report Format
        report_text = f"""
==================================================
         INTELLITRAFFIC ANALYTICS REPORT          
==================================================
REPORT DETAILS:
---------------
Session ID      : #00{item['id']}
Source Video    : {item['filename']}
Processed Date  : {item['date']}

TRAFFIC ANALYSIS SUMMARY:
-------------------------
Total Vehicles  : {item['cars']} units
Avg. Density    : {item['congestion_level']}
System Status   : Analysis Successfully Completed

--------------------------------------------------
Generated by: IntelliTraffic AI Surveillance Node
Admin: Prachi Pandhurnekar
==================================================
        """
        proxy = io.BytesIO()
        proxy.write(report_text.encode('utf-8'))
        proxy.seek(0)
        return send_file(proxy, as_attachment=True, download_name=f"Report_Session_{session_id}.txt", mimetype='text/plain')
    return "Not Found", 404

if __name__ == "__main__":
    app.run(debug=True, threaded=True)