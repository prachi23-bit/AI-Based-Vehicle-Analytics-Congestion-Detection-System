import ultralytics
ultralytics.checks()

import cv2
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Run congestion detection on a video.")
parser.add_argument("--weights", type=str, required=True, help="Path to the YOLO model weights")
parser.add_argument("--input", type=str, required=True, help="Path to the input video")
parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to save the output video")

args = parser.parse_args()

@dataclass
class VehicleData:
    position: Tuple[int, int]
    class_id: int
    class_name: str
    speed: float = 0.0

class CongestionDetector:
    def __init__(self, model_path: str, vehicle_threshold: int = 6, speed_threshold: float = 75, speed_window: int = 30):
        self.model = YOLO(model_path)
        self.vehicle_threshold = vehicle_threshold
        self.speed_threshold = speed_threshold
        self.speed_window = speed_window
        self.vehicles: Dict[int, VehicleData] = {}
        self.speed_history: Dict[int, list] = {}
        self.unique_objects = defaultdict(set)

        # IDs: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
        self.allowed_classes = [0, 1, 2, 3, 5, 7, 9]

        # BHAI COLORS YAHAN SWAP KAR DIYE HAIN
        self.class_colors = {
            0: (255, 0, 0),      # Pedestrian (Blue)
            1: (0, 255, 0),      # Bicycle (Green)
            2: (128, 0, 128),    # Car (PURPLE) - Updated!
            3: (255, 255, 0),    # Motorcycle (Cyan)
            5: (0, 255, 255),    # Bus (YELLOW) - Updated!
            7: (255, 0, 255),    # Truck (Magenta)
            9: (0, 200, 0)       # Traffic Light (Dark Green)
        }

    def calculate_speed(self, current_pos, previous_pos, fps):
        dx, dy = current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1]
        return math.sqrt(dx * dx + dy * dy) * fps

    def process_frame(self, frame: np.ndarray, fps: float) -> Tuple[np.ndarray, bool]:
        results = self.model.track(frame, persist=True, conf=0.25)
        vehicle_speeds: List[float] = []

        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                if class_id not in self.allowed_classes: continue

                obj_id = int(box.id.item()) if box.id is not None else None
                if obj_id is None: continue

                x, y, w, h = box.xywh[0]
                center = (int(x), int(y))
                class_name = self.model.names[class_id]

                self._track_unique_objects(obj_id, class_name)
                speed = self._update_vehicle_data(obj_id, center, class_id, class_name.lower(), fps)

                # Speed calculation skip for traffic lights in avg speed
                if speed is not None and class_id != 9:
                    vehicle_speeds.append(speed)

                self._draw_vehicle_info(frame, box, class_id, class_name)

        self._draw_unique_object_count(frame)
        is_congested = self._check_congestion(len(results[0].boxes), vehicle_speeds)
        self._draw_status_info(frame, is_congested, vehicle_speeds)

        return frame, is_congested

    def _track_unique_objects(self, obj_id, class_name):
        self.unique_objects[class_name].add(obj_id)

    def _draw_unique_object_count(self, frame):
        total_unique_objects = sum(len(objects) for objects in self.unique_objects.values())
        unique_counts = [f"{cls}: {len(objs)}" for cls, objs in self.unique_objects.items()]
        x_pos, y_start = 10, 30
        
        total_text = f"Total Unique Objects: {total_unique_objects}"
        (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x_pos-5, y_start-th-5), (x_pos+tw+5, y_start+5), (0,0,0), -1)
        cv2.putText(frame, total_text, (x_pos, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        for i, count_text in enumerate(unique_counts):
            y_pos = y_start + (i + 1) * 25
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x_pos-5, y_pos-th-5), (x_pos+tw+5, y_pos+5), (0,0,0), -1)
            cv2.putText(frame, count_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def _update_vehicle_data(self, vehicle_id, position, class_id, class_name, fps):
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = VehicleData(position, class_id, class_name)
            self.speed_history[vehicle_id] = []
            return None
        speed = self.calculate_speed(position, self.vehicles[vehicle_id].position, fps)
        self.speed_history[vehicle_id].append(speed)
        if len(self.speed_history[vehicle_id]) > self.speed_window: self.speed_history[vehicle_id].pop(0)
        avg_speed = np.mean(self.speed_history[vehicle_id])
        self.vehicles[vehicle_id] = VehicleData(position, class_id, class_name, avg_speed)
        return avg_speed

    def _check_congestion(self, vehicle_count, speeds):
        if not speeds: return False
        avg_speed = np.mean(speeds)
        return vehicle_count > self.vehicle_threshold and avg_speed < self.speed_threshold

    def _draw_vehicle_info(self, frame, box, class_id, class_name):
        x, y, w, h = box.xywh[0]
        tl = (int(x - w/2), int(y - h/2))
        br = (int(x + w/2), int(y + h/2))
        color = self.class_colors.get(class_id, (255, 255, 255))
        
        cv2.rectangle(frame, tl, br, color, 2)
        v_info = self.vehicles.get(int(box.id.item()) if box.id is not None else -1)
        
        # Traffic light (ID 9) logic: No speed text
        if class_id != 9 and v_info and v_info.speed > 0:
            speed_text = f" {v_info.speed:.1f}px/s"
        else:
            speed_text = ""

        cv2.putText(frame, f"{class_name}{speed_text}", (tl[0], tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_status_info(self, frame, is_congested, speeds):
        avg_speed = np.mean(speeds) if speeds else 0
        status_color = (0, 0, 255) if is_congested else (0, 255, 0)
        x_pos, y_start = frame.shape[1] - 220, 30
        texts = [(f"Congestion: {'YES' if is_congested else 'NO'}", status_color), (f"Avg Speed: {avg_speed:.2f} px/s", (255, 255, 255))]

        for i, (text, color) in enumerate(texts):
            y_pos = y_start + i * 35
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x_pos-5, y_pos-th-5), (x_pos+tw+5, y_pos+5), (0,0,0), -1)
            cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cap = cv2.VideoCapture(args.input)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    video_writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    # Path settings
    detector = CongestionDetector(r"D:\Vehicle\Yolov8\traffic_yolo_best.pt")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        annotated_frame, _ = detector.process_frame(frame, fps)
        video_writer.write(annotated_frame)
    cap.release()
    video_writer.release()

if __name__ == "__main__":
    main()