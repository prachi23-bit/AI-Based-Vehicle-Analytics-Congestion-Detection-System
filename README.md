# 🚦 AI-Based Vehicle Analytics & Congestion Detection System

An intelligent traffic monitoring and congestion detection system built using Python, Flask, OpenCV, and YOLOv8. This project performs real-time vehicle detection, tracking, speed estimation, and congestion analysis from uploaded traffic videos.

---

## 📌 Features

- 🚗 Real-time Vehicle Detection using YOLOv8
- 📊 Traffic Congestion Analysis
- 🧠 Unique Vehicle Counting
- ⚡ Vehicle Speed Estimation
- 🎥 Video Upload & Live Processing
- 📈 Live Dashboard Statistics
- 📝 Downloadable Traffic Analysis Reports
- 🕒 Session History Tracking
- 🌐 Flask-Based Web Application

---

## 🛠️ Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```bash
├── app.py
├── main.py
├── web.py
├── yolov8n.pt
├── uploads/
├── templates/
│   ├── index.html
│   └── history.html
└── README.md
```

---

## 🚀 How It Works

1. User uploads a traffic video.
2. YOLOv8 detects and tracks vehicles frame-by-frame.
3. Vehicles are classified into:
   - Car
   - Bus
   - Truck
   - Motorcycle
4. Vehicle movement is analyzed to estimate speed.
5. Congestion level is determined based on:
   - Vehicle count
   - Average vehicle speed
6. Processed video is streamed live on the dashboard.
7. Traffic analysis reports are generated automatically.

---

## 📸 Supported Vehicle Classes

- 🚘 Car
- 🚌 Bus
- 🚚 Truck
- 🏍️ Motorcycle
- 🚲 Bicycle
- 🚦 Traffic Light
- 🚶 Person

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-congestion-detection.git
cd vehicle-congestion-detection
```

### 2️⃣ Install Required Libraries

```bash
pip install flask ultralytics opencv-python numpy
```

### 3️⃣ Run the Flask Application

```bash
python app.py
```

### 4️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```

---

## ▶️ Run Standalone Detection Script

```bash
python main.py --weights yolov8n.pt --input input_video.mp4 --output output_video.mp4
```

---

## 📊 Congestion Detection Logic

The system marks traffic as congested when:

- Vehicle count exceeds a threshold
- Average vehicle speed becomes low

This helps in detecting:
- Traffic jams
- Slow-moving traffic
- Dense traffic areas

---

## 📷 Output Features

- Vehicle Bounding Boxes
- Vehicle ID Tracking
- Speed Display
- Congestion Status Detection
- Unique Object Counting
- Downloadable Analysis Reports

---

## 🧠 Future Enhancements

- Real-time CCTV Camera Integration
- Smart Traffic Signal Management
- AI-Based Traffic Prediction
- Cloud Deployment
- Heatmap Visualization
- Emergency Vehicle Detection

---

## 👩‍💻 Author

**Prachi Pandhurnekar**  
B.Sc. Data Science Student  
Passionate about AI, Computer Vision & Intelligent Systems

GitHub: https://github.com/prachi23-bit

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!