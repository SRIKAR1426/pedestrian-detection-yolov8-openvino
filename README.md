# Pedestrian Detection using YOLOv8 and OpenVINO

## Overview

This project presents a real-time pedestrian detection system using the YOLOv8 deep learning model combined with OpenVINO optimization for high-performance inference. The system is designed to detect pedestrians accurately in images and videos while improving inference speed and deployment efficiency.

The project was developed as a major academic project focusing on computer vision, deep learning, and AI-based surveillance applications.

---

## Features

- Real-time pedestrian detection
- YOLOv8-based object detection
- OpenVINO optimized inference
- Image and video detection support
- Web application interface
- Performance benchmarking
- Before vs after optimization comparison

---

## Technologies Used

- Python
- YOLOv8
- OpenCV
- OpenVINO Toolkit
- Flask
- HTML
- CSS
- JavaScript

---

## Project Structure

```bash
pedestrian-detection-yolov8-openvino/
│
├── models/
├── results/
├── test_images/
├── webapp/
├── benchmark.py
├── demo.py
├── inference_video.py
├── .gitignore
└── README.md
```

---

## Dataset

The model was trained and evaluated using the KITTI Vision Benchmark Dataset for pedestrian detection tasks.

Dataset Link:
https://www.cvlibs.net/datasets/kitti/

---

## Installation

### Clone Repository

```bash
git clone https://github.com/SRIKAR1426/pedestrian-detection-yolov8-openvino.git
```

### Navigate to Project Folder

```bash
cd pedestrian-detection-yolov8-openvino
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Detection Demo

```bash
python demo.py
```

### Run Video Inference

```bash
python inference_video.py
```

### Run Web Application

```bash
python app.py
```

---

## Results

The project achieved accurate pedestrian detection with improved inference performance using OpenVINO optimization techniques.

Performance improvements include:
- Faster inference speed
- Reduced latency
- Better deployment efficiency

---

## Applications

- Smart surveillance systems
- Traffic monitoring
- Autonomous driving support
- Smart city solutions
- Public safety systems

---

## Future Scope

- Integration with edge devices
- Multi-object tracking support
- Live CCTV integration
- Cloud deployment
- Advanced analytics dashboard

---

## Author

Sai Srikar

---

## License

This project is developed for educational and research purposes.