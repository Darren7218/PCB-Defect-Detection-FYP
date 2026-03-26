# 🔍 Multi-Model PCB Defect Detection System

An interactive, multi-architecture Computer Vision system built for automated Printed Circuit Board (PCB) quality inspection. This project serves as a comparative study evaluating the performance of state-of-the-art Convolutional Neural Networks (CNNs) against Vision Transformers (ViTs) in industrial defect detection.

### 🧠 Architectures Compared
* **YOLOv9-c:** A highly optimized Convolutional Neural Network designed for rapid edge-inference.
* **RT-DETR-Large:** A Real-Time DEtection TRansformer utilizing global self-attention mechanisms.

## 📂 Dataset Acknowledgment
The training and testing data used to build these models is based on the open-source PCB defect dataset provided by [Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB). I extend my gratitude to the original authors for curating and providing the foundational dataset containing standard PCB defects (e.g., missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper) which made this comparative study possible.

## 📊 Performance Metrics (Official Blind Test)
Both models were trained for 100 epochs on a custom 640x640 dataset using a Tesla L4 GPU, with hyperparameter tuning customized to the mathematical constraints of each architecture.

| Metric | YOLOv9 (CNN) | RT-DETR (Transformer) |
| :--- | :--- | :--- |
| **mAP50** | **99.1%** | 98.9% |
| **mAP50-95** | **68.3%** | 60.2% |
| **Precision** | **98.1%** | 97.5% |
| **Recall** | **99.2%** | 98.9% |

*Result:* In a direct comparative analysis, the Convolutional Neural Network (YOLOv9) outperformed the Vision Transformer (RT-DETR) across all key metrics. While both models achieved near-perfect general detection (mAP50), YOLOv9 demonstrated a significant advantage in strict bounding box accuracy (mAP50-95), proving to be the superior architecture for highly precise industrial PCB inspection.

## 💻 Streamlit Web Application
This repository includes a fully functional, local web application that allows users to seamlessly swap between the CNN and Transformer architectures to analyze raw PCB images in real-time.

### Installation & Usage
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install streamlit ultralytics torch torchvision pillow opencv-python albumentations pycocotools-windows
3. Ensure both yolov9_best.pt and rtdetr_best.pt weights are in the root directory.
4. Launch the application:
   ```bash
   streamlit run app.py
   
