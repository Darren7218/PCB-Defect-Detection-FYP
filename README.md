# 🔍 Multi-Model PCB Defect Detection System — Comparative Study of CNNs and Vision Transformers for Industrial Inspection

The PCB Defect Detection System is an interactive, multi-architecture Computer Vision web application built for automated Printed Circuit Board (PCB) quality inspection. This Final Year Project (FYP) serves as a comprehensive comparative study evaluating the performance of state-of-the-art Convolutional Neural Networks (CNNs) against Vision Transformers (ViTs) in highly precise industrial environments.

### 🧠 Architectures Compared
* **YOLOv9-c:** A highly optimized Convolutional Neural Network designed for rapid edge-inference.
* **RT-DETR-Large:** A Real-Time DEtection TRansformer utilizing global self-attention mechanisms.

## 📂 Dataset Acknowledgment
The training and testing data used to build these models is based on the open-source PCB defect dataset provided by [Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB). I extend my gratitude to the original authors for curating and providing the foundational dataset containing standard PCB defects (e.g., missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper) which made this comparative study possible.

## Project Overview

Traditional Automated Optical Inspection (AOI) systems in PCB manufacturing often struggle with complex, nuanced, or irregular defects, leading to high false-positive rates and bottlenecks in production lines. This project explores Deep Learning as a modern solution by building a robust inspection pipeline. Rather than just deploying a single model, this system allows users to seamlessly toggle between a highly optimized edge-inference CNN (**YOLOv9-c**) and a global self-attention Vision Transformer (**RT-DETR-Large**) to directly compare their efficacy in real-time defect detection.

## Problem Statement

In industrial PCB manufacturing, defects such as *missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper* can cause catastrophic electronic failures. Manual inspection is slow and error-prone, while standard computer vision algorithms lack adaptability. Deep learning offers a solution, but engineers face a critical architectural choice: do they use mathematically efficient Convolutional Neural Networks (YOLO) designed for real-time edge devices, or do they leverage the heavy, context-aware global attention mechanisms of Vision Transformers (RT-DETR)? This project addresses that dilemma through rigorous empirical testing and an interactive visual comparison.

## Tech Stack

[![Tech Stack Architecture](docs/img/tech-stack.png)](docs/img/tech-stack.png)
*(Placeholder: Add an architecture tech-stack graphic here)*

| Category | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Language** | Python |
| **Deep Learning Framework** | PyTorch, Ultralytics |
| **CNN Architecture** | YOLOv9-c (PyTorch Hub) |
| **ViT Architecture** | RT-DETR-Large (Real-Time DEtection TRansformer) |
| **Image Processing** | OpenCV, PIL |
| **Hardware / Acceleration** | Tesla L4 GPU, CUDA |

## Additional Technologies

| Category | Tool / Technology | Usage in Project |
| :--- | :--- | :--- |
| **Data Augmentation** | Albumentations | Enhances dataset robustness via dynamic image transformations |
| **Format Conversion** | Pycocotools | Handles bounding box conversions and annotations formatting |
| **Deployment Export** | ONNX Runtime | Converts raw PyTorch `.pt` weights to optimized `.onnx` for faster FPS |
| **Environment Control** | Jupyter Notebook | Segregated execution for training, hyperparameter tuning, and validation |

## Key Features

* **Multi-Architecture Integration:** Hot-swap between YOLOv9 and RT-DETR within the same user interface.
* **Real-Time Streamlit Interface:** Upload raw PCB images and run instant defect detection visualizations with interactive bounding boxes and confidence scores.
* **Strict Defect Classification:** Accurately identifies 6 distinct industrial PCB defects.
* **Deployment Optimization:** Includes pipelines to convert heavy PyTorch models into lightweight ONNX graphs for edge-device deployment benchmarking.
* **End-to-End Reproducibility:** Segregated Jupyter notebooks detailing data ingestion, bounding box conversion, hyperparameter tuning, and blind-testing.

## What This Project Demonstrates

* Deep learning model orchestration and comparison.
* Handling domain-specific constraints (e.g., stabilizing Vision Transformer attention matrices using `AdamW`).
* Industrial computer vision preprocessing and annotation workflows.
* Model deployment optimization and FPS benchmarking.
* Practical product thinking by wrapping complex ML models in an accessible, user-friendly frontend dashboard.

## How It Works

[![Workflow Diagram](docs/img/workflow.png)](docs/img/workflow.png)
*(Placeholder: Add a diagram showing the data flow from upload to detection)*

### 1. Data Ingestion & Preprocessing
Raw PCB images and annotations are formatted and automated bounding box conversions are executed via the `dataPreprocessing_yaml.ipynb` pipeline. This generates a master `data.yaml` configuration ensuring the models train on an identical, standardized 640x640 resolution dataset.

### 2. Dual-Model Training Pipeline
Training is strictly segregated to respect the mathematical constraints of each architecture:
* **CNN Route:** Evaluates YOLOv9 utilizing standard SGD/Adam optimization for rapid convergence.
* **Transformer Route:** Evaluates RT-DETR, specifically requiring hyperparameter adjustments like utilizing `AdamW` to stabilize the heavy self-attention mechanisms over 100 epochs.

### 3. Web UI Inference & Routing
When a user uploads an image to the Streamlit app, they select their desired architecture from the sidebar. The application dynamically loads the specific pre-trained weights (`yolov9_best.pt` or `rtdetr_best.pt`) and routes the image tensor through the selected neural network.

### 4. Evaluation & Visualization
The output generates precise bounding boxes overlaid on the PCB image, highlighting the defect location, the specific defect class, and the model's confidence threshold.

## Current Implementation Status

This FYP is a **fully functional proof-of-concept**. Both YOLOv9 and RT-DETR models have been successfully trained for 100 epochs. The Streamlit web application is operational, successfully handling local image uploads, model swapping, and real-time bounding box visualization. The ONNX conversion scripts are functional and ready for future edge-device integration.

## Performance Summary

### Direct Model Comparison (Official Blind Test)

Models were evaluated on an unseen test split to ensure unbiased metric extraction.

| Metric | YOLOv9-c (CNN) | RT-DETR-Large (Transformer) | Status |
| :--- | :--- | :--- | :--- |
| **mAP50** (General Detection) | **99.1%** | 98.9% | YOLOv9 Slight Edge |
| **mAP50-95** (Strict BBox Accuracy)| **68.3%** | 60.2% | **YOLOv9 Superior** |
| **Precision** | **98.1%** | 97.5% | YOLOv9 Superior |
| **Recall** | **99.2%** | 98.9% | YOLOv9 Superior |

**Conclusion:** In a direct comparative analysis, the Convolutional Neural Network (YOLOv9) outperformed the Vision Transformer (RT-DETR) across all key metrics. While both achieved near-perfect general detection (mAP50), YOLOv9 demonstrated a significant 8.1% advantage in strict bounding box accuracy (mAP50-95), proving to be the superior and more efficient architecture for highly precise industrial PCB inspection.

## Technical Highlights

### Architecture Constraints & Tuning
A major highlight of this research was handling the differing training requirements of the architectures. While YOLOv9 converged smoothly with standard practices, the Vision Transformer (RT-DETR) required careful application of the `AdamW` optimizer and specific learning rate schedules to prevent its self-attention matrices from destabilizing early in the training loop.

### ONNX FPS Optimization
To move beyond just academic metrics, the project includes `weightConvertOnnx_checkFPS.ipynb`. This demonstrates practical engineering by stripping the models of their heavy PyTorch dependencies, converting them to `.onnx` formats, and benchmarking frames-per-second (FPS) to simulate real-world factory conveyor belt speeds.

## Challenges

* **GPU Resource Management:** Training a Large Vision Transformer (RT-DETR) requires significantly more VRAM and computational power than YOLOv9. Batch sizes had to be carefully managed to prevent CUDA Out-Of-Memory (OOM) errors on the Tesla L4 GPU.
* **Annotation Formatting:** Converting and maintaining annotation integrity (YOLO txt formats vs COCO JSON formats) during data augmentations required building custom preprocessing scripts to ensure no defect labels were lost or misaligned.

## Future Improvements

* **TensorRT Optimization:** Upgrading the ONNX models to NVIDIA TensorRT engines for maximum inference speed on industrial edge devices.
* **Edge Deployment:** Porting the system to an NVIDIA Jetson Nano or Raspberry Pi to test real-world factory floor latency.
* **Instance Segmentation:** Upgrading bounding boxes to pixel-perfect segmentation masks to calculate the exact surface area of PCB defects (e.g., measuring the exact size of a "mouse bite").

## Dataset Acknowledgment & Disclaimer

This is an academic Final Year Project developed at **Universiti Tunku Abdul Rahman (UTAR)**. 

The training and testing data used to build these models is based on the open-source PCB defect dataset provided by [Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB). Deepest gratitude is extended to the original authors for curating and providing the foundational dataset containing standard PCB defects, without which this comparative study would not have been possible.


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

## 📓 Notebooks & Reproducibility
To ensure full transparency and reproducibility of the comparative study, the complete training and evaluation pipelines are provided in the notebooks/ directory:

* **dataPreprocessing_yaml.ipynb**: Scripts for dataset formatting, automated bounding box conversions, and generating the master data.yaml configuration.

* **YOLO.ipynb & RTDETR_100epoch.ipynb**: The core training loops for both architectures, detailing the specific hyperparameter tuning (e.g., utilizing AdamW to stabilize the Vision Transformer's attention matrices).

* **testYOLO.ipynb & testRT_DETR.ipynb**: The rigorous validation and blind-testing scripts used to generate the final confusion matrices and performance metrics.

* **weightConvertOnnx_checkFPS.ipynb**: Deployment optimization scripts converting raw PyTorch weights (.pt) into the ONNX format for accelerated inference and FPS benchmarking.
