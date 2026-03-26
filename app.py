import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import RTDETR

# ==========================================
# --- MASTER LEVEL PYTHON HACKS ---
# ==========================================

# Fix 1: Bypass PyTorch 2.6+ Security Block for YOLOv9
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# Fix 2: Numpy Version Translation (Colab used Numpy 2, Laptop has Numpy 1)
# This tricks PyTorch into seamlessly reading the Colab model without crashing.
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']

# ==========================================
# --- WEB APPLICATION CODE ---
# ==========================================

# --- 1. Page Configuration ---
st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")

# --- 2. Sidebar Configuration ---
st.sidebar.title("⚙️ System Settings")
st.sidebar.write("Configure the AI models and parameters:")

model_choice = st.sidebar.selectbox(
    "🧠 Choose AI Architecture",
    ("YOLOv9 (Convolutional Neural Network)", "RT-DETR (Vision Transformer)")
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
st.sidebar.markdown("---")

if model_choice == "YOLOv9 (Convolutional Neural Network)":
    st.sidebar.success("Model Status: Loaded\n\nArchitecture: YOLOv9-c\n\nOfficial mAP50: 99.1%")
else:
    st.sidebar.info("Model Status: Loaded\n\nArchitecture: RT-DETR-Large\n\nOfficial mAP50: 98.9%")

# --- 3. Main UI ---
st.title("🔍 Multi-Model PCB Defect Detection")
st.write("Upload a raw image of a Printed Circuit Board to run the comparative inference.")

# --- 4. Load the AI Model (Dynamically) ---
@st.cache_resource
def load_model(choice):
    if choice == "YOLOv9 (Convolutional Neural Network)":
        # Load YOLOv9 using the original PyTorch codebase via Hub
        model = torch.hub.load('WongKinYiu/yolov9', 'custom', 'yolov9_best.pt', trust_repo=True)
        return model, "yolov9"
    else:
        # Load RT-DETR using the modern Ultralytics library
        model = RTDETR('rtdetr_best.pt')
        return model, "rtdetr"

try:
    model, model_type = load_model(model_choice)
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")

# --- 5. File Upload & Inference ---
uploaded_file = st.file_uploader("Upload PCB Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    # Create a writable copy for YOLOv9 (OpenCV can't draw on read-only arrays)
    if model_type == "yolov9":
        writable_image = np.array(image.convert("RGB")).copy()
    
    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')
        
    with col2:
        st.subheader(f"Results: {model_choice.split(' ')[0]}")
        with st.spinner(f'Running {model_choice.split(" ")[0]} inference...'):
            
            # --- ARCHITECTURE BRANCHING LOGIC ---
            if model_type == "yolov9":
                # YOLOv9 Inference
                model.conf = confidence_threshold
                results = model(writable_image, size=640)
                
                annotated_img = results.render()[0]
                st.image(annotated_img, channels="RGB", width='stretch')
                num_defects = len(results.pandas().xyxy[0])
                
            else:
                # RT-DETR Inference 
                results = model.predict(image, conf=confidence_threshold)
                annotated_img = results[0].plot()
                st.image(annotated_img, channels="BGR", width='stretch')
                num_defects = len(results[0].boxes)
            
    if num_defects > 0:
        st.error(f"⚠️ Warning: {num_defects} defect(s) detected using {model_choice.split(' ')[0]}.")
    else:
        st.success(f"✅ PCB passed inspection. No defects detected by {model_choice.split(' ')[0]}.")