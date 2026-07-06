import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from ultralytics import RTDETR

try:
    from patchcore_l2l3 import PatchCoreL2L3
except ImportError:
    PatchCoreL2L3 = None


YOLO_LABEL = "YOLOv9 (Convolutional Neural Network)"
RTDETR_LABEL = "RT-DETR (Vision Transformer)"
PATCHCORE_LABEL = "PatchCore L2/L3 (Unsupervised Anomaly)"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_runtime_patches():
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(
        *args, **{**kwargs, "weights_only": False}
    )

    numpy_core = sys.modules.get("numpy.core")
    numpy_multiarray = sys.modules.get("numpy.core.multiarray")
    if "numpy._core" not in sys.modules and numpy_core is not None:
        sys.modules["numpy._core"] = numpy_core
    if "numpy._core.multiarray" not in sys.modules and numpy_multiarray is not None:
        sys.modules["numpy._core.multiarray"] = numpy_multiarray


@st.cache_resource(show_spinner=False)
def load_model(choice):
    if choice == YOLO_LABEL:
        model = torch.hub.load(
            "WongKinYiu/yolov9",
            "custom",
            "yolov9_best.pt",
            trust_repo=True,
        )
        return model, "yolov9"

    if choice == RTDETR_LABEL:
        model = RTDETR("rtdetr_best.pt")
        return model, "rtdetr"

    if PatchCoreL2L3 is None:
        raise RuntimeError("patchcore_l2l3.py must be next to app.py to use PatchCore.")
    model = PatchCoreL2L3.from_checkpoint("patchcore_l2l3.pt")
    return model, "patchcore"


def normalize_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def open_uploaded_image(uploaded_file):
    uploaded_file.seek(0)
    return Image.open(uploaded_file)


def yolo_records(results):
    table = results.pandas().xyxy[0]
    records = []
    for _, row in table.iterrows():
        records.append(
            {
                "class": str(row.get("name", row.get("class", "unknown"))),
                "confidence": float(row.get("confidence", 0.0)),
                "xmin": float(row.get("xmin", 0.0)),
                "ymin": float(row.get("ymin", 0.0)),
                "xmax": float(row.get("xmax", 0.0)),
                "ymax": float(row.get("ymax", 0.0)),
            }
        )
    return records


def rtdetr_records(result):
    records = []
    names = result.names or {}
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        xmin, ymin, xmax, ymax = box.xyxy[0].detach().cpu().numpy().tolist()
        records.append(
            {
                "class": label,
                "confidence": float(box.conf[0].item()),
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
            }
        )
    return records


def patchcore_records(result, image):
    if not result["is_anomaly"]:
        return []
    width, height = image.size
    return [
        {
            "class": "anomaly",
            "confidence": float(result["score"]),
            "xmin": 0.0,
            "ymin": 0.0,
            "xmax": float(width),
            "ymax": float(height),
        }
    ]


def inspect_image(model, model_type, image, image_name, confidence, image_size, patchcore_threshold=None):
    start = time.perf_counter()

    if model_type == "yolov9":
        model.conf = confidence
        writable_image = np.array(image).copy()
        results = model(writable_image, size=image_size)
        annotated_image = results.render()[0]
        detections = yolo_records(results)
        image_channels = "RGB"
    elif model_type == "rtdetr":
        results = model.predict(image, conf=confidence, imgsz=image_size, verbose=False)
        result = results[0]
        annotated_image = result.plot()
        detections = rtdetr_records(result)
        image_channels = "BGR"
    else:
        result = model.predict(image, threshold=patchcore_threshold)
        annotated_image = result["overlay"]
        detections = patchcore_records(result, image)
        image_channels = "RGB"

    inference_ms = (time.perf_counter() - start) * 1000
    classes = sorted({detection["class"] for detection in detections})
    max_confidence = max((detection["confidence"] for detection in detections), default=0.0)

    return {
        "image": image_name,
        "status": "Defects found" if detections else "Pass",
        "defects": len(detections),
        "classes": ", ".join(classes) if classes else "-",
        "max_confidence": round(max_confidence, 4),
        "inference_ms": round(inference_ms, 1),
        "detections": detections,
        "annotated_image": annotated_image,
        "image_channels": image_channels,
    }


def find_local_images(folder_path, recursive):
    root = Path(folder_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_summary(results):
    inspected = len(results)
    failed = sum(1 for result in results if result["defects"] > 0)
    defects = sum(result["defects"] for result in results)
    avg_ms = sum(result["inference_ms"] for result in results) / inspected if inspected else 0
    return inspected, failed, defects, avg_ms


def results_dataframe(results):
    rows = [
        {
            "image": result["image"],
            "status": result["status"],
            "defects": result["defects"],
            "classes": result["classes"],
            "max_confidence": result["max_confidence"],
            "inference_ms": result["inference_ms"],
        }
        for result in results
    ]
    return pd.DataFrame(rows)


def detections_dataframe(results):
    rows = []
    for result in results:
        for detection in result["detections"]:
            rows.append({"image": result["image"], **detection})
    return pd.DataFrame(rows)


def run_bulk_inspection(
    items,
    model,
    model_type,
    confidence,
    image_size,
    max_previews,
    patchcore_threshold=None,
):
    progress = st.progress(0)
    status = st.empty()
    results = []
    errors = []

    for index, item in enumerate(items, start=1):
        image_name, image_loader = item
        status.write(f"Inspecting {index} of {len(items)}: {image_name}")

        try:
            image = normalize_image(image_loader())
            results.append(
                inspect_image(
                    model=model,
                    model_type=model_type,
                    image=image,
                    image_name=image_name,
                    confidence=confidence,
                    image_size=image_size,
                    patchcore_threshold=patchcore_threshold,
                )
            )
        except Exception as exc:
            errors.append({"image": image_name, "error": str(exc)})

        progress.progress(index / len(items))

    status.empty()
    render_bulk_results(results, errors, max_previews)


def render_bulk_results(results, errors, max_previews):
    if not results and not errors:
        st.info("No images were inspected.")
        return

    inspected, failed, defects, avg_ms = build_summary(results)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Images inspected", inspected)
    metric_cols[1].metric("Images with defects", failed)
    metric_cols[2].metric("Total detections", defects)
    metric_cols[3].metric("Avg inference", f"{avg_ms:.1f} ms")

    if results:
        summary_df = results_dataframe(results)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download summary CSV",
            summary_df.to_csv(index=False).encode("utf-8"),
            file_name="pcb_inspection_summary.csv",
            mime="text/csv",
        )

        detection_df = detections_dataframe(results)
        if not detection_df.empty:
            st.download_button(
                "Download detection CSV",
                detection_df.to_csv(index=False).encode("utf-8"),
                file_name="pcb_detection_details.csv",
                mime="text/csv",
            )

        preview_results = [result for result in results if result["defects"] > 0][:max_previews]
        if preview_results:
            st.subheader("Defect previews")
            preview_cols = st.columns(2)
            for index, result in enumerate(preview_results):
                with preview_cols[index % 2]:
                    st.image(
                        result["annotated_image"],
                        caption=f"{result['image']} - {result['defects']} detection(s)",
                        channels=result["image_channels"],
                        use_container_width=True,
                    )

    if errors:
        st.warning(f"{len(errors)} image(s) could not be inspected.")
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)


apply_runtime_patches()

st.set_page_config(page_title="PCB Inspection Console", layout="wide")

st.sidebar.title("Inspection Controls")
model_choice = st.sidebar.selectbox("Model", (YOLO_LABEL, RTDETR_LABEL, PATCHCORE_LABEL))
confidence_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05
)
image_size = st.sidebar.select_slider("Inference image size", options=[320, 480, 640, 960, 1280], value=640)
max_previews = st.sidebar.slider("Bulk preview limit", min_value=2, max_value=24, value=8, step=2)
patchcore_threshold = None

if model_choice == PATCHCORE_LABEL:
    patchcore_threshold_input = st.sidebar.number_input(
        "PatchCore threshold override",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Use 0 to keep the threshold stored in patchcore_l2l3.pt.",
    )
    patchcore_threshold = patchcore_threshold_input or None

try:
    with st.spinner("Loading model..."):
        model, model_type = load_model(model_choice)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

st.title("PCB Inspection Console")
st.caption("Tune model settings, inspect one board, or run a batch over many PCB images.")

single_tab, bulk_tab = st.tabs(["Single image", "Bulk inspection"])

with single_tab:
    uploaded_file = st.file_uploader(
        "Upload PCB image",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
    )

    if uploaded_file is not None:
        image = normalize_image(open_uploaded_image(uploaded_file))
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Inspection result")
            with st.spinner("Running inference..."):
                result = inspect_image(
                    model=model,
                    model_type=model_type,
                    image=image,
                    image_name=uploaded_file.name,
                    confidence=confidence_threshold,
                    image_size=image_size,
                    patchcore_threshold=patchcore_threshold,
                )
            st.image(
                result["annotated_image"],
                channels=result["image_channels"],
                use_container_width=True,
            )

        inspected, failed, defects, avg_ms = build_summary([result])
        metric_cols = st.columns(4)
        metric_cols[0].metric("Status", result["status"])
        metric_cols[1].metric("Detections", defects)
        metric_cols[2].metric("Top confidence", f"{result['max_confidence']:.2f}")
        metric_cols[3].metric("Inference", f"{avg_ms:.1f} ms")

        detection_df = detections_dataframe([result])
        if not detection_df.empty:
            st.dataframe(detection_df, use_container_width=True, hide_index=True)

with bulk_tab:
    source_mode = st.radio(
        "Bulk source",
        ("Upload images", "Local folder path"),
        horizontal=True,
    )

    items = []
    if source_mode == "Upload images":
        uploaded_files = st.file_uploader(
            "Upload multiple PCB images",
            type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        items = [
            (uploaded_file.name, lambda uploaded_file=uploaded_file: open_uploaded_image(uploaded_file))
            for uploaded_file in uploaded_files
        ]
    else:
        folder_path = st.text_input("Folder path on this machine", placeholder=r"C:\path\to\pcb-images")
        recursive = st.checkbox("Include subfolders", value=True)
        if folder_path:
            try:
                image_paths = find_local_images(folder_path, recursive)
                st.caption(f"Found {len(image_paths)} supported image(s).")
                items = [
                    (str(path), lambda path=path: Image.open(path))
                    for path in image_paths
                ]
            except Exception as exc:
                st.error(str(exc))

    run_disabled = len(items) == 0
    if st.button("Run bulk inspection", type="primary", disabled=run_disabled):
        run_bulk_inspection(
            items=items,
            model=model,
            model_type=model_type,
            confidence=confidence_threshold,
            image_size=image_size,
            max_previews=max_previews,
            patchcore_threshold=patchcore_threshold,
        )
