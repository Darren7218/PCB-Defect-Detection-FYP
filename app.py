"""
Unified Multi-Product & Multi-Paradigm PCB Defect Inspection Console.

Combines:
  - Closed-Set Multi-Product Specialists: YOLOv11-m (10-class), YOLOv9-c, RT-DETR-Large
  - Open-Vocabulary Detectors: Fine-Tuned YOLO-World-S (with Post-Hoc Platt Calibration),
    Zero-Shot YOLO-World, and Zero-Shot GroundingDINO (10 unified PCB/PCBA defect prompts)
  - Unsupervised / Open-Set Multi-Product Anomaly Engine: PatchCore++ (DINOv2-reg + WRN50 Hybrid,
    Automatic Product-SKU Router, Greedy Coreset + FAISS, True-RGB Overlay, ROI Prototype Classifier)
  - Two-Stage Hybrid Cascade: Specialist Detector + PatchCore++ Open-Set Novelty & Pixel-Area Fusion
  - High-Resolution SAHI Sliced Inference for full-board (2000px-3000px+) PCB images
  - Single-Image, Bulk Folder / Multi-Upload Inspection, & Live Benchmark / Ablation Dashboard
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.anomaly.patchcore_plus import SUPPORTED_EXTENSIONS, PatchCorePlus
from src.datasets.taxonomy import DEFAULT_OVD_PROMPTS_10CLS, UNIFIED_CLASS_NAMES
from src.detection.sahi_and_calibration import (
    ConfidenceCalibrator,
    class_aware_nms_nmm,
    compute_iou_and_ios,
    fuse_detector_and_patchcore,
    render_detections_rgb,
    run_sliced_inference,
)


# ============================================================================
# Runtime Compatibility Patches (PyTorch 2.6+ YOLOv9 Hub & NumPy 1.x/2.x)
# ============================================================================

def apply_runtime_patches() -> None:
    logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
    original_load = torch.load
    if not getattr(original_load, "_pcb_patched", False):
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        _patched_load._pcb_patched = True  # type: ignore[attr-defined]
        torch.load = _patched_load

    numpy_core = sys.modules.get("numpy.core")
    numpy_multiarray = sys.modules.get("numpy.core.multiarray")
    if "numpy._core" not in sys.modules and numpy_core is not None:
        sys.modules["numpy._core"] = numpy_core
    if "numpy._core.multiarray" not in sys.modules and numpy_multiarray is not None:
        sys.modules["numpy._core.multiarray"] = numpy_multiarray


apply_runtime_patches()

REPO_ROOT = Path(__file__).resolve().parent
SEARCH_DIRS = (
    REPO_ROOT / "weights",
    REPO_ROOT,
)


def resolve_weight_path(*candidates: str) -> Optional[Path]:
    """Locate a checkpoint across `weights/` and repository root."""
    for name in candidates:
        for base in SEARCH_DIRS:
            cand = base / name
            if cand.exists():
                return cand
    return None


# ============================================================================
# Model Labels
# ============================================================================

MODEL_YOLOV11M = "YOLOv11-m (Multi-Product 10-Class)"
MODEL_YOLOV9C = "YOLOv9-c (CNN Specialist)"
MODEL_RTDETR = "RT-DETR-Large (Vision Transformer)"
MODEL_YOLOWORLD_FT = "Fine-Tuned YOLO-World-S (Calibrated OVD)"
MODEL_YOLOWORLD_ZS = "YOLO-World-S (Zero-Shot OVD)"
MODEL_GDINO_ZS = "GroundingDINO-Tiny (Zero-Shot OVD)"
MODEL_PATCHCORE_PLUS = "PatchCore++ (DINOv2 Hybrid + Product Router)"
MODEL_HYBRID_CASCADE = "Two-Stage Hybrid (Specialist + PatchCore++ Open-Set)"
MODEL_HYBRID_OVD = "Hybrid Specialist + Calibrated YOLO-World-S"

ALL_MODELS = (
    MODEL_YOLOV11M,
    MODEL_YOLOV9C,
    MODEL_RTDETR,
    MODEL_YOLOWORLD_FT,
    MODEL_YOLOWORLD_ZS,
    MODEL_GDINO_ZS,
    MODEL_PATCHCORE_PLUS,
    MODEL_HYBRID_CASCADE,
    MODEL_HYBRID_OVD,
)
OPENVOCAB_PROMPT_MODELS = {MODEL_YOLOWORLD_ZS, MODEL_GDINO_ZS}


# ============================================================================
# Cached Model Loaders
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_yolov11m():
    from ultralytics import YOLO

    ckpt = resolve_weight_path("yolo11m_multiproduct_best.pt", "yolo11m_best.pt")
    if ckpt is None:
        raise FileNotFoundError(
            "yolo11m_multiproduct_best.pt not found in `weights/` or repo root. "
            "Run `notebooks/colab_01_train_multiproduct_detectors_L4.ipynb` on Colab Pro L4 first."
        )
    return YOLO(str(ckpt))


@st.cache_resource(show_spinner=False)
def load_yolov9c():
    ckpt = resolve_weight_path("yolov9_multiproduct_best.pt", "yolov9_best.pt")
    if ckpt is None:
        raise FileNotFoundError("yolov9_best.pt not found in `weights/` or repo root.")
    return torch.hub.load("WongKinYiu/yolov9", "custom", str(ckpt), trust_repo=True)


@st.cache_resource(show_spinner=False)
def load_rtdetr():
    from ultralytics import RTDETR

    ckpt = resolve_weight_path("rtdetr_multiproduct_best.pt", "rtdetr_best.pt")
    if ckpt is None:
        raise FileNotFoundError("rtdetr_best.pt not found in `weights/` or repo root.")
    return RTDETR(str(ckpt))


@st.cache_resource(show_spinner=False)
def load_yoloworld_ft():
    from ultralytics import YOLOWorld

    ckpt = resolve_weight_path("yoloworld_multiproduct_best.pt", "yoloworld_best.pt")
    if ckpt is None:
        raise FileNotFoundError("yoloworld_best.pt not found in `weights/` or repo root.")
    return YOLOWorld(str(ckpt))


@st.cache_resource(show_spinner=False)
def load_yoloworld_zs():
    from ultralytics import YOLOWorld

    ckpt = resolve_weight_path("yolov8s-worldv2.pt")
    return YOLOWorld(str(ckpt) if ckpt is not None else "yolov8s-worldv2.pt")


@st.cache_resource(show_spinner=False)
def load_groundingdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model


@st.cache_resource(show_spinner=False)
def load_patchcore_plus() -> Tuple[PatchCorePlus, str]:
    ckpt = resolve_weight_path(
        "patchcore_plus_multiproduct.pt",
        "patchcore_l2l3.pt",
        "patchcore_memory_bank.npz",
    )
    if ckpt is None:
        raise FileNotFoundError(
            "No PatchCore++ checkpoint found (`patchcore_plus_multiproduct.pt`, `patchcore_l2l3.pt`, "
            "or `patchcore_memory_bank.npz`). Run `kaggle_01_dataset_builder_and_patchcore_ablation.ipynb` "
            "or `python patchcore_l2l3.py fit ...`."
        )
    return PatchCorePlus.from_checkpoint(ckpt), ckpt.name


@st.cache_resource(show_spinner=False)
def load_ovd_calibrator() -> ConfidenceCalibrator:
    cal_path = resolve_weight_path("ovd_calibrator.json")
    if cal_path is not None:
        return ConfidenceCalibrator.load(cal_path)
    return ConfidenceCalibrator(default_operating_raw_conf=0.002)


# ============================================================================
# Unified Inference Adapters
# ============================================================================

def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def open_uploaded_image(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    return Image.open(uploaded_file)


def _ultralytics_to_records(result, prompt_class_map: Optional[Dict[str, str]] = None, prompts: Optional[List[str]] = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return records
    names = result.names or {}
    for box in result.boxes:
        cid = int(box.cls[0].item())
        if prompts is not None and 0 <= cid < len(prompts):
            desc = prompts[cid]
            label = (prompt_class_map or {}).get(desc, desc)
        else:
            label = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            if label not in UNIFIED_CLASS_NAMES and 0 <= cid < len(UNIFIED_CLASS_NAMES):
                label = UNIFIED_CLASS_NAMES[cid]
        xmin, ymin, xmax, ymax = box.xyxy[0].detach().cpu().numpy().tolist()
        records.append(
            {
                "class": str(label),
                "confidence": float(box.conf[0].item()),
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "source": "detector",
            }
        )
    return records


def _predict_specialist_records(
    specialist_label: str,
    pil_image: Image.Image,
    conf: float,
    img_size: int,
    use_sahi: bool,
) -> List[Dict[str, object]]:
    def _tile_fn(tile_img: Image.Image) -> List[Dict[str, object]]:
        if specialist_label == MODEL_YOLOV11M:
            model = load_yolov11m()
            res = model.predict(tile_img, conf=conf, imgsz=img_size, verbose=False)[0]
            return _ultralytics_to_records(res)
        if specialist_label == MODEL_YOLOV9C:
            model = load_yolov9c()
            model.conf = conf
            arr = np.asarray(tile_img).copy()
            res = model(arr, size=img_size)
            table = res.pandas().xyxy[0]
            return [
                {
                    "class": str(row.get("name", row.get("class", "unknown"))),
                    "confidence": float(row.get("confidence", 0.0)),
                    "xmin": float(row.get("xmin", 0.0)),
                    "ymin": float(row.get("ymin", 0.0)),
                    "xmax": float(row.get("xmax", 0.0)),
                    "ymax": float(row.get("ymax", 0.0)),
                    "source": "detector",
                }
                for _, row in table.iterrows()
            ]
        model = load_rtdetr()
        res = model.predict(tile_img, conf=conf, imgsz=img_size, verbose=False)[0]
        return _ultralytics_to_records(res)

    if use_sahi:
        return run_sliced_inference(pil_image, _tile_fn, slice_size=img_size, overlap_ratio=0.25)
    return _tile_fn(pil_image)


def _predict_ovd_ft_records(
    pil_image: Image.Image,
    conf: float,
    img_size: int,
    apply_calibration: bool,
    use_sahi: bool,
) -> List[Dict[str, object]]:
    model = load_yoloworld_ft()
    raw_query_conf = 0.0005 if apply_calibration else conf
    calibrator = load_ovd_calibrator() if apply_calibration else None

    def _tile_fn(tile_img: Image.Image) -> List[Dict[str, object]]:
        res = model.predict(tile_img, conf=raw_query_conf, imgsz=img_size, verbose=False)[0]
        recs = _ultralytics_to_records(res)
        if calibrator is not None:
            recs = calibrator.calibrate_detections(recs, min_calibrated_conf=conf)
        return recs

    if use_sahi:
        return run_sliced_inference(pil_image, _tile_fn, slice_size=img_size, overlap_ratio=0.25)
    return _tile_fn(pil_image)


def _predict_zeroshot_records(
    model_label: str,
    pil_image: Image.Image,
    conf: float,
    img_size: int,
    prompts: List[str],
    class_map: Dict[str, str],
) -> List[Dict[str, object]]:
    if model_label == MODEL_YOLOWORLD_ZS:
        model = load_yoloworld_zs()
        model.set_classes(prompts)
        res = model.predict(pil_image, conf=conf, imgsz=img_size, verbose=False)[0]
        return _ultralytics_to_records(res, prompt_class_map=class_map, prompts=prompts)

    processor, model = load_groundingdino()
    text = ". ".join(prompts) + "."
    inputs = processor(images=pil_image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[pil_image.height, pil_image.width]])
    res = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=conf,
        text_threshold=conf,
        target_sizes=target_sizes,
    )[0]
    boxes = res["boxes"].detach().cpu().numpy()
    scores = res["scores"].detach().cpu().numpy()
    raw_labels = res["labels"]

    records: List[Dict[str, object]] = []
    for i in range(len(boxes)):
        raw_lbl = raw_labels[i] if isinstance(raw_labels[i], str) else str(raw_labels[i])
        short_name = raw_lbl
        for desc, cname in class_map.items():
            if raw_lbl.lower() in desc.lower() or desc.lower() in raw_lbl.lower():
                short_name = cname
                break
        records.append(
            {
                "class": short_name,
                "confidence": float(scores[i]),
                "xmin": float(boxes[i][0]),
                "ymin": float(boxes[i][1]),
                "xmax": float(boxes[i][2]),
                "ymax": float(boxes[i][3]),
                "source": "zeroshot_ovd",
            }
        )
    return records


def inspect_board(
    model_choice: str,
    image: Image.Image,
    image_name: str,
    confidence: float,
    image_size: int,
    use_sahi: bool = False,
    apply_ovd_calibration: bool = True,
    prompts: Optional[List[str]] = None,
    prompt_class_map: Optional[Dict[str, str]] = None,
    hybrid_specialist: str = MODEL_YOLOV11M,
    patchcore_threshold: Optional[float] = None,
    product_sku: str = "auto",
) -> Dict[str, object]:
    """Run unified single-board inspection across any selected paradigm."""
    t0 = time.perf_counter()
    base_rgb = np.asarray(image).copy()
    detections: List[Dict[str, object]] = []
    annotated_rgb = base_rgb
    detected_sku = "-"
    anomaly_score: Optional[float] = None

    if model_choice in {MODEL_YOLOV11M, MODEL_YOLOV9C, MODEL_RTDETR}:
        detections = _predict_specialist_records(model_choice, image, confidence, image_size, use_sahi)
        annotated_rgb = render_detections_rgb(base_rgb, detections)

    elif model_choice == MODEL_YOLOWORLD_FT:
        detections = _predict_ovd_ft_records(image, confidence, image_size, apply_ovd_calibration, use_sahi)
        annotated_rgb = render_detections_rgb(base_rgb, detections)

    elif model_choice in OPENVOCAB_PROMPT_MODELS:
        detections = _predict_zeroshot_records(
            model_choice,
            image,
            confidence,
            image_size,
            prompts or list(DEFAULT_OVD_PROMPTS_10CLS.values()),
            prompt_class_map or {v: k for k, v in DEFAULT_OVD_PROMPTS_10CLS.items()},
        )
        annotated_rgb = render_detections_rgb(base_rgb, detections)

    elif model_choice == MODEL_PATCHCORE_PLUS:
        pc_model, _ = load_patchcore_plus()
        pc_res = pc_model.predict(image, threshold=patchcore_threshold, product_sku=product_sku)
        detected_sku = f"{pc_res['product_sku']} ({pc_res['sku_confidence']:.2f})"
        anomaly_score = float(pc_res["score"])
        annotated_rgb = pc_res["overlay"] if pc_res["overlay"] is not None else base_rgb

        if pc_res["is_anomaly"]:
            rois = list(pc_res["rois"])
            if rois:
                for r in rois:
                    detections.append(
                        {
                            "class": str(r["class"]),
                            "confidence": float(r["score"]),
                            "xmin": float(r["xmin"]),
                            "ymin": float(r["ymin"]),
                            "xmax": float(r["xmax"]),
                            "ymax": float(r["ymax"]),
                            "area_px": int(r["area_px"]),
                            "source": "patchcore_plus_roi",
                        }
                    )
            else:
                w, h = image.size
                detections.append(
                    {
                        "class": "anomaly",
                        "confidence": anomaly_score,
                        "xmin": 0.0,
                        "ymin": 0.0,
                        "xmax": float(w),
                        "ymax": float(h),
                        "area_px": w * h,
                        "source": "patchcore_plus_global",
                    }
                )

    elif model_choice == MODEL_HYBRID_CASCADE:
        spec_dets = _predict_specialist_records(hybrid_specialist, image, confidence, image_size, use_sahi)
        pc_model, _ = load_patchcore_plus()
        pc_res = pc_model.predict(image, threshold=patchcore_threshold, product_sku=product_sku)
        detected_sku = f"{pc_res['product_sku']} ({pc_res['sku_confidence']:.2f})"
        anomaly_score = float(pc_res["score"])
        detections, _ = fuse_detector_and_patchcore(spec_dets, pc_res)
        heat_base = pc_res["overlay"] if pc_res["overlay"] is not None else base_rgb
        annotated_rgb = render_detections_rgb(heat_base, detections)

    elif model_choice == MODEL_HYBRID_OVD:
        spec_dets = _predict_specialist_records(hybrid_specialist, image, confidence, image_size, use_sahi)
        ovd_dets = _predict_ovd_ft_records(image, confidence, image_size, apply_ovd_calibration, use_sahi)
        merged = list(spec_dets)
        for cand in ovd_dets:
            c_box = (float(cand["xmin"]), float(cand["ymin"]), float(cand["xmax"]), float(cand["ymax"]))
            dup = any(
                compute_iou_and_ios(c_box, (float(s["xmin"]), float(s["ymin"]), float(s["xmax"]), float(s["ymax"])))[0] >= 0.45
                for s in spec_dets
            )
            if not dup:
                novel_cand = dict(cand)
                novel_cand["source"] = "ovd_complement"
                merged.append(novel_cand)
        detections = class_aware_nms_nmm(merged)
        annotated_rgb = render_detections_rgb(base_rgb, detections)

    inference_ms = (time.perf_counter() - t0) * 1000.0
    classes = sorted({str(d["class"]) for d in detections})
    max_conf = max((float(d["confidence"]) for d in detections), default=0.0)

    return {
        "image": image_name,
        "product_sku": detected_sku,
        "status": "Defects found" if detections else "Pass",
        "defects": len(detections),
        "classes": ", ".join(classes) if classes else "-",
        "max_confidence": round(max_conf, 4),
        "anomaly_score": round(anomaly_score, 4) if anomaly_score is not None else None,
        "inference_ms": round(inference_ms, 1),
        "detections": detections,
        "annotated_image": annotated_rgb,
    }


# ============================================================================
# Bulk Folder & Reporting Utilities
# ============================================================================

def find_local_images(folder_path: str, recursive: bool) -> List[Path]:
    root = Path(folder_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def results_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = [
        {
            "image": r["image"],
            "product_sku": r["product_sku"],
            "status": r["status"],
            "defects": r["defects"],
            "classes": r["classes"],
            "max_confidence": r["max_confidence"],
            "anomaly_score": r["anomaly_score"],
            "inference_ms": r["inference_ms"],
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def detections_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in results:
        for det in r["detections"]:  # type: ignore[union-attr]
            rows.append({"image": r["image"], "product_sku": r["product_sku"], **det})
    return pd.DataFrame(rows)


def load_json_if_exists(path: Path) -> Optional[Dict[str, object]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ============================================================================
# Streamlit Page & Sidebar Controls
# ============================================================================

st.set_page_config(page_title="Multi-Product PCB Inspection Console", page_icon="🔍", layout="wide")

st.sidebar.title("⚙️ Inspection Controls")
model_choice = st.sidebar.selectbox("🧠 Inspection Pipeline", ALL_MODELS, index=0)

hybrid_specialist = MODEL_YOLOV11M
if model_choice in {MODEL_HYBRID_CASCADE, MODEL_HYBRID_OVD}:
    hybrid_specialist = st.sidebar.selectbox(
        "Specialist Detector (Stage 1)",
        (MODEL_YOLOV11M, MODEL_YOLOV9C, MODEL_RTDETR),
        index=0,
    )

apply_ovd_calibration = True
if model_choice in {MODEL_YOLOWORLD_FT, MODEL_HYBRID_OVD}:
    apply_ovd_calibration = st.sidebar.checkbox(
        "Apply Post-Hoc Platt Calibration (OVD)",
        value=True,
        help="Rescales Fine-Tuned YOLO-World's low raw scores (~0.002) onto calibrated [0, 1] probabilities.",
    )

min_slider = 0.001 if (model_choice == MODEL_YOLOWORLD_FT and not apply_ovd_calibration) else 0.05
default_conf = 0.002 if (model_choice == MODEL_YOLOWORLD_FT and not apply_ovd_calibration) else 0.45
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=float(min_slider),
    max_value=0.95,
    value=float(default_conf),
    step=0.001 if min_slider < 0.01 else 0.05,
)

image_size = st.sidebar.select_slider(
    "Inference tile / image size",
    options=[320, 512, 640, 960, 1280],
    value=640,
)

use_sahi = False
if model_choice != MODEL_PATCHCORE_PLUS:
    use_sahi = st.sidebar.checkbox(
        "Enable SAHI Full-Board Slicing",
        value=False,
        help="Slices high-resolution full PCB boards (e.g. 3034x1586) into overlapping windows so 5px micro-defects are preserved.",
    )

patchcore_threshold: Optional[float] = None
product_sku = "auto"
if model_choice in {MODEL_PATCHCORE_PLUS, MODEL_HYBRID_CASCADE}:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧩 PatchCore++ Multi-Product Settings**")
    product_sku = st.sidebar.text_input(
        "Product SKU (`auto` uses DINOv2 Router)",
        value="auto",
        help="Leave as `auto` to automatically detect board SKU (e.g. visa_pcb1..4, pku_pcb_01..12, deeppcb_*).",
    ).strip() or "auto"
    thr_override = st.sidebar.number_input(
        "Anomaly threshold override (0 = calibrated)",
        min_value=0.0,
        value=0.0,
        step=0.01,
    )
    patchcore_threshold = thr_override if thr_override > 0 else None

text_prompts = list(DEFAULT_OVD_PROMPTS_10CLS.values())
prompt_class_map = {desc: name for name, desc in DEFAULT_OVD_PROMPTS_10CLS.items()}

if model_choice in OPENVOCAB_PROMPT_MODELS:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📝 Open-Vocabulary Defect Prompts (10 Classes)**")
    default_prompt_blob = "\n".join(f"{name}: {desc}" for name, desc in DEFAULT_OVD_PROMPTS_10CLS.items())
    raw_prompts = st.sidebar.text_area("class_name: visual description", value=default_prompt_blob, height=220)
    text_prompts = []
    prompt_class_map = {}
    for line in raw_prompts.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            cname, cdesc = [x.strip() for x in line.split(":", 1)]
        else:
            cname, cdesc = line, line
        text_prompts.append(cdesc)
        prompt_class_map[cdesc] = cname

max_previews = st.sidebar.slider("Bulk defect preview limit", min_value=2, max_value=24, value=8, step=2)

# ============================================================================
# Main Console Tabs
# ============================================================================

st.title("🔍 Multi-Product & Multi-Paradigm PCB Defect Inspection Console")
st.caption(
    "Unified 10-Class Bare-PCB + Assembled-PCBA Detection · PatchCore++ (DINOv2 Hybrid + Product Router) · SAHI Full-Board Slicing"
)

single_tab, bulk_tab, bench_tab = st.tabs(
    [
        "🖼️ Single Board Inspection",
        "📂 Bulk Folder / Batch Inspection",
        "📊 Model Benchmark & Ablation Dashboard",
    ]
)

with single_tab:
    uploaded_file = st.file_uploader(
        "Upload PCB or PCBA image",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
    )

    if uploaded_file is not None:
        pil_img = normalize_image(open_uploaded_image(uploaded_file))
        try:
            with st.spinner(f"Running {model_choice}..."):
                result = inspect_board(
                    model_choice=model_choice,
                    image=pil_img,
                    image_name=uploaded_file.name,
                    confidence=confidence_threshold,
                    image_size=image_size,
                    use_sahi=use_sahi,
                    apply_ovd_calibration=apply_ovd_calibration,
                    prompts=text_prompts,
                    prompt_class_map=prompt_class_map,
                    hybrid_specialist=hybrid_specialist,
                    patchcore_threshold=patchcore_threshold,
                    product_sku=product_sku,
                )

            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Verdict", str(result["status"]))
            kpi_cols[1].metric("Detections / ROIs", int(result["defects"]))
            kpi_cols[2].metric("Product SKU", str(result["product_sku"]))
            kpi_cols[3].metric(
                "Peak Score",
                f"{result['anomaly_score']:.3f}" if result["anomaly_score"] is not None else f"{result['max_confidence']:.2f}",
            )
            kpi_cols[4].metric("Latency", f"{result['inference_ms']:.1f} ms")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Original ({pil_img.width}×{pil_img.height})")
                st.image(pil_img, use_container_width=True)
            with col2:
                st.subheader(f"Inspection Result — {model_choice}")
                st.image(result["annotated_image"], channels="RGB", use_container_width=True)

            det_df = detections_dataframe([result])
            if not det_df.empty:
                st.subheader("Detected Defect Regions")
                st.dataframe(det_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")

with bulk_tab:
    source_mode = st.radio("Batch input source", ("Upload multiple images", "Local folder path"), horizontal=True)
    items: List[Tuple[str, Callable[[], Image.Image]]] = []

    if source_mode == "Upload multiple images":
        uploaded_files = st.file_uploader(
            "Select multiple PCB images",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        items = [(f.name, lambda f=f: open_uploaded_image(f)) for f in uploaded_files]
    else:
        folder_path = st.text_input("Local folder path", placeholder=r"C:\path\to\pcb-batch")
        recursive = st.checkbox("Scan subfolders recursively", value=True)
        if folder_path:
            try:
                paths = find_local_images(folder_path, recursive)
                st.caption(f"Discovered {len(paths)} supported PCB image(s).")
                items = [(str(p), lambda p=p: Image.open(p)) for p in paths]
            except Exception as exc:
                st.error(str(exc))

    if st.button("Run Bulk Inspection", type="primary", disabled=len(items) == 0):
        progress = st.progress(0)
        status_box = st.empty()
        batch_results: List[Dict[str, object]] = []
        batch_errors: List[Dict[str, str]] = []

        for idx, (img_name, loader_fn) in enumerate(items, start=1):
            status_box.write(f"Inspecting {idx} / {len(items)}: `{img_name}`")
            try:
                img_obj = normalize_image(loader_fn())
                batch_results.append(
                    inspect_board(
                        model_choice=model_choice,
                        image=img_obj,
                        image_name=img_name,
                        confidence=confidence_threshold,
                        image_size=image_size,
                        use_sahi=use_sahi,
                        apply_ovd_calibration=apply_ovd_calibration,
                        prompts=text_prompts,
                        prompt_class_map=prompt_class_map,
                        hybrid_specialist=hybrid_specialist,
                        patchcore_threshold=patchcore_threshold,
                        product_sku=product_sku,
                    )
                )
            except Exception as exc:
                batch_errors.append({"image": img_name, "error": str(exc)})
            progress.progress(idx / len(items))

        status_box.empty()

        if batch_results:
            n_total = len(batch_results)
            n_failed = sum(1 for r in batch_results if int(r["defects"]) > 0)
            n_defects = sum(int(r["defects"]) for r in batch_results)
            avg_ms = sum(float(r["inference_ms"]) for r in batch_results) / max(1, n_total)

            mcols = st.columns(4)
            mcols[0].metric("Boards Inspected", n_total)
            mcols[1].metric("Boards Flagged", n_failed)
            mcols[2].metric("Total Defect ROIs", n_defects)
            mcols[3].metric("Mean Latency", f"{avg_ms:.1f} ms")

            sum_df = results_dataframe(batch_results)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Board Summary CSV",
                    sum_df.to_csv(index=False).encode("utf-8"),
                    file_name="pcb_inspection_summary.csv",
                    mime="text/csv",
                )
            det_df = detections_dataframe(batch_results)
            if not det_df.empty:
                with dl_col2:
                    st.download_button(
                        "⬇️ Download Per-Defect Details CSV",
                        det_df.to_csv(index=False).encode("utf-8"),
                        file_name="pcb_detection_details.csv",
                        mime="text/csv",
                    )

            flagged = [r for r in batch_results if int(r["defects"]) > 0][:max_previews]
            if flagged:
                st.subheader("Flagged Board Previews")
                pcols = st.columns(2)
                for p_idx, r in enumerate(flagged):
                    with pcols[p_idx % 2]:
                        st.image(
                            r["annotated_image"],
                            caption=f"{r['image']} | SKU: {r['product_sku']} | {r['defects']} finding(s)",
                            channels="RGB",
                            use_container_width=True,
                        )

        if batch_errors:
            st.warning(f"{len(batch_errors)} image(s) encountered errors.")
            st.dataframe(pd.DataFrame(batch_errors), use_container_width=True, hide_index=True)

with bench_tab:
    pc_ablation = load_json_if_exists(REPO_ROOT / "results" / "patchcore_ablation_metrics.json")
    det_metrics = load_json_if_exists(REPO_ROOT / "results" / "multiproduct_detector_metrics.json")
    ovd_metrics = load_json_if_exists(REPO_ROOT / "results" / "ovd_calibration_metrics.json")

    st.subheader("🏆 Multi-Product & Multi-Class Benchmark Summary")
    if pc_ablation and det_metrics:
        best_pc = pc_ablation.get("patchcore_plus_dinov2_hybrid_518", pc_ablation.get("patchcore_plus_wrn50_v2_512", {}))
        y11_test = det_metrics.get("yolov11m_10cls", {}).get("test", det_metrics.get("yolov11m_10cls", {}).get("val", {}))
        ovd_cal = (ovd_metrics or {}).get("calibrated_platt_threshold_0_50", {})
        top_cols = st.columns(4)
        top_cols[0].metric(
            "PatchCore++ Image-AUROC",
            f"{float(best_pc.get('image_auroc', 0.0)):.4f}",
            delta=f"+{float(best_pc.get('image_auroc', 0.0)) - 0.5624:.4f} vs v1",
        )
        top_cols[1].metric(
            "PatchCore++ Pixel-AUROC (Med)",
            f"{float(best_pc.get('pixel_auroc_median', 0.0)):.4f}",
            delta=f"+{float(best_pc.get('pixel_auroc_median', 0.0)) - 0.7127:.4f} vs v1",
        )
        top_cols[2].metric(
            "YOLOv11-m 10-Class mAP@0.50",
            f"{float(y11_test.get('mAP50', 0.0)):.4f}",
            delta="10 Unified Classes",
        )
        top_cols[3].metric(
            "Calibrated OVD Recall @ 0.50",
            f"{float(ovd_cal.get('recall', 0.0)):.4f}",
            delta=f"Precision {float(ovd_cal.get('precision', 0.0)):.4f}",
        )

    if pc_ablation:
        st.markdown("### 1. Multi-Product PatchCore++ Ablation (`PKU-PCB` + `DeepPCB` + `VisA PCBA`)")
        pc_rows = []
        for key, label in (
            ("legacy_v1_audit_baseline", "Legacy PatchCore v1 (Single-Board Audit)"),
            ("baseline_wrn50_l2l3_256_pooled", "Baseline WRN50 (L2+L3 @ 256px, Pooled)"),
            ("patchcore_plus_wrn50_v2_512", "PatchCore++ WRN50-v2 (L1+L2 @ 512px + SKU Router)"),
            ("patchcore_plus_dinov2_hybrid_518", "PatchCore++ DINOv2-Hybrid (ViT-B/14-reg + WRN50 @ 518px)"),
        ):
            item = pc_ablation.get(key)
            if isinstance(item, dict):
                doms = item.get("domains", {})
                pc_rows.append(
                    {
                        "Configuration": label,
                        "Image-AUROC": item.get("image_auroc"),
                        "F1-Max": item.get("f1_max"),
                        "Pixel-AUROC (Mean)": item.get("pixel_auroc_mean"),
                        "Pixel-AUROC (Median)": item.get("pixel_auroc_median"),
                        "PKU-PCB Pixel-AUROC": doms.get("pku_pcb", {}).get("pixel_auroc_median"),
                        "DeepPCB Pixel-AUROC": doms.get("deeppcb", {}).get("pixel_auroc_median"),
                        "VisA PCBA Image-AUROC": doms.get("visa_pcb", {}).get("image_auroc"),
                        "VisA PCBA Pixel-AUROC": doms.get("visa_pcb", {}).get("pixel_auroc_median"),
                        "Latency (ms)": item.get("avg_latency_ms"),
                    }
                )
        st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

    if det_metrics:
        st.markdown("### 2. 10-Class Multi-Product Specialist Per-Class AP@0.50")
        y11 = det_metrics.get("yolov11m_10cls", {})
        rtd = det_metrics.get("rtdetr_l_10cls", {})
        y_val_pc = y11.get("val", {}).get("per_class_ap50", {})
        y_tst_pc = y11.get("test", {}).get("per_class_ap50", {})
        r_tst_pc = rtd.get("test", rtd.get("val", {})).get("per_class_ap50", {})
        cls_rows = []
        for idx, cname in enumerate(UNIFIED_CLASS_NAMES):
            cls_rows.append(
                {
                    "Class ID": idx,
                    "Unified Defect Class": cname,
                    "Domain": "Bare PCB (0–5)" if idx < 6 else "Assembled PCBA / Surface (6–9)",
                    "YOLOv11-m Val AP@0.50": y_val_pc.get(cname),
                    "YOLOv11-m Blind Test AP@0.50": y_tst_pc.get(cname),
                    "RT-DETR-L AP@0.50": r_tst_pc.get(cname),
                }
            )
        st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

    fig_dir = REPO_ROOT / "results" / "figures"
    if fig_dir.exists():
        st.markdown("### 3. Visual Inspection & Benchmark Figures")
        fcols = st.columns(2)
        for i, (fname, cap) in enumerate(
            [
                ("fig1_patchcore_v1_vs_v2_comparison.png", "Fig 1: Legacy PatchCore v1 vs Upgraded PatchCore++ v2"),
                ("fig2_multiproduct_4domain_gallery.png", "Fig 2: 4-Dataset Multi-Product Inspection Gallery"),
                ("fig3_benchmark_charts.png", "Fig 3: Quantitative Ablation & 10-Class AP@0.50 Charts"),
                ("fig4_sahi_vs_naive_fullboard.png", "Fig 4: Full-Board Naive Downsampling vs SAHI Sliced Inference"),
            ]
        ):
            fpath = fig_dir / fname
            if fpath.exists():
                with fcols[i % 2]:
                    st.image(str(fpath), caption=cap, use_container_width=True)
