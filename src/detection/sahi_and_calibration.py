"""
High-Resolution SAHI Sliced Inference, Post-Hoc OVD Confidence Calibration,
and Two-Stage Detector + PatchCore++ Open-Set Fusion.

Components:
  1. `run_sliced_inference`: Tiles full-resolution PCB boards (e.g. 3034x1586) into overlapping
     640x640 crops + global context pass, merging predictions with Class-Aware NMS + NMM
     (Containment / Non-Max Merging) so 4-10px micro-defects are never lost to downsampling.
  2. `ConfidenceCalibrator`: Fixes structural low-confidence collapse in fine-tuned OVD heads
     (e.g. YOLO-World operating at conf=0.001) via per-class log-odds Platt scaling
     `p_cal = sigmoid(a_c * logit(p_raw) + b_c)`.
  3. `fuse_detector_and_patchcore`: Combines closed-set multi-class detector boxes (YOLOv11/YOLOv9/RT-DETR)
     with PatchCore++ connected-component anomaly ROIs to attach pixel defect areas (`px^2`)
     and surface novel/unseen PCB defects missed by closed-set models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


def compute_iou_and_ios(box_a: Sequence[float], box_b: Sequence[float]) -> Tuple[float, float]:
    """Return `(IoU, IoSmallest)` for two `[xmin, ymin, xmax, ymax]` boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    iou = inter / union if union > 1e-8 else 0.0
    smallest = min(area_a, area_b)
    ios = inter / smallest if smallest > 1e-8 else 0.0
    return iou, ios


def class_aware_nms_nmm(
    detections: List[Dict[str, object]],
    iou_threshold: float = 0.45,
    ios_threshold: float = 0.80,
) -> List[Dict[str, object]]:
    """
    Merge duplicate detections across overlapping SAHI tiles using both IoU and
    IoS (Intersection-over-Smallest, preventing truncated tile-border sub-boxes).
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
    kept: List[Dict[str, object]] = []

    for cand in ordered:
        c_box = (float(cand["xmin"]), float(cand["ymin"]), float(cand["xmax"]), float(cand["ymax"]))
        c_cls = str(cand["class"])
        suppressed = False
        for prev in kept:
            if str(prev["class"]) != c_cls:
                continue
            p_box = (float(prev["xmin"]), float(prev["ymin"]), float(prev["xmax"]), float(prev["ymax"]))
            iou, ios = compute_iou_and_ios(c_box, p_box)
            if iou >= iou_threshold or ios >= ios_threshold:
                # Expand kept box slightly when tile-seam partial boxes are merged
                if iou >= iou_threshold:
                    w_prev = float(prev["confidence"])
                    w_cand = float(cand["confidence"])
                    w_sum = w_prev + w_cand + 1e-8
                    prev["xmin"] = min(float(prev["xmin"]), (w_prev * p_box[0] + w_cand * c_box[0]) / w_sum)
                    prev["ymin"] = min(float(prev["ymin"]), (w_prev * p_box[1] + w_cand * c_box[1]) / w_sum)
                    prev["xmax"] = max(float(prev["xmax"]), (w_prev * p_box[2] + w_cand * c_box[2]) / w_sum)
                    prev["ymax"] = max(float(prev["ymax"]), (w_prev * p_box[3] + w_cand * c_box[3]) / w_sum)
                suppressed = True
                break
        if not suppressed:
            kept.append(dict(cand))
    return kept


def generate_sahi_slices(
    width: int,
    height: int,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """Generate `(x1, y1, x2, y2)` overlapping windows covering `(width, height)`."""
    if width <= int(slice_size * 1.15) and height <= int(slice_size * 1.15):
        return [(0, 0, width, height)]

    stride = max(64, int(round(slice_size * (1.0 - overlap_ratio))))
    xs = list(range(0, max(1, width - slice_size + 1), stride))
    ys = list(range(0, max(1, height - slice_size + 1), stride))
    if width > slice_size and xs[-1] != width - slice_size:
        xs.append(width - slice_size)
    if height > slice_size and ys[-1] != height - slice_size:
        ys.append(height - slice_size)

    slices: List[Tuple[int, int, int, int]] = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(width, x1 + slice_size)
            y2 = min(height, y1 + slice_size)
            slices.append((x1, y1, x2, y2))
    return slices


def run_sliced_inference(
    pil_image: Image.Image,
    predict_tile_fn: Callable[[Image.Image], List[Dict[str, object]]],
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    include_full_image: bool = True,
    iou_threshold: float = 0.45,
) -> List[Dict[str, object]]:
    """
    SAHI-style sliced high-resolution inference wrapper compatible with any detector callback
    `predict_tile_fn(tile_pil) -> [{"class": ..., "confidence": ..., "xmin": ..., ...}]`.
    """
    w, h = pil_image.size
    slices = generate_sahi_slices(w, h, slice_size=slice_size, overlap_ratio=overlap_ratio)
    if len(slices) == 1:
        return predict_tile_fn(pil_image)

    aggregated: List[Dict[str, object]] = []
    for x1, y1, x2, y2 in slices:
        crop = pil_image.crop((x1, y1, x2, y2))
        tile_dets = predict_tile_fn(crop)
        for det in tile_dets:
            shifted = dict(det)
            shifted["xmin"] = float(det["xmin"]) + x1
            shifted["ymin"] = float(det["ymin"]) + y1
            shifted["xmax"] = float(det["xmax"]) + x1
            shifted["ymax"] = float(det["ymax"]) + y1
            aggregated.append(shifted)

    if include_full_image:
        aggregated.extend(predict_tile_fn(pil_image))

    return class_aware_nms_nmm(aggregated, iou_threshold=iou_threshold)


# ============================================================================
# Post-Hoc Log-Odds Platt Confidence Calibrator for Fine-Tuned OVD (YOLO-World)
# ============================================================================

def _logit(p: float, eps: float = 1e-6) -> float:
    p_c = min(1.0 - eps, max(eps, float(p)))
    return math.log(p_c / (1.0 - p_c))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


@dataclass
class ConfidenceCalibrator:
    """
    Per-class log-odds Platt scaling calibrator:
        `p_calibrated = sigmoid(scale_c * logit(p_raw) + shift_c)`

    Default analytical anchor maps Fine-Tuned YOLO-World's `p_raw = 0.002` operating point
    (`logit ~ -6.21`) cleanly to `p_cal = 0.50`, while `fit_from_validation_matches()` fits exact
    per-class parameters on validation TP/FP detections.
    """

    default_operating_raw_conf: float = 0.002
    params_by_class: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def _default_params(self) -> Tuple[float, float]:
        # Anchor: raw=operating_point -> cal=0.50; raw=0.05 -> cal~0.92
        x0 = _logit(self.default_operating_raw_conf)
        x1 = _logit(max(0.05, self.default_operating_raw_conf * 20.0))
        scale = 2.4 / max(x1 - x0, 1e-4)
        shift = -scale * x0
        return (scale, shift)

    def calibrate_score(self, raw_conf: float, class_name: str = "") -> float:
        scale, shift = self.params_by_class.get(class_name, self._default_params())
        return round(_sigmoid(scale * _logit(raw_conf) + shift), 4)

    def calibrate_detections(
        self,
        detections: Iterable[Dict[str, object]],
        min_calibrated_conf: float = 0.50,
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for det in detections:
            raw_c = float(det["confidence"])
            cls_name = str(det["class"])
            cal_c = self.calibrate_score(raw_c, cls_name)
            if cal_c >= min_calibrated_conf:
                updated = dict(det)
                updated["raw_confidence"] = round(raw_c, 6)
                updated["confidence"] = cal_c
                out.append(updated)
        return out

    def fit_from_validation_matches(
        self,
        observations_by_class: Dict[str, Sequence[Tuple[float, int]]],
        lr: float = 0.05,
        steps: int = 400,
        l2_reg: float = 1e-3,
    ) -> "ConfidenceCalibrator":
        """
        Fit `(scale_c, shift_c)` per class from `[(raw_conf, is_true_positive_0_or_1), ...]`
        via L2-regularized Newton/gradient steps on binary cross-entropy.
        """
        init_a, init_b = self._default_params()
        for cls_name, pairs in observations_by_class.items():
            if len(pairs) < 8:
                continue
            xs = np.asarray([_logit(p) for p, _ in pairs], dtype=np.float64)
            ys = np.asarray([float(y) for _, y in pairs], dtype=np.float64)
            a, b = float(init_a), float(init_b)
            for _ in range(steps):
                z = np.clip(a * xs + b, -40.0, 40.0)
                preds = 1.0 / (1.0 + np.exp(-z))
                err = preds - ys
                grad_a = float(np.mean(err * xs) + l2_reg * (a - 1.0))
                grad_b = float(np.mean(err))
                a = max(0.15, a - lr * grad_a)
                b = b - lr * grad_b
            self.params_by_class[cls_name] = (round(a, 6), round(b, 6))
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            "default_operating_raw_conf": self.default_operating_raw_conf,
            "params_by_class": {k: list(v) for k, v in self.params_by_class.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            default_operating_raw_conf=float(payload.get("default_operating_raw_conf", 0.002)),
            params_by_class={k: (float(v[0]), float(v[1])) for k, v in payload.get("params_by_class", {}).items()},
        )


# ============================================================================
# Two-Stage Closed-Set Detector + Open-Set PatchCore++ Hybrid Fusion
# ============================================================================

def fuse_detector_and_patchcore(
    detector_records: Sequence[Dict[str, object]],
    patchcore_result: Dict[str, object],
    match_iou_threshold: float = 0.15,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Fuse Stage-1 supervised/OVD bounding boxes with Stage-2 PatchCore++ anomaly ROIs:
      - Detector boxes overlapping an anomaly ROI (`IoU >= 0.15` or center inside) receive
        pixel defect area (`area_px`) and `source="specialist+patchcore_verified"`.
      - Unmatched PatchCore++ ROIs (when `patchcore_result["is_anomaly"]` is True) are
        appended as open-set findings (`class` from Prototype Bank or `novel_unseen_defect`),
        ensuring rare/unseen PCB defects are never missed!
    """
    fused: List[Dict[str, object]] = [dict(d, source=d.get("source", "detector")) for d in detector_records]
    rois: List[Dict[str, object]] = list(patchcore_result.get("rois", []))  # type: ignore
    matched_roi_indices: set[int] = set()

    for det in fused:
        d_box = (float(det["xmin"]), float(det["ymin"]), float(det["xmax"]), float(det["ymax"]))
        best_idx = -1
        best_overlap = 0.0
        for r_idx, roi in enumerate(rois):
            r_box = (float(roi["xmin"]), float(roi["ymin"]), float(roi["xmax"]), float(roi["ymax"]))
            iou, ios = compute_iou_and_ios(d_box, r_box)
            score = max(iou, 0.5 * ios)
            if score > best_overlap:
                best_overlap = score
                best_idx = r_idx
        if best_idx >= 0 and best_overlap >= match_iou_threshold:
            matched_roi_indices.add(best_idx)
            det["area_px"] = int(rois[best_idx].get("area_px", 0))
            det["anomaly_score"] = float(rois[best_idx].get("score", 0.0))
            det["source"] = "verified_by_patchcore"

    novel_added = 0
    if patchcore_result.get("is_anomaly", False):
        for r_idx, roi in enumerate(rois):
            if r_idx in matched_roi_indices:
                continue
            fused.append(
                {
                    "class": str(roi.get("class", "novel_unseen_defect")),
                    "confidence": float(roi.get("score", 0.0)),
                    "xmin": float(roi["xmin"]),
                    "ymin": float(roi["ymin"]),
                    "xmax": float(roi["xmax"]),
                    "ymax": float(roi["ymax"]),
                    "area_px": int(roi.get("area_px", 0)),
                    "source": "patchcore_open_set",
                }
            )
            novel_added += 1

    meta = {
        "product_sku": patchcore_result.get("product_sku", "unknown"),
        "sku_confidence": patchcore_result.get("sku_confidence", 0.0),
        "anomaly_score": patchcore_result.get("score", 0.0),
        "is_anomaly": patchcore_result.get("is_anomaly", False),
        "detector_count": len(detector_records),
        "verified_count": len(matched_roi_indices),
        "open_set_novel_count": novel_added,
        "total_unique": len(fused),
    }
    return fused, meta


def render_detections_rgb(
    base_rgb: np.ndarray,
    detections: Sequence[Dict[str, object]],
) -> np.ndarray:
    """Draw color-coded bounding boxes onto an RGB image array."""
    canvas = base_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        cls_name = str(det["class"])
        conf = float(det["confidence"])
        src = str(det.get("source", "detector"))
        area_px = det.get("area_px")

        if src == "patchcore_open_set" or "novel" in cls_name.lower():
            color = (255, 65, 54)   # Red: novel / open-set anomaly ROI
        elif src == "verified_by_patchcore":
            color = (0, 230, 118)   # Emerald: dual-verified known defect
        else:
            color = (56, 189, 248)  # Sky blue: standard detector prediction

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        area_tag = f" | {area_px}px2" if area_px else ""
        label = f"{cls_name} {conf:.2f}{area_tag}"
        ty = max(16, y1 - 6)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    return canvas
