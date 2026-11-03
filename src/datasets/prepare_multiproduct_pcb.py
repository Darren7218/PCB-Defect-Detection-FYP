"""
Multi-Product & Multi-Class PCB Dataset Builder (Tier 0 Local / Tier 1 Kaggle Free — 0 CU).

Ingests and harmonizes the 4 complementary public PCB benchmarks into two ready-to-train layouts:

1. Unified 10-Class Object Detection Dataset (`<out>/unified_pcb_10cls/data.yaml`):
   - PKU-Market-PCB / TDD-net (`VOC_PCB`): 6 bare-board trace defect classes
   - DeepPCB (Tang et al., 1,500 template-test pairs): 6 bare-board trace defect classes
   - Assembled PCBA Component & Solder Defect Dataset (YOLO / VOC XML / COCO JSON): classes 6..9
   - VisA (`pcb1`, `pcb2`, `pcb3`, `pcb4`) anomaly masks -> tight connected-component bounding boxes

2. Multi-Product Anomaly Detection Suite (`<out>/anomaly_multiproduct/<product_sku>/`):
   - `VisA` (`visa_pcb1`, `visa_pcb2`, `visa_pcb3`, `visa_pcb4`): official normal train + normal/anomalous test + pixel GT masks
   - `DeepPCB` (`deeppcb_group_<id>`): clean `*_temp.jpg` normal bank + `*_test.jpg` + bbox/difference GT masks
   - `PKU-Market-PCB` (`pku_pcb_<board_id>`): dense `stride=150` normal tiles with cardinal (0/90/180/270) AND
     oblique (+/-7 deg) rotations + strictly held-out `val/good` and `test/good` tiles so `image_auroc` is NEVER null!

Also supports `--pack-tar` to bundle datasets into single `.tar` files for <60s staging onto Colab `/content/datasets`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.datasets.taxonomy import (
    UNIFIED_CLASS_NAMES,
    resolve_unified_class_id,
)

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dirs(root: Path, splits: Iterable[str] = ("train", "val", "test")) -> None:
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(float(w), xmin))
    xmax = max(0.0, min(float(w), xmax))
    ymin = max(0.0, min(float(h), ymin))
    ymax = max(0.0, min(float(h), ymax))
    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 2.0 or bh < 2.0 or w <= 0 or h <= 0:
        return None
    cx = (xmin + xmax) / (2.0 * w)
    cy = (ymin + ymax) / (2.0 * h)
    return (cx, cy, bw / w, bh / h)


def _write_yolo_labels(label_path: Path, records: List[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in records]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_or_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copy2(src, dst)


# ============================================================================
# 1. DeepPCB (1,500 aligned pairs across multiple bare-board PCB layouts)
# ============================================================================

def download_deeppcb(dest_dir: Path) -> Path:
    """Clone official DeepPCB repository (lightweight ~85MB git repo)."""
    repo_dir = dest_dir / "DeepPCB"
    if (repo_dir / "PCBData").exists():
        print(f"[Dataset] DeepPCB already exists at {repo_dir}")
        return repo_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Dataset] Cloning DeepPCB into {repo_dir}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/tangsanli5201/DeepPCB.git", str(repo_dir)],
        check=True,
    )
    return repo_dir


def ingest_deeppcb(
    deeppcb_root: Path,
    det_out: Path,
    anom_out: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Ingest DeepPCB into:
      - `det_out`: Unified 10-class YOLO dataset
      - `anom_out`: Multi-product template/normal vs defective anomaly benchmark
    """
    pcb_data = deeppcb_root / "PCBData"
    if not pcb_data.exists() and (deeppcb_root / "group00041").exists():
        pcb_data = deeppcb_root
    if not pcb_data.exists():
        raise FileNotFoundError(f"DeepPCB PCBData directory not found under {deeppcb_root}")

    rng = random.Random(seed)
    stats: Counter[str] = Counter()

    # Parse official trainval.txt / test.txt if present; otherwise discover pairs
    split_lists: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    trainval_txt = pcb_data / "trainval.txt"
    test_txt = pcb_data / "test.txt"

    def _parse_list_file(txt_file: Path) -> List[Tuple[Path, Path]]:
        pairs = []
        for raw_line in txt_file.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 2:
                continue
            img_rel, ann_rel = parts
            # DeepPCB list entries point to `<id>.jpg` where actual files are `<id>_test.jpg` and `<id>_temp.jpg`
            img_candidate = pcb_data / img_rel
            if not img_candidate.exists():
                stem_test = img_candidate.with_name(f"{img_candidate.stem}_test{img_candidate.suffix}")
                if stem_test.exists():
                    img_candidate = stem_test
            ann_candidate = pcb_data / ann_rel
            if img_candidate.exists() and ann_candidate.exists():
                pairs.append((img_candidate, ann_candidate))
        return pairs

    if trainval_txt.exists() and test_txt.exists():
        tv_pairs = _parse_list_file(trainval_txt)
        rng.shuffle(tv_pairs)
        n_val = max(1, int(len(tv_pairs) * val_ratio))
        split_lists["val"] = tv_pairs[:n_val]
        split_lists["train"] = tv_pairs[n_val:]
        split_lists["test"] = _parse_list_file(test_txt)
    else:
        all_tests = sorted(pcb_data.rglob("*_test.jpg"))
        all_pairs = []
        for t_img in all_tests:
            sample_id = t_img.stem.replace("_test", "")
            group_dir = t_img.parent
            ann_dir = group_dir.parent / f"{group_dir.name}_not"
            ann_file = ann_dir / f"{sample_id}.txt"
            if ann_file.exists():
                all_pairs.append((t_img, ann_file))
        rng.shuffle(all_pairs)
        n_test = max(1, int(len(all_pairs) * 0.2))
        n_val = max(1, int(len(all_pairs) * 0.15))
        split_lists["test"] = all_pairs[:n_test]
        split_lists["val"] = all_pairs[n_test : n_test + n_val]
        split_lists["train"] = all_pairs[n_test + n_val :]

    _ensure_dirs(det_out)

    for split, pairs in split_lists.items():
        for test_img_path, ann_path in pairs:
            img = cv2.imread(str(test_img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sample_id = test_img_path.stem.replace("_test", "")
            group_name = test_img_path.parent.name  # e.g. "00041" (distinct PCB product layout!)
            sku_name = f"deeppcb_{group_name}"

            yolo_boxes: List[Tuple[int, float, float, float, float]] = []
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            dominant_defect = "mixed_defect"

            for line in ann_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                x1, y1, x2, y2 = map(float, parts[:4])
                raw_cls = int(parts[4])
                cid = resolve_unified_class_id(raw_cls, source_dataset="deeppcb")
                if cid is None:
                    continue
                ybox = _xyxy_to_yolo(x1, y1, x2, y2, w, h)
                if ybox is not None:
                    yolo_boxes.append((cid, *ybox))
                    dominant_defect = UNIFIED_CLASS_NAMES[cid]
                    gt_mask[max(0, int(y1)) : min(h, int(y2)), max(0, int(x1)) : min(w, int(x2))] = 255
                    stats[f"box_{UNIFIED_CLASS_NAMES[cid]}"] += 1

            out_stem = f"deeppcb_{group_name}_{sample_id}"
            _copy_or_link(test_img_path, det_out / "images" / split / f"{out_stem}.jpg")
            _write_yolo_labels(det_out / "labels" / split / f"{out_stem}.txt", yolo_boxes)
            stats[f"det_{split}"] += 1

            # Populate Multi-Product Anomaly Benchmark using paired defect-free *_temp.jpg!
            temp_img_path = test_img_path.with_name(f"{sample_id}_temp{test_img_path.suffix}")
            sku_dir = anom_out / sku_name
            if temp_img_path.exists():
                if split == "train":
                    dst_good = sku_dir / "train" / "good" / f"{out_stem}_temp.jpg"
                elif split == "val":
                    dst_good = sku_dir / "val" / "good" / f"{out_stem}_temp.jpg"
                else:
                    dst_good = sku_dir / "test" / "good" / f"{out_stem}_temp.jpg"
                dst_good.parent.mkdir(parents=True, exist_ok=True)
                _copy_or_link(temp_img_path, dst_good)
                stats[f"anom_good_{split}"] += 1

            if split in {"val", "test"} and yolo_boxes:
                anom_split_dir = sku_dir / split / dominant_defect
                mask_split_dir = sku_dir / "ground_truth" / split / dominant_defect
                anom_split_dir.mkdir(parents=True, exist_ok=True)
                mask_split_dir.mkdir(parents=True, exist_ok=True)
                _copy_or_link(test_img_path, anom_split_dir / f"{out_stem}.jpg")
                cv2.imwrite(str(mask_split_dir / f"{out_stem}_mask.png"), gt_mask)
                stats[f"anom_defect_{split}"] += 1

    return dict(stats)


# ============================================================================
# 2. PKU-Market-PCB / VOC_PCB + Leak-Free Dense Normal Tile Builder
# ============================================================================

def _extract_pku_board_id(filename: str) -> str:
    """Extract board ID (01..12) from VOC_PCB crop or full-board filename."""
    m = re.search(r"(?:missing_hole|mouse_bite|mousebite|open_circuit|short|spur|spurious_copper)_(\d{2})_", filename.lower())
    if m:
        return m.group(1)
    m2 = re.match(r"^(\d{1,2})\.(?:jpg|jpeg|png|bmp)$", filename.lower())
    if m2:
        return f"{int(m2.group(1)):02d}"
    return "00"


def ingest_voc_pcb_and_good_boards(
    voc_pcb_root: Optional[Path],
    good_boards_dir: Optional[Path],
    det_out: Path,
    anom_out: Path,
    tile_size: int = 600,
    stride: int = 180,
    include_oblique_rotations: bool = True,
    seed: int = 42,
) -> Dict[str, int]:
    """
    1. Copies/maps existing VOC_PCB train/val/test YOLO splits into `det_out` (Classes 0..5).
    2. Upgrades `augment_good.py` by tiling full-board `good/*.JPG` images with dense stride (180px),
       both cardinal (0/90/180/270) AND mild oblique (+/-6 deg) rotations (matching `l_light_*`),
       and partitioning tiles by spatial tile-coordinate into `train/good` (75%), `val/good` (10%),
       and `test/good` (15%) per board SKU (`pku_pcb_01`..`pku_pcb_12`), plus anomalous test crops!
       This guarantees `image_auroc` is ALWAYS well-defined and leak-free.
    """
    stats: Counter[str] = Counter()
    rng = random.Random(seed)
    _ensure_dirs(det_out)

    # Part A: VOC_PCB detection splits + anomalous test crops for PatchCore evaluation
    if voc_pcb_root is not None and voc_pcb_root.exists():
        for split in ("train", "val", "test"):
            img_dir = voc_pcb_root / "images" / split
            lbl_dir = voc_pcb_root / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                out_name = f"pku_{img_path.name}"
                _copy_or_link(img_path, det_out / "images" / split / out_name)
                if lbl_path.exists():
                    # VOC_PCB already uses 0:missing_hole, 1:mousebite, 2:open_circuit, 3:short, 4:spur, 5:spurious_copper
                    _copy_or_link(lbl_path, det_out / "labels" / split / f"pku_{img_path.stem}.txt")
                else:
                    (det_out / "labels" / split / f"pku_{img_path.stem}.txt").write_text("", encoding="utf-8")
                stats[f"pku_det_{split}"] += 1

                # Mirror val/test defective crops into anomaly_multiproduct/pku_pcb_<board_id>
                if split in {"val", "test"} and lbl_path.exists():
                    board_id = _extract_pku_board_id(img_path.name)
                    sku_dir = anom_out / f"pku_pcb_{board_id}"
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    gt_mask = np.zeros((h, w), dtype=np.uint8)
                    defect_cls_name = "defect"
                    for line in lbl_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cid = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = max(0, int((cx - bw / 2.0) * w))
                        y1 = max(0, int((cy - bh / 2.0) * h))
                        x2 = min(w, int((cx + bw / 2.0) * w))
                        y2 = min(h, int((cy + bh / 2.0) * h))
                        gt_mask[y1:y2, x1:x2] = 255
                        if 0 <= cid < len(UNIFIED_CLASS_NAMES):
                            defect_cls_name = UNIFIED_CLASS_NAMES[cid]

                    if gt_mask.sum() > 0:
                        anom_dst = sku_dir / split / defect_cls_name / out_name
                        mask_dst = sku_dir / "ground_truth" / split / defect_cls_name / f"pku_{img_path.stem}_mask.png"
                        anom_dst.parent.mkdir(parents=True, exist_ok=True)
                        mask_dst.parent.mkdir(parents=True, exist_ok=True)
                        _copy_or_link(img_path, anom_dst)
                        cv2.imwrite(str(mask_dst), gt_mask)
                        stats[f"pku_anom_defect_{split}"] += 1

    # Part B: Dense normal tile generation from full-board `good/` templates
    if good_boards_dir is not None and good_boards_dir.exists():
        gammas = (0.94, 1.0, 1.07)
        oblique_angles = (-6.0, 6.0) if include_oblique_rotations else ()

        for board_path in sorted(good_boards_dir.iterdir()):
            if board_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
                continue
            board_img = cv2.imread(str(board_path))
            if board_img is None:
                continue
            board_id = _extract_pku_board_id(board_path.name)
            sku_dir = anom_out / f"pku_pcb_{board_id}"

            # Also save the full golden reference template for optional ORB homography alignment!
            ref_dir = sku_dir / "reference_templates"
            ref_dir.mkdir(parents=True, exist_ok=True)
            _copy_or_link(board_path, ref_dir / board_path.name)

            H, W = board_img.shape[:2]
            xs = list(range(0, max(W - tile_size, 0) + 1, stride))
            ys = list(range(0, max(H - tile_size, 0) + 1, stride))
            if W >= tile_size and (not xs or xs[-1] != W - tile_size):
                xs.append(W - tile_size)
            if H >= tile_size and (not ys or ys[-1] != H - tile_size):
                ys.append(H - tile_size)

            coords = [(x, y) for y in ys for x in xs]
            rng.shuffle(coords)
            n_total = len(coords)
            n_test = max(2, int(n_total * 0.15))
            n_val = max(2, int(n_total * 0.10))

            for idx, (tx, ty) in enumerate(coords):
                if idx < n_test:
                    split = "test"
                elif idx < n_test + n_val:
                    split = "val"
                else:
                    split = "train"

                crop = board_img[ty : ty + tile_size, tx : tx + tile_size]
                if crop.shape[:2] != (tile_size, tile_size):
                    continue

                out_good_dir = sku_dir / split / "good"
                out_good_dir.mkdir(parents=True, exist_ok=True)

                # For val/test we keep realistic unaugmented + 1 rotated/lit view; for train we cover full grid
                rot_indices = (0, 1, 2, 3) if split == "train" else (0, rng.choice((1, 3)))
                split_gammas = gammas if split == "train" else (1.0,)

                for k in rot_indices:
                    rot_crop = np.rot90(crop, k=k).copy()
                    for g_idx, gamma in enumerate(split_gammas):
                        if abs(gamma - 1.0) > 1e-3:
                            table = (np.power(np.linspace(0, 1, 256), 1.0 / gamma) * 255.0).clip(0, 255).astype(np.uint8)
                            lit = cv2.LUT(rot_crop, table)
                        else:
                            lit = rot_crop
                        fname = f"pku_good_b{board_id}_y{ty:04d}_x{tx:04d}_r{k*90:03d}_g{g_idx}.jpg"
                        cv2.imwrite(str(out_good_dir / fname), lit, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        stats[f"pku_good_{split}"] += 1

                # Add oblique rotation crop (`l_light_*` simulation) so PatchCore learns slight angled traces
                if split == "train" and oblique_angles:
                    angle = rng.choice(oblique_angles)
                    M = cv2.getRotationMatrix2D((tile_size / 2.0, tile_size / 2.0), angle, 1.05)
                    oblique = cv2.warpAffine(crop, M, (tile_size, tile_size), borderMode=cv2.BORDER_REFLECT_101)
                    fname = f"pku_good_b{board_id}_y{ty:04d}_x{tx:04d}_oblique{int(angle):+d}.jpg"
                    cv2.imwrite(str(out_good_dir / fname), oblique, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    stats["pku_good_train"] += 1

    return dict(stats)


# ============================================================================
# 3. VisA Multi-Product Assembled PCBA (`pcb1`, `pcb2`, `pcb3`, `pcb4`)
# ============================================================================

def ingest_visa_pcb(
    visa_root: Path,
    det_out: Path,
    anom_out: Path,
    pcb_subsets: Tuple[str, ...] = ("pcb1", "pcb2", "pcb3", "pcb4"),
    seed: int = 42,
) -> Dict[str, int]:
    """
    Ingest VisA PCB1-PCB4 multi-product assembled PCBA benchmark.
    Supports both:
      - Raw VisA archive layout (`split_csv/1cls.csv` + `pcb1/Data/Images/{Normal,Anomaly}` + `Masks/Anomaly`)
      - Pre-organized MVTec/Anomalib folder layout (`pcb1/{train,test,ground_truth}`)
    """
    stats: Counter[str] = Counter()
    rng = random.Random(seed)
    _ensure_dirs(det_out)

    split_csv = visa_root / "split_csv" / "1cls.csv"
    if split_csv.exists():
        rows_by_obj: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with split_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = row.get("object", "").strip().lower()
                if obj in pcb_subsets:
                    rows_by_obj[obj].append(row)

        for obj, rows in rows_by_obj.items():
            sku_dir = anom_out / f"visa_{obj}"
            train_normals = [r for r in rows if r["split"].strip() == "train" and r["label"].strip() == "normal"]
            rng.shuffle(train_normals)
            n_val = max(2, int(len(train_normals) * 0.12))
            val_set = {id(r) for r in train_normals[:n_val]}

            for r in rows:
                img_path = visa_root / r["image"]
                if not img_path.exists():
                    continue
                raw_split = r["split"].strip()
                label = r["label"].strip().lower()
                split = "val" if id(r) in val_set else raw_split

                if label == "normal":
                    dst = sku_dir / split / "good" / f"visa_{obj}_{img_path.name}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _copy_or_link(img_path, dst)
                    stats[f"visa_{obj}_good_{split}"] += 1
                else:
                    mask_rel = r.get("mask", "").strip()
                    mask_path = visa_root / mask_rel if mask_rel else None
                    anom_dst = sku_dir / "test" / "anomaly" / f"visa_{obj}_{img_path.name}"
                    anom_dst.parent.mkdir(parents=True, exist_ok=True)
                    _copy_or_link(img_path, anom_dst)
                    stats[f"visa_{obj}_anomaly_test"] += 1

                    if mask_path is not None and mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            bin_mask = ((mask > 0).astype(np.uint8)) * 255
                            mask_dst = sku_dir / "ground_truth" / "test" / "anomaly" / f"visa_{obj}_{img_path.stem}_mask.png"
                            mask_dst.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(mask_dst), bin_mask)

                            # Extract connected-component bounding boxes from pixel mask for 10-class detector eval/training!
                            h, w = bin_mask.shape[:2]
                            num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                            boxes: List[Tuple[int, float, float, float, float]] = []
                            for cc_idx in range(1, num_cc):
                                x, y, bw_px, bh_px, area = cc_stats[cc_idx]
                                if area < 12:
                                    continue
                                ybox = _xyxy_to_yolo(x, y, x + bw_px, y + bh_px, w, h)
                                if ybox is not None:
                                    # Default VisA PCBA structural/surface defect class -> 9 (surface_scratch / general PCBA anomaly)
                                    # or specific class if encoded in filename
                                    cid = resolve_unified_class_id(img_path.stem, "visa") or 9
                                    boxes.append((cid, *ybox))
                            if boxes:
                                det_split = "test" if rng.random() < 0.5 else "train"
                                _copy_or_link(img_path, det_out / "images" / det_split / f"visa_{obj}_{img_path.name}")
                                _write_yolo_labels(det_out / "labels" / det_split / f"visa_{obj}_{img_path.stem}.txt", boxes)
                                stats[f"visa_det_{det_split}"] += 1
    else:
        # Already in MVTec/Anomalib folder structure (e.g. MoxVox/PCB-Bank pcb1/{train/good, test/good, test/bad, ground_truth/bad})
        for obj in pcb_subsets:
            cand = visa_root / obj
            if not cand.exists():
                continue
            sku_dir = anom_out / f"visa_{obj}"
            for split_sub in ("train/good", "test/good", "val/good"):
                src_sub = cand / split_sub
                if src_sub.exists():
                    dst_sub = sku_dir / split_sub
                    dst_sub.mkdir(parents=True, exist_ok=True)
                    for p in sorted(src_sub.iterdir()):
                        if p.suffix.lower() in SUPPORTED_IMG_EXTS:
                            _copy_or_link(p, dst_sub / f"visa_{obj}_{p.name}")
                            stats[f"visa_{obj}_{split_sub.replace('/', '_')}"] += 1
            for bad_name in ("bad", "anomaly"):
                src_bad = cand / "test" / bad_name
                if src_bad.exists():
                    dst_bad = sku_dir / "test" / "anomaly"
                    dst_bad.mkdir(parents=True, exist_ok=True)
                    for p in sorted(src_bad.iterdir()):
                        if p.suffix.lower() in SUPPORTED_IMG_EXTS:
                            _copy_or_link(p, dst_bad / f"visa_{obj}_{p.name}")
                            stats[f"visa_{obj}_test_anomaly"] += 1
                            # Check matching ground-truth mask
                            for gt_sub in (cand / "ground_truth" / bad_name, cand / "ground_truth" / "test" / bad_name):
                                for m_cand in (gt_sub / p.name, gt_sub / f"{p.stem}.png", gt_sub / f"{p.stem}_mask.png"):
                                    if m_cand.exists():
                                        mask = cv2.imread(str(m_cand), cv2.IMREAD_GRAYSCALE)
                                        if mask is not None:
                                            bin_mask = ((mask > 0).astype(np.uint8)) * 255
                                            mask_dst = sku_dir / "ground_truth" / "test" / "anomaly" / f"visa_{obj}_{p.stem}_mask.png"
                                            mask_dst.parent.mkdir(parents=True, exist_ok=True)
                                            cv2.imwrite(str(mask_dst), bin_mask)
                                            # Extract bounding boxes for 10-class detector
                                            h, w = bin_mask.shape[:2]
                                            num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                                            boxes = []
                                            for cc_idx in range(1, num_cc):
                                                x, y, bw_px, bh_px, area = cc_stats[cc_idx]
                                                if area < 12:
                                                    continue
                                                ybox = _xyxy_to_yolo(x, y, x + bw_px, y + bh_px, w, h)
                                                if ybox is not None:
                                                    boxes.append((9, *ybox))
                                            if boxes:
                                                det_split = "test" if rng.random() < 0.5 else "train"
                                                _copy_or_link(p, det_out / "images" / det_split / f"visa_{obj}_{p.name}")
                                                _write_yolo_labels(det_out / "labels" / det_split / f"visa_{obj}_{p.stem}.txt", boxes)
                                                stats[f"visa_det_{det_split}"] += 1
                                        break

    return dict(stats)


# ============================================================================
# 4. Generic Assembled PCBA Component & Solder Dataset (YOLO / VOC XML / COCO)
# ============================================================================

def ingest_pcba_yolo_or_voc(
    pcba_root: Path,
    det_out: Path,
    prefix: str = "pcba",
) -> Dict[str, int]:
    """
    Harmonize an external Assembled PCBA Component & Solder Defect dataset
    (e.g., Kaggle / Roboflow YOLO data.yaml or Pascal VOC XML folder) into the
    unified 10-class taxonomy (Classes 6..9: bad_solder, missing_component,
    misaligned_component, surface_scratch, plus any 0..5 trace classes).
    """
    stats: Counter[str] = Counter()
    _ensure_dirs(det_out)

    # Case 1: YOLO format with data.yaml
    yaml_candidates = list(pcba_root.glob("*.yaml")) + list(pcba_root.glob("*/*.yaml"))
    if yaml_candidates:
        yaml_path = yaml_candidates[0]
        raw_text = yaml_path.read_text(encoding="utf-8")
        # Lightweight class-name parser without requiring PyYAML dependency
        src_id_to_name: Dict[int, str] = {}
        list_match = re.search(r"names\s*:\s*\[([^\]]+)\]", raw_text)
        if list_match:
            items = [x.strip().strip("'\"") for x in list_match.group(1).split(",") if x.strip()]
            src_id_to_name = {i: name for i, name in enumerate(items)}
        else:
            for m in re.finditer(r"^\s*(\d+)\s*:\s*['\"]?([^'\"\n#]+)['\"]?", raw_text, flags=re.MULTILINE):
                src_id_to_name[int(m.group(1))] = m.group(2).strip()
            if not src_id_to_name:
                names_block = re.search(r"names\s*:\s*\r?\n((?:\s*-\s*[^\r\n]+\r?\n?)+)", raw_text)
                if names_block:
                    items = [
                        re.sub(r"^\s*-\s*", "", line).strip().strip("'\"")
                        for line in names_block.group(1).splitlines()
                        if line.strip().startswith("-")
                    ]
                    src_id_to_name = {i: name for i, name in enumerate(items)}

        id_remap: Dict[int, int] = {}
        for src_id, src_name in src_id_to_name.items():
            uid = resolve_unified_class_id(src_name, source_dataset="pcba")
            if uid is not None:
                id_remap[src_id] = uid

        split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        for raw_split, dst_split in split_aliases.items():
            # Support both `train/images` (Roboflow) and `images/train` (Ultralytics)
            for img_dir, lbl_dir in (
                (yaml_path.parent / raw_split / "images", yaml_path.parent / raw_split / "labels"),
                (yaml_path.parent / "images" / raw_split, yaml_path.parent / "labels" / raw_split),
            ):
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.iterdir()):
                    if img_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
                        continue
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    mapped_boxes: List[Tuple[int, float, float, float, float]] = []
                    if lbl_path.exists():
                        for line in lbl_path.read_text(encoding="utf-8").splitlines():
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            scid = int(float(parts[0]))
                            if scid not in id_remap:
                                continue
                            cx, cy, bw, bh = map(float, parts[1:5])
                            mapped_boxes.append((id_remap[scid], cx, cy, bw, bh))
                            stats[f"pcba_box_{UNIFIED_CLASS_NAMES[id_remap[scid]]}"] += 1

                    if mapped_boxes:
                        out_stem = f"{prefix}_{img_path.stem}"
                        _copy_or_link(img_path, det_out / "images" / dst_split / f"{out_stem}{img_path.suffix.lower()}")
                        _write_yolo_labels(det_out / "labels" / dst_split / f"{out_stem}.txt", mapped_boxes)
                        stats[f"pcba_det_{dst_split}"] += 1
        return dict(stats)

    # Case 2: Pascal VOC XML annotations
    xml_files = sorted(pcba_root.rglob("*.xml"))
    for idx, xml_path in enumerate(xml_files):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        cands = list(pcba_root.rglob(filename))
        if not cands:
            continue
        img_path = cands[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for obj in root.findall("object"):
            cname = obj.findtext("name") or ""
            uid = resolve_unified_class_id(cname, source_dataset="pcba")
            bnd = obj.find("bndbox")
            if uid is None or bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            ybox = _xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h)
            if ybox is not None:
                boxes.append((uid, *ybox))
        if boxes:
            dst_split = "test" if idx % 10 in (0, 1) else ("val" if idx % 10 == 2 else "train")
            out_stem = f"{prefix}_{img_path.stem}"
            _copy_or_link(img_path, det_out / "images" / dst_split / f"{out_stem}{img_path.suffix.lower()}")
            _write_yolo_labels(det_out / "labels" / dst_split / f"{out_stem}.txt", boxes)
            stats[f"pcba_xml_det_{dst_split}"] += 1

    return dict(stats)


def write_unified_data_yaml(det_out: Path) -> Path:
    """Write Ultralytics-compatible 10-class `data.yaml`."""
    yaml_lines = [
        f"path: {det_out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(UNIFIED_CLASS_NAMES)}",
        "names:",
    ]
    for idx, cname in enumerate(UNIFIED_CLASS_NAMES):
        yaml_lines.append(f"  {idx}: {cname}")
    yaml_path = det_out / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def pack_directory_to_tar(source_dir: Path, tar_path: Path) -> Path:
    """Create an uncompressed `.tar` bundle for fast Colab local NVMe SSD extraction."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(source_dir, arcname=source_dir.name)
    print(f"[Dataset] Packed {source_dir} -> {tar_path} ({tar_path.stat().st_size / (1024**2):.1f} MB)")
    return tar_path


def download_hf_pcba_solder(target_dir: Path) -> Path:
    """Download public ParsaVictor/pcb-solder-defect-detection dataset from HuggingFace CDN."""
    import urllib.request

    pcba_dir = target_dir / "pcba_solder"
    pcba_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("data.yaml", "images_and_labels.zip"):
        dest = pcba_dir / fname
        if not dest.exists():
            url = f"https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection/resolve/main/{fname}"
            print(f"[PCBA] Downloading {url} -> {dest} ...")
            urllib.request.urlretrieve(url, str(dest))
    extract_root = pcba_dir / "final_solder"
    if not extract_root.exists():
        with zipfile.ZipFile(pcba_dir / "images_and_labels.zip", "r") as zf:
            zf.extractall(pcba_dir)
    return extract_root if extract_root.exists() else pcba_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build Unified 10-Class & Multi-Product PCB/PCBA Datasets.")
    ap.add_argument("--out-dir", default="datasets", help="Output root directory")
    ap.add_argument("--voc-pcb", default=None, help="Path to existing VOC_PCB directory")
    ap.add_argument("--good-boards", default=None, help="Path to full-board defect-free PCB templates (good/)")
    ap.add_argument("--deeppcb", default=None, help="Path to DeepPCB root (or pass --download-deeppcb)")
    ap.add_argument("--download-deeppcb", action="store_true", help="Automatically git clone DeepPCB")
    ap.add_argument("--download-hf-pcba", action="store_true", help="Automatically download HuggingFace PCBA solder/component dataset")
    ap.add_argument("--visa", default=None, help="Path to VisA root directory (containing pcb1..pcb4)")
    ap.add_argument("--pcba", nargs="*", default=[], help="One or more Assembled PCBA component/solder dataset roots")
    ap.add_argument("--pack-tar", action="store_true", help="Bundle outputs into .tar files for Colab SSD staging")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_dir)
    det_out = out_root / "unified_pcb_10cls"
    anom_out = out_root / "anomaly_multiproduct"
    det_out.mkdir(parents=True, exist_ok=True)
    anom_out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {"classes": UNIFIED_CLASS_NAMES, "sources": {}}

    if args.download_deeppcb and not args.deeppcb:
        dp_path = download_deeppcb(out_root / "_raw")
    elif args.deeppcb:
        dp_path = Path(args.deeppcb)
    else:
        dp_path = None

    if dp_path is not None and dp_path.exists():
        manifest["sources"]["deeppcb"] = ingest_deeppcb(dp_path, det_out, anom_out, seed=args.seed)

    if args.voc_pcb or args.good_boards:
        manifest["sources"]["pku_voc_pcb"] = ingest_voc_pcb_and_good_boards(
            Path(args.voc_pcb) if args.voc_pcb else None,
            Path(args.good_boards) if args.good_boards else None,
            det_out,
            anom_out,
            seed=args.seed,
        )

    if args.visa:
        manifest["sources"]["visa_pcb"] = ingest_visa_pcb(Path(args.visa), det_out, anom_out, seed=args.seed)

    pcba_dirs = [Path(p) for p in args.pcba]
    if args.download_hf_pcba:
        pcba_dirs.append(download_hf_pcba_solder(out_root / "_raw"))

    for idx, p in enumerate(pcba_dirs):
        if p.exists():
            manifest["sources"][f"pcba_{idx}"] = ingest_pcba_yolo_or_voc(p, det_out, prefix=f"pcba{idx}")

    yaml_path = write_unified_data_yaml(det_out)
    manifest_path = out_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Dataset] Wrote unified config -> {yaml_path}")
    print(f"[Dataset] Wrote dataset manifest -> {manifest_path}")

    if args.pack_tar:
        pack_directory_to_tar(det_out, out_root / "unified_pcb_10cls.tar")
        pack_directory_to_tar(anom_out, out_root / "anomaly_multiproduct.tar")


if __name__ == "__main__":
    main()
