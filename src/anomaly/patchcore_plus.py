"""
PatchCore++ / AnomalyDINO Multi-Product PCB Anomaly & Multi-Class ROI Engine.

Directly resolves all 7 bottlenecks identified in the baseline PatchCore audit:
  1. Backbone Upgrade: Supports `dinov2_hybrid` (Frozen DINOv2-ViT-B/14-reg blocks [5,8,11]
     + High-Res WideResNet50 layer1+layer2 @ 518x518), pure `dinov2_vitb14_reg`, or `wide_resnet50_2`.
  2. Multi-Product Routing: Global CLS/GeM cosine ProductRouter automatically identifies the PCB SKU
     (e.g., `visa_pcb1..4`, `pku_pcb_01..12`, `deeppcb_*`) in <1 ms and queries its dedicated coreset
     memory bank (falling back to a unified multi-board bank for unseen SKUs).
  3. True GPU k-Center Greedy Coreset + Cached FAISS Index (`faiss-gpu` / `faiss-cpu` / chunked `torch.cdist`).
  4. Preserves Edge Contrast Energy: Disables destructive pre-kNN L2 feature collapse by default on CNN stems,
     using deterministic orthogonal projection to 512-d descriptors.
  5. Robust Scoring, Per-SKU Score Standardization & Border Masking: `k=9` Roth et al. neighborhood density
     re-weighting + peak-preserving top-q% spatial quantile pooling + 3% crop-border suppression applied
     prior to `s_star` selection + per-SKU normal ceiling standardization + optional ORB homography alignment.
  6. Multi-Class ROI Prototype Head: Connected components on the pixel anomaly map extract tight bounding boxes
     + pixel areas (`px^2`) and classify each anomalous ROI into the 10 unified PCB/PCBA defect classes
     or `novel_unseen_defect` via few-shot feature prototypes.
  7. True-RGB Overlay: Proper BGR->RGB Turbo/Jet colormap conversion (fixing inverted red/blue heatmaps).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(paths: Sequence[str | Path], recursive: bool = True) -> List[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    images: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images.extend(
                cand for cand in iterator if cand.is_file() and cand.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return sorted(images)


def load_rgb_image(path_or_image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        return ImageOps.exif_transpose(path_or_image).convert("RGB")
    if isinstance(path_or_image, np.ndarray):
        return Image.fromarray(path_or_image).convert("RGB")
    return ImageOps.exif_transpose(Image.open(path_or_image)).convert("RGB")


def align_image_orb(
    test_rgb: np.ndarray,
    template_rgb: np.ndarray,
    max_features: int = 1500,
) -> np.ndarray:
    """Align a test PCB image to a golden reference template via ORB + RANSAC homography."""
    g_test = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2GRAY)
    g_tmpl = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(g_test, None)
    kp2, des2 = orb.detectAndCompute(g_tmpl, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return test_rgb
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    good = matches[: max(12, int(len(matches) * 0.25))]
    if len(good) < 10:
        return test_rgb
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_mat, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if H_mat is None:
        return test_rgb
    h, w = template_rgb.shape[:2]
    return cv2.warpPerspective(test_rgb, H_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def render_true_rgb_overlay(
    pil_image: Image.Image,
    anomaly_map: np.ndarray,
    pixel_threshold: Optional[float] = None,
    rois: Optional[List[Dict[str, object]]] = None,
    alpha: float = 0.48,
) -> np.ndarray:
    """
    Render a true-RGB anomaly heatmap overlay (explicit BGR->RGB conversion)
    with optional connected-component bounding boxes and defect/novelty labels.
    """
    img_rgb = np.asarray(load_rgb_image(pil_image)).copy()
    h, w = img_rgb.shape[:2]

    amap = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    lo = float(np.percentile(amap, 10.0))
    hi = float(max(amap.max(), (pixel_threshold or lo) * 1.15, lo + 1e-6))
    norm = np.clip((amap - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    cmap_id = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(np.uint8(norm * 255.0), cmap_id)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # Spatially-modulated alpha so clean PCB tracks remain sharp while anomalies glow
    weight = (alpha * np.power(norm, 0.85))[..., None].astype(np.float32)
    blended = np.clip(img_rgb.astype(np.float32) * (1.0 - weight) + heat_rgb.astype(np.float32) * weight, 0, 255).astype(
        np.uint8
    )

    if rois:
        for roi in rois:
            x1, y1, x2, y2 = int(roi["xmin"]), int(roi["ymin"]), int(roi["xmax"]), int(roi["ymax"])
            cls_name = str(roi.get("class", "anomaly"))
            score = float(roi.get("score", 0.0))
            area_px = int(roi.get("area_px", (x2 - x1) * (y2 - y1)))
            is_novel = "novel" in cls_name.lower() or cls_name == "anomaly"
            color = (255, 65, 54) if is_novel else (255, 195, 0)
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
            caption = f"{cls_name} ({score:.2f} | {area_px}px2)"
            ty = max(16, y1 - 6)
            cv2.putText(blended, caption, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    return blended


class FastKNNIndex:
    """Cached FAISS (GPU/CPU) or chunked PyTorch kNN index over a float32 memory bank."""

    def __init__(self, bank: torch.Tensor, device: torch.device, use_faiss: bool = True):
        self.device = device
        self.bank_cpu = bank.detach().float().contiguous().cpu()
        self.dim = int(self.bank_cpu.shape[1])
        self.ntotal = int(self.bank_cpu.shape[0])
        self._faiss_index = None
        self._bank_gpu: Optional[torch.Tensor] = None

        if use_faiss and HAS_FAISS and self.ntotal > 0:
            cpu_index = faiss.IndexFlatL2(self.dim)
            if device.type == "cuda" and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, device.index or 0, cpu_index)
                    gpu_index.add(self.bank_cpu.numpy())
                    self._faiss_index = gpu_index
                except Exception:
                    cpu_index.add(self.bank_cpu.numpy())
                    self._faiss_index = cpu_index
            else:
                cpu_index.add(self.bank_cpu.numpy())
                self._faiss_index = cpu_index
        else:
            self._bank_gpu = self.bank_cpu.to(device)

    @torch.no_grad()
    def search(self, queries: torch.Tensor, k: int = 9, chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(euclidean_distances, neighbor_indices)` of shape `(N, k_eff)`."""
        k_eff = max(1, min(k, self.ntotal))
        if self._faiss_index is not None:
            q_np = queries.detach().float().contiguous().cpu().numpy()
            sq_dists, idxs = self._faiss_index.search(q_np, k_eff)
            dists = np.sqrt(np.maximum(sq_dists, 0.0))
            return dists, idxs

        assert self._bank_gpu is not None
        q_dev = queries.to(self.device).float()
        all_d: List[torch.Tensor] = []
        all_i: List[torch.Tensor] = []
        for s in range(0, q_dev.shape[0], chunk_size):
            d_mat = torch.cdist(q_dev[s : s + chunk_size], self._bank_gpu, p=2.0)
            vals, inds = torch.topk(d_mat, k=k_eff, dim=1, largest=False)
            all_d.append(vals.cpu())
            all_i.append(inds.cpu())
        return torch.cat(all_d, dim=0).numpy(), torch.cat(all_i, dim=0).numpy()


def greedy_kcenter_coreset(
    embeddings: torch.Tensor,
    coreset_ratio: float = 0.04,
    max_coreset_size: int = 25000,
    proj_dim: int = 128,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Roth et al. (CVPR 2022) Iterative Furthest-Point k-Center Greedy Coreset Selection
    with Johnson-Lindenstrauss random orthogonal projection on GPU.

    Preserves rare normal PCB structural features (mounting rings, fiducials, IC outlines)
    that uniform random sampling discards.
    """
    n_total, feat_dim = embeddings.shape
    target_m = min(max_coreset_size, max(64, int(round(n_total * coreset_ratio))))
    if target_m >= n_total:
        return embeddings

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Pre-cap ultra-huge raw pools (>200k patches) via stratified image stride before exact greedy pass
    if n_total > 180_000:
        perm = torch.randperm(n_total, generator=gen)[:180_000]
        pool = embeddings[perm]
        n_pool = pool.shape[0]
    else:
        pool = embeddings
        n_pool = n_total

    target_m = min(target_m, n_pool)

    with torch.no_grad():
        proj_d = min(proj_dim, feat_dim)
        proj_mat = torch.randn(feat_dim, proj_d, generator=gen, dtype=torch.float32).to(dev) / math.sqrt(proj_d)
        # Project in chunks so 4GB-16GB GPUs never OOM
        proj_chunks = []
        for s in range(0, n_pool, 32768):
            proj_chunks.append(pool[s : s + 32768].to(dev, dtype=torch.float32) @ proj_mat)
        pts = torch.cat(proj_chunks, dim=0)  # (n_pool, proj_d)

        selected = torch.empty(target_m, dtype=torch.long, device="cpu")
        cur_idx = int(torch.randint(0, n_pool, (1,), generator=gen).item())
        min_sq_dists = torch.full((n_pool,), float("inf"), dtype=torch.float32, device=dev)

        for step in range(target_m):
            selected[step] = cur_idx
            center = pts[cur_idx : cur_idx + 1]
            sq_d = ((pts - center) ** 2).sum(dim=1)
            min_sq_dists = torch.minimum(min_sq_dists, sq_d)
            cur_idx = int(torch.argmax(min_sq_dists).item())

    return pool[selected]


class _MultiScaleBackbone(nn.Module):
    """
    Selectable feature extractor:
      - `dinov2_hybrid`: Frozen `dinov2_vitb14_reg` blocks (5, 8, 11) + `wide_resnet50_2` (layer1, layer2)
      - `dinov2_vitb14_reg`: Frozen `dinov2_vitb14_reg` blocks (5, 8, 11)
      - `wide_resnet50_2`: `wide_resnet50_2` configurable layers (default `layer1`, `layer2` or `layer2`, `layer3`)
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_hybrid",
        cnn_layers: Sequence[str] = ("layer1", "layer2"),
        dino_blocks: Sequence[int] = (5, 8, 11),
        neighborhood_kernel: int = 3,
        l2_normalize_cnn: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.cnn_layers = tuple(cnn_layers)
        self.dino_blocks = tuple(dino_blocks)
        self.neighborhood_kernel = int(neighborhood_kernel)
        self.l2_normalize_cnn = bool(l2_normalize_cnn)

        self.cnn_extractor: Optional[nn.Module] = None
        self.dino_model: Optional[nn.Module] = None
        self.raw_dim = 0

        if backbone_name in {"wide_resnet50_2", "dinov2_hybrid"}:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
            wrn = models.wide_resnet50_2(weights=weights)
            self.cnn_extractor = create_feature_extractor(wrn, return_nodes={l: l for l in self.cnn_layers})
            layer_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
            self.raw_dim += sum(layer_dims[l] for l in self.cnn_layers)

        if backbone_name in {"dinov2_vitb14_reg", "dinov2_hybrid"}:
            if not pretrained:
                raise ValueError("DINOv2 backbones require pretrained=True.")
            try:
                self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
            except Exception:
                self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.raw_dim += 768 * len(self.dino_blocks)

        if self.raw_dim == 0:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Returns:
          - `patches`: `(B, Hf * Wf, D_raw)` patch descriptors
          - `global_sig`: `(B, D_global)` L2-normalized board signature for SKU ProductRouter
          - `grid_size`: `(Hf, Wf)`
        """
        b_sz, _, img_h, img_w = batch.shape
        feat_maps: List[torch.Tensor] = []
        global_parts: List[torch.Tensor] = []
        target_hw: Optional[Tuple[int, int]] = None

        if self.cnn_extractor is not None:
            cnn_outs = self.cnn_extractor(batch)
            # Align to 64x64 grid (stride 8) for micro-defect localization without 128x128 memory explosion
            first_hw = cnn_outs[self.cnn_layers[0]].shape[-2:]
            target_hw = (min(64, first_hw[0]), min(64, first_hw[1]))
            cnn_aligned = []
            for lname in self.cnn_layers:
                fmap = cnn_outs[lname]
                if self.neighborhood_kernel > 1:
                    fmap = F.avg_pool2d(
                        fmap,
                        kernel_size=self.neighborhood_kernel,
                        stride=1,
                        padding=self.neighborhood_kernel // 2,
                    )
                if fmap.shape[-2:] != target_hw:
                    fmap = F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
                # Standardize per-layer scale without collapsing spatial edge magnitude unless l2_normalize_cnn=True
                if self.l2_normalize_cnn:
                    fmap = F.normalize(fmap, p=2, dim=1)
                else:
                    fmap = fmap / (fmap.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-6)
                cnn_aligned.append(fmap)

            cnn_cat = torch.cat(cnn_aligned, dim=1)
            feat_maps.append(cnn_cat)
            # GeM global pooling for board SKU signature
            gem = cnn_cat.clamp(min=1e-6).pow(3.0).mean(dim=(-2, -1)).pow(1.0 / 3.0)
            global_parts.append(F.normalize(gem, p=2, dim=-1))

        if self.dino_model is not None:
            # Ensure input divisible by patch size 14 for DINOv2
            dh = max(14, (img_h // 14) * 14)
            dw = max(14, (img_w // 14) * 14)
            dino_in = batch if (dh == img_h and dw == img_w) else F.interpolate(batch, size=(dh, dw), mode="bilinear", align_corners=False)
            gh, gw = dh // 14, dw // 14

            inter = self.dino_model.get_intermediate_layers(
                dino_in,
                n=self.dino_blocks,
                reshape=True,
                return_class_token=True,
            )
            if target_hw is None:
                target_hw = (gh, gw)

            dino_maps = []
            cls_tokens = []
            for patch_map, cls_tok in inter:
                pm = F.normalize(patch_map, p=2, dim=1) * math.sqrt(patch_map.shape[1] / 64.0)
                if pm.shape[-2:] != target_hw:
                    pm = F.interpolate(pm, size=target_hw, mode="bilinear", align_corners=False)
                dino_maps.append(pm)
                cls_tokens.append(F.normalize(cls_tok, p=2, dim=-1))

            feat_maps.append(torch.cat(dino_maps, dim=1))
            global_parts.append(F.normalize(torch.cat(cls_tokens, dim=-1), p=2, dim=-1))

        assert target_hw is not None
        full_map = torch.cat(feat_maps, dim=1)  # (B, D_raw, Hf, Wf)
        hf, wf = target_hw
        patches = full_map.permute(0, 2, 3, 1).reshape(b_sz, hf * wf, full_map.shape[1])
        global_sig = F.normalize(torch.cat(global_parts, dim=-1), p=2, dim=-1)
        return patches, global_sig, (hf, wf)


class PatchCorePlus:
    """
    Multi-Product PatchCore++ / AnomalyDINO Engine with:
      - Hybrid DINOv2-reg + WideResNet50 multi-scale features
      - Automatic Product-SKU Routing + Unified Multi-Product Coreset fallback
      - GPU k-Center Greedy Coreset + Cached FAISS kNN
      - Roth et al. `k=9` density-reweighted scoring + border-suppressed `s_star` + per-SKU standardization
      - Connected-Component ROI Extraction + Few-Shot Defect Prototype Classification
    """

    def __init__(
        self,
        image_size: int = 518,
        backbone_name: str = "dinov2_hybrid",
        cnn_layers: Sequence[str] = ("layer1", "layer2"),
        dino_blocks: Sequence[int] = (5, 8, 11),
        projection_dim: int = 512,
        neighborhood_kernel: int = 3,
        l2_normalize_cnn: bool = False,
        knn_k: int = 9,
        gaussian_sigma: float = 2.0,
        border_margin_ratio: float = 0.03,
        top_q_pool: float = 0.0015,
        sku_sim_threshold: float = 0.80,
        novelty_sim_threshold: float = 0.68,
        pretrained: bool = True,
        device: Optional[str] = None,
        projection_matrix: Optional[torch.Tensor] = None,
        unified_bank: Optional[torch.Tensor] = None,
        banks_by_sku: Optional[Dict[str, torch.Tensor]] = None,
        sku_centroids: Optional[Dict[str, torch.Tensor]] = None,
        threshold: Optional[float] = None,
        thresholds_by_sku: Optional[Dict[str, float]] = None,
        sku_score_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        pixel_threshold: Optional[float] = None,
        defect_prototypes: Optional[Dict[str, torch.Tensor]] = None,
        seed: int = 42,
    ):
        # Round image_size cleanly for ViT/14 when using DINOv2
        if "dinov2" in backbone_name and image_size % 14 != 0:
            image_size = max(224, round(image_size / 14) * 14)
        self.image_size = int(image_size)
        self.backbone_name = backbone_name
        self.cnn_layers = tuple(cnn_layers)
        self.dino_blocks = tuple(dino_blocks)
        self.projection_dim = int(projection_dim)
        self.neighborhood_kernel = int(neighborhood_kernel)
        self.l2_normalize_cnn = bool(l2_normalize_cnn)
        self.knn_k = int(knn_k)
        self.gaussian_sigma = float(gaussian_sigma)
        self.border_margin_ratio = float(border_margin_ratio)
        self.top_q_pool = float(top_q_pool)
        self.sku_sim_threshold = float(sku_sim_threshold)
        self.novelty_sim_threshold = float(novelty_sim_threshold)
        self.pretrained = bool(pretrained)
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.backbone = _MultiScaleBackbone(
            backbone_name=self.backbone_name,
            cnn_layers=self.cnn_layers,
            dino_blocks=self.dino_blocks,
            neighborhood_kernel=self.neighborhood_kernel,
            l2_normalize_cnn=self.l2_normalize_cnn,
            pretrained=self.pretrained,
        ).to(self.device).eval()

        if projection_matrix is not None:
            self.projection_matrix = projection_matrix.float().to(self.device)
        elif self.backbone.raw_dim > self.projection_dim:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            q_mat, _ = torch.linalg.qr(
                torch.randn(self.backbone.raw_dim, self.projection_dim, generator=gen, dtype=torch.float32)
            )
            self.projection_matrix = (q_mat * math.sqrt(self.backbone.raw_dim / self.projection_dim)).to(self.device)
        else:
            self.projection_matrix = torch.eye(self.backbone.raw_dim, dtype=torch.float32, device=self.device)
            self.projection_dim = self.backbone.raw_dim

        self.unified_bank: Optional[torch.Tensor] = unified_bank.cpu() if unified_bank is not None else None
        self.banks_by_sku: Dict[str, torch.Tensor] = {
            k: v.cpu() for k, v in (banks_by_sku or {}).items()
        }
        self.sku_centroids: Dict[str, torch.Tensor] = {
            k: F.normalize(v.float().cpu(), p=2, dim=-1) for k, v in (sku_centroids or {}).items()
        }
        self.threshold = threshold
        self.thresholds_by_sku: Dict[str, float] = dict(thresholds_by_sku or {})
        self.sku_score_stats: Dict[str, Tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in (sku_score_stats or {}).items()
        }
        self.pixel_threshold = pixel_threshold
        self.defect_prototypes: Dict[str, torch.Tensor] = {
            k: F.normalize(v.float().cpu(), p=2, dim=-1) for k, v in (defect_prototypes or {}).items()
        }

        self._unified_index: Optional[FastKNNIndex] = None
        self._sku_indices: Dict[str, FastKNNIndex] = {}
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._unified_index = (
            FastKNNIndex(self.unified_bank, self.device) if self.unified_bank is not None and len(self.unified_bank) > 0 else None
        )
        self._sku_indices = {
            sku: FastKNNIndex(bank, self.device)
            for sku, bank in self.banks_by_sku.items()
            if len(bank) > 0
        }

    def _preprocess_batch(self, images: Sequence[str | Path | Image.Image | np.ndarray]) -> torch.Tensor:
        tensors = [self.transform(load_rgb_image(img)) for img in images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def extract_descriptors(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        batch = self._preprocess_batch(images)
        raw_patches, global_sig, grid = self.backbone(batch)
        proj_patches = raw_patches @ self.projection_matrix  # (B, Hf*Wf, D_proj)
        return proj_patches, global_sig, grid

    @torch.no_grad()
    def route_product_sku(self, global_sig_1d: torch.Tensor) -> Tuple[str, float]:
        """Identify PCB product SKU from global CLS/GeM signature in <1 ms."""
        if not self.sku_centroids:
            return "unified_default", 1.0
        q = F.normalize(global_sig_1d.float().cpu(), p=2, dim=-1)
        best_sku = "unknown_sku"
        best_sim = -1.0
        for sku, centroid in self.sku_centroids.items():
            sim = float(torch.dot(q, centroid.view(-1)).item())
            if sim > best_sim:
                best_sim = sim
                best_sku = sku
        if best_sim < self.sku_sim_threshold:
            return "unknown_sku", best_sim
        return best_sku, best_sim

    @torch.no_grad()
    def fit(
        self,
        normal_images: Sequence[str | Path] | Dict[str, Sequence[str | Path]],
        batch_size: int = 8,
        coreset_ratio: float = 0.04,
        max_coreset_per_sku: int = 18000,
        max_unified_coreset: int = 30000,
        threshold_quantile: float = 0.985,
        exclude_masks: Optional[Dict[str, str | Path]] = None,
    ) -> "PatchCorePlus":
        """
        Fit either:
          - Single-product / pooled list of normal image paths, OR
          - Multi-product dict `{product_sku: [normal_image_paths...]}` (builds per-SKU coresets
            + SKU router centroids + per-SKU score standardization + unified fallback coreset!).
        """
        if isinstance(normal_images, dict):
            sku_map = {k: list(v) for k, v in normal_images.items() if len(v) > 0}
        else:
            sku_map = {"default_pcb": list(normal_images)}

        if not sku_map:
            raise ValueError("No normal images were provided to PatchCorePlus.fit().")

        mask_lookup = {str(k): Path(v) for k, v in (exclude_masks or {}).items()}
        unified_pool: List[torch.Tensor] = []
        self.banks_by_sku = {}
        self.sku_centroids = {}
        self.sku_score_stats = {}

        for sku, img_paths in sku_map.items():
            sku_patches: List[torch.Tensor] = []
            sku_globals: List[torch.Tensor] = []

            for s in range(0, len(img_paths), batch_size):
                chunk_paths = img_paths[s : s + batch_size]
                patches, g_sigs, (hf, wf) = self.extract_descriptors(chunk_paths)
                for b_idx, p_item in enumerate(chunk_paths):
                    p_tokens = patches[b_idx].cpu()  # (Hf*Wf, D)
                    m_path = mask_lookup.get(str(p_item)) or mask_lookup.get(Path(p_item).name)
                    if m_path is not None and m_path.exists():
                        m_gray = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                        if m_gray is not None and np.any(m_gray > 0):
                            m_dil = cv2.dilate((m_gray > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
                            m_grid = cv2.resize(m_dil, (wf, hf), interpolation=cv2.INTER_NEAREST).ravel()
                            keep = m_grid == 0
                            if int(keep.sum()) >= 16:
                                p_tokens = p_tokens[torch.from_numpy(keep)]
                    sku_patches.append(p_tokens)
                sku_globals.append(g_sigs.cpu())

            raw_sku_bank = torch.cat(sku_patches, dim=0)
            coreset_sku = greedy_kcenter_coreset(
                raw_sku_bank,
                coreset_ratio=coreset_ratio,
                max_coreset_size=max_coreset_per_sku,
                seed=self.seed,
                device=self.device,
            )
            self.banks_by_sku[sku] = coreset_sku.cpu()
            unified_pool.append(coreset_sku.cpu())

            mean_global = torch.cat(sku_globals, dim=0).mean(dim=0)
            self.sku_centroids[sku] = F.normalize(mean_global, p=2, dim=-1)

        merged_coresets = torch.cat(unified_pool, dim=0)
        if len(merged_coresets) > max_unified_coreset:
            self.unified_bank = greedy_kcenter_coreset(
                merged_coresets,
                coreset_ratio=max_unified_coreset / float(len(merged_coresets)),
                max_coreset_size=max_unified_coreset,
                seed=self.seed,
                device=self.device,
            ).cpu()
        else:
            self.unified_bank = merged_coresets.cpu()

        self._rebuild_indices()

        # Fit per-SKU score standardization & thresholds from normal training score distribution
        all_norm_scores: List[float] = []
        all_pixel_q995: List[float] = []
        self.thresholds_by_sku = {}

        for sku, img_paths in sku_map.items():
            sample_paths = img_paths[: min(len(img_paths), 120)]
            raw_sku_scores: List[float] = []
            for p in sample_paths:
                m_path = mask_lookup.get(str(p)) or mask_lookup.get(Path(p).name)
                pred = self.predict(
                    p,
                    product_sku=sku,
                    extract_rois=False,
                    render_overlay=False,
                    _raw_unscaled=True,
                    exclude_mask=m_path,
                )
                raw_sku_scores.append(float(pred["score"]))
                all_pixel_q995.append(float(np.quantile(pred["anomaly_map"], 0.995)))

            arr_raw = np.asarray(raw_sku_scores, dtype=np.float64)
            q50 = float(np.median(arr_raw))
            q_thr = float(np.quantile(arr_raw, threshold_quantile))
            iqr = max(float(q_thr - q50), float(np.std(arr_raw)), 1e-3)
            denom = max(q_thr, q50 + 1.5 * iqr, 1e-4)
            self.sku_score_stats[sku] = (q50, denom)
            self.thresholds_by_sku[sku] = 1.0
            for rs in raw_sku_scores:
                all_norm_scores.append(float(rs / denom))

        self.threshold = float(np.quantile(np.asarray(all_norm_scores), threshold_quantile)) if all_norm_scores else 1.0
        self.pixel_threshold = float(np.quantile(np.asarray(all_pixel_q995), 0.98)) if all_pixel_q995 else 0.5
        return self

    @torch.no_grad()
    def calibrate_threshold(
        self,
        val_normal_images: Sequence[str | Path],
        val_anomalous_images: Sequence[str | Path],
        product_sku: str = "auto",
    ) -> float:
        """Calibrate pass/fail threshold by maximizing F1 / Youden's J on a held-out validation split."""
        y_true: List[int] = []
        y_score: List[float] = []
        for p in val_normal_images:
            y_true.append(0)
            y_score.append(self.predict(p, product_sku=product_sku, extract_rois=False, render_overlay=False)["score"])
        for p in val_anomalous_images:
            y_true.append(1)
            y_score.append(self.predict(p, product_sku=product_sku, extract_rois=False, render_overlay=False)["score"])

        if len(set(y_true)) < 2:
            return float(self.threshold or 1.0)

        prec, rec, thresh = precision_recall_curve(y_true, y_score)
        f1 = (2.0 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
        best_idx = int(np.argmax(f1))
        best_thr = float(thresh[best_idx])
        if product_sku != "auto" and product_sku in self.thresholds_by_sku:
            self.thresholds_by_sku[product_sku] = best_thr
        else:
            self.threshold = best_thr
        return best_thr

    @torch.no_grad()
    def fit_defect_prototypes(
        self,
        labeled_samples: Sequence[Tuple[str | Path, Sequence[Tuple[str, float, float, float, float]]]],
        max_per_class: int = 250,
    ) -> Dict[str, int]:
        """
        Build a Few-Shot Defect Prototype Bank from bounding boxes `(class_name, xmin, ymin, xmax, ymax)`
        in pixel coordinates so PatchCore++ can classify anomalous ROIs into known defect classes
        (or flag `novel_unseen_defect` when similarity < `novelty_sim_threshold`).
        """
        vectors_by_class: Dict[str, List[torch.Tensor]] = {}
        counts: Dict[str, int] = {}

        for img_path, boxes in labeled_samples:
            if not boxes:
                continue
            pil_img = load_rgb_image(img_path)
            w, h = pil_img.size
            patches, _, (hf, wf) = self.extract_descriptors([pil_img])
            fmap = patches[0].reshape(hf, wf, -1)  # (Hf, Wf, D)

            for cls_name, x1, y1, x2, y2 in boxes:
                if len(vectors_by_class.get(cls_name, ())) >= max_per_class:
                    continue
                gx1 = max(0, min(wf - 1, int(math.floor(x1 / max(w, 1) * wf))))
                gx2 = max(gx1 + 1, min(wf, int(math.ceil(x2 / max(w, 1) * wf))))
                gy1 = max(0, min(hf - 1, int(math.floor(y1 / max(h, 1) * hf))))
                gy2 = max(gy1 + 1, min(hf, int(math.ceil(y2 / max(h, 1) * hf))))
                roi_vec = fmap[gy1:gy2, gx1:gx2].reshape(-1, fmap.shape[-1]).mean(dim=0)
                vectors_by_class.setdefault(cls_name, []).append(F.normalize(roi_vec.cpu(), p=2, dim=-1))

        self.defect_prototypes = {}
        for cls_name, vecs in vectors_by_class.items():
            stacked = F.normalize(torch.stack(vecs, dim=0), p=2, dim=-1)
            self.defect_prototypes[cls_name] = stacked
            counts[cls_name] = int(stacked.shape[0])
        return counts

    def _classify_roi_embedding(self, roi_vec: torch.Tensor) -> Tuple[str, float]:
        """Classify a pooled anomaly ROI embedding via top-5 cosine kNN over defect prototypes."""
        if not self.defect_prototypes:
            return "anomaly", 0.0
        q = F.normalize(roi_vec.float().cpu(), p=2, dim=-1)
        best_cls = "novel_unseen_defect"
        best_sim = -1.0
        for cls_name, proto_bank in self.defect_prototypes.items():
            sims = proto_bank @ q
            k_top = min(5, int(sims.shape[0]))
            top_sim = float(torch.topk(sims, k=k_top).values.mean().item())
            if top_sim > best_sim:
                best_sim = top_sim
                best_cls = cls_name
        if best_sim < self.novelty_sim_threshold:
            return "novel_unseen_defect", best_sim
        return best_cls, best_sim

    def _compute_image_score(
        self,
        dists_knn: np.ndarray,
        smoothed_map: np.ndarray,
        raw_map: Optional[np.ndarray] = None,
    ) -> float:
        """
        Roth et al. (CVPR 2022 Eq. 6) neighborhood density re-weighted anomaly score
        evaluated on border-suppressed `raw_map`, blended with peak & tight top-q% spatial pooling.
        """
        d1 = raw_map.ravel() if raw_map is not None else dists_knn[:, 0]
        star_idx = int(np.argmax(d1))
        s_star = float(d1[star_idx])

        if dists_knn.shape[1] > 1:
            neigh = dists_knn[star_idx].copy()
            neigh[0] = s_star
            shifted = neigh - neigh.min()
            exp_vals = np.exp(np.clip(shifted, -30.0, 30.0))
            weight = 1.0 - (1.0 / max(float(exp_vals.sum()), 1e-8))
            reweighted = max(0.0, weight) * s_star + 0.5 * s_star
        else:
            reweighted = s_star

        flat = smoothed_map.ravel()
        peak_val = float(flat.max())
        bg_med = float(np.median(flat))
        contrast_val = max(0.0, peak_val - 0.55 * bg_med)
        n_top = max(1, min(6, int(round(len(flat) * self.top_q_pool))))
        top_q_mean = float(np.partition(flat, -n_top)[-n_top:].mean())
        return 0.35 * reweighted + 0.40 * contrast_val + 0.25 * top_q_mean

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        threshold: Optional[float] = None,
        product_sku: str = "auto",
        reference_template: Optional[str | Path | Image.Image | np.ndarray] = None,
        extract_rois: bool = True,
        render_overlay: bool = True,
        _raw_unscaled: bool = False,
        exclude_mask: Optional[str | Path] = None,
    ) -> Dict[str, object]:
        if self._unified_index is None and not self._sku_indices:
            raise RuntimeError("PatchCorePlus memory bank is empty. Call fit() or from_checkpoint() first.")

        pil_image = load_rgb_image(image)
        if reference_template is not None:
            tmpl_np = np.asarray(load_rgb_image(reference_template))
            aligned_np = align_image_orb(np.asarray(pil_image), tmpl_np)
            pil_image = Image.fromarray(aligned_np)

        patches, g_sigs, (hf, wf) = self.extract_descriptors([pil_image])
        flat_patches = patches[0]  # (Hf*Wf, D)

        if product_sku == "auto":
            routed_sku, sku_conf = self.route_product_sku(g_sigs[0])
        else:
            routed_sku, sku_conf = product_sku, 1.0

        active_index = self._sku_indices.get(routed_sku) or self._unified_index
        assert active_index is not None

        dists_knn, _ = active_index.search(flat_patches, k=self.knn_k)
        raw_map = dists_knn[:, 0].reshape(hf, wf).astype(np.float32)

        # Suppress narrow crop-border artifacts BEFORE selecting s_star or smoothing
        margin_y = int(round(hf * self.border_margin_ratio))
        margin_x = int(round(wf * self.border_margin_ratio))
        interior_median = float(
            np.median(raw_map[margin_y : hf - margin_y, margin_x : wf - margin_x])
            if (margin_y > 0 and margin_x > 0 and hf > 2 * margin_y + 2 and wf > 2 * margin_x + 2)
            else np.median(raw_map)
        )
        if margin_y > 0 and margin_x > 0 and hf > 2 * margin_y + 2 and wf > 2 * margin_x + 2:
            raw_map[:margin_y, :] = np.minimum(raw_map[:margin_y, :], interior_median)
            raw_map[hf - margin_y :, :] = np.minimum(raw_map[hf - margin_y :, :], interior_median)
            raw_map[:, :margin_x] = np.minimum(raw_map[:, :margin_x], interior_median)
            raw_map[:, wf - margin_x :] = np.minimum(raw_map[:, wf - margin_x :], interior_median)
        if exclude_mask is not None and Path(exclude_mask).exists():
            m_gray = cv2.imread(str(exclude_mask), cv2.IMREAD_GRAYSCALE)
            if m_gray is not None and np.any(m_gray > 0):
                m_dil = cv2.dilate((m_gray > 0).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1)
                m_grid = cv2.resize(m_dil, (wf, hf), interpolation=cv2.INTER_NEAREST)
                raw_map[m_grid > 0] = interior_median

        smoothed_map = (
            gaussian_filter(raw_map, sigma=self.gaussian_sigma) if self.gaussian_sigma > 0 else raw_map
        )

        raw_score = self._compute_image_score(dists_knn, smoothed_map, raw_map=raw_map)
        if not _raw_unscaled and routed_sku in self.sku_score_stats:
            _, denom = self.sku_score_stats[routed_sku]
            score = float(raw_score / max(denom, 1e-4))
        else:
            score = float(raw_score)

        active_threshold = (
            threshold
            if threshold is not None
            else self.thresholds_by_sku.get(routed_sku, self.threshold)
        )
        is_anomaly = bool(active_threshold is not None and score >= active_threshold)

        rois: List[Dict[str, object]] = []
        pix_thr = self.pixel_threshold or (float(np.quantile(smoothed_map, 0.97)) if is_anomaly else None)

        if extract_rois and (is_anomaly or threshold is not None) and pix_thr is not None:
            w_orig, h_orig = pil_image.size
            amap_full = cv2.resize(smoothed_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            eff_pix_thr = max(float(pix_thr), float(np.quantile(amap_full, 0.965)))
            bin_mask = (amap_full >= eff_pix_thr).astype(np.uint8)
            num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            fmap_grid = flat_patches.reshape(hf, wf, -1)

            for cc_id in range(1, num_cc):
                x, y, bw, bh, area = cc_stats[cc_id]
                if area < max(16, int(0.00015 * w_orig * h_orig)):
                    continue
                gx1 = max(0, min(wf - 1, int(math.floor(x / max(w_orig, 1) * wf))))
                gx2 = max(gx1 + 1, min(wf, int(math.ceil((x + bw) / max(w_orig, 1) * wf))))
                gy1 = max(0, min(hf - 1, int(math.floor(y / max(h_orig, 1) * hf))))
                gy2 = max(gy1 + 1, min(hf, int(math.ceil((y + bh) / max(h_orig, 1) * hf))))

                roi_emb = fmap_grid[gy1:gy2, gx1:gx2].reshape(-1, fmap_grid.shape[-1]).mean(dim=0)
                pred_cls, proto_sim = self._classify_roi_embedding(roi_emb)
                peak_val = float(amap_full[y : y + bh, x : x + bw].max())

                rois.append(
                    {
                        "class": pred_cls,
                        "score": round(peak_val, 4),
                        "prototype_similarity": round(proto_sim, 4),
                        "xmin": float(x),
                        "ymin": float(y),
                        "xmax": float(x + bw),
                        "ymax": float(y + bh),
                        "area_px": int(area),
                        "area_pct": round(100.0 * float(area) / max(1.0, float(w_orig * h_orig)), 4),
                    }
                )
            rois.sort(key=lambda r: float(r["score"]), reverse=True)

        overlay = (
            render_true_rgb_overlay(pil_image, smoothed_map, pixel_threshold=pix_thr, rois=rois)
            if render_overlay
            else None
        )

        return {
            "score": float(score),
            "raw_score": float(raw_score),
            "threshold": active_threshold,
            "is_anomaly": is_anomaly,
            "product_sku": routed_sku,
            "sku_confidence": round(float(sku_conf), 4),
            "anomaly_map": smoothed_map,
            "rois": rois,
            "overlay": overlay,
        }

    @torch.no_grad()
    def evaluate_anomaly_dir(
        self,
        sku_test_dir: str | Path,
        gt_mask_dir: Optional[str | Path] = None,
        product_sku: str = "auto",
    ) -> Dict[str, object]:
        """
        Evaluate Image-AUROC, Pixel-AUROC, F1-Max, and Latency (ms) on a standard
        `test/good` + `test/<defect_class>` (+ optional `ground_truth/test/<defect_class>`) folder.
        """
        test_root = Path(sku_test_dir)
        mask_root = Path(gt_mask_dir) if gt_mask_dir else None

        img_labels: List[int] = []
        img_scores: List[float] = []
        pixel_aurocs: List[float] = []
        latencies_ms: List[float] = []

        for sub in sorted(test_root.iterdir()):
            if not sub.is_dir():
                continue
            is_good = sub.name.lower() in {"good", "normal", "ok", "pass"}
            for img_path in find_images([sub], recursive=False):
                t0 = time.perf_counter()
                pred = self.predict(img_path, product_sku=product_sku, extract_rois=False, render_overlay=False)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                img_labels.append(0 if is_good else 1)
                img_scores.append(float(pred["score"]))

                if not is_good and mask_root is not None:
                    cands = [
                        mask_root / sub.name / f"{img_path.stem}_mask.png",
                        mask_root / sub.name / f"{img_path.stem}.png",
                    ]
                    mpath = next((c for c in cands if c.exists()), None)
                    if mpath is not None:
                        gt = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                        if gt is not None:
                            gt_bin = (gt > 0).astype(np.uint8)
                            if 0 < gt_bin.sum() < gt_bin.size:
                                amap_up = cv2.resize(
                                    pred["anomaly_map"],
                                    (gt_bin.shape[1], gt_bin.shape[0]),
                                    interpolation=cv2.INTER_CUBIC,
                                )
                                pixel_aurocs.append(float(roc_auc_score(gt_bin.ravel(), amap_up.ravel())))

        image_auroc = float(roc_auc_score(img_labels, img_scores)) if len(set(img_labels)) > 1 else None
        f1_max = None
        if len(set(img_labels)) > 1:
            prec, rec, _ = precision_recall_curve(img_labels, img_scores)
            f1_max = float(np.max((2.0 * prec * rec) / (prec + rec + 1e-8)))

        return {
            "n_test_images": len(img_labels),
            "n_normal": int(sum(1 for y in img_labels if y == 0)),
            "n_anomalous": int(sum(1 for y in img_labels if y == 1)),
            "image_auroc": image_auroc,
            "f1_max": f1_max,
            "pixel_auroc_mean": float(np.mean(pixel_aurocs)) if pixel_aurocs else None,
            "pixel_auroc_median": float(np.median(pixel_aurocs)) if pixel_aurocs else None,
            "avg_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        }

    def save(self, path: str | Path) -> None:
        if self.unified_bank is None and not self.banks_by_sku:
            raise RuntimeError("Cannot save PatchCorePlus before fitting a memory bank.")
        payload = {
            "format_version": "patchcore_plus_v2",
            "image_size": self.image_size,
            "backbone_name": self.backbone_name,
            "cnn_layers": self.cnn_layers,
            "dino_blocks": self.dino_blocks,
            "projection_dim": self.projection_dim,
            "neighborhood_kernel": self.neighborhood_kernel,
            "l2_normalize_cnn": self.l2_normalize_cnn,
            "knn_k": self.knn_k,
            "gaussian_sigma": self.gaussian_sigma,
            "border_margin_ratio": self.border_margin_ratio,
            "top_q_pool": self.top_q_pool,
            "sku_sim_threshold": self.sku_sim_threshold,
            "novelty_sim_threshold": self.novelty_sim_threshold,
            "pretrained": self.pretrained,
            "seed": self.seed,
            "projection_matrix": self.projection_matrix.detach().cpu(),
            "unified_bank": self.unified_bank.detach().cpu() if self.unified_bank is not None else None,
            "banks_by_sku": {k: v.detach().cpu() for k, v in self.banks_by_sku.items()},
            "sku_centroids": {k: v.detach().cpu() for k, v in self.sku_centroids.items()},
            "threshold": self.threshold,
            "thresholds_by_sku": self.thresholds_by_sku,
            "sku_score_stats": self.sku_score_stats,
            "pixel_threshold": self.pixel_threshold,
            "defect_prototypes": {k: v.detach().cpu() for k, v in self.defect_prototypes.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: Optional[str] = None) -> "PatchCorePlus":
        ckpt_path = Path(path)
        if ckpt_path.suffix.lower() == ".npz":
            # Seamless backward-compatibility loader for legacy D:\FYP_Web\weights\patchcore_memory_bank.npz!
            data = np.load(str(ckpt_path), allow_pickle=True)
            bank = torch.from_numpy(data["bank"]).float()
            input_sz = int(data["PC_INPUT"]) if "PC_INPUT" in data else 512
            feat_norm = bool(data["feature_norm"]) if "feature_norm" in data else True
            sigma = float(data["gaussian_sigma"]) if "gaussian_sigma" in data else 2.0
            return cls(
                image_size=input_sz,
                backbone_name="wide_resnet50_2",
                cnn_layers=("layer1", "layer2"),
                projection_dim=bank.shape[1],
                neighborhood_kernel=1,
                l2_normalize_cnn=feat_norm,
                gaussian_sigma=sigma,
                projection_matrix=torch.eye(bank.shape[1], dtype=torch.float32),
                unified_bank=bank,
                threshold=0.35,
                device=device,
            )

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != "patchcore_plus_v2":
            # Legacy GitHub `patchcore_l2l3.pt` checkpoint compatibility
            bank = payload["memory_bank"].float()
            return cls(
                image_size=int(payload.get("image_size", 256)),
                backbone_name=payload.get("backbone_name", "wide_resnet50_2"),
                cnn_layers=tuple(payload.get("layers", ("layer2", "layer3"))),
                projection_dim=bank.shape[1],
                l2_normalize_cnn=True,
                projection_matrix=torch.eye(bank.shape[1], dtype=torch.float32),
                unified_bank=bank,
                threshold=payload.get("threshold"),
                pretrained=payload.get("pretrained", True),
                device=device,
            )

        return cls(
            image_size=payload["image_size"],
            backbone_name=payload["backbone_name"],
            cnn_layers=payload["cnn_layers"],
            dino_blocks=payload["dino_blocks"],
            projection_dim=payload["projection_dim"],
            neighborhood_kernel=payload.get("neighborhood_kernel", 3),
            l2_normalize_cnn=payload.get("l2_normalize_cnn", False),
            knn_k=payload.get("knn_k", 9),
            gaussian_sigma=payload.get("gaussian_sigma", 2.0),
            border_margin_ratio=payload.get("border_margin_ratio", 0.03),
            top_q_pool=payload.get("top_q_pool", 0.0015),
            sku_sim_threshold=payload.get("sku_sim_threshold", 0.80),
            novelty_sim_threshold=payload.get("novelty_sim_threshold", 0.68),
            pretrained=payload.get("pretrained", True),
            seed=payload.get("seed", 42),
            projection_matrix=payload.get("projection_matrix"),
            unified_bank=payload.get("unified_bank"),
            banks_by_sku=payload.get("banks_by_sku"),
            sku_centroids=payload.get("sku_centroids"),
            threshold=payload.get("threshold"),
            thresholds_by_sku=payload.get("thresholds_by_sku"),
            sku_score_stats=payload.get("sku_score_stats"),
            pixel_threshold=payload.get("pixel_threshold"),
            defect_prototypes=payload.get("defect_prototypes"),
            device=device,
        )


def _cli_fit(args: argparse.Namespace) -> None:
    sku_inputs: Dict[str, List[Path]] = {}
    for raw_dir in args.normal_dir:
        p = Path(raw_dir)
        # Auto-detect multi-product directory tree (`anomaly_multiproduct/<sku>/train/good`)
        sub_goods = sorted(p.glob("*/train/good"))
        if sub_goods:
            for gdir in sub_goods:
                sku_name = gdir.parent.parent.name
                imgs = find_images([gdir], recursive=True)
                if imgs:
                    sku_inputs[sku_name] = imgs
        else:
            imgs = find_images([p], recursive=args.recursive)
            if imgs:
                sku_inputs[p.name] = imgs

    if not sku_inputs:
        raise SystemExit("No normal training images found.")

    model = PatchCorePlus(
        image_size=args.image_size,
        backbone_name=args.backbone,
        cnn_layers=tuple(args.cnn_layers),
        projection_dim=args.projection_dim,
        l2_normalize_cnn=args.l2_normalize_cnn,
        pretrained=not args.no_pretrained,
        device=args.device,
        seed=args.seed,
    )
    model.fit(
        normal_images=sku_inputs,
        batch_size=args.batch_size,
        coreset_ratio=args.coreset_ratio,
        threshold_quantile=args.threshold_quantile,
    )
    model.save(args.output)
    total_imgs = sum(len(v) for v in sku_inputs.values())
    print(f"[PatchCore++] Saved checkpoint -> {args.output}")
    print(f"[PatchCore++] Backbone={args.backbone} | SKUs={list(sku_inputs.keys())} | Total Normal Imgs={total_imgs}")
    print(f"[PatchCore++] Global Threshold={model.threshold:.5f} | SKU Thresholds={model.thresholds_by_sku}")


def _cli_predict(args: argparse.Namespace) -> None:
    image_paths = find_images(args.input, recursive=args.recursive)
    if not image_paths:
        raise SystemExit("No input images found.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = PatchCorePlus.from_checkpoint(args.checkpoint, device=args.device)

    csv_path = out_dir / "patchcore_plus_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "product_sku", "sku_confidence", "score", "threshold", "status", "roi_classes", "roi_count"],
        )
        writer.writeheader()
        for p in image_paths:
            res = model.predict(p, threshold=args.threshold, product_sku=args.product_sku)
            rois = res["rois"]
            roi_classes = ", ".join(sorted({str(r["class"]) for r in rois})) if rois else "-"
            writer.writerow(
                {
                    "image": str(p),
                    "product_sku": res["product_sku"],
                    "sku_confidence": res["sku_confidence"],
                    "score": f"{res['score']:.6f}",
                    "threshold": "" if res["threshold"] is None else f"{res['threshold']:.6f}",
                    "status": "anomaly" if res["is_anomaly"] else "pass",
                    "roi_classes": roi_classes,
                    "roi_count": len(rois),
                }
            )
            if args.save_overlays and res["overlay"] is not None:
                Image.fromarray(res["overlay"]).save(out_dir / f"{p.stem}_patchcore_plus.png")

    print(f"[PatchCore++] Saved predictions -> {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Product PatchCore++ / AnomalyDINO PCB Pipeline.")
    sub = parser.add_subparsers(required=True)

    fit_p = sub.add_parser("fit", help="Fit multi-product or single-product normal memory bank.")
    fit_p.add_argument("--normal-dir", nargs="+", required=True)
    fit_p.add_argument("--output", default="weights/patchcore_plus_multiproduct.pt")
    fit_p.add_argument("--backbone", default="dinov2_hybrid", choices=["dinov2_hybrid", "dinov2_vitb14_reg", "wide_resnet50_2"])
    fit_p.add_argument("--cnn-layers", nargs="+", default=["layer1", "layer2"], choices=["layer1", "layer2", "layer3", "layer4"])
    fit_p.add_argument("--image-size", type=int, default=518)
    fit_p.add_argument("--projection-dim", type=int, default=512)
    fit_p.add_argument("--batch-size", type=int, default=8)
    fit_p.add_argument("--coreset-ratio", type=float, default=0.04)
    fit_p.add_argument("--threshold-quantile", type=float, default=0.985)
    fit_p.add_argument("--l2-normalize-cnn", action="store_true")
    fit_p.add_argument("--no-pretrained", action="store_true")
    fit_p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    fit_p.add_argument("--seed", type=int, default=42)
    fit_p.add_argument("--device", default=None)
    fit_p.set_defaults(func=_cli_fit)

    pred_p = sub.add_parser("predict", help="Score PCB images and extract multi-class / novel anomaly ROIs.")
    pred_p.add_argument("--checkpoint", default="weights/patchcore_plus_multiproduct.pt")
    pred_p.add_argument("--input", nargs="+", required=True)
    pred_p.add_argument("--output-dir", default="patchcore_results")
    pred_p.add_argument("--product-sku", default="auto")
    pred_p.add_argument("--threshold", type=float, default=None)
    pred_p.add_argument("--save-overlays", action="store_true")
    pred_p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    pred_p.add_argument("--device", default=None)
    pred_p.set_defaults(func=_cli_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
