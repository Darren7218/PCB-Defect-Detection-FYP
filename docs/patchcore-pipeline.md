# PatchCore++ (`DINOv2-reg + WRN50` Hybrid) Multi-Product Pipeline

`src/anomaly/patchcore_plus.py` (and backward-compatible wrapper `patchcore_l2l3.py`) upgrades the baseline PatchCore branch into a **Multi-Product Anomaly Localization & Few-Shot Defect Classification Engine**.

## Key Architectural Upgrades Over Baseline `PatchCoreL2L3`

1. **Hybrid Foundation Backbone (`dinov2_hybrid` @ `518x518`)**: Concatenates frozen **`DINOv2-ViT-B/14-reg`** intermediate blocks `(5, 8, 11)` (structural layout without register background artifacts) with **`WideResNet50` `layer1 + layer2`** (`64x64` high-frequency copper trace edges), projected via an orthogonal matrix to `512-d` descriptors.
2. **DINOv2 Global CLS/GeM Product-SKU Router**: Identifies which PCB product (`visa_pcb1..4`, `pku_pcb_01..12`, `deeppcb_*`, etc.) is under inspection in `<1 ms` and routes to its dedicated memory bank (falling back to a unified multi-product coreset for unseen boards).
3. **True GPU k-Center Greedy Coreset + Cached FAISS Index**: Preserves rare normal landmarks (fiducials, mounting holes, IC silkscreen corners) that random subsampling dropped, while accelerating kNN lookup by `>25x`.
4. **Density-Reweighted `k=9` Scoring + Crop-Border Suppression**: Eliminates single-pixel border spikes and oblique crop-edge false positives.
5. **Connected-Component ROI & Few-Shot Defect Prototype Head**: Extracts tight bounding boxes + pixel defect areas (`px²`) and classifies each anomaly into the 10 unified PCB/PCBA defect classes or `novel_unseen_defect`.
6. **True-RGB Turbo/Jet Overlay**: Fixes the BGR-on-RGB channel inversion bug.

## Fit Multi-Product Memory Bank (CLI)

```bash
python -m src.anomaly.patchcore_plus fit \
  --normal-dir datasets/anomaly_multiproduct \
  --output weights/patchcore_plus_multiproduct.pt \
  --backbone dinov2_hybrid \
  --image-size 518 \
  --coreset-ratio 0.05
```

## Score & Extract Defect ROIs (CLI)

```bash
python -m src.anomaly.patchcore_plus predict \
  --checkpoint weights/patchcore_plus_multiproduct.pt \
  --input data/inspection_batch \
  --product-sku auto \
  --output-dir patchcore_results \
  --save-overlays
```
