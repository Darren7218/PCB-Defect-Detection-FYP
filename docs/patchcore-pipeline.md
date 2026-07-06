# PatchCore L2/L3 Pipeline

This adds an unsupervised PatchCore path for cases where the detector misses novel or rare PCB defects. It trains from normal/good boards only, stores a memory bank, and scores new boards by nearest-neighbor distance in feature space.

## Why Layer2 + Layer3

Use `layer2` and `layer3` by default. `layer2` preserves local texture and small geometry cues that matter for PCB defects. `layer3` adds stronger context without becoming as spatially coarse as `layer4`. For small industrial defects, `layer4` often reduces localization quality and increases compute, so it is intentionally excluded from the default pipeline.

## Fit A Memory Bank

Train on good/normal boards only:

```bash
python patchcore_l2l3.py fit ^
  --normal-dir data/pcb_normal ^
  --output patchcore_l2l3.pt ^
  --layers layer2 layer3 ^
  --image-size 256 ^
  --batch-size 8 ^
  --coreset-ratio 0.1 ^
  --threshold-quantile 0.995
```

The output `patchcore_l2l3.pt` should be placed next to `app.py` for Streamlit use.

## Score A Folder

```bash
python patchcore_l2l3.py predict ^
  --checkpoint patchcore_l2l3.pt ^
  --input data/inspection_batch ^
  --output-dir patchcore_results ^
  --save-overlays
```

This writes:

- `patchcore_results.csv` with per-image anomaly scores
- optional red heatmap overlays for visual review

## Streamlit Integration

Copy `patchcore_l2l3.py` and `patchcore_l2l3.pt` next to `app.py`. The app then exposes `PatchCore L2/L3 (Unsupervised Anomaly)` as a third model in the sidebar.

PatchCore does not produce class labels or bounding boxes. The app reports one `anomaly` finding when the image score is above the threshold, and the preview image shows a red heatmap overlay.
