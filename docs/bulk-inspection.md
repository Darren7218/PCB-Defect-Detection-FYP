# Bulk Inspection UI Update

This update refactors `app.py` into a Streamlit inspection console with:

- Single-image inspection
- Bulk inspection from multiple uploaded images
- Bulk inspection from a local folder path, with optional recursive scanning
- Model controls for YOLOv9 or RT-DETR
- Confidence threshold tuning
- Inference image-size tuning
- Per-image status, defect count, classes, max confidence, and inference time
- Downloadable summary CSV
- Downloadable per-detection CSV
- Defect preview thumbnails for failed boards

## Apply

Replace the repository root `app.py` with `outputs/app.py`.

Add `requirements.txt` from `outputs/requirements.txt` to the repository root.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `yolov9_best.pt` and `rtdetr_best.pt` in the repository root before launching.

## Bulk Folder Inspection

Open the `Bulk inspection` tab, choose `Local folder path`, enter the image folder on the machine running Streamlit, and run inspection. Supported image types are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
