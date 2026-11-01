# 3-Tier Compute Workflow: Local + Kaggle Free (`0 CU`) + Google Colab Pro (`200 CU/mo`)

## 1. Connect Google Colab Pro Inside Antigravity IDE

1. Open this repository folder as your active workspace in Antigravity IDE (`File -> Open Folder`).
2. Press `Ctrl+Shift+X` and install:
   - **Jupyter** (`ms-toolsai.jupyter`)
   - **Google Colab** (`Google.colab`)
3. Open `notebooks/colab_00_connect_and_verify.ipynb`, click **Select Kernel -> Colab**, sign in with your Google Colab Pro account, and run all 3 cells on a **CPU or T4** runtime to verify `/content/drive/MyDrive/PCB_FYP_Experiments` and `~/.kaggle/kaggle.json`.

## 2. Execution Order Across Compute Tiers (Maximizing Your 200 Credits)

| Step | Notebook | Recommended Platform | Est. Colab Credits | Output Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | `notebooks/kaggle_01_dataset_builder_and_patchcore_ablation.ipynb` | **Kaggle Free (`GPU T4 x2`)** | **`0.0 CU`** | `patchcore_plus_multiproduct.pt`, `patchcore_ablation_metrics.json`, `unified_pcb_10cls/` |
| **Step 2** | `notebooks/colab_01_train_multiproduct_detectors_L4.ipynb` | **Google Colab Pro (`L4 24GB`)** | **`~18–26 CU` / model** (`~65 CU` for all 3) | `yolo11m_multiproduct_best.pt`, `yolov9_multiproduct_best.pt`, `rtdetr_multiproduct_best.pt` |
| **Step 3** | `notebooks/kaggle_02_ovd_calibration_and_sahi_eval.ipynb` | **Kaggle Free (`GPU T4`)** | **`0.0 CU`** | `ovd_calibrator.json` |
| **Step 4** | `streamlit run app.py` | **Local PC (`RTX 3050`)** | **`0.0 CU`** | Unified Single & Bulk Inspection Web Console |

## 3. Built-In Colab Credit Protections (`src/cloud/colab_guard.py`)

- **Local NVMe SSD Staging**: Extracts dataset tarballs or pulls Kaggle datasets directly into `/content/datasets/` so GPU utilization stays `>90%` instead of stalling on Google Drive FUSE reads.
- **10-Epoch Drive Checkpointing**: Copies `best.pt` to `/content/drive/MyDrive/PCB_FYP_Experiments/weights/` every 10 epochs.
- **Automatic VM Shutoff**: Calls `google.colab.runtime.unassign()` inside `finally:` immediately upon completion or error so an unattended overnight run never burns idle Compute Units.
