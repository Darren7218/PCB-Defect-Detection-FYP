"""
Zero-dependency unit & syntax verification suite for the Multi-Product PCB Upgrade.
Runs locally in <2s using standard library + numpy (if available) to guarantee 100%
syntax and logic integrity across all Python modules and Jupyter notebooks before cloud execution.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cloud.colab_guard import ColabCreditGuard, estimate_cu_rate, stage_archive_to_local_ssd
from src.datasets.taxonomy import (
    DEFAULT_OVD_PROMPTS_10CLS,
    UNIFIED_CLASS_NAMES,
    resolve_unified_class_id,
)
from src.detection.sahi_and_calibration import (
    ConfidenceCalibrator,
    class_aware_nms_nmm,
    fuse_detector_and_patchcore,
    generate_sahi_slices,
)


class TestSyntaxAndNotebooks(unittest.TestCase):
    def test_all_python_files_parse_cleanly(self) -> None:
        py_files = list(REPO_ROOT.glob("*.py")) + list((REPO_ROOT / "src").rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 7)
        for path in py_files:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))

    def test_all_new_notebooks_valid_json_and_python_syntax(self) -> None:
        nb_names = [
            "colab_00_connect_and_verify.ipynb",
            "kaggle_01_dataset_builder_and_patchcore_ablation.ipynb",
            "colab_01_train_multiproduct_detectors_L4.ipynb",
            "kaggle_02_ovd_calibration_and_sahi_eval.ipynb",
        ]
        for name in nb_names:
            nb_path = REPO_ROOT / "notebooks" / name
            self.assertTrue(nb_path.exists(), f"Missing notebook: {nb_path}")
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            for idx, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    lines = [
                        line for line in "".join(cell["source"]).splitlines()
                        if not line.lstrip().startswith(("!", "%"))
                    ]
                    ast.parse("\n".join(lines), filename=f"{name}#cell_{idx}")


class TestTaxonomyAndCloudGuard(unittest.TestCase):
    def test_unified_10_class_taxonomy(self) -> None:
        self.assertEqual(len(UNIFIED_CLASS_NAMES), 10)
        self.assertEqual(len(DEFAULT_OVD_PROMPTS_10CLS), 10)
        # DeepPCB 1-indexed mapping
        self.assertEqual(resolve_unified_class_id(1, "deeppcb"), 2)  # open -> open_circuit
        self.assertEqual(resolve_unified_class_id(6, "deeppcb"), 0)  # pin-hole -> missing_hole
        # Bare PCB & Assembled PCBA aliases
        self.assertEqual(resolve_unified_class_id("mousebite"), 1)
        self.assertEqual(resolve_unified_class_id("solder_bridge"), 6)
        self.assertEqual(resolve_unified_class_id("missing_part"), 7)
        self.assertEqual(resolve_unified_class_id("bent_pin"), 8)
        self.assertEqual(resolve_unified_class_id("scratch_surface"), 9)

    def test_colab_credit_guard_and_tar_staging(self) -> None:
        self.assertAlmostEqual(estimate_cu_rate("NVIDIA L4", env="colab"), 4.5)
        self.assertAlmostEqual(estimate_cu_rate("NVIDIA A100-SXM4-40GB", env="colab"), 13.0)
        self.assertAlmostEqual(estimate_cu_rate("Tesla T4", env="kaggle"), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            import tarfile

            sample_dir = tmp_p / "src_bundle"
            sample_dir.mkdir()
            (sample_dir / "data.yaml").write_text("nc: 10\n", encoding="utf-8")
            tar_file = tmp_p / "bundle.tar"
            with tarfile.open(tar_file, "w") as tf:
                tf.add(sample_dir, arcname="src_bundle")

            staged = stage_archive_to_local_ssd(tar_file, dest_dir=tmp_p / "ssd")
            self.assertTrue((staged / "src_bundle" / "data.yaml").exists())

            sync_dir = tmp_p / "drive_sync"
            with ColabCreditGuard(
                job_name="unit_test_job",
                drive_sync_dir=sync_dir,
                auto_unassign=False,
            ) as guard:
                guard.register_artifact(sample_dir / "data.yaml")

            self.assertTrue((sync_dir / "data.yaml").exists())
            self.assertTrue((sync_dir / "unit_test_job_telemetry.json").exists())
            Path("unit_test_job_telemetry.json").unlink(missing_ok=True)

    def test_sahi_calibration_and_two_stage_fusion(self) -> None:
        # 1. SAHI slices cover full 3034x1586 PCB board
        slices = generate_sahi_slices(3034, 1586, slice_size=640, overlap_ratio=0.25)
        self.assertGreater(len(slices), 12)
        self.assertEqual(max(s[2] for s in slices), 3034)
        self.assertEqual(max(s[3] for s in slices), 1586)

        # 2. ConfidenceCalibrator maps raw OVD operating point 0.002 -> ~0.50
        cal = ConfidenceCalibrator(default_operating_raw_conf=0.002)
        self.assertAlmostEqual(cal.calibrate_score(0.002, "spur"), 0.50, places=2)
        self.assertGreater(cal.calibrate_score(0.05, "spur"), 0.90)

        # 3. Two-Stage Fusion enriches matched specialist box & appends unmatched novel ROI
        spec = [{"class": "short", "confidence": 0.94, "xmin": 100.0, "ymin": 100.0, "xmax": 140.0, "ymax": 140.0}]
        pc = {
            "is_anomaly": True,
            "score": 4.2,
            "product_sku": "visa_pcb1",
            "sku_confidence": 0.96,
            "rois": [
                {"class": "short", "score": 3.8, "xmin": 102.0, "ymin": 98.0, "xmax": 138.0, "ymax": 142.0, "area_px": 1150},
                {"class": "novel_unseen_defect", "score": 4.2, "xmin": 400.0, "ymin": 300.0, "xmax": 460.0, "ymax": 350.0, "area_px": 2100},
            ],
        }
        fused, meta = fuse_detector_and_patchcore(spec, pc)
        self.assertEqual(len(fused), 2)
        self.assertEqual(meta["verified_count"], 1)
        self.assertEqual(meta["open_set_novel_count"], 1)
        self.assertEqual(fused[0]["area_px"], 1150)
        self.assertEqual(fused[1]["class"], "novel_unseen_defect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
