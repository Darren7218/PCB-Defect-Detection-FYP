"""
Google Colab Pro (200 Compute Units/month) & Kaggle Free Tier Guardrails.

Protects Colab Compute Units via four automated mechanisms:
  1. Hardware & CU Burn-Rate Telemetry (flags expensive A100 attachment when L4 24GB suffices).
  2. Fast Local NVMe Staging (/content/datasets) from tar/zip archives or Kaggle Datasets so
     training never bottlenecks on slow Google Drive FUSE file I/O.
  3. Incremental Checkpoint & Metrics Mirroring to Google Drive (/content/drive/MyDrive/...).
  4. Mandatory `google.colab.runtime.unassign()` in `finally:` so finished or failed jobs
     immediately release the cloud GPU VM (zero overnight idle credit burn).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


# Approximate Colab Compute Unit hourly rates for budget telemetry
APPROX_CU_PER_HOUR: Dict[str, float] = {
    "A100": 13.0,
    "L4": 4.5,
    "V100": 5.0,
    "T4": 1.9,
    "P100": 0.0,  # Kaggle Free
    "CPU": 0.1,
}


def detect_cloud_environment() -> str:
    """Return 'colab', 'kaggle', or 'local'."""
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists() and not Path("/kaggle").exists():
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def detect_gpu_name() -> str:
    """Detect attached NVIDIA GPU model string without requiring torch import first."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "CPU"


def estimate_cu_rate(gpu_name: str, env: Optional[str] = None) -> float:
    """Estimate Colab Compute Units burned per hour."""
    active_env = env or detect_cloud_environment()
    if active_env != "colab":
        return 0.0
    upper = gpu_name.upper()
    for key, rate in APPROX_CU_PER_HOUR.items():
        if key in upper:
            return rate
    return 2.0


def mount_google_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive when running inside Google Colab."""
    if detect_cloud_environment() != "colab":
        return None
    mp = Path(mount_point)
    if (mp / "MyDrive").exists():
        return mp / "MyDrive"
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point, force_remount=False)
        return mp / "MyDrive"
    except Exception as exc:
        print(f"[ColabGuard] Warning: Google Drive mount skipped ({exc})")
        return None


def stage_archive_to_local_ssd(
    archive_path: str | Path,
    dest_dir: str | Path = "/content/datasets",
    marker_name: str = ".staged_ok",
) -> Path:
    """
    Extract a .tar / .tar.gz / .zip bundle from Google Drive onto local NVMe SSD once.
    Eliminates 5x-10x training slowdown from reading thousands of small images over Drive FUSE.
    """
    archive = Path(archive_path)
    dest = Path(dest_dir) / archive.stem.replace(".tar", "")
    marker = dest / marker_name
    if marker.exists():
        print(f"[ColabGuard] Local SSD cache hit: {dest}")
        return dest

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found for SSD staging: {archive}")

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[ColabGuard] Staging {archive.name} -> {dest} (local NVMe SSD)...")

    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif ".tar" in suffixes or suffixes.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(dest, filter="data")
            else:
                tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")

    marker.write_text(f"staged_from={archive}\nelapsed_s={time.perf_counter() - t0:.1f}\n", encoding="utf-8")
    print(f"[ColabGuard] Staged in {time.perf_counter() - t0:.1f}s -> {dest}")
    return dest


def stage_kaggle_dataset_to_ssd(
    dataset_slug: str,
    dest_dir: str | Path = "/content/datasets",
) -> Path:
    """
    Pull a Kaggle Dataset directly onto Colab's local NVMe SSD at ~150-250 MB/s.
    Requires KAGGLE_USERNAME/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json.
    """
    slug_safe = dataset_slug.replace("/", "__")
    dest = Path(dest_dir) / slug_safe
    marker = dest / ".staged_ok"
    if marker.exists():
        print(f"[ColabGuard] Kaggle dataset already staged on SSD: {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[ColabGuard] Downloading Kaggle dataset '{dataset_slug}' -> {dest}...")
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Kaggle dataset download failed ({dataset_slug}): {res.stderr or res.stdout}")
    marker.write_text(f"slug={dataset_slug}\nelapsed_s={time.perf_counter() - t0:.1f}\n", encoding="utf-8")
    print(f"[ColabGuard] Kaggle dataset ready in {time.perf_counter() - t0:.1f}s -> {dest}")
    return dest


def sync_artifacts_to_drive(
    sources: Iterable[str | Path],
    drive_dest_dir: str | Path,
) -> list[Path]:
    """Copy checkpoints, FAISS banks, and JSON reports to Google Drive."""
    dst_root = Path(drive_dest_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for raw_src in sources:
        src = Path(raw_src)
        if not src.exists():
            continue
        target = dst_root / src.name
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
        copied.append(target)
        print(f"[ColabGuard] Synced -> {target}")
    return copied


@dataclass
class ColabCreditGuard:
    """
    Context manager wrapping high-resource Colab Pro L4 runs.

    Example:
        with ColabCreditGuard(
            job_name="yolov11m_unified_10cls",
            drive_sync_dir="/content/drive/MyDrive/PCB_FYP_Experiments/yolov11m",
            max_cu_budget=25.0,
            auto_unassign=True,
        ) as guard:
            ...
            guard.register_artifact("runs/detect/train/weights/best.pt")
    """

    job_name: str
    drive_sync_dir: Optional[str | Path] = None
    max_cu_budget: float = 30.0
    recommend_l4_over_a100: bool = True
    auto_unassign: bool = True
    grace_seconds: int = 8
    tracked_artifacts: list[Path] = field(default_factory=list)
    _start_time: float = field(default=0.0, init=False)
    _env: str = field(default="local", init=False)
    _gpu: str = field(default="CPU", init=False)
    _cu_rate: float = field(default=0.0, init=False)

    def register_artifact(self, path: str | Path) -> None:
        self.tracked_artifacts.append(Path(path))

    def elapsed_hours(self) -> float:
        return max(0.0, (time.perf_counter() - self._start_time) / 3600.0)

    def estimated_cu_used(self) -> float:
        return self.elapsed_hours() * self._cu_rate

    def checkpoint_now(self, extra_paths: Iterable[str | Path] = ()) -> None:
        if self.drive_sync_dir is None:
            return
        all_paths = [*self.tracked_artifacts, *(Path(p) for p in extra_paths)]
        sync_artifacts_to_drive(all_paths, self.drive_sync_dir)

    def __enter__(self) -> "ColabCreditGuard":
        self._start_time = time.perf_counter()
        self._env = detect_cloud_environment()
        self._gpu = detect_gpu_name()
        self._cu_rate = estimate_cu_rate(self._gpu, self._env)

        print("=" * 72)
        print(f"[ColabGuard] Job       : {self.job_name}")
        print(f"[ColabGuard] Env / GPU : {self._env.upper()} | {self._gpu}")
        print(f"[ColabGuard] Burn Rate : ~{self._cu_rate:.2f} CU/hr (Max job budget: {self.max_cu_budget:.1f} CU)")
        if self._env == "colab" and self.recommend_l4_over_a100 and "A100" in self._gpu.upper():
            print(
                "[ColabGuard] NOTICE: A100 (~13 CU/hr) is attached. "
                "L4 (24GB, ~4.5 CU/hr) delivers ~2.5-3x more epochs per Compute Unit for 640px PCB models!"
            )
        if self._env == "colab" and self.drive_sync_dir is not None:
            mount_google_drive()
        print("=" * 72)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_h = self.elapsed_hours()
        cu_used = self.estimated_cu_used()
        status = "SUCCESS" if exc_type is None else f"ERROR ({exc_type.__name__}: {exc_val})"

        telemetry = {
            "job_name": self.job_name,
            "environment": self._env,
            "gpu": self._gpu,
            "status": status,
            "elapsed_minutes": round(elapsed_h * 60.0, 2),
            "estimated_cu_rate_per_hr": self._cu_rate,
            "estimated_cu_consumed": round(cu_used, 3),
            "timestamp_epoch": time.time(),
        }

        telemetry_file = Path(f"{self.job_name}_telemetry.json")
        telemetry_file.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
        self.register_artifact(telemetry_file)

        try:
            self.checkpoint_now()
        except Exception as sync_exc:
            print(f"[ColabGuard] Warning during final Drive sync: {sync_exc}")

        print("=" * 72)
        print(
            f"[ColabGuard] Finished '{self.job_name}' [{status}] | "
            f"Time: {elapsed_h * 60:.1f} min | Est. Colab Credits Used: {cu_used:.2f} CU"
        )
        print("=" * 72)

        if self._env == "colab" and self.auto_unassign:
            print(
                f"[ColabGuard] Auto-unassigning Colab VM in {self.grace_seconds}s "
                "to prevent idle Compute Unit drain..."
            )
            time.sleep(self.grace_seconds)
            try:
                from google.colab import runtime  # type: ignore

                runtime.unassign()
            except Exception as unassign_exc:
                print(f"[ColabGuard] Could not call runtime.unassign(): {unassign_exc}")

        return False
