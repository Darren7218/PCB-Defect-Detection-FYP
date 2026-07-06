import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LAYERS = ("layer2", "layer3")


def find_images(paths, recursive=True):
    images = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images.extend(
                candidate
                for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return sorted(images)


def load_rgb_image(path_or_image):
    if isinstance(path_or_image, Image.Image):
        image = path_or_image
    else:
        image = Image.open(path_or_image)
    return ImageOps.exif_transpose(image).convert("RGB")


def overlay_anomaly_map(image, anomaly_map, alpha=0.45):
    image = load_rgb_image(image)
    heatmap = anomaly_map.astype(np.float32)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(
        image.size, Image.Resampling.BILINEAR
    )
    heatmap_np = np.asarray(heatmap_image).astype(np.float32) / 255.0
    image_np = np.asarray(image).astype(np.float32)

    red_overlay = np.zeros_like(image_np)
    red_overlay[..., 0] = 255.0
    blended = image_np * (1.0 - alpha * heatmap_np[..., None]) + red_overlay * (
        alpha * heatmap_np[..., None]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


class PatchCoreL2L3:
    def __init__(
        self,
        image_size=256,
        layers=DEFAULT_LAYERS,
        memory_bank=None,
        threshold=None,
        backbone_name="wide_resnet50_2",
        pretrained=True,
        device=None,
    ):
        self.image_size = int(image_size)
        self.layers = tuple(layers)
        self.threshold = threshold
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.memory_bank = memory_bank

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.extractor = self._build_extractor().to(self.device).eval()
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.to(self.device)

    def _build_extractor(self):
        if self.backbone_name != "wide_resnet50_2":
            raise ValueError("Only wide_resnet50_2 is supported by this lightweight pipeline.")

        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if self.pretrained else None
        backbone = models.wide_resnet50_2(weights=weights)
        return create_feature_extractor(backbone, return_nodes={layer: layer for layer in self.layers})

    def _preprocess_batch(self, images):
        tensors = [self.transform(load_rgb_image(image)) for image in images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def extract_embeddings(self, images):
        batch = self._preprocess_batch(images)
        feature_maps = self.extractor(batch)
        target_size = feature_maps[self.layers[0]].shape[-2:]

        aligned_features = []
        for layer in self.layers:
            feature = feature_maps[layer]
            feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            aligned_features.append(feature)

        features = torch.cat(aligned_features, dim=1)
        batch_size, channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        patches = F.normalize(patches, dim=-1)
        return patches, (height, width)

    @torch.no_grad()
    def fit(
        self,
        normal_images,
        batch_size=8,
        coreset_ratio=0.1,
        threshold_quantile=0.995,
        seed=42,
    ):
        all_embeddings = []
        for start in range(0, len(normal_images), batch_size):
            batch_paths = normal_images[start : start + batch_size]
            embeddings, _ = self.extract_embeddings(batch_paths)
            all_embeddings.append(embeddings.reshape(-1, embeddings.shape[-1]).cpu())

        if not all_embeddings:
            raise ValueError("No normal images were provided for PatchCore fitting.")

        memory_bank = torch.cat(all_embeddings, dim=0)
        memory_bank = self._sample_memory_bank(memory_bank, coreset_ratio, seed)
        self.memory_bank = memory_bank.to(self.device)

        train_scores = [
            self.predict(path, threshold=None)["score"]
            for path in normal_images
        ]
        self.threshold = float(np.quantile(np.array(train_scores), threshold_quantile))
        return self

    def _sample_memory_bank(self, memory_bank, coreset_ratio, seed):
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in the range (0, 1].")
        sample_count = max(1, int(len(memory_bank) * coreset_ratio))
        if sample_count >= len(memory_bank):
            return memory_bank

        random.seed(seed)
        indices = random.sample(range(len(memory_bank)), sample_count)
        return memory_bank[indices]

    @torch.no_grad()
    def predict(self, image, threshold=None, distance_chunk_size=4096):
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is not loaded. Run fit or load a checkpoint first.")

        pil_image = load_rgb_image(image)
        embeddings, patch_grid = self.extract_embeddings([pil_image])
        patches = embeddings.reshape(-1, embeddings.shape[-1])

        min_distances = []
        for start in range(0, len(patches), distance_chunk_size):
            patch_chunk = patches[start : start + distance_chunk_size]
            distances = torch.cdist(patch_chunk, self.memory_bank)
            min_distances.append(distances.min(dim=1).values.detach().cpu())

        anomaly_scores = torch.cat(min_distances).reshape(patch_grid).numpy()
        score = float(anomaly_scores.max())
        active_threshold = self.threshold if threshold is None else threshold
        is_anomaly = bool(active_threshold is not None and score >= active_threshold)

        return {
            "score": score,
            "threshold": active_threshold,
            "is_anomaly": is_anomaly,
            "anomaly_map": anomaly_scores,
            "overlay": overlay_anomaly_map(pil_image, anomaly_scores),
        }

    def save(self, path):
        if self.memory_bank is None:
            raise RuntimeError("Cannot save PatchCore before fitting a memory bank.")
        payload = {
            "image_size": self.image_size,
            "layers": self.layers,
            "threshold": self.threshold,
            "backbone_name": self.backbone_name,
            "pretrained": self.pretrained,
            "memory_bank": self.memory_bank.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path, device=None):
        payload = torch.load(path, map_location="cpu")
        return cls(
            image_size=payload["image_size"],
            layers=payload["layers"],
            threshold=payload.get("threshold"),
            backbone_name=payload.get("backbone_name", "wide_resnet50_2"),
            pretrained=payload.get("pretrained", True),
            memory_bank=payload["memory_bank"],
            device=device,
        )


def fit_command(args):
    normal_images = find_images(args.normal_dir, recursive=args.recursive)
    if not normal_images:
        raise SystemExit("No normal training images found.")

    model = PatchCoreL2L3(
        image_size=args.image_size,
        layers=tuple(args.layers),
        pretrained=not args.no_pretrained,
        device=args.device,
    )
    model.fit(
        normal_images=normal_images,
        batch_size=args.batch_size,
        coreset_ratio=args.coreset_ratio,
        threshold_quantile=args.threshold_quantile,
        seed=args.seed,
    )
    model.save(args.output)
    print(f"Saved PatchCore memory bank to {args.output}")
    print(f"Images: {len(normal_images)}")
    print(f"Layers: {', '.join(args.layers)}")
    print(f"Threshold: {model.threshold:.6f}")


def predict_command(args):
    image_paths = find_images(args.input, recursive=args.recursive)
    if not image_paths:
        raise SystemExit("No input images found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PatchCoreL2L3.from_checkpoint(args.checkpoint, device=args.device)
    threshold = args.threshold if args.threshold is not None else model.threshold

    csv_path = output_dir / "patchcore_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "score", "threshold", "status"])
        writer.writeheader()

        for path in image_paths:
            result = model.predict(path, threshold=threshold)
            status = "anomaly" if result["is_anomaly"] else "pass"
            writer.writerow(
                {
                    "image": str(path),
                    "score": f"{result['score']:.6f}",
                    "threshold": "" if threshold is None else f"{threshold:.6f}",
                    "status": status,
                }
            )

            if args.save_overlays:
                output_name = f"{path.stem}_patchcore_overlay.png"
                Image.fromarray(result["overlay"]).save(output_dir / output_name)

    print(f"Saved results to {csv_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="PatchCore L2/L3 anomaly pipeline for PCB inspection.")
    subparsers = parser.add_subparsers(required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a normal-image memory bank.")
    fit_parser.add_argument("--normal-dir", nargs="+", required=True)
    fit_parser.add_argument("--output", default="patchcore_l2l3.pt")
    fit_parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS), choices=["layer1", "layer2", "layer3", "layer4"])
    fit_parser.add_argument("--image-size", type=int, default=256)
    fit_parser.add_argument("--batch-size", type=int, default=8)
    fit_parser.add_argument("--coreset-ratio", type=float, default=0.1)
    fit_parser.add_argument("--threshold-quantile", type=float, default=0.995)
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    fit_parser.add_argument("--no-pretrained", action="store_true")
    fit_parser.add_argument("--device", default=None)
    fit_parser.set_defaults(func=fit_command)

    predict_parser = subparsers.add_parser("predict", help="Score images with a fitted memory bank.")
    predict_parser.add_argument("--checkpoint", default="patchcore_l2l3.pt")
    predict_parser.add_argument("--input", nargs="+", required=True)
    predict_parser.add_argument("--output-dir", default="patchcore_results")
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--save-overlays", action="store_true")
    predict_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    predict_parser.add_argument("--device", default=None)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
