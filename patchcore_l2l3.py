"""
Backward-compatible entrypoint for `patchcore_l2l3.py` upgraded onto `PatchCorePlus`.

Existing commands (`python patchcore_l2l3.py fit ...` / `predict ...`) and imports
(`from patchcore_l2l3 import PatchCoreL2L3`) now leverage Greedy k-Center Coreset,
FAISS kNN, True-RGB heatmaps, Multi-Product routing, and optional DINOv2 Hybrid backbones,
while remaining 100% able to load legacy `patchcore_l2l3.pt` and `patchcore_memory_bank.npz` files.
"""

from __future__ import annotations

from src.anomaly.patchcore_plus import (
    SUPPORTED_EXTENSIONS,
    PatchCorePlus,
    find_images,
    load_rgb_image,
    main,
    render_true_rgb_overlay,
)


class PatchCoreL2L3(PatchCorePlus):
    """Compatibility alias defaulting to WideResNet50 512px Greedy-Coreset PatchCore++ if instantiated directly."""

    def __init__(
        self,
        image_size: int = 512,
        layers=("layer1", "layer2"),
        backbone_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        device=None,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            backbone_name=backbone_name,
            cnn_layers=tuple(layers),
            pretrained=pretrained,
            device=device,
            **kwargs,
        )


overlay_anomaly_map = render_true_rgb_overlay

__all__ = [
    "PatchCoreL2L3",
    "PatchCorePlus",
    "SUPPORTED_EXTENSIONS",
    "find_images",
    "load_rgb_image",
    "overlay_anomaly_map",
]


if __name__ == "__main__":
    main()
