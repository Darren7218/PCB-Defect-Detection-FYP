"""
Unified 10-Class Multi-Product PCB & PCBA Defect Taxonomy.

Covers both:
  - Bare-board copper/substrate micro-defects (Classes 0-5: PKU-Market-PCB / VOC_PCB, DeepPCB)
  - Assembled PCBA component, pin, solder, and surface defects (Classes 6-9: VisA PCB1-4, PCBA SMT/THT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DefectClassSpec:
    class_id: int
    name: str
    domain: str  # "bare_pcb" | "assembled_pcba" | "both"
    ovd_prompt: str
    aliases: tuple[str, ...]


UNIFIED_DEFECT_SPECS: tuple[DefectClassSpec, ...] = (
    DefectClassSpec(
        class_id=0,
        name="missing_hole",
        domain="bare_pcb",
        ovd_prompt="pcb pad with a small missing drill hole or pinhole defect",
        aliases=("missing_hole", "missinghole", "pinhole", "pin_hole", "pin-hole"),
    ),
    DefectClassSpec(
        class_id=1,
        name="mouse_bite",
        domain="bare_pcb",
        ovd_prompt="pcb copper trace edge showing small irregular mouse bite notch damage",
        aliases=("mouse_bite", "mousebite", "mouse-bite"),
    ),
    DefectClassSpec(
        class_id=2,
        name="open_circuit",
        domain="bare_pcb",
        ovd_prompt="small break or gap in copper trace on pcb causing open circuit",
        aliases=("open_circuit", "opencircuit", "open", "trace_break"),
    ),
    DefectClassSpec(
        class_id=3,
        name="short",
        domain="bare_pcb",
        ovd_prompt="two adjacent pcb copper traces unintentionally bridged or shorted together",
        aliases=("short", "short_circuit", "copper_short"),
    ),
    DefectClassSpec(
        class_id=4,
        name="spur",
        domain="bare_pcb",
        ovd_prompt="tiny thin copper spike or spur protruding from pcb trace",
        aliases=("spur", "copper_spur"),
    ),
    DefectClassSpec(
        class_id=5,
        name="spurious_copper",
        domain="bare_pcb",
        ovd_prompt="small unwanted isolated copper residue or blob on pcb surface",
        aliases=("spurious_copper", "spuriouscopper", "spurious", "copper_residue"),
    ),
    DefectClassSpec(
        class_id=6,
        name="bad_solder",
        domain="assembled_pcba",
        ovd_prompt="defective pcb solder joint with solder bridge excess solder or cold solder",
        aliases=(
            "bad_solder",
            "solder_bridge",
            "cold_solder",
            "excess_solder",
            "insufficient_solder",
            "solder_ball",
            "solder_defect",
            "melt",
        ),
    ),
    DefectClassSpec(
        class_id=7,
        name="missing_component",
        domain="assembled_pcba",
        ovd_prompt="empty solder pads on assembled pcb where an electronic component is missing",
        aliases=(
            "missing_component",
            "missing_part",
            "absent_component",
            "missing_cap",
            "missing",
        ),
    ),
    DefectClassSpec(
        class_id=8,
        name="misaligned_component",
        domain="assembled_pcba",
        ovd_prompt="misaligned shifted tilted electronic component or bent metal pin on pcb",
        aliases=(
            "misaligned_component",
            "misaligned",
            "shifted_component",
            "tombstone",
            "bent_pin",
            "bent_lead",
            "bent",
            "wrong_polarity",
            "orientation",
        ),
    ),
    DefectClassSpec(
        class_id=9,
        name="surface_scratch",
        domain="both",
        ovd_prompt="visible surface scratch dent foreign debris or contamination mark on pcb board",
        aliases=(
            "surface_scratch",
            "scratch",
            "scratches",
            "dent",
            "foreign_object",
            "contamination",
            "spot",
            "damage",
        ),
    ),
)

UNIFIED_CLASS_NAMES: List[str] = [spec.name for spec in UNIFIED_DEFECT_SPECS]
BARE_PCB_6_CLASS_NAMES: List[str] = [spec.name for spec in UNIFIED_DEFECT_SPECS[:6]]

DEFAULT_OVD_PROMPTS_10CLS: Dict[str, str] = {
    spec.name: spec.ovd_prompt for spec in UNIFIED_DEFECT_SPECS
}
DEFAULT_OVD_PROMPTS_6CLS: Dict[str, str] = {
    spec.name: spec.ovd_prompt for spec in UNIFIED_DEFECT_SPECS[:6]
}

# DeepPCB integer class IDs (Tang et al., 2019: 1-indexed in annotation txt files)
# 1: open, 2: short, 3: mousebite, 4: spur, 5: copper (spurious_copper), 6: pin-hole (missing_hole)
DEEPPCB_ID_TO_UNIFIED_ID: Dict[int, int] = {
    1: 2,  # open -> open_circuit
    2: 3,  # short -> short
    3: 1,  # mousebite -> mouse_bite
    4: 4,  # spur -> spur
    5: 5,  # copper -> spurious_copper
    6: 0,  # pin-hole -> missing_hole
}

# Build case-insensitive alias -> unified class_id lookup
ALIAS_TO_CLASS_ID: Dict[str, int] = {}
for _spec in UNIFIED_DEFECT_SPECS:
    ALIAS_TO_CLASS_ID[_spec.name.lower()] = _spec.class_id
    for _alias in _spec.aliases:
        ALIAS_TO_CLASS_ID[_alias.lower().strip()] = _spec.class_id


def resolve_unified_class_id(raw_label: str | int, source_dataset: str = "generic") -> Optional[int]:
    """Map a dataset-specific label name or integer ID into the unified 0..9 taxonomy."""
    src = source_dataset.lower().strip()
    if isinstance(raw_label, int) or (isinstance(raw_label, str) and raw_label.strip().isdigit()):
        idx = int(raw_label)
        if src == "deeppcb":
            return DEEPPCB_ID_TO_UNIFIED_ID.get(idx)
        if src in {"voc_pcb", "pku_pcb", "tdd_net", "unified"} and 0 <= idx < len(UNIFIED_DEFECT_SPECS):
            return idx
        return None

    normalized = raw_label.lower().strip().replace(" ", "_").replace("-", "_")
    if normalized in ALIAS_TO_CLASS_ID:
        return ALIAS_TO_CLASS_ID[normalized]

    # Substring fallback for compound VisA / Roboflow defect folder names (e.g. "scratch_surface", "bent_pins")
    for alias, cid in ALIAS_TO_CLASS_ID.items():
        if len(alias) >= 4 and alias in normalized:
            return cid
    return None
