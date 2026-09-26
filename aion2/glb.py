"""Read GLB metadata with the standard library only.

Cheap introspection (without importing into Blender) is what lets the catalog
and the UI panel tell a real hairstyle apart from the degenerate ``M_Hide``
placeholder, and show triangle counts in the picker.

A GLB is a 12-byte header followed by length-prefixed chunks; the first chunk
holds JSON.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A  # 'JSON'
_CHUNK_BIN = 0x004E4942   # 'BIN\0'


@dataclass
class GlbSummary:
    """Lightweight facts about one exported mesh."""

    path: Path
    meshes: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    vertices: int = 0
    triangles: int = 0
    joints: int = 0
    primitives: int = 0
    error: Optional[str] = None

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def is_degenerate(self) -> bool:
        """Placeholder meshes (e.g. ``GF_Hair_000``) carry only a few verts."""
        return self.vertices < 64

    @property
    def is_hidden(self) -> bool:
        """True when every primitive uses the ``M_Hide`` mask material."""
        return bool(self.materials) and all(
            m.upper().startswith("M_HIDE") for m in self.materials
        )

    @property
    def usable(self) -> bool:
        return not self.error and not self.is_degenerate and not self.is_hidden

    def label(self) -> str:
        if self.error:
            return f"{self.stem}  (unreadable)"
        flags = []
        if self.is_hidden:
            flags.append("HIDE")
        if self.is_degenerate:
            flags.append("empty")
        extra = f"  [{','.join(flags)}]" if flags else ""
        return f"{self.stem}  ({self.vertices} v / {self.triangles} tri){extra}"


def read_glb_json(path: Path) -> Optional[dict]:
    """Return the JSON chunk of a GLB, or ``None`` if it cannot be read."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != _GLB_MAGIC:
                return None
            _magic, _version, total = struct.unpack("<4sII", header)
            offset = 12
            while offset < total:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    return None
                length, chunk_type = struct.unpack("<II", chunk_header)
                if chunk_type == _CHUNK_JSON:
                    return json.loads(handle.read(length).decode("utf-8"))
                handle.seek(length, 1)
                offset += 8 + length
    except (OSError, ValueError, struct.error):
        return None
    return None


def summarize(path: Path) -> GlbSummary:
    """Summarise a GLB's meshes, materials and vertex/triangle counts."""
    path = Path(path)
    summary = GlbSummary(path=path)
    data = read_glb_json(path)
    if data is None:
        summary.error = "not a readable GLB"
        return summary

    accessors = data.get("accessors") or []
    summary.meshes = [m.get("name") or path.stem for m in data.get("meshes") or []]
    summary.materials = [m.get("name") or "?" for m in data.get("materials") or []]
    skins = data.get("skins") or []
    if skins:
        summary.joints = len(skins[0].get("joints") or [])

    for mesh in data.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            summary.primitives += 1
            pos = (prim.get("attributes") or {}).get("POSITION")
            if pos is not None and pos < len(accessors):
                summary.vertices += int(accessors[pos].get("count", 0) or 0)
            idx = prim.get("indices")
            if idx is not None and idx < len(accessors):
                summary.triangles += int(accessors[idx].get("count", 0) or 0) // 3
    return summary


__all__ = ["GlbSummary", "read_glb_json", "summarize"]


def _self_test() -> None:  # pragma: no cover - manual helper
    import sys
    for arg in sys.argv[1:]:
        print(summarize(Path(arg)).label())
