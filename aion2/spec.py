"""Description of the pieces that make up one assembled character.

Kept deliberately free of any ``bpy`` import so blueprints can be built and
inspected under plain CPython - handy for the CLI's ``--catalog`` mode, for the
sidebar panel's "preview what will be loaded" step, and for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class CharacterSpec:
    """Which pieces make up one assembled character.

    Every field holds an exported ``.glb`` path (or ``None`` to skip that part).
    ``armor`` is a list because an armour set is split across several meshes
    (body, pants, boots, gloves, shoulder, cape, helmet).
    """

    basebody: Optional[Path] = None
    head: Optional[Path] = None
    hair: Optional[Path] = None
    armor: List[Path] = field(default_factory=list)
    extra: List[Path] = field(default_factory=list)
    gender: str = "GF"

    def all_paths(self) -> List[Tuple[str, Path]]:
        """Ordered ``(label, path)`` pairs for every piece to import."""
        items: List[Tuple[str, Path]] = []
        for label, path in (
            ("Basebody", self.basebody),
            ("Head", self.head),
            ("Hair", self.hair),
        ):
            if path is not None:
                items.append((label, Path(path)))
        for path in self.armor:
            items.append(("Armor", Path(path)))
        for path in self.extra:
            items.append(("Extra", Path(path)))
        return items

    def is_empty(self) -> bool:
        return not self.all_paths()

    def describe(self) -> str:
        lines = [f"gender: {self.gender}"]
        for label, path in self.all_paths():
            lines.append(f"  {label:9s} {path.name}")
        if self.is_empty():
            lines.append("  (nothing selected)")
        return "\n".join(lines)


__all__ = ["CharacterSpec"]
