"""Discover the female-player (``GF``) character pieces that FModel exported.

Aion2 lays characters out as::

    Content/Character/Player/GF/Basebody/GF_BaseFull.glb      <- body + pants
    Content/Character/Player/GF/Basebody/GF_BaseHead.glb      <- head, eyes, lashes
    Content/Character/Player/GF/Hair/GF_Hair_001/GF_Hair_001.glb
    Content/Character/Player/GF/Armor/0096/GF_0096_T04/GF_0096_T04_{Body,Boots,...}.glb
    Content/Character/Player/GF/Head/GF_Head_001/GF_Head_001.glb

This module turns that tree into pickable, labelled choices for the CLI and the
Blender sidebar panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .glb import GlbSummary, summarize
from .paths import GamePaths

#: Folder (relative to ``Content``) holding the female player assets.
GF_DIR = "Character/Player/GF"
GM_DIR = "Character/Player/GM"

#: Order armour parts so a torso/legs assembly reads sensibly in the outliner.
PART_ORDER = ("Body", "Pants", "Boots", "Glove", "Gloves", "Shoulder", "Cape", "Helmet")

#: Base-body pieces that are useful on their own, most complete first.
BASEBODY_PRIORITY = ("GF_BaseFull", "GF_BaseBody", "GF_BaseBoots", "GF_BasePants", "GF_BaseFlatshoes")

#: Meshes that exist in the export but should never be part of a character.
EXCLUDED_STEMS = {"GF_Shadow"}

#: ``GF_BaseHead`` lives in ``Basebody/`` but is a *head*, not a body.  Listing it
#: as a body let a build import it alongside a real head and produce two heads.
BASEBODY_EXCLUDED = EXCLUDED_STEMS | {"GF_BaseHead", "GM_BaseHead"}

#: Material names that mean "this mesh already contains the skin underneath".
BODY_SHELL_PREFIXES = ("GF_BASEBODY", "GM_BASEBODY", "MI_GF_BASE_", "MI_GM_BASE_")

#: Character-creator eye customisation.  Each ``IrisShape_##`` material picks a
#: shared iris height map plus an iris mask, and each ``PupilShape_##`` picks a
#: ``Cstm_Pupil_##`` map - the same 11 + 21 options the in-game creator offers.
EYE_CUSTOM_DIR = "Character/Player/Common/CustomMI"
EYE_IRIS_DIR = f"{EYE_CUSTOM_DIR}/IrisShape"
EYE_PUPIL_DIR = f"{EYE_CUSTOM_DIR}/PupilShape"
#: Shared fallbacks when a shape inherits instead of naming its own textures
#: (``CMI_Eye_CstmBase`` / ``CMM_Eye``); ``Iris_02_M`` is absent from the export.
EYE_IRIS_FALLBACK = (
    "/Game/Character/Player/Common/Iris_02_H.Iris_02_H",
    "/Game/Character/Player/Common/Iris_02_M.Iris_02_M",
)
EYE_PUPIL_FALLBACK = "/Game/Character/Player/Customize/Cstm_Pupil_01.Cstm_Pupil_01"


@dataclass
class EyeIris:
    """One ``CustomMI/IrisShape_##`` option: an iris height + mask map pair."""

    variant: str
    height: Optional[Path] = None
    masks: Optional[Path] = None
    #: True when the shape names a texture that was not extracted, so a shared
    #: fallback stands in for it (``IrisShape_09``/``10`` reference
    #: ``T_Iris_00#_*`` maps this export never wrote out).
    approximated: bool = False

    @property
    def usable(self) -> bool:
        return self.height is not None and self.masks is not None

    @property
    def label(self) -> str:
        # IrisShape_03 -> "Iris 03"
        tail = self.variant.rsplit("_", 1)[-1]
        return f"Iris {tail}" + (" *" if self.approximated else "")

    def detail(self) -> str:
        parts = [p.stem for p in (self.height, self.masks) if p is not None]
        if self.approximated:
            parts.append("(textures not extracted - shared maps used)")
        return " + ".join(parts) or "no textures exported"


@dataclass
class EyePupil:
    """One ``CustomMI/PupilShape_##`` option: a ``Cstm_Pupil_##`` map."""

    variant: str
    pupil: Optional[Path] = None
    scale: float = 1.0
    #: True when the shape's own texture was not extracted.
    approximated: bool = False

    @property
    def usable(self) -> bool:
        return self.pupil is not None

    @property
    def label(self) -> str:
        return f"Pupil {self.variant.rsplit('_', 1)[-1]}" + (
            " *" if self.approximated else "")

    def detail(self) -> str:
        if self.pupil is None:
            return "no pupil texture exported"
        return f"{self.pupil.stem}  x{self.scale:g}"


_BODY_SHELL_CACHE: Dict[str, bool] = {}


def supplies_base_body(mesh_path: Path) -> bool:
    """True when *mesh_path* bundles its own copy of the body/legs shell.

    Cached because the sidebar asks this on every redraw.
    """
    key = str(Path(mesh_path))
    cached = _BODY_SHELL_CACHE.get(key)
    if cached is None:
        cached = any(m.upper().startswith(BODY_SHELL_PREFIXES)
                     for m in summarize(Path(mesh_path)).materials)
        _BODY_SHELL_CACHE[key] = cached
    return cached


@dataclass
class ArmorSet:
    """One armour variant, e.g. ``0096/GF_0096_T04``."""

    group: str
    variant: str
    parts: List[Path] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.group}  {self.variant}"

    def part(self, name: str) -> Optional[Path]:
        for p in self.parts:
            if name.lower() in p.stem.lower():
                return p
        return None

    def ordered_parts(self, only: Optional[Iterable[str]] = None) -> List[Path]:
        chosen = [p for p in self.parts if not only or any(
            key.lower() in p.stem.lower() for key in only)]
        def rank(p: Path) -> int:
            for i, key in enumerate(PART_ORDER):
                if p.stem.lower().endswith(key.lower()):
                    return i
            return len(PART_ORDER)
        return sorted(chosen, key=lambda p: (rank(p), p.stem))


class Catalog:
    """Views over the exported GF (and optionally GM) asset tree."""

    #: Parsing a GLB header is cheap, but the picker asks repeatedly.
    _SUMMARY_CACHE: Dict[str, GlbSummary] = {}

    def __init__(self, gpaths: Optional[GamePaths] = None, gender: str = "GF") -> None:
        self.gpaths = gpaths or GamePaths()
        self.gender = gender.upper()

    # -- roots ------------------------------------------------------------- #

    @property
    def gender_dir(self) -> Path:
        sub = GF_DIR if self.gender == "GF" else GM_DIR
        return self.gpaths.content / sub

    # -- piece listings ---------------------------------------------------- #

    def basebody(self, usable_only: bool = True) -> List[Path]:
        """Naked body shells.  Excludes ``GF_BaseHead`` (it is a head)."""
        base = self.gender_dir / "Basebody"
        if not base.is_dir():
            return []
        glbs = [p for p in sorted(base.glob("*.glb"))
                if p.stem not in BASEBODY_EXCLUDED]
        if usable_only:
            glbs = [p for p in glbs if self.info(p).usable]

        def rank(p: Path) -> int:
            for i, name in enumerate(BASEBODY_PRIORITY):
                if p.stem == name:
                    return i
            return len(BASEBODY_PRIORITY)
        return sorted(glbs, key=lambda p: (rank(p), p.stem))

    def heads(self, usable_only: bool = True) -> List[Path]:
        """Head meshes, default first.

        ``GF_BaseHead`` (from ``Basebody/``) is the in-game default head and is
        listed first; the numbered styles from ``Head/`` follow.
        """
        glbs: List[Path] = []
        default_head = self.gender_dir / "Basebody" / f"{self.gender}_BaseHead.glb"
        if default_head.is_file():
            glbs.append(default_head)
        head_dir = self.gender_dir / "Head"
        if head_dir.is_dir():
            glbs.extend(sorted(head_dir.glob("*/GF_Head_*.glb")))
            glbs.extend(sorted(head_dir.glob("*/GM_Head_*.glb")))
        if usable_only:
            glbs = [p for p in glbs if self.info(p).usable]
        return glbs

    def hairs(self, usable_only: bool = True) -> List[Path]:
        """Hairstyles.

        ``usable_only`` drops placeholder styles such as ``GF_Hair_000``, which
        is a 3-vertex mesh flagged with the ``M_Hide`` mask material (the game's
        "bald" option) - selecting it by default is what makes a build look
        broken.
        """
        base = self.gender_dir / "Hair"
        if not base.is_dir():
            return []
        # Only the primary mesh per style; *_Type1/_Type2 are alt card layouts.
        # Hair assets are stored under the gender‑specific folder (``GF`` or ``GM``).
        # Previously the implementation only globs for ``GF_Hair_*.glb`` which
        # unintentionally excludes male hair assets (``GM_Hair_*.glb``).  To support
        # both genders we glob for any hair file matching ``*_Hair_*.glb`` and then
        # filter out alternative type meshes (``*_Type``) which are not primary
        # hairstyles.
        # Hair assets are stored in a sub‑folder per style (e.g. ``GF_Hair_001/GF_Hair_001.glb``).
        # ``Path.glob`` only searches the directory itself, so it missed the nested files.
        # Using ``rglob`` walks the hierarchy and finds every ``*_Hair_*.glb`` file while
        # still filtering out the alternative layout meshes (``*_Type``) which are not the
        # primary hairstyles.
        glbs = sorted(
            p for p in base.rglob("*_Hair_*.glb")
            if "_Type" not in p.stem
        )
        if usable_only:
            glbs = [p for p in glbs if self.info(p).usable]
        return glbs

    def creature_assets(self, family: str, usable_only: bool = True) -> List[Path]:
        """List generic NPC/monster GLBs without assuming player part names."""
        folder = {"NPC": "NPC", "MONSTER": "Monster"}.get(
            (family or "").upper(), family
        )
        root = self.gpaths.content / "Character" / folder
        if not root.is_dir():
            return []
        glbs = sorted(root.rglob("*.glb"))
        if usable_only:
            glbs = [p for p in glbs if self.info(p).usable]
        return glbs

    # -- metadata ---------------------------------------------------------- #

    def summary_of(self, path: Path) -> GlbSummary:
        return self.info(path)

    def summaries(self, paths: Iterable[Path]) -> List[GlbSummary]:
        return [self.info(p) for p in paths]

    def info(self, path: Path) -> GlbSummary:
        """Cached GLB metadata for *path* (see :mod:`aion2.glb`)."""
        key = str(Path(path))
        cached = Catalog._SUMMARY_CACHE.get(key)
        if cached is None:
            cached = summarize(Path(path))
            Catalog._SUMMARY_CACHE[key] = cached
        return cached

    def hair_variants(self, hair_mesh: Path) -> List[Path]:
        """All card variants (base + Type1/Type2) for one hairstyle."""
        return sorted(hair_mesh.parent.glob(f"{hair_mesh.stem}*.glb"))

    # -- eye customisation ------------------------------------------------- #

    def eye_iris_shapes(self) -> List[EyeIris]:
        """Every ``IrisShape_##``, ordered by index.

        A shape only overrides ``IrisHeight``/``IrisMasks``; the ones that do
        not (``IrisShape_00``/``07``) inherit the shared ``CM_Eye`` maps, so the
        shipped fallbacks are filled in rather than leaving them unusable.
        """
        root = self.gpaths.content / EYE_IRIS_DIR
        if not root.is_dir():
            return []
        fallback_h = self.gpaths.resolve(EYE_IRIS_FALLBACK[0])
        fallback_m = self.gpaths.resolve(EYE_IRIS_FALLBACK[1])
        out: List[EyeIris] = []
        for json_path in sorted(root.glob("*.json")):
            md = self.gpaths.material(json_path)
            height = masks = None
            named = False
            if md is not None:
                named = bool(md.ue_textures.get("IrisHeight")
                             or md.ue_textures.get("IrisMasks"))
                height = self._eye_map(md.ue_textures.get("IrisHeight"))
                masks = self._eye_map(md.ue_textures.get("IrisMasks"))
            approximate = named and (height is None or masks is None)
            out.append(EyeIris(json_path.stem, height or fallback_h,
                               masks or fallback_m, approximate))
        return out

    def eye_pupil_shapes(self) -> List[EyePupil]:
        """Every ``PupilShape_##``, ordered by index."""
        root = self.gpaths.content / EYE_PUPIL_DIR
        if not root.is_dir():
            return []
        fallback = self.gpaths.resolve(EYE_PUPIL_FALLBACK)
        out: List[EyePupil] = []
        for json_path in sorted(root.glob("*.json")):
            md = self.gpaths.material(json_path)
            pupil = scale = None
            named = False
            if md is not None:
                named = bool(md.ue_textures.get("PupilTex"))
                pupil = self._eye_map(md.ue_textures.get("PupilTex"))
                scale = md.scalars.get("PupilOnly_Scale")
            out.append(EyePupil(json_path.stem, pupil or fallback,
                                float(scale) if scale else 1.0,
                                named and pupil is None))
        return out

    def _eye_map(self, ue_reference: Optional[str]) -> Optional[Path]:
        """Resolve a UE texture reference from an eye-customisation JSON."""
        if not ue_reference:
            return None
        return self.gpaths.resolve(ue_reference)

    def iris_shape(self, variant: str) -> Optional[EyeIris]:
        for shape in self.eye_iris_shapes():
            if shape.variant == variant:
                return shape
        return None

    def pupil_shape(self, variant: str) -> Optional[EyePupil]:
        for shape in self.eye_pupil_shapes():
            if shape.variant == variant:
                return shape
        return None

    def armor_sets(self) -> List[ArmorSet]:
        base = self.gender_dir / "Armor"
        if not base.is_dir():
            return []
        sets: List[ArmorSet] = []
        for group_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for variant_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                glbs = sorted(variant_dir.glob("*.glb"))
                if glbs:
                    sets.append(ArmorSet(group_dir.name, variant_dir.name, glbs))
        return sets

    def set_by_label(self, label: str) -> Optional[ArmorSet]:
        """Find an armour set by group (``0103``), variant (``GF_0103_T01``) or label.

        Callers reasonably ask for just the group number, so an exact match is
        tried against each field first and only then a (unique) substring match -
        otherwise ``0103`` silently fell through to the first set in the list.
        """
        target = (label or "").strip().lower()
        if not target:
            return None
        sets = self.armor_sets()
        for s in sets:
            if target in (s.group.lower(), s.variant.lower(), s.label.lower()):
                return s
        matches = [s for s in sets
                   if target in s.group.lower() or target in s.variant.lower()]
        return matches[0] if len(matches) == 1 else None

    # -- summaries --------------------------------------------------------- #

    def summary(self) -> Dict[str, int]:
        return {
            "basebody": len(self.basebody()),
            "heads": len(self.heads()),
            "hairs": len(self.hairs()),
            "armor_sets": len(self.armor_sets()),
        }

    def describe(self, usable_only: bool = True) -> str:
        lines = [f"gender={self.gender}  root={self.gender_dir}"]
        base = self.basebody(usable_only)
        lines.append(f"basebody ({len(base)}): " + ", ".join(p.stem for p in base))
        heads = self.heads(usable_only)
        lines.append(f"heads ({len(heads)}): " +
                     ", ".join(p.stem for p in heads[:8]) +
                     (" ..." if len(heads) > 8 else ""))
        hairs = self.hairs(usable_only)
        lines.append(f"hairs ({len(hairs)}): " +
                     ", ".join(p.stem for p in hairs[:8]) +
                     (" ..." if len(hairs) > 8 else ""))
        sets = self.armor_sets()
        lines.append(f"armor sets ({len(sets)}):")
        for s in sets:
            lines.append(f"   {s.label:24s} parts={[p.stem for p in s.ordered_parts()]}")
        return "\n".join(lines)


def default_spec(
    gpaths: Optional[GamePaths] = None,
    *,
    gender: str = "GF",
    basebody: Optional[str] = None,
    head: Optional[str] = None,
    hair: Optional[str] = None,
    armor: Optional[str] = None,
    armor_parts: Optional[Iterable[str]] = None,
    include_basebody: Optional[bool] = None,
):
    """Build a :class:`~aion2.assembly.CharacterSpec` from friendly names.

    ``basebody``/``head``/``hair`` accept a stem such as ``GF_BaseFull`` or
    ``GF_Hair_001``; ``armor`` accepts a set label such as ``0096`` or
    ``GF_0096_T04``.  Unset values fall back to the first available choice.
    """
    from .spec import CharacterSpec

    cat = Catalog(gpaths, gender)

    def pick(options: List[Path], want: Optional[str], fallback: Optional[Path]) -> Optional[Path]:
        if not options:
            return fallback
        if want:
            for p in options:
                if want.lower() == p.stem.lower() or want.lower() in p.stem.lower():
                    return p
        return options[0]

    armor_paths: List[Path] = []
    sets = cat.armor_sets()
    if sets:
        chosen = cat.set_by_label(armor) if armor else None
        chosen = chosen or sets[0]
        armor_paths = chosen.ordered_parts(armor_parts)

    # Armour meshes already bundle the skin beneath them (the 0096 pants carry a
    # base-legs shell of exactly the same size), so a standalone base body would
    # simply overlap it.  Only add one when nothing else supplies the body, or
    # when the caller explicitly asks for it.
    armour_supplies_body = any(supplies_base_body(p) for p in armor_paths)
    if include_basebody is None:
        include_basebody = not armour_supplies_body
    body = pick(cat.basebody(), basebody, None) if include_basebody else None

    head_mesh = pick(cat.heads(), head, (cat.gender_dir / "Basebody" / f"{gender}_BaseHead.glb"))

    # Hairs: search the full list (so an explicit request can still reach a
    # placeholder) but default to a style with real geometry.
    all_hairs = cat.hairs(usable_only=False)
    hair_mesh = None
    if hair is not None:
        for p in all_hairs:
            if hair.lower() in p.stem.lower():
                hair_mesh = p
                break
    if hair_mesh is None:
        usable = cat.hairs(usable_only=True)
        hair_mesh = usable[0] if usable else (all_hairs[0] if all_hairs else None)

    return CharacterSpec(basebody=body, head=head_mesh, hair=hair_mesh,
                         armor=armor_paths, gender=gender.upper())
