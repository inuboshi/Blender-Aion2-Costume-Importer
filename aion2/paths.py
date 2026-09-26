"""Unreal Engine object-path resolution against an FModel export tree.

FModel with ``KeepDirectoryStructure`` mirrors a game's ``Content`` folder, so a
UE reference such as::

    /Game/Character/Player/GF/Basebody/Materials/GF_BaseBody_D.GF_BaseBody_D

resolves to::

    <export root>/Content/Character/Player/GF/Basebody/Materials/GF_BaseBody_D.tga

This module owns that translation plus discovery of the mesh/material/JSON
artifacts that FModel produced.  It has **no** ``bpy`` dependency.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Default FModel output root (set ``AION2_EXPORT_ROOT`` to override).
DEFAULT_EXPORT_ROOT = r"your FModel export root"

#: Image extensions FModel may write.  TGA (uncompressed BGRA) is the default.
TEXTURE_EXTS: Tuple[str, ...] = (".tga", ".png", ".tif", ".tiff", ".dds", ".bmp")

#: UE mount points that appear in FModel material JSON.
_MOUNT_PREFIXES = ("/Game/", "/Script/", "/Engine/")


def export_root(explicit: Optional[str] = None) -> Path:
    """Return the active export root, honouring an explicit path or env var."""
    raw = explicit or os.environ.get("AION2_EXPORT_ROOT") or DEFAULT_EXPORT_ROOT
    return Path(raw).expanduser()


def ue_to_content_rel(ue_path: str) -> Optional[str]:
    """``/Game/A/B.C`` -> ``A/B`` (a path relative to ``Content/``).

    Returns ``None`` for references that are not ``/Game/`` assets (engine or
    script references carry no exported files).
    """
    if not ue_path:
        return None
    u = ue_path.strip().strip('"')
    for prefix in _MOUNT_PREFIXES:
        if u.startswith(prefix):
            if prefix != "/Game/":
                return None
            u = u[len("/Game/"):]
            break
    else:
        return None
    # Strip the UE "package.object" suffix -> keep the package path only.
    u = u.split(".")[0]
    return u.replace("\\", "/") or None


# --------------------------------------------------------------------------- #
# Texture resolution
# --------------------------------------------------------------------------- #

class TextureResolver:
    """Indexes every exported image once, then resolves UE paths in O(1).

    FModel appends ``_LAYER0``/``_tmp`` style suffixes to some exports and may
    write PNG instead of TGA, so resolution is done with ordered fallbacks
    rather than one exact string match.
    """

    def __init__(self, root: Path, content_dir: str = "Content") -> None:
        self.root = Path(root)
        self.content = self.root / content_dir
        self._by_rel: Dict[str, Path] = {}
        self._scanned = False

    # -- indexing ---------------------------------------------------------- #

    def scan(self, force: bool = False) -> "TextureResolver":
        if self._scanned and not force:
            return self
        self._by_rel.clear()
        if not self.content.is_dir():
            self._scanned = True
            return self
        for dirpath, _dirnames, filenames in os.walk(self.content):
            for name in filenames:
                _stem, ext = os.path.splitext(name)
                if ext.lower() not in TEXTURE_EXTS:
                    continue
                abs_path = Path(dirpath) / name
                rel = os.path.relpath(abs_path, self.content).replace("\\", "/")
                self._by_rel.setdefault(rel.lower(), abs_path)
        self._scanned = True
        return self

    # -- lookup ------------------------------------------------------------ #

    def resolve(self, ue_path: str) -> Optional[Path]:
        """Resolve a UE texture reference to an on-disk image, or ``None``."""
        rel = ue_to_content_rel(ue_path)
        if not rel:
            return None
        self.scan()
        key = rel.lower()
        # 1) exact stem, any known extension
        for ext in TEXTURE_EXTS:
            hit = self._by_rel.get(key + ext)
            if hit:
                return hit
        # 2) FModel layer suffixes, e.g. T2A_DeriveNM -> T2A_DeriveNM_LAYER0
        for candidate_key, candidate in self._by_rel.items():
            stem, ext = os.path.splitext(candidate_key)
            if ext not in TEXTURE_EXTS:
                continue
            if stem.startswith(key) and stem[len(key):].startswith("_"):
                return candidate
        return None

    def resolve_stem(self, stem: str) -> Optional[Path]:
        """Find an exported image by file *stem*, ignoring where it lives.

        Needed because Aion2's texture *arrays* export one file per element with
        an FModel suffix, while the material names a different asset.  The face
        makeup is the case in point: ``MI_NPC_Leda_0_Head`` asks for
        ``MK_Head_HBS_05``, which FModel wrote as
        ``MK_Head_HBS_T2A_01_LAYER5.tga``.
        """
        want = (stem or "").lower()
        if not want:
            return None
        self.scan()
        for key, path in self._by_rel.items():
            if key.rsplit('/', 1)[-1].rsplit('.', 1)[0] == want:
                return path
        for key, path in self._by_rel.items():
            if key.rsplit('/', 1)[-1].startswith(want):
                return path
        return None

    def all_textures(self) -> List[Path]:
        self.scan()
        return sorted(self._by_rel.values())

    def stats(self) -> Dict[str, int]:
        self.scan()
        return {"textures_indexed": len(self._by_rel)}


# --------------------------------------------------------------------------- #
# Material definitions
# --------------------------------------------------------------------------- #

#: Slot-name aliases grouped by the *role* we care about.  FModel preserves the
#: original material parameter names, and Aion2 uses several spellings for the
#: same logical map, so each role is resolved through an ordered candidate list.
BASE_COLOR_SLOTS = ("BaseColor", "PM_Diffuse", "BaseColorTex", "Diffuse")
NORMAL_SLOTS = ("Normal", "PM_Normals", "NormalBase", "NormalMap")
PACKED_SLOTS = ("ARSC", "ARS", "ARM", "PM_SpecularMasks", "PackedMask")
EMISSIVE_SLOTS = ("SE", "Emissive", "PM_Emissive", "EmissiveColor")
#: A dedicated opacity map, when a material has one.
#:
#: ``Cstm`` deliberately is **not** here.  It used to be, and because the custom
#: mask is bound on 1852 of the exported armour instances it won the opacity
#: slot on nearly all of them - overriding the real source.  Aion2's ``CSTM``
#: maps are flat-opaque (alpha 1.0 everywhere), so every ``_DO`` cutout
#: (``GF_0108_T04_Pants_DO`` has 23% of its pixels at alpha < 0.5) rendered
#: solid.  Opacity comes from the base colour's alpha instead; see
#: ``MaterialDef.uses_base_alpha``.
OPACITY_SLOTS = ("OpacityMask", "Opacity", "AO_Mask")
CUSTOM_SLOTS = ("CSTM", "Cstm", "CustomMask", "G_Mask")

#: Texture stems that are *masks*, not albedo.  Aion2's head materials point
#: ``PM_Diffuse`` at ``T_Mask_Black`` (a shaved-hair/stubble mask), so treating
#: that slot as base colour would paint faces solid black - these are rejected as
#: albedo sources and surfaced as a data gap instead.
MASK_STEMS = {
    "t_mask_black", "t_mask_white", "t_mask", "black", "white",
    "b_mask", "g_mask", "r_mask", "a_mask", "t_base_m",
}


def is_mask_reference(ue_path: str) -> bool:
    """True when *ue_path* points at a known mask/utility texture."""
    rel = ue_to_content_rel(ue_path)
    if not rel:
        return False
    return Path(rel).stem.lower() in MASK_STEMS


#: "Layered body" material shells - the base body/legs wearing whatever armour
#: happened to be equipped when FModel captured the material instance.  Their
#: JSON therefore points at *another* armour set's maps: Aion2's ``GF_BaseFull``
#: ships ``MI_GF_Base_Body`` textured with set ``0103``, which paints bare skin
#: with cloth.  These are redirected onto the character's own skin maps.
LAYERED_BODY_RE = re.compile(
    r"^(?:MI_(?:GF|GM)_Base_(?:Body|Pants|Glove|Boots|Head)|(?:GF|GM)_BaseBody)",
    re.IGNORECASE,
)


def is_layered_body_material(name: str) -> bool:
    """True for base-body shell materials whose maps come from an armour set."""
    return bool(LAYERED_BODY_RE.match(name or ""))


#: Head (face) material instances: ``MI_GF_Head_019``, ``GF_Head_001``,
#: ``GF_BaseHead``.  These are the materials that must not fall back to the
#: *body* skin - see :func:`is_head_material`.  ``_Eye`` companions are a
#: different shader entirely and are matched out.
HEAD_MATERIAL_RE = re.compile(r"^(?:MI_)?(?:GF|GM)_(?:Base)?Head(?:_\d+)?$", re.IGNORECASE)


def is_head_material(name: str) -> bool:
    """True for a face skin material.

    Aion2's face shader (``MF_CombinePartsForHead``) composes the face from a
    shared base-head albedo plus tinted makeup layers.  The per-style material
    instances only name a *mask* in ``PM_Diffuse``, so a face built from its own
    JSON alone has no albedo at all - it must borrow the shared head maps rather
    than the body's, otherwise the face renders as blank skin with no features.
    """
    stem = _strip_blender_suffix(name)
    if "eye" in stem.lower():
        return False
    return bool(HEAD_MATERIAL_RE.match(stem))


def _strip_blender_suffix(name: str) -> str:
    """``GF_Head_001.002`` -> ``GF_Head_001`` (Blender's duplicate naming)."""
    base, dot, tail = (name or "").rpartition(".")
    return base if dot and tail.isdigit() else (name or "")


#: The translucent eye-AO / tear-line overlay every head mesh carries.
#: ``MI_Eye_AO_02`` ships **no** material JSON - FModel never exported
#: ``Common/MI_Eye_AO`` - so the map has to be recovered from the instance name.
EYE_AO_RE = re.compile(r"^MI_(?:EyeAO_Tear|Eye_AO_(?P<num>\d+))$", re.IGNORECASE)


def is_eye_ao_material(name: str) -> bool:
    """True for the eye-AO/tear overlay shells (``MI_Eye_AO_02``)."""
    return bool(EYE_AO_RE.match(_strip_blender_suffix(name)))


def eye_ao_texture_ue(name: str) -> Optional[str]:
    """Recover ``MI_Eye_AO_02`` -> ``Customize/Eye_AO_M_02``.

    ``MI_EyeAO_Tear`` has no number of its own; its JSON names
    ``Customize/Eye_AO_M_07`` explicitly, so ``07`` is used as the default.
    """
    match = EYE_AO_RE.match(_strip_blender_suffix(name))
    if not match:
        return None
    stem = f"Eye_AO_M_{match.group('num') or '07'}"
    return f"/Game/Character/Player/Customize/{stem}.{stem}"

#: Shading-model enum values from UE's ``EMaterialShadingModel`` (as emitted by
#: FModel).  5 == MSM_SubsurfaceProfile, 14 == MSM_FromMaterialExpression.
SHADING_SUBSURFACE_PROFILE = 5
SHADING_FROM_EXPRESSION = 14


#: The face's makeup decals, as Aion2 selects them.
#:
#: ``(part, array group, index scalar, colour key, strength key, mask decode)``.
#: The layer index is the part's own ``*_NUMBER`` scalar - ``MI_NPC_Leda_0_Head``
#: pins it explicitly (``Blusher_NUMBER_Texture = MK_Head_HBS_05`` alongside
#: ``Blusher_NUMBER``), which is what identifies the scheme.  ``*Use_Color`` is
#: the blend strength; when it is absent the part is off, so a head that never
#: enables a decal never gets one.
#:
#: The decode matters: EEE/HBS layers are black with the art bright (mask =
#: ``max(r, g, b)``), but the **LFL** layers key their art *green* -
#: ``MK_Head_LFL_T2A_01_LAYER1`` has a pure-green background (median G 0.97)
#: with the lip drawn where green drops out - so ``max(r, g, b)`` there reads
#: the *background* as a full-face mask and paints the whole face with the lip
#: colour.  Those use ``background_key`` (mask = max-channel distance from the
#: decal's own measured background colour; see :meth:`MakeupLayer`).
MAKEUP_PARTS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("EyeLiner",    "EEE", "EyeLiner_NUMBER",    "EyeLiner_Color",    "EyeLinerUse_Color",    "max_rgb"),
    ("EyeShadowA",  "EEE", "EyeShadowA_NUMBER",  "EyeShadowA_Color",  "EyeShadowAUse_Color",  "max_rgb"),
    ("LipA",        "LFL", "LipA_NUMBER",        "LipA_Color",        "LipAUse_Color",        "background_key"),
    ("LipB",        "LFL", "LipB_NUMBER",        "LipB_Color",        "LipBUse_Color",        "background_key"),
    ("Blusher",     "HBS", "Blusher_NUMBER",     "Blusher_Color",     "BlusherUse_Color",     "max_rgb"),
    ("Highlighter", "HBS", "Highlighter_NUMBER", "Highlighter_Color", "HighlighterUse_Color", "max_rgb"),
)


def makeup_array_asset(group: str, index: int) -> str:
    """``("HBS", 5)`` -> ``MK_Head_HBS_T2A_01_LAYER5`` (the exported file stem)."""
    return f"MK_Head_{group}_T2A_01_LAYER{index}"


@dataclass
class MakeupLayer:
    """One makeup decal to composite onto the face albedo.

    ``decode`` is how the layer's mask is read out of the decal texture:

    * ``max_rgb`` - black background, bright art (EEE/HBS): the mask is
      ``max(r, g, b)``;
    * ``background_key`` - art keyed against a bright background (the LFL
      lip layers are grey-on-green): the mask is the max-channel distance
      from the decal's own background colour, measured from the texture's
      pixel median at build time.
    """

    part: str
    mask: Path
    colour: Tuple[float, float, float]
    strength: float
    decode: str = "max_rgb"


#: Name of the extra UV layer the assembler bakes onto a head mesh.
#:
#: The makeup decals are authored in head UV space, so UV0 samples them
#: directly.  The eyebrow is not: ``Cstm_EyeBrow_T2A`` is a 24-element array of
#: *single* brows that Aion2 stamps onto the forehead with a planar projection,
#: so it needs coordinates that do not exist in UV0.  They are baked once, at
#: bind pose, by :func:`aion2.faceproj.project_brow` - per-vertex, so the brow
#: deforms with the face instead of sliding across it.  Those coordinates are
#: *continuous* over the whole head (clamped to the brow box outside it), with
#: :data:`BROW_MASK_ATTRIBUTE` saying where they mean anything.
BROW_UV_LAYER = "A2_BrowUV"

#: Per-vertex weight (``0..1``) for the same projection: 1 over the forehead the
#: brow is stamped onto, 0 elsewhere.  The eyebrow atlas *cannot* be gated by
#: sending off-brow vertices to a transparent corner of the image: the UV would
#: still be interpolated across any triangle that spans the boundary, and that
#: interpolation sweeps straight through the sprite.  On head 002 that is 190
#: triangles painting thin strokes over the forehead, so the "not brow here"
#: signal has to be a separate attribute rather than part of the UV.
BROW_MASK_ATTRIBUTE = "A2_BrowMask"

#: The single-brow texture array asset (one exported file per element).
EYEBROW_ARRAY = "Cstm_EyeBrow_T2A"


def eyebrow_array_asset(index: int) -> str:
    """``12`` -> ``Cstm_EyeBrow_T2A_LAYER12`` (the exported file stem)."""
    return f"{EYEBROW_ARRAY}_LAYER{index}"


@dataclass
class Eyebrow:
    """The face's eyebrow decal.

    Unlike the makeup masks, this one is **not** in head UV space: its content
    sits in the middle of a 512x256 atlas while the head's eyebrow region lives
    somewhere else entirely in UV0, and Aion2 stamps it on with
    ``EyeBrow_Scale*``/``_Rotate``/``Eyebrow_Distance``/``EyeBrow_Height``.

    Its **alpha** is the brow mask and its RGB is a bluish detail map, so only
    the mask is used and the colour comes from the material's
    ``EyeBrowHair_Color``/``EyeBrowPencil_Color``.
    """

    #: Element of the texture array (``EyeBrow_NUMBER``), i.e. the brow shape.
    index: int
    mask: Path
    hair_colour: Tuple[float, float, float]
    pencil_colour: Tuple[float, float, float]
    #: ``EyeBrowHair_intensity`` / ``EyeBrowPencil_Intensity``.
    hair_strength: float
    pencil_strength: float
    #: The atlas's own alpha bounding box ``(u0, u1, v0, v1)`` measured from the
    #: **top** of the texture - the projection maps that box onto the forehead,
    #: so the atlas's empty margins never matter.
    content_box: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    #: UE's own placement parameters, reported rather than applied.
    raw: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaterialDef:
    """A parsed FModel material-instance JSON, with textures resolved."""

    name: str
    json_path: Path
    ue_textures: Dict[str, str] = field(default_factory=dict)
    colors: Dict[str, object] = field(default_factory=dict)
    scalars: Dict[str, float] = field(default_factory=dict)
    switches: Dict[str, bool] = field(default_factory=dict)
    properties: Dict[str, object] = field(default_factory=dict)
    blend_mode: Optional[int] = None
    shading_model: Optional[int] = None
    subsurface_profile: Optional[str] = None
    two_sided: bool = False
    opacity_mask_clip: float = 0.333
    #: role -> resolved on-disk image path
    resolved: Dict[str, Path] = field(default_factory=dict)
    #: role -> original UE reference that could not be resolved
    unresolved: Dict[str, str] = field(default_factory=dict)
    #: role -> UE reference that resolved to a mask, so was rejected as albedo
    mask_not_albedo: Dict[str, str] = field(default_factory=dict)
    #: Set by the assembler for layered body shells: forces skin packing/SSS.
    packed_kind_override: Optional[str] = None
    force_kind: Optional[str] = None
    #: Repairs the assembler applied (shared head maps, body-skin redirect...),
    #: surfaced verbatim in the build report so a fix is never silent.
    notes: List[str] = field(default_factory=list)
    #: Face makeup decals, filled in by the assembler for head materials.
    makeup: List[MakeupLayer] = field(default_factory=list)
    #: The face's eyebrow decal, filled in by the assembler for head materials.
    eyebrow: Optional[Eyebrow] = None

    # -- classification ---------------------------------------------------- #

    @property
    def is_subsurface(self) -> bool:
        return self.shading_model == SHADING_SUBSURFACE_PROFILE

    @property
    def is_skin(self) -> bool:
        """Skin: SSS shading model, or a recognised skin subsurface profile."""
        profile = (self.subsurface_profile or "").lower()
        if "skin" in profile:
            return True
        return self.is_subsurface

    @property
    def packed_slot(self) -> Optional[str]:
        """Original slot name of the packed (AO/Rough/...) map, if any."""
        for slot in PACKED_SLOTS:
            if slot in self.ue_textures:
                return slot
        return None

    @property
    def packed_kind(self) -> Optional[str]:
        """``"ARM"`` (blue == metallic) or ``"ARSC"`` (blue == specular).

        Prefers an explicit override, then the material's own slot name; falls
        back to the packed texture's file name so materials whose slot is named
        only ``PM_SpecularMasks`` still get the right channel semantics.
        """
        if self.packed_kind_override:
            return self.packed_kind_override
        slot = self.packed_slot or ""
        up = slot.upper()
        if up == "ARM":
            return "ARM"
        if up in ("ARSC", "ARS"):
            return "ARSC"
        # Fall back to inspecting the resolved / referenced file names.
        candidates = [str(v) for v in self.ue_textures.values()]
        candidates += [str(p.name) for p in self.resolved.values()]
        for value in candidates:
            v = value.upper()
            if "_ARM" in v or v.endswith("ARM"):
                return "ARM"
            if "_ARSC" in v or "_ARS" in v:
                return "ARSC"
        return None

    @property
    def is_hair(self) -> bool:
        if self.name.upper().startswith(("MI_GF_HAIR", "MI_GM_HAIR")):
            return True
        return any("HAIR" in s.upper() for s in self.ue_textures.values())

    @property
    def is_eye(self) -> bool:
        """True for the eyeball itself, not the AO/tear overlay on top of it.

        ``MI_Eye_AO_02`` and ``MI_EyeAO_Tear`` are translucent *shells* carried
        by every head mesh.  Treating them as eyes turned them into two extra
        opaque dark spheres layered over each eyeball.
        """
        if is_eye_ao_material(self.name):
            return False
        return "EYE" in self.name.upper()

    @property
    def is_hide(self) -> bool:
        """Aion2's ``M_Hide`` mask material (used by placeholder meshes)."""
        return self.name.upper().startswith("M_HIDE")

    @property
    def uses_base_alpha(self) -> bool:
        """True when opacity lives in the base-colour texture's alpha.

        Aion2 flags this with ``Use_BaseColorTexAlpha`` (set on 946 of the
        exported armour instances) and ships the map as ``_DO`` - *diffuse +
        opacity*.  A plain ``_D`` is uniformly opaque, so falling back to its
        alpha is harmless either way.
        """
        if self.switches.get("Use_BaseColorTexAlpha"):
            return True
        for value in list(self.ue_textures.values()) + [str(p.name) for p in self.resolved.values()]:
            if "_DO" in str(value).upper():
                return True
        return False


def _first_present(d: Dict[str, object], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def parse_material(
    json_path: Path,
    resolver: Optional[TextureResolver] = None,
    name_override: Optional[str] = None,
) -> Optional[MaterialDef]:
    """Parse one FModel material JSON into a :class:`MaterialDef`."""
    try:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    asset = raw
    if isinstance(raw, list):
        asset = next(
            (item for item in raw if isinstance(item, dict) and
             item.get("Type") in ("Material", "MaterialInstanceConstant")),
            None,
        )
    if not isinstance(asset, dict):
        return None

    is_fmodel_asset = asset.get("Type") in ("Material", "MaterialInstanceConstant")
    if is_fmodel_asset:
        props = asset.get("Properties") or {}
        textures = {}
        colors = {}
        scalars = {}
        switches = {}

        for item in props.get("TextureParameterValues") or ():
            if not isinstance(item, dict):
                continue
            info = item.get("ParameterInfo") or {}
            value = item.get("ParameterValue") or {}
            name = info.get("Name")
            path = value.get("ObjectPath") if isinstance(value, dict) else None
            if isinstance(name, str) and isinstance(path, str):
                textures[name] = path

        for item in props.get("VectorParameterValues") or ():
            if not isinstance(item, dict):
                continue
            info = item.get("ParameterInfo") or {}
            name = info.get("Name")
            value = item.get("ParameterValue")
            if isinstance(name, str) and isinstance(value, dict):
                colors[name] = value

        for item in props.get("ScalarParameterValues") or ():
            if not isinstance(item, dict):
                continue
            info = item.get("ParameterInfo") or {}
            name = info.get("Name")
            value = item.get("ParameterValue")
            if isinstance(name, str) and isinstance(value, (int, float)):
                scalars[name] = float(value)

        static = props.get("StaticParametersRuntime") or {}
        for item in static.get("StaticSwitchParameters") or ():
            if not isinstance(item, dict):
                continue
            info = item.get("ParameterInfo") or {}
            name = info.get("Name")
            value = item.get("Value")
            if isinstance(name, str) and isinstance(value, bool):
                switches[name] = value

        overrides = props.get("BasePropertyOverrides") or {}
        params = {}
        blend_mode = props.get("BlendMode")
        shading_model = props.get("ShadingModel")
        profile = props.get("SubsurfaceProfile")
    elif isinstance(asset, dict):
        textures = asset.get("Textures") or {}
        params = asset.get("Parameters") or {}
        props = params.get("Properties") or {}
        overrides = props.get("BasePropertyOverrides") or {}
        colors = params.get("Colors") or {}
        scalars = {
            key: float(value)
            for key, value in (params.get("Scalars") or {}).items()
            if isinstance(value, (int, float))
        }
        switches = params.get("Switches") or {}
        blend_mode = params.get("BlendMode")
        shading_model = params.get("ShadingModel")
        profile = props.get("SubsurfaceProfile")
    else:
        return None

    sss = None
    if isinstance(profile, dict):
        sss = profile.get("ObjectName")
    elif isinstance(profile, str):
        sss = profile

    md = MaterialDef(
        name=name_override or asset.get("Name") or Path(json_path).stem,
        json_path=Path(json_path),
        ue_textures={k: v for k, v in textures.items() if isinstance(v, str)},
        colors=colors,
        scalars=scalars,
        switches=switches,
        properties=props,
        blend_mode=blend_mode,
        shading_model=shading_model,
        subsurface_profile=sss,
        two_sided=bool(overrides.get("TwoSided", False)),
        opacity_mask_clip=float(overrides.get("OpacityMaskClipValue", 0.333) or 0.333),
    )

    if resolver is not None:
        for role, candidates in (
            ("base_color", BASE_COLOR_SLOTS),
            ("normal", NORMAL_SLOTS),
            ("packed", PACKED_SLOTS),
            ("emissive", EMISSIVE_SLOTS),
            ("opacity", OPACITY_SLOTS),
            ("custom", CUSTOM_SLOTS),
        ):
            ue = _first_present(md.ue_textures, candidates)
            if not ue:
                continue
            if role == "base_color" and is_mask_reference(ue):
                md.mask_not_albedo[role] = ue
                continue
            hit = resolver.resolve(ue)
            if hit:
                md.resolved[role] = hit
            else:
                md.unresolved[role] = ue
        if md.is_hair:
            for role, parameter in (
                ("hair_card_index", "Card_Index_Texture"),
                ("hair_card_array", "DIO_HairCard"),
                ("hair_diro", "DIRO"),
                ("hair_flow", "FlowMap_HairCard"),
                ("hair_cap_roi", "ROI_HairCap"),
            ):
                ue = md.ue_textures.get(parameter)
                if not ue:
                    continue
                hit = resolver.resolve(ue)
                if hit:
                    md.resolved[role] = hit
                else:
                    md.unresolved[role] = ue
    return md


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def _walk(root: Path, suffixes: Iterable[str]) -> List[Path]:
    want = tuple(s.lower() for s in suffixes)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(want):
                out.append(Path(dirpath) / name)
    return sorted(out)


def discover_meshes(root: Path, subdir: Optional[str] = None) -> List[Path]:
    """All exported mesh files (``.glb``/``.gltf``) under *root*."""
    base = Path(root) / "Content" / subdir if subdir else Path(root) / "Content"
    return _walk(base, (".glb", ".gltf"))


def discover_materials(root: Path, subdir: Optional[str] = None) -> List[Path]:
    """All material JSON files under *root*."""
    base = Path(root) / "Content" / subdir if subdir else Path(root) / "Content"
    return _walk(base, (".json",))


# --------------------------------------------------------------------------- #
# Convenience facade
# --------------------------------------------------------------------------- #

class GamePaths:
    """Bundle of resolver + root used by the Blender-side code."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = export_root(root)
        self.content = self.root / "Content"
        self.textures = TextureResolver(self.root)

    def resolve(self, ue_path: str) -> Optional[Path]:
        return self.textures.resolve(ue_path)

    def meshes(self, subdir: Optional[str] = None) -> List[Path]:
        return discover_meshes(self.root, subdir)

    def materials(self, subdir: Optional[str] = None) -> List[Path]:
        return discover_materials(self.root, subdir)

    def material(self, json_path: Path) -> Optional[MaterialDef]:
        return parse_material(json_path, self.textures)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"GamePaths(root={self.root!r})"
