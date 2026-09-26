"""Rebuild Aion2's packed PBR shading in Blender from FModel material JSON.

Reconstruction rules (per project spec)
---------------------------------------
* **Base colour** - ``_D`` map -> Base Color, image colour space ``sRGB``.
  Multiplied by the *red* channel of the packed map (= ambient occlusion).
* **Normal** - ``_N`` map -> Normal Map node (``Non-Color``, Tangent Space,
  OpenGL/+Y up) -> Principled ``Normal``.
* **Packed maps**
    - skin / ``ARSC``: ``G`` -> Roughness, ``B`` -> Specular IOR Level,
      ``Metallic`` forced to 0.
    - cloth / ``ARM``: ``G`` -> Roughness, ``B`` -> Metallic.
* **Subsurface** (skin only) - Random Walk, weight ~0.1, skin radius profile
  ``(1.0, 0.2, 0.1)`` in R/G/B.
* **Emission** - ``_SE`` / emissive map -> Emission Color.

Anything the JSON references but that is not on disk is reported rather than
silently replaced, so a missing map can never masquerade as default grey.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bpy

from .paths import BROW_MASK_ATTRIBUTE, BROW_UV_LAYER, MaterialDef

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SRGB = "sRGB"
NON_COLOR = "Non-Color"


# --------------------------------------------------------------------------- #
# Small node-graph helpers
# --------------------------------------------------------------------------- #

def _input(node, *names: str):
    """Return the first matching input socket on *node*, or ``None``."""
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            return sock
    return None


def _set_value(node, names: Sequence[str], value) -> bool:
    sock = _input(node, *names)
    if sock is None:
        return False
    try:
        sock.default_value = value
    except (TypeError, ValueError):
        return False
    return True


def _set_enum(owner, name: str, value: str) -> bool:
    """Set an enum property, tolerating builds that dropped or renamed it."""
    try:
        setattr(owner, name, value)
        return True
    except (TypeError, AttributeError, ValueError):
        return False


def _is_translucent(md: MaterialDef) -> bool:
    """True when the material declares UE translucency (``BLEND_Translucent*``).

    Used to tell the two eye overlays apart - both are called "eye AO" but the
    lash shells are ``BLEND_Masked`` (clipped strokes) while ``MI_EyeAO_Tear`` is
    ``BLEND_TranslucentGreyTransmittance``.
    """
    overrides = (md.properties or {}).get("BasePropertyOverrides", {})
    blend = str(overrides.get("BlendMode", "")) if isinstance(overrides, dict) else ""
    if "Translucent" in blend:
        return True
    if "Masked" in blend:
        return False
    return bool(md.properties.get("IsTranslucent", False))


def _new_node(tree, node_type: str, name: str, location: Tuple[float, float]):
    node = tree.nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


_GRAYSCALE_CACHE: Dict[str, bool] = {}


def _image_is_grayscale(image: "bpy.types.Image", tolerance: float = 0.02,
                        max_samples: int = 4096) -> bool:
    """True when an image carries no colour information (a luminance map).

    Aion2 armour ``_D`` maps are all pure greyscale - the colour comes from the
    ``CSTM`` custom-colour system - while skin ``_D`` maps are properly coloured.
    Detecting this is what lets the builder tint armour without also tinting skin.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - Blender always ships numpy
        return False
    key = image.name
    cached = _GRAYSCALE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        buffer = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(buffer)
        pixels = buffer.reshape(-1, 4)
        if len(pixels) > max_samples:
            pixels = pixels[:: max(1, len(pixels) // max_samples)]
        chroma = np.abs(pixels[:, 0] - pixels[:, 1]) + np.abs(pixels[:, 1] - pixels[:, 2])
        result = bool(float(chroma.mean()) < tolerance)
    except Exception:  # pragma: no cover - never fail a build over a heuristic
        result = False
    _GRAYSCALE_CACHE[key] = result
    return result


def _image_bg_colour(image: "bpy.types.Image", max_samples: int = 8192) -> Tuple[float, float, float]:
    """The decal's background colour: the per-channel *median* of its pixels.

    Makeup decals are a flat key colour with art on top, so the median pixel is
    the background: black on EEE/HBS layers, pure green on the LFL lip layers
    (``MK_Head_LFL_T2A_01_LAYER1`` medians (0.0, 0.97, 0.0)).  The makeup mask
    is then the pixel's distance from this colour, which reads bright art on
    black correctly *and* keys green-background layers without a separate
    decode.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - Blender always ships numpy
        return (0.0, 0.0, 0.0)
    try:
        buffer = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(buffer)
        pixels = buffer.reshape(-1, 4)[:, :3]
        if len(pixels) > max_samples:
            pixels = pixels[:: max(1, len(pixels) // max_samples)]
        med = np.median(pixels, axis=0)
        return (float(med[0]), float(med[1]), float(med[2]))
    except Exception:  # pragma: no cover - never fail a build over a heuristic
        return (0.0, 0.0, 0.0)


def _load_image(path: Path, colorspace: str) -> "bpy.types.Image":
    """Load (or reuse) an image and pin its colour space."""
    abs_path = str(Path(path).resolve())
    img = None
    for existing in bpy.data.images:
        if not existing.filepath:
            continue
        try:
            resolved = os.path.abspath(bpy.path.abspath(existing.filepath))
        except (ValueError, RuntimeError):
            continue
        if os.path.normcase(resolved) == os.path.normcase(abs_path):
            img = existing
            break
    if img is None:
        img = bpy.data.images.load(abs_path, check_existing=True)
    try:
        img.colorspace_settings.name = colorspace
    except TypeError:
        pass
    # Aion2 alpha is used as a mask on masked materials; keep it visible.
    try:
        img.alpha_mode = "STRAIGHT"
    except (AttributeError, TypeError):
        pass
    return img


# --------------------------------------------------------------------------- #
# Result reporting
# --------------------------------------------------------------------------- #

@dataclass
class MaterialReport:
    """What actually got wired up for one material."""

    material: str
    kind: str = "other"
    packed_kind: Optional[str] = None
    assigned: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    used_images: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.material} [{self.kind}]"]
        if self.packed_kind:
            parts.append(f"packed={self.packed_kind}")
        parts.append("maps=" + (",".join(sorted(self.assigned)) or "none"))
        if self.missing:
            parts.append("MISSING=" + ",".join(self.missing))
        return "  ".join(parts)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

class Aion2MaterialBuilder:
    """Builds Blender node graphs for Aion2 materials.

    Parameters mirror the project spec so callers can retune without editing the
    graph logic.
    """

    #: Channel order of the CSTM custom-mask map.
    CUSTOM_CHANNELS = ("R", "G", "B", "A")

    def __init__(
        self,
        *,
        sss_weight: float = 0.1,
        sss_radius: Tuple[float, float, float] = (1.0, 0.2, 0.1),
        sss_scale: float = 0.05,
        sss_method: str = "RANDOM_WALK_SKIN",
        normal_strength: float = 1.0,
        ao_strength: float = 1.0,
        apply_roughness_offset: bool = True,
        emission_strength: float = 1.0,
        use_vertex_color: bool = True,
        hair_detail_gain: float = 1.0,
        hair_detail_floor: float = 0.15,
        hair_opacity_pow: float = 1.0,
        hair_opacity_floor: float = 0.35,
        hair_opacity_boost: float = 1.0,
        hair_detail_channel: str = "green",
        hair_cutout: str = "alpha",
        fallback_base_color: Optional[Path] = None,
        fallback_packed: Optional[Path] = None,
        skin_roughness_default: float = 0.45,
        custom_color_tint: bool = True,
        iris_texture: Optional[Path] = None,
        iris_mask_texture: Optional[Path] = None,
        pupil_texture: Optional[Path] = None,
        iris_detail: float = 0.75,
        iris_mask_strength: float = 0.45,
        iris_scale: float = 1.0,
        pupil_scale: float = 1.0,
        iris_mid_color: Optional[Tuple[float, float, float]] = None,
        iris_edge_color: Optional[Tuple[float, float, float]] = None,
        face_makeup: bool = False,
        face_eyebrow: bool = True,
        sclera_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        pupil_color: Tuple[float, float, float, float] = (0.02, 0.02, 0.025, 1.0),
    ) -> None:
        self.sss_weight = sss_weight
        self.sss_radius = sss_radius
        self.sss_scale = sss_scale
        self.sss_method = sss_method
        self.normal_strength = normal_strength
        self.ao_strength = ao_strength
        self.apply_roughness_offset = apply_roughness_offset
        self.emission_strength = emission_strength
        self.use_vertex_color = use_vertex_color
        #: Hair card albedo is a *greyscale strand pattern* that is very dark in
        #: the exported atlas.  Which channel carries it was measured, not
        #: assumed: sampling ``T_Hair_03_DIRO`` at the mesh's own UVs gives
        #: R mean 0.073 / p90 0.157 (essentially empty - reading it left the
        #: cards as smooth blobs) against G mean 0.204 / p90 0.365, which is the
        #: green strand art visible in the atlas.  B is the root->tip gradient.
        #: The gradient tint carries the hue; ``gain`` scales the map and
        #: ``floor`` remaps it into ``[floor, 1]`` so the roots do not crush to
        #: pure black.
        self.hair_detail_gain = hair_detail_gain
        self.hair_detail_floor = hair_detail_floor
        self.hair_detail_channel = hair_detail_channel.lower()
        #: Card cutout source: ``"alpha"`` (``DIRO.A``, the default), ``"green"``
        #: or ``"max"`` of the two.
        #:
        #: ``DIRO.A`` measures empty where the cards sample it (mean 0.048,
        #: median 0.000, 0.8% of loops over 0.9) while the strand art sits in
        #: green, so the card *cutout* has to come from the floor standing in for
        #: the un-exported hair cap: rendered side by side the floor + alpha pair
        #: reads as a dense, well-shaped head of hair, whereas the honest strand
        #: cut (``"green"``/``"max"``) drops half the texels below the 0.333 mask
        #: clip and leaves a lacy see-through hairline.  ``"max"`` is there for
        #: anyone who wants the lacy cut.
        self.hair_cutout = hair_cutout.lower()
        #: Card-mask alpha curve: ``alpha = clamp((floor + (1-floor)*mask)** ... * boost)``.
        #:
        #: The mask is a *soft* strand coverage map, so alpha-tested raw it drops
        #: the thin strands and shows the scalp between cards - the game covers
        #: that with a separate opaque hair *cap*, which the export does not carry
        #: (the ``USE_Haircap`` switch is on).  ``hair_opacity_floor`` stands in
        #: for the cap: it lifts the gaps to a soft hair-coloured veil so the
        #: scalp reads as dense hair, and softens the card silhouette the way real
        #: hair fringes do.  Push it too high and whole cards go solid, which is
        #: what created the grey quads.
        #:
        #: The instance's own ``OpacityPowNear``/``OpacityNear`` are reported, not
        #: applied - they are view-dependent and over-solidify here.
        self.hair_opacity_pow = hair_opacity_pow
        self.hair_opacity_floor = hair_opacity_floor
        self.hair_opacity_boost = hair_opacity_boost

        # Aion2 exports the head's ``PM_Diffuse`` slot pointing at a *mask*, so a
        # face material arrives with no albedo at all.  These opt-in fallbacks
        # let a skin material borrow the base-body skin maps (same character,
        # same skin) instead of rendering default grey.  Leave as None to
        # disable and surface the gap instead.
        self.fallback_base_color = Path(fallback_base_color) if fallback_base_color else None
        self.fallback_packed = Path(fallback_packed) if fallback_packed else None
        self.skin_roughness_default = skin_roughness_default
        # Armour albedo is a greyscale pattern tinted through the CSTM mask +
        # CustomMask*_Color palette; without this every armour set renders grey.
        self.custom_color_tint = custom_color_tint
        # Shared iris map (``Common/Iris_0#_H``).  Every eye material instance
        # only carries ``IrisColor_Mid``/``IrisColor_Edge``, so this map is what
        # actually places the iris on the eyeball's UVs.
        self.iris_texture = Path(iris_texture) if iris_texture else None
        self.iris_mask_texture = Path(iris_mask_texture) if iris_mask_texture else None
        #: Shared pupil map (``Customize/Cstm_Pupil_01``).  White with green
        #: pupil art at the centre (median background G 1.0, art where green
        #: drops out), so the pupil mask is the *inverted* green channel and
        #: the pupil is drawn near-black on top of the iris ramp.
        self.pupil_texture = Path(pupil_texture) if pupil_texture else None
        #: How strongly the iris map's greyscale *relief* modulates the iris
        #: colour.  ``Common/Iris_0#_H`` is a height/fibre map (flat ~0.38 grey
        #: with radiating strands, mean ~0.30 outside the iris), so its red
        #: channel is used as a luminance multiplier around 1.0 rather than as
        #: the ramp input it used to be - driving the two-colour ramp straight
        #: from it is what collapsed the whole eyeball to one flat grey.
        self.iris_detail = iris_detail
        #: How strongly the ``IrisMasks`` map's green channel (the same fibre
        #: pattern at higher contrast) is folded into that relief.
        self.iris_mask_strength = iris_mask_strength
        #: Iris / pupil size multipliers.  ``Iris_Scale`` ships as 0.5-0.7 on the
        #: shared instances and 0.97-0.98 on the shipped heads (a *size* scalar,
        #: so it scales the disc); ``PupilOnly_Scale`` (0.9-1.3) rides on the
        #: chosen ``PupilShape_##``.
        self.iris_scale = iris_scale
        self.pupil_scale = pupil_scale
        # Optional per‑eye colour overrides – if ``None`` the defaults from the
        # exported ``IrisColor_Mid`` / ``IrisColor_Edge`` are used.
        self.iris_mid_color = iris_mid_color
        self.iris_edge_color = iris_edge_color
        #: Composite the face's lips / eyeliner / eyeshadow / blush decals.
        self.face_makeup = face_makeup
        #: Composite the face's eyebrow decal, which is *projected* onto the
        #: forehead through the ``A2_BrowUV`` layer instead of head UV0.
        self.face_eyebrow = face_eyebrow
        #: Sclera colour outside the iris disc.  Aion2's iris map is flat grey
        #: there, so this is supplied rather than sampled (``Sclera_Color``, then
        #: scaled by ``ScleraBrightness`` in the shader).
        self.sclera_color = sclera_color
        #: Pupil colour drawn where the pupil map's art lands.
        self.pupil_color = pupil_color

    # -- public API -------------------------------------------------------- #

    def build(self, material: "bpy.types.Material", md: Optional[MaterialDef]) -> MaterialReport:
        """(Re)build *material*'s node tree from *md*."""
        report = MaterialReport(material=material.name)
        if "eyeao_tear" in material.name.lower():
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            transparent = _new_node(tree, "ShaderNodeBsdfTransparent", "A2_TearTransparent", (420, 0))
            output = _new_node(tree, "ShaderNodeOutputMaterial", "A2_Output", (720, 0))
            tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
            report.kind = "eye_ao"
            report.notes.append("tear-line overlay hidden until its UV registration is extracted")
            return report
        if md is None:
            report.warnings.append("no material JSON found - left as imported")
            return report

        kind = self.classify(md)
        report.kind = kind
        report.packed_kind = md.packed_kind

        # The eye-AO/tear overlay is translucent over the eyeball.  With no map
        # exported its alpha is unknown, and leaving it on the imported glTF
        # shader can end up masking the eye entirely - so it is hidden instead,
        # loudly, and is restored the moment the map is extracted.
        if kind == "eye_ao" and "base_color" not in md.resolved:
            report.warnings.append(
                "eye-AO map not extracted - overlay hidden "
                f"(need {md.unresolved.get('base_color', 'Customize/Eye_AO_M_##')})"
            )
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            out = _new_node(tree, "ShaderNodeOutputMaterial", "A2_Output", (320, 0))
            bsdf = _new_node(tree, "ShaderNodeBsdfPrincipled", "A2_Principled", (20, 0))
            tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            _set_value(bsdf, ("Alpha",), 0.0)
            return report

        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()

        out = _new_node(tree, "ShaderNodeOutputMaterial", "A2_Output", (720, 0))
        bsdf = _new_node(tree, "ShaderNodeBsdfPrincipled", "A2_Principled", (420, 0))
        tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Fidelity first: default to a non-shiny dielectric, then let the real
        # maps take over.  Nothing here is allowed to remain grey.
        _set_value(bsdf, ("Metallic",), 0.0)
        _set_value(bsdf, ("Roughness",), 0.5)
        _set_value(bsdf, ("Specular IOR Level", "Specular"), 0.5)
        _set_value(bsdf, ("Emission Strength",), 0.0)

        if md.is_hide:
            self._build_hide(material, tree, bsdf, md, report)
        elif kind == "hair":
            self._build_hair(material, tree, bsdf, md, report)
        elif kind == "eye":
            self._build_eye(material, tree, bsdf, md, report)
        elif kind == "eye_ao":
            self._build_eye_ao(material, tree, bsdf, md, report)
        else:
            self._build_surface(material, tree, bsdf, md, report, kind)

        self._apply_blend_settings(material, md, report)
        return report

    # -- classification ---------------------------------------------------- #

    @staticmethod
    def classify(md: MaterialDef) -> str:
        """Coarse material family used to pick packing/SSS behaviour."""
        if md.force_kind:
            # The assembler overrides ``MI_GF_Base_*`` shells to "skin" because
            # their own JSON describes the armour they were captured wearing.
            return md.force_kind
        if md.is_hide:
            return "hide"
        if md.is_hair:
            return "hair"
        if md.is_eye:
            return "eye"
        if md.is_skin:
            return "skin"
        return "cloth"

    def _build_hide(self, material, tree, bsdf, md, report) -> None:
        """Aion2's ``M_Hide`` mask: renders nothing (used to blank placeholders)."""
        report.kind = "hide"
        _set_value(bsdf, ("Base Color",), (0.0, 0.0, 0.0, 1.0))
        _set_value(bsdf, ("Alpha",), 0.0)
        _set_value(bsdf, ("Roughness",), 1.0)
        report.notes.append("M_Hide mask material - fully transparent")

    # -- skin / cloth ------------------------------------------------------ #

    def _build_surface(self, material, tree, bsdf, md, report, kind) -> None:
        # Skin may borrow the base-body maps when the export has none.
        fb_color = self.fallback_base_color if kind == "skin" else None
        fb_packed = self.fallback_packed if kind == "skin" else None

        # --- base colour (+ AO) ------------------------------------------- #
        base_img = self._image_for(md, "base_color", report, fallback=fb_color)
        color_socket = None
        if base_img is not None:
            tex = _new_node(tree, "ShaderNodeTexImage", "A2_BaseColor_D", (-620, 260))
            tex.image = base_img
            color_socket = tex.outputs["Color"]

        if base_img is None and md.mask_not_albedo.get("base_color"):
            report.notes.append(
                f"PM_Diffuse was a mask ({md.mask_not_albedo['base_color'].split('/')[-1]}), "
                "not albedo"
            )

        packed_img = self._image_for(md, "packed", report, fallback=fb_packed)
        packed_sep = None
        if packed_img is not None:
            ptex = _new_node(tree, "ShaderNodeTexImage", "A2_Packed", (-620, -180))
            ptex.image = packed_img
            packed_sep = _new_node(tree, "ShaderNodeSeparateColor", "A2_PackedSplit", (-380, -180))
            tree.links.new(ptex.outputs["Color"], packed_sep.inputs["Color"])

        # Greyscale albedo + a custom-colour palette means the real colour is
        # carried by the CSTM mask, so tint before the AO multiply.
        if base_img is not None and color_socket is not None:
            tinted = self._apply_custom_color(tree, bsdf, md, base_img, color_socket, report)
            if tinted is not None:
                color_socket = tinted

        # Makeup sits on top of the finished skin, under the AO multiply.  The
        # layers are drawn in the order _apply_makeup returns them - lips last,
        # so a later eye-area decal can never paint over the lower face (which
        # is how one green-keyed lip layer once washed the whole face red).
        if md.makeup and color_socket is not None:
            made_up = self._apply_makeup(tree, md, color_socket, report)
            if made_up is not None:
                color_socket = made_up

        # The eyebrow is a projected decal, not a UV0 one - see _apply_eyebrow.
        if md.eyebrow is not None and color_socket is not None:
            brow = self._apply_eyebrow(tree, md, color_socket, report)
            if brow is not None:
                color_socket = brow

        if color_socket is not None and packed_sep is not None:
            # AO = red channel of the packed map, multiplied into albedo.
            mix = _new_node(tree, "ShaderNodeMixRGB", "A2_BaseColor_x_AO", (-140, 260))
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = self.ao_strength
            tree.links.new(color_socket, mix.inputs["Color1"])
            tree.links.new(packed_sep.outputs["Red"], mix.inputs["Color2"])
            tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
            report.notes.append("AO from packed.R multiplied into base colour")
        elif color_socket is not None:
            tree.links.new(color_socket, bsdf.inputs["Base Color"])

        # --- normal -------------------------------------------------------- #
        normal_img = self._image_for(md, "normal", report)
        if normal_img is not None:
            ntex = _new_node(tree, "ShaderNodeTexImage", "A2_Normal_N", (-620, -520))
            ntex.image = normal_img
            nmap = _new_node(tree, "ShaderNodeNormalMap", "A2_NormalMap", (-320, -520))
            nmap.space = "TANGENT"
            nmap.inputs["Strength"].default_value = self.normal_strength
            tree.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
            tree.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
            report.notes.append("normal map (Tangent/OpenGL) connected")

        # --- roughness / metallic / specular ------------------------------- #
        if packed_sep is not None:
            rough_out = packed_sep.outputs["Green"]
            if self.apply_roughness_offset:
                offset = float(md.scalars.get("Roughness_Add", 0.0) or 0.0)
                if abs(offset) > 1e-4:
                    add = _new_node(tree, "ShaderNodeMath", "A2_Roughness_Add", (-180, -180))
                    add.operation = "ADD"
                    add.use_clamp = True
                    add.inputs[1].default_value = offset
                    tree.links.new(rough_out, add.inputs[0])
                    rough_out = add.outputs["Value"]
                    report.notes.append(f"roughness offset {offset:+.3f}")
            tree.links.new(rough_out, bsdf.inputs["Roughness"])

            if md.packed_kind == "ARM":
                tree.links.new(packed_sep.outputs["Blue"], bsdf.inputs["Metallic"])
                report.notes.append("ARM: G->Roughness, B->Metallic")
            else:
                # ARSC / ARS (skin): blue carries specular, metal stays 0.
                _set_value(bsdf, ("Metallic",), 0.0)
                spec_in = _input(bsdf, "Specular IOR Level", "Specular")
                if spec_in is not None:
                    tree.links.new(packed_sep.outputs["Blue"], spec_in)
                report.notes.append("ARSC: G->Roughness, B->Specular IOR Level, Metallic=0")

        # --- subsurface (skin) --------------------------------------------- #
        if kind == "skin":
            self._apply_sss(bsdf, report)
            if packed_sep is None:
                # No packed map: use a plausible skin roughness rather than 0.5.
                _set_value(bsdf, ("Roughness",), self.skin_roughness_default)
                report.notes.append(f"roughness default {self.skin_roughness_default} (no packed map)")

        # --- emission ------------------------------------------------------ #
        self._apply_emission(tree, bsdf, md, report)

        # --- alpha (masked materials) -------------------------------------- #
        if self._needs_alpha(md) and base_img is not None:
            alpha_src = self._alpha_source(tree, md, report)
            if alpha_src is not None:
                tree.links.new(alpha_src, bsdf.inputs["Alpha"])
                report.notes.append("alpha driven by mask map")

    def _apply_custom_color(self, tree, bsdf, md, base_img, base_socket, report):
        """Tint a greyscale albedo with Aion2's custom-colour palette.

        The ``CSTM`` map's channels *select* which ``CustomMask{R,G,B,A}_Color``
        applies, so the armour colour is reconstructed as::

            tint = white
            for each enabled channel X:
                tint = lerp(tint, CustomMaskX_Color * A, mask.X * CustomMaskX_Value)
            albedo = greyscale_diffuse * tint

        Only channels the material enables take part: the rest sit at their
        pure-primary defaults with ``A == 0`` (see the note in the body).  A flat
        mask swatch such as ``G_Mask`` therefore collapses to a single uniform
        dye colour, which is exactly the in-game result for undyed gear.  The
        ``Use_ColorCustom`` scalar scales how strongly the tint is applied.

        Returns the new colour socket, or ``None`` when this material does not
        use the system.
        """
        if not self.custom_color_tint:
            return None
        strength = float(md.scalars.get("Use_ColorCustom", 0.0) or 0.0)
        if strength <= 0.0:
            return None

        # Which palette entries are actually live?  Aion2 leaves the channels a
        # material does not use at their pure-primary defaults (``FF0000`` /
        # ``00FF00`` / ``0000FF`` / ``FFFF00``) with ``A == 0.0`` *and*
        # ``CustomMaskX_Use == 0``.  Those two always agree, so either test is
        # enough; the ``Use`` flag wins when it is present.
        #
        # Summing all four unconditionally - as an earlier revision did - mixed
        # the placeholder hues into every set.  Because the flat mask swatches
        # (``G_Mask``/``R_Mask``/``B_Mask``) are fully *opaque* it also dragged
        # ``CustomMaskA_Color`` over the whole suit, so the greyscale pattern was
        # replaced by one wrong flat hue instead of the in-game dyed colour.
        palette: List[Tuple[str, Tuple[float, float, float, float], float]] = []
        for channel in self.CUSTOM_CHANNELS:
            entry = md.colors.get(f"CustomMask{channel}_Color")
            if not isinstance(entry, dict):
                continue
            alpha = float(entry.get("A", 0.0) or 0.0)
            use = md.scalars.get(f"CustomMask{channel}_Use")
            enabled = float(use) > 0.0 if use is not None else alpha > 0.0
            if not enabled:
                continue
            # ``A`` doubles as an HDR intensity (1.5 on 0103's cloth, 1.1 on
            # 0110), so an enabled entry is brighter than its swatch colour.
            intensity = alpha if alpha > 0.0 else 1.0
            colour = tuple(
                float(entry.get(axis, 1.0)) * intensity for axis in ("R", "G", "B")
            ) + (1.0,)
            value = float(md.scalars.get(f"CustomMask{channel}_Value", 1.0) or 1.0)
            palette.append((channel, colour, value))
        if not palette:
            return None

        mask_img = self._image_for(md, "custom", report)
        if mask_img is None:
            return None
        if not _image_is_grayscale(base_img):
            # A genuinely coloured diffuse already carries the hue.
            report.notes.append("custom-colour tint skipped (diffuse is already coloured)")
            return None

        mask = _new_node(tree, "ShaderNodeTexImage", "A2_CustomMask_CSTM", (-700, -1200))
        mask.image = mask_img
        split = _new_node(tree, "ShaderNodeSeparateColor", "A2_CustomSplit", (-500, -1200))
        tree.links.new(mask.outputs["Color"], split.inputs["Color"])

        # ``Separate Color`` only splits RGB - alpha has to come straight off the
        # image node, so resolve sockets through one mapping.
        def channel_socket(channel: str):
            if channel == "A":
                return mask.outputs["Alpha"]
            return split.outputs[{"R": "Red", "G": "Green", "B": "Blue"}[channel]]

        # Accumulate the enabled entries with ``lerp``, which collapses the
        # original shader's mask *selection* into one node per channel: wherever
        # a channel's mask is set its colour is substituted, and wherever no
        # channel covers the surface the accumulator stays white so the
        # greyscale diffuse shows through untouched.
        tint_socket = None
        for index, (channel, colour, value) in enumerate(palette):
            source = channel_socket(channel)
            if abs(value - 1.0) > 1e-4:
                scale = _new_node(tree, "ShaderNodeMath", f"A2_CustomV_{channel}",
                                  (-330, -1150 - 150 * index))
                scale.operation = "MULTIPLY"
                scale.use_clamp = True
                scale.inputs[1].default_value = value
                tree.links.new(source, scale.inputs[0])
                source = scale.outputs["Value"]

            step = _new_node(tree, "ShaderNodeMixRGB", f"A2_Custom_{channel}",
                             (-140, -1500 - 170 * index))
            step.blend_type = "MIX"
            tree.links.new(source, step.inputs["Factor"])
            if tint_socket is None:
                step.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
            else:
                tree.links.new(tint_socket, step.inputs["Color1"])
            step.inputs["Color2"].default_value = colour
            tint_socket = step.outputs["Color"]
        if tint_socket is None:
            return None

        # The greyscale map is the *pattern*; the palette supplies the hue.  Both
        # have to survive, so the tint multiplies the albedo - a plain MIX here
        # would replace the pattern with a flat colour, which is what made armour
        # look both wrong and detail-free.
        tinted = _new_node(tree, "ShaderNodeMixRGB", "A2_BaseColor_x_Tint", (20, -1400))
        tinted.blend_type = "MULTIPLY"
        tinted.inputs["Factor"].default_value = 1.0
        tree.links.new(base_socket, tinted.inputs["Color1"])
        tree.links.new(tint_socket, tinted.inputs["Color2"])

        result = _new_node(tree, "ShaderNodeMixRGB", "A2_BaseColor_x_Custom", (200, -1400))
        result.blend_type = "MIX"
        result.inputs["Factor"].default_value = min(max(strength, 0.0), 1.0)
        tree.links.new(base_socket, result.inputs["Color1"])
        tree.links.new(tinted.outputs["Color"], result.inputs["Color2"])
        report.notes.append(
            f"custom-colour tint from CSTM "
            f"({', '.join(c for c, _, _ in palette)}) strength={strength:.2f}"
        )
        return result.outputs["Color"]

    def _apply_makeup(self, tree, md: MaterialDef, base_socket, report: MaterialReport):
        """Composite the face's makeup decals onto the skin albedo.

        Each decal is a face-space mask blended toward the slot's ``*_Color``
        at ``*Use_Color`` strength, drawn lips-last so a later decal can never
        repaint the lower face::

            albedo = lerp(albedo, colour, mask * strength)

        The mask decode is per layer (``MakeupLayer.decode``): ``max_rgb``
        reads ``max(r, g, b)`` directly - correct for the black-background
        EEE/HBS decals - while ``background_key`` measures each decal's own
        background colour (the pixel median) and masks the distance from it,
        which also handles the green-keyed LFL lip layers whose *background*
        is bright.  Reading LFL with a plain ``max`` is what washed the whole
        face with the lip colour.
        """
        if not self.face_makeup or not md.makeup:
            return None
        # Lips last: LFL decals cover the lower face, and a later MIX over them
        # would repaint it wherever the later decal's UV0 footprint reaches.
        layers = sorted(md.makeup, key=lambda l: 1 if l.decode == "background_key" else 0)
        socket = base_socket
        applied: List[str] = []
        for index, layer in enumerate(layers):
            try:
                img = _load_image(layer.mask, NON_COLOR)
            except (RuntimeError, OSError) as exc:
                report.warnings.append(f"makeup {layer.part} unusable ({exc})")
                continue
            tex = _new_node(tree, "ShaderNodeTexImage", f"A2_Mkup_{layer.part}_Map",
                            (-820, -1900 - 220 * index))
            tex.image = img
            split = _new_node(tree, "ShaderNodeSeparateColor", f"A2_Mkup_{layer.part}_Split",
                              (-600, -1900 - 220 * index))
            tree.links.new(tex.outputs["Color"], split.inputs["Color"])
            if layer.decode == "background_key":
                wide = _new_node(tree, "ShaderNodeMath", f"A2_Mkup_{layer.part}_Mask",
                                 (-60, -1900 - 220 * index))
                # LFL uses bright green as its key, but the texture also has a
                # black lower gradient.  Green-drop decoding mistakes that
                # gradient for lip art; the actual lip is the red/blue chroma.
                wide.operation = "MAXIMUM"
                tree.links.new(split.outputs["Red"], wide.inputs[0])
                tree.links.new(split.outputs["Blue"], wide.inputs[1])
                colour = layer.colour
            else:
                top = _new_node(tree, "ShaderNodeMath", f"A2_Mkup_{layer.part}_MaxRG",
                                (-420, -1900 - 220 * index))
                top.operation = "MAXIMUM"
                tree.links.new(split.outputs["Red"], top.inputs[0])
                tree.links.new(split.outputs["Green"], top.inputs[1])
                wide = _new_node(tree, "ShaderNodeMath", f"A2_Mkup_{layer.part}_Max",
                                 (-240, -1900 - 220 * index))
                wide.operation = "MAXIMUM"
                tree.links.new(top.outputs["Value"], wide.inputs[0])
                tree.links.new(split.outputs["Blue"], wide.inputs[1])
                colour = layer.colour

            mix = _new_node(tree, "ShaderNodeMixRGB", f"A2_Mkup_{layer.part}",
                            (0, -1900 - 220 * index))
            mix.blend_type = "MIX"
            mix.inputs["Color2"].default_value = (*colour, 1.0)
            # Mask * strength, clamped: MixRGB's Fac expects 0..1.
            if layer.strength < 1.0:
                scale = _new_node(tree, "ShaderNodeMath",
                                  f"A2_Mkup_{layer.part}_Strength",
                                  (-60, -2150 - 220 * index))
                scale.operation = "MULTIPLY"
                scale.use_clamp = True
                scale.inputs[1].default_value = layer.strength
                tree.links.new(wide.outputs["Value"], scale.inputs[0])
                factor = scale.outputs["Value"]
            else:
                factor = wide.outputs["Value"]
            tree.links.new(factor, mix.inputs["Factor"])
            tree.links.new(socket, mix.inputs["Color1"])
            socket = mix.outputs["Color"]
            report.assigned[f"makeup:{layer.part}"] = str(layer.mask)
            report.used_images.append(img.name)
            applied.append(layer.part)
        if not applied:
            return None
        report.notes.append("face makeup composited: " + ", ".join(applied))
        return socket

    def _apply_eyebrow(self, tree, md: MaterialDef, base_socket, report: MaterialReport):
        """Composite the face's eyebrow decal onto the skin albedo.

        ``Cstm_EyeBrow_T2A_LAYER<n>`` is a **single** brow whose *alpha* is the
        mask, and it is not authored in head UV space, so it is sampled through
        the ``A2_BrowUV`` layer the assembler baked onto the head mesh (see
        :mod:`aion2.faceproj`).  Two tints are in play - ``EyeBrowHair_Color`` at
        ``EyeBrowHair_intensity``, the brow the atlas actually draws, and
        ``EyeBrowPencil_Color`` at ``EyeBrowPencil_Intensity``, the softer pencil
        beneath it - so::

            colour = mix(pencil, hair, hair_strength)
            albedo = mix(albedo, colour, alpha * min(1, hair + pencil))

        The atlas's RGB is a bluish *detail* map rather than a colour (mean
        ``(0.50, 0.48, 0.70)`` on ``LAYER12``), so it is deliberately unused: a
        flat tint keeps the brow the colour the material asked for.
        """
        if not self.face_eyebrow or md.eyebrow is None:
            return None
        brow = md.eyebrow
        try:
            img = _load_image(brow.mask, NON_COLOR)
        except (RuntimeError, OSError) as exc:
            report.warnings.append(f"eyebrow atlas unusable ({exc})")
            return None

        uvn = _new_node(tree, "ShaderNodeUVMap", "A2_BrowUV", (-1040, -2900))
        uvn.uv_map = BROW_UV_LAYER
        tex = _new_node(tree, "ShaderNodeTexImage", "A2_Brow_Map", (-840, -2900))
        tex.image = img
        # The baked coordinates are continuous over the whole head (clamped to
        # the brow box outside it), so they stay in range and there is nothing
        # for an extension mode to do.  Where they *mean* anything is decided by
        # A2_Brow_Mask below.
        tree.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

        gate = _new_node(tree, "ShaderNodeAttribute", "A2_Brow_Mask", (-1040, -2540))
        gate.attribute_type = "GEOMETRY"
        gate.attribute_name = BROW_MASK_ATTRIBUTE

        colour = _new_node(tree, "ShaderNodeMixRGB", "A2_Brow_Colour", (-620, -2820))
        colour.blend_type = "MIX"
        colour.inputs["Factor"].default_value = brow.hair_strength
        colour.inputs["Color1"].default_value = (*brow.pencil_colour, 1.0)
        colour.inputs["Color2"].default_value = (*brow.hair_colour, 1.0)

        strength = min(1.0, brow.hair_strength + brow.pencil_strength)
        fac = _new_node(tree, "ShaderNodeMapRange", "A2_Brow_Alpha", (-620, -2660))
        fac.interpolation_type = "SMOOTHSTEP"
        fac.inputs["From Min"].default_value = 0.25
        fac.inputs["From Max"].default_value = 0.5
        fac.inputs["To Min"].default_value = 0.0
        fac.inputs["To Max"].default_value = strength
        tree.links.new(tex.outputs["Alpha"], fac.inputs["Value"])

        # Only the forehead the projection actually covers may take the decal.
        # Gating here (on the vertex, after the atlas alpha) is what keeps the
        # blend from following a triangle that spans the brow's edge, which the
        # texture alone cannot express - see BROW_MASK_ATTRIBUTE.
        gate_mul = _new_node(tree, "ShaderNodeMath", "A2_Brow_Gate", (-500, -2540))
        gate_mul.operation = "MULTIPLY"
        tree.links.new(fac.outputs["Result"], gate_mul.inputs[0])
        tree.links.new(gate.outputs["Fac"], gate_mul.inputs[1])

        mix = _new_node(tree, "ShaderNodeMixRGB", "A2_Brow", (-380, -2800))
        mix.blend_type = "MIX"
        tree.links.new(gate_mul.outputs["Value"], mix.inputs["Factor"])
        tree.links.new(base_socket, mix.inputs["Color1"])
        tree.links.new(colour.outputs["Color"], mix.inputs["Color2"])

        report.assigned["eyebrow"] = str(brow.mask)
        report.used_images.append(img.name)
        report.notes.append(
            f"eyebrow {brow.mask.name} on {BROW_UV_LAYER} "
            f"(hair {brow.hair_strength:.2f}, pencil {brow.pencil_strength:.2f})"
        )
        return mix.outputs["Color"]

    def _apply_sss(self, bsdf, report: MaterialReport) -> None:
        try:
            bsdf.subsurface_method = self.sss_method
        except (TypeError, AttributeError):
            try:
                bsdf.subsurface_method = "RANDOM_WALK"
            except (TypeError, AttributeError):
                report.warnings.append("subsurface_method unsupported")
        _set_value(bsdf, ("Subsurface Weight",), self.sss_weight)
        _set_value(bsdf, ("Subsurface Radius",), tuple(self.sss_radius))
        _set_value(bsdf, ("Subsurface Scale",), self.sss_scale)
        _set_value(bsdf, ("Metallic",), 0.0)
        report.notes.append(
            f"SSS {self.sss_method} weight={self.sss_weight} radius={tuple(self.sss_radius)}"
        )

    def _apply_emission(self, tree, bsdf, md, report) -> None:
        emis_img = self._image_for(md, "emissive", report)
        if emis_img is None:
            return
        etex = _new_node(tree, "ShaderNodeTexImage", "A2_Emissive_SE", (-620, -860))
        etex.image = emis_img
        tree.links.new(etex.outputs["Color"], bsdf.inputs["Emission Color"])
        _set_value(bsdf, ("Emission Strength",), self.emission_strength)
        report.notes.append(f"emissive {os.path.basename(emis_img.filepath)} -> Emission")

    # -- hair -------------------------------------------------------------- #

    #: ``T_Hair_##_DIRO`` channel roles, **measured from the exported pixels**
    #: (see ``scripts/analyze_textures.py``) rather than inferred from the name:
    #:
    #:   * ``R`` - a greyscale *strand/diffuse* pattern (the albedo pattern);
    #:   * ``G`` - the per-card *index* (flat horizontal bands).  Not wired: it
    #:     only selects which ``CardA..E_{Index,Section}`` parameters apply to a
    #:     region, and ``_hair_inheritance`` has already folded those into the
    #:     instance;
    #:   * ``B`` - the *root->tip* gradient (dark at the root, bright at the tip);
    #:   * ``A`` - the card *opacity* mask (1.0 inside each card, 0 between).
    #:
    #: ``T_Hair_##_DOI`` (the ``Card_Index_Texture``) is the same atlas repacked
    #: to **D/O/I**: ``R`` == DIRO.R, ``G`` == DIRO.A (opacity), ``B`` == DIRO.G
    #: (index), ``A`` flat 1.0 - verified by correlating the two maps
    #: (``R``+0.95, ``G`` vs ``A`` +0.99, ``B`` vs ``G`` +0.96).  It is the
    #: fallback when a hairstyle ships no ``DIRO``.

    def _hair_cutout(self, tree, tex_node, split, has_diro: bool = True):
        """The socket the card cutout comes from (see ``hair_cutout``).

        ``DIRO.A`` is the *card* mask rather than the strand mask, and it is
        empty where the cards sample it, so on its own every card face falls
        below the 0.333 mask clip and the opacity floor has to carry the shape.
        ``"green"``/``"max"`` use the strand art instead (a lacy cut).
        """
        mode = self.hair_cutout
        green = split.outputs["Green"]
        if not has_diro or mode == "green":
            return green
        alpha = tex_node.outputs["Alpha"]
        if mode == "alpha":
            return alpha
        node = _new_node(tree, "ShaderNodeMath", "A2_HairCutout_Max", (-760, -220))
        node.operation = "MAXIMUM"
        node.use_clamp = True
        tree.links.new(alpha, node.inputs[0])
        tree.links.new(green, node.inputs[1])
        return node.outputs["Value"]

    def _build_hair(self, material, tree, bsdf, md, report) -> None:
        """Hair-card shader reconstructed from the exported card atlases.

        Aion2 hair is a *card* material: a greyscale strand pattern
        (``DIRO.R``) tinted by the ``RootColor``/``MidColor``/``TipColor`` gradient
        along the strand, with the card shape taken from the atlas alpha
        (``DIRO.A``).  The mesh's ``COLOR_0.R`` is itself a baked root->tip
        gradient (it correlates at -0.86 with vertex height), so it drives the
        gradient blend at ``RootUseVertexColor`` exactly as the source names it.

        Anything the material references but that is *not* wired (the
        ``FlowMap``, the ``DIO_HairCard`` gradient array, the view-dependent
        ``OpacityNear/Far`` falloff) is reported rather than silently dropped.
        """
        diro = self._hair_map(md, "hair_diro", report)
        card = self._hair_map(md, "hair_card_index", report)
        tex = diro or card
        if tex is None:
            report.missing.append("hair card atlas (DIRO/DOI)")
            _set_value(bsdf, ("Base Color",), self._hair_tint(md))
            _set_value(bsdf, ("Roughness",), 0.4)
            _set_value(bsdf, ("Metallic",), 0.0)
            return

        role = "DIRO" if diro is not None else "Card_Index(DOI)"
        atlas_path = (md.resolved.get("hair_diro") if diro is not None
                      else md.resolved.get("hair_card_index"))
        report.assigned["hair_atlas"] = str(atlas_path)

        tex_node = _new_node(tree, "ShaderNodeTexImage", "A2_Hair_Atlas", (-900, 240))
        tex_node.image = tex
        report.used_images.append(tex.name)
        split = _new_node(tree, "ShaderNodeSeparateColor", "A2_HairSplit", (-700, 240))
        tree.links.new(tex_node.outputs["Color"], split.inputs["Color"])

        # -- albedo: gradient tint x strand pattern ------------------------ #
        # Which channel carries the strand art is measured (see __init__): it is
        # the green one; ``DIRO.R`` is nearly empty.
        detail = (split.outputs["Green"] if self.hair_detail_channel == "green"
                  else split.outputs["Red"])
        if abs(self.hair_detail_gain - 1.0) > 1e-4:
            gain = _new_node(tree, "ShaderNodeMath", "A2_HairDetail_Gain", (-540, 420))
            gain.operation = "MULTIPLY"
            gain.use_clamp = True
            gain.inputs[1].default_value = self.hair_detail_gain
            tree.links.new(detail, gain.inputs[0])
            detail = gain.outputs["Value"]
        if self.hair_detail_floor > 0.0:
            lift = _new_node(tree, "ShaderNodeMapRange", "A2_HairDetail_Lift", (-360, 420))
            lift.interpolation_type = "LINEAR"
            lift.inputs["From Min"].default_value = 0.0
            lift.inputs["From Max"].default_value = 1.0
            lift.inputs["To Min"].default_value = self.hair_detail_floor
            lift.inputs["To Max"].default_value = 1.0
            tree.links.new(detail, lift.inputs["Value"])
            detail = lift.outputs["Result"]

        root_factor = self._hair_root_factor(tree, md, split.outputs["Blue"] if diro is not None else None)
        root_c, mid_c, tip_c = self._hair_colours(md)

        low = _new_node(tree, "ShaderNodeMixRGB", "A2_Hair_GradLow", (-380, 560))
        low.blend_type = "MIX"
        low.inputs["Color1"].default_value = (*root_c, 1.0)
        low.inputs["Color2"].default_value = (*mid_c, 1.0)
        tree.links.new(self._clamp_affine(tree, "A2_Hair_tLow", root_factor, 2.0, 0.0,
                                          (-560, 700)),
                       low.inputs["Factor"])
        high = _new_node(tree, "ShaderNodeMixRGB", "A2_Hair_GradHigh", (-180, 560))
        high.blend_type = "MIX"
        tree.links.new(low.outputs["Color"], high.inputs["Color1"])
        high.inputs["Color2"].default_value = (*tip_c, 1.0)
        tree.links.new(self._clamp_affine(tree, "A2_Hair_tHigh", root_factor, 2.0, -1.0,
                                          (-560, 760)),
                       high.inputs["Factor"])

        albedo = _new_node(tree, "ShaderNodeMixRGB", "A2_Hair_Albedo", (40, 560))
        albedo.blend_type = "MULTIPLY"
        albedo.inputs["Factor"].default_value = 1.0
        tree.links.new(high.outputs["Color"], albedo.inputs["Color1"])
        tree.links.new(detail, albedo.inputs["Color2"])
        tree.links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])

        # -- card cutout --------------------------------------------------- #
        # ``mask`` is the strand coverage; lift its gaps by ``hair_opacity_floor``
        # (a stand-in for the game's separate hair cap - see the note in __init__),
        # then apply ``pow``/``boost`` for taste.
        alpha = self._hair_cutout(tree, tex_node, split, diro is not None)
        floor = min(max(self.hair_opacity_floor, 0.0), 1.0)
        if floor > 1e-4:
            scale = _new_node(tree, "ShaderNodeMath", "A2_HairOpacity_Scale", (-660, -40))
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = 1.0 - floor
            tree.links.new(alpha, scale.inputs[0])
            lift = _new_node(tree, "ShaderNodeMath", "A2_HairOpacity_Lift", (-480, -40))
            lift.operation = "ADD"
            lift.use_clamp = True
            lift.inputs[1].default_value = floor
            tree.links.new(scale.outputs["Value"], lift.inputs[0])
            alpha = lift.outputs["Value"]
        power = self.hair_opacity_pow
        if abs(power - 1.0) > 1e-4 and power > 0.0:
            pw = _new_node(tree, "ShaderNodeMath", "A2_HairOpacity_Pow", (-320, -40))
            pw.operation = "POWER"
            pw.inputs[1].default_value = power
            tree.links.new(alpha, pw.inputs[0])
            alpha = pw.outputs["Value"]
        if abs(self.hair_opacity_boost - 1.0) > 1e-4:
            boost = _new_node(tree, "ShaderNodeMath", "A2_HairOpacity_Boost", (-160, -40))
            boost.operation = "MULTIPLY"
            boost.use_clamp = True
            boost.inputs[1].default_value = self.hair_opacity_boost
            tree.links.new(alpha, boost.inputs[0])
            alpha = boost.outputs["Value"]
        tree.links.new(alpha, bsdf.inputs["Alpha"])
        report.notes.append(
            f"hair card shader from {role}: "
            + (f"{self.hair_detail_channel[0].upper()}->albedo detail, "
               "B->root gradient, "
               + (f"{self.hair_cutout}->cutout" if diro is not None
                  else "G->cutout (no DIRO root channel)"))
            + f"; alpha floor {floor:g}, pow {power:g}, boost {self.hair_opacity_boost:g}"
        )

        # -- roughness / specular from the instance's own hair values ------ #
        near = md.scalars.get("RoughnessNear")
        far = md.scalars.get("RoughnessFar")
        if near is not None and far is not None:
            rough = (float(near) + float(far)) / 2.0
        elif near is not None:
            rough = float(near)
        elif far is not None:
            rough = float(far)
        else:
            rough = 0.4
        _set_value(bsdf, ("Roughness",), min(max(rough, 0.0), 1.0))
        _set_value(bsdf, ("Metallic",), 0.0)
        spec_min = md.scalars.get("SpecMin")
        _set_value(bsdf, ("Specular IOR Level", "Specular"),
                   min(max(float(spec_min), 0.0), 1.0) if spec_min is not None else 0.35)
        if near is not None or far is not None:
            report.notes.append(
                f"hair roughness {rough:.2f} (RoughnessNear={near}, RoughnessFar={far})"
            )

        # -- things the source shader uses, surfaced rather than guessed --- #
        if any(md.resolved.get(r) for r in ("hair_flow", "hair_card_array")):
            present = [r for r in ("hair_flow", "hair_card_array") if md.resolved.get(r)]
            for r in present:
                report.notes.append(f"{r} present (unwired): {md.resolved[r].name}")
        if card is not None and diro is not None:
            report.notes.append("card index atlas present (DIRO.G / DOI.B) - not wired")
        if any(k in md.scalars for k in ("OpacityNear", "OpacityFar", "OpacityPowNear",
                                         "OpacityPowFar", "OpacityMsk_NearAdd",
                                         "OpacityMsk_FarAdd")):
            report.notes.append(
                "view-dependent opacity (OpacityNear/Far/Pow*) not applied - "
                "alpha is the atlas card mask, curved by hair_opacity_pow/boost"
            )

        self._apply_emission(tree, bsdf, md, report)

    @staticmethod
    def _hair_tint(md: MaterialDef) -> Tuple[float, float, float, float]:
        """Blend Root/Mid/Tip colours into a single representative tint."""
        def col(key):
            c = md.colors.get(key)
            if isinstance(c, dict):
                return (float(c.get("R", 1)), float(c.get("G", 1)), float(c.get("B", 1)))
            return None

        cols = [c for c in (
            col("RootColor"), col("MidColor"), col("TipColor"),
            col("HairColor"), col("CustomMaskG_Color"),
        ) if c]
        if not cols:
            return (1.0, 1.0, 1.0, 1.0)
        n = len(cols)
        r = sum(c[0] for c in cols) / n
        g = sum(c[1] for c in cols) / n
        b = sum(c[2] for c in cols) / n
        # Very dark tints read as black through an atlas; lift the floor.
        return (max(r, 0.06), max(g, 0.06), max(b, 0.06), 1.0)

    @staticmethod
    def _hair_colours(md: MaterialDef) -> Tuple[
            Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """The ``RootColor``/``MidColor``/``TipColor`` gradient stops.

        Aion2 pins all three to the same value on most hairstyles (the vanilla
        ``MI_GF_Hair_001`` uses ``(0.145, 0.012, 0.012)`` for all three), so the
        gradient collapses to a flat tint and the strand *pattern* carries the
        variation - which is exactly what the atlas diffuse is for.
        """
        def col(key, fallback):
            c = md.colors.get(key)
            if isinstance(c, dict):
                return (float(c.get("R", fallback[0])),
                        float(c.get("G", fallback[1])),
                        float(c.get("B", fallback[2])))
            return fallback

        root = col("RootColor", (0.1, 0.05, 0.04))
        mid = col("MidColor", root)
        tip = col("TipColor", mid)
        return root, mid, tip

    def _hair_map(self, md: MaterialDef, role: str,
                  report: MaterialReport) -> Optional["bpy.types.Image"]:
        """Load a hair atlas (``DIRO``/``Card_Index``) as **Non-Color** data.

        These are linear data maps, not sRGB colour (``SRGB = False`` on the
        exported ``Texture2D``), so they must not go through ``_image_for``,
        which assumes a colour map for any role outside packed/normal/custom.
        """
        path = md.resolved.get(role)
        if path is None:
            if role in md.unresolved:
                report.missing.append(f"{role}:{md.unresolved[role]}")
            return None
        try:
            img = _load_image(path, NON_COLOR)
        except (RuntimeError, OSError) as exc:
            report.missing.append(f"{role}({exc})")
            return None
        report.used_images.append(img.name)
        return img

    def _hair_root_factor(self, tree, md: MaterialDef, root_out):
        """Root->tip blend factor from the atlas root band x ``COLOR_0.R``.

        ``DIRO.B`` is the atlas' own root->tip gradient and ``COLOR_0.R`` is the
        mesh's baked one (measured -0.86 against vertex height), so they are
        multiplied and the vertex term is weighted by ``RootUseVertexColor``
        (``0.8`` on ``MI_GF_Hair_001``, ``1.0`` on ``MI_HairCap_01``)::

            t = DIRO.B * ((1 - u) + u * COLOR_0.R)

        Returns a socket, or ``None`` when there is nothing to drive it.
        """
        t = root_out
        if self.use_vertex_color:
            vcol = _new_node(tree, "ShaderNodeVertexColor", "A2_HairRootVCol", (-1000, 60))
            # Blender's glTF importer names the imported COLOR_0 attribute
            # "Color"; that is what the exported hair meshes carry.
            vcol.layer_name = "Color"
            vsplit = _new_node(tree, "ShaderNodeSeparateColor", "A2_HairVColSplit", (-820, 60))
            tree.links.new(vcol.outputs["Color"], vsplit.inputs["Color"])
            try:
                use = float(md.scalars.get("RootUseVertexColor", 1.0))
            except (TypeError, ValueError):
                use = 1.0
            use = min(max(use, 0.0), 1.0)
            if use >= 0.999:
                vterm = vsplit.outputs["Red"]
            else:
                scale = _new_node(tree, "ShaderNodeMath", "A2_HairVCol_Scale", (-640, 0))
                scale.operation = "MULTIPLY"
                scale.use_clamp = True
                scale.inputs[1].default_value = use
                tree.links.new(vsplit.outputs["Red"], scale.inputs[0])
                add = _new_node(tree, "ShaderNodeMath", "A2_HairVCol_Add", (-480, 0))
                add.operation = "ADD"
                add.use_clamp = True
                add.inputs[1].default_value = 1.0 - use
                tree.links.new(scale.outputs["Value"], add.inputs[0])
                vterm = add.outputs["Value"]
            if t is None:
                t = vterm
            else:
                mul = _new_node(tree, "ShaderNodeMath", "A2_HairRoot_Mul", (-300, 160))
                mul.operation = "MULTIPLY"
                mul.use_clamp = True
                tree.links.new(t, mul.inputs[0])
                tree.links.new(vterm, mul.inputs[1])
                t = mul.outputs["Value"]
        return t

    @staticmethod
    def _clamp_affine(tree, name: str, source, factor: float, offset: float, location):
        """``clamp(source * factor + offset, 0, 1)`` as Math nodes."""
        mul = _new_node(tree, "ShaderNodeMath", f"{name}_Mul", location)
        mul.operation = "MULTIPLY"
        mul.use_clamp = True
        mul.inputs[1].default_value = factor
        tree.links.new(source, mul.inputs[0])
        if abs(offset) < 1e-9:
            return mul.outputs["Value"]
        add = _new_node(tree, "ShaderNodeMath", f"{name}_Add",
                        (location[0] + 150, location[1]))
        add.operation = "ADD"
        add.use_clamp = True
        add.inputs[1].default_value = offset
        tree.links.new(mul.outputs["Value"], add.inputs[0])
        return add.outputs["Value"]

    # -- eyes -------------------------------------------------------------- #

    #: Where the iris ends, as a fraction of ``|uv - (0.5, 0.5)| * 2``.
    #:
    #: Measured from the shipped head rather than assumed.  The eyeball is a
    #: 26.4 mm sphere (13.2 mm radius, centre at +/-28.8 mm) whose UV0 is a polar
    #: unwrap about the pupil axis: the vertex at the equator (13.2 mm lateral)
    #: sits at r = 0.76-0.83, and ``scripts/probe_eye.py`` reads the shader's own
    #: radial value off a calibrated render, where the palpebral opening spans
    #: **r = 0.00-0.34** - a 20.1 x 9.1 mm slit, measured against the projected
    #: eyeball silhouette.
    #:
    #: r = 0.19 is therefore the disc that puts an 11.5 mm iris - the real
    #: iris/eyeball ratio of 0.44 - inside that opening, leaving the sclera
    #: visible on both sides.  The previous revision used 0.28, which measured
    #: **100% of the aperture** in the same render (0 sclera pixels) and is what
    #: read as "the iris is stretched over the whole eyeball".
    IRIS_RADIUS = 0.19
    IRIS_SOFTNESS = 0.035
    #: Measured support radii of the exported maps, same ``|uv - 0.5| * 2`` units:
    #: ``Cstm_Iris_M_##`` stays fibre-coloured out to r ~= 0.95 (cyan background
    #: past it), and ``Cstm_Pupil_##`` is black - the pupil itself - out to
    #: r ~= 0.13.  Sampling either through head UV0 (r up to 1.32 across the
    #: sphere) is what stretched the iris over the eyeball in the first place.
    IRIS_TEX_RADIUS = 0.92
    PUPIL_TEX_RADIUS = 0.13
    #: Pupil radius as a fraction of the iris radius: 4 mm of an 11.5 mm iris.
    PUPIL_RATIO = 0.35

    def _build_eye(self, material, tree, bsdf, md, report) -> None:
        """Aion2 eyes: a fitted radial iris disc over a white sclera.

        ``MI_GF_Head_###_Eye`` is a parameter-only instance (FModel reports it as
        ``IsNull``) carrying ``IrisColor_Mid``/``IrisColor_Edge``, ``Iris_Scale``
        and ``Pupil_Scale``, plus the texture slots of the shared ``CM_Eye``
        chain (``IrisHeight`` / ``IrisMasks`` / ``PupilTex``).  Those are exactly
        what the character creator picks through ``CustomMI/IrisShape_##`` and
        ``CustomMI/PupilShape_##``, so the eye picker in the sidebar drives them
        directly.

        What the maps are, measured (``scripts/probe_eye.py``):

        * ``Iris_0#_H`` - a greyscale *relief/fibre* map: ~0.38 grey with
          radiating strands, brighter towards the pupil.  It is **not** an
          albedo, so driving the iris colour ramp straight from it (an earlier
          revision) collapsed the iris to one flat grey.
        * ``Cstm_Iris_M_##`` / ``Iris_##_M`` - radial profiles: R flat ~0.25,
          G is the fibre relief, and B is a *hard-edged disc mask* (0 for
          r < 0.35, 1 past r = 0.40) - the pupil/collarette zone.  The fibre art
          runs out to r ~= 0.95, past which the map is cyan background, which is
          what fixes ``IRIS_TEX_RADIUS``.
        * ``Cstm_Pupil_##`` - white with the pupil drawn at the centre as black
          out to r ~= 0.13, so the pupil mask is the inverted green channel.
        * ``T_Sclera_*`` - a tangent-space normal map with no albedo, so the
          sclera is the ``Sclera_Color`` vector dimmed by ``ScleraBrightness``.
        """
        def col(key, fallback):
            c = md.colors.get(key)
            if isinstance(c, dict):
                return (float(c.get("R", fallback[0])),
                        float(c.get("G", fallback[1])),
                        float(c.get("B", fallback[2])), 1.0)
            return (*fallback, 1.0)

        # Base colours from the exported JSON; may be overridden by the
        # per‑eye colour picker supplied via ``iris_mid_color`` / ``iris_edge_color``.
        mid = col("IrisColor_Mid", (0.11, 0.15, 0.13))
        edge = col("IrisColor_Edge", (0.08, 0.11, 0.15))
        if self.iris_mid_color is not None:
            mid = (*self.iris_mid_color, 1.0)
        if self.iris_edge_color is not None:
            edge = (*self.iris_edge_color, 1.0)
        # ``ScleraBrightness`` is an additive -0.15 on the shipped instances: the
        # sclera is a slightly dimmed white, never pure white.
        bright = float(md.scalars.get("ScleraBrightness", 0.0) or 0.0)
        sclera_colour = tuple(
            max(0.0, min(1.0, c * (1.0 + bright)))
            for c in self.sclera_color[:3]
        ) + (1.0,)
        # ``LimbusDarkAmount`` (4.46 on CM_Eye, absent on the head instances)
        # darkens the iris rim; the mild default stops the edge reading as a
        # flat disc.
        limbus = float(md.scalars.get("LimbusDarkAmount", 1.8) or 0.0)
        rim = max(0.3, 1.0 - 0.14 * limbus)

        def _map(path, role):
            if path is None:
                return None
            try:
                return _load_image(path, NON_COLOR)
            except (RuntimeError, OSError) as exc:
                report.warnings.append(f"{role} map unusable ({exc})")
                return None

        iris_img = _map(self.iris_texture, "iris")
        mask_img = _map(self.iris_mask_texture, "iris mask")
        pupil_img = _map(self.pupil_texture, "pupil")

        # Radial coordinate from the eyeball's own UVs (see the constants).
        tex_coord = _new_node(tree, "ShaderNodeTexCoord", "A2_EyeUV", (-1820, 260))
        centred = _new_node(tree, "ShaderNodeVectorMath", "A2_EyeCentred", (-1640, 180))
        centred.operation = "SUBTRACT"
        centred.inputs[1].default_value = (0.5, 0.5, 0.180)
        tree.links.new(tex_coord.outputs["UV"], centred.inputs[0])
        radius = _new_node(tree, "ShaderNodeVectorMath", "A2_EyeRadius", (-1460, 260))
        radius.operation = "LENGTH"
        tree.links.new(centred.outputs["Vector"], radius.inputs[0])

        # Disc mask: 1 inside the iris, 0 on the sclera.  ``Iris_Scale`` is a
        # 0..1 iris *size* scalar (0.97-0.98 on the shipped heads, 0.5-0.7 on
        # the shared instances), so it grows the measured base radius.
        scale = float(md.scalars.get("Iris_Scale", 1.0) or 1.0)
        disc_r = max(0.06, self.IRIS_RADIUS * self.iris_scale * scale)
        disc = _new_node(tree, "ShaderNodeMapRange", "A2_IrisDisc", (-1280, 120))
        disc.interpolation_type = "SMOOTHSTEP"
        # Updated defaults for the iris disc mask (static values for all eyes)
        # From Min / From Max control the radial range of the disc mask (in UV units).
        # To Min / To Max control the output interpolation (1 = fully inside iris, 0 = outside).
        disc.inputs["From Min"].default_value = 0.182
        disc.inputs["From Max"].default_value = 0.317
        disc.inputs["To Min"].default_value = 1.300
        disc.inputs["To Max"].default_value = 0.300
        tree.links.new(radius.outputs["Value"], disc.inputs["Value"])

        def fitted(k: float, name: str, location, scale=None, offset=None):
            """``uv -> 0.5 + (uv - 0.5) * k``: rescale a map about the pupil.

            With ``k = map_radius / disc_radius`` the map's own iris (or pupil)
            art lands exactly on the disc edge instead of being stretched over
            the whole eyeball.
            """
            node = _new_node(tree, "ShaderNodeVectorMath", name, location)
            node.operation = "MULTIPLY_ADD"
            node.inputs[1].default_value = scale if scale is not None else (k, k, 0.0)
            node.inputs[2].default_value = (0.5, 0.5, 0.0)
            tree.links.new(centred.outputs["Vector"], node.inputs[0])
            return node.outputs["Vector"]

        # ``t`` = 0 at the pupil, 1 at the limbus: the iris ramp's own input.
        t = _new_node(tree, "ShaderNodeMath", "A2_IrisT", (-1100, 420))
        t.operation = "DIVIDE"
        t.use_clamp = True
        t.inputs[1].default_value = disc_r
        tree.links.new(radius.outputs["Value"], t.inputs[0])

        # Iris relief: the map's greyscale, remapped *around 1.0* so it modulates
        # the ramp instead of defining it.
        curve = None
        relief = None
        if iris_img is not None:
            k_iris = self.IRIS_TEX_RADIUS / disc_r
            tex = _new_node(tree, "ShaderNodeTexImage", "A2_Iris_H", (-1820, 700))
            tex.image = iris_img
            tree.links.new(fitted(k_iris, "A2_IrisUV", (-2.735, 4000), scale=(2.735, 2.735, 4.000)),
                           tex.inputs["Vector"])
            split = _new_node(tree, "ShaderNodeSeparateColor", "A2_IrisSplit",
                              (-1640, 700))
            tree.links.new(tex.outputs["Color"], split.inputs["Color"])
            # The ``Iris_##_H`` art is low contrast (its fibres wander inside
            # 0.28-0.48 grey, mean 0.38), so the window is narrow: a wide one
            # left the fibre detail invisible at this disc size.
            curve = _new_node(tree, "ShaderNodeMapRange", "A2_IrisRange", (-1460, 700))
            curve.interpolation_type = "LINEAR"
            curve.inputs["From Min"].default_value = 0.29
            curve.inputs["From Max"].default_value = 0.50
            curve.inputs["To Min"].default_value = 0.66
            curve.inputs["To Max"].default_value = 1.22
            tree.links.new(split.outputs["Red"], curve.inputs["Value"])
            relief = curve.outputs["Result"]
            report.assigned["iris"] = str(self.iris_texture)
            report.used_images.append(iris_img.name)
            report.notes.append(
                f"iris from {Path(self.iris_texture).name} (disc r={disc_r:.2f}, "
                f"map fitted x{k_iris:.2f})"
            )
        else:
            report.notes.append(
                "procedural iris fallback - no Common/Iris_0#_H extracted"
            )

        # The mask map is the same fibres at higher contrast; fold it in gently.
        if mask_img is not None and relief is not None:
            strength = max(0.0, min(1.0, self.iris_mask_strength))
            mask_tex = _new_node(tree, "ShaderNodeTexImage", "A2_Iris_M", (-1820, 900))
            mask_tex.image = mask_img
            tree.links.new(fitted(k_iris, "A2_IrisMaskUV", (-2000, 900)),
                           mask_tex.inputs["Vector"])
            mask_split = _new_node(tree, "ShaderNodeSeparateColor",
                                   "A2_IrisMaskSplit", (-1640, 900))
            tree.links.new(mask_tex.outputs["Color"], mask_split.inputs["Color"])
            fold = _new_node(tree, "ShaderNodeMath", "A2_IrisMaskFold", (-1460, 900))
            fold.operation = "MULTIPLY_ADD"
            fold.inputs[1].default_value = strength
            fold.inputs[2].default_value = 1.0 - strength
            tree.links.new(mask_split.outputs["Green"], fold.inputs[0])
            combined = _new_node(tree, "ShaderNodeMath", "A2_IrisRelief", (-1280, 800))
            combined.operation = "MULTIPLY"
            combined.use_clamp = True
            tree.links.new(curve.outputs["Result"], combined.inputs[0])
            tree.links.new(fold.outputs["Value"], combined.inputs[1])
            relief = combined.outputs["Value"]
            report.assigned["iris_mask"] = str(self.iris_mask_texture)
            report.used_images.append(mask_img.name)
            report.notes.append(
                "iris fibre contrast from "
                f"{Path(self.iris_mask_texture).name} green channel"
            )
        elif mask_img is not None:
            report.notes.append("iris mask ignored - no iris height map to modulate")

        # Iris colour: ``IrisColor_Mid`` in the middle through to a darkened
        # ``IrisColor_Edge`` at the limbus, with a dimmer ring just inside it.
        ramp = _new_node(tree, "ShaderNodeValToRGB", "A2_IrisColors", (-920, 420))
        ramp.color_ramp.interpolation = "EASE"
        stops = ramp.color_ramp.elements
        stops[0].position, stops[0].color = 0.0, (*mid[:3], 1.0)
        stops[1].position, stops[1].color = 1.0, (*[c * rim for c in edge[:3]], 1.0)
        stops.new(0.62).color = (*[c * 0.82 for c in mid[:3]], 1.0)
        tree.links.new(t.outputs["Value"], ramp.inputs["Fac"])
        out_socket = ramp.outputs["Color"]

        # Fold the fibre relief into the colour at ``iris_detail`` strength.
        if relief is not None and self.iris_detail > 0.0:
            amount = max(0.0, min(1.0, self.iris_detail))
            shade = _new_node(tree, "ShaderNodeMath", "A2_IrisShade", (-740, 420))
            shade.operation = "MULTIPLY_ADD"
            shade.use_clamp = True
            shade.inputs[1].default_value = amount
            shade.inputs[2].default_value = 1.0 - amount
            tree.links.new(relief, shade.inputs[0])
            tint = _new_node(tree, "ShaderNodeMixRGB", "A2_IrisDetail", (-560, 420))
            tint.blend_type = "MULTIPLY"
            tint.inputs["Factor"].default_value = 1.0
            tree.links.new(ramp.outputs["Color"], tint.inputs["Color1"])
            tree.links.new(shade.outputs["Value"], tint.inputs["Color2"])
            out_socket = tint.outputs["Color"]

        # Everything the disc does not cover is sclera, not iris.
        sclera = _new_node(tree, "ShaderNodeMixRGB", "A2_IrisOnSclera", (-320, 300))
        sclera.blend_type = "MIX"
        sclera.inputs["Color1"].default_value = sclera_colour
        tree.links.new(disc.outputs["Result"], sclera.inputs["Factor"])
        tree.links.new(out_socket, sclera.inputs["Color2"])
        out_socket = sclera.outputs["Color"]

        # Pupil from the shared ``Cstm_Pupil_##`` map: white with the pupil art
        # at the centre, so the inverted green channel is the pupil mask.  Fitted
        # so the art disc lands at ``PUPIL_RATIO`` of the iris, then gated by the
        # disc so it cannot bleed onto the sclera.
        if pupil_img is not None:
            pupil_r = max(0.02, disc_r * self.PUPIL_RATIO * self.pupil_scale)
            k_pupil = self.PUPIL_TEX_RADIUS / pupil_r
            ptex = _new_node(tree, "ShaderNodeTexImage", "A2_Pupil_M", (-1820, 60))
            ptex.image = pupil_img
            tree.links.new(fitted(k_pupil, "A2_PupilUV", (-2000, 60)),
                           ptex.inputs["Vector"])
            psplit = _new_node(tree, "ShaderNodeSeparateColor", "A2_PupilSplit",
                               (-1640, 60))
            tree.links.new(ptex.outputs["Color"], psplit.inputs["Color"])
            pinv = _new_node(tree, "ShaderNodeMath", "A2_Pupil_OneMinusG",
                             (-1460, 60))
            pinv.operation = "SUBTRACT"
            pinv.inputs[0].default_value = 1.0
            tree.links.new(psplit.outputs["Green"], pinv.inputs[1])
            prange = _new_node(tree, "ShaderNodeMapRange", "A2_PupilRange",
                               (-1280, 60))
            prange.interpolation_type = "SMOOTHSTEP"
            prange.inputs["From Min"].default_value = 0.35
            prange.inputs["From Max"].default_value = 0.75
            prange.inputs["To Min"].default_value = 0.0
            prange.inputs["To Max"].default_value = 1.0
            tree.links.new(pinv.outputs["Value"], prange.inputs["Value"])
            pfac = _new_node(tree, "ShaderNodeMath", "A2_Pupil_OnDisc", (-1100, 60))
            pfac.operation = "MULTIPLY"
            pfac.use_clamp = True
            tree.links.new(prange.outputs["Result"], pfac.inputs[0])
            tree.links.new(disc.outputs["Result"], pfac.inputs[1])
            pmix = _new_node(tree, "ShaderNodeMixRGB", "A2_Pupil", (-160, 260))
            pmix.blend_type = "MIX"
            pmix.inputs["Color2"].default_value = col("Pupil_Color",
                                                      self.pupil_color[:3])
            tree.links.new(pfac.outputs["Value"], pmix.inputs["Factor"])
            tree.links.new(out_socket, pmix.inputs["Color1"])
            out_socket = pmix.outputs["Color"]
            report.assigned["pupil"] = str(self.pupil_texture)
            report.used_images.append(pupil_img.name)
            report.notes.append(
                f"pupil from {Path(self.pupil_texture).name} (inverted green, "
                f"r={pupil_r:.3f}, fitted x{k_pupil:.2f})"
            )

        # EyeAO / tear maps, when the material actually ships them.
        if "custom" in md.resolved:
            ao = _new_node(tree, "ShaderNodeTexImage", "A2_EyeAO_CSTM", (-620, -180))
            ao.image = md.resolved["custom"]
            mix = _new_node(tree, "ShaderNodeMixRGB", "A2_EyeAO_Mix", (280, 260))
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = float(md.scalars.get("EyeAO_intensity", 0.4))
            tree.links.new(out_socket, mix.inputs["Color1"])
            tree.links.new(ao.outputs["Color"], mix.inputs["Color2"])
            out_socket = mix.outputs["Color"]

        tree.links.new(out_socket, bsdf.inputs["Base Color"])
        _set_value(bsdf, ("Roughness",),
                   float(md.scalars.get("IrisRoughness", 0.08)))
        _set_value(bsdf, ("Metallic",), 0.0)
        _set_value(bsdf, ("Specular IOR Level", "Specular"),
                   float(md.scalars.get("IrisSpecularity", 0.4)))
        _set_value(bsdf, ("IOR",), float(md.scalars.get("IoR", 1.38)))
        # Iris_Scale / Pupil_Scale are applied above as the size scalars they
        # measurably are; the exact UE reading still lives in the unextracted
        # Material/Chr/MM/Cstm/CMM_Eye chain.
        self._apply_emission(tree, bsdf, md, report)

    #: Base colour for the lash shells.  The maps are *masks* (see
    #: :meth:`_build_eye_ao`), so they carry no colour of their own and the real
    #: lash tone lives in ``Common/Materials/MI_GF_Head`` plus ``Common/CustomMI/
    #: Eyelash/Eyelash_00..10``, none of which is extracted.  Until then the
    #: shells render as the near-black lash tone the game shows.
    LASH_COLOR = (0.055, 0.048, 0.052, 1.0)

    def _build_eye_ao(self, material, tree, bsdf, md, report) -> None:
        """Eyelash / eye-AO / tear-line overlay over the eyeball.

        The JSONs spell the semantics out, and it is *not* what the texture
        suggests: ``Eye_AO_M_*`` is 512x512 black with green-only art (mean ink
        rgb ~= (0.03, 0.71, 0.07)) and ``alpha`` flat at 1.0, i.e. the **green
        channel is the mask**.  Both consumers agree:

        * lash/eye-AO shells (``GF_Head_*_EyeAO``, ``MI_Eye_AO_0#``) are
          ``BlendMode = 1`` / ``EBlendMode::BLEND_Masked`` with
          ``OpacityMaskClipValue = 0.33333`` - a *clipped* lash-stroke mask,
          lash opacity scaled by ``EyelashUpper/Lower_intensity`` (per lid) and
          ``EyeLashAO_Opacity``, tinted by ``EyeAO_Color``;
        * ``MI_EyeAO_Tear`` is ``BlendMode = 2`` /
          ``BLEND_TranslucentGreyTransmittance`` - genuinely translucent, its
          alpha scaled by ``Tearline_Intensity`` and tinted by
          ``Tearline_Color``.

        Reading ``alpha`` instead (flat 1.0) is what hid the eyeballs: the shell
        would render as an opaque visor across the whole eye, and the ray cast
        confirmed it sat 0.7-4 mm *in front* of the eyeball at the pupil.

        Verified on ``GF_Head_002``: the mask reads >0.333 over 98% of the
        shell's rendered footprint (median 0.62), so the lashes draw as line art
        around the eye instead of a flat visor.  The eyeballs - which measured
        **0 px before this fix** - now show iris through the gap: 19 px at a
        600x800 portrait framing (8 left / 11 right, rows 381-404), and 6 px of
        a 32-px aperture in the tighter 860x1150 face shot, the rest of the
        aperture being upper and lower lash.  ``scripts/probe_eye.py``
        reproduces both numbers.

        The lash strokes only register with ~10% of the shell's camera-facing
        faces at ``UVMap``, which is why they read as a heavy band rather than
        individual hairs; the per-shell UV transform (``Tearline_Scale`` /
        ``Tearline_Offset`` name the idea on the tear line) lives in the
        unextracted ``Common/Materials/MI_GF_Head``.
        """
        img = self._image_for(md, "base_color", report)
        if img is None:
            report.warnings.append("eye-AO map unresolved")
            _set_value(bsdf, ("Alpha",), 0.0)
            return
        tex = _new_node(tree, "ShaderNodeTexImage", "A2_EyeAO_Map", (-680, 260))
        tex.image = img

        def col(key, fallback):
            c = md.colors.get(key)
            if isinstance(c, dict):
                return (float(c.get("R", fallback[0])),
                        float(c.get("G", fallback[1])),
                        float(c.get("B", fallback[2])), 1.0)
            return (*fallback, 1.0)

        def scalar(*keys, default):
            for key in keys:
                value = md.scalars.get(key)
                if value:
                    return float(value)
            return default

        tear = _is_translucent(md) or "tear" in material.name.lower()
        if tear:
            tree.nodes.clear()
            transparent = _new_node(tree, "ShaderNodeBsdfTransparent", "A2_TearTransparent", (420, 0))
            output = _new_node(tree, "ShaderNodeOutputMaterial", "A2_Output", (720, 0))
            tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
            report.notes.append("tear-line overlay hidden until its UV registration is extracted")
            return
        else:
            # ``EyeAO_Color`` is C5D5E4 with A == 0.0, i.e. *switched off* by this
            # project's own convention, so fall back to the lash tone.
            eye_ao = md.colors.get("EyeAO_Color")
            if isinstance(eye_ao, dict) and float(eye_ao.get("A", 0.0)) > 0.0:
                base = col("EyeAO_Color", self.LASH_COLOR[:3])
            else:
                base = self.LASH_COLOR
            upper = scalar("EyelashUpper_intensity", default=1.0)
            lower = scalar("EyelashLower_intensity", default=1.0)
            strength = scalar("EyeLashAO_Opacity", default=1.0) * (upper + lower) / 2.0
            label = "eyelash"

        # The mask is the *green* channel (the only non-zero one in the atlas);
        # alpha is flat 1.0 and luminance only tracks the same green art.
        sep = _new_node(tree, "ShaderNodeSeparateColor", "A2_LashMaskRGB", (-500, 60))
        sep.mode = "RGB"
        tree.links.new(tex.outputs["Color"], sep.inputs["Color"])
        scale = _new_node(tree, "ShaderNodeMath", "A2_LashOpacity", (-320, 60))
        scale.operation = "MULTIPLY"
        scale.use_clamp = True
        scale.inputs[1].default_value = strength
        tree.links.new(sep.outputs["Green"], scale.inputs[0])
        tree.links.new(scale.outputs["Value"], bsdf.inputs["Alpha"])

        _set_value(bsdf, ("Base Color",), base)
        _set_value(bsdf, ("Roughness",),
                   float(md.scalars.get("Tearline_Roughness")
                         or md.scalars.get("EyeAO_Roughness") or 0.6))
        _set_value(bsdf, ("Metallic",), 0.0)
        _set_value(bsdf, ("Specular IOR Level", "Specular"),
                   float(md.scalars.get("Tearline_Specular", 0.3)))
        report.notes.append(
            f"{label} overlay: green channel of the atlas -> Alpha "
            f"(x{strength:g}; alpha is flat 1.0 and carries no shape), "
            + ("blended translucent" if tear else "alpha-clipped strokes")
        )

    # -- helpers ----------------------------------------------------------- #

    def _image_for(self, md: MaterialDef, role: str, report: MaterialReport,
                   fallback: Optional[Path] = None):
        path = md.resolved.get(role)
        if path is None and fallback is not None:
            path = Path(fallback)
            report.notes.append(f"{role}: borrowed skin fallback '{path.name}'")
        if path is None:
            if role in md.unresolved:
                report.missing.append(f"{role}:{md.unresolved[role]}")
            return None
        try:
            img = _load_image(path, NON_COLOR if role in ("normal", "packed", "custom") else SRGB)
        except (RuntimeError, OSError) as exc:
            report.missing.append(f"{role}({exc})")
            return None
        report.assigned[role] = str(path)
        report.used_images.append(img.name)
        return img

    @staticmethod
    def _needs_alpha(md: MaterialDef) -> bool:
        overrides = (md.properties or {}).get("BasePropertyOverrides", {})
        blend = str(overrides.get("BlendMode", "")) if isinstance(overrides, dict) else ""
        if "Masked" in blend or "Translucent" in blend:
            return True
        return bool(md.properties.get("IsTranslucent", False))

    def _alpha_source(self, tree, md: MaterialDef, report: MaterialReport):
        """Best available alpha: a dedicated mask, else the base-colour alpha.

        The custom-colour (``CSTM``) mask is deliberately **not** a candidate -
        it is an opaque channel selector, not an opacity map, and letting it win
        here is what hid every ``_DO`` cutout.
        """
        mask = md.resolved.get("opacity")
        if mask is not None:
            try:
                img = _load_image(mask, NON_COLOR)
            except (RuntimeError, OSError):
                img = None
            if img is not None:
                tex = self._find_image_node(tree, img)
                if tex is None:
                    tex = _new_node(tree, "ShaderNodeTexImage", "A2_AlphaMask", (140, -420))
                    tex.image = img
                report.assigned.setdefault("opacity", str(mask))
                report.used_images.append(img.name)
                return tex.outputs["Alpha"]
        if not md.uses_base_alpha:
            return None
        base_node = self._find_image_node_by_name(tree, "A2_BaseColor_D")
        if base_node is not None:
            report.notes.append("alpha from base-colour map (Use_BaseColorTexAlpha)")
            return base_node.outputs["Alpha"]
        return None

    @staticmethod
    def _find_image_node(tree, image):
        for node in tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is image:
                return node
        return None

    @staticmethod
    def _find_image_node_by_name(tree, name):
        node = tree.nodes.get(name)
        if node is not None and node.type == "TEX_IMAGE":
            return node
        return None

    def _apply_blend_settings(self, material, md: MaterialDef, report: MaterialReport) -> None:
        """Map UE blend modes onto Blender's surface settings (version-tolerant)."""
        overrides = (md.properties or {}).get("BasePropertyOverrides", {})
        blend = str(overrides.get("BlendMode", "")) if isinstance(overrides, dict) else ""
        # Lash/eye-AO shells split by their own declared blend mode: the tear
        # line is ``BLEND_TranslucentGreyTransmittance`` and must be *blended*,
        # while the lashes are ``BLEND_Masked`` and must be *alpha-clipped* at
        # ``OpacityMaskClipValue`` (0.33333) or the whole shell renders as an
        # opaque visor over the eye.  Set both spellings: ``blend_method`` is
        # what 3.x/4.0 read, 4.2+ EEVEE Next reads ``surface_render_method``.
        if report.kind == "eye_ao":
            material.use_backface_culling = False
            if _is_translucent(md):
                _set_enum(material, "blend_method", "BLEND")
                _set_enum(material, "surface_render_method", "BLENDED")
                report.notes.append("blended translucent surface")
            else:
                if not _set_enum(material, "blend_method", "CLIP"):
                    # 4.2+/5.x renamed CLIP to HASHED and moved the real switch
                    # to ``surface_render_method``.
                    _set_enum(material, "blend_method", "HASHED")
                _set_enum(material, "surface_render_method", "DITHERED")
                try:
                    material.alpha_threshold = float(md.opacity_mask_clip)
                except (AttributeError, TypeError):
                    pass
                report.notes.append("alpha-clipped lash surface")
            return

        # Hair stays alpha-clipped like the game's ``BLEND_Masked``: measured on
        # the rebuilt file that is 98% crown coverage with 96% of it fully
        # opaque, whereas rendering the same mask *blended* collapses to 2%
        # coverage (the cards are two-sided and overlap, so a blended stack
        # never accumulates and the hair turns into a pale film).  The soft
        # strand mask is instead filled by ``hair_opacity_floor``.
        masked = ("Masked" in blend or report.kind in ("hair", "hide")
                  or self._needs_alpha(md))

        # Aion2 relies on two-sided cloth/hair; keep it visible from both sides.
        material.use_backface_culling = False

        # Blender 4.2+ replaced CLIP with DITHERED and moved the property to
        # ``surface_render_method``; set whichever this build understands.
        if not _set_enum(material, "blend_method", "CLIP" if masked else "OPAQUE"):
            _set_enum(material, "blend_method", "BLEND" if masked else "OPAQUE")
        _set_enum(material, "surface_render_method",
                  "DITHERED" if masked else "OPAQUE")
        try:
            material.alpha_threshold = float(md.opacity_mask_clip)
        except (AttributeError, TypeError):
            pass
        if masked:
            report.notes.append("alpha-clip surface")
