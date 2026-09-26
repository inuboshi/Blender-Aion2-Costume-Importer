"""Projecting Aion2's face decals onto the head mesh.

Aion2's face shader (``MF_CombinePartsForHead`` -> ``CM_Head_EyeBrow``) does not
sample the eyebrow out of head UV space.  The makeup decals *are* authored in
head UV space - ``MK_Head_EEE_T2A``'s eyeliner really is drawn where UV0 puts the
eyes - but ``Cstm_EyeBrow_T2A`` is a texture *array of single brows*, and the
shader stamps one onto each side of the forehead with

    EyeBrow_ScaleU / EyeBrow_ScaleV / EyeBrow_Rotate / Eyebrow_Distance /
    EyeBrow_Height / EyeBrow_Mirror

so its atlas coordinates and the head's UV0 are unrelated.  ``CM_Head_EyeBrow``
itself is not in the export, and the seven placement scalars do not admit an
unambiguous reconstruction (``Eyebrow_Distance`` is 0.0654 on head 002 and
0.0206 on head 019 - the same mesh - and ``EyeBrow_Height`` is ~-1.03, which is
neither metres nor a UV coordinate).

What *is* unambiguous is the anatomy, so the projection is derived from it.  Two
measurements come off the mesh itself (the eyeball's own bounding box and the
eye-AO shell that traces the visible aperture), and the brow is placed above the
lid.  Every constant below is therefore a proportion of a measured feature
rather than an absolute position, which is what lets the same code serve all 50
odd head styles.

There is a second reason this cannot be a material node graph: object
coordinates deform with the mesh, so a projection evaluated in the shader would
stay glued to the head's origin and slide off the brow as soon as the head is
posed.  Baking coordinates per vertex keeps the brow attached to the skin.

The module is ``bpy``-free so the geometry can be checked without Blender.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #

#: Brow lower edge, above the *visible* upper lid (metres).  Scaled to the
#: subject's own anatomy by :func:`brow_box`, and the one value worth tuning:
#: the shipped faces are ~1:1 with a real head, where the brow sits 15-20 mm
#: above the lid margin and ~25 mm above the eye's centre.
LID_GAP = 0.013

#: Vertical extent of the whole brow, i.e. of the atlas's alpha bounding box.
#: The atlas brow is a wide flat shape (roughly 6:1), so this is also roughly
#: its thickness at the thick (medial) end.
BROW_HEIGHT = 0.009

#: How far the brow reaches inboard of the eyeball's medial edge, and past its
#: lateral edge.  The medial inset is what gives the brow a proper head; the
#: lateral extension is the thin tail.
INNER_INSET = 0.011
OUTER_EXTEND = 0.019

#: The brow may never cross the midline (that is what ``EyeBrow_Mirror`` is).
MIDLINE_GUARD = 0.002

#: Verts whose normal points away from the face by more than this are excluded,
#: which is what keeps the back of the skull from catching the brow (an ortho
#: projection in x/z otherwise paints a second brow behind the head).  Negative
#: because the character faces ``-Y``.
FRONT_NORMAL = -0.15

#: How far outside the brow box the :func:`project_brow` weight fades to 0, as
#: a fraction of the box's own size.  The gate exists because a per-vertex UV
#: cannot express "no brow here": any triangle with one vertex on the brow and
#: one outside it interpolates *through the whole sprite* on the way, leaving a
#: thin smear behind (190 such triangles on head 002).  So off-brow vertices get
#: an ordinary clamped coordinate - continuous with their neighbours, sweeping
#: nothing - and a weight that removes them from the blend outright.
#:
#: Vertical fade, as a fraction of the brow's height (~3 mm).
GATE_MARGIN = 0.35

#: Lateral fade, as a fraction of the *brow's length*.  This one has to be much
#: tighter than the vertical margin: the box is ~56 mm long, so the old 0.35
#: put weight on vertices 70 mm out - i.e. on the temple and the side of the
#: skull, where the clamped atlas coordinate smeared the brow's tail into the
#: horizontal streaks that read as "hair cards poking through".
GATE_MARGIN_X = 0.08


@dataclass(frozen=True)
class BrowBox:
    """Where a single brow sits on one side of the face.

    ``x_in``/``x_out`` are distances from the midline (always positive - the
    mirrored side is produced by ``|x|``), ``z`` is height.
    """

    x_in: float
    x_out: float
    z_lo: float
    z_hi: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x_in, self.x_out, self.z_lo, self.z_hi)


@dataclass(frozen=True)
class HeadAnchors:
    """The measurements the projection is built from."""

    #: Distance from the midline to the eyeball's medial / lateral edge.
    eye_inner: float
    eye_outer: float
    #: Height of the visible upper lid (the eye-AO shell's top, when present).
    lid_z: float
    #: Height of the eyeball's centre, for the report.
    eye_z: float

    def describe(self) -> str:
        return (f"eye centre z={self.eye_z:.4f}, lid z={self.lid_z:.4f}, "
                f"eye |x| {self.eye_inner:.4f}..{self.eye_outer:.4f}")


def anchors_from_mesh(
    eye_points: Sequence[Sequence[float]],
    aperture_points: Sequence[Sequence[float]] = (),
) -> Optional[HeadAnchors]:
    """Derive :class:`HeadAnchors` from the eyeball and lid vertices.

    *eye_points* are the vertices of the two ``*_Eye`` primitives, *aperture*
    those of the eye-AO/tear shells (the only geometry that outlines the actual
    eye opening - the eyeball itself is a full sphere, so its top is well above
    the lid).  Only the ``x > 0`` side is measured; the face is symmetric and
    the projection mirrors.
    """
    right = [p for p in eye_points if p[0] > 0.0]
    if not right:
        return None
    xs = [p[0] for p in right]
    zs = [p[2] for p in right]
    eye_inner, eye_outer = min(xs), max(xs)
    eye_z = 0.5 * (min(zs) + max(zs))
    lid_z = max(zs)
    if aperture_points:
        lid_z = max(aperture_points, key=lambda p: p[2])[2]
    return HeadAnchors(eye_inner=eye_inner, eye_outer=eye_outer,
                       lid_z=lid_z, eye_z=eye_z)


def brow_box(anchors: HeadAnchors) -> BrowBox:
    """Place one brow from the measured anatomy."""
    x_in = max(anchors.eye_inner - INNER_INSET, MIDLINE_GUARD)
    x_out = max(anchors.eye_outer + OUTER_EXTEND, x_in + 0.02)
    z_lo = anchors.lid_z + LID_GAP
    return BrowBox(x_in=x_in, x_out=x_out, z_lo=z_lo, z_hi=z_lo + BROW_HEIGHT)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _ramp(value: float, low: float, high: float, fade_low: float, fade_high: float) -> float:
    """1 inside ``[low, high]``, easing to 0 over each side's own fade distance."""
    if low <= value <= high:
        return 1.0
    over, fade = (low - value, fade_low) if value < low else (value - high, fade_high)
    if fade <= 0.0:
        return 0.0
    return max(0.0, 1.0 - over / fade)


def project_brow(
    positions: Iterable[Sequence[float]],
    normals: Iterable[Sequence[float]],
    box: BrowBox,
    content: Tuple[float, float, float, float],
    *,
    margin: float = GATE_MARGIN,
    margin_x: float = GATE_MARGIN_X,
    front_normal: float = FRONT_NORMAL,
) -> List[Tuple[Tuple[float, float], float]]:
    """Per-vertex ``(atlas uv, weight)`` for the eyebrow decal.

    ``content`` is the atlas's own alpha bounding box, so the brow's shape,
    arch and taper (all of which are *in the texture*) land unmodified; only the
    box it is drawn into is chosen here.

    Coordinates are returned in **texture** space: ``v`` runs downwards, as it
    does in the TGA and in glTF.  Blender's UV layers run the other way, so the
    caller flips ``v`` (see :func:`to_blender_uv`).

    The coordinates are defined for **every** vertex - a vertex outside the box
    is projected as if it were on the box's edge, so the field is continuous and
    no triangle interpolates across the sprite.  The returned weight is what
    confines the decal: 1 inside the box on a front-facing vertex, easing to 0
    over ``margin`` of the box's own size outside it, and 0 behind the face.
    """
    u0, u1, v0, v1 = content
    span_u = (u1 - u0) or 1.0
    span_v = (v1 - v0) or 1.0
    span_x = (box.x_out - box.x_in) or 1.0
    span_z = (box.z_hi - box.z_lo) or 1.0
    # How far outside the box still gets *some* weight, in metres.  The medial
    # fade is shortened so it lands exactly on the midline rather than putting
    # weight across it - the mirror is what makes the other brow (see
    # ``MIDLINE_GUARD``), so one decal must never reach x = 0.
    fade_x = max(margin_x, 0.0) * span_x
    fade_z = max(margin, 0.0) * span_z
    fade_in = min(fade_x, box.x_in)

    out: List[Tuple[Tuple[float, float], float]] = []
    for pos, nrm in zip(positions, normals):
        ax = abs(pos[0])
        z = pos[2]
        u = u0 + (_clamp(ax, box.x_in, box.x_out) - box.x_in) / span_x * span_u
        # v grows downwards while z grows upwards: the box top is v0.
        v = v0 + (box.z_hi - _clamp(z, box.z_lo, box.z_hi)) / span_z * span_v
        weight = (_ramp(ax, box.x_in, box.x_out, fade_in, fade_x)
                  * _ramp(z, box.z_lo, box.z_hi, fade_z, fade_z))
        if nrm is not None and nrm[1] > front_normal:
            # Back of the skull: an ortho x/z projection would otherwise paint a
            # second brow behind the head.
            weight = 0.0
        out.append(((u, v), weight))
    return out


def to_blender_uv(uv: Tuple[float, float]) -> Tuple[float, float]:
    """Texture space (v down) -> Blender UV space (v up)."""
    return (uv[0], 1.0 - uv[1])


# --------------------------------------------------------------------------- #
# Atlas content box
# --------------------------------------------------------------------------- #

def tga_alpha_box(path: Path, threshold: float = 0.02) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box of the opaque part of a TGA's **alpha**, as UV fractions.

    Returns ``(u0, u1, v0, v1)`` with ``v`` measured from the **top** of the
    image, or ``None`` when the read fails or the image is empty (in which case
    the caller falls back to the whole texture).

    Only uncompressed true-colour and greyscale TGAs occur in an FModel export,
    which is all this handles - it is deliberately a few lines rather than a
    dependency, because all it needs is the alpha extremes.
    """
    try:
        data = Path(path).read_bytes()
        id_len, _cmap, image_type = data[0], data[1], data[2]
        width, height, bpp, desc = struct.unpack("<HHBB", data[12:18])
    except (OSError, struct.error, IndexError):
        return None
    offset = 18 + id_len
    top_down = bool(desc & 0x20)
    try:
        if image_type == 3:          # greyscale: the sample *is* the mask
            alpha = data[offset:offset + width * height]
            if len(alpha) < width * height:
                return None
        elif image_type == 2 and bpp // 8 in (3, 4):
            nch = bpp // 8
            stride = nch
            raw = data[offset:offset + width * height * nch]
            if len(raw) < width * height * nch:
                return None
            if nch == 4:
                alpha = raw[3::stride]
            else:                     # no alpha channel at all: use luminance
                alpha = raw[0::stride]  # BGRA order, any channel will do
        else:
            return None
    except (struct.error, IndexError):
        return None

    limit = int(threshold * 255.0)
    u0 = width
    u1 = -1
    v0 = height
    v1 = -1
    for row in range(height):
        base = row * width
        line = alpha[base:base + width]
        if max(line) <= limit:
            continue
        if row < v0:
            v0 = row
        v1 = row
        for col in range(width):
            if line[col] > limit:
                if col < u0:
                    u0 = col
                if col > u1:
                    u1 = col
    if u1 < 0:
        return None
    if not top_down:                  # TGA origin is bottom-left by default
        v0, v1 = height - 1 - v1, height - 1 - v0
    u0f, u1f = u0 / width, (u1 + 1) / width
    v0f, v1f = v0 / height, (v1 + 1) / height
    if u1f - u0f < 0.02 or v1f - v0f < 0.02:
        return None
    return (u0f, u1f, v0f, v1f)


def _self_test() -> int:
    """Check the projection's contract. ``python -m aion2.faceproj``."""
    box = BrowBox(x_in=0.005, x_out=0.060, z_lo=1.742, z_hi=1.751)
    content = (0.09, 0.93, 0.20, 0.85)
    up = (0.0, -1.0, 0.0)
    back = (0.0, 1.0, 0.0)
    cases = [
        ((box.x_in, -0.06, box.z_hi), up, (content[0], content[2]), 1.0, "inner/top"),
        ((box.x_out, -0.04, box.z_lo), up, (content[1], content[3]), 1.0, "outer/bottom"),
        ((-0.030, -0.05, 1.7465), up, None, 1.0, "mirrored side"),
        ((0.030, -0.05, 1.7465), up, None, 1.0, "right side"),
        ((0.030, -0.05, 1.7465), back, None, 0.0, "back-facing normal"),
        ((0.001, -0.05, 1.7465), up, None, -1.0, "inside the medial fade"),
        ((0.000, -0.05, 1.7465), up, None, 0.0, "on the midline"),
        ((0.030, -0.05, 1.7000), up, None, 0.0, "below the box"),
        ((0.030, -0.05, 1.9000), up, None, 0.0, "above the box"),
        # Just outside the box still carries a little weight, so the brow's own
        # soft edge is not cut off by the gate.
        ((box.x_out + 0.001, -0.05, 1.7465), up, None, -1.0, "just past the tail"),
    ]
    got = project_brow([c[0] for c in cases], [c[1] for c in cases], box, content)
    failures = 0
    for index, (point, _nrm, expect, weight, label) in enumerate(cases):
        actual, actual_weight = got[index]
        if expect is not None and any(abs(a - b) > 1e-6 for a, b in zip(actual, expect)):
            print(f"  FAIL {label}: {actual} != {expect}")
            failures += 1
        if weight >= 0.0 and abs(actual_weight - weight) > 1e-6:
            print(f"  FAIL {label} weight: {actual_weight} != {weight}")
            failures += 1
    # Weight is a *soft* ramp, not a cliff: a vertex a hair outside the box must
    # still be fading, or the brow's own taper would be clipped.
    for index, label in ((5, "inside the medial fade"), (9, "just past the tail")):
        if not 0.0 < got[index][1] < 1.0:
            print(f"  FAIL gate ramp {label}: {got[index][1]}")
            failures += 1
    # Nothing may reach the midline: that would draw a second brow's worth of
    # decal between the eyes.
    if got[6][1] != 0.0:
        print(f"  FAIL midline weight: {got[6][1]}")
        failures += 1
    # The projection must stay inside the atlas even far off the brow - that is
    # what keeps a triangle from sweeping across the sprite.
    for label, point in (("far inboard", (0.0, -0.05, 1.7465)),
                         ("far outboard", (0.09, -0.05, 1.7465)),
                         ("far below", (0.03, -0.05, 1.60)),
                         ("far above", (0.03, -0.05, 1.90))):
        uv, weight = project_brow([point], [up], box, content)[0]
        if not (content[0] - 1e-6 <= uv[0] <= content[1] + 1e-6
                and content[2] - 1e-6 <= uv[1] <= content[3] + 1e-6):
            print(f"  FAIL {label} left the atlas: {uv}")
            failures += 1
        if weight != 0.0:
            print(f"  FAIL {label} should have no weight: {weight}")
            failures += 1
    # The two mirrored points must share a coordinate, which is what
    # ``EyeBrow_Mirror`` does in the game's shader.
    if any(abs(a - b) > 1e-9 for a, b in zip(got[3][0], got[2][0])):
        print(f"  FAIL mirroring: {got[3]} != {got[2]}")
        failures += 1
    # v runs downwards in texture space while z runs up.
    if not got[0][0][1] < got[1][0][1]:
        print("  FAIL v direction")
        failures += 1
    print(f"faceproj self-test: {len(cases)} cases, {failures} failure(s)")
    return 1 if failures else 0


__all__ = [
    "BROW_HEIGHT", "GATE_MARGIN", "GATE_MARGIN_X", "LID_GAP", "BrowBox",
    "HeadAnchors",
    "anchors_from_mesh", "brow_box", "project_brow", "tga_alpha_box",
    "to_blender_uv",
]


if __name__ == "__main__":  # pragma: no cover - manual helper
    raise SystemExit(_self_test())
