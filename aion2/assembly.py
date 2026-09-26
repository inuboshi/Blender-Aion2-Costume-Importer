"""Import Aion2 glTF exports and assemble them into one coherent scene.

FModel exports one GLB per asset, and - crucially - **each GLB carries only the
subset of the skeleton its own mesh is skinned to**.  A shoulder pad ships 12
bones, a hair piece 52, the body 115, the head 170, and the union over all
pieces is over a thousand.  Importing several pieces therefore yields several
partial armatures that no single file owns.

:func:`merge_armatures` solves this: it snapshots the bones of every imported
armature and writes the *union* into one master armature.  All Aion2 GLBs put
their skeletons at the origin with an identity transform and an identical
``Root`` bone, so bone rest data from different files is directly comparable and
can be merged without any space conversion.  Meshes are then re-targeted to the
master and their redundant armatures deleted, while their vertex groups keep
working unchanged because Blender binds weights by bone *name*.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bpy

from .glb import summarize
from .materials import Aion2MaterialBuilder, MaterialReport
from .paths import (
    BROW_MASK_ATTRIBUTE,
    BROW_UV_LAYER,
    MAKEUP_PARTS,
    Eyebrow,
    GamePaths,
    MakeupLayer,
    MaterialDef,
    eyebrow_array_asset,
    eye_ao_texture_ue,
    is_eye_ao_material,
    is_head_material,
    is_layered_body_material,
    makeup_array_asset,
)
from .spec import CharacterSpec
from . import faceproj

__all__ = [
    "Aion2Assembler",
    "AssemblyReport",
    "CharacterSpec",
    "MaterialLibrary",
    "PieceInfo",
    "assign_materials",
    "bake_brow_uvs",
    "import_glb",
    "merge_armatures",
    "purge_helper_objects",
    "purge_previous_build",
]

#: Collection the Blender glTF importer drops helper objects into.  It always
#: contains a stray material-less ``Icosphere`` for these Aion2 files; leaving it
#: in the scene is what produced the junk object in the earlier hand-built blend.
HELPER_COLLECTION = "glTF_not_exported"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

@dataclass
class PieceInfo:
    source: Path
    mesh_objects: List[str] = field(default_factory=list)
    armature: Optional[str] = None
    collection: Optional[str] = None
    materials: List[str] = field(default_factory=list)
    vertices: int = 0
    #: CharacterSpec section this piece came from (Basebody/Head/Hair/Armor/Extra).
    label: str = ""

    @property
    def supplies_base_body(self) -> bool:
        """True when this piece ships its own copy of the body/legs shell.

        Aion2 armour meshes bundle the skin underneath them - the 0096 pants
        carry a ``GF_BaseBody`` shell of exactly the same size as the standalone
        base legs - so importing a base body alongside armour leaves two shells
        fighting for the same surface.
        """
        return any(_is_body_shell(m) for m in self.materials)

    def __str__(self) -> str:
        return (f"{self.source.name}: meshes={self.mesh_objects} rig={self.armature} "
                f"mats={self.materials}")


def _is_body_shell(material_name: str) -> bool:
    """Base-body/naked-skin material names embedded in armour meshes."""
    up = (material_name or "").upper()
    return (up.startswith(("GF_BASEBODY", "GM_BASEBODY"))
            or up.startswith(("MI_GF_BASE_", "MI_GM_BASE_")))


@dataclass
class AssemblyReport:
    pieces: List[PieceInfo] = field(default_factory=list)
    materials: List[MaterialReport] = field(default_factory=list)
    master_armature: Optional[str] = None
    bones_added: int = 0
    total_bones: int = 0
    purged_helpers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Informational only - e.g. layered body shells redirected onto skin maps.
    notes: List[str] = field(default_factory=list)
    #: Pieces that ship their own copy of the base body/legs shell.
    redundant_shells: List[str] = field(default_factory=list)
    #: Objects removed from a previous build so this one replaces it.
    replaced_objects: List[str] = field(default_factory=list)

    @property
    def body_providers(self) -> List[PieceInfo]:
        return [p for p in self.pieces if p.supplies_base_body]

    @property
    def missing_maps(self) -> List[str]:
        out = []
        for m in self.materials:
            out.extend(f"{m.material}: {x}" for x in m.missing)
        return out


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _get_collection(name: str, parent: Optional["bpy.types.Collection"] = None):
    """Fetch or create a collection, linked under *parent* (or the scene)."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    elif parent is not None and coll.name not in {
        c.name for c in parent.children
    }:
        parent.children.link(coll)
    return coll


def _link_only(obj, collection) -> None:
    """Move *obj* into *collection*, preserving no other membership."""
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for other in list(obj.users_collection):
        if other is not collection:
            other.objects.unlink(obj)


def purge_helper_objects(report: Optional[AssemblyReport] = None) -> List[str]:
    """Delete the importer's placeholder collection and everything in it."""
    removed: List[str] = []
    coll = bpy.data.collections.get(HELPER_COLLECTION)
    if coll is None:
        return removed
    for obj in list(coll.objects):
        removed.append(obj.name)
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except RuntimeError:
            pass
    try:
        bpy.data.collections.remove(coll)
    except RuntimeError:
        pass
    if report is not None:
        report.purged_helpers.extend(removed)
    return removed


def _new_objects(before: Sequence[str]) -> List["bpy.types.Object"]:
    seen = set(before)
    return [o for o in bpy.data.objects if o.name not in seen]


def _remove_datablock(item) -> bool:
    """Delete an orphaned datablock whichever collection owns it."""
    if item is None:
        return False
    try:
        if item.users > 0:
            return False
    except ReferenceError:
        return False
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                       bpy.data.images, bpy.data.node_groups, bpy.data.curves):
        try:
            collection.remove(item)
            return True
        except (TypeError, RuntimeError, ReferenceError):
            continue
    return False


def purge_previous_build(root_name: str = "Aion2") -> List[str]:
    """Remove a previous build's collection tree and everything inside it.

    Without this the build operator is not idempotent: pressing it twice imports
    a second copy of the whole character, which shows up as two heads, two hairs
    and armour fighting over the same surface.  Only objects under the Aion2
    root collection are touched, so the user's own scene survives.

    The *datablocks* are purged too.  Deleting an object leaves its mesh,
    armature, materials and images behind as zero-user orphans that still hold
    their names, so the next import would silently rename everything to
    ``GF_BaseHead.001`` - which looks like a duplicate even when it is not.
    """
    root = bpy.data.collections.get(root_name)
    if root is None:
        return []

    removed: List[str] = []
    orphans: List[object] = []
    for obj in list(root.all_objects):
        removed.append(obj.name)
        try:
            orphans.append(obj.data)
            if obj.type == "MESH":
                orphans.extend(slot.material for slot in obj.material_slots)
        except (ReferenceError, AttributeError):
            pass
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except (RuntimeError, ReferenceError):
            pass

    for child in list(root.children):
        try:
            bpy.data.collections.remove(child)
        except RuntimeError:
            pass
    try:
        bpy.data.collections.remove(root)
    except RuntimeError:
        pass

    # Now that the objects are gone, drop their datablocks and any images or
    # node groups those materials were the last users of.
    for item in orphans:
        _remove_datablock(item)
    for orphan in list(bpy.data.images) + list(bpy.data.materials):
        if orphan.users == 0:
            _remove_datablock(orphan)
    return removed


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #

def import_glb(path: Path, *, report: Optional[AssemblyReport] = None) -> PieceInfo:
    """Import one Aion2 GLB, dropping the importer's placeholder objects."""
    source = Path(path)
    before = [o.name for o in bpy.data.objects]
    bpy.ops.import_scene.gltf(filepath=str(source))
    new_objs = _new_objects(before)

    info = PieceInfo(source=source)
    for obj in new_objs:
        if obj.type == "MESH":
            info.mesh_objects.append(obj.name)
        elif obj.type == "ARMATURE" and info.armature is None:
            info.armature = obj.name

    # Read the source GLB cheaply to learn what skin shells it bundles.
    summary = summarize(source)
    if summary.error is None:
        info.materials = list(summary.materials)
        info.vertices = summary.vertices

    # Safe to call every import; removes the Icosphere and friends.
    purge_helper_objects(report)
    return info


# --------------------------------------------------------------------------- #
# Rig merging
# --------------------------------------------------------------------------- #

def _ensure_editable(obj: "bpy.types.Object") -> None:
    """Make *obj* the active, visible, selectable object of the view layer.

    ``mode_set`` refuses to enter Edit Mode on a hidden or excluded object, and
    the import step hides donor rigs, so visibility is forced back on here.
    """
    try:
        obj.hide_viewport = False
    except AttributeError:
        pass
    try:
        obj.hide_set(False)
    except (AttributeError, RuntimeError):
        pass
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _snapshot_bones(arm_obj: "bpy.types.Object") -> List[dict]:
    """Read a bone hierarchy out of an armature as plain data."""
    view_layer = bpy.context.view_layer
    previous = view_layer.objects.active
    _ensure_editable(arm_obj)
    snapshot: List[dict] = []
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for eb in arm_obj.data.edit_bones:
            snapshot.append({
                "name": eb.name,
                "head": tuple(eb.head),
                "tail": tuple(eb.tail),
                "roll": float(eb.roll),
                "parent": eb.parent.name if eb.parent else None,
                "use_connect": bool(eb.use_connect),
                "use_deform": bool(eb.use_deform),
            })
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        if previous is not None:
            view_layer.objects.active = previous
    return snapshot


def _write_bones(master: "bpy.types.Object", snapshots: Iterable[dict]) -> int:
    """Add every bone in *snapshots* that the master armature lacks.

    Parents are resolved in repeated passes because a snapshot taken from one
    file can reference a parent that only another file defines.
    """
    view_layer = bpy.context.view_layer
    previous = view_layer.objects.active
    _ensure_editable(master)
    added = 0
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bones = master.data.edit_bones
        pending = [s for s in snapshots if s["name"] not in bones]
        for _pass in range(8):
            if not pending:
                break
            progressed = False
            still: List[dict] = []
            for snap in pending:
                if snap["name"] in bones:
                    progressed = True
                    continue
                parent_name = snap["parent"]
                # Defer until the parent exists, but never block a root bone.
                if parent_name and parent_name not in bones:
                    still.append(snap)
                    continue
                eb = bones.new(snap["name"])
                eb.head = snap["head"]
                eb.tail = snap["tail"]
                try:
                    eb.roll = snap["roll"]
                except (AttributeError, TypeError):
                    pass
                eb.use_deform = snap["use_deform"]
                if parent_name:
                    eb.parent = bones[parent_name]
                    eb.use_connect = snap["use_connect"]
                added += 1
                progressed = True
            if not progressed:
                # Orphans with unresolvable parents - add them unparented.
                for snap in still:
                    eb = bones.new(snap["name"])
                    eb.head = snap["head"]
                    eb.tail = snap["tail"]
                    added += 1
                break
            pending = still
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        if previous is not None:
            view_layer.objects.active = previous
    return added


def merge_armatures(
    armatures: Sequence["bpy.types.Object"],
    report: Optional[AssemblyReport] = None,
) -> Optional["bpy.types.Object"]:
    """Merge *armatures* into a single master rig and remove the rest.

    The armature with the most bones becomes the master; every other armature's
    bones are unioned into it.  Meshes are re-pointed at the master first, so
    their ``Armature`` modifiers and parenting stay valid.
    """
    armatures = [a for a in armatures if a and a.type == "ARMATURE"]
    if not armatures:
        return None

    master = max(armatures, key=lambda a: len(a.data.bones))
    others = [a for a in armatures if a is not master]

    snapshots: List[dict] = []
    for arm in others:
        snapshots.extend(_snapshot_bones(arm))
    added = _write_bones(master, snapshots) if snapshots else 0

    # Re-point meshes at the master before deleting the donor armatures.
    for arm in others:
        for child in list(arm.children):
            _retarget_mesh(child, arm, master)

    for arm in others:
        try:
            bpy.data.objects.remove(arm, do_unlink=True)
        except RuntimeError:
            pass

    if report is not None:
        report.master_armature = master.name
        report.bones_added = added
        report.total_bones = len(master.data.bones)
    return master


def _retarget_mesh(mesh_obj: "bpy.types.Object", old: "bpy.types.Object",
                   master: "bpy.types.Object") -> None:
    """Move *mesh_obj* off *old* and onto *master* without breaking its skin."""
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE" and mod.object is old:
            mod.object = master
    if mesh_obj.parent is old:
        matrix = mesh_obj.matrix_world.copy()
        mesh_obj.parent = master
        mesh_obj.matrix_parent_inverse = master.matrix_world.inverted()
        mesh_obj.matrix_world = matrix


#: Bones that lift the upper lid.  ``LidUA..E`` run along the lash line from
#: inner (A) to outer (E); ``LidCover*`` shape the crease above it.
UPPER_LID_RE = re.compile(r"^b_[RL]_LidU[A-E]_\d+$")


def open_eyes_pose(rig: "bpy.types.Object", amount: float = 0.6,
                   report: Optional[AssemblyReport] = None) -> List[str]:
    """Rotate the upper-lid bones open by *amount* radians (pose, not rest).

    The exported bind pose carries the lids nearly shut - a lens-shaped opening
    only ~4-6 mm tall - and no eye-open morph exists among the head's shape
    keys, so posing the lid bones is the only way the eyes read as open.
    Every ``b_{R,L}_LidU*`` bone points along the lash line (inner to outer,
    roughly -X for the right eye), so rotating around the bone's own local Y
    rolls the lid up over the eyeball.  Rotations are set on top of whatever
    pose existed (the rig ships at rest), and ``amount`` 0 disables.
    """
    if amount <= 0.0 or rig is None or rig.type != "ARMATURE":
        return []
    # Pose transforms need the depsgraph-free pose API; ensure the rig is not
    # in Edit Mode (merge_armatures leaves it in Object Mode).
    names = [b.name for b in rig.pose.bones if UPPER_LID_RE.match(b.name)]
    if not names:
        if report is not None:
            report.warnings.append("open_eyes: no upper-lid bones found")
        return []
    for name in names:
        bone = rig.pose.bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, amount, 0.0)
    if report is not None:
        report.notes.append(
            f"open_eyes: posed {len(names)} upper-lid bones "
            f"({', '.join(sorted(names))})"
        )
    return sorted(names)


# --------------------------------------------------------------------------- #
# Material wiring
# --------------------------------------------------------------------------- #

class MaterialLibrary:
    """Maps GLB material names onto the FModel material JSON definitions.

    Also repairs "layered body" shells.  ``MI_GF_Base_Body`` / ``MI_GF_Base_Pants``
    are the base body wearing whichever armour was equipped when FModel captured
    the material instance, so their JSON points at *that* armour set's maps (set
    ``0103`` for the vanilla GF export).  Left alone they paint bare skin with
    cloth, so those materials are redirected onto the character's own skin maps
    and forced to skin shading.
    """

    def __init__(self, gpaths: GamePaths, gender: str = "GF") -> None:
        self.gpaths = gpaths
        self.gender = gender.upper()
        self._index: Dict[str, Path] = {}
        self._collisions: List[str] = []
        self._built = False
        self._skin_overrides: Optional[Dict[str, Path]] = None
        self._head_overrides: Optional[Dict[str, Path]] = None
        self._iris: Optional[Path] = None
        self._pupil: Optional[Path] = None
        self._alpha_boxes: Dict[str, Tuple[float, float, float, float]] = {}
        self.repaired: List[str] = []
        self.heads_repaired: List[str] = []

    # -- layered body repair ----------------------------------------------- #

    def _skin_maps(self) -> Dict[str, Path]:
        """The character's own skin maps, used for layered body shells."""
        if self._skin_overrides is not None:
            return self._skin_overrides
        base = f"/Game/Character/Player/{self.gender}/Basebody/Materials/{self.gender}_BaseBody"
        found: Dict[str, Path] = {}
        for role, suffix in (("base_color", "_D"), ("normal", "_N"),
                             ("packed", "_ARSC")):
            hit = self.gpaths.resolve(f"{base}{suffix}.{self.gender}_BaseBody{suffix}")
            if hit is not None:
                found[role] = hit
        self._skin_overrides = found
        return found

    def _repair_layered_body(self, md: MaterialDef) -> MaterialDef:
        skin = self._skin_maps()
        if not skin:
            return md
        for role, path in skin.items():
            md.resolved[role] = path
            md.unresolved.pop(role, None)
        # Skin packs specular in blue (ARSC), never metallic, and shades as skin.
        md.packed_kind_override = "ARSC"
        md.force_kind = "skin"
        if md.name not in self.repaired:
            self.repaired.append(md.name)
        return md

    def _head_maps(self) -> Dict[str, Path]:
        """The shared face maps every per-style head material composes onto.

        ``Common/GF_Head_D`` (+ ``GF_Head_ARSC``) sits next to ``MI_GF_Head``,
        the parent material instance of every ``MI_GF_Head_###`` - the same
        adjacency ``MI_GF_Base_Body``/``GF_BaseBody_D`` has in ``Basebody/``.
        ``Basebody/Materials/GF_BaseHead_D`` (+ ``_N``) is the alternate pair
        used by the ``GF_BaseHead`` mesh and is tried second.
        """
        if self._head_overrides is not None:
            return self._head_overrides
        gender = self.gender
        found: Dict[str, Path] = {}
        for base in (
            f"/Game/Character/Player/Common/{gender}_Head",
            f"/Game/Character/Player/{gender}/Basebody/Materials/{gender}_BaseHead",
        ):
            stem = base.rsplit("/", 1)[1]
            for role, suffix in (("base_color", "_D"), ("normal", "_N"),
                                 ("packed", "_ARSC")):
                if role in found:
                    continue
                hit = self.gpaths.resolve(f"{base}{suffix}.{stem}{suffix}")
                if hit is not None:
                    found[role] = hit
        self._head_overrides = found
        return found

    def _repair_head(self, md: MaterialDef) -> MaterialDef:
        """Give a face material the shared head maps it does not carry itself.

        Only fills roles the instance is missing, so the three styles that ship
        a real ``_D`` (``GF_Head_019``/``024``/``041``) keep their own albedo.
        """
        head = self._head_maps()
        if not head:
            return md
        filled: List[str] = []
        for role, path in head.items():
            if role in md.resolved:
                continue
            md.resolved[role] = path
            md.unresolved.pop(role, None)
            md.mask_not_albedo.pop(role, None)
            filled.append(role)
        if filled:
            md.notes.append(
                "shared head maps -> " + ", ".join(sorted(filled))
            )
            if md.name not in self.heads_repaired:
                self.heads_repaired.append(md.name)
        md.makeup = self._head_makeup(md)
        md.eyebrow = self._head_eyebrow(md)
        if md.eyebrow is not None:
            e = md.eyebrow
            md.notes.append(
                f"eyebrow -> Customize/{eyebrow_array_asset(e.index)} "
                f"(hair {e.hair_strength:.2f}, pencil {e.pencil_strength:.2f}, "
                f"atlas box {e.content_box[0]:.3f}-{e.content_box[1]:.3f} x "
                f"{e.content_box[2]:.3f}-{e.content_box[3]:.3f})"
            )
            placed = ", ".join(f"{k}={v:g}" for k, v in sorted(e.raw.items()))
            if placed:
                md.notes.append(f"eyebrow source placement (unapplied): {placed}")
        return md

    def _head_makeup(self, md: MaterialDef) -> List[MakeupLayer]:
        """Resolve the face's makeup decals (lips, eyeliner, eyeshadow, blush).

        Aion2 composites these as ``MK_Head_{EEE,HBS,LFL}_T2A_01`` texture-array
        elements over the base face, one per makeup slot, tinted by the slot's
        ``*_Color`` and blended at ``*Use_Color``.  The layer index is the
        slot's own ``*_NUMBER`` scalar; FModel writes element *n* as
        ``..._LAYER<n>`` (see :meth:`TextureResolver.resolve_stem`).

        Only slots the material actually enables are returned, so a head that
        leaves a decal off is never given one.

        The eyebrow is deliberately excluded: it lives in
        ``Cstm_EyeBrow_T2A`` (mask in **alpha**, hair colour in RGB) and is
        positioned *in the shader* by ``EyeBrow_Mirror``/``_ScaleU``/``_ScaleV``/
        ``_Rotate``/``Eyebrow_Distance``, so sampling it through UV0 would put
        one brow across half the face.
        """
        layers: List[MakeupLayer] = []
        for part, group, index_key, colour_key, strength_key, decode in MAKEUP_PARTS:
            strength = md.scalars.get(strength_key)
            if not strength:
                continue
            number = md.scalars.get(index_key)
            if not number:
                continue
            entry = md.colors.get(colour_key)
            if not isinstance(entry, dict):
                continue
            mask = self.gpaths.textures.resolve_stem(
                makeup_array_asset(group, int(number)))
            if mask is None:
                continue
            layers.append(MakeupLayer(
                part=part,
                mask=mask,
                colour=(float(entry.get("R", 1.0)),
                        float(entry.get("G", 1.0)),
                        float(entry.get("B", 1.0))),
                strength=min(max(float(strength), 0.0), 1.0),
                decode=decode,
            ))
        if layers:
            md.notes.append(
                "face makeup -> " + ", ".join(f"{l.part}@{l.strength:.2f}" for l in layers)
            )
        return layers

    # -- eyebrow ------------------------------------------------------------ #

    @staticmethod
    def _colour(entry, fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """A JSON ``Colors`` entry as linear RGB (FModel writes linear + a hex)."""
        if not isinstance(entry, dict):
            return fallback
        return (float(entry.get("R", fallback[0])),
                float(entry.get("G", fallback[1])),
                float(entry.get("B", fallback[2])))

    def _head_eyebrow(self, md: MaterialDef) -> Optional[Eyebrow]:
        """Resolve the face's eyebrow decal, or ``None`` when it has none.

        ``EyeBrow_NUMBER`` selects the brow *shape* - one element of the
        ``Cstm_EyeBrow_T2A`` texture array - and is set on every shipped face.
        ``Use_EyeBrow`` is the switch, and the faces that pin it to ``0`` are
        almost exactly the ones with no number at all.  A **missing**
        ``Use_EyeBrow`` is not "off": FModel only writes the parameters an
        instance overrides, and head ``002`` - the base head ``GF_BaseHead``
        uses - carries ``EyeBrow_NUMBER=12`` and both colours without it, so
        reading absence as off would leave the vanilla face brow-less.

        ``EyeBrow_Mirror`` is likewise present on only some instances, but the
        array holds a *single* brow, so mirroring is the only reading that
        produces a face; it is always applied.  The exact UE placement scalars
        are not reconstructable (see :mod:`aion2.faceproj`), so they are
        carried through as :attr:`Eyebrow.raw` for the report instead.
        """
        number = md.scalars.get("EyeBrow_NUMBER")
        use = md.scalars.get("Use_EyeBrow")
        if not number or (use is not None and use <= 0.0):
            return None
        index = int(number)
        stem = eyebrow_array_asset(index)
        mask = self.gpaths.textures.resolve_stem(stem)
        if mask is None:
            md.unresolved["eyebrow"] = f"Customize/{stem}"
            return None

        box = self._alpha_boxes.get(str(mask))
        if box is None:
            box = faceproj.tga_alpha_box(mask) or (0.0, 1.0, 0.0, 1.0)
            self._alpha_boxes[str(mask)] = box

        hair = self._colour(md.colors.get("EyeBrowHair_Color"), (0.09, 0.05, 0.04))
        pencil = self._colour(md.colors.get("EyeBrowPencil_Color"), hair)
        if not isinstance(md.colors.get("EyeBrowHair_Color"), dict):
            md.notes.append("eyebrow hair colour missing - defaulted to dark brown")
        raw = {k: float(v) for k, v in md.scalars.items()
               if k.startswith("EyeBrow") or k == "Eyebrow_Distance"}
        return Eyebrow(
            index=index,
            mask=mask,
            hair_colour=hair,
            pencil_colour=pencil,
            hair_strength=min(max(float(md.scalars.get("EyeBrowHair_intensity", 1.0)), 0.0), 1.0),
            pencil_strength=min(max(float(md.scalars.get("EyeBrowPencil_Intensity", 0.0)), 0.0), 1.0),
            content_box=box,
            raw=raw,
        )

    def iris_texture(self) -> Optional[Path]:
        """The shared iris map (``Common/Iris_0#_H``), or ``None``.

        Every ``MI_GF_Head_###_Eye`` is a parameter-only instance (FModel even
        reports it as ``IsNull``), so the iris pattern lives in one shared
        texture plus ``IrisColor_Mid``/``IrisColor_Edge`` from the JSON.
        """
        if self._iris is None:
            for name in ("Iris_02_H", "Iris_01_H", "Iris_03_H", "Iris_01_M"):
                hit = self.gpaths.resolve(
                    f"/Game/Character/Player/Common/{name}.{name}")
                if hit is not None:
                    self._iris = hit
                    break
            else:
                self._iris = False  # type: ignore[assignment]
        return self._iris or None

    def pupil_texture(self) -> Optional[Path]:
        """The shared pupil map (``Customize/Cstm_Pupil_##``), or ``None``.

        White background with green pupil art at the centre; the builder reads
        the pupil from the inverted green channel.
        """
        if self._pupil is None:
            for name in ("Cstm_Pupil_01", "Cstm_Pupil_02", "Cstm_Pupil_03"):
                hit = self.gpaths.resolve(
                    f"/Game/Character/Player/Customize/{name}.{name}")
                if hit is not None:
                    self._pupil = hit
                    break
            else:
                self._pupil = False  # type: ignore[assignment]
        return self._pupil or None

    def iris_mask_texture(self) -> Optional[Path]:
        """The default character-creator iris mask from ``IrisShape_01``."""
        return self.gpaths.resolve(
            "/Game/Character/Player/Customize/Cstm_Iris_M_01.Cstm_Iris_M_01"
        )

    def build_index(self) -> "MaterialLibrary":
        if self._built:
            return self
        for json_path in self.gpaths.materials():
            stem = json_path.stem.lower()
            if stem in self._index:
                self._collisions.append(stem)
                continue
            self._index[stem] = json_path
        self._built = True
        return self

    @staticmethod
    def _clean_name(name: str) -> str:
        """Blender de-duplicates as ``Name.001``; strip that for lookup."""
        base, dot, tail = name.rpartition(".")
        if dot and tail.isdigit() and len(tail) == 3:
            return base
        return name

    def lookup(self, material_name: str) -> Optional[Path]:
        self.build_index()
        candidates = [self._clean_name(material_name).lower()]
        # FModel sometimes suffixes instance names, e.g. "MI_X" vs "MI_X_LT".
        candidates.append(candidates[0].replace("_lt", ""))
        for cand in candidates:
            hit = self._index.get(cand)
            if hit:
                return hit
        # Fall back to a unique prefix match.
        matches = [p for stem, p in self._index.items() if stem.startswith(candidates[0])]
        if len(matches) == 1:
            return matches[0]
        return None

    def _as_eye_ao(self, md: MaterialDef, material_name: str) -> MaterialDef:
        """Steer an eye-AO/tear overlay onto the translucent overlay path.

        ``MI_Eye_AO_02`` and ``MI_EyeAO_Tear`` sit on every head mesh.  Without
        this they were classified as *eyes*, so each eyeball ended up with two
        extra opaque dark spheres layered over it.  ``MI_Eye_AO_02`` has no
        exported JSON at all, so its map is recovered from the instance name
        (``Customize/Eye_AO_M_02``); ``MI_EyeAO_Tear`` names it in its own JSON.
        """
        if not is_eye_ao_material(material_name):
            return md
        md.force_kind = "eye_ao"
        ue = None
        for value in md.ue_textures.values():
            if "Eye_AO_M_" in value:
                ue = value
                break
        ue = ue or eye_ao_texture_ue(material_name)
        if ue:
            hit = self.gpaths.resolve(ue)
            if hit is not None:
                md.resolved["base_color"] = hit
            else:
                md.unresolved["base_color"] = ue
        return md

    def _synthetic_eye_ao(self, material_name: str) -> Optional[MaterialDef]:
        """Describe an eye-AO overlay that has no material JSON at all."""
        if not is_eye_ao_material(material_name):
            return None
        md = MaterialDef(name=material_name, json_path=Path("<eye-ao-synthetic>"))
        return self._as_eye_ao(md, material_name)

    def definition(self, material_name: str) -> Optional[MaterialDef]:
        json_path = self.lookup(material_name)
        if json_path is None:
            return self._synthetic_eye_ao(material_name)
        md = self.gpaths.material(json_path)
        if md is None:
            return None
        if md.is_hair:
            md = self._hair_inheritance(md)
        elif is_layered_body_material(material_name):
            md = self._repair_layered_body(md)
        elif is_head_material(material_name):
            md = self._repair_head(md)
        elif is_eye_ao_material(material_name):
            md = self._as_eye_ao(md, material_name)
        return md

    def _hair_inheritance(self, child: MaterialDef) -> MaterialDef:
        """Merge exported hair material-instance parameters through their parents."""
        self.build_index()
        chain = [child]
        seen = {child.name.lower()}
        current = child

        while True:
            parent = current.properties.get("Parent")
            parent_path = parent.get("ObjectPath") if isinstance(parent, dict) else None
            if not isinstance(parent_path, str):
                break
            parent_name = parent_path.rsplit("/", 1)[-1].split(".", 1)[0]
            if not parent_name or parent_name.lower() in seen:
                break
            parent_json = self._index.get(parent_name.lower())
            if parent_json is None:
                break
            parent_md = self.gpaths.material(parent_json)
            if parent_md is None:
                break
            chain.append(parent_md)
            seen.add(parent_name.lower())
            current = parent_md

        if len(chain) == 1:
            return child

        merged_properties: Dict[str, object] = {}
        merged_overrides: Dict[str, object] = {}
        merged_textures: Dict[str, str] = {}
        merged_colors: Dict[str, object] = {}
        merged_scalars: Dict[str, float] = {}
        merged_switches: Dict[str, bool] = {}
        merged_resolved: Dict[str, Path] = {}
        merged_unresolved: Dict[str, str] = {}
        inherited_names = [item.name for item in reversed(chain)]

        for item in reversed(chain):
            merged_properties.update(item.properties)
            overrides = item.properties.get("BasePropertyOverrides")
            if isinstance(overrides, dict):
                merged_overrides.update(overrides)
            merged_textures.update(item.ue_textures)
            merged_colors.update(item.colors)
            merged_scalars.update(item.scalars)
            merged_switches.update(item.switches)
            merged_resolved.update(item.resolved)
            merged_unresolved.update(item.unresolved)

        if merged_overrides:
            merged_properties["BasePropertyOverrides"] = merged_overrides

        child.properties = merged_properties
        child.ue_textures = merged_textures
        child.colors = merged_colors
        child.scalars = merged_scalars
        child.switches = merged_switches
        child.resolved = merged_resolved
        child.unresolved = merged_unresolved
        child.two_sided = bool(merged_overrides.get("TwoSided", child.two_sided))
        child.opacity_mask_clip = float(
            merged_overrides.get("OpacityMaskClipValue", child.opacity_mask_clip)
        )
        child.blend_mode = merged_overrides.get("BlendMode", child.blend_mode)
        child.shading_model = merged_overrides.get(
            "ShadingModel", child.shading_model
        )
        child.notes.append(
            "hair parameters inherited: " + " -> ".join(inherited_names)
        )
        return child

    @property
    def collisions(self) -> List[str]:
        return list(self._collisions)

    @property
    def size(self) -> int:
        self.build_index()
        return len(self._index)


def assign_materials(
    meshes: Sequence["bpy.types.Object"],
    library: MaterialLibrary,
    builder: Aion2MaterialBuilder,
    report: Optional[AssemblyReport] = None,
) -> List[MaterialReport]:
    """Rebuild every material slot on *meshes* from its FModel definition."""
    reports: List[MaterialReport] = []
    done: Dict[str, MaterialReport] = {}

    # Every eye instance shares one iris map (and a pupil map); hand both to
    # the builder once.
    if getattr(builder, "iris_texture", None) is None:
        builder.iris_texture = library.iris_texture()
    if getattr(builder, "pupil_texture", None) is None:
        builder.pupil_texture = library.pupil_texture()
    if getattr(builder, "iris_mask_texture", None) is None:
        builder.iris_mask_texture = library.iris_mask_texture()

    for mesh in meshes:
        for slot in mesh.material_slots:
            mat = slot.material
            if mat is None:
                continue
            key = mat.name
            if key in done:
                reports.append(done[key])
                continue
            md = library.definition(mat.name)
            try:
                result = builder.build(mat, md)
            except Exception as exc:  # pragma: no cover - keep the build alive
                result = MaterialReport(material=mat.name)
                result.warnings.append(f"build failed: {exc}")
            if md is None:
                # Preserve the imported glTF shader rather than blanking it.
                result.warnings.append("no FModel JSON; kept imported graph")
            else:
                # Repairs the library applied (shared head maps, body-skin
                # redirect) are surfaced here so a fix is never invisible.
                result.notes.extend(md.notes)
            done[key] = result
            reports.append(result)
    if report is not None:
        report.materials.extend(reports)
    return reports


def _brow_uv_layer(me: "bpy.types.Mesh"):
    """Return (creating if needed) the baked eyebrow UV layer, or ``None``.

    ``uv_layers.new()`` cannot be relied on: on Blender 5.2 it returns ``None``
    and adds nothing to a *glTF-imported* mesh (those carry a ``custom_normal``
    attribute), while working fine on a primitive.  The generic attribute API
    always works and the result shows up in ``uv_layers`` immediately.  There is
    deliberately **no** fallback to "the last layer" - writing the brow into an
    existing UV set would corrupt whatever else samples it.
    """
    layer = me.uv_layers.get(BROW_UV_LAYER)
    if layer is not None:
        return layer
    try:
        me.attributes.new(BROW_UV_LAYER, "FLOAT2", "CORNER")
    except (AttributeError, RuntimeError, TypeError):
        try:
            me.uv_layers.new(name=BROW_UV_LAYER)
        except (AttributeError, RuntimeError, TypeError):
            return None
    return me.uv_layers.get(BROW_UV_LAYER)


def _write_uvs(me: "bpy.types.Mesh", layer, flat: List[float]) -> bool:
    """Fill a UV layer from a flat ``[u0, v0, u1, v1, ...]`` list.

    Blender 4.x moved UVs onto a generic ``FLOAT2`` attribute (``layer.uv``)
    and 5.x dropped ``layer.data`` entirely, so the modern path is tried first.
    """
    for attempt in (lambda: layer.uv.foreach_set("vector", flat),
                    lambda: layer.data.foreach_set("uv", flat)):
        try:
            attempt()
            return True
        except (AttributeError, RuntimeError, TypeError):
            continue
    try:  # pragma: no cover - very old Blender
        for index in range(len(me.loops)):
            layer.data[index].uv = (flat[2 * index], flat[2 * index + 1])
        return True
    except (AttributeError, RuntimeError, TypeError):
        return False


def _write_vertex_floats(me: "bpy.types.Mesh", name: str, values: Sequence[float]) -> bool:
    """Store one float per vertex in a ``POINT`` attribute, creating it.

    The eyebrow gate has to be an attribute rather than part of the baked UV:
    see :data:`aion2.paths.BROW_MASK_ATTRIBUTE`.
    """
    attribute = me.attributes.get(name)
    if attribute is None:
        try:
            attribute = me.attributes.new(name, "FLOAT", "POINT")
        except (AttributeError, RuntimeError, TypeError):
            return False
    try:
        attribute.data.foreach_set("value", list(values))
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def bake_brow_uvs(
    meshes: Sequence["bpy.types.Object"],
    library: MaterialLibrary,
    report: Optional[AssemblyReport] = None,
) -> List[str]:
    """Bake the eyebrow projection into an extra UV layer on the head meshes.

    Nothing else in Aion2 needs baked coordinates: the makeup decals are in head
    UV space.  The eyebrow is not (see :mod:`aion2.faceproj`), so its projection
    is written once per vertex into :data:`aion2.paths.BROW_UV_LAYER`, where it
    deforms with the face instead of sliding across it.

    Returns the names of the objects it baked.
    """
    baked: List[str] = []
    for obj in meshes:
        me = obj.data
        definitions: Dict[str, Optional[MaterialDef]] = {}
        for slot in obj.material_slots:
            if slot.material is not None:
                definitions[slot.material.name] = library.definition(slot.material.name)
        eyebrow = next((d.eyebrow for d in definitions.values() if d and d.eyebrow), None)
        if eyebrow is None:
            continue

        # Group the mesh's vertices by what their polygons' materials are, so
        # the projection can be anchored on the eyeball and the eye-AO shell.
        eye_idx: set = set()
        lid_idx: set = set()
        for poly in me.polygons:
            slot = obj.material_slots[poly.material_index] if poly.material_index < len(obj.material_slots) else None
            name = slot.material.name if slot and slot.material else ""
            md = definitions.get(name)
            if md is not None and md.is_eye:
                eye_idx.update(poly.vertices)
            elif is_eye_ao_material(name):
                lid_idx.update(poly.vertices)

        coords = [v.co for v in me.vertices]
        eye_pts = [coords[i] for i in sorted(eye_idx)]
        lid_pts = [coords[i] for i in sorted(lid_idx)]
        anchors = faceproj.anchors_from_mesh(eye_pts, lid_pts)
        if anchors is None:
            message = f"{obj.name}: no eyeball geometry - eyebrow not placed"
            if report is not None:
                report.warnings.append(message)
            continue

        box = faceproj.brow_box(anchors)
        normals = [v.normal for v in me.vertices]
        projected = faceproj.project_brow(coords, normals, box, eyebrow.content_box)
        coords_uv = [uv for uv, _w in projected]
        weights = [w for _uv, w in projected]
        inside = sum(1 for w in weights if w >= 1.0)

        layer = _brow_uv_layer(me)
        if layer is None:
            message = f"{obj.name}: could not add a '{BROW_UV_LAYER}' UV layer"
            if report is not None:
                report.warnings.append(message)
            continue
        flat: List[float] = []
        for loop in me.loops:
            u, v = faceproj.to_blender_uv(coords_uv[loop.vertex_index])
            flat.append(u)
            flat.append(v)
        if not _write_uvs(me, layer, flat):
            message = f"{obj.name}: could not write '{BROW_UV_LAYER}'"
            if report is not None:
                report.warnings.append(message)
            continue
        if not _write_vertex_floats(me, BROW_MASK_ATTRIBUTE, weights):
            message = f"{obj.name}: could not write '{BROW_MASK_ATTRIBUTE}'"
            if report is not None:
                report.warnings.append(message)
            continue
        # The decal's own layer must not become the render UV: everything else
        # still samples UV0.
        try:
            layer.active_render = False
            me.uv_layers[0].active_render = True
            me.uv_layers.active_index = 0
        except (AttributeError, RuntimeError, TypeError):
            pass

        baked.append(obj.name)
        if report is not None:
            report.notes.append(
                f"{obj.name}: eyebrow baked ({eyebrow.mask.name}) on {inside} verts "
                f"- {anchors.describe()}, brow x {box.x_in:.4f}..{box.x_out:.4f} "
                f"z {box.z_lo:.4f}..{box.z_hi:.4f}"
            )
    return baked


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

class Aion2Assembler:
    """Imports pieces, merges the rig, wires materials and organises collections."""

    def __init__(
        self,
        gpaths: Optional[GamePaths] = None,
        builder: Optional[Aion2MaterialBuilder] = None,
        *,
        gender: str = "GF",
        skin_fallback: bool = True,
        root_collection: str = "Aion2",
        open_eyes: float = 0.6,
        log=print,
    ) -> None:
        self.gpaths = gpaths or GamePaths()
        self.builder = builder or Aion2MaterialBuilder()
        self.gender = gender.upper()
        #: The exported bind pose has the lids nearly shut.  When the master rig
        #: is merged, the upper-lid bones are rotated open by this fraction
        #: (0 disables) so the eyes read as open without any manual posing.
        self.open_eyes = open_eyes
        self.library = MaterialLibrary(self.gpaths, self.gender)
        self.root_collection = root_collection
        self.log = log
        if skin_fallback:
            self._apply_skin_fallbacks()

    def _apply_skin_fallbacks(self) -> None:
        """Point the builder at the base-body skin maps.

        Aion2 ships no albedo for the head (its ``PM_Diffuse`` slot names a
        mask), so without this the face renders as flat grey. Reusing the base
        body's skin maps keeps the face consistent with the body. Only applied
        when the caller has not supplied their own fallbacks.
        """
        skin = self.library._skin_maps()
        if getattr(self.builder, "fallback_base_color", None) is None:
            self.builder.fallback_base_color = skin.get("base_color")
        if getattr(self.builder, "fallback_packed", None) is None:
            self.builder.fallback_packed = skin.get("packed")

    # -- public ------------------------------------------------------------ #

    def assemble(self, spec: CharacterSpec, *, merge_rig: bool = True,
                 replace: bool = True) -> AssemblyReport:
        """Import *spec* into the scene.

        ``replace`` (the default) deletes the previous Aion2 build first, which
        is what keeps the operator idempotent - rebuilding must not stack a
        second copy of the character on top of the first.
        """
        report = AssemblyReport()
        if replace:
            report.replaced_objects = purge_previous_build(self.root_collection)
            if report.replaced_objects:
                self.log(f"  replaced previous build "
                         f"({len(report.replaced_objects)} objects removed)")
        root = _get_collection(self.root_collection)

        armatures: List["bpy.types.Object"] = []
        meshes: List["bpy.types.Object"] = []

        for label, path in spec.all_paths():
            if not path.is_file():
                report.warnings.append(f"missing file: {path}")
                self.log(f"  !! missing {label}: {path}")
                continue
            self.log(f"  + {label}: {path.name}")
            info = import_glb(path, report=report)
            coll = _get_collection(f"{self.root_collection}_{label}", root)
            info.collection = coll.name
            info.label = label

            for name in info.mesh_objects:
                obj = bpy.data.objects.get(name)
                if obj is None:
                    continue
                _link_only(obj, coll)
                meshes.append(obj)
            if info.armature:
                arm = bpy.data.objects.get(info.armature)
                if arm is not None:
                    _link_only(arm, root)
                    # Keep donor rigs visible until merging; mode_set cannot
                    # enter Edit Mode on a hidden object.
                    armatures.append(arm)
            report.pieces.append(info)

        master = None
        if merge_rig:
            master = merge_armatures(armatures, report)
            if master is not None:
                _link_only(master, root)
                master.name = "A2_Rig"
                master.data.name = master.name
                report.master_armature = master.name
                # Hide the finished rig so the viewport shows only geometry.
                try:
                    master.hide_viewport = True
                except AttributeError:
                    pass
                if self.open_eyes > 0.0:
                    opened = open_eyes_pose(master, self.open_eyes, report)
                    if opened:
                        report.notes.append(
                            f"eyes posed open ({', '.join(opened)}), "
                            f"upper-lid rotation +{self.open_eyes:.2f} rad"
                        )

        self.library.build_index()
        self.log(f"  material definitions indexed: {self.library.size}")
        baked = bake_brow_uvs(meshes, self.library, report)
        if baked:
            self.log(f"  eyebrow UV baked: {', '.join(baked)}")
        assign_materials(meshes, self.library, self.builder, report)

        if self.library.collisions:
            report.warnings.append(
                f"{len(self.library.collisions)} duplicate material json stems ignored"
            )
        for name in self.library.repaired:
            report.notes.append(
                f"{name}: layered body shell - redirected to the character's skin maps"
            )

        # Armour meshes bundle the skin beneath them: Body carries the torso
        # shell, Pants the legs.  Those two are complementary, not duplicates,
        # so only flag the genuinely wrong combination - a *standalone* body
        # imported alongside armour that already supplies one.
        providers = [p for p in report.pieces if p.supplies_base_body]
        standalone = [p for p in providers if p.label == "Basebody"]
        from_armour = [p for p in providers if p.label == "Armor"]
        if standalone and from_armour:
            report.redundant_shells = [p.source.stem for p in providers]
            report.warnings.append(
                f"a standalone body ({', '.join(p.source.stem for p in standalone)}) "
                f"overlaps armour that already contains skin "
                f"({', '.join(p.source.stem for p in from_armour)}); "
                "drop the base body or the armour"
            )
        return report

    def import_any(self, paths: Iterable[Path], *, merge_rig: bool = True,
                   replace: bool = False) -> AssemblyReport:
        """Import an arbitrary set of GLBs (used by the viewer/browser).

        ``replace`` defaults to False here so extra assets can be added to an
        existing character without wiping it.
        """
        spec = CharacterSpec(extra=list(paths))
        return self.assemble(spec, merge_rig=merge_rig, replace=replace)
