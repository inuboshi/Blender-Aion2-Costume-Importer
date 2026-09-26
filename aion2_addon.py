"""Aion2 Character Builder - a Blender sidebar panel for the FModel export tree.

Adds a **Aion2** tab to the 3D Viewport sidebar (press ``N``) that lets you pick
a female/male base body, head, hairstyle and armour set, then assemble the whole
character onto one merged rig with fully rebuilt Aion2 shading in a single click.

The heavy lifting lives in the sibling ``aion2`` package; this file is only the
UI shell.  Set the *Project root* to the folder containing that package (default
`the project folder`).

Install::

    Install via Blender's Preferences > Add-ons > Install from Disk
"""

from __future__ import annotations

import os
import re
import sys
import traceback

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

bl_info = {
    "name": "Aion2 Character Builder",
    "author": "Inuboshi",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Aion2",
    "description": "Import AION2 characters from an FModel export and rebuild their shading",
    "category": "Import-Export",
}

DEFAULT_PROJECT_ROOT = r"your project root"
DEFAULT_EXPORT_ROOT = r"your FModel export root"

#: Armour sub-parts the picker can include, in the order they read in the list.
ARMOR_PARTS = ("Body", "Pants", "Boots", "Glove", "Shoulder", "Cape", "Helmet")

#: Eye-preset identifier meaning "whatever the head material itself ships".
EYE_HEAD_DEFAULT = "HEAD"

#: Eye materials are the eyeball itself (``..._Eye``), including the split
#: NPC variants (``..._EyeL`` / ``..._EyeR``).  The lash/tear overlays
#: (``MI_Eye_AO_*``, ``MI_EyeAO_Tear``) must **not** match - they are masks.
EYE_MATERIAL_RE = re.compile(r"_eye[lr]?$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Bridge to the aion2 package
# --------------------------------------------------------------------------- #

def _project_root() -> str:
    """Where the ``aion2`` package lives (addon preference, then sensible guesses)."""
    prefs = bpy.context.preferences.addons.get(__name__)
    if prefs is not None and getattr(prefs.preferences, "project_root", ""):
        return prefs.preferences.project_root
    here = os.path.dirname(os.path.abspath(__file__))
    return here or DEFAULT_PROJECT_ROOT


def _ensure_importable() -> None:
    root = _project_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _load_modules():
    """Import (or re-import) the aion2 building blocks.

    Re-imported on every use so edits to the package are picked up without
    restarting Blender.
    """
    _ensure_importable()
    import importlib

    import aion2.paths
    import aion2.glb
    import aion2.catalog
    import aion2.spec
    import aion2.materials
    import aion2.assembly

    for mod in (aion2.paths, aion2.glb, aion2.catalog, aion2.spec,
                aion2.materials, aion2.assembly):
        importlib.reload(mod)

    from aion2.assembly import Aion2Assembler, MaterialLibrary
    from aion2.catalog import Catalog
    from aion2.materials import Aion2MaterialBuilder
    from aion2.paths import GamePaths
    from aion2.spec import CharacterSpec
    return {
        "Aion2Assembler": Aion2Assembler,
        "MaterialLibrary": MaterialLibrary,
        "Catalog": Catalog,
        "Aion2MaterialBuilder": Aion2MaterialBuilder,
        "GamePaths": GamePaths,
        "CharacterSpec": CharacterSpec,
    }


# --------------------------------------------------------------------------- #
# Cached dynamic enum data
# --------------------------------------------------------------------------- #

_CACHE: dict = {"signature": None, "items": {}}
_CREATURE_CACHE: dict = {"signature": None, "items": []}
_LIBRARY_CACHE: dict = {"signature": None, "library": None}


def _paths() -> "object":
    mods = _load_modules()
    return mods["GamePaths"](bpy.context.scene.aion2.export_root or None)


def _refresh_cache(force: bool = False) -> dict:
    """Build (and cache) the enum item lists for every picker."""
    scene = bpy.context.scene
    signature = (scene.aion2.export_root, scene.aion2.gender)
    if not force and _CACHE["signature"] == signature:
        return _CACHE["items"]

    mods = _load_modules()
    GamePaths, Catalog = mods["GamePaths"], mods["Catalog"]
    gpaths = GamePaths(scene.aion2.export_root or None)
    cat = Catalog(gpaths, scene.aion2.gender)

    def items_from(paths, include_none=None):
        out = list(include_none or [])
        for path in paths:
            info = cat.info(path)
            out.append((path.stem, info.label(), str(path)))
        return out

    def eye_items(shapes):
        """Eye presets, with the material's own shipped look offered first."""
        out = [(EYE_HEAD_DEFAULT, "Head default",
                "Keep the iris/pupil the selected head material ships")]
        for shape in shapes:
            if shape.usable:
                out.append((shape.variant, shape.label, shape.detail()))
        return out

    irises = cat.eye_iris_shapes()
    pupils = cat.eye_pupil_shapes()
    items = {
        "basebody": items_from(cat.basebody(usable_only=True)) or [("NONE", "-", "")],  # noqa: E501
        "head": items_from(cat.heads(usable_only=True)) or [("NONE", "-", "")],
        "hair": items_from(cat.hairs(usable_only=True)) or [("NONE", "-", "")],
        # Real sets first: Blender uses the first item as the implicit default,
        # so listing "none" first silently built unarmoured characters.
        "armor": [(s.variant, s.label, str(s.variant)) for s in cat.armor_sets()]
                 + [("NONE", "- no armour -", "")],
        "iris_shape": eye_items(irises),
        "pupil_shape": eye_items(pupils),
        "iris_shapes": {s.variant: s for s in irises},
        "pupil_shapes": {s.variant: s for s in pupils},
        "counts": cat.summary(),
        "gpaths": gpaths,
    }
    items["counts"]["irises"] = len(irises)
    items["counts"]["pupils"] = len(pupils)
    _CACHE["signature"] = signature
    _CACHE["items"] = items
    return items


def _enum_items(key: str):
    def callback(self, context):
        try:
            return _refresh_cache()[key]
        except Exception:  # pragma: no cover - never break the UI
            return [("NONE", "unavailable", "")]
    return callback


def _creature_items(self, context):
    try:
        settings = context.scene.aion2
        signature = (settings.export_root, settings.creature_family)
        if _CREATURE_CACHE["signature"] != signature:
            mods = _load_modules()
            gpaths = mods["GamePaths"](settings.export_root or None)
            cat = mods["Catalog"](gpaths, "GF")
            paths = cat.creature_assets(settings.creature_family)
            _CREATURE_CACHE["items"] = [
                (str(path), f"{path.stem} [{path.parent.name}]", str(path))
                for path in paths
            ] or [("NONE", "No exported GLBs found", "")]
            _CREATURE_CACHE["signature"] = signature
        return _CREATURE_CACHE["items"]
    except Exception:
        return [("NONE", "unavailable", "")]


def _on_settings_changed(self, context) -> None:
    """Property ``update`` hook: drop the cached picker lists.

    Declared as a real function rather than a lambda because Blender stores the
    callback in RNA and may invoke it from a context where a closure's globals
    are not yet populated.  It must never raise - a failure here would abort the
    property assignment itself.
    """
    try:
        _refresh_cache(force=True)
    except Exception:  # pragma: no cover - UI must stay alive
        pass


def _on_eye_choice_changed(self, context) -> None:
    """Property ``update`` hook: re-shade the eyes already in the scene.

    That is what makes the eye pickers *previews* rather than settings you only
    see after a full rebuild.  It never raises, and it does nothing when there
    is no character yet - pressing Build applies the same choice anyway.
    """
    try:
        settings = context.scene.aion2 if context.scene else None
        if settings is None or not settings.eye_preview:
            return
        _preview_eyes(context)
    except Exception:  # pragma: no cover - UI must stay alive
        traceback.print_exc()


def _material_library(gpaths, gender: str):
    """A cached :class:`MaterialLibrary` - indexing 4k JSONs is not cheap.

    The eye presets preview on every change, so rebuilding that index each time
    made the picker feel sticky.  Keyed on the export root and gender, which are
    the only inputs.
    """
    signature = (str(gpaths.root), gender)
    cached = _LIBRARY_CACHE
    if cached["library"] is None or signature != cached["signature"]:
        library = None
    else:
        library = cached["library"]
    if library is None:
        mods = _load_modules()
        library = mods["MaterialLibrary"](gpaths, gender)
        _LIBRARY_CACHE["signature"] = signature
        _LIBRARY_CACHE["library"] = library
    return library


def _preview_eyes(context) -> int:
    """Rebuild every eyeball material in the scene from the current choices."""
    settings = context.scene.aion2
    mods = _load_modules()
    gpaths = mods["GamePaths"](settings.export_root or None)
    library = _material_library(gpaths, settings.gender)
    builder = mods["Aion2MaterialBuilder"](**settings.builder_kwargs())
    touched = 0
    for material in list(bpy.data.materials):
        if not EYE_MATERIAL_RE.search(material.name):
            continue
        try:
            builder.build(material, library.definition(material.name))
        except Exception:  # pragma: no cover - a bad preset must not kill the UI
            traceback.print_exc()
            continue
        touched += 1
    for area in context.screen.areas if context.screen else []:
        if area.type == "VIEW_3D":
            area.tag_redraw()
    return touched


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

class Aion2Settings(PropertyGroup):
    export_root: StringProperty(
        name="Export root",
        description="FModel output folder that contains the Content/ tree",
        subtype="DIR_PATH",
        default=DEFAULT_EXPORT_ROOT,
        update=_on_settings_changed,
    )
    gender: EnumProperty(
        name="Gender",
        description="Which player character tree to list",
        items=[
            ("GF", "Female (GF)", "Character/Player/GF"),
            ("GM", "Male (GM)", "Character/Player/GM"),
        ],
        default="GF",
        update=_on_settings_changed,
    )
    creature_family: EnumProperty(
        name="Family",
        description="Generic exported character family",
        items=[
            ("NPC", "NPC", "Character/NPC"),
            ("Monster", "Monster", "Character/Monster"),
        ],
        default="NPC",
        update=_on_settings_changed,
    )
    creature_asset: EnumProperty(
        name="Asset",
        description="NPC or monster mesh to import",
        items=_creature_items,
    )

    basebody: EnumProperty(name="Base body", items=_enum_items("basebody"))
    head: EnumProperty(name="Head", items=_enum_items("head"))
    hair: EnumProperty(name="Hair", items=_enum_items("hair"))
    armor: EnumProperty(name="Armour set", items=_enum_items("armor"))

    # Off by default: the armour meshes already bundle the skin underneath them,
    # so importing a naked body as well just stacks two shells on the same surface.
    include_basebody: BoolProperty(
        name="Base body (naked)",
        description=("Import a standalone naked body. Usually unnecessary - armour meshes "
                     "already contain the skin beneath them - and enabling it with "
                     "armour produces overlapping shells"),
        default=False,
    )
    include_head: BoolProperty(name="Head", default=True)
    include_hair: BoolProperty(name="Hair", default=True)
    include_armor: BoolProperty(name="Armour", default=True)

    part_body: BoolProperty(name="Body", default=True)
    part_pants: BoolProperty(name="Pants", default=True)
    part_boots: BoolProperty(name="Boots", default=True)
    part_glove: BoolProperty(name="Gloves", default=True)
    part_shoulder: BoolProperty(name="Shoulder", default=True)
    part_cape: BoolProperty(name="Cape", default=True)
    part_helmet: BoolProperty(name="Helmet", default=True)

    merge_rig: BoolProperty(
        name="Merge into one rig",
        description="Union every piece's partial skeleton into a single armature",
        default=True,
    )
    replace_existing: BoolProperty(
        name="Replace previous build",
        description=("Delete the previous Aion2 build before importing. Leave on - "
                     "otherwise every press adds another copy of the character"
                     " (two heads, two hairs, overlapping armour)"),
        default=True,
    )
    skin_fallback: BoolProperty(
        name="Borrow body skin for the face",
        description=("Aion2 exports no face albedo (its diffuse slot is a mask), so the "
                     "head reuses the base body's skin maps instead of rendering grey"),
        default=True,
    )
    sss_weight: FloatProperty(
        name="Skin subsurface",
        description="Principled subsurface weight for skin materials",
        default=0.1, min=0.0, max=1.0, precision=3,
    )
    ao_strength: FloatProperty(
        name="AO strength",
        description="How strongly the packed map's red channel darkens base colour",
        default=1.0, min=0.0, max=1.0,
    )
    custom_color_tint: BoolProperty(
        name="Armour custom colours",
        description=("Aion2 armour diffuse maps are greyscale; the real colour comes from "
                     "the CSTM mask plus the CustomMask_Color palette. Enable to "
                     "reconstruct that tint instead of rendering neutral grey armour"),
        default=True,
    )
    face_makeup: BoolProperty(
        name="Face make-up",
        description=("Composite the head materials' lips / eyeliner / eyeshadow / blush "
                     "decals (MK_Head_* texture arrays) over the face albedo. Off by "
                     "default: the decals need the exported mask arrays and mostly "
                     "darken the face, so the bare head reads better until you want "
                     "a specific look"),
        default=False,
    )
    face_eyebrow: BoolProperty(
        name="Eyebrows",
        description=("Project the head material's Cstm_EyeBrow_T2A brow onto the "
                     "forehead (baked into an extra A2_BrowUV layer, since the atlas "
                     "is not in head UV space). Turn off to compare against a "
                     "brow-less face"),
        default=True,
    )
    iris_shape: EnumProperty(
        name="Iris shape",
        description="Character-creator iris preset (Common/CustomMI/IrisShape)",
        items=_enum_items("iris_shape"),
        update=_on_eye_choice_changed,
    )
    pupil_shape: EnumProperty(
        name="Pupil shape",
        description="Character-creator pupil preset (Common/CustomMI/PupilShape)",
        items=_enum_items("pupil_shape"),
        update=_on_eye_choice_changed,
    )
    eye_preview: BoolProperty(
        name="Re-shade eyes live",
        description=("Rebuild the eyes in the scene as soon as an eye preset changes, "
                     "so the choice can be previewed before building"),
        default=True,
    )

    # ---------------------------------------------------------------------
    # Iris colour overrides – per‑eye colour picker (RGBA, alpha unused)
    # ---------------------------------------------------------------------
    iris_mid_color: FloatVectorProperty(
        name="Iris Mid Colour",
        description="Override the default iris middle colour (RGB).",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.11, 0.15, 0.13, 1.0),
        update=_on_settings_changed,
    )
    iris_edge_color: FloatVectorProperty(
        name="Iris Edge Colour",
        description="Override the default iris edge colour (RGB).",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.08, 0.11, 0.15, 1.0),
        update=_on_settings_changed,
    )
    open_eyes: FloatProperty(
        name="Open eyes (rad)",
        description=("The exported bind pose has the lids nearly shut; rotate the "
                     "upper-lid bones open by this many radians after the rig is "
                     "merged (0 keeps the exported pose)"),
        default=0.6,
        min=0.0,
        max=1.2,
        subtype="ANGLE",
    )
    pack_textures: BoolProperty(
        name="Pack textures into .blend",
        description="Embed every image so the file is self-contained",
        default=False,
    )

    def eye_options(self) -> dict:
        """Builder kwargs for the picked eye presets (empty = head default)."""
        try:
            items = _refresh_cache()
        except Exception:  # pragma: no cover - UI must stay alive
            return {}
        options = {}
        iris = items.get("iris_shapes", {}).get(self.iris_shape)
        if iris is not None and iris.usable:
            options["iris_texture"] = iris.height
            options["iris_mask_texture"] = iris.masks
        pupil = items.get("pupil_shapes", {}).get(self.pupil_shape)
        if pupil is not None and pupil.usable:
            options["pupil_texture"] = pupil.pupil
            options["pupil_scale"] = pupil.scale
        return options

    def builder_kwargs(self) -> dict:
        """Every shading choice, ready to splat into ``Aion2MaterialBuilder``."""
        kwargs = dict(
            sss_weight=self.sss_weight,
            ao_strength=self.ao_strength,
            custom_color_tint=self.custom_color_tint,
            face_makeup=self.face_makeup,
            face_eyebrow=self.face_eyebrow,
        )
        kwargs.update(self.eye_options())
        # Add iris colour overrides – they are stored as RGBA vectors; only RGB is needed.
        kwargs["iris_mid_color"] = tuple(self.iris_mid_color[:3])
        kwargs["iris_edge_color"] = tuple(self.iris_edge_color[:3])
        return kwargs

    def selected_parts(self):
        flags = (
            ("Body", self.part_body), ("Pants", self.part_pants),
            ("Boots", self.part_boots), ("Glove", self.part_glove),
            ("Shoulder", self.part_shoulder), ("Cape", self.part_cape),
            ("Helmet", self.part_helmet),
        )
        return [name for name, on in flags if on]

    def selected_armor(self, cat):
        if not (self.include_armor and self.armor and self.armor != "NONE"):
            return []
        chosen = cat.set_by_label(self.armor)
        return chosen.ordered_parts(self.selected_parts()) if chosen else []

    def body_status(self, gpaths) -> str:
        """Warn about the two ways a body selection can go wrong."""
        try:
            from aion2.catalog import Catalog, supplies_base_body
            cat = Catalog(gpaths, self.gender)
            armor = self.selected_armor(cat)
            armour_has_body = any(supplies_base_body(p) for p in armor)
        except Exception:
            return ""
        if self.include_basebody and armour_has_body:
            return "Base body + armour both supply skin: expect overlapping shells"
        if not self.include_basebody and not armour_has_body and not armor:
            return "No body selected: enable Base body to get a complete character"
        return ""

    def build_spec(self, gpaths):
        from aion2.catalog import Catalog
        from aion2.spec import CharacterSpec

        cat = Catalog(gpaths, self.gender)
        armor_paths = self.selected_armor(cat)

        def pick(options, stem):
            for p in options:
                if p.stem == stem:
                    return p
            return None

        return CharacterSpec(
            basebody=pick(cat.basebody(), self.basebody) if self.include_basebody else None,
            head=pick(cat.heads(), self.head) if self.include_head else None,
            hair=pick(cat.hairs(), self.hair) if self.include_hair else None,
            armor=armor_paths,
            gender=self.gender,
        )


class Aion2Preferences(AddonPreferences):
    bl_idname = __name__

    project_root: StringProperty(
        name="Project root",
        description="Folder containing the 'aion2' package (and scripts/)",
        subtype="DIR_PATH",
        default=DEFAULT_PROJECT_ROOT,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "project_root")
        layout.label(text="The aion2 package must live directly inside this folder.",
                     icon="INFO")


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #

class AION2_OT_refresh(Operator):
    """Re-scan the export tree and rebuild the picker lists"""

    bl_idname = "aion2.refresh_catalog"
    bl_label = "Refresh Catalog"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            items = _refresh_cache(force=True)
        except Exception as exc:
            self.report({"ERROR"}, f"Aion2: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}
        counts = items["counts"]
        self.report({"INFO"},
                    f"Aion2: {counts['basebody']} bodies, {counts['heads']} heads, "
                    f"{counts['hairs']} hairs, {counts['armor_sets']} armour sets")
        return {"FINISHED"}


class AION2_OT_build_character(Operator):
    """Import the selected pieces and rebuild their Aion2 shading"""

    bl_idname = "aion2.build_character"
    bl_label = "Run Pipeline / Build Character"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.aion2
        try:
            mods = _load_modules()
            gpaths = mods["GamePaths"](settings.export_root or None)
            spec = settings.build_spec(gpaths)
            if spec.is_empty():
                self.report({"ERROR"}, "Nothing selected to build")
                return {"CANCELLED"}

            builder = mods["Aion2MaterialBuilder"](**settings.builder_kwargs())
            assembler = mods["Aion2Assembler"](
                gpaths, builder, gender=settings.gender,
                skin_fallback=settings.skin_fallback, open_eyes=settings.open_eyes,
                log=lambda m: None,
            )
            report = assembler.assemble(spec, merge_rig=settings.merge_rig,
                                       replace=settings.replace_existing)

            if settings.pack_textures:
                try:
                    bpy.ops.file.pack_all()
                except RuntimeError:
                    pass

            # Make the new character visible in whatever view is open.
            for area in context.screen.areas if context.screen else []:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            region.tag_redraw()
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Aion2 build failed: {exc}")
            return {"CANCELLED"}

        missing = len(report.missing_maps)
        replaced = len(report.replaced_objects)
        message = (f"Built {len(report.pieces)} pieces, {len(report.materials)} materials"
                   + (f", replaced {replaced} old objects" if replaced else "")
                   + (f", {missing} missing maps" if missing else ""))
        if report.redundant_shells:
            message += f"; WARNING duplicated body shells: {report.redundant_shells}"
        self.report({"WARNING"} if report.redundant_shells else {"INFO"}, message)
        return {"FINISHED"}


class AION2_OT_preview_eyes(Operator):
    """Re-shade the eyes in the scene with the picked iris and pupil presets"""

    bl_idname = "aion2.preview_eyes"
    bl_label = "Preview Eyes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.aion2
        try:
            touched = _preview_eyes(context)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Aion2 eye preview failed: {exc}")
            return {"CANCELLED"}
        if not touched:
            self.report({"INFO"},
                        "Aion2: no eyeballs in this scene yet - build a character first")
            return {"CANCELLED"}
        self.report({"INFO"},
                    f"Aion2: re-shaded {touched} eye material(s) "
                    f"({settings.iris_shape} / {settings.pupil_shape})")
        return {"FINISHED"}


class AION2_OT_drop_basebody(Operator):
    """Stop importing the naked body, since the armour already supplies it"""

    bl_idname = "aion2.drop_basebody"
    bl_label = "Use armour's body only"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.aion2.include_basebody = False
        return {"FINISHED"}


class AION2_OT_import_single(Operator):
    """Import one exported .glb and rebuild just its materials"""

    bl_idname = "aion2.import_single"
    bl_label = "Import Single Asset"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from pathlib import Path

        settings = context.scene.aion2
        try:
            mods = _load_modules()
            gpaths = mods["GamePaths"](settings.export_root or None)
            builder = mods["Aion2MaterialBuilder"](**settings.builder_kwargs())
            assembler = mods["Aion2Assembler"](
                gpaths, builder, gender=settings.gender, log=lambda m: None,
            )
            report = assembler.import_any([Path(self.filepath)], merge_rig=settings.merge_rig)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Aion2 import failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"},
                    f"Imported {Path(self.filepath).name}: "
                    f"{len(report.pieces)} piece(s), {len(report.materials)} materials")
        return {"FINISHED"}


class AION2_OT_audit_textures(Operator):
    """Report how many material texture references resolve on disk"""

    bl_idname = "aion2.audit_textures"
    bl_label = "Audit Texture Coverage"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            mods = _load_modules()
            gpaths = mods["GamePaths"](context.scene.aion2.export_root or None)
            definitions = gpaths.materials()
            resolved = missing = 0
            for json_path in definitions:
                md = gpaths.material(json_path)
                if md is None:
                    continue
                resolved += len(md.resolved)
                missing += len(md.unresolved)
        except Exception as exc:
            self.report({"ERROR"}, f"Aion2 audit failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"},
                    f"Aion2: {resolved} texture refs resolved, {missing} unresolved "
                    f"across {len(definitions)} materials")
        return {"FINISHED"}


class AION2_OT_build_creature(Operator):
    """Import one generic NPC or monster and rebuild its materials."""

    bl_idname = "aion2.build_creature"
    bl_label = "Import NPC / Monster"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.aion2
        if not settings.creature_asset or settings.creature_asset == "NONE":
            self.report({"ERROR"}, "No NPC or monster asset selected")
            return {"CANCELLED"}
        try:
            mods = _load_modules()
            from pathlib import Path

            gpaths = mods["GamePaths"](settings.export_root or None)
            builder = mods["Aion2MaterialBuilder"](
                sss_weight=settings.sss_weight,
                ao_strength=settings.ao_strength,
                custom_color_tint=settings.custom_color_tint,
            )
            assembler = mods["Aion2Assembler"](
                gpaths, builder, gender="GF", skin_fallback=False,
                open_eyes=0.0, log=lambda _message: None,
            )
            report = assembler.import_any(
                [Path(settings.creature_asset)],
                merge_rig=settings.merge_rig,
                replace=settings.replace_existing,
            )
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Creature import failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"},
                    f"Imported {len(report.pieces)} creature piece(s), "
                    f"rebuilt {len(report.materials)} materials")
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #

class AION2_PT_main(Panel):
    bl_label = "Aion2 Character Builder"
    bl_idname = "AION2_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Aion2"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.aion2
        items = _refresh_cache()
        counts = items["counts"]

        col = layout.column(align=True)
        col.prop(settings, "export_root")
        row = col.row(align=True)
        row.prop(settings, "gender", expand=True)

        layout.separator()
        layout.label(text=f"{counts['basebody']} bodies / {counts['heads']} heads / "
                          f"{counts['hairs']} hairs / {counts['armor_sets']} sets",
                     icon="INFO")

        # --- pieces --------------------------------------------------------- #
        box = layout.box()
        box.label(text="Character parts", icon="OUTLINER_OB_GROUP_INSTANCE")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(settings, "include_basebody", text="")
        row.prop(settings, "basebody", text="")
        row = col.row(align=True)
        row.prop(settings, "include_head", text="")
        row.prop(settings, "head", text="")
        row = col.row(align=True)
        row.prop(settings, "include_hair", text="")
        row.prop(settings, "hair", text="")

        # --- eyes ----------------------------------------------------------- #
        eye_box = layout.box()
        eye_box.label(text=f"Eyes ({counts.get('irises', 0)} iris / "
                           f"{counts.get('pupils', 0)} pupil presets)",
                      icon="HIDE_OFF")
        col = eye_box.column(align=True)
        col.prop(settings, "iris_shape")
        col.prop(settings, "pupil_shape")
        row = eye_box.row(align=True)
        row.prop(settings, "eye_preview")
        row.operator(AION2_OT_preview_eyes.bl_idname, text="Preview",
                     icon="FILE_REFRESH")
        # -----------------------------------------------------------------
        # Iris colour pickers – only show when a preview is enabled (matches UI style)
        # -----------------------------------------------------------------
        col = eye_box.column(align=True)
        col.prop(settings, "iris_mid_color")
        col.prop(settings, "iris_edge_color")

        # --- armour --------------------------------------------------------- #
        box = layout.box()
        box.label(text="Armour", icon="MOD_CLOTH")
        row = box.row(align=True)
        row.prop(settings, "include_armor", text="")
        row.prop(settings, "armor", text="")
        grid = box.grid_flow(columns=3, even_columns=True, align=True)
        for part in ARMOR_PARTS:
            grid.prop(settings, f"part_{part.lower()}")

        warning = settings.body_status(items["gpaths"])
        if warning:
            note = layout.box()
            note.alert = True
            row = note.row()
            row.label(text=warning, icon="ERROR")
            if "overlapping" in warning:
                note.operator(AION2_OT_drop_basebody.bl_idname)

        # --- shading -------------------------------------------------------- #
        box = layout.box()
        box.label(text="Shading", icon="MATERIAL")
        box.prop(settings, "sss_weight", slider=True)
        box.prop(settings, "ao_strength", slider=True)
        box.prop(settings, "custom_color_tint")
        box.prop(settings, "face_makeup")
        box.prop(settings, "face_eyebrow")
        box.prop(settings, "open_eyes")
        box.prop(settings, "merge_rig")
        box.prop(settings, "skin_fallback")
        box.prop(settings, "replace_existing")
        box.prop(settings, "pack_textures")

        # --- actions -------------------------------------------------------- #
        layout.separator()
        layout.scale_y = 1.5
        layout.operator(AION2_OT_build_character.bl_idname, icon="ARMATURE_DATA")
        layout.scale_y = 1.0
        row = layout.row(align=True)
        row.operator(AION2_OT_refresh.bl_idname, icon="FILE_REFRESH")
        row.operator(AION2_OT_audit_textures.bl_idname, icon="IMAGE_DATA")
        layout.operator(AION2_OT_import_single.bl_idname, icon="IMPORT")


class AION2_PT_creatures(Panel):
    bl_label = "Aion2 NPCs and Monsters"
    bl_idname = "AION2_PT_creatures"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Creatures"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.aion2
        layout.prop(settings, "export_root")
        layout.prop(settings, "creature_family", expand=True)
        layout.prop(settings, "creature_asset", text="Asset")
        layout.separator()
        layout.prop(settings, "merge_rig")
        layout.prop(settings, "replace_existing")
        layout.prop(settings, "custom_color_tint")
        layout.operator(AION2_OT_build_creature.bl_idname, icon="ARMATURE_DATA")
        layout.operator(AION2_OT_import_single.bl_idname, icon="IMPORT")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_CLASSES = (
    Aion2Preferences,
    Aion2Settings,
    AION2_OT_refresh,
    AION2_OT_build_character,
    AION2_OT_preview_eyes,
    AION2_OT_drop_basebody,
    AION2_OT_import_single,
    AION2_OT_audit_textures,
    AION2_OT_build_creature,
    AION2_PT_main,
    AION2_PT_creatures,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.aion2 = PointerProperty(type=Aion2Settings)


def unregister() -> None:
    del bpy.types.Scene.aion2
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
