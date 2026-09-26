"""Aion2 (AION2_TW) Unreal Engine asset tooling for Blender.

Sub-modules
-----------
paths      UE object-path (<->) exported-file resolution + asset discovery
materials  Principled BSDF / shading graph reconstruction from FModel material JSON
assembly   GLB import, master-rig merging and coherent scene collection assembly

``paths`` is import-safe outside Blender (no ``bpy`` dependency), so discovery
and auditing can run under a plain CPython interpreter.  ``materials`` and
``assembly`` require ``bpy``.
"""

from .glb import GlbSummary, summarize
from .paths import (
    DEFAULT_EXPORT_ROOT,
    GamePaths,
    MaterialDef,
    TextureResolver,
    discover_materials,
    discover_meshes,
    export_root,
    is_mask_reference,
    parse_material,
    ue_to_content_rel,
)
from .spec import CharacterSpec

__all__ = [
    # bpy-free
    "DEFAULT_EXPORT_ROOT",
    "GamePaths",
    "GlbSummary",
    "MaterialDef",
    "TextureResolver",
    "CharacterSpec",
    "discover_materials",
    "discover_meshes",
    "export_root",
    "is_mask_reference",
    "parse_material",
    "summarize",
    "ue_to_content_rel",
    # Blender-only symbols are imported lazily so that the helpers above keep
    # working under plain CPython.
    "Aion2MaterialBuilder",
    "MaterialReport",
    "Aion2Assembler",
    "AssemblyReport",
    "Catalog",
    "default_spec",
]


def __getattr__(name: str):
    """Lazily expose the bpy-dependent classes to keep this package import-safe."""
    if name in ("Aion2MaterialBuilder", "MaterialReport"):
        from .materials import Aion2MaterialBuilder, MaterialReport
        return {"Aion2MaterialBuilder": Aion2MaterialBuilder,
                "MaterialReport": MaterialReport}[name]
    if name in ("Aion2Assembler", "AssemblyReport"):
        from .assembly import Aion2Assembler, AssemblyReport
        return {"Aion2Assembler": Aion2Assembler,
                "AssemblyReport": AssemblyReport}[name]
    if name in ("Catalog", "default_spec"):
        from .catalog import Catalog, default_spec
        return {"Catalog": Catalog, "default_spec": default_spec}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
