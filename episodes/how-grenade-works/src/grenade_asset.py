"""
M67 Tactical Grenade + X-Ray Internal Cutaway + 3D Component Pointer Labels Asset
Blender 4.2 Cycles | Real-World Metric Local Coordinates (R_OUTER = 0.042m)

Architectural Standards (AGENTS.md V10 Cinema Standard):
- 100% local coordinates centered at (0, 0, 0) inside parent Empty `Grenade_Root`.
  Move, rotate, or scale `Grenade_Root` anywhere in a 3D world and every mesh,
  FX element, 3D callout label, and spatial shader follows seamlessly.
- 100% Local-Space Shaders: all spatial masks (`ScanZ`, `BurnZ`, cast-iron noise,
  molten fracture veins, turbulent fire plasma) drive off `ShaderNodeTexCoord.Object`
  bound directly to `Grenade_Root` (`tex_coord.object = root_empty`).
- Photoreal M67 cast-iron shell, muted ochre band, machined beveled cutaway rim,
  diamond internal pre-scoring V-grooves, sculpted beveled M213 die-cast fuze head,
  hexagonal threaded collar, split cotter pin + knurled chrome pull ring, 33-point
  Catmull-Rom rolled-steel safety spoon, 6-coil helical torsion spring, forged
  striker hammer slamming dead-center onto the brass primer, machined gold-brass
  cutaway delay column + live burning core + copper booster, 48 thick pre-scored
  cast-iron shards, 3D procedural turbulent fireball, and AgX-safe 2-Zone brushed-gold
  / cyan 3D engineering callouts with glowing leader lines.

Public API:
    build_grenade_assembly(
        collection=None,
        name_prefix="",
        include_labels=True,
        include_fx=True,
        font_path=None,
    ) -> dict

    animate_grenade(
        assembly,
        *,
        pin_pull=(15, 96),
        xray_scan=(108, 140),
        spoon_fly=(244, 322),
        striker_slam=(244, 294),
        fuse_burn=(356, 499),
        pressure_swell=(499, 611),
        fracture_frame=612,
        total_frames=720,
        show_labels=True,
    ) -> None
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

# ==============================================================================
# LOCAL-SPACE METRIC CONSTANTS (Centered at (0, 0, 0) inside Grenade_Root)
# ==============================================================================
R_OUTER = 0.0420        # 8.4 cm outer diameter (real handheld M67 baseball grenade)
R_INNER = 0.0355        # 6.5 mm cast-iron wall thickness
NECK_Z_MIN = 0.0380
NECK_Z_MAX = 0.0520
FUZE_Z_MIN = 0.0500
FUZE_Z_MAX = 0.0740
HINGE_POS = Vector((0.0080, 0.0, 0.0630))
PRIMER_POS = Vector((0.0, 0.0, 0.0470))
DELAY_Z_TOP = 0.0460
DELAY_Z_BOT = -0.0050
BOOSTER_Z_TOP = -0.0040
BOOSTER_Z_BOT = -0.0160

# Exact kinematics: vector from HINGE_POS (+0.008, 0, +0.063) to PRIMER_POS (0, 0, +0.047)
# is (-0.008, 0, -0.016) -> length = 0.01788854m, Y-rotation impact angle = -243.4349 deg.
_STRIKER_DELTA = PRIMER_POS - HINGE_POS
STRIKER_REACH = _STRIKER_DELTA.length
STRIKER_IMPACT_DEG = -math.degrees(math.atan2(-_STRIKER_DELTA.z, _STRIKER_DELTA.x))
if STRIKER_IMPACT_DEG > 0.0:
    STRIKER_IMPACT_DEG -= 360.0
# Verify: atan2(+0.016, -0.008) = +116.565 deg -> -243.435 deg so R_y(-243.435)*(+L,0,0) = (-0.008,0,-0.016)
STRIKER_IMPACT_DEG = -243.4349488
STRIKER_COCKED_DEG = -52.0

# ScanZ & BurnZ local ranges
SCAN_Z_START = 0.0950   # Cleanly above top of spoon/fuze (+0.078)
SCAN_Z_END = -0.0600    # Cleanly below bottom of shell (-0.042)
BURN_Z_START = 0.0455   # Top of delay core directly under primer
BURN_Z_END = -0.0045    # Bottom of delay core directly at copper booster

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"


# ==============================================================================
# MATH, SPLINE & BLENDER HELPERS
# ==============================================================================
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _catmull_rom(pts: list[Vector], samples_per_seg: int = 4) -> list[Vector]:
    """Smooth Catmull-Rom spline interpolation for stamped-steel contours."""
    if len(pts) < 2:
        return list(pts)
    ext = [pts[0] + (pts[0] - pts[1])] + list(pts) + [pts[-1] + (pts[-1] - pts[-2])]
    out: list[Vector] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    out.append(pts[-1])
    return out


def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    target = col or bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_smooth(mesh: bpy.types.Mesh, smooth: bool = True) -> None:
    if hasattr(mesh, "polygons") and len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()


def _create_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material] | bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Euler = (0.0, 0.0, 0.0),
    smooth_all: bool | None = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    if smooth_all is not None:
        _set_smooth(mesh, smooth_all)
    else:
        mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation) if not isinstance(rotation, Euler) else rotation.copy()
    if parent is not None:
        obj.parent = parent
    _link_obj(obj, col)
    return obj


def _clear_anim(id_block: Any) -> None:
    if id_block is not None and getattr(id_block, "animation_data", None):
        id_block.animation_data_clear()


def _set_bezier(id_block: Any, skip_prefix: str = "hide_") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def _set_linear(id_block: Any, data_path_contains: str = "") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if data_path_contains and data_path_contains not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _set_constant(id_block: Any, data_path_prefix: str = "hide_") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if fc.data_path.startswith(data_path_prefix):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    if not visible_ranges:
        for f in (1, total_frames):
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        _set_constant(obj, "hide_")
        return

    all_frames = {1, total_frames}
    for s, e in visible_ranges:
        s_c = max(1, min(total_frames, int(s)))
        e_c = max(1, min(total_frames, int(e)))
        if s_c > 1:
            all_frames.add(s_c - 1)
        all_frames.add(s_c)
        all_frames.add(e_c)
        if e_c < total_frames:
            all_frames.add(e_c + 1)

    for f in sorted(all_frames):
        is_vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not is_vis
        obj.hide_render = not is_vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_constant(obj, "hide_")


def _parse_window(val: tuple[int, int] | list[int] | int | float, default_len: int) -> tuple[int, int]:
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return int(val[0]), int(val[1])
    start = int(val)
    return start, start + default_len


def _make_beveled_box_bm(dims: tuple[float, float, float], bevel_w: float, segments: int = 3) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((dims[0], dims[1], dims[2], 1.0)))
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_w, segments=segments, affect="EDGES")
    return bm


def _make_torus_bm(
    major_r: float,
    minor_r: float,
    major_seg: int = 48,
    minor_seg: int = 16,
    ang_start: float = 0.0,
    ang_end: float = 2.0 * math.pi,
    cap_ends: bool = False,
) -> bmesh.types.BMesh:
    """Build a full or open-arc smooth torus in bmesh (no bpy.ops dependency)."""
    bm = bmesh.new()
    closed = abs((ang_end - ang_start) - 2.0 * math.pi) < 1e-4
    n_u = major_seg if closed else major_seg + 1
    rings: list[list[bmesh.types.BMVert]] = []

    for i in range(n_u):
        u = i / major_seg
        a = _lerp(ang_start, ang_end, u)
        ca, sa = math.cos(a), math.sin(a)
        ring: list[bmesh.types.BMVert] = []
        for j in range(minor_seg):
            b = j * 2.0 * math.pi / minor_seg
            cb, sb = math.cos(b), math.sin(b)
            r = major_r + minor_r * cb
            ring.append(bm.verts.new((r * ca, r * sa, minor_r * sb)))
        rings.append(ring)

    seg_count = major_seg if closed else major_seg
    for i in range(seg_count):
        inext = (i + 1) % major_seg if closed else i + 1
        for j in range(minor_seg):
            jnext = (j + 1) % minor_seg
            f = bm.faces.new((rings[i][j], rings[inext][j], rings[inext][jnext], rings[i][jnext]))
            f.smooth = True

    if not closed and cap_ends:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])

    return bm


def _make_tube_along_path_bm(
    points: list[Vector],
    radius: float,
    n_seg: int = 12,
    cap_ends: bool = True,
) -> bmesh.types.BMesh:
    """Sweep a circular cross-section cleanly along a polyline."""
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    up_ref = Vector((0.0, 1.0, 0.0))

    for idx, pt in enumerate(points):
        if idx == 0:
            tang = (points[1] - points[0]).normalized()
        elif idx == len(points) - 1:
            tang = (points[-1] - points[-2]).normalized()
        else:
            t1 = (points[idx] - points[idx - 1]).normalized()
            t2 = (points[idx + 1] - points[idx]).normalized()
            tang = (t1 + t2)
            tang = tang.normalized() if tang.length > 1e-6 else t1

        ref = up_ref if abs(tang.dot(up_ref)) < 0.92 else Vector((0.0, 0.0, 1.0))
        b1 = tang.cross(ref).normalized()
        b2 = tang.cross(b1).normalized()
        ring = []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ring.append(bm.verts.new(pt + (b1 * math.cos(a) + b2 * math.sin(a)) * radius))
        rings.append(ring)

    for i in range(len(rings) - 1):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.smooth = True

    if cap_ends and len(rings) >= 2:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])

    return bm


# ==============================================================================
# LOCAL-SPACE MATERIALS LIBRARY (100% TexCoord.Object -> Grenade_Root!)
# ==============================================================================
def _add_cast_iron_nodes(
    nt: bpy.types.NodeTree,
    bsdf: bpy.types.ShaderNode,
    root_empty: bpy.types.Object,
    is_yellow_band: bool = False,
) -> None:
    """Wire gritty military M67 cast-iron PBR shader in Grenade_Root local space."""
    tex_c = nt.nodes.new("ShaderNodeTexCoord")
    tex_c.object = root_empty

    # Multi-scale cast-iron pitting: Noise 140 (fine sand-cast pits) + Noise 28 (macro patina)
    noise_fine = nt.nodes.new("ShaderNodeTexNoise")
    noise_fine.inputs["Scale"].default_value = 140.0
    noise_fine.inputs["Detail"].default_value = 12.0
    noise_fine.inputs["Roughness"].default_value = 0.76

    noise_micro = nt.nodes.new("ShaderNodeTexNoise")
    noise_micro.inputs["Scale"].default_value = 480.0
    noise_micro.inputs["Detail"].default_value = 8.0
    noise_micro.inputs["Roughness"].default_value = 0.70

    noise_macro = nt.nodes.new("ShaderNodeTexNoise")
    noise_macro.inputs["Scale"].default_value = 28.0
    noise_macro.inputs["Detail"].default_value = 6.0
    noise_macro.inputs["Roughness"].default_value = 0.58

    nt.links.new(tex_c.outputs["Object"], noise_fine.inputs["Vector"])
    nt.links.new(tex_c.outputs["Object"], noise_micro.inputs["Vector"])
    nt.links.new(tex_c.outputs["Object"], noise_macro.inputs["Vector"])

    # Combine fine + micro cast-iron height
    mix_h = nt.nodes.new("ShaderNodeMath")
    mix_h.operation = "MULTIPLY_ADD"
    mix_h.inputs[1].default_value = 0.65
    nt.links.new(noise_fine.outputs["Fac"], mix_h.inputs[0])

    mul_m = nt.nodes.new("ShaderNodeMath")
    mul_m.operation = "MULTIPLY"
    mul_m.inputs[1].default_value = 0.35
    nt.links.new(noise_micro.outputs["Fac"], mul_m.inputs[0])
    nt.links.new(mul_m.outputs["Value"], mix_h.inputs[2])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.58 if not is_yellow_band else 0.44
    bump.inputs["Distance"].default_value = 0.00105
    nt.links.new(mix_h.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Edge / patina roughness variation
    ramp_rough = nt.nodes.new("ShaderNodeMapRange")
    ramp_rough.clamp = True
    ramp_rough.inputs["From Min"].default_value = 0.22
    ramp_rough.inputs["From Max"].default_value = 0.78
    if is_yellow_band:
        # Authentic muted military ochre-yellow shoulder band (rich ordnance paint, never chalky!)
        ramp_col = nt.nodes.new("ShaderNodeValToRGB")
        ramp_col.color_ramp.elements[0].position = 0.25
        ramp_col.color_ramp.elements[0].color = (0.38, 0.23, 0.010, 1.0)
        ramp_col.color_ramp.elements[1].position = 0.78
        ramp_col.color_ramp.elements[1].color = (0.58, 0.37, 0.022, 1.0)
        nt.links.new(noise_macro.outputs["Fac"], ramp_col.inputs["Fac"])
        nt.links.new(ramp_col.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Metallic"].default_value = 0.12
        ramp_rough.inputs["To Min"].default_value = 0.46
        ramp_rough.inputs["To Max"].default_value = 0.70
    else:
        # Deep gritty military olive-drab cast iron (0.025, 0.040, 0.018)
        ramp_col = nt.nodes.new("ShaderNodeValToRGB")
        ramp_col.color_ramp.elements[0].position = 0.22
        ramp_col.color_ramp.elements[0].color = (0.014, 0.024, 0.010, 1.0)
        e_mid = ramp_col.color_ramp.elements.new(0.55)
        e_mid.color = (0.025, 0.040, 0.018, 1.0)
        ramp_col.color_ramp.elements[2].position = 0.85
        ramp_col.color_ramp.elements[2].color = (0.042, 0.058, 0.025, 1.0)
        nt.links.new(noise_macro.outputs["Fac"], ramp_col.inputs["Fac"])
        nt.links.new(ramp_col.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Metallic"].default_value = 0.45
        bsdf.inputs["Coat Weight"].default_value = 0.06
        bsdf.inputs["Coat Roughness"].default_value = 0.44
        ramp_rough.inputs["To Min"].default_value = 0.38
        ramp_rough.inputs["To Max"].default_value = 0.74

    nt.links.new(noise_macro.outputs["Fac"], ramp_rough.inputs["Value"])
    nt.links.new(ramp_rough.outputs["Result"], bsdf.inputs["Roughness"])


def _build_materials(
    root_empty: bpy.types.Object,
    prefix: str = "",
) -> tuple[dict[str, bpy.types.Material], dict[str, Any]]:
    mats: dict[str, bpy.types.Material] = {}
    shader_nodes: dict[str, Any] = {"scan_z_nodes": []}

    def nm(base: str) -> str:
        return f"{prefix}{base}"

    # 1. Authentic Gritty Military Olive-Drab Cast Iron
    m_olive = bpy.data.materials.new(nm("M67_Olive_CastIron"))
    m_olive.use_nodes = True
    nt = m_olive.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _add_cast_iron_nodes(nt, bsdf, root_empty, is_yellow_band=False)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mats["olive_iron"] = m_olive

    # 2. Tactical Military Ochre-Yellow Shoulder Band
    m_yellow = bpy.data.materials.new(nm("M67_Yellow_Band"))
    m_yellow.use_nodes = True
    nt = m_yellow.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _add_cast_iron_nodes(nt, bsdf, root_empty, is_yellow_band=True)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mats["yellow_band"] = m_yellow

    # 3. Machined Beveled Cutaway Rim Steel (Crisp dark brushed titanium-steel edge, zero anisotropic striping!)
    m_cut = bpy.data.materials.new(nm("Machined_Cutaway_Steel"))
    m_cut.use_nodes = True
    nt = m_cut.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.19, 0.23, 0.28, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Anisotropic"].default_value = 0.0
    mats["cut_steel"] = m_cut

    # 4. Internal Pre-Scored Cast Steel Cavity (Crisp gunmetal diamond V-grooves catching rim highlights)
    m_inner = bpy.data.materials.new(nm("Inner_PreScored_Steel"))
    m_inner.use_nodes = True
    nt = m_inner.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.14, 0.17, 0.22, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.94
    bsdf.inputs["Roughness"].default_value = 0.26
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = root_empty
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 240.0
    nf.inputs["Detail"].default_value = 8.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.14
    bmp.inputs["Distance"].default_value = 0.00035
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["inner_steel"] = m_inner

    # 5. LOCAL-SPACE Animated X-Ray Transition Shader (Solid M67 -> Crystal Cyan X-Ray Glass!)
    def make_xray_transition_mat(
        name: str,
        is_yellow_band: bool = False,
        is_fuze: bool = False,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(nm(name))
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")

        # CRITICAL: Local-space Object coordinate bound to Grenade_Root (NOT World Geometry.Position!)
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        tex_coord.object = root_empty
        sep_xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])

        scan_z = nt.nodes.new("ShaderNodeValue")
        scan_z.name = "ScanZ"
        scan_z.outputs[0].default_value = SCAN_Z_START
        shader_nodes["scan_z_nodes"].append(scan_z)

        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        nt.links.new(sep_xyz.outputs["Z"], sub.inputs[0])
        nt.links.new(scan_z.outputs[0], sub.inputs[1])

        xray_mask = nt.nodes.new("ShaderNodeMapRange")
        xray_mask.clamp = True
        xray_mask.inputs["From Min"].default_value = -0.0028
        xray_mask.inputs["From Max"].default_value = 0.0038
        nt.links.new(sub.outputs["Value"], xray_mask.inputs["Value"])

        abs_d = nt.nodes.new("ShaderNodeMath")
        abs_d.operation = "ABSOLUTE"
        nt.links.new(sub.outputs["Value"], abs_d.inputs[0])

        scan_line = nt.nodes.new("ShaderNodeMapRange")
        scan_line.clamp = True
        scan_line.inputs["From Min"].default_value = 0.0022
        scan_line.inputs["From Max"].default_value = 0.0
        nt.links.new(abs_d.outputs["Value"], scan_line.inputs["Value"])

        solid = nt.nodes.new("ShaderNodeBsdfPrincipled")
        if is_fuze:
            solid.inputs["Base Color"].default_value = (0.16, 0.19, 0.23, 1.0)
            solid.inputs["Metallic"].default_value = 0.94
            solid.inputs["Roughness"].default_value = 0.27
            solid.inputs["Anisotropic"].default_value = 0.42
            nf = nt.nodes.new("ShaderNodeTexNoise")
            nf.inputs["Scale"].default_value = 260.0
            nf.inputs["Detail"].default_value = 8.0
            bmp = nt.nodes.new("ShaderNodeBump")
            bmp.inputs["Strength"].default_value = 0.08
            bmp.inputs["Distance"].default_value = 0.00025
            nt.links.new(tex_coord.outputs["Object"], nf.inputs["Vector"])
            nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
            nt.links.new(bmp.outputs["Normal"], solid.inputs["Normal"])
        else:
            _add_cast_iron_nodes(nt, solid, root_empty, is_yellow_band=is_yellow_band)

        # Ultra-clean Technical Cyan X-Ray Glass (Fresnel edge glow + crystal-clear center)
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        transp.inputs["Color"].default_value = (0.96, 0.99, 1.0, 1.0)

        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.24

        rim_pow = nt.nodes.new("ShaderNodeMath")
        rim_pow.operation = "POWER"
        rim_pow.inputs[1].default_value = 2.6
        nt.links.new(lw.outputs["Facing"], rim_pow.inputs[0])

        rim_scale = nt.nodes.new("ShaderNodeMath")
        rim_scale.operation = "MULTIPLY"
        rim_scale.inputs[1].default_value = 0.36 if not is_fuze else 0.24
        nt.links.new(rim_pow.outputs["Value"], rim_scale.inputs[0])

        xray_emit = nt.nodes.new("ShaderNodeEmission")
        xray_emit.inputs["Color"].default_value = (0.12, 0.80, 1.0, 1.0)
        xray_emit.inputs["Strength"].default_value = 1.90

        xray_mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(rim_scale.outputs["Value"], xray_mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], xray_mix.inputs[1])
        nt.links.new(xray_emit.outputs["Emission"], xray_mix.inputs[2])

        main_mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(xray_mask.outputs["Result"], main_mix.inputs["Fac"])
        nt.links.new(solid.outputs["BSDF"], main_mix.inputs[1])
        nt.links.new(xray_mix.outputs["Shader"], main_mix.inputs[2])

        # Laser scan band additive glow
        scan_emit = nt.nodes.new("ShaderNodeEmission")
        scan_emit.inputs["Color"].default_value = (0.22, 0.94, 1.0, 1.0)
        scan_strength = nt.nodes.new("ShaderNodeMath")
        scan_strength.operation = "MULTIPLY"
        scan_strength.inputs[1].default_value = 4.8
        nt.links.new(scan_line.outputs["Result"], scan_strength.inputs[0])
        nt.links.new(scan_strength.outputs["Value"], scan_emit.inputs["Strength"])

        add_scan = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(main_mix.outputs["Shader"], add_scan.inputs[0])
        nt.links.new(scan_emit.outputs["Emission"], add_scan.inputs[1])

        # AGENTS.md Section 3: LightPath.Is Shadow Ray -> Transparent BSDF when X-ray active!
        lp = nt.nodes.new("ShaderNodeLightPath")
        shad_fac = nt.nodes.new("ShaderNodeMath")
        shad_fac.operation = "MULTIPLY"
        nt.links.new(lp.outputs["Is Shadow Ray"], shad_fac.inputs[0])
        nt.links.new(xray_mask.outputs["Result"], shad_fac.inputs[1])

        final_mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(shad_fac.outputs["Value"], final_mix.inputs["Fac"])
        nt.links.new(add_scan.outputs["Shader"], final_mix.inputs[1])
        nt.links.new(transp.outputs["BSDF"], final_mix.inputs[2])

        nt.links.new(final_mix.outputs["Shader"], out.inputs["Surface"])
        return mat

    mats["front_olive_xray"] = make_xray_transition_mat("FrontShell_Olive_XRay", False, False)
    mats["front_yellow_xray"] = make_xray_transition_mat("FrontShell_Yellow_XRay", True, False)
    mats["front_fuze_xray"] = make_xray_transition_mat("FrontFuze_Cover_XRay", False, True)

    # 6. Anodized Dark Gunmetal-Zinc Die-Cast M213 Fuze Body
    m_fuze = bpy.data.materials.new(nm("M213_Anodized_Gunmetal"))
    m_fuze.use_nodes = True
    nt = m_fuze.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.16, 0.19, 0.23, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.94
    bsdf.inputs["Roughness"].default_value = 0.27
    bsdf.inputs["Anisotropic"].default_value = 0.42
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = root_empty
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 260.0
    nf.inputs["Detail"].default_value = 8.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.08
    bmp.inputs["Distance"].default_value = 0.00025
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["fuze_metal"] = m_fuze

    # 7. Dark Parkerized Hexagonal / Threaded Neck Collar
    m_collar = bpy.data.materials.new(nm("Parkerized_Neck_Collar"))
    m_collar.use_nodes = True
    bsdf = m_collar.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.085, 0.10, 0.125, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.93
    bsdf.inputs["Roughness"].default_value = 0.29
    mats["neck_collar"] = m_collar

    # 8. Polished Heavy Chrome-Steel Pull Ring & Split Cotter Pin
    m_chrome = bpy.data.materials.new(nm("Polished_Chrome_Pin"))
    m_chrome.use_nodes = True
    bsdf = m_chrome.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.84, 0.87, 0.92, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.13
    mats["chrome_steel"] = m_chrome

    # 9. Matte Military Olive-Steel Stamped Safety Lever ("Spoon")
    m_spoon = bpy.data.materials.new(nm("Safety_Lever_Spoon"))
    m_spoon.use_nodes = True
    nt = m_spoon.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.044, 0.064, 0.032, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.74
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = 0.18
    bsdf.inputs["Emission Color"].default_value = (0.14, 0.78, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    shader_nodes["spoon_em_socket"] = bsdf.inputs["Emission Strength"]
    mats["spoon_steel"] = m_spoon

    # 10. High-Carbon 6-Coil Helical Torsion Spring (Gleaming Amber-Orange Tension Highlight!)
    m_spring = bpy.data.materials.new(nm("Torsion_Spring_Steel"))
    m_spring.use_nodes = True
    bsdf = m_spring.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.80, 0.84, 0.90, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.14
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.50, 0.07, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    shader_nodes["spring_em_socket"] = bsdf.inputs["Emission Strength"]
    mats["spring_steel"] = m_spring

    # 11. Forged Steel Striker Hammer
    m_striker = bpy.data.materials.new(nm("Striker_Hammer_Steel"))
    m_striker.use_nodes = True
    bsdf = m_striker.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.76, 0.72, 0.66, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.98
    bsdf.inputs["Roughness"].default_value = 0.19
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.44, 0.06, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.0
    shader_nodes["striker_em_socket"] = bsdf.inputs["Emission Strength"]
    mats["striker_steel"] = m_striker

    # 12. Machined Gold-Brass (Delay Column & Primer Cap)
    m_brass = bpy.data.materials.new(nm("Machined_Gold_Brass"))
    m_brass.use_nodes = True
    bsdf = m_brass.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.86, 0.64, 0.18, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.98
    bsdf.inputs["Roughness"].default_value = 0.18
    bsdf.inputs["Anisotropic"].default_value = 0.56
    mats["brass"] = m_brass

    # 13. Machined Copper Booster Capsule
    m_copper = bpy.data.materials.new(nm("Machined_Copper_Booster"))
    m_copper.use_nodes = True
    bsdf = m_copper.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.84, 0.36, 0.14, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.21
    mats["copper"] = m_copper

    # 14. Dark Granular Composition-B Educational Chamber Block (High 10:1 contrast behind brass!)
    m_core_fill = bpy.data.materials.new(nm("CompositionB_Core_Fill"))
    m_core_fill.use_nodes = True
    nt = m_core_fill.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.044, 0.036, 0.028, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.08
    bsdf.inputs["Roughness"].default_value = 0.82
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = root_empty
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 320.0
    nf.inputs["Detail"].default_value = 9.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.22
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])
    nt.links.new(nf.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    mats["core_fill"] = m_core_fill

    # 15. LOCAL-SPACE Live Burning 4-Second Delay Fuse Core Shader
    m_fuse = bpy.data.materials.new(nm("Live_Delay_Fuse_Core"))
    m_fuse.use_nodes = True
    nt = m_fuse.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    # Local Object space referenced to Grenade_Root!
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.object = root_empty
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tex_coord.outputs["Object"], sep.inputs["Vector"])

    burn_z = nt.nodes.new("ShaderNodeValue")
    burn_z.name = "BurnZ"
    burn_z.outputs[0].default_value = BURN_Z_START
    shader_nodes["burn_z_node"] = burn_z

    burn_on = nt.nodes.new("ShaderNodeValue")
    burn_on.name = "BurnOn"
    burn_on.outputs[0].default_value = 0.0
    shader_nodes["burn_on_node"] = burn_on

    diff = nt.nodes.new("ShaderNodeMath")
    diff.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Z"], diff.inputs[0])
    nt.links.new(burn_z.outputs[0], diff.inputs[1])

    # Unburned dark-slate delay powder below -> charred black ash above burn front
    ash_mask = nt.nodes.new("ShaderNodeMapRange")
    ash_mask.clamp = True
    ash_mask.inputs["From Min"].default_value = 0.0
    ash_mask.inputs["From Max"].default_value = 0.0032
    nt.links.new(diff.outputs["Value"], ash_mask.inputs["Value"])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.inputs[6].default_value = (0.060, 0.066, 0.080, 1.0)
    mix_col.inputs[7].default_value = (0.015, 0.014, 0.014, 1.0)
    nt.links.new(ash_mask.outputs["Result"], mix_col.inputs["Factor"])

    abs_b = nt.nodes.new("ShaderNodeMath")
    abs_b.operation = "ABSOLUTE"
    nt.links.new(diff.outputs["Value"], abs_b.inputs[0])

    front_glow = nt.nodes.new("ShaderNodeMapRange")
    front_glow.clamp = True
    front_glow.inputs["From Min"].default_value = 0.0068
    front_glow.inputs["From Max"].default_value = 0.0
    nt.links.new(abs_b.outputs["Value"], front_glow.inputs["Value"])

    glow_ramp = nt.nodes.new("ShaderNodeValToRGB")
    glow_ramp.color_ramp.elements[0].position = 0.0
    glow_ramp.color_ramp.elements[0].color = (0.92, 0.14, 0.006, 1.0)
    e_mid = glow_ramp.color_ramp.elements.new(0.48)
    e_mid.color = (1.0, 0.46, 0.03, 1.0)
    glow_ramp.color_ramp.elements[2].position = 0.90
    glow_ramp.color_ramp.elements[2].color = (1.0, 0.94, 0.68, 1.0)
    nt.links.new(front_glow.outputs["Result"], glow_ramp.inputs["Fac"])

    em_mul1 = nt.nodes.new("ShaderNodeMath")
    em_mul1.operation = "MULTIPLY"
    nt.links.new(front_glow.outputs["Result"], em_mul1.inputs[0])
    nt.links.new(burn_on.outputs[0], em_mul1.inputs[1])

    em_mul2 = nt.nodes.new("ShaderNodeMath")
    em_mul2.operation = "MULTIPLY"
    em_mul2.inputs[1].default_value = 9.5
    nt.links.new(em_mul1.outputs["Value"], em_mul2.inputs[0])

    nf_fuse = nt.nodes.new("ShaderNodeTexNoise")
    nf_fuse.inputs["Scale"].default_value = 420.0
    nf_fuse.inputs["Detail"].default_value = 8.0
    bmp_fuse = nt.nodes.new("ShaderNodeBump")
    bmp_fuse.inputs["Strength"].default_value = 0.28
    bmp_fuse.inputs["Distance"].default_value = 0.00035
    nt.links.new(tex_coord.outputs["Object"], nf_fuse.inputs["Vector"])
    nt.links.new(nf_fuse.outputs["Fac"], bmp_fuse.inputs["Height"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.78
    nt.links.new(mix_col.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(glow_ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(em_mul2.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bmp_fuse.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mats["fuse_core"] = m_fuse

    # 16. Additive Golden-Orange Spark Needles (AGENTS.md Section 3: AddShader + Camera Ray)
    m_spark = bpy.data.materials.new(nm("Additive_Spark_Gold"))
    m_spark.use_nodes = True
    nt = m_spark.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.68, 0.16, 1.0)

    # Camera Ray * (1 - Backfacing)
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], cam_front.inputs[1])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 7.2
    nt.links.new(cam_front.outputs["Value"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], em.inputs["Strength"])

    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    mats["spark_gold"] = m_spark

    # 17. PROCEDURAL 3D TURBULENT FIERY PLASMA FIREBALL (Zero flat bullseye circles!)
    def make_turbulent_plasma(
        name: str,
        noise_scale: float,
        falloff_power: float,
        strength: float,
        c_edge: tuple[float, float, float],
        c_mid: tuple[float, float, float],
        c_hot: tuple[float, float, float],
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(nm(name))
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        lp = nt.nodes.new("ShaderNodeLightPath")
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        em = nt.nodes.new("ShaderNodeEmission")

        # (1 - Facing)^p soft radial edge -> ZERO hard sphere outline
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.50
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(lw.outputs["Facing"], inv.inputs[1])

        soft_edge = nt.nodes.new("ShaderNodeMath")
        soft_edge.operation = "POWER"
        soft_edge.inputs[1].default_value = falloff_power
        nt.links.new(inv.outputs["Value"], soft_edge.inputs[0])

        # Local Object-space 3D turbulent fire noise
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = noise_scale
        noise.inputs["Detail"].default_value = 9.0
        noise.inputs["Roughness"].default_value = 0.65
        noise.inputs["Distortion"].default_value = 1.65
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

        n_map = nt.nodes.new("ShaderNodeMapRange")
        n_map.clamp = True
        n_map.inputs["From Min"].default_value = 0.30
        n_map.inputs["From Max"].default_value = 0.78
        n_map.inputs["To Min"].default_value = 0.14
        n_map.inputs["To Max"].default_value = 1.0
        nt.links.new(noise.outputs["Fac"], n_map.inputs["Value"])

        density = nt.nodes.new("ShaderNodeMath")
        density.operation = "MULTIPLY"
        nt.links.new(soft_edge.outputs["Value"], density.inputs[0])
        nt.links.new(n_map.outputs["Result"], density.inputs[1])

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.08
        ramp.color_ramp.elements[0].color = (*c_edge, 1.0)
        e1 = ramp.color_ramp.elements.new(0.46)
        e1.color = (*c_mid, 1.0)
        ramp.color_ramp.elements[2].position = 0.85
        ramp.color_ramp.elements[2].color = (*c_hot, 1.0)
        nt.links.new(density.outputs["Value"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], em.inputs["Color"])

        s_mul = nt.nodes.new("ShaderNodeMath")
        s_mul.operation = "MULTIPLY"
        s_mul.inputs[1].default_value = strength
        nt.links.new(density.outputs["Value"], s_mul.inputs[0])

        inv_bf = nt.nodes.new("ShaderNodeMath")
        inv_bf.operation = "SUBTRACT"
        inv_bf.inputs[0].default_value = 1.0
        nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

        cam_front = nt.nodes.new("ShaderNodeMath")
        cam_front.operation = "MULTIPLY"
        nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
        nt.links.new(inv_bf.outputs["Value"], cam_front.inputs[1])

        cam_mul = nt.nodes.new("ShaderNodeMath")
        cam_mul.operation = "MULTIPLY"
        nt.links.new(s_mul.outputs["Value"], cam_mul.inputs[0])
        nt.links.new(cam_front.outputs["Value"], cam_mul.inputs[1])
        nt.links.new(cam_mul.outputs["Value"], em.inputs["Strength"])

        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(tr.outputs["BSDF"], add.inputs[0])
        nt.links.new(em.outputs["Emission"], add.inputs[1])
        nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
        return mat

    # Noise scales calibrated for real 0.042m sphere local coordinates!
    mats["plasma_core"] = make_turbulent_plasma(
        "Turbulent_Plasma_Core", 45.0, 2.2, 5.0,
        (0.95, 0.22, 0.01), (1.0, 0.62, 0.08), (1.0, 0.96, 0.80),
    )
    mats["plasma_mid"] = make_turbulent_plasma(
        "Turbulent_Plasma_Fire", 62.0, 2.8, 3.5,
        (0.55, 0.06, 0.002), (1.0, 0.34, 0.02), (1.0, 0.78, 0.20),
    )
    mats["plasma_rim"] = make_turbulent_plasma(
        "Turbulent_Pressure_Wave", 88.0, 3.2, 2.3,
        (0.08, 0.38, 0.85), (0.28, 0.76, 1.0), (1.0, 0.82, 0.45),
    )
    mats["shockwave_ring"] = make_turbulent_plasma(
        "Additive_Shockwave_Ring", 72.0, 2.5, 3.6,
        (0.95, 0.30, 0.02), (1.0, 0.65, 0.16), (1.0, 0.94, 0.72),
    )

    # 18. Photoreal Shattered Dark Cast-Iron Fracture Side Walls (Molten Orange Ember Veins!)
    m_hot = bpy.data.materials.new(nm("Hot_Fracture_Edge"))
    m_hot.use_nodes = True
    nt = m_hot.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    # Local Object space of each shard (stays locked to shard surface while tumbling!)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nf = nt.nodes.new("ShaderNodeTexNoise")
    nf.inputs["Scale"].default_value = 260.0
    nf.inputs["Detail"].default_value = 9.0
    nf.inputs["Roughness"].default_value = 0.72
    nt.links.new(tc.outputs["Object"], nf.inputs["Vector"])

    bsdf.inputs["Base Color"].default_value = (0.075, 0.085, 0.105, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.92
    bsdf.inputs["Roughness"].default_value = 0.46
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.36
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(nf.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    em_mask = nt.nodes.new("ShaderNodeMapRange")
    em_mask.clamp = True
    em_mask.inputs["From Min"].default_value = 0.56
    em_mask.inputs["From Max"].default_value = 0.76
    nt.links.new(nf.outputs["Fac"], em_mask.inputs["Value"])

    em_gain = nt.nodes.new("ShaderNodeValue")
    em_gain.name = "FractureGlowGain"
    em_gain.outputs[0].default_value = 0.0
    shader_nodes["hot_edge_gain_node"] = em_gain

    em_mul = nt.nodes.new("ShaderNodeMath")
    em_mul.operation = "MULTIPLY"
    nt.links.new(em_mask.outputs["Result"], em_mul.inputs[0])
    nt.links.new(em_gain.outputs[0], em_mul.inputs[1])

    bsdf.inputs["Emission Color"].default_value = (1.0, 0.36, 0.025, 1.0)
    nt.links.new(em_mul.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mats["hot_edge"] = m_hot

    # 19. 2-Zone Brushed Satin-Gold / Cyan / Amber 3D Typography (AGENTS.md Section 2!)
    def make_3d_text_mat(
        name: str,
        face_col: tuple[float, float, float],
        emit_strength: float = 1.85,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(nm(name))
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")

        # Object-Space |N.z| > 0.85 separates crisp satin face from dark obsidian-bronze sides
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        vec_t = nt.nodes.new("ShaderNodeVectorTransform")
        vec_t.vector_type = "NORMAL"
        vec_t.convert_from = "WORLD"
        vec_t.convert_to = "OBJECT"
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        abs_z = nt.nodes.new("ShaderNodeMath")
        abs_z.operation = "ABSOLUTE"
        mask = nt.nodes.new("ShaderNodeMapRange")
        mask.clamp = True
        mask.inputs["From Min"].default_value = 0.83
        mask.inputs["From Max"].default_value = 0.92

        nt.links.new(geom.outputs["Normal"], vec_t.inputs["Vector"])
        nt.links.new(vec_t.outputs["Vector"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], abs_z.inputs[0])
        nt.links.new(abs_z.outputs["Value"], mask.inputs["Value"])

        face_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        face_bsdf.inputs["Base Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Metallic"].default_value = 0.62
        face_bsdf.inputs["Roughness"].default_value = 0.24
        face_bsdf.inputs["Emission Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Emission Strength"].default_value = emit_strength

        side_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        side_bsdf.inputs["Base Color"].default_value = (0.025, 0.028, 0.036, 1.0)
        side_bsdf.inputs["Metallic"].default_value = 0.90
        side_bsdf.inputs["Roughness"].default_value = 0.30

        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mask.outputs["Result"], mix.inputs["Fac"])
        nt.links.new(side_bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(face_bsdf.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        return mat

    mats["text_gold"] = make_3d_text_mat("Text3D_SatinGold", (1.0, 0.82, 0.28), 1.88)
    mats["text_cyan"] = make_3d_text_mat("Text3D_CyanTech", (0.28, 0.88, 1.0), 1.92)
    mats["text_amber"] = make_3d_text_mat("Text3D_AmberAlert", (1.0, 0.48, 0.10), 2.00)

    # Dark obsidian keyline backing
    m_keyline = bpy.data.materials.new(nm("Text3D_Obsidian_Keyline"))
    m_keyline.use_nodes = True
    bsdf = m_keyline.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.008, 0.010, 0.016, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.22
    bsdf.inputs["Roughness"].default_value = 0.62
    mats["text_keyline"] = m_keyline

    # Sleek Glowing Cyan & Gold Pointer Leader Lines (AgX-safe emission!)
    def make_pointer_mat(name: str, col_rgb: tuple[float, float, float], emit_s: float) -> bpy.types.Material:
        mat = bpy.data.materials.new(nm(name))
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*col_rgb, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.45
        bsdf.inputs["Roughness"].default_value = 0.20
        bsdf.inputs["Emission Color"].default_value = (*col_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit_s
        return mat

    mats["pointer_cyan"] = make_pointer_mat("Pointer_Line_Cyan", (0.22, 0.88, 1.0), 2.10)
    mats["pointer_gold"] = make_pointer_mat("Pointer_Line_Gold", (1.0, 0.78, 0.22), 2.05)

    return mats, shader_nodes


# ==============================================================================
# HYPER-REALISTIC M67 MECHANICAL GEOMETRY BUILDERS (LOCAL SPACE!)
# ==============================================================================
def _build_hollow_hemisphere(
    name: str,
    phi_start: float,
    phi_end: float,
    n_lat: int,
    n_lon: int,
    mats_list: list[bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    with_inner_scoring: bool = True,
    neck_theta_cut: float = 0.375,
    single_surface_xray: bool = False,
) -> bpy.types.Object:
    """
    Build back or front cast-iron hemisphere centered at (0, 0, 0) with:
    - Integral cast-iron upper neck boss collar meeting the threaded fuze neck cleanly
    - Muted ochre-yellow shoulder identification band
    - Crisp geometric internal diamond pre-scoring V-grooves on back hemisphere
    - Machined cutaway wall rim along Y = 0 (back hemisphere only -> zero Z-fighting!)
    """
    bm = bmesh.new()
    theta_min = neck_theta_cut
    theta_max = math.pi - 0.025

    outer_grid: list[list[bmesh.types.BMVert]] = []
    inner_grid: list[list[bmesh.types.BMVert]] = []

    # Row 0: integral cast neck boss top edge (Z = +0.0424)
    r_boss_o = R_OUTER * math.sin(theta_min) * 0.96
    r_boss_i = R_INNER * math.sin(theta_min) * 0.97
    z_boss_top = 0.0424
    boss_o_row, boss_i_row = [], []
    for j in range(n_lon + 1):
        phi = _lerp(phi_start, phi_end, j / n_lon)
        cp, sp = math.cos(phi), math.sin(phi)
        boss_o_row.append(bm.verts.new((r_boss_o * cp, r_boss_o * sp, z_boss_top)))
        if not single_surface_xray:
            boss_i_row.append(bm.verts.new((r_boss_i * cp, r_boss_i * sp, z_boss_top)))
    outer_grid.append(boss_o_row)
    if not single_surface_xray:
        inner_grid.append(boss_i_row)

    # Sphere rows (1 .. n_lat + 1)
    for i in range(n_lat + 1):
        theta = _lerp(theta_min, theta_max, i / n_lat)
        st, ct = math.sin(theta), math.cos(theta)
        o_row, i_row = [], []
        for j in range(n_lon + 1):
            phi = _lerp(phi_start, phi_end, j / n_lon)
            cp, sp = math.cos(phi), math.sin(phi)
            o_row.append(bm.verts.new((R_OUTER * st * cp, R_OUTER * st * sp, R_OUTER * ct)))
            if not single_surface_xray:
                r_in = R_INNER
                if with_inner_scoring and 1 <= i < n_lat - 1 and 1 <= j < n_lon - 1:
                    # Crisp deep geometric diamond pre-scoring V-grooves on concave inner wall
                    g1 = abs(math.sin(theta * 9.0 + phi * 4.5))
                    g2 = abs(math.sin(theta * 9.0 - phi * 4.5))
                    d_groove = min(g1, g2)
                    if d_groove < 0.25:
                        r_in += (1.0 - d_groove / 0.25) * 0.0023
                i_row.append(bm.verts.new((r_in * st * cp, r_in * st * sp, r_in * ct)))
        outer_grid.append(o_row)
        if not single_surface_xray:
            inner_grid.append(i_row)

    bm.verts.ensure_lookup_table()
    total_rows = len(outer_grid) - 1

    for i in range(total_rows):
        if i == 0:
            is_band = False
        else:
            theta_mid = _lerp(theta_min, theta_max, (i - 0.5) / n_lat)
            is_band = 0.48 <= theta_mid <= 0.61

        for j in range(n_lon):
            fo = bm.faces.new((
                outer_grid[i][j],
                outer_grid[i + 1][j],
                outer_grid[i + 1][j + 1],
                outer_grid[i][j + 1],
            ))
            fo.material_index = 1 if is_band else 0
            fo.smooth = True

            if not single_surface_xray:
                fi = bm.faces.new((
                    inner_grid[i][j],
                    inner_grid[i][j + 1],
                    inner_grid[i + 1][j + 1],
                    inner_grid[i + 1][j],
                ))
                fi.material_index = 3 if with_inner_scoring else 0
                fi.smooth = not with_inner_scoring

    if not single_surface_xray:
        # Machined cutaway wall faces along Y = 0 plane (j=0 and j=n_lon)
        for i in range(total_rows):
            f_r = bm.faces.new((
                outer_grid[i][0], inner_grid[i][0], inner_grid[i + 1][0], outer_grid[i + 1][0],
            ))
            f_r.material_index = 2
            f_r.smooth = False
            f_l = bm.faces.new((
                outer_grid[i][n_lon], outer_grid[i + 1][n_lon], inner_grid[i + 1][n_lon], inner_grid[i][n_lon],
            ))
            f_l.material_index = 2
            f_l.smooth = False

        # Top boss flat lip + bottom pole cap
        for j in range(n_lon):
            f_top = bm.faces.new((
                outer_grid[0][j], outer_grid[0][j + 1], inner_grid[0][j + 1], inner_grid[0][j],
            ))
            f_top.material_index = 2
            f_top.smooth = False

            f_bot = bm.faces.new((
                outer_grid[total_rows][j],
                inner_grid[total_rows][j],
                inner_grid[total_rows][j + 1],
                outer_grid[total_rows][j + 1],
            ))
            f_bot.material_index = 2
            f_bot.smooth = False

    # Close bottom pole disc cleanly so no pinhole appears at Z = -0.042
    pole_o = bm.verts.new((0.0, 0.0, -R_OUTER))
    pole_i = None if single_surface_xray else bm.verts.new((0.0, 0.0, -R_INNER))
    for j in range(n_lon):
        fo_p = bm.faces.new((outer_grid[total_rows][j], pole_o, outer_grid[total_rows][j + 1]))
        fo_p.material_index = 0
        fo_p.smooth = True
        if pole_i is not None:
            fi_p = bm.faces.new((inner_grid[total_rows][j], inner_grid[total_rows][j + 1], pole_i))
            fi_p.material_index = 3 if with_inner_scoring else 0
            fi_p.smooth = False

    return _create_mesh_obj(name, bm, mats_list, col, parent=parent, smooth_all=None)


def _build_fracture_shards(
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """
    48 thick pre-scored dark cast-iron shards parented to Grenade_Root.
    Returns list of dicts with {'obj', 'centroid', 'n_dir'} for bullet-time animation.
    """
    n_lat = 6
    n_lon = 8
    theta_min = 0.375
    theta_max = math.pi - 0.04
    shards: list[dict[str, Any]] = []

    for i in range(n_lat):
        t0 = _lerp(theta_min, theta_max, i / n_lat)
        t1 = _lerp(theta_min, theta_max, (i + 1) / n_lat)
        t_mid = 0.5 * (t0 + t1)
        is_band = 0.46 <= t_mid <= 0.62

        for j in range(n_lon):
            p0 = _lerp(0.0, 2.0 * math.pi, j / n_lon)
            p1 = _lerp(0.0, 2.0 * math.pi, (j + 1) / n_lon)
            p_mid = 0.5 * (p0 + p1)

            n_dir = Vector((
                math.sin(t_mid) * math.cos(p_mid),
                math.sin(t_mid) * math.sin(p_mid),
                math.cos(t_mid),
            )).normalized()
            r_mid = 0.5 * (R_OUTER + R_INNER)
            centroid = n_dir * r_mid

            bm = bmesh.new()
            sub = 3
            og: list[list[bmesh.types.BMVert]] = []
            ig: list[list[bmesh.types.BMVert]] = []
            for si in range(sub + 1):
                th = _lerp(t0, t1, si / sub)
                st, ct = math.sin(th), math.cos(th)
                orow, irow = [], []
                for sj in range(sub + 1):
                    ph = _lerp(p0, p1, sj / sub)
                    cp, sp = math.cos(ph), math.sin(ph)
                    po = Vector((R_OUTER * st * cp, R_OUTER * st * sp, R_OUTER * ct)) - centroid
                    pi = Vector((R_INNER * st * cp, R_INNER * st * sp, R_INNER * ct)) - centroid
                    orow.append(bm.verts.new(po))
                    irow.append(bm.verts.new(pi))
                og.append(orow)
                ig.append(irow)

            for si in range(sub):
                for sj in range(sub):
                    fo = bm.faces.new((og[si][sj], og[si + 1][sj], og[si + 1][sj + 1], og[si][sj + 1]))
                    fo.material_index = 1 if is_band else 0
                    fo.smooth = True
                    fi = bm.faces.new((ig[si][sj], ig[si][sj + 1], ig[si + 1][sj + 1], ig[si + 1][sj]))
                    fi.material_index = 3
                    fi.smooth = False

            for k in range(sub):
                for verts in (
                    (og[k][0], ig[k][0], ig[k + 1][0], og[k + 1][0]),
                    (og[k][sub], og[k + 1][sub], ig[k + 1][sub], ig[k][sub]),
                    (og[0][k], og[0][k + 1], ig[0][k + 1], ig[0][k]),
                    (og[sub][k], ig[sub][k], ig[sub][k + 1], og[sub][k + 1]),
                ):
                    fs = bm.faces.new(verts)
                    fs.material_index = 2
                    fs.smooth = False

            obj = _create_mesh_obj(
                f"{prefix}Shard_{i}_{j}",
                bm,
                [mats["olive_iron"], mats["yellow_band"], mats["hot_edge"], mats["inner_steel"]],
                col,
                parent=parent,
                location=centroid,
                smooth_all=None,
            )
            shards.append({"obj": obj, "centroid": centroid.copy(), "n_dir": n_dir.copy()})

    return shards


def _build_helical_spring(
    name: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Prominent 6-coil helical steel torsion spring wound around the Y-axis hinge axle
    with multi-radius spiral stepping + two visible torsion tension legs!
    """
    coils = 6
    radius = 0.0046
    wire_r = 0.00078
    length = 0.0112
    n_steps = coils * 22
    n_seg = 10
    bm = bmesh.new()

    rings: list[list[bmesh.types.BMVert]] = []
    for s in range(n_steps + 1):
        u = s / n_steps
        ang = u * coils * 2.0 * math.pi
        y = _lerp(-0.5 * length, 0.5 * length, u)
        # Multi-radius spiral stepping so concentric coils catch specular highlights from front & 3/4 view!
        r_here = radius * _lerp(0.84, 1.15, u)
        c_pt = Vector((r_here * math.cos(ang), y, r_here * math.sin(ang)))

        tang = Vector((-r_here * math.sin(ang), length / (coils * 2.0 * math.pi), r_here * math.cos(ang))).normalized()
        rad_out = Vector((math.cos(ang), 0.0, math.sin(ang))).normalized()
        binorm = tang.cross(rad_out).normalized()

        ring = []
        for k in range(n_seg):
            a = k * 2.0 * math.pi / n_seg
            ring.append(bm.verts.new(c_pt + rad_out * (wire_r * math.cos(a)) + binorm * (wire_r * math.sin(a))))
        rings.append(ring)

    for s in range(n_steps):
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm.faces.new((rings[s][k], rings[s][kn], rings[s + 1][kn], rings[s + 1][k]))
            f.smooth = True
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])

    # Add two visible straight torsion legs extending from coil ends (anchor + striker push leg)
    for leg_pts in (
        [Vector((radius * 0.84, -0.5 * length, 0.0)), Vector((radius * 0.84, -0.5 * length, -0.0068))],
        [Vector((radius * 1.15, 0.5 * length, 0.0)), Vector((radius * 1.15 + 0.0055, 0.5 * length, 0.0042))],
    ):
        leg_bm = _make_tube_along_path_bm(leg_pts, wire_r * 0.94, n_seg=8, cap_ends=True)
        temp_mesh = bpy.data.meshes.new("_tmp_leg")
        leg_bm.to_mesh(temp_mesh)
        leg_bm.free()
        bm.from_mesh(temp_mesh)
        bpy.data.meshes.remove(temp_mesh)

    return _create_mesh_obj(name, bm, mat, col, parent=parent, location=HINGE_POS, smooth_all=True)


def _build_safety_spoon(
    name: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Contoured 33-point Catmull-Rom stamped-steel safety lever ("Spoon") on +X side
    with rolled U-channel flanges hugging the grenade shoulder cleanly!
    Origin placed at HINGE_POS so rotational release choreography is natural.
    """
    ctrl_local = [
        Vector((-0.0068, 0.0, 0.0748)),   # Left retaining hook saddle over fuze
        Vector((0.0028, 0.0, 0.0772)),    # Top saddle crest over hinge
        Vector((0.0135, 0.0, 0.0758)),    # Upper right shoulder bend
        Vector((0.0235, 0.0, 0.0685)),    # Neck clearance curve
        Vector((0.0332, 0.0, 0.0545)),    # Upper body sweep
        Vector((0.0422, 0.0, 0.0365)),    # Upper shell hug
        Vector((0.0462, 0.0, 0.0165)),    # Equatorial shoulder hug
        Vector((0.0464, 0.0, -0.0045)),   # Lower grip curve
        Vector((0.0432, 0.0, -0.0215)),   # Bottom flared thumb tab
    ]
    smooth_pts = _catmull_rom(ctrl_local, samples_per_seg=4)  # 33 smooth points
    spine = [p - HINGE_POS for p in smooth_pts]

    bm = bmesh.new()
    half_w = 0.0058
    thick = 0.0012
    flange = 0.0030
    profile_pts = [
        (-half_w, -flange),
        (-half_w, 0.0),
        (half_w, 0.0),
        (half_w, -flange),
        (half_w - thick, -flange),
        (half_w - thick, -thick),
        (-half_w + thick, -thick),
        (-half_w + thick, -flange),
    ]

    rings: list[list[bmesh.types.BMVert]] = []
    for idx, p in enumerate(spine):
        if idx == 0:
            tang = (spine[1] - spine[0]).normalized()
        elif idx == len(spine) - 1:
            tang = (spine[-1] - spine[-2]).normalized()
        else:
            tang = (spine[idx + 1] - spine[idx - 1]).normalized()
        n_out = Vector((-tang.z, 0.0, tang.x)).normalized()
        y_axis = Vector((0.0, 1.0, 0.0))
        w_scale = _lerp(1.04, 0.78, idx / (len(spine) - 1))
        ring = [bm.verts.new(p + y_axis * (py * w_scale) + n_out * pn) for py, pn in profile_pts]
        rings.append(ring)

    n_prof = len(profile_pts)
    for i in range(len(rings) - 1):
        for k in range(n_prof):
            kn = (k + 1) % n_prof
            f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
            f.smooth = True
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])

    return _create_mesh_obj(name, bm, mat, col, parent=parent, location=HINGE_POS, smooth_all=None)


def _build_striker_hammer(
    name: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Forged steel striker hammer + tapered firing pin pivoting around HINGE_POS.
    Exact firing-pin tip at local (+STRIKER_REACH, 0, 0) so rotating around Y to
    STRIKER_IMPACT_DEG (-243.435 deg) slams dead-center onto PRIMER_POS (0, 0, +0.047)!
    """
    bm = bmesh.new()
    pin_len = 0.0036
    arm_len = STRIKER_REACH - pin_len  # ~0.0143m

    # Cylindrical pivot collar around hinge axle
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.0028, radius2=0.0028, depth=0.0068,
        matrix=Matrix.Rotation(math.pi / 2.0, 4, "X"),
    )
    # Forged beveled striker arm
    arm_bm = _make_beveled_box_bm((arm_len, 0.0046, 0.0028), 0.00045, segments=2)
    bmesh.ops.translate(arm_bm, verts=arm_bm.verts, vec=(0.5 * arm_len, 0.0, 0.0))
    tmp = bpy.data.meshes.new("_tmp_arm")
    arm_bm.to_mesh(tmp)
    arm_bm.free()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)

    # Weighted forged hammer head block
    head_bm = _make_beveled_box_bm((0.0042, 0.0054, 0.0048), 0.0006, segments=2)
    bmesh.ops.translate(head_bm, verts=head_bm.verts, vec=(arm_len - 0.0012, 0.0, 0.0010))
    tmp2 = bpy.data.meshes.new("_tmp_head")
    head_bm.to_mesh(tmp2)
    head_bm.free()
    bm.from_mesh(tmp2)
    bpy.data.meshes.remove(tmp2)

    # Tapered machined firing pin tip ending right at X = +STRIKER_REACH
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=16,
        radius1=0.00145, radius2=0.00035, depth=pin_len,
        matrix=Matrix.Translation((arm_len + 0.5 * pin_len, 0.0, 0.0)) @ Matrix.Rotation(math.pi / 2.0, 4, "Y"),
    )

    return _create_mesh_obj(
        name, bm, mat, col, parent=parent,
        location=HINGE_POS,
        rotation=(0.0, math.radians(STRIKER_COCKED_DEG), 0.0),
        smooth_all=True,
    )


def _build_pull_pin_assembly(
    mat_chrome: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    prefix: str = "",
) -> bpy.types.Object:
    """
    Authentic split cotter pin (two shafts + splayed retaining ends + forged eye)
    coupled to a heavy knurled/split chrome pull ring on the -X side!
    """
    pin_rig = bpy.data.objects.new(f"{prefix}PullPin_Rig", None)
    pin_rig.location = Vector((0.0, 0.0, HINGE_POS.z))
    pin_rig.parent = parent
    _link_obj(pin_rig, col)

    # Upper & lower split cotter shafts with authentic bent retaining tips on +X exit
    shaft_r = 0.00080
    top_pts = [
        Vector((-0.0195, 0.0, +0.00080)),
        Vector((+0.0132, 0.0, +0.00080)),
        Vector((+0.0168, 0.0, +0.00245)),
    ]
    bot_pts = [
        Vector((-0.0195, 0.0, -0.00080)),
        Vector((+0.0132, 0.0, -0.00080)),
        Vector((+0.0162, 0.0, -0.00220)),
    ]
    _create_mesh_obj(
        f"{prefix}Cotter_Leg_Top",
        _make_tube_along_path_bm(top_pts, shaft_r, n_seg=12, cap_ends=True),
        mat_chrome, col, parent=pin_rig,
    )
    _create_mesh_obj(
        f"{prefix}Cotter_Leg_Bot",
        _make_tube_along_path_bm(bot_pts, shaft_r, n_seg=12, cap_ends=True),
        mat_chrome, col, parent=pin_rig,
    )

    # Forged cotter pin eye loop on -X side
    eye_bm = _make_torus_bm(major_r=0.0024, minor_r=0.00085, major_seg=28, minor_seg=12)
    _create_mesh_obj(
        f"{prefix}Cotter_Eye",
        eye_bm, mat_chrome, col, parent=pin_rig,
        location=(-0.0208, 0.0, 0.0),
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )

    # Heavy circular chrome split pull ring (double-turn overlapping ring detail!)
    ring_bm = _make_torus_bm(major_r=0.0146, minor_r=0.00162, major_seg=64, minor_seg=18)
    _create_mesh_obj(
        f"{prefix}Pull_Ring_Main",
        ring_bm, mat_chrome, col, parent=pin_rig,
        location=(-0.0342, 0.0, -0.0018),
        rotation=(math.pi / 2.0, 0.0, 0.14),
    )
    # Second offset split-coil ring seam for authentic key-ring construction
    ring2_bm = _make_torus_bm(
        major_r=0.0146, minor_r=0.00120, major_seg=48, minor_seg=14,
        ang_start=0.25, ang_end=2.0 * math.pi - 0.35, cap_ends=True,
    )
    _create_mesh_obj(
        f"{prefix}Pull_Ring_SplitCoil",
        ring2_bm, mat_chrome, col, parent=pin_rig,
        location=(-0.0342, 0.0011, -0.0018),
        rotation=(math.pi / 2.0, 0.0, 0.14),
    )

    return pin_rig


def _build_fuze_and_internals(
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    prefix: str = "",
) -> dict[str, Any]:
    parts: dict[str, Any] = {}

    def nm(base: str) -> str:
        return f"{prefix}{base}"

    # 1. Threaded Neck Collar + Lathe-Turned Threads + Hexagonal Wrench Collar (Z = +0.038 .. +0.052)
    # Front faces (y < 0) use front_fuze_xray so they stay 100% solid in Beat 1 and reveal the Brass Primer in X-ray!
    bm_col = bmesh.new()
    bmesh.ops.create_cone(
        bm_col, cap_ends=True, cap_tris=False, segments=48,
        radius1=0.0132, radius2=0.0132, depth=0.0102,
    )
    for f in bm_col.faces:
        f.material_index = 1 if f.calc_center_median().y < -0.001 else 0
        f.smooth = True
    parts["collar"] = _create_mesh_obj(
        nm("Neck_Threaded_Collar"), bm_col, [mats["neck_collar"], mats["front_fuze_xray"]], col,
        parent=parent, location=(0.0, 0.0, 0.0434), smooth_all=None,
    )

    # 4 machined thread rings around neck barrel (front half X-ray transparent in Beat 2–5!)
    bm_thr = bmesh.new()
    for tz in (0.0402, 0.0422, 0.0442, 0.0462):
        t_bm = _make_torus_bm(0.0134, 0.00055, major_seg=40, minor_seg=8)
        bmesh.ops.translate(t_bm, verts=t_bm.verts, vec=(0.0, 0.0, tz))
        tmp = bpy.data.meshes.new("_tmp_thr")
        t_bm.to_mesh(tmp)
        t_bm.free()
        bm_thr.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    for f in bm_thr.faces:
        f.material_index = 1 if f.calc_center_median().y < -0.001 else 0
        f.smooth = True
    parts["neck_threads"] = _create_mesh_obj(
        nm("Neck_Threads"), bm_thr, [mats["cut_steel"], mats["front_fuze_xray"]], col,
        parent=parent, smooth_all=None,
    )

    # Authentic beveled Hexagonal Collar Nut (Z = +0.0478 .. +0.0520, front faces X-ray!)
    bm_hex = bmesh.new()
    bmesh.ops.create_cone(
        bm_hex, cap_ends=True, cap_tris=False, segments=6,
        radius1=0.0158, radius2=0.0156, depth=0.0044,
    )
    bmesh.ops.bevel(bm_hex, geom=list(bm_hex.edges), offset=0.00055, segments=2, affect="EDGES")
    for f in bm_hex.faces:
        f.material_index = 1 if f.calc_center_median().y < -0.001 else 0
        f.smooth = False
    parts["flange"] = _create_mesh_obj(
        nm("Neck_Hex_Flange"), bm_hex, [mats["fuze_metal"], mats["front_fuze_xray"]], col,
        parent=parent, location=(0.0, 0.0, 0.0498), smooth_all=None,
    )

    # 2. Sculpted Chamfered Die-Cast M213 Fuze Head (Z = +0.050 .. +0.074)
    parts["fuze_back"] = _create_mesh_obj(
        nm("Fuze_Back_Body"),
        _make_beveled_box_bm((0.0246, 0.0066, 0.0226), 0.0010, segments=3),
        mats["fuze_metal"], col, parent=parent, location=(0.0, 0.0064, 0.0622),
    )
    parts["fuze_wall_L"] = _create_mesh_obj(
        nm("Fuze_Wall_L"),
        _make_beveled_box_bm((0.0030, 0.0144, 0.0216), 0.0006, segments=2),
        mats["fuze_metal"], col, parent=parent, location=(-0.0110, 0.0006, 0.0622),
    )
    parts["fuze_wall_R"] = _create_mesh_obj(
        nm("Fuze_Wall_R"),
        _make_beveled_box_bm((0.0030, 0.0144, 0.0185), 0.0006, segments=2),
        mats["fuze_metal"], col, parent=parent, location=(+0.0110, 0.0006, 0.0608),
    )

    # Curved die-cast hinge ears flanking the hinge axle at (+0.008, ±0.0062, +0.063)
    for side, sy in (("Back", +0.0064), ("Front", -0.0056)):
        bm_ear = bmesh.new()
        bmesh.ops.create_cone(
            bm_ear, cap_ends=True, cap_tris=False, segments=24,
            radius1=0.0064, radius2=0.0062, depth=0.0024,
            matrix=Matrix.Rotation(math.pi / 2.0, 4, "X"),
        )
        ear_mat = mats["fuze_metal"] if side == "Back" else mats["front_fuze_xray"]
        parts[f"fuze_ear_{side}"] = _create_mesh_obj(
            nm(f"Fuze_HingeEar_{side}"), bm_ear, ear_mat, col,
            parent=parent, location=(HINGE_POS.x, sy, HINGE_POS.z + 0.0018),
        )

    # Recessed chamfered pin-hole boss on -X wall
    boss_bm = _make_torus_bm(0.0024, 0.00065, major_seg=24, minor_seg=10)
    parts["pin_boss"] = _create_mesh_obj(
        nm("Fuze_PinBoss"), boss_bm, mats["neck_collar"], col,
        parent=parent, location=(-0.0128, 0.0, HINGE_POS.z),
        rotation=(0.0, math.pi / 2.0, 0.0),
    )

    # Front X-Ray Cover Plate on Fuze Head (solid dark gunmetal -> cyan X-ray glass!)
    parts["fuze_front"] = _create_mesh_obj(
        nm("Fuze_Front_Cover_XRay"),
        _make_beveled_box_bm((0.0248, 0.0024, 0.0222), 0.00075, segments=3),
        mats["front_fuze_xray"], col, parent=parent, location=(0.0, -0.0068, 0.0622),
    )

    # Hinge Axle Pin (flush inside hinge ears so it never pokes through front solid plate!)
    bm_ax = bmesh.new()
    bmesh.ops.create_cone(
        bm_ax, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.00145, radius2=0.00145, depth=0.0136,
        matrix=Matrix.Rotation(math.pi / 2.0, 4, "X"),
    )
    parts["axle"] = _create_mesh_obj(
        nm("Hinge_Axle"), bm_ax, mats["chrome_steel"], col,
        parent=parent, location=HINGE_POS,
    )

    # 3. Prominent 6-Coil Helical Torsion Spring
    parts["spring"] = _build_helical_spring(nm("Torsion_Spring"), mats["spring_steel"], col, parent)

    # 4. Forged Steel Striker Hammer + Tapered Firing Pin
    parts["striker"] = _build_striker_hammer(nm("Striker_Hammer"), mats["striker_steel"], col, parent)

    # 5. Contoured 33-Point Safety Lever ("Spoon")
    parts["spoon"] = _build_safety_spoon(nm("Safety_Spoon"), mats["spoon_steel"], col, parent)

    # 6. Split Cotter Pin & Heavy Chrome Pull Ring
    parts["pull_pin"] = _build_pull_pin_assembly(mats["chrome_steel"], col, parent, prefix=prefix)

    # 7. Brass Percussion Primer Cap at PRIMER_POS (0, 0, +0.047)
    bm_pr = bmesh.new()
    bmesh.ops.create_cone(
        bm_pr, cap_ends=True, cap_tris=False, segments=32,
        radius1=0.0056, radius2=0.0056, depth=0.0052,
    )
    parts["primer"] = _create_mesh_obj(
        nm("Brass_Primer_Cap"), bm_pr, mats["brass"], col,
        parent=parent, location=(0.0, 0.0, PRIMER_POS.z - 0.0006),
    )

    bm_pd = bmesh.new()
    bmesh.ops.create_cone(
        bm_pd, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.0029, radius2=0.0026, depth=0.0014,
    )
    parts["primer_dot"] = _create_mesh_obj(
        nm("Primer_Anvil_Dot"), bm_pd, mats["copper"], col,
        parent=parent, location=(0.0, 0.0, PRIMER_POS.z + 0.0022),
    )

    # 8. Machined Gold-Brass Cutaway Delay Column (Z = -0.005 .. +0.046, bold open front channel!)
    bm_tube = bmesh.new()
    r_o, r_i = 0.0074, 0.0044
    ang_start = -math.pi * 0.18
    ang_end = math.pi * 1.18
    n_seg = 36
    o_bot, o_top, i_bot, i_top = [], [], [], []
    for k in range(n_seg + 1):
        a = _lerp(ang_start, ang_end, k / n_seg)
        ca, sa = math.cos(a), math.sin(a)
        o_bot.append(bm_tube.verts.new((r_o * ca, r_o * sa, DELAY_Z_BOT)))
        o_top.append(bm_tube.verts.new((r_o * ca, r_o * sa, DELAY_Z_TOP)))
        i_bot.append(bm_tube.verts.new((r_i * ca, r_i * sa, DELAY_Z_BOT)))
        i_top.append(bm_tube.verts.new((r_i * ca, r_i * sa, DELAY_Z_TOP)))

    for k in range(n_seg):
        fo = bm_tube.faces.new((o_bot[k], o_bot[k + 1], o_top[k + 1], o_top[k]))
        fo.material_index = 0
        fo.smooth = True
        fi = bm_tube.faces.new((i_bot[k], i_top[k], i_top[k + 1], i_bot[k + 1]))
        fi.material_index = 0
        fi.smooth = True
        ft = bm_tube.faces.new((o_top[k], o_top[k + 1], i_top[k + 1], i_top[k]))
        ft.material_index = 1
        ft.smooth = False
        fb = bm_tube.faces.new((o_bot[k], i_bot[k], i_bot[k + 1], o_bot[k + 1]))
        fb.material_index = 1
        fb.smooth = False

    for verts in (
        (o_bot[0], o_top[0], i_top[0], i_bot[0]),
        (o_bot[n_seg], i_bot[n_seg], i_top[n_seg], o_top[n_seg]),
    ):
        fc = bm_tube.faces.new(verts)
        fc.material_index = 1
        fc.smooth = False

    parts["delay_tube"] = _create_mesh_obj(
        nm("Brass_Delay_Housing"), bm_tube, [mats["brass"], mats["cut_steel"]],
        col, parent=parent, smooth_all=None,
    )

    # 3 Stepped Machined C-Section Brass Collar Rings (matching cutaway open front channel!)
    delay_collars: list[bpy.types.Object] = []
    for idx, rz in enumerate((0.0365, 0.0205, 0.0045)):
        ring_bm = _make_torus_bm(
            major_r=0.0078, minor_r=0.00095, major_seg=32, minor_seg=12,
            ang_start=ang_start, ang_end=ang_end, cap_ends=True,
        )
        c_obj = _create_mesh_obj(
            nm(f"Brass_Delay_Collar_{idx}"), ring_bm, mats["brass"],
            col, parent=parent, location=(0.0, 0.0, rz),
        )
        delay_collars.append(c_obj)
    parts["delay_collars"] = delay_collars

    # 9. Live Burning 4-Second Delay Fuse Core (Z = -0.0045 .. +0.0455, slim so gold brass channel gleams!)
    bm_fc = bmesh.new()
    bmesh.ops.create_cone(
        bm_fc, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.00285, radius2=0.00285, depth=0.0500,
    )
    parts["fuse_core"] = _create_mesh_obj(
        nm("Delay_Fuse_Core"), bm_fc, mats["fuse_core"], col,
        parent=parent, location=(0.0, 0.0, 0.0205),
    )

    # 10. Machined Copper Booster Capsule (Z = -0.016 .. -0.004)
    bm_bo = bmesh.new()
    bmesh.ops.create_cone(
        bm_bo, cap_ends=True, cap_tris=False, segments=32,
        radius1=0.0078, radius2=0.0080, depth=0.0120,
    )
    bmesh.ops.bevel(bm_bo, geom=list(bm_bo.edges), offset=0.00075, segments=2, affect="EDGES")
    parts["booster"] = _create_mesh_obj(
        nm("Copper_Booster_Cap"), bm_bo, mats["copper"], col,
        parent=parent, location=(0.0, 0.0, 0.5 * (BOOSTER_Z_TOP + BOOSTER_Z_BOT)),
    )

    # 11. Deep Recessed Educational Core Chamber Backing (leaves 80%+ of diamond pre-scored inner shell visible!)
    bm_core = bmesh.new()
    n_la, n_lo = 14, 16
    r_c = R_INNER - 0.0032
    grid: list[list[bmesh.types.BMVert]] = []
    for i in range(n_la + 1):
        th = _lerp(0.85, math.pi - 0.65, i / n_la)
        row = []
        for j in range(n_lo + 1):
            ph = _lerp(0.68, math.pi - 0.68, j / n_lo)
            x = r_c * math.sin(th) * math.cos(ph)
            y = max(0.0230, r_c * math.sin(th) * math.sin(ph))
            z = r_c * math.cos(th)
            row.append(bm_core.verts.new((x, y, z)))
        grid.append(row)
    for i in range(n_la):
        for j in range(n_lo):
            f = bm_core.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
            f.smooth = True
    parts["core_fill"] = _create_mesh_obj(
        nm("Core_Chamber_Fill"), bm_core, mats["core_fill"], col, parent=parent,
    )

    return parts


# ==============================================================================
# VOLUMETRIC OPTICS: SPARKS, BURN EMBERS, TURBULENT FIREBALL & SHOCKWAVE
# ==============================================================================
def _build_optics_fx(
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    prefix: str = "",
) -> dict[str, Any]:
    rng = random.Random(108)
    fx: dict[str, Any] = {}

    def nm(base: str) -> str:
        return f"{prefix}{base}"

    # 1. Primer Strike Additive Spark Burst (Open-ended cones, end_fill_type="NOTHING"!)
    spark_rig = bpy.data.objects.new(nm("Primer_Spark_Rig"), None)
    spark_rig.location = PRIMER_POS + Vector((0.0, -0.0018, 0.0016))
    spark_rig.parent = parent
    _link_obj(spark_rig, col)

    bm_orb = bmesh.new()
    bmesh.ops.create_icosphere(bm_orb, subdivisions=2, radius=0.0046)
    _create_mesh_obj(nm("Primer_Flash_Orb"), bm_orb, mats["plasma_core"], col, parent=spark_rig)

    for k in range(18):
        az = rng.uniform(0.0, 2.0 * math.pi)
        el = rng.uniform(0.18, 1.38)
        d = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))).normalized()
        length = rng.uniform(0.011, 0.025)
        bm_n = bmesh.new()
        # Open-ended cone (cap_ends=False == end_fill_type="NOTHING") per AGENTS.md Section 3!
        bmesh.ops.create_cone(
            bm_n, cap_ends=False, cap_tris=False, segments=8,
            radius1=0.00095, radius2=0.00010, depth=length,
        )
        _create_mesh_obj(
            nm(f"Primer_Spark_{k:02d}"), bm_n, mats["spark_gold"], col,
            parent=spark_rig,
            location=d * (0.5 * length),
            rotation=d.to_track_quat("Z", "Y").to_euler(),
        )
    fx["primer_spark_rig"] = spark_rig

    # 2. Internal Dynamic Point Light (calibrated for real 0.042m metric inverse-square falloff!)
    burn_light_data = bpy.data.lights.new(nm("Fuse_Burn_Light"), type="POINT")
    burn_light_data.color = (1.0, 0.48, 0.08)
    burn_light_data.energy = 0.0
    burn_light_data.shadow_soft_size = 0.0055
    burn_light = bpy.data.objects.new(nm("Fuse_Burn_Light"), burn_light_data)
    burn_light.location = Vector((0.0, -0.0075, BURN_Z_START))
    burn_light.parent = parent
    _link_obj(burn_light, col)
    fx["burn_light"] = burn_light

    # 3. 14 Sleek Upward Ember Needles rising off the travelling burn front
    fuse_embers: list[bpy.types.Object] = []
    for k in range(14):
        bm_e = bmesh.new()
        bmesh.ops.create_cone(
            bm_e, cap_ends=False, cap_tris=False, segments=6,
            radius1=0.00055, radius2=0.00006, depth=0.0028,
        )
        emb = _create_mesh_obj(
            nm(f"Fuse_Ember_{k:02d}"), bm_e, mats["spark_gold"], col, parent=parent,
        )
        fuse_embers.append(emb)
    fx["fuse_embers"] = fuse_embers

    # 4. 3D Procedural Turbulent Fiery Plasma Fireball (3 displaced icospheres, ZERO flat circles!)
    plasma_specs = [
        ("Pressure_Core_White", mats["plasma_core"], 0.0235, 0.0030, 11),
        ("Pressure_Mid_Amber", mats["plasma_mid"], 0.0335, 0.0044, 23),
        ("Pressure_Outer_Wave", mats["plasma_rim"], 0.0405, 0.0035, 37),
    ]
    plasma_spheres: list[bpy.types.Object] = []
    for pname, pmat, base_r, disp_amp, seed in plasma_specs:
        bm_p = bmesh.new()
        bmesh.ops.create_icosphere(bm_p, subdivisions=4, radius=base_r)
        for v in bm_p.verts:
            p = v.co.normalized()
            d = (
                math.sin(p.x * 5.2 + seed) * math.cos(p.y * 4.7 - seed)
                + 0.5 * math.sin(p.z * 7.5 + p.x * 4.1)
            )
            v.co = v.co + p * (disp_amp * d)
        ps = _create_mesh_obj(nm(pname), bm_p, pmat, col, parent=parent, location=(0.0, 0.0, 0.0))
        plasma_spheres.append(ps)
    fx["plasma_spheres"] = plasma_spheres

    # 5. Pulsing Internal Pressure Wave Rings (3 tilted rings)
    pressure_rings: list[bpy.types.Object] = []
    for idx in range(3):
        bm_r = _make_torus_bm(0.031, 0.0012, major_seg=48, minor_seg=12)
        pr = _create_mesh_obj(
            nm(f"Pressure_Ring_{idx}"), bm_r, mats["plasma_rim"], col,
            parent=parent,
            location=(0.0, 0.0, 0.0),
            rotation=(math.radians(76 - idx * 22), math.radians(14 * idx), 0.0),
        )
        pressure_rings.append(pr)
    fx["pressure_rings"] = pressure_rings

    # 6. Soft-Edge Glowing Radial Shockwave Ring (Beat 6)
    bm_sh = _make_torus_bm(0.054, 0.0040, major_seg=72, minor_seg=18)
    shock = _create_mesh_obj(
        nm("BulletTime_Shockwave_Ring"), bm_sh, mats["shockwave_ring"], col,
        parent=parent,
        location=(0.0, 0.0, 0.0),
        rotation=(math.radians(82.0), math.radians(-8.0), 0.0),
    )
    fx["shockwave_ring"] = shock

    # 7. 38 Sleek Radial Molten Blast Spark Streaks (Open-ended additive cones)
    blast_embers: list[dict[str, Any]] = []
    for k in range(38):
        th = rng.uniform(0.32, math.pi - 0.28)
        ph = rng.uniform(0.0, 2.0 * math.pi)
        nd = Vector((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
        length = rng.uniform(0.0055, 0.0135)
        bm_eb = bmesh.new()
        bmesh.ops.create_cone(
            bm_eb, cap_ends=False, cap_tris=False, segments=6,
            radius1=0.00072, radius2=0.00010, depth=length,
        )
        eb = _create_mesh_obj(
            nm(f"Blast_Ember_{k:02d}"), bm_eb, mats["spark_gold"], col,
            parent=parent,
            rotation=nd.to_track_quat("Z", "Y").to_euler(),
        )
        blast_embers.append({
            "obj": eb,
            "nd": nd,
            "d_fast": rng.uniform(0.075, 0.120),
            "d_slow": rng.uniform(0.095, 0.142),
        })
    fx["blast_embers"] = blast_embers

    return fx


# ==============================================================================
# CRISP 3D ENGINEERING COMPONENT LABELS + GLOWING POINTER LEADER LINES
# ==============================================================================
def _build_3d_labels(
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object,
    font_path: Path | str | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """
    Build all 3D engineering callouts, pointer leader lines + target dots, and live
    Beat 4 countdown numbers, ALL parented directly to `Grenade_Root` so they travel
    in lockstep with the grenade at any world transform!
    """
    resolved_font = Path(font_path) if font_path else FONT_PATH
    font = bpy.data.fonts.load(str(resolved_font)) if resolved_font.is_file() else None

    def nm(base: str) -> str:
        return f"{prefix}{base}"

    def make_label_with_pointer(
        key: str,
        text: str,
        text_pos: tuple[float, float, float],
        size: float,
        text_mat: bpy.types.Material,
        pointer_mat: bpy.types.Material | None = None,
        pointer_pts: list[tuple[float, float, float]] | None = None,
        shelf_span: tuple[float, float, float, float] | None = None,
        align_x: str = "CENTER",
    ) -> dict[str, Any]:
        rig = bpy.data.objects.new(nm(f"LabelRig_{key}"), None)
        rig.location = Vector((0.0, 0.0, 0.0))
        rig.parent = parent
        _link_obj(rig, col)

        # Text holder oriented facing -Y camera (local +Z -> -Y, local +Y -> +Z)
        text_anchor = bpy.data.objects.new(nm(f"TextAnchor_{key}"), None)
        text_anchor.location = Vector(text_pos)
        text_anchor.rotation_euler = Euler((math.pi / 2.0, 0.0, 0.0))
        text_anchor.parent = rig
        _link_obj(text_anchor, col)

        cdata = bpy.data.curves.new(nm(f"Text_{key}"), type="FONT")
        cdata.body = text
        if font:
            cdata.font = font
        cdata.size = size
        cdata.align_x = align_x
        cdata.align_y = "CENTER"
        cdata.extrude = size * 0.078
        cdata.bevel_depth = size * 0.011
        cdata.bevel_resolution = 2
        cdata.space_character = 1.04
        tobj = bpy.data.objects.new(nm(f"Text_{key}"), cdata)
        tobj.data.materials.append(text_mat)
        tobj.parent = text_anchor
        _link_obj(tobj, col)

        # Dark obsidian keyline backing recessed behind text (+Y in Grenade_Root = -Z in text_anchor)
        kdata = bpy.data.curves.new(nm(f"TextKeyline_{key}"), type="FONT")
        kdata.body = text
        if font:
            kdata.font = font
        kdata.size = size
        kdata.align_x = align_x
        kdata.align_y = "CENTER"
        kdata.extrude = size * 0.055
        kdata.offset = size * 0.068
        kdata.space_character = 1.04
        kobj = bpy.data.objects.new(nm(f"TextKeyline_{key}"), kdata)
        kobj.location = Vector((0.0, 0.0, -size * 0.14))
        kobj.data.materials.append(mats["text_keyline"])
        kobj.parent = text_anchor
        _link_obj(kobj, col)

        pointer_objs: list[bpy.types.Object] = []
        if pointer_pts and len(pointer_pts) >= 2 and pointer_mat is not None:
            pts_v = [Vector(p) for p in pointer_pts]
            line_bm = _make_tube_along_path_bm(pts_v, radius=0.00038, n_seg=8, cap_ends=True)
            line_obj = _create_mesh_obj(
                nm(f"PointerLine_{key}"), line_bm, pointer_mat, col, parent=rig,
            )
            pointer_objs.append(line_obj)

            if shelf_span is not None:
                x0, x1, sy, sz = shelf_span
                sh_bm = _make_tube_along_path_bm(
                    [Vector((x0, sy, sz)), Vector((x1, sy, sz))],
                    radius=0.00038, n_seg=8, cap_ends=True,
                )
                sh_obj = _create_mesh_obj(
                    nm(f"PointerShelf_{key}"), sh_bm, pointer_mat, col, parent=rig,
                )
                pointer_objs.append(sh_obj)

            # Glowing endpoint dot at component anchor (pts_v[0])
            dot_bm = bmesh.new()
            bmesh.ops.create_icosphere(dot_bm, subdivisions=2, radius=0.00110)
            dot_obj = _create_mesh_obj(
                nm(f"PointerDot_{key}"), dot_bm, pointer_mat, col,
                parent=rig, location=pts_v[0],
            )
            pointer_objs.append(dot_obj)

            # Subtle precision reticle ring around target dot facing -Y camera
            ret_bm = _make_torus_bm(0.00205, 0.00024, major_seg=24, minor_seg=8)
            ret_obj = _create_mesh_obj(
                nm(f"PointerReticle_{key}"), ret_bm, pointer_mat, col,
                parent=rig, location=pts_v[0], rotation=(math.pi / 2.0, 0.0, 0.0),
            )
            pointer_objs.append(ret_obj)

        return {
            "rig": rig,
            "text_anchor": text_anchor,
            "text_obj": tobj,
            "keyline_obj": kobj,
            "pointer_objs": pointer_objs,
            "all_objs": [tobj, kobj] + pointer_objs,
        }

    labels: dict[str, Any] = {}

    # 1. SAFETY PIN (points to pull pin/ring on -X side, Beat 1/2)
    labels["safety_pin"] = make_label_with_pointer(
        "SafetyPin", "SAFETY PIN",
        text_pos=(-0.024, -0.004, 0.091),
        size=0.0054,
        text_mat=mats["text_gold"],
        pointer_mat=mats["pointer_gold"],
        pointer_pts=[
            (-0.024, -0.0035, 0.0645),
            (-0.024, -0.0045, 0.085),
        ],
        shelf_span=(-0.042, -0.006, -0.0045, 0.085),
        align_x="CENTER",
    )

    # 2. SAFETY LEVER (SPOON) (points directly onto curved spoon front contour, Beat 2)
    labels["safety_lever"] = make_label_with_pointer(
        "SafetyLever", "SAFETY LEVER (SPOON)",
        text_pos=(0.006, -0.004, -0.056),
        size=0.0046,
        text_mat=mats["text_cyan"],
        pointer_mat=mats["pointer_cyan"],
        pointer_pts=[
            (0.0460, -0.0045, -0.008),
            (0.0460, -0.0045, -0.049),
            (-0.024, -0.0045, -0.049),
        ],
        align_x="CENTER",
    )

    # 3. TORSION SPRING (points straight to gleaming 6-coil helical spring top-right, Beat 2–3)
    labels["torsion_spring"] = make_label_with_pointer(
        "TorsionSpring", "TORSION SPRING",
        text_pos=(0.015, -0.004, 0.092),
        size=0.0052,
        text_mat=mats["text_gold"],
        pointer_mat=mats["pointer_gold"],
        pointer_pts=[
            (0.0090, -0.0052, 0.0650),
            (0.015, -0.0045, 0.086),
        ],
        shelf_span=(-0.009, 0.039, -0.0045, 0.086),
        align_x="CENTER",
    )

    # 4. STRIKER HAMMER (points to striker hammer top-left, Beat 3)
    labels["striker_hammer"] = make_label_with_pointer(
        "StrikerHammer", "STRIKER HAMMER",
        text_pos=(-0.014, -0.004, 0.092),
        size=0.0052,
        text_mat=mats["text_gold"],
        pointer_mat=mats["pointer_gold"],
        pointer_pts=[
            (-0.0030, -0.0045, 0.0695),
            (-0.014, -0.0045, 0.086),
        ],
        shelf_span=(-0.038, 0.010, -0.0045, 0.086),
        align_x="CENTER",
    )

    # 5. BRASS PRIMER (points directly to gleaming percussion primer cap center, Beat 3)
    labels["brass_primer"] = make_label_with_pointer(
        "BrassPrimer", "BRASS PRIMER",
        text_pos=(0.034, -0.004, 0.053),
        size=0.0048,
        text_mat=mats["text_cyan"],
        pointer_mat=mats["pointer_cyan"],
        pointer_pts=[
            (0.0052, -0.0050, 0.0470),
            (0.053, -0.0045, 0.0470),
        ],
        align_x="CENTER",
    )

    # 6. 4-SEC DELAY FUSE (points to glowing machined brass column, Beat 4)
    labels["delay_fuse"] = make_label_with_pointer(
        "DelayFuse", "4-SEC DELAY FUSE",
        text_pos=(0.000, -0.004, -0.056),
        size=0.0050,
        text_mat=mats["text_gold"],
        pointer_mat=mats["pointer_gold"],
        pointer_pts=[
            (0.0076, -0.0055, 0.0180),
            (0.026, -0.0050, 0.0180),
            (0.026, -0.0045, -0.049),
            (-0.026, -0.0045, -0.049),
        ],
        align_x="CENTER",
    )

    # 7. PRE-SCORED SHELL (points to internal diamond V-grooves, Beat 5–6)
    labels["pre_scored_shell"] = make_label_with_pointer(
        "PreScoredShell", "PRE-SCORED SHELL",
        text_pos=(0.0, -0.004, 0.094),
        size=0.0054,
        text_mat=mats["text_gold"],
        pointer_mat=mats["pointer_gold"],
        pointer_pts=[
            (-0.025, -0.0020, 0.022),
            (-0.028, -0.0045, 0.087),
            (0.028, -0.0045, 0.087),
        ],
        align_x="CENTER",
    )

    # 8. Live 3D Countdown ("4.0 s", "3.0 s", "2.0 s", "1.0 s", "0.0 s") cleanly above grenade!
    countdown: dict[str, dict[str, Any]] = {}
    for step_key, body, mat_key in (
        ("4.0", "4.0 s", "text_gold"),
        ("3.0", "3.0 s", "text_gold"),
        ("2.0", "2.0 s", "text_amber"),
        ("1.0", "1.0 s", "text_amber"),
        ("0.0", "0.0 s", "text_amber"),
    ):
        countdown[step_key] = make_label_with_pointer(
            f"Timer_{step_key.replace('.', '_')}",
            body,
            text_pos=(0.0, -0.003, 0.095),
            size=0.0138,
            text_mat=mats[mat_key],
            pointer_mat=None,
            pointer_pts=None,
            align_x="CENTER",
        )
    labels["countdown"] = countdown

    labels["all_rigs"] = [
        labels["safety_pin"]["rig"],
        labels["safety_lever"]["rig"],
        labels["torsion_spring"]["rig"],
        labels["striker_hammer"]["rig"],
        labels["brass_primer"]["rig"],
        labels["delay_fuse"]["rig"],
        labels["pre_scored_shell"]["rig"],
    ] + [entry["rig"] for entry in countdown.values()]

    return labels


# ==============================================================================
# PUBLIC API: BUILD & ANIMATE
# ==============================================================================
def build_grenade_assembly(
    collection: bpy.types.Collection | None = None,
    name_prefix: str = "",
    include_labels: bool = True,
    include_fx: bool = True,
    font_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Build the complete M67 Tactical Grenade + X-Ray Cutaway + FX + 3D Labels
    in local coordinates centered at (0, 0, 0) inside `Grenade_Root`.

    Moving, rotating, or scaling `assembly["root"]` moves the entire grenade,
    all internal mechanisms, all local-space X-ray/burn shaders, FX, and 3D labels.
    """
    if collection is None:
        col_name = f"{name_prefix}Grenade_Assembly"
        col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(col)
    else:
        col = collection

    root_empty = bpy.data.objects.new(f"{name_prefix}Grenade_Root", None)
    root_empty.empty_display_type = "PLAIN_AXES"
    root_empty.empty_display_size = 0.025
    root_empty.location = Vector((0.0, 0.0, 0.0))
    _link_obj(root_empty, col)

    mats, shader_nodes = _build_materials(root_empty, prefix=name_prefix)

    # Outer Cast-Iron Hemispheres:
    # Back = thick cast-iron cutaway wall + crisp diamond V-grooves
    # Front = single-surface solid->X-ray shell (ZERO coplanar Z-fighting on Y=0 cutaway wall!)
    back_shell = _build_hollow_hemisphere(
        f"{name_prefix}M67_Back_Hemisphere",
        phi_start=0.0, phi_end=math.pi,
        n_lat=60, n_lon=60,
        mats_list=[mats["olive_iron"], mats["yellow_band"], mats["cut_steel"], mats["inner_steel"]],
        col=col, parent=root_empty, with_inner_scoring=True, single_surface_xray=False,
    )
    front_shell = _build_hollow_hemisphere(
        f"{name_prefix}M67_Front_Hemisphere_XRay",
        phi_start=math.pi + 0.002, phi_end=2.0 * math.pi - 0.002,
        n_lat=40, n_lon=40,
        mats_list=[
            mats["front_olive_xray"],
            mats["front_yellow_xray"],
            mats["front_olive_xray"],
            mats["front_olive_xray"],
        ],
        col=col, parent=root_empty, with_inner_scoring=False, single_surface_xray=True,
    )

    parts = _build_fuze_and_internals(mats, col, root_empty, prefix=name_prefix)
    parts["back_shell"] = back_shell
    parts["front_shell"] = front_shell
    parts["shards"] = _build_fracture_shards(mats, col, root_empty, prefix=name_prefix)

    fx = _build_optics_fx(mats, col, root_empty, prefix=name_prefix) if include_fx else {}
    labels = (
        _build_3d_labels(mats, col, root_empty, font_path=font_path, prefix=name_prefix)
        if include_labels
        else {}
    )

    return {
        "root": root_empty,
        "collection": col,
        "materials": mats,
        "shader_nodes": shader_nodes,
        "parts": parts,
        "fx": fx,
        "labels": labels,
        "local_bounds": {
            "r_outer": R_OUTER,
            "r_inner": R_INNER,
            "z_min": -R_OUTER,
            "z_max": 0.0790,
            "hinge_pos": HINGE_POS.copy(),
            "primer_pos": PRIMER_POS.copy(),
        },
    }


# Convenience alias
create_grenade = build_grenade_assembly


def animate_grenade(
    assembly: dict[str, Any],
    *,
    pin_pull: tuple[int, int] | int = (15, 96),
    xray_scan: tuple[int, int] | int = (108, 140),
    spoon_fly: tuple[int, int] | int = (244, 322),
    striker_slam: tuple[int, int] | int = (244, 294),
    fuse_burn: tuple[int, int] | int = (356, 499),
    pressure_swell: tuple[int, int] | int = (499, 611),
    fracture_frame: int = 612,
    total_frames: int = 720,
    show_labels: bool = True,
) -> None:
    """
    Keyframe every internal mechanical motion, local-space shader (`ScanZ`, `BurnZ`,
    tension highlights, molten fracture glow), optical FX, visibility switch, and
    3D component label schedule cleanly on `assembly`.
    """
    pin_s, pin_e = _parse_window(pin_pull, 80)
    xray_s, xray_e = _parse_window(xray_scan, 32)
    spoon_s, spoon_e = _parse_window(spoon_fly, 78)
    str_s, str_e = _parse_window(striker_slam, 50)
    burn_s, burn_e = _parse_window(fuse_burn, 143)
    pres_s, pres_e = _parse_window(pressure_swell, 112)
    frac_f = int(fracture_frame)
    T = int(total_frames)

    parts = assembly["parts"]
    fx = assembly.get("fx", {})
    labels = assembly.get("labels", {})
    sn = assembly["shader_nodes"]
    mats = assembly["materials"]

    # Clear prior keyframes on materials & parts so calling animate_grenade() is idempotent
    for m in mats.values():
        _clear_anim(m.node_tree)

    # ------------------------------------------------------------------
    # 1. LOCAL-SPACE X-RAY LASER SCAN (`ScanZ` in Grenade_Root coords)
    # ------------------------------------------------------------------
    for scan_node in sn["scan_z_nodes"]:
        sock = scan_node.outputs[0]
        for f, val in [(1, SCAN_Z_START), (xray_s, SCAN_Z_START), (xray_e, SCAN_Z_END), (T, SCAN_Z_END)]:
            sock.default_value = val
            sock.keyframe_insert("default_value", frame=max(1, min(T, f)))
    for m_key in ("front_olive_xray", "front_yellow_xray", "front_fuze_xray"):
        _set_bezier(mats[m_key].node_tree)

    # ------------------------------------------------------------------
    # 2. LOCAL-SPACE 4-SEC DELAY FUSE BURN (`BurnZ` + `BurnOn`)
    # ------------------------------------------------------------------
    bz_sock = sn["burn_z_node"].outputs[0]
    for f, val in [
        (1, BURN_Z_START),
        (burn_s - 4, BURN_Z_START),
        (burn_s, BURN_Z_START - 0.0020),
        (burn_e, BURN_Z_END),
        (T, BURN_Z_END),
    ]:
        bz_sock.default_value = val
        bz_sock.keyframe_insert("default_value", frame=max(1, min(T, f)))
    _set_linear(mats["fuse_core"].node_tree, "BurnZ")

    bon_sock = sn["burn_on_node"].outputs[0]
    for f, val in [
        (1, 0.0),
        (str_e - 1, 0.0),
        (str_e + 2, 1.30),
        (burn_s, 1.20),
        (burn_e, 1.48),
        (burn_e + 18, 0.32),
        (min(T, burn_e + 52), 0.0),
    ]:
        bon_sock.default_value = val
        bon_sock.keyframe_insert("default_value", frame=max(1, min(T, f)))

    # ------------------------------------------------------------------
    # 3. COMPONENT TENSION & FRACTURE EMISSION HIGHLIGHTS
    # ------------------------------------------------------------------
    # Safety spoon subtle cyan educational highlight during Beat 2 reveal
    sp_em = sn["spoon_em_socket"]
    for f, val in [
        (1, 0.0),
        (xray_e + 12, 0.0),
        (xray_e + 30, 0.28),
        (spoon_s - 18, 0.28),
        (spoon_s, 0.0),
    ]:
        sp_em.default_value = val
        sp_em.keyframe_insert("default_value", frame=max(1, min(T, f)))

    # Torsion spring warm copper-amber tension highlight during Beat 2–3
    spr_em = sn["spring_em_socket"]
    for f, val in [
        (1, 0.0),
        (xray_e - 8, 0.0),
        (xray_e + 10, 0.95),
        (spoon_s - 8, 1.15),
        (spoon_s + 12, 1.55),
        (str_e + 4, 0.20),
        (str_e + 34, 0.0),
    ]:
        spr_em.default_value = val
        spr_em.keyframe_insert("default_value", frame=max(1, min(T, f)))

    # Striker hammer hot highlight during Beat 3 slam
    str_em = sn["striker_em_socket"]
    for f, val in [
        (1, 0.0),
        (str_s - 6, 0.0),
        (str_s + 8, 0.65),
        (str_e, 1.25),
        (str_e + 30, 0.0),
    ]:
        str_em.default_value = val
        str_em.keyframe_insert("default_value", frame=max(1, min(T, f)))

    # Hot fracture side-wall ember vein glow at Beat 6 detonation
    hot_sock = sn["hot_edge_gain_node"].outputs[0]
    for f, val in [
        (1, 0.0),
        (max(1, frac_f - 2), 0.0),
        (min(T, frac_f + 3), 3.8),
        (min(T, frac_f + 44), 1.8),
        (T, 0.9),
    ]:
        hot_sock.default_value = val
        hot_sock.keyframe_insert("default_value", frame=f)

    # ------------------------------------------------------------------
    # 4. MECHANICAL MOTION CHOREOGRAPHY (ALL IN LOCAL Grenade_Root SPACE)
    # ------------------------------------------------------------------
    # Pull Pin extraction to -X
    pin = parts["pull_pin"]
    _clear_anim(pin)
    pin_d = max(1, pin_e - pin_s)
    for f, loc, rot in [
        (1, (0.0, 0.0, HINGE_POS.z), (0.0, 0.0, 0.0)),
        (pin_s, (0.0, 0.0, HINGE_POS.z), (0.0, 0.0, 0.0)),
        (int(pin_s + 0.18 * pin_d), (-0.022, 0.0, HINGE_POS.z + 0.0004), (0.0, 0.04, 0.0)),
        (int(pin_s + 0.36 * pin_d), (-0.044, -0.0025, HINGE_POS.z + 0.0009), (0.06, 0.12, -0.08)),
        (int(pin_s + 0.66 * pin_d), (-0.052, -0.0048, HINGE_POS.z - 0.0012), (-0.08, 0.18, -0.14)),
        (pin_e, (-0.165, -0.010, HINGE_POS.z - 0.0045), (0.15, 0.32, -0.25)),
    ]:
        pin.location = Vector(loc)
        pin.rotation_euler = Euler(rot)
        pin.keyframe_insert("location", frame=max(1, min(T, f)))
        pin.keyframe_insert("rotation_euler", frame=max(1, min(T, f)))
    _set_bezier(pin)

    # Safety Spoon: tiny tension twitch on pin exit, then dramatic spinning launch at spoon_fly!
    spoon = parts["spoon"]
    _clear_anim(spoon)
    sp_d = max(1, spoon_e - spoon_s)
    twitch_f = int(pin_s + 0.20 * pin_d)
    for f, loc, rot in [
        (1, HINGE_POS, (0.0, 0.0, 0.0)),
        (twitch_f, HINGE_POS, (0.0, 0.0, 0.0)),
        (twitch_f + 5, HINGE_POS, (0.0, math.radians(-2.2), 0.0)),
        (twitch_f + 12, HINGE_POS, (0.0, 0.0, 0.0)),
        (spoon_s, HINGE_POS, (0.0, 0.0, 0.0)),
        (int(spoon_s + 0.14 * sp_d), HINGE_POS + Vector((0.015, -0.0025, 0.008)), (0.05, math.radians(-38.0), 0.08)),
        (int(spoon_s + 0.32 * sp_d), HINGE_POS + Vector((0.046, -0.0055, 0.020)), (0.12, math.radians(-115.0), 0.15)),
        (int(spoon_s + 0.62 * sp_d), HINGE_POS + Vector((0.112, -0.0085, 0.030)), (0.22, math.radians(-245.0), 0.28)),
        (spoon_e, HINGE_POS + Vector((0.210, -0.0130, 0.012)), (0.35, math.radians(-390.0), 0.40)),
    ]:
        spoon.location = Vector(loc)
        spoon.rotation_euler = Euler(rot)
        spoon.keyframe_insert("location", frame=max(1, min(T, f)))
        spoon.keyframe_insert("rotation_euler", frame=max(1, min(T, f)))
    _set_bezier(spoon)

    # Torsion Spring unwind
    spring = parts["spring"]
    _clear_anim(spring)
    str_d = max(1, str_e - str_s)
    for f, deg in [
        (1, 0.0),
        (str_s, 0.0),
        (int(str_s + 0.44 * str_d), -75.0),
        (str_e, -185.0),
        (min(T, str_e + 6), -176.0),
        (T, -176.0),
    ]:
        spring.rotation_euler = Euler((0.0, math.radians(deg), 0.0))
        spring.keyframe_insert("rotation_euler", frame=max(1, min(T, f)))
    _set_bezier(spring)

    # Striker Hammer slam dead-center onto PRIMER_POS
    striker = parts["striker"]
    _clear_anim(striker)
    for f, deg in [
        (1, STRIKER_COCKED_DEG),
        (str_s, STRIKER_COCKED_DEG),
        (int(str_s + 0.24 * str_d), -56.0),
        (int(str_s + 0.56 * str_d), -122.0),
        (int(str_s + 0.82 * str_d), -194.0),
        (str_e, STRIKER_IMPACT_DEG),
        (min(T, str_e + 4), STRIKER_IMPACT_DEG + 7.2),
        (min(T, str_e + 10), STRIKER_IMPACT_DEG + 0.3),
        (T, STRIKER_IMPACT_DEG + 0.3),
    ]:
        striker.rotation_euler = Euler((0.0, math.radians(deg), 0.0))
        striker.keyframe_insert("rotation_euler", frame=max(1, min(T, f)))
    _set_bezier(striker)

    # ------------------------------------------------------------------
    # 5. INTACT SHELL VISIBILITY & BEAT 6 FRACTURE SHARDS + FUZE EJECT
    # ------------------------------------------------------------------
    for intact_key in ("back_shell", "front_shell", "fuze_front", "fuze_ear_Front", "core_fill"):
        obj = parts[intact_key]
        _clear_anim(obj)
        _key_visibility(obj, [(1, frac_f - 1)], T)

    for core_key in ("delay_tube", "fuse_core", "booster"):
        obj = parts[core_key]
        _clear_anim(obj)
        _key_visibility(obj, [(1, min(T, frac_f + 4))], T)
    for c_obj in parts["delay_collars"]:
        _clear_anim(c_obj)
        _key_visibility(c_obj, [(1, min(T, frac_f + 4))], T)

    # Fuze assembly ejects gently upward-back-right on detonation
    fuze_eject_keys = [
        "collar", "neck_threads", "flange", "fuze_back",
        "fuze_wall_L", "fuze_wall_R", "fuze_ear_Back", "pin_boss",
        "axle", "spring", "striker", "primer", "primer_dot",
    ]
    for fk in fuze_eject_keys:
        fp = parts[fk]
        base_loc = fp.location.copy()
        fp.keyframe_insert("location", frame=1)
        fp.keyframe_insert("location", frame=frac_f)
        fp.location = base_loc + Vector((0.015, 0.022, 0.020))
        fp.keyframe_insert("location", frame=min(T, frac_f + 20))
        fp.location = base_loc + Vector((0.022, 0.032, 0.027))
        fp.keyframe_insert("location", frame=T)
        _set_bezier(fp)

    # 48 Pre-Scored Shards bullet-time radial fracture
    rng = random.Random(42)
    for sh in parts["shards"]:
        obj = sh["obj"]
        centroid = sh["centroid"]
        n_dir = sh["n_dir"]
        _clear_anim(obj)
        _key_visibility(obj, [(frac_f, T)], T)

        # Slightly damp upward Z so shards frame cleanly below top label!
        burst_vec = Vector((n_dir.x, n_dir.y, n_dir.z * (0.62 if n_dir.z > 0.35 else 0.95)))
        dist_fast = rng.uniform(0.036, 0.060)
        dist_slow = dist_fast + rng.uniform(0.012, 0.020)
        rot_fast = Euler((rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)))
        rot_slow = Euler((rot_fast.x * 1.28, rot_fast.y * 1.28, rot_fast.z * 1.28))

        obj.location = centroid
        obj.rotation_euler = Euler((0.0, 0.0, 0.0))
        obj.keyframe_insert("location", frame=frac_f)
        obj.keyframe_insert("rotation_euler", frame=frac_f)

        obj.location = centroid + burst_vec * dist_fast
        obj.rotation_euler = rot_fast
        obj.keyframe_insert("location", frame=min(T, frac_f + 18))
        obj.keyframe_insert("rotation_euler", frame=min(T, frac_f + 18))

        obj.location = centroid + burst_vec * dist_slow
        obj.rotation_euler = rot_slow
        obj.keyframe_insert("location", frame=T)
        obj.keyframe_insert("rotation_euler", frame=T)
        _set_bezier(obj)

    # ------------------------------------------------------------------
    # 6. OPTICAL FX CHOREOGRAPHY
    # ------------------------------------------------------------------
    if fx:
        # Primer spark burst
        spark_rig = fx["primer_spark_rig"]
        _clear_anim(spark_rig)
        _key_visibility(spark_rig, [(str_e - 1, min(T, str_e + 42))], T)
        for child in spark_rig.children:
            _clear_anim(child)
            _key_visibility(child, [(str_e - 1, min(T, str_e + 42))], T)
        for f, sc in [
            (str_e - 1, 0.02),
            (str_e + 2, 1.38),
            (str_e + 11, 1.02),
            (str_e + 26, 0.42),
            (min(T, str_e + 42), 0.01),
        ]:
            spark_rig.scale = (sc, sc, sc)
            spark_rig.keyframe_insert("scale", frame=max(1, min(T, f)))
        _set_bezier(spark_rig)

        # Internal Burn / Pressure Point Light (calibrated for 0.042m metric scale!)
        bl = fx["burn_light"]
        _clear_anim(bl)
        _clear_anim(bl.data)
        bl.location = Vector((0.0, -0.0072, BURN_Z_START))
        bl.keyframe_insert("location", frame=max(1, str_e - 1))
        bl.keyframe_insert("location", frame=burn_s)
        bl.location = Vector((0.0, -0.0072, BURN_Z_END))
        bl.keyframe_insert("location", frame=burn_e)
        bl.location = Vector((0.0, -0.0025, 0.0))
        bl.keyframe_insert("location", frame=min(T, frac_f + 2))

        for f, e in [
            (1, 0.0),
            (str_e - 2, 0.0),
            (str_e + 2, 2.2),
            (str_e + 20, 0.45),
            (burn_s, 0.95),
            (burn_e, 1.25),
            (pres_s + 8, 4.8),
            (min(T, frac_f + 3), 6.5),
            (min(T, frac_f + 48), 3.0),
            (T, 1.8),
        ]:
            bl.data.energy = e
            bl.data.keyframe_insert("energy", frame=max(1, min(T, f)))

        # Upward travelling fuse embers
        burn_span = max(1, burn_e - burn_s)
        for k, emb in enumerate(fx["fuse_embers"]):
            _clear_anim(emb)
            _key_visibility(emb, [(burn_s, min(T, burn_e + 3))], T)
            phase = k * 9
            step = max(4, burn_span // 24)
            for f in range(burn_s, min(T, burn_e + 5), step):
                t_b = max(0.0, min(1.0, (f - burn_s) / burn_span))
                z_base = _lerp(BURN_Z_START - 0.002, BURN_Z_END, t_b)
                cy = ((f + phase) % 18) / 18.0
                ang = k * 2.1 + f * 0.14
                rad = 0.0015 + 0.0032 * cy
                emb.location = (
                    rad * math.cos(ang),
                    -0.0032 - 0.0020 * abs(math.sin(ang)),
                    z_base + cy * 0.0092,
                )
                emb.keyframe_insert("location", frame=f)

        # 3D Turbulent Fiery Plasma Fireball
        pres_span = max(1, pres_e - pres_s)
        for idx, ps in enumerate(fx["plasma_spheres"]):
            _clear_anim(ps)
            _key_visibility(ps, [(pres_s, T)], T)
            kfs = [
                (pres_s, 0.10, 0.0),
                (int(pres_s + 0.18 * pres_span), 0.46, 0.35),
                (int(pres_s + 0.50 * pres_span), 0.74, 0.85),
                (int(pres_s + 0.82 * pres_span), 0.89, 1.30),
                (pres_e, 0.96, 1.65),
                (min(T, frac_f + 8), 1.36 + 0.12 * idx, 2.05),
                (min(T, frac_f + 44), 1.15 + 0.08 * idx, 2.55),
                (T, 0.94 + 0.06 * idx, 3.15),
            ]
            for f, sc, rz in kfs:
                ps.scale = (sc, sc, sc)
                ps.rotation_euler = Euler((
                    rz * 0.3 * (idx + 1),
                    rz * 0.2,
                    rz * (1.0 if idx % 2 == 0 else -0.8),
                ))
                ps.keyframe_insert("scale", frame=max(1, min(T, f)))
                ps.keyframe_insert("rotation_euler", frame=max(1, min(T, f)))
            _set_bezier(ps)

        # Pulsing Internal Pressure Wave Rings
        for idx, pr in enumerate(fx["pressure_rings"]):
            _clear_anim(pr)
            rs = int(pres_s + (0.12 + 0.30 * idx) * pres_span)
            re = min(frac_f - 1, rs + 34)
            _key_visibility(pr, [(rs, re)], T)
            pr.scale = (0.15, 0.15, 0.15)
            pr.keyframe_insert("scale", frame=rs)
            pr.scale = (1.05, 1.05, 1.05)
            pr.keyframe_insert("scale", frame=re)
            _set_bezier(pr)

        # Beat 6 Radial Shockwave Ring
        shock = fx["shockwave_ring"]
        _clear_anim(shock)
        _key_visibility(shock, [(frac_f, T)], T)
        for f, sc in [
            (frac_f, 0.25),
            (min(T, frac_f + 16), 1.70),
            (min(T, frac_f + 54), 2.08),
            (T, 2.28),
        ]:
            shock.scale = (sc, sc, sc * 0.42)
            shock.keyframe_insert("scale", frame=f)
        _set_bezier(shock)

        # 38 Radial Blast Ember Streaks
        for be in fx["blast_embers"]:
            eb = be["obj"]
            nd = be["nd"]
            _clear_anim(eb)
            _key_visibility(eb, [(frac_f, T)], T)
            burst_n = Vector((nd.x, nd.y, nd.z * (0.60 if nd.z > 0.35 else 0.95)))
            eb.location = burst_n * 0.030
            eb.keyframe_insert("location", frame=frac_f)
            eb.location = burst_n * be["d_fast"]
            eb.keyframe_insert("location", frame=min(T, frac_f + 18))
            eb.location = burst_n * be["d_slow"]
            eb.keyframe_insert("location", frame=T)
            _set_bezier(eb)

    # ------------------------------------------------------------------
    # 7. 3D COMPONENT POINTER LABELS & LIVE COUNTDOWN SCHEDULE
    # ------------------------------------------------------------------
    if labels:
        def set_label_vis(entry: dict[str, Any], ranges: list[tuple[int, int]], ptr_ranges: list[tuple[int, int]] | None = None) -> None:
            r_text = ranges if show_labels else []
            r_ptr = (ptr_ranges if ptr_ranges is not None else ranges) if show_labels else []
            for obj in (entry["text_obj"], entry["keyline_obj"]):
                _clear_anim(obj)
                _key_visibility(obj, r_text, T)
            for pobj in entry["pointer_objs"]:
                _clear_anim(pobj)
                _key_visibility(pobj, r_ptr, T)

        set_label_vis(labels["safety_pin"], [(1, max(2, xray_s - 2))])
        set_label_vis(labels["safety_lever"], [(xray_e - 6, min(T, spoon_s + 8))])
        set_label_vis(labels["torsion_spring"], [(xray_e, max(xray_e + 1, spoon_s - 2))])
        set_label_vis(labels["striker_hammer"], [(spoon_s, min(T, burn_s - 4))])
        set_label_vis(labels["brass_primer"], [(max(spoon_s, str_e - 18), min(T, burn_s - 4))])
        set_label_vis(labels["delay_fuse"], [(burn_s, burn_e)])
        set_label_vis(
            labels["pre_scored_shell"],
            ranges=[(pres_s + 4, T)],
            ptr_ranges=[(pres_s + 4, max(pres_s + 5, frac_f - 1))],
        )

        # Gently raise PRE-SCORED SHELL slightly at fracture so exploding shards stay well clear
        ps_anchor = labels["pre_scored_shell"]["text_anchor"]
        _clear_anim(ps_anchor)
        ps_anchor.location = Vector((0.0, -0.004, 0.094))
        ps_anchor.keyframe_insert("location", frame=max(1, frac_f - 4))
        ps_anchor.location = Vector((0.0, -0.004, 0.112))
        ps_anchor.keyframe_insert("location", frame=min(T, frac_f + 12))
        _set_bezier(ps_anchor)

        # Live 4.0 s -> 0.0 s 3D countdown across fuse_burn window
        steps = ["4.0", "3.0", "2.0", "1.0", "0.0"]
        span = max(5, burn_e - burn_s)
        for idx, step_key in enumerate(steps):
            s_f = int(burn_s + idx * span / len(steps))
            e_f = int(burn_s + (idx + 1) * span / len(steps)) - 1 if idx < len(steps) - 1 else burn_e
            set_label_vis(labels["countdown"][step_key], [(s_f, max(s_f, e_f))])


# ==============================================================================
# STANDALONE STUDIO VISUAL VERIFICATION TEST
# ==============================================================================
def _run_standalone_test() -> None:
    out_dir = Path("/tmp/grenade_v3_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_extra = "--extra" in sys.argv

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 35
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = 720
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 94

    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.10

    cycles = scene.cycles
    cycles.device = "CPU"
    cycles.samples = 6
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.04
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 3
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 8
    cycles.transparent_max_bounces = 16
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False

    # Dramatic dark studio backdrop
    world = bpy.data.worlds.new("Grenade_Test_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.012, 0.018, 0.030, 1.0)
    bg.inputs["Strength"].default_value = 0.45
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    col_studio = bpy.data.collections.new("Test_Studio")
    scene.collection.children.link(col_studio)

    def add_light(name: str, loc: tuple[float, float, float], target: tuple[float, float, float],
                  energy: float, color: tuple[float, float, float], size: float, size_y: float | None = None) -> None:
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.color = color
        if size_y is not None:
            ld.shape = "RECTANGLE"
            ld.size = size
            ld.size_y = size_y
        else:
            ld.shape = "DISK"
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = Vector(loc)
        lo.rotation_euler = (Vector(target) - lo.location).to_track_quat("-Z", "Y").to_euler()
        col_studio.objects.link(lo)

    # Studio lighting scaled cleanly for a 0.084m handheld grenade!
    # Also test non-origin placement to prove 100% local-space shaders work anywhere in world!
    test_offset = Vector((1.25, -0.40, 0.85))
    add_light("Key_Softbox", tuple(test_offset + Vector((-0.24, -0.26, 0.22))), tuple(test_offset + Vector((0, 0, 0.015))), 4.5, (1.0, 0.95, 0.88), 0.20, 0.28)
    add_light("Rim_CoolBlue", tuple(test_offset + Vector((0.28, 0.14, 0.16))), tuple(test_offset + Vector((0, 0, 0.015))), 5.8, (0.45, 0.82, 1.0), 0.18, 0.28)
    add_light("Rim_WarmAmber", tuple(test_offset + Vector((-0.26, 0.16, 0.10))), tuple(test_offset + Vector((0, 0, 0.020))), 3.6, (1.0, 0.62, 0.24), 0.16, 0.24)
    add_light("Fill_Front", tuple(test_offset + Vector((0.08, -0.32, 0.04))), tuple(test_offset + Vector((0, 0, 0.020))), 1.4, (0.85, 0.92, 1.0), 0.22)
    add_light("Top_Strip", tuple(test_offset + Vector((0.0, -0.04, 0.34))), tuple(test_offset + Vector((0, 0, 0.040))), 2.2, (0.95, 0.97, 1.0), 0.22, 0.07)

    # Build and animate grenade, then move Grenade_Root to test_offset to prove local-space shader invariance!
    assembly = build_grenade_assembly()
    assembly["root"].location = test_offset
    animate_grenade(
        assembly,
        pin_pull=(15, 96),
        xray_scan=(108, 140),
        spoon_fly=(244, 322),
        striker_slam=(244, 294),
        fuse_burn=(356, 499),
        pressure_swell=(499, 611),
        fracture_frame=612,
        total_frames=720,
    )

    # Close-up portrait camera
    cam_d = bpy.data.cameras.new("Test_Camera")
    cam_d.lens = 50.0
    cam_d.sensor_width = 36.0
    cam_d.clip_start = 0.01
    cam_d.clip_end = 50.0
    cam_o = bpy.data.objects.new("Test_Camera", cam_d)
    col_studio.objects.link(cam_o)
    scene.camera = cam_o

    target_empty = bpy.data.objects.new("Test_Cam_Target", None)
    target_empty.location = test_offset + Vector((0.0, 0.0, 0.020))
    col_studio.objects.link(target_empty)
    tr = cam_o.constraints.new(type="TRACK_TO")
    tr.target = target_empty
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"

    renders = [
        ("solid.jpg", 26, Vector((-0.032, -0.345, 0.036)), Vector((-0.004, 0.0, 0.020))),
        ("xray_with_labels.jpg", 185, Vector((0.036, -0.345, 0.034)), Vector((0.004, 0.0, 0.020))),
    ]
    if render_extra:
        renders.extend([
            ("striker_primer.jpg", 295, Vector((0.015, -0.305, 0.048)), Vector((0.004, 0.0, 0.042))),
            ("burn_countdown.jpg", 420, Vector((0.018, -0.340, 0.030)), Vector((0.0, 0.0, 0.020))),
            ("fracture_shards.jpg", 648, Vector((0.075, -0.430, 0.042)), Vector((0.0, 0.0, 0.016))),
        ])

    for fname, frame, cam_rel, tgt_rel in renders:
        scene.frame_set(frame)
        cam_o.location = test_offset + cam_rel
        target_empty.location = test_offset + tgt_rel
        out_path = str(out_dir / fname)
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[GrenadeAsset Test] Rendered {out_path} (frame {frame})", flush=True)


if __name__ == "__main__":
    _run_standalone_test()
