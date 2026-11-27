"""
parachute_asset.py — Parachutes Asset Module for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames).

Blender 4.2 Cycles | Real-World Metric Coordinates | Prefix: `Chute_`

Builds and animates both recovery parachute systems directly from `ejection_contract.py`:
1. Gyro-Stabilized Drogue Parachute (`Chute_Drogue_*`):
   - Visible frames EV["drogue_fire"] (1400) .. EV["seat_hidden"] - 1 (1739);
     CONSTANT-hidden outside [1400, 1739].
   - 16-gore domed stabilization canopy (1.5 m diameter) with open central apex vent,
     alternating international-orange & off-white ripstop nylon gores, raised radial
     seam tapes, brass/steel mortar extractor slug + apex lanyard, confluence swivel,
     8 braided suspension lines, and heavy nylon bridle strap anchored with 0.000 mm
     drift to the seat's evaluated world-space `SEAT["drogue_mortar"]` port.

2. Main Ring-Sail Parachute (`Chute_Main_*`):
   - Visible frames EV["separation"] (1545) .. total_frames (1800);
     CONSTANT-hidden before frame 1545.
   - Classic USAF / Martin-Baker 24-gore, 3-tier vented Ring-Sail canopy (7.6 m diameter)
     with louvered annular exhaust vent gaps, continuous radial load-tape ribs, open
     apex vent, alternating international-orange, off-white, and desert-sand panels,
     local-space procedural ripstop weave bump, and warm golden-hour translucent backlighting.
   - Two-stage inflation driven by `C.main_chute_open_fraction(f)`:
     * u in [0.00, 0.25] (1545..1580): deployment bag + line stretch to 6.4 m.
     * u in [0.25, 1.00] (1580..1650): dramatic radial blossom (X/Y 0.16 -> 1.00)
       with subtle pneumatic dome breathing oscillation through 1800.
   - Left & Right Kevlar shoulder riser straps (`Chute_Main_Riser_L/R`) from
     `C.RISER_ATTACH_L/R` on `pilot_carrier` up to `Chute_Main_Swivel`, fanning into
     16 braided suspension lines attached cleanly to 16 reinforced canopy skirt tabs.

Public API:
    build_parachutes(
        col: bpy.types.Collection,
        seat_root: bpy.types.Object | None,
        pilot_carrier: bpy.types.Object | None,
        total_frames: int = 1800,
    ) -> dict[str, Any]

Returned dict keys & exact types:
    - "drogue_canopy":     bpy.types.Object (`Chute_Drogue_Canopy`)
    - "drogue_lines":      list[bpy.types.Object] (8 suspension lines)
    - "drogue_bridle":     bpy.types.Object (`Chute_Drogue_Bridle`)
    - "drogue_confluence": bpy.types.Object (`Chute_Drogue_Confluence`)
    - "drogue_extractor":  bpy.types.Object (`Chute_Drogue_ExtractorSlug`)
    - "main_canopy":       bpy.types.Object (`Chute_Main_Canopy`)
    - "main_lines":        list[bpy.types.Object] (16 suspension lines)
    - "main_risers":       list[bpy.types.Object] (`Chute_Main_Riser_L`, `Chute_Main_Riser_R`)
    - "main_swivel":       bpy.types.Object (`Chute_Main_Swivel`)
    - "main_deploy_bag":   bpy.types.Object (`Chute_Main_DeployBag`)
    - "all_objs":          list[bpy.types.Object] (all `Chute_*` objects created)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402

# ---------------------------------------------------------------------------
# PARACHUTE DESIGN CONSTANTS (derived from ejection_contract)
# ---------------------------------------------------------------------------
DROGUE_GORES = 16
DROGUE_LINE_COUNT = 8
DROGUE_RADIUS = C.DROGUE_DIAMETER_M * 0.5            # 0.75 m
DROGUE_HEIGHT = 0.58                                 # m (domed profile)
DROGUE_VENT_RADIUS = 0.17                            # m (stabilization apex vent)
DROGUE_CONFLUENCE_OFFSET = 0.95                      # m below skirt at full deploy
DROGUE_TRAIL_DIR = Vector((0.10, -1.6, 4.2)).normalized()

MAIN_GORES = 24
MAIN_LINE_COUNT = 16
MAIN_RADIUS = C.MAIN_CHUTE_DIAMETER_M * 0.5          # 3.80 m
MAIN_HEIGHT = 2.66                                   # m (billowed ring-sail dome)
MAIN_SWIVEL_MAX_LEN = 1.25                           # m above shoulder riser midpoint
MAIN_BAG_HIDE_FRAME = 1605                           # deployment bag cleanly exits & hides


# ---------------------------------------------------------------------------
# BLENDER SCENE, MESH & KEYFRAME HELPERS
# ---------------------------------------------------------------------------
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None) -> bpy.types.Object:
    target = col or bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_bsdf(bsdf: bpy.types.Node, name: str, val: Any) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _finalize_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material] | bpy.types.Material,
    col: bpy.types.Collection | None,
    *,
    recalc_normals: bool = True,
    smooth: bool = True,
) -> bpy.types.Object:
    """Attach materials BEFORE bm.to_mesh(mesh), finalize normals, and link object."""
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for mat in mat_list:
        mesh.materials.append(mat)

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    else:
        bm.normal_update()

    bm.to_mesh(mesh)
    bm.free()

    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.rotation_mode = "QUATERNION"
    _link_obj(obj, col)
    return obj


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe hide_viewport & hide_render with strict CONSTANT interpolation."""
    frames = {1, total_frames}
    clamped_ranges: list[tuple[int, int]] = []
    for s, e in visible_ranges:
        s_c = max(1, min(total_frames, int(s)))
        e_c = max(1, min(total_frames, int(e)))
        if s_c <= e_c:
            clamped_ranges.append((s_c, e_c))
            if s_c > 1:
                frames.add(s_c - 1)
            frames.add(s_c)
            frames.add(e_c)
            if e_c < total_frames:
                frames.add(e_c + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in clamped_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)

    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path.startswith("hide_"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _set_linear_transform_fcurves(obj: bpy.types.Object) -> None:
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if fc.data_path in ("location", "rotation_quaternion", "rotation_euler", "scale"):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def _basis_quat(z_dir: Vector, right_hint: Vector, prev_q: Quaternion | None = None) -> Quaternion:
    """Build a gimbal-free quaternion whose local +Z aligns with `z_dir` and +X near `right_hint`."""
    z_ax = z_dir.normalized() if z_dir.length > 1e-6 else Vector((0.0, 0.0, 1.0))
    proj = right_hint - z_ax * right_hint.dot(z_ax)
    if proj.length < 1e-5:
        fallback = Vector((0.0, 1.0, 0.0)) if abs(z_ax.x) > 0.85 else Vector((1.0, 0.0, 0.0))
        proj = fallback - z_ax * fallback.dot(z_ax)
    x_ax = proj.normalized()
    y_ax = z_ax.cross(x_ax).normalized()
    rot_m = Matrix((x_ax, y_ax, z_ax)).transposed()
    q = rot_m.to_quaternion()
    if prev_q is not None and prev_q.dot(q) < 0.0:
        q = -q
    return q


def _keyframe_segment_endpoints(
    obj: bpy.types.Object,
    frame: int,
    p_start: Vector,
    p_end: Vector,
    right_hint: Vector,
    prev_q: Quaternion | None,
    min_len: float = 1e-3,
) -> Quaternion:
    """Transform a unit-+Z segment mesh so local z=0 sits at `p_start` and z=1 at `p_end`."""
    delta = p_end - p_start
    dist = delta.length
    z_dir = delta / dist if dist > 1e-6 else Vector((0.0, 0.0, 1.0))
    q = _basis_quat(z_dir, right_hint, prev_q)

    obj.location = p_start
    obj.rotation_quaternion = q
    obj.scale = (1.0, 1.0, max(dist, min_len))
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)
    obj.keyframe_insert("scale", frame=frame)
    return q


# ---------------------------------------------------------------------------
# PROCEDURAL MATERIALS (LOCAL-SPACE RIPSTOP NYLON, KEVLAR, METAL)
# ---------------------------------------------------------------------------
def _make_ripstop_material(
    name: str,
    base_rgb: tuple[float, float, float],
    *,
    weave_scale: float = 14.0,
    translucent_fac: float = 0.22,
    roughness: float = 0.44,
) -> bpy.types.Material:
    """Local-space procedural ripstop nylon with crosshatch weave bump and sun backlighting."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    trans = nt.nodes.new("ShaderNodeBsdfTranslucent")

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    wave_x = nt.nodes.new("ShaderNodeTexWave")
    wave_y = nt.nodes.new("ShaderNodeTexWave")
    math_max = nt.nodes.new("ShaderNodeMath")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    rough_ramp = nt.nodes.new("ShaderNodeValToRGB")
    bump = nt.nodes.new("ShaderNodeBump")

    # Orthogonal ripstop grid in object-local coordinates
    for wv, axis in ((wave_x, "X"), (wave_y, "Y")):
        wv.wave_type = "BANDS"
        wv.bands_direction = axis
        wv.wave_profile = "SIN"
        wv.inputs["Scale"].default_value = weave_scale
        wv.inputs["Distortion"].default_value = 0.15
        wv.inputs["Detail"].default_value = 1.0
        nt.links.new(tex_coord.outputs["Object"], wv.inputs["Vector"])

    math_max.operation = "MAXIMUM"
    nt.links.new(wave_x.outputs["Fac"], math_max.inputs[0])
    nt.links.new(wave_y.outputs["Fac"], math_max.inputs[1])

    bump.inputs["Strength"].default_value = 0.085
    bump.inputs["Distance"].default_value = 0.0025
    nt.links.new(math_max.outputs["Value"], bump.inputs["Height"])

    # Subtle fabric grain in roughness
    noise.inputs["Scale"].default_value = weave_scale * 2.2
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    rough_ramp.color_ramp.elements[0].position = 0.25
    rough_ramp.color_ramp.elements[0].color = (max(0.1, roughness - 0.08),) * 3 + (1.0,)
    rough_ramp.color_ramp.elements[1].position = 0.78
    rough_ramp.color_ramp.elements[1].color = (min(0.9, roughness + 0.10),) * 3 + (1.0,)
    nt.links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])

    rgba = (*base_rgb, 1.0)
    _set_bsdf(bsdf, "Base Color", rgba)
    _set_bsdf(bsdf, "Metallic", 0.0)
    _set_bsdf(bsdf, "Specular IOR Level", 0.36)
    _set_bsdf(bsdf, "Sheen Weight", 0.35)
    _set_bsdf(bsdf, "Sheen Roughness", 0.45)
    _set_bsdf(bsdf, "Sheen Tint", (0.95, 0.90, 0.80, 1.0))
    nt.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Warm golden-hour sunlight transmission through thin nylon fabric
    warm_trans = (
        min(1.0, base_rgb[0] * 1.05 + 0.05),
        min(1.0, base_rgb[1] * 0.96 + 0.03),
        min(1.0, base_rgb[2] * 0.82),
        1.0,
    )
    trans.inputs["Color"].default_value = warm_trans
    nt.links.new(bump.outputs["Normal"], trans.inputs["Normal"])

    mix.inputs[0].default_value = translucent_fac
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(trans.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_webbing_material(
    name: str,
    base_rgb: tuple[float, float, float],
    *,
    roughness: float = 0.62,
    weave_scale: float = 35.0,
) -> bpy.types.Material:
    """Heavy opaque braided/woven Kevlar or nylon webbing material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    bump = nt.nodes.new("ShaderNodeBump")

    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = weave_scale
    wave.inputs["Distortion"].default_value = 0.2
    nt.links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])

    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Roughness", roughness)
    _set_bsdf(bsdf, "Specular IOR Level", 0.30)
    _set_bsdf(bsdf, "Sheen Weight", 0.20)
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_metal_material(
    name: str,
    base_rgb: tuple[float, float, float],
    *,
    roughness: float = 0.25,
    metallic: float = 0.92,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    ramp = nt.nodes.new("ShaderNodeValToRGB")

    noise.inputs["Scale"].default_value = 28.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color = (max(0.05, roughness - 0.07),) * 3 + (1.0,)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (min(0.85, roughness + 0.08),) * 3 + (1.0,)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_materials() -> dict[str, bpy.types.Material]:
    return {
        "orange": _make_ripstop_material(
            "Chute_Mat_Orange", (0.86, 0.17, 0.018), weave_scale=11.0, translucent_fac=0.22
        ),
        "white": _make_ripstop_material(
            "Chute_Mat_White", (0.83, 0.81, 0.76), weave_scale=11.0, translucent_fac=0.20
        ),
        "sand": _make_ripstop_material(
            "Chute_Mat_Sand", (0.63, 0.49, 0.30), weave_scale=11.0, translucent_fac=0.20
        ),
        "webbing": _make_webbing_material(
            "Chute_Mat_Webbing", (0.065, 0.070, 0.075), roughness=0.66, weave_scale=30.0
        ),
        "kevlar_riser": _make_webbing_material(
            "Chute_Mat_KevlarRiser", (0.16, 0.165, 0.12), roughness=0.58, weave_scale=40.0
        ),
        "cord": _make_webbing_material(
            "Chute_Mat_SuspensionCord", (0.68, 0.67, 0.63), roughness=0.48, weave_scale=45.0
        ),
        "bag": _make_webbing_material(
            "Chute_Mat_DeployBag", (0.18, 0.19, 0.14), roughness=0.68, weave_scale=24.0
        ),
        "steel": _make_metal_material(
            "Chute_Mat_Steel", (0.58, 0.60, 0.64), roughness=0.24, metallic=0.92
        ),
        "brass": _make_metal_material(
            "Chute_Mat_Brass", (0.78, 0.56, 0.17), roughness=0.22, metallic=0.94
        ),
    }


# ---------------------------------------------------------------------------
# SHARED GEOMETRY PRIMITIVES
# ---------------------------------------------------------------------------
def _build_unit_line_mesh_obj(
    name: str,
    radius: float,
    sides: int,
    mat: bpy.types.Material,
    col: bpy.types.Collection | None,
) -> bpy.types.Object:
    """Create a unit-length cylinder along local +Z from z=0.0 to z=1.0."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=sides,
        radius1=radius,
        radius2=radius,
        depth=1.0,
    )
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.5), verts=bm.verts)
    for f in bm.faces:
        f.material_index = 0
    return _finalize_mesh_obj(name, bm, [mat], col, recalc_normals=True, smooth=True)


def _build_unit_strap_mesh_obj(
    name: str,
    width: float,
    thickness: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection | None,
) -> bpy.types.Object:
    """Create a beveled flat webbing strap along local +Z from z=0.0 to z=1.0."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(width, thickness, 1.0), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.5), verts=bm.verts)
    long_edges = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 0.5]
    if long_edges:
        bmesh.ops.bevel(bm, geom=long_edges, offset=thickness * 0.28, segments=2, affect="EDGES")
    for f in bm.faces:
        f.material_index = 0
    return _finalize_mesh_obj(name, bm, [mat], col, recalc_normals=True, smooth=True)


def _add_thin_band_ring(
    bm: bmesh.types.BMesh,
    n_cols: int,
    r_inner: float,
    r_outer: float,
    z_bot: float,
    z_top: float,
    mat_idx: int,
) -> None:
    """Add a closed rectangular-section hoop/collar ring to `bm`."""
    ring_verts: list[list[bmesh.types.BMVert]] = []
    profile = [(r_outer, z_bot), (r_outer, z_top), (r_inner, z_top), (r_inner, z_bot)]
    for rp, zp in profile:
        row = []
        for c in range(n_cols):
            th = (2.0 * math.pi * c) / float(n_cols)
            row.append(bm.verts.new((rp * math.cos(th), rp * math.sin(th), zp)))
        ring_verts.append(row)
    for c in range(n_cols):
        cn = (c + 1) % n_cols
        for p in range(4):
            pn = (p + 1) % 4
            face = bm.faces.new(
                (ring_verts[p][c], ring_verts[p][cn], ring_verts[pn][cn], ring_verts[pn][c])
            )
            face.material_index = mat_idx


def _add_skirt_tab(
    bm: bmesh.types.BMesh,
    radius: float,
    theta: float,
    tab_width: float,
    tab_height: float,
    tab_thick: float,
    mat_idx: int,
) -> None:
    """Add a small reinforced webbing attachment tab on the canopy skirt at `theta`."""
    res = bmesh.ops.create_cube(bm, size=1.0)
    verts = res["verts"]
    bmesh.ops.scale(bm, vec=(tab_thick, tab_width, tab_height), verts=verts)
    rot_m = Matrix.Rotation(theta, 4, "Z")
    pos = Vector((radius * math.cos(theta), radius * math.sin(theta), -tab_height * 0.25))
    xform = Matrix.Translation(pos) @ rot_m
    bmesh.ops.transform(bm, matrix=xform, verts=verts)
    for v in verts:
        for f in v.link_faces:
            f.material_index = mat_idx


# ---------------------------------------------------------------------------
# 1. DROGUE PARACHUTE BUILDER (16 GORES, 1.5 m DIAMETER, 8 LINES + BRIDLE)
# ---------------------------------------------------------------------------
def _build_drogue_canopy_mesh(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Build 16-gore domed drogue canopy (diameter=1.5m) with open apex vent & seam ribs."""
    bm = bmesh.new()
    sub_u = 4
    n_cols = DROGUE_GORES * sub_u
    n_rings = 9

    # Build dome shell rings with lobe billow inside each gore
    grid: list[list[bmesh.types.BMVert]] = []
    for ri in range(n_rings + 1):
        v = ri / float(n_rings)
        phi = v * math.radians(74.0)
        r_base = C.lerp(DROGUE_RADIUS, DROGUE_VENT_RADIUS, math.sin(phi) / math.sin(math.radians(74.0)))
        z_base = DROGUE_HEIGHT * (math.sin(phi * 1.06) / math.sin(math.radians(74.0) * 1.06))

        row: list[bmesh.types.BMVert] = []
        for ci in range(n_cols):
            theta = (2.0 * math.pi * ci) / float(n_cols)
            gore_u = (ci % sub_u) / float(sub_u)
            lobe = 0.028 * math.sin(math.pi * gore_u) * (0.35 + 0.65 * math.sin(math.pi * v))
            r = r_base + lobe
            z = z_base + 0.012 * math.sin(math.pi * gore_u) * math.sin(math.pi * v)
            row.append(bm.verts.new((r * math.cos(theta), r * math.sin(theta), z)))
        grid.append(row)

    # Material slots: 0=orange, 1=white, 2=webbing
    for ri in range(n_rings):
        for ci in range(n_cols):
            cn = (ci + 1) % n_cols
            gore_idx = ci // sub_u
            mat_idx = 0 if (gore_idx % 2 == 0) else 1
            f = bm.faces.new((grid[ri][ci], grid[ri][cn], grid[ri + 1][cn], grid[ri + 1][ci]))
            f.material_index = mat_idx

    # Skirt hem collar & open apex vent collar + 16 radial seam ribs + 8 skirt tabs
    _add_thin_band_ring(bm, n_cols, DROGUE_RADIUS - 0.008, DROGUE_RADIUS + 0.010, -0.015, 0.018, 2)
    _add_thin_band_ring(
        bm, n_cols, DROGUE_VENT_RADIUS - 0.008, DROGUE_VENT_RADIUS + 0.010, DROGUE_HEIGHT - 0.014, DROGUE_HEIGHT + 0.012, 2
    )

    # 16 raised solid seam tapes along gore edges
    for g in range(DROGUE_GORES):
        ci = g * sub_u
        th = (2.0 * math.pi * ci) / float(n_cols)
        nx, ny = math.cos(th), math.sin(th)
        tx, ty = -ny, nx
        rib_half_w = 0.008
        rib_thick = 0.005
        rib_rings: list[list[bmesh.types.BMVert]] = []
        for ri in range(n_rings + 1):
            co = grid[ri][ci].co
            p_in_l = Vector((co.x - tx * rib_half_w - nx * 0.001, co.y - ty * rib_half_w - ny * 0.001, co.z - 0.001))
            p_in_r = Vector((co.x + tx * rib_half_w - nx * 0.001, co.y + ty * rib_half_w - ny * 0.001, co.z - 0.001))
            p_out_r = Vector((co.x + tx * rib_half_w + nx * rib_thick, co.y + ty * rib_half_w + ny * rib_thick, co.z + rib_thick))
            p_out_l = Vector((co.x - tx * rib_half_w + nx * rib_thick, co.y - ty * rib_half_w + ny * rib_thick, co.z + rib_thick))
            rib_rings.append([bm.verts.new(p) for p in (p_in_r, p_in_l, p_out_l, p_out_r)])
        for ri in range(n_rings):
            for s in range(4):
                sn = (s + 1) % 4
                f = bm.faces.new((rib_rings[ri][s], rib_rings[ri][sn], rib_rings[ri + 1][sn], rib_rings[ri + 1][s]))
                f.material_index = 2

    # 2 crossed apex vent bridle tapes meeting at vent center for the extractor lanyard
    for ang in (0.0, math.pi * 0.5):
        res = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(DROGUE_VENT_RADIUS * 2.02, 0.016, 0.006), verts=res["verts"])
        bmesh.ops.rotate(bm, matrix=Matrix.Rotation(ang, 3, "Z"), verts=res["verts"])
        bmesh.ops.translate(bm, vec=(0.0, 0.0, DROGUE_HEIGHT), verts=res["verts"])
        for v in res["verts"]:
            for f in v.link_faces:
                f.material_index = 2

    # 8 attachment tabs at every 2nd gore seam
    for k in range(DROGUE_LINE_COUNT):
        th = (2.0 * math.pi * k) / float(DROGUE_LINE_COUNT)
        _add_skirt_tab(bm, DROGUE_RADIUS, th, tab_width=0.024, tab_height=0.036, tab_thick=0.010, mat_idx=2)

    return _finalize_mesh_obj(
        "Chute_Drogue_Canopy",
        bm,
        [mats["orange"], mats["white"], mats["webbing"]],
        col,
        recalc_normals=False,
        smooth=True,
    )


def _build_drogue_extractor_slug(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Machined brass/steel mortar extractor slug ahead of drogue apex vent."""
    bm = bmesh.new()
    # Brass cylindrical slug body
    res_body = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=20, radius1=0.042, radius2=0.038, depth=0.13
    )
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.065), verts=res_body["verts"])
    for v in res_body["verts"]:
        for f in v.link_faces:
            f.material_index = 0

    # Steel tail clevis collar
    res_collar = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=16, radius1=0.022, radius2=0.030, depth=0.032
    )
    bmesh.ops.translate(bm, vec=(0.0, 0.0, -0.012), verts=res_collar["verts"])
    for v in res_collar["verts"]:
        for f in v.link_faces:
            f.material_index = 1

    return _finalize_mesh_obj(
        "Chute_Drogue_ExtractorSlug",
        bm,
        [mats["brass"], mats["steel"]],
        col,
        recalc_normals=True,
        smooth=True,
    )


def _build_drogue_confluence(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Forged steel confluence swivel connecting bridle to 8 drogue lines."""
    bm = bmesh.new()
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=16, radius1=0.022, radius2=0.028, depth=0.06
    )
    for v in res["verts"]:
        for f in v.link_faces:
            f.material_index = 0
    return _finalize_mesh_obj(
        "Chute_Drogue_Confluence", bm, [mats["steel"]], col, recalc_normals=True, smooth=True
    )


# ---------------------------------------------------------------------------
# 2. MAIN RING-SAIL PARACHUTE BUILDER (24 GORES, 3 RING-SAIL TIERS, 7.6 m)
# ---------------------------------------------------------------------------
def _build_main_ringsail_canopy_mesh(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Build classic USAF/Martin-Baker 24-gore 3-tier vented Ring-Sail canopy (7.6m)."""
    bm = bmesh.new()
    sub_u = 4
    n_cols = MAIN_GORES * sub_u
    phi_max = math.radians(78.0)

    # 3 concentric ring-sail tiers with louvered annular vent steps between tiers
    tiers = [
        # (phi_start_deg, phi_end_deg, radial_step_bot, radial_step_top, v_segs)
        (0.0, 26.0, 0.00, -0.050, 5),
        (28.5, 53.0, +0.024, -0.045, 4),
        (55.5, 78.0, +0.024, 0.00, 4),
    ]

    # Material slots: 0=orange, 1=white, 2=sand, 3=webbing
    for tier_idx, (p0_d, p1_d, r_off0, r_off1, v_segs) in enumerate(tiers):
        p0 = math.radians(p0_d)
        p1 = math.radians(p1_d)
        grid: list[list[bmesh.types.BMVert]] = []

        for ri in range(v_segs + 1):
            tv = ri / float(v_segs)
            phi = C.lerp(p0, p1, tv)
            r_profile = MAIN_RADIUS * math.cos(phi) + C.lerp(r_off0, r_off1, tv)
            z_profile = MAIN_HEIGHT * (math.sin(phi) / math.sin(phi_max))

            row: list[bmesh.types.BMVert] = []
            for ci in range(n_cols):
                theta = (2.0 * math.pi * ci) / float(n_cols)
                gore_u = (ci % sub_u) / float(sub_u)
                # Pneumatic lobe puff inside each gore panel & sail tier
                billow = 0.095 * math.sin(math.pi * gore_u) * (0.25 + 0.75 * math.sin(math.pi * tv))
                r = r_profile + billow * math.cos(phi * 0.6)
                z = z_profile + 0.045 * math.sin(math.pi * gore_u) * math.sin(math.pi * tv)
                row.append(bm.verts.new((r * math.cos(theta), r * math.sin(theta), z)))
            grid.append(row)

        for ri in range(v_segs):
            for ci in range(n_cols):
                cn = (ci + 1) % n_cols
                gore_idx = ci // sub_u
                # Iconic 2-gore wide international-orange / off-white / desert-sand ring-sail pattern
                sector = (gore_idx // 2 + (1 if tier_idx == 1 else 0)) % 3
                f = bm.faces.new((grid[ri][ci], grid[ri][cn], grid[ri + 1][cn], grid[ri + 1][ci]))
                f.material_index = sector

    # 24 continuous solid radial load-tape ribs spanning across all 3 tiers & vent slots
    rib_segs = 14
    rib_half_w = 0.016
    for g in range(MAIN_GORES):
        th = (2.0 * math.pi * g) / float(MAIN_GORES)
        nx, ny = math.cos(th), math.sin(th)
        tx, ty = -ny, nx
        rib_rings: list[list[bmesh.types.BMVert]] = []
        for ri in range(rib_segs + 1):
            phi = (ri / float(rib_segs)) * phi_max
            r_in = MAIN_RADIUS * math.cos(phi) + 0.025
            r_out = r_in + 0.010
            z_in = MAIN_HEIGHT * (math.sin(phi) / math.sin(phi_max)) + 0.005
            z_out = z_in + 0.008
            p_in_r = Vector((r_in * nx + tx * rib_half_w, r_in * ny + ty * rib_half_w, z_in))
            p_in_l = Vector((r_in * nx - tx * rib_half_w, r_in * ny - ty * rib_half_w, z_in))
            p_out_l = Vector((r_out * nx - tx * rib_half_w, r_out * ny - ty * rib_half_w, z_out))
            p_out_r = Vector((r_out * nx + tx * rib_half_w, r_out * ny + ty * rib_half_w, z_out))
            rib_rings.append([bm.verts.new(p) for p in (p_in_r, p_in_l, p_out_l, p_out_r)])
        for ri in range(rib_segs):
            for s in range(4):
                sn = (s + 1) % 4
                f = bm.faces.new((rib_rings[ri][s], rib_rings[ri][sn], rib_rings[ri + 1][sn], rib_rings[ri + 1][s]))
                f.material_index = 3

    # Reinforced skirt hem band & open apex vent collar
    _add_thin_band_ring(bm, n_cols, MAIN_RADIUS - 0.022, MAIN_RADIUS + 0.025, -0.032, 0.035, 3)
    r_apex = MAIN_RADIUS * math.cos(phi_max)
    _add_thin_band_ring(bm, n_cols, r_apex - 0.025, r_apex + 0.025, MAIN_HEIGHT - 0.025, MAIN_HEIGHT + 0.025, 3)

    # 16 reinforced skirt attachment tabs matching the 16 suspension line anchors
    for k in range(MAIN_LINE_COUNT):
        th = (k + 0.5) * (2.0 * math.pi / float(MAIN_LINE_COUNT))
        _add_skirt_tab(bm, MAIN_RADIUS, th, tab_width=0.055, tab_height=0.075, tab_thick=0.024, mat_idx=3)

    return _finalize_mesh_obj(
        "Chute_Main_Canopy",
        bm,
        [mats["orange"], mats["white"], mats["sand"], mats["webbing"]],
        col,
        recalc_normals=False,
        smooth=True,
    )


def _build_main_deploy_bag(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Ballistic-nylon parachute deployment bag visible during line stretch (1545..1604)."""
    bm = bmesh.new()
    res_bag = bmesh.ops.create_cube(bm, size=1.0)
    verts = res_bag["verts"]
    bmesh.ops.scale(bm, vec=(0.40, 0.32, 0.56), verts=verts)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.045, segments=2, affect="EDGES")
    for f in bm.faces:
        f.material_index = 0

    # Orange release webbing strap around deployment pack
    for z_off in (-0.14, 0.14):
        res_strap = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(0.415, 0.335, 0.045), verts=res_strap["verts"])
        bmesh.ops.translate(bm, vec=(0.0, 0.0, z_off), verts=res_strap["verts"])
        for v in res_strap["verts"]:
            for f in v.link_faces:
                f.material_index = 1

    return _finalize_mesh_obj(
        "Chute_Main_DeployBag",
        bm,
        [mats["bag"], mats["orange"]],
        col,
        recalc_normals=True,
        smooth=True,
    )


def _build_main_swivel(
    mats: dict[str, bpy.types.Material], col: bpy.types.Collection | None
) -> bpy.types.Object:
    """Forged titanium/steel confluence yoke & swivel ring above pilot's shoulders."""
    bm = bmesh.new()
    # Cross yoke bar for Left/Right risers (+-0.036 m in X)
    res_bar = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.11, 0.032, 0.030), verts=res_bar["verts"])
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.005, segments=2, affect="EDGES")

    # Central swivel barrel & upper suspension-line collar
    res_cyl = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=16, radius1=0.024, radius2=0.032, depth=0.060
    )
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.030), verts=res_cyl["verts"])

    for f in bm.faces:
        f.material_index = 0
    return _finalize_mesh_obj(
        "Chute_Main_Swivel", bm, [mats["steel"]], col, recalc_normals=True, smooth=True
    )


# ---------------------------------------------------------------------------
# ANIMATION DRIVERS (EVALUATED STRAIGHT FROM ejection_contract.py)
# ---------------------------------------------------------------------------
def _animate_drogue_system(
    canopy: bpy.types.Object,
    slug: bpy.types.Object,
    lanyard: bpy.types.Object,
    confluence: bpy.types.Object,
    bridle: bpy.types.Object,
    lines: list[bpy.types.Object],
    total_frames: int,
) -> None:
    f_start = C.EV["drogue_fire"]
    f_end = min(C.EV["seat_hidden"] - 1, total_frames)
    vis_ranges = [(f_start, f_end)]

    all_drogue = [canopy, slug, lanyard, confluence, bridle] + lines
    for obj in all_drogue:
        _key_visibility(obj, vis_ranges, total_frames)

    prev_q: dict[str, Quaternion | None] = {o.name: None for o in all_drogue}

    for f in range(f_start, f_end + 1):
        u = C.drogue_open_fraction(f)
        seat_pos = Vector(C.seat_world_location(f))
        seat_rot = Euler(C.seat_world_rotation_euler(f), "XYZ").to_matrix()
        mortar_world = seat_pos + seat_rot @ Vector(C.SEAT["drogue_mortar"])
        canopy_pos = Vector(C.drogue_world_location(f))

        chord = canopy_pos - mortar_world
        dist = chord.length
        chord_dir = chord / dist if dist > 1e-4 else (seat_rot @ DROGUE_TRAIL_DIR)
        mortar_dir = (seat_rot @ DROGUE_TRAIL_DIR).normalized()
        axis_dir = mortar_dir.lerp(chord_dir, C.smoothstep(0.08, 0.45, dist)).normalized()

        # Subtle high-speed slipstream wake flutter after inflation starts
        t_s = (f - f_start) / float(C.FPS)
        flutter_w = C.smoothstep(0.35, 1.0, u)
        right_hint = (seat_rot @ Vector((1.0, 0.0, 0.0))).normalized()
        q_base = _basis_quat(axis_dir, right_hint, prev_q[canopy.name])
        flutter_eul = Euler(
            (
                math.radians(2.2) * math.sin(t_s * 7.2) * flutter_w,
                math.radians(1.8) * math.cos(t_s * 5.9 + 0.7) * flutter_w,
                0.0,
            ),
            "XYZ",
        )
        q_canopy = q_base @ flutter_eul.to_quaternion()
        if prev_q[canopy.name] is not None and prev_q[canopy.name].dot(q_canopy) < 0.0:
            q_canopy = -q_canopy
        prev_q[canopy.name] = q_canopy
        canopy_rot_m = q_canopy.to_matrix()

        # Two-stage drogue radial blossom: packed mortar cylinder -> mushroom -> 1.5m stabilized dome
        if u <= 0.30:
            tl = C.smoothstep(0.0, 0.30, u)
            s_xy = C.lerp(0.11, 0.24, tl)
            s_z = C.lerp(0.42, 1.28, tl)
        else:
            tb = C.smootherstep(0.30, 1.0, u)
            snap = 0.035 * (math.sin(math.pi * tb) ** 2) * C.smoothstep(0.4, 0.9, tb)
            breath = 0.014 * math.sin(t_s * 9.0) * C.smoothstep(0.85, 1.0, u)
            s_xy = C.lerp(0.24, 1.00, tb) + snap + breath
            s_z = C.lerp(1.28, 1.00, tb) - 0.5 * breath

        canopy.location = canopy_pos
        canopy.rotation_quaternion = q_canopy
        canopy.scale = (s_xy, s_xy, s_z)
        canopy.keyframe_insert("location", frame=f)
        canopy.keyframe_insert("rotation_quaternion", frame=f)
        canopy.keyframe_insert("scale", frame=f)

        # Confluence swivel along chord from mortar -> canopy skirt
        confl_len = min(DROGUE_CONFLUENCE_OFFSET * (0.20 + 0.80 * u), 0.55 * max(dist, 0.05))
        confl_pos = canopy_pos - chord_dir * confl_len

        prev_q[confluence.name] = _keyframe_segment_endpoints(
            confluence, f, confl_pos, confl_pos + chord_dir, right_hint, prev_q[confluence.name], min_len=1.0
        )
        # Keep confluence scale uniform 1.0 (not stretched like a line)
        confluence.scale = (1.0, 1.0, 1.0)
        confluence.keyframe_insert("scale", frame=f)

        # Heavy nylon bridle from evaluated seat mortar -> confluence swivel
        prev_q[bridle.name] = _keyframe_segment_endpoints(
            bridle, f, mortar_world, confl_pos, right_hint, prev_q[bridle.name], min_len=0.01
        )

        # 8 suspension lines from confluence swivel -> exact canopy skirt attachment points
        for k, line_obj in enumerate(lines):
            th = (2.0 * math.pi * k) / float(DROGUE_LINE_COUNT)
            skirt_local = Vector((DROGUE_RADIUS * s_xy * math.cos(th), DROGUE_RADIUS * s_xy * math.sin(th), 0.0))
            skirt_world = canopy_pos + canopy_rot_m @ skirt_local
            prev_q[line_obj.name] = _keyframe_segment_endpoints(
                line_obj, f, confl_pos, skirt_world, right_hint, prev_q[line_obj.name], min_len=0.01
            )

        # Extractor slug & apex lanyard directly ahead of canopy apex vent
        canopy_z = canopy_rot_m @ Vector((0.0, 0.0, 1.0))
        apex_world = canopy_pos + canopy_z * (DROGUE_HEIGHT * s_z)
        slug_offset = 0.24 * (0.35 + 0.65 * u)
        slug_world = apex_world + canopy_z * slug_offset
        q_slug = _basis_quat(canopy_z, right_hint, prev_q[slug.name])
        prev_q[slug.name] = q_slug
        slug.location = slug_world
        slug.rotation_quaternion = q_slug
        slug.scale = (1.0, 1.0, 1.0)
        slug.keyframe_insert("location", frame=f)
        slug.keyframe_insert("rotation_quaternion", frame=f)
        slug.keyframe_insert("scale", frame=f)

        prev_q[lanyard.name] = _keyframe_segment_endpoints(
            lanyard, f, apex_world, slug_world, right_hint, prev_q[lanyard.name], min_len=0.01
        )

    for obj in all_drogue:
        _set_linear_transform_fcurves(obj)


def _animate_main_system(
    canopy: bpy.types.Object,
    deploy_bag: bpy.types.Object,
    swivel: bpy.types.Object,
    riser_l: bpy.types.Object,
    riser_r: bpy.types.Object,
    lines: list[bpy.types.Object],
    total_frames: int,
) -> None:
    f_start = C.EV["separation"]
    f_end = total_frames
    vis_ranges = [(f_start, f_end)]

    main_objs = [canopy, swivel, riser_l, riser_r] + lines
    for obj in main_objs:
        _key_visibility(obj, vis_ranges, total_frames)

    # Deployment bag visible during line stretch & release, then CONSTANT-hidden
    _key_visibility(deploy_bag, [(f_start, min(MAIN_BAG_HIDE_FRAME - 1, total_frames))], total_frames)

    all_objs = main_objs + [deploy_bag]
    prev_q: dict[str, Quaternion | None] = {o.name: None for o in all_objs}

    for f in range(f_start, f_end + 1):
        u = C.main_chute_open_fraction(f)
        stretch = C.smoothstep(0.0, 0.25, u)

        carrier_pos = Vector(C.pilot_carrier_location(f))
        carrier_eul = Euler(C.pilot_carrier_rotation_euler(f), "XYZ")
        carrier_rot = carrier_eul.to_matrix()
        carrier_mat = Matrix.Translation(carrier_pos) @ carrier_rot.to_4x4()

        riser_l_world = carrier_mat @ Vector(C.RISER_ATTACH_L)
        riser_r_world = carrier_mat @ Vector(C.RISER_ATTACH_R)
        riser_mid = 0.5 * (riser_l_world + riser_r_world)
        carrier_right = (carrier_rot @ Vector((1.0, 0.0, 0.0))).normalized()

        canopy_pos = Vector(C.main_chute_world_location(f))
        chord = canopy_pos - riser_mid
        dist = chord.length
        chord_dir = chord / dist if dist > 1e-4 else Vector((0.0, 0.0, 1.0))

        # Canopy tilt: aligns with stretching lines -> gentle pendulum sway synced to pilot_carrier
        if u <= 0.25:
            canopy_z = chord_dir
        else:
            t_open = C.clamp01((u - 0.25) / 0.75)
            sway_eul = Euler((0.40 * carrier_eul.x, 0.40 * carrier_eul.y, 0.0), "XYZ")
            sway_z = (sway_eul.to_matrix() @ Vector((0.0, 0.0, 1.0))).normalized()
            canopy_z = chord_dir.lerp(sway_z, C.smoothstep(0.0, 0.35, t_open)).normalized()

        q_canopy = _basis_quat(canopy_z, carrier_right, prev_q[canopy.name])
        prev_q[canopy.name] = q_canopy
        canopy_rot_m = q_canopy.to_matrix()

        # Two-stage main canopy inflation (line stretch u<=0.25 -> blossom u in 0.25..1.0)
        if u <= 0.25:
            tl = C.clamp01(u / 0.25)
            s_xy = C.lerp(0.045, 0.16, C.smoothstep(0.0, 1.0, tl))
            s_z = C.lerp(0.20, 1.25, C.smoothstep(0.0, 1.0, tl))
        else:
            to = C.clamp01((u - 0.25) / 0.75)
            snap = 0.040 * (math.sin(math.pi * to) ** 2) * C.smoothstep(0.35, 0.90, to)
            s_xy = 0.16 + 0.84 * C.smootherstep(0.0, 1.0, to) + snap
            s_z = C.lerp(1.25, 1.00, C.smootherstep(0.0, 1.0, to))
            if f > C.EV["main_full"]:
                dt = (f - C.EV["main_full"]) / float(C.FPS)
                breath = 0.015 * math.sin(dt * 3.1) * math.exp(-0.20 * dt)
                s_xy += breath
                s_z -= 0.65 * breath

        canopy.location = canopy_pos
        canopy.rotation_quaternion = q_canopy
        canopy.scale = (s_xy, s_xy, s_z)
        canopy.keyframe_insert("location", frame=f)
        canopy.keyframe_insert("rotation_quaternion", frame=f)
        canopy.keyframe_insert("scale", frame=f)

        # Deployment bag streaks up during line stretch and clears upward off the apex
        if f < MAIN_BAG_HIDE_FRAME:
            lift_off = 3.8 * C.smoothstep(C.EV["main_line_stretch"] - 4, MAIN_BAG_HIDE_FRAME - 1, f)
            bag_pos = canopy_pos + canopy_z * (0.35 * s_z + lift_off)
            q_bag = _basis_quat(canopy_z, carrier_right, prev_q[deploy_bag.name])
            prev_q[deploy_bag.name] = q_bag
            deploy_bag.location = bag_pos
            deploy_bag.rotation_quaternion = q_bag
            deploy_bag.scale = (1.0, 1.0, 1.0)
            deploy_bag.keyframe_insert("location", frame=f)
            deploy_bag.keyframe_insert("rotation_quaternion", frame=f)
            deploy_bag.keyframe_insert("scale", frame=f)

        # Confluence swivel positioned cleanly above pilot's shoulders/helmet
        riser_len = min(C.lerp(0.20, MAIN_SWIVEL_MAX_LEN, stretch), 0.45 * max(dist, 0.1))
        swivel_pos = riser_mid + chord_dir * riser_len
        q_swivel = _basis_quat(chord_dir, carrier_right, prev_q[swivel.name])
        prev_q[swivel.name] = q_swivel
        swivel_rot_m = q_swivel.to_matrix()

        swivel.location = swivel_pos
        swivel.rotation_quaternion = q_swivel
        swivel.scale = (1.0, 1.0, 1.0)
        swivel.keyframe_insert("location", frame=f)
        swivel.keyframe_insert("rotation_quaternion", frame=f)
        swivel.keyframe_insert("scale", frame=f)

        swivel_right = swivel_rot_m @ Vector((1.0, 0.0, 0.0))
        swivel_fwd = swivel_rot_m @ Vector((0.0, 1.0, 0.0))
        swivel_l = swivel_pos - swivel_right * 0.036
        swivel_r = swivel_pos + swivel_right * 0.036
        swivel_top = swivel_pos + chord_dir * 0.045

        # Left & Right Kevlar shoulder risers: 0.000mm contact with RISER_ATTACH_L/R & swivel
        prev_q[riser_l.name] = _keyframe_segment_endpoints(
            riser_l, f, riser_l_world, swivel_l, carrier_right, prev_q[riser_l.name], min_len=0.02
        )
        prev_q[riser_r.name] = _keyframe_segment_endpoints(
            riser_r, f, riser_r_world, swivel_r, carrier_right, prev_q[riser_r.name], min_len=0.02
        )

        # 16 suspension lines from swivel collar ring -> 16 exact skirt attachment tabs
        for k, line_obj in enumerate(lines):
            th = (k + 0.5) * (2.0 * math.pi / float(MAIN_LINE_COUNT))
            c_th, s_th = math.cos(th), math.sin(th)
            swivel_anchor = swivel_top + (swivel_right * c_th + swivel_fwd * s_th) * 0.022
            skirt_local = Vector((MAIN_RADIUS * s_xy * c_th, MAIN_RADIUS * s_xy * s_th, 0.0))
            skirt_world = canopy_pos + canopy_rot_m @ skirt_local
            prev_q[line_obj.name] = _keyframe_segment_endpoints(
                line_obj, f, swivel_anchor, skirt_world, carrier_right, prev_q[line_obj.name], min_len=0.02
            )

    for obj in all_objs:
        _set_linear_transform_fcurves(obj)


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def build_parachutes(
    col: bpy.types.Collection,
    seat_root: bpy.types.Object | None,
    pilot_carrier: bpy.types.Object | None,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build and keyframe Drogue + Main Ring-Sail parachute systems inside `col`."""
    del seat_root, pilot_carrier  # World kinematics read directly from ejection_contract for 0.000mm drift
    target_col = col or bpy.context.scene.collection
    mats = _build_materials()

    # 1. Drogue Parachute objects
    drogue_canopy = _build_drogue_canopy_mesh(mats, target_col)
    drogue_slug = _build_drogue_extractor_slug(mats, target_col)
    drogue_lanyard = _build_unit_line_mesh_obj(
        "Chute_Drogue_ApexLanyard", radius=0.0055, sides=8, mat=mats["cord"], col=target_col
    )
    drogue_confluence = _build_drogue_confluence(mats, target_col)
    drogue_bridle = _build_unit_strap_mesh_obj(
        "Chute_Drogue_Bridle", width=0.036, thickness=0.010, mat=mats["webbing"], col=target_col
    )
    drogue_lines = [
        _build_unit_line_mesh_obj(
            f"Chute_Drogue_Line_{k:02d}", radius=0.0045, sides=8, mat=mats["cord"], col=target_col
        )
        for k in range(DROGUE_LINE_COUNT)
    ]

    # 2. Main Ring-Sail Parachute objects
    main_canopy = _build_main_ringsail_canopy_mesh(mats, target_col)
    main_bag = _build_main_deploy_bag(mats, target_col)
    main_swivel = _build_main_swivel(mats, target_col)
    riser_l = _build_unit_strap_mesh_obj(
        "Chute_Main_Riser_L", width=0.048, thickness=0.012, mat=mats["kevlar_riser"], col=target_col
    )
    riser_r = _build_unit_strap_mesh_obj(
        "Chute_Main_Riser_R", width=0.048, thickness=0.012, mat=mats["kevlar_riser"], col=target_col
    )
    main_lines = [
        _build_unit_line_mesh_obj(
            f"Chute_Main_Line_{k:02d}", radius=0.0070, sides=8, mat=mats["cord"], col=target_col
        )
        for k in range(MAIN_LINE_COUNT)
    ]

    # 3. Frame-by-frame trajectory & attachment animation
    _animate_drogue_system(
        drogue_canopy, drogue_slug, drogue_lanyard, drogue_confluence, drogue_bridle, drogue_lines, total_frames
    )
    _animate_main_system(
        main_canopy, main_bag, main_swivel, riser_l, riser_r, main_lines, total_frames
    )

    all_objs: list[bpy.types.Object] = [
        drogue_canopy,
        drogue_slug,
        drogue_lanyard,
        drogue_confluence,
        drogue_bridle,
        *drogue_lines,
        main_canopy,
        main_bag,
        main_swivel,
        riser_l,
        riser_r,
        *main_lines,
    ]

    return {
        "drogue_canopy": drogue_canopy,
        "drogue_lines": drogue_lines,
        "drogue_bridle": drogue_bridle,
        "drogue_confluence": drogue_confluence,
        "drogue_extractor": drogue_slug,
        "main_canopy": main_canopy,
        "main_lines": main_lines,
        "main_risers": [riser_l, riser_r],
        "main_swivel": main_swivel,
        "main_deploy_bag": main_bag,
        "all_objs": all_objs,
    }


# ---------------------------------------------------------------------------
# STANDALONE SELF-TEST (factory reset & render f=1430, f=1700 -> /tmp/fje_chutes/)
# ---------------------------------------------------------------------------
def _build_dummy_proxy(name: str, dims: tuple[float, float, float], col: bpy.types.Collection) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=dims, verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0.0, 0.0, dims[2] * 0.5), verts=bm.verts)
    mat = bpy.data.materials.new(f"{name}_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        _set_bsdf(bsdf, "Base Color", (0.20, 0.23, 0.26, 1.0))
    return _finalize_mesh_obj(name, bm, [mat], col, recalc_normals=True, smooth=False)


def _run_self_test() -> None:
    out_dir = Path("/tmp/fje_chutes")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES
    scene.render.fps = C.FPS

    # Simple golden-hour sky + sun for standalone QA only
    world = bpy.data.worlds.new("ChuteTestWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(22.0)
    sky.sun_rotation = math.radians(150.0)
    w_out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], w_out.inputs["Surface"])

    sun_data = bpy.data.lights.new("TestSun", type="SUN")
    sun_data.energy = 4.0
    sun_obj = bpy.data.objects.new("TestSun", sun_data)
    sun_dir = Vector(C.SUN_TOWARD).normalized()
    sun_obj.rotation_euler = (-sun_dir).to_track_quat("-Z", "Y").to_euler("XYZ")
    scene.collection.objects.link(sun_obj)

    col = bpy.data.collections.new("Parachutes_Test")
    scene.collection.children.link(col)

    dummy_seat = _build_dummy_proxy("Dummy_Seat", (0.55, 0.65, 1.15), col)
    dummy_pilot = _build_dummy_proxy("Dummy_Pilot", (0.45, 0.35, 0.90), col)
    for f in (1430, 1700):
        dummy_seat.location = C.seat_world_location(f)
        dummy_seat.rotation_quaternion = Euler(C.seat_world_rotation_euler(f), "XYZ").to_quaternion()
        dummy_seat.keyframe_insert("location", frame=f)
        dummy_seat.keyframe_insert("rotation_quaternion", frame=f)

        dummy_pilot.location = C.pilot_carrier_location(f)
        dummy_pilot.rotation_quaternion = Euler(C.pilot_carrier_rotation_euler(f), "XYZ").to_quaternion()
        dummy_pilot.keyframe_insert("location", frame=f)
        dummy_pilot.keyframe_insert("rotation_quaternion", frame=f)

    res = build_parachutes(col, dummy_seat, dummy_pilot, total_frames=C.TOTAL_FRAMES)
    print(f"[parachute_asset] Built {len(res['all_objs'])} Chute_* objects.")

    cam_data = bpy.data.cameras.new("TestCam")
    cam_data.lens = 28.0
    cam_obj = bpy.data.objects.new("TestCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 6
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 25
    scene.render.image_settings.file_format = "PNG"

    # Render Beat 5 drogue test frame (f=1430) and Beat 6 main ring-sail test frame (f=1700)
    shots = [
        (1430, Vector(C.seat_world_location(1430)) + Vector((4.8, -5.8, 2.6)), Vector(C.seat_world_location(1430)) + Vector((0.1, -0.8, 2.3))),
        (1700, Vector(C.pilot_carrier_location(1700)) + Vector((10.5, -13.5, 4.2)), Vector(C.pilot_carrier_location(1700)) + Vector((0.0, 0.0, 4.6))),
    ]
    for f, cam_pos, target in shots:
        scene.frame_set(f)
        cam_obj.location = cam_pos
        cam_obj.rotation_euler = (target - cam_pos).normalized().to_track_quat("-Z", "Y").to_euler("XYZ")
        out_path = out_dir / f"frame_{f:04d}.png"
        scene.render.filepath = os.fspath(out_path)
        bpy.ops.render.render(write_still=True)
        print(f"[parachute_asset] Rendered {out_path}")


if __name__ == "__main__":
    _run_self_test()
