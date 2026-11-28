"""
Environment, Vibrant Daylight Sky, 3D Cumulus Cloud Banks, Enemy Radar Command Room & Lighting Rig
for Episode #4: "How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Daylight Hollywood-VFX & Enemy Bunker Edition (V2):
  1. High-Chromaticity AgX-Calibrated Cobalt/Azure Daylight Sky World + Procedural Cirrus Strands
     (eliminates washed-out grey AgX sky so the sky renders rich aviation blue!).
  2. Real 3D Sculpted Sunlit Cumulus Cloud Banks (`Env_3DCumulusClouds`, Beats 1, 2, 5 & 6) floating
     at multiple depths (`Y = 5.2..9.5m`) for true Hollywood aerial parallax behind the jets!
  3. Architectural Enemy Air-Defense Radar Command Room (`Env_EnemyCommandRoom`, Beats 1 & 6):
     - Full-depth console cabinet extending down to `Z = -0.75m` (ZERO grey gap at bottom of frame!)
     - Low-profile console sill (`Z = 0.82m`) & wrist bolster (`Z = 0.80m`) so the Beat 1 & Beat 6
       green PPI radar scopes (`Z = 1.18m`) stand 100% unobstructed (ZERO bars cutting across scopes!)
     - Wide panoramic armored bunker window + backlit telemetry screens + illuminated tactical buttons
     - Sculpted Military Air-Defense Radar Officer (`Env_EnemyOperator`) wearing a peaked military
       visor cap, uniform collar/epaulets, and tactical over-ear comms headset + boom mic!
  4. Sunlit Outdoor Military Parabolic Radar Tower (`Env_GroundRadar_Root`, Beats 1, 2 & 5).
  5. Dramatic Multi-Peak Sunlit Mountain Range (`Env_DaylightRidge`, Beats 1 & 6).
  6. Crisp Golden-White Daylight SUN Key + Cool Sky Rim + Anti-Glare Front Fill (`visible_glossy = False`).

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules (iter_action_fcurves, get_mix_sockets,
materials appended before bm.to_mesh).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stealth_contract as C  # noqa: E402

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]


def _set_const_hide(obj: bpy.types.Object) -> None:
    ad = getattr(obj, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in C.iter_action_fcurves(ad.action):
        if fc.data_path.startswith("hide_"):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def _key_visibility(obj: bpy.types.Object, windows: list[tuple[int, int]], total_frames: int) -> None:
    frames = {1, total_frames}
    for s, e in windows:
        sc = max(1, min(total_frames, s))
        ec = max(1, min(total_frames, e))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < total_frames:
            frames.add(ec + 1)
    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in windows)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_const_hide(obj)


def _mesh_from_bm(
    name: str,
    bm: bmesh.types.BMesh,
    mats: list[bpy.types.Material],
    col: bpy.types.Collection,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent: bpy.types.Object | None = None,
    smooth: bool = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    for m in mats:
        mesh.materials.append(m)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ")
    if parent is not None:
        obj.parent = parent
    col.objects.link(obj)
    return obj


def _add_box(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    bevel_w: float = 0.0,
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    sx, sy, sz = size
    rot_m = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ").to_matrix().to_4x4()
    mat = Matrix.Translation(Vector(center)) @ rot_m @ Matrix.Diagonal((sx, sy, sz, 1.0))
    res = bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    vset = set(res["verts"])
    for f in bm.faces:
        if all(v in vset for v in f.verts):
            f.material_index = mat_index
    if bevel_w > 0.0:
        edges = [e for e in bm.edges if e.verts[0] in vset and e.verts[1] in vset]
        bmesh.ops.bevel(bm, geom=edges, offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)


def _add_strut(
    bm: bmesh.types.BMesh,
    p0: Vector,
    p1: Vector,
    radius: float,
    mat_index: int = 0,
    segs: int = 10,
) -> None:
    seg = p1 - p0
    length = seg.length
    if length < 1e-5:
        return
    mid = 0.5 * (p0 + p1)
    rot_m = seg.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    mat = Matrix.Translation(mid) @ rot_m
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs,
        radius1=radius, radius2=radius, depth=length, matrix=mat,
    )
    vset = set(res["verts"])
    for f in bm.faces:
        if all(v in vset for v in f.verts):
            f.material_index = mat_index


# ==============================================================================
# 1. VIBRANT AGX-CALIBRATED COBALT-AZURE DAYLIGHT SKY WORLD
# ==============================================================================
def _setup_world(total_frames: int) -> bpy.types.World:
    """
    High-chromaticity deep aviation blue sky calibrated specifically for `AgX - Punchy`.
    Maps the horizon-to-zenith gradient tightly across `Generated.Z in [0.42..0.64]` so the camera
    sees rich cobalt/azure blue with crisp white high-altitude cirrus bands (never washed-out grey!).
    """
    _ = total_frames
    world = bpy.data.worlds.new("World_DaylightAviationSky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    # Tight vertical sky gradient tuned to camera view band (Generated Z: 0.43 horizon -> 0.64 zenith)
    sky_ramp = nt.nodes.new("ShaderNodeValToRGB")
    sky_ramp.color_ramp.elements[0].position = 0.38
    sky_ramp.color_ramp.elements[0].color = (0.16, 0.38, 0.72, 1.0)   # Lower horizon blue
    e_hor = sky_ramp.color_ramp.elements.new(0.46)
    e_hor.color = (0.18, 0.48, 0.88, 1.0)                             # Crisp sunlit azure horizon
    e_mid = sky_ramp.color_ramp.elements.new(0.54)
    e_mid.color = (0.03, 0.20, 0.64, 1.0)                             # Rich saturated aviation blue
    sky_ramp.color_ramp.elements[-1].position = 0.66
    sky_ramp.color_ramp.elements[-1].color = (0.008, 0.06, 0.34, 1.0) # Deep stratospheric cobalt
    nt.links.new(sep.outputs["Z"], sky_ramp.inputs["Fac"])

    # Crisp high-contrast cirrus cloud wisps (high threshold so sky stays 75% rich blue!)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (2.4, 2.4, 8.5)
    nt.links.new(tc.outputs["Generated"], mapping.inputs["Vector"])

    noise_c = nt.nodes.new("ShaderNodeTexNoise")
    noise_c.inputs["Scale"].default_value = 3.8
    noise_c.inputs["Detail"].default_value = 7.0
    noise_c.inputs["Roughness"].default_value = 0.55
    nt.links.new(mapping.outputs["Vector"], noise_c.inputs["Vector"])

    cloud_ramp = nt.nodes.new("ShaderNodeValToRGB")
    cloud_ramp.color_ramp.elements[0].position = 0.62
    cloud_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cloud_ramp.color_ramp.elements[1].position = 0.84
    cloud_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(noise_c.outputs["Fac"], cloud_ramp.inputs["Fac"])

    cloud_soft = nt.nodes.new("ShaderNodeMath")
    cloud_soft.operation = "MULTIPLY"
    cloud_soft.inputs[1].default_value = 0.45
    nt.links.new(cloud_ramp.outputs["Color"], cloud_soft.inputs[0])

    mix_sky = nt.nodes.new("ShaderNodeMix")
    mix_sky.data_type = "RGBA"
    fac_s, col_a, col_b = C.get_mix_sockets(mix_sky)
    nt.links.new(cloud_soft.outputs[0], fac_s)
    nt.links.new(sky_ramp.outputs["Color"], col_a)
    col_b.default_value = (0.92, 0.96, 1.00, 1.0)

    nt.links.new(mix_sky.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.96
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world


# ==============================================================================
# MATERIALS
# ==============================================================================
def _mat_solid(
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.2,
    roughness: float = 0.45,
    emit_col: tuple[float, float, float] | None = None,
    emit_str: float = 0.0,
    spec_ior: float = 0.5,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = spec_ior
    if emit_col is not None and emit_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_col, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    return mat


def _mat_cumulus_cloud() -> bpy.types.Material:
    """Sunlit 3D cumulus cloud shader: warm ivory sunlit top + soft sky-blue underside shadow."""
    mat = bpy.data.materials.new("Mat_Env_3DCumulus")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep.inputs["Vector"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.4
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    # Normal.Z (-1..+1) -> top sunlit white-ivory, bottom cool sky-blue shadow
    nz_map = nt.nodes.new("ShaderNodeMapRange")
    nz_map.clamp = True
    nz_map.inputs["From Min"].default_value = -0.45
    nz_map.inputs["From Max"].default_value = 0.65
    nt.links.new(sep.outputs["Z"], nz_map.inputs["Value"])

    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac, ca, cb = C.get_mix_sockets(mix_c)
    nt.links.new(nz_map.outputs["Result"], fac)
    ca.default_value = (0.34, 0.48, 0.68, 1.0)   # Cool blue cloud base shadow
    cb.default_value = (0.96, 0.97, 1.00, 1.0)   # Bright sunlit cumulus crown

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.28
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Emission Color"].default_value = (0.88, 0.94, 1.00, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.22
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_radar_dish_mesh() -> bpy.types.Material:
    """Realistic sunlit military perforated aluminum-titanium radar reflector dish material."""
    mat = bpy.data.materials.new("Mat_Env_RadarDishMesh")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 46.0
    vor.inputs["Randomness"].default_value = 0.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])

    grid = nt.nodes.new("ShaderNodeMapRange")
    grid.clamp = True
    grid.inputs["From Min"].default_value = 0.04
    grid.inputs["From Max"].default_value = 0.14
    nt.links.new(vor.outputs["Distance"], grid.inputs["Value"])

    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac, ca, cb = C.get_mix_sockets(mix_c)
    nt.links.new(grid.outputs["Result"], fac)
    ca.default_value = (0.52, 0.58, 0.65, 1.0)
    cb.default_value = (0.18, 0.22, 0.27, 1.0)

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.006
    nt.links.new(grid.outputs["Result"], bump.inputs["Height"])

    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.86
    bsdf.inputs["Roughness"].default_value = 0.26
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_tactical_screen(name: str, accent_col: tuple[float, float, float]) -> bpy.types.Material:
    """Backlit tactical bunker side-monitor with procedural green/amber scanlines & waveform bars."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.005, 0.018, 0.014, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.38
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.08

    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave_h = nt.nodes.new("ShaderNodeTexWave")
    wave_h.wave_type = "BANDS"
    wave_h.bands_direction = "Z"
    wave_h.inputs["Scale"].default_value = 18.0
    wave_h.inputs["Distortion"].default_value = 1.6
    nt.links.new(tc.outputs["Object"], wave_h.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.55
    ramp.color_ramp.elements[0].color = (0.01, 0.04, 0.03, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*accent_col, 1.0)
    nt.links.new(wave_h.outputs["Fac"], ramp.inputs["Fac"])

    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.35
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_terrain_ridge() -> bpy.types.Material:
    """Sunlit atmospheric mountain terrain with snow-kissed sunlit ridges & deep slate valleys."""
    mat = bpy.data.materials.new("Mat_Env_DaylightRidge")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.2
    noise.inputs["Detail"].default_value = 7.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac, ca, cb = C.get_mix_sockets(mix_c)
    nt.links.new(noise.outputs["Fac"], fac)
    ca.default_value = (0.10, 0.18, 0.30, 1.0)  # Deep atmospheric blue-slate mountain shadow
    cb.default_value = (0.34, 0.40, 0.46, 1.0)  # Sunlit granite ridge peak

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.86
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


# ==============================================================================
# 2. ARCHITECTURAL ENEMY AIR-DEFENSE RADAR COMMAND ROOM (BEATS 1 & 6)
# ==============================================================================
def _build_enemy_command_room(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Dark steel Enemy Air-Defense Radar Command Room framing a wide unobstructed panoramic
    armored observation window looking out into the bright sunlit daylight sky!
    - Full-height lower console cabinet (`Z = -0.75..+0.82m`) so ZERO grey gap ever appears at bottom of frame!
    - Low-profile window sill (`Z = 0.82m`) & wrist bolster (`Z = 0.80m`) so the Beat 1 & Beat 6
      radar scopes (`Z = 1.18m`) stand 100% ABOVE the console deck (ZERO horizontal bars cutting scopes!)
    - Seated Military Air-Defense Radar Officer (`Env_EnemyOperator`) with peaked military visor cap,
      uniform collar & shoulder epaulets, and twin-earcup tactical comms headset + boom mic!
    """
    mat_bunker = _mat_solid("Mat_Cmd_BunkerSteel", (0.040, 0.050, 0.065), metallic=0.78, roughness=0.40, spec_ior=0.32)
    mat_desk = _mat_solid("Mat_Cmd_ConsoleDesk", (0.052, 0.064, 0.080), metallic=0.68, roughness=0.38, spec_ior=0.25)
    mat_trim = _mat_solid("Mat_Cmd_BezelTrim", (0.14, 0.17, 0.21), metallic=0.88, roughness=0.28)
    mat_scr_grn = _mat_tactical_screen("Mat_Cmd_SideScreenGrn", (0.06, 0.92, 0.56))
    mat_scr_amb = _mat_tactical_screen("Mat_Cmd_SideScreenAmb", (1.00, 0.62, 0.10))
    mat_led_grn = _mat_solid("Mat_Cmd_LedGrn", (0.08, 0.95, 0.45), 0.1, 0.25, (0.08, 0.95, 0.45), 1.75)
    mat_led_amb = _mat_solid("Mat_Cmd_LedAmb", (1.00, 0.64, 0.08), 0.1, 0.25, (1.00, 0.64, 0.08), 1.80)
    mat_led_red = _mat_solid("Mat_Cmd_LedRed", (1.00, 0.18, 0.08), 0.1, 0.25, (1.00, 0.18, 0.08), 1.85)
    mat_op_suit = _mat_solid("Mat_Cmd_OperatorSuit", (0.035, 0.048, 0.042), metallic=0.08, roughness=0.72, spec_ior=0.20)
    mat_op_skin = _mat_solid("Mat_Cmd_OperatorSkin", (0.36, 0.26, 0.20), metallic=0.02, roughness=0.52, spec_ior=0.25)
    mat_op_cap = _mat_solid("Mat_Cmd_OperatorCap", (0.025, 0.035, 0.032), metallic=0.15, roughness=0.48, spec_ior=0.30)

    mats = [
        mat_bunker, mat_desk, mat_trim, mat_scr_grn, mat_scr_amb,
        mat_led_grn, mat_led_amb, mat_led_red, mat_op_suit, mat_op_skin, mat_op_cap,
    ]

    room_root = bpy.data.objects.new("Env_EnemyCommandRoom_Root", None)
    col.objects.link(room_root)

    # --- A. Wide Panoramic Armored Bunker Window Frame (Zero Center Mullion!) ---
    bm_room = bmesh.new()
    # Left & Right heavy angled armored window pillars framing outer portrait edges
    _add_box(bm_room, (-1.42, -1.08, 2.15), (0.30, 0.45, 3.35), mat_index=0, bevel_w=0.020)
    _add_box(bm_room, (+1.42, -1.08, 2.15), (0.30, 0.45, 3.35), mat_index=0, bevel_w=0.020)
    # Inner chamfered metallic window jamb trim
    _add_box(bm_room, (-1.25, -1.02, 2.15), (0.045, 0.24, 2.50), mat_index=2, bevel_w=0.008)
    _add_box(bm_room, (+1.25, -1.02, 2.15), (0.045, 0.24, 2.50), mat_index=2, bevel_w=0.008)
    # Overhead armored bunker ceiling header beam (frames top of window cleanly)
    _add_box(bm_room, (0.0, -1.18, 3.62), (3.40, 1.35, 0.42), mat_index=0, bevel_w=0.024)
    _add_box(bm_room, (0.0, -0.96, 3.39), (2.60, 0.14, 0.050), mat_index=2, bevel_w=0.008)
    # Red/amber overhead tactical bunker status strips on top left & right corners
    _add_box(bm_room, (-0.92, -1.05, 3.37), (0.28, 0.04, 0.022), mat_index=7)
    _add_box(bm_room, (+0.92, -1.05, 3.37), (0.28, 0.04, 0.022), mat_index=6)
    # Low-profile armored window sill shelf (Z = 0.80, safely below radar scopes!)
    _add_box(bm_room, (0.0, -0.82, 0.80), (3.20, 0.52, 0.12), mat_index=0, bevel_w=0.014)

    # --- B. Full-Depth Military Radar Tracking Console Desk (Extends to Z=-0.75m: ZERO Bottom Gap!) ---
    _add_box(bm_room, (0.0, -1.12, 0.02), (3.10, 0.90, 1.56), mat_index=1, bevel_w=0.018)
    # Flat horizontal instrument deck shelf at Z = 0.83m (never tilts up in front of radar scopes!)
    _add_box(bm_room, (0.0, -1.12, 0.83), (3.00, 0.82, 0.08), mat_index=1, bevel_w=0.012)
    # Protective steel wrist-rest bolster bar along front console edge at Z = 0.82m (safely below scopes!)
    _add_box(bm_room, (0.0, -1.50, 0.82), (2.96, 0.10, 0.075), mat_index=2, bevel_w=0.016)

    # Left & Right low-profile angled tactical telemetry monitors on outer console wings
    for side, scr_idx in ((-1.0, 3), (+1.0, 4)):
        mx = side * 0.90
        _add_box(
            bm_room, (mx, -0.98, 1.00), (0.42, 0.12, 0.22),
            mat_index=0, bevel_w=0.010, rot_deg=(-18.0, 0.0, -side * 18.0),
        )
        _add_box(
            bm_room, (mx - side * 0.014, -1.04, 1.00), (0.35, 0.02, 0.16),
            mat_index=scr_idx, rot_deg=(-18.0, 0.0, -side * 18.0),
        )

    # Tactile rows of backlit military console pushbuttons (green / amber / red) & rotary dials
    for bx_off in (-1.02, -0.90, -0.78, -0.66, -0.54, -0.42, -0.30, -0.18, 0.82, 0.94):
        led_i = 7 if abs(bx_off) > 0.88 else (6 if abs(bx_off) > 0.60 else 5)
        _add_box(bm_room, (bx_off, -1.32, 0.875), (0.048, 0.034, 0.018), mat_index=led_i)
        _add_box(bm_room, (bx_off, -1.40, 0.872), (0.044, 0.028, 0.015), mat_index=2)

    room_obj = _mesh_from_bm("Env_EnemyCommandRoom", bm_room, mats, col, parent=room_root, smooth=False)
    _key_visibility(room_obj, [(1, 220), (1561, total_frames)], total_frames)

    # --- C. Sculpted Military Air-Defense Radar Officer (Peaked Visor Cap + Epaulets + Headset!) ---
    bm_op = bmesh.new()
    # Uniform jacket torso + tailored shoulders
    res_t = bmesh.ops.create_icosphere(
        bm_op, subdivisions=3, radius=1.0,
        matrix=Matrix.Translation((0.0, 0.0, -0.02)) @ Matrix.Diagonal((0.23, 0.14, 0.26, 1.0)),
    )
    tv = set(res_t["verts"])
    for ft in bm_op.faces:
        if all(v in tv for v in ft.verts):
            ft.material_index = 8
    # Structured military uniform collar & gold-trimmed shoulder epaulets
    _add_box(bm_op, (-0.01, 0.01, 0.21), (0.14, 0.12, 0.05), mat_index=10, bevel_w=0.008)
    for sx in (-1.0, +1.0):
        _add_box(
            bm_op, (sx * 0.15, 0.0, 0.17), (0.11, 0.07, 0.018),
            mat_index=10, bevel_w=0.004, rot_deg=(0.0, sx * 18.0, 0.0),
        )
        # Gold officer rank bar on each shoulder epaulet
        _add_box(
            bm_op, (sx * 0.18, 0.0, 0.182), (0.032, 0.045, 0.008),
            mat_index=6, rot_deg=(0.0, sx * 18.0, 0.0),
        )

    # Sculpted neck + head turned slightly left toward radar scope
    res_h = bmesh.ops.create_icosphere(
        bm_op, subdivisions=3, radius=1.0,
        matrix=Matrix.Translation((-0.02, 0.03, 0.32)) @ Matrix.Diagonal((0.094, 0.106, 0.118, 1.0)),
    )
    hv = set(res_h["verts"])
    for fh in bm_op.faces:
        if all(v in hv for v in fh.verts):
            fh.material_index = 9

    # Military Officer Peaked Visor Cap / Tactical Beret Crown + Front Visor Brim + Gold Badge!
    # Cap band around forehead
    res_cb = bmesh.ops.create_cone(
        bm_op, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.098, radius2=0.104, depth=0.042,
        matrix=Matrix.Translation((-0.02, 0.032, 0.392))
        @ Euler((math.radians(8.0), math.radians(-4.0), 0.0), "XYZ").to_matrix().to_4x4(),
    )
    cbv = set(res_cb["verts"])
    for fcb in bm_op.faces:
        if all(v in cbv for v in fcb.verts):
            fcb.material_index = 10
    # Peaked flare crown of military officer cap
    res_cc = bmesh.ops.create_cone(
        bm_op, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.104, radius2=0.132, depth=0.048,
        matrix=Matrix.Translation((-0.02, 0.040, 0.432))
        @ Euler((math.radians(14.0), math.radians(-5.0), 0.0), "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((0.96, 1.10, 1.0, 1.0)),
    )
    ccv = set(res_cc["verts"])
    for fcc in bm_op.faces:
        if all(v in ccv for v in fcc.verts):
            fcc.material_index = 8
    # Front visor brim projecting forward toward console
    _add_box(
        bm_op, (-0.025, 0.118, 0.378), (0.145, 0.095, 0.012),
        mat_index=10, bevel_w=0.004, rot_deg=(-22.0, 0.0, 6.0),
    )

    # Twin-earcup tactical military comms headset + headband + boom mic
    for sx in (-1.0, +1.0):
        res_ec = bmesh.ops.create_cone(
            bm_op, cap_ends=True, cap_tris=False, segments=16,
            radius1=0.044, radius2=0.038, depth=0.036,
            matrix=Matrix.Translation((-0.02 + sx * 0.100, 0.03, 0.32))
            @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        ev = set(res_ec["verts"])
        for fe in bm_op.faces:
            if all(v in ev for v in fe.verts):
                fe.material_index = 0
    _add_strut(bm_op, Vector((-0.115, 0.04, 0.30)), Vector((-0.070, 0.15, 0.27)), 0.006, mat_index=2, segs=6)

    op_obj = _mesh_from_bm(
        "Env_EnemyOperator", bm_op, mats, col,
        location=(0.64, -1.42, 0.90),
        rotation=(0.0, 0.0, math.radians(28.0)),
        parent=room_root,
        smooth=True,
    )
    _key_visibility(op_obj, [(1561, total_frames)], total_frames)

    for f in (1561, 1650, 1730, total_frames):
        u = C.smootherstep(1561.0, float(total_frames), float(f))
        op_obj.rotation_euler = Euler((math.radians(C.lerp(-4.0, 6.0, u)), 0.0, math.radians(C.lerp(32.0, 20.0, u))), "XYZ")
        op_obj.keyframe_insert("rotation_euler", frame=f)

    return [room_obj, op_obj]


# ==============================================================================
# 3. REAL 3D SCULPTED SUNLIT CUMULUS CLOUD BANKS (AERIAL PARALLAX!)
# ==============================================================================
def _build_3d_cumulus_clouds(col: bpy.types.Collection, total_frames: int) -> bpy.types.Object:
    """
    Build real 3D sunlit cumulus cloud banks at multiple depths (`Y = 5.4..9.2m`)
    so the bright blue daylight sky has true volumetric-look cloud billows & parallax!
    """
    mat_cloud = _mat_cumulus_cloud()
    bm = bmesh.new()

    cloud_clusters = [
        # (cx, cy, cz, base_scale_x, base_scale_y, base_scale_z)
        (-2.20, 6.20, 2.85, 0.85, 0.42, 0.38),
        (-1.45, 6.35, 3.05, 0.65, 0.38, 0.44),
        (-2.85, 6.40, 2.72, 0.62, 0.36, 0.32),
        (+1.85, 5.80, 2.45, 0.92, 0.44, 0.40),
        (+2.55, 5.95, 2.68, 0.72, 0.40, 0.46),
        (+1.20, 6.10, 2.38, 0.58, 0.35, 0.30),
        (-0.25, 7.80, 3.45, 1.10, 0.48, 0.42),
        (+0.55, 7.95, 3.62, 0.82, 0.44, 0.48),
        (-1.05, 7.90, 3.32, 0.74, 0.40, 0.34),
        (-2.60, 5.50, 1.65, 0.78, 0.36, 0.28),
        (+2.40, 5.60, 1.72, 0.82, 0.38, 0.30),
    ]
    for cx, cy, cz, sx, sy, sz in cloud_clusters:
        m = Matrix.Translation((cx, cy, cz)) @ Matrix.Diagonal((sx, sy, sz, 1.0))
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0, matrix=m)
        # Add 2 overlapping puff lobes per cluster for organic cumulus silhouette
        for dx, dz, sc in ((-0.48 * sx, -0.10 * sz, 0.68), (+0.52 * sx, +0.08 * sz, 0.74)):
            m_sub = Matrix.Translation((cx + dx, cy - 0.08, cz + dz)) @ Matrix.Diagonal((sx * sc, sy * sc, sz * sc, 1.0))
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0, matrix=m_sub)

    for f in bm.faces:
        f.material_index = 0

    clouds_obj = _mesh_from_bm("Env_3DCumulusClouds", bm, [mat_cloud], col, smooth=True)
    clouds_obj.visible_shadow = False
    # Visible across all outdoor daylight sky beats (Beats 1, 2, 5 & 6)
    _key_visibility(clouds_obj, [(1, 540), (1321, total_frames)], total_frames)
    return clouds_obj


# ==============================================================================
# 4. OUTDOOR SUNLIT MILITARY PARABOLIC RADAR TOWER (BEATS 1, 2 & 5)
# ==============================================================================
def _build_ground_radar_station(col: bpy.types.Collection, total_frames: int) -> bpy.types.Object:
    """
    Sunlit realistic military parabolic radar station (Beats 1, 2 & 5):
    - Concrete octagonal command pad + weathered olive-slate military steel pedestal
    - Side antenna mast with red aviation strobe, rotary azimuth collar, hydraulic elevation struts
    - Perforated aluminum parabolic reflector dish with 8 back-ribs, rim ring & 3-leg tripod feed-horn!
    """
    mat_steel = _mat_solid("Mat_Env_RadarSteel", (0.20, 0.24, 0.26), metallic=0.80, roughness=0.36)
    mat_concrete = _mat_solid("Mat_Env_RadarConcrete", (0.36, 0.37, 0.35), metallic=0.04, roughness=0.82)
    mat_dish = _mat_radar_dish_mesh()
    mat_cyan = _mat_solid(
        "Mat_Env_RadarCore", (0.06, 0.88, 1.00), metallic=0.1, roughness=0.20,
        emit_col=(0.06, 0.88, 1.00), emit_str=1.95,
    )
    mat_warn = _mat_solid(
        "Mat_Env_RadarWarn", (1.00, 0.22, 0.05), metallic=0.1, roughness=0.22,
        emit_col=(1.00, 0.22, 0.05), emit_str=1.80,
    )

    root = bpy.data.objects.new("Env_GroundRadar_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    bm_base = bmesh.new()
    res_pad = bmesh.ops.create_cone(
        bm_base, cap_ends=True, cap_tris=False, segments=8,
        radius1=0.40, radius2=0.34, depth=0.18,
        matrix=Matrix.Translation((0.0, 0.0, -0.50)),
    )
    pad_vs = set(res_pad["verts"])
    for f_p in bm_base.faces:
        if all(v in pad_vs for v in f_p.verts):
            f_p.material_index = 4

    bmesh.ops.create_cone(
        bm_base, cap_ends=True, cap_tris=False, segments=12,
        radius1=0.28, radius2=0.20, depth=0.28,
        matrix=Matrix.Translation((0.0, 0.0, -0.27)),
    )
    bmesh.ops.create_cone(
        bm_base, cap_ends=True, cap_tris=False, segments=20,
        radius1=0.22, radius2=0.22, depth=0.05,
        matrix=Matrix.Translation((0.0, 0.0, -0.11)),
    )
    _add_box(bm_base, (0.0, 0.0, -0.02), (0.20, 0.18, 0.16), mat_index=0, bevel_w=0.012)
    _add_box(bm_base, (-0.125, 0.0, 0.08), (0.042, 0.085, 0.20), mat_index=0, bevel_w=0.006)
    _add_box(bm_base, (+0.125, 0.0, 0.08), (0.042, 0.085, 0.20), mat_index=0, bevel_w=0.006)
    _add_strut(bm_base, Vector((-0.18, -0.14, -0.22)), Vector((-0.12, -0.02, 0.04)), 0.012, mat_index=0)
    _add_strut(bm_base, Vector((+0.18, -0.14, -0.22)), Vector((+0.12, -0.02, 0.04)), 0.012, mat_index=0)
    _add_strut(bm_base, Vector((-0.26, 0.10, -0.42)), Vector((-0.26, 0.10, 0.24)), 0.009, mat_index=0)
    res_bcn = bmesh.ops.create_icosphere(
        bm_base, subdivisions=2, radius=0.018, matrix=Matrix.Translation((-0.26, 0.10, 0.255))
    )
    bcn_vs = set(res_bcn["verts"])
    for f_b in bm_base.faces:
        if all(v in bcn_vs for v in f_b.verts):
            f_b.material_index = 3
    for sx in (-0.16, 0.16):
        _add_box(bm_base, (sx, -0.15, -0.18), (0.022, 0.022, 0.026), mat_index=3)

    base_obj = _mesh_from_bm(
        "Env_RadarPedestal", bm_base, [mat_steel, mat_dish, mat_cyan, mat_warn, mat_concrete], col, parent=root
    )

    bm_dish = bmesh.new()
    segs = 36
    rings = 10
    r_max = 0.32
    depth_k = 0.24
    ring_verts: list[list[bmesh.types.BMVert]] = []
    for ri in range(rings + 1):
        rr = (ri / rings) * r_max
        dz = depth_k * (rr / r_max) ** 2
        row = []
        for si in range(segs):
            a = 2.0 * math.pi * si / segs
            row.append(bm_dish.verts.new((rr * math.cos(a), rr * math.sin(a), dz)))
        ring_verts.append(row)
    for ri in range(rings):
        for si in range(segs):
            sn = (si + 1) % segs
            f_face = bm_dish.faces.new(
                (ring_verts[ri][si], ring_verts[ri][sn], ring_verts[ri + 1][sn], ring_verts[ri + 1][si])
            )
            f_face.material_index = 1
    bmesh.ops.solidify(bm_dish, geom=list(bm_dish.faces), thickness=0.015)

    res_hub = bmesh.ops.create_cone(
        bm_dish, cap_ends=True, cap_tris=False, segments=16,
        radius1=0.085, radius2=0.065, depth=0.075,
        matrix=Matrix.Translation((0.0, 0.0, -0.04)),
    )
    hub_vs = set(res_hub["verts"])
    for fh in bm_dish.faces:
        if all(v in hub_vs for v in fh.verts):
            fh.material_index = 0
    for k in range(8):
        a = 2.0 * math.pi * k / 8.0
        p_in = Vector((0.05 * math.cos(a), 0.05 * math.sin(a), -0.025))
        p_out = Vector((r_max * 0.96 * math.cos(a), r_max * 0.96 * math.sin(a), depth_k * 0.92 - 0.015))
        _add_strut(bm_dish, p_in, p_out, 0.008, mat_index=0, segs=6)

    for si in range(segs):
        a0 = 2.0 * math.pi * si / segs
        a1 = 2.0 * math.pi * ((si + 1) % segs) / segs
        _add_strut(
            bm_dish,
            Vector((r_max * math.cos(a0), r_max * math.sin(a0), depth_k)),
            Vector((r_max * math.cos(a1), r_max * math.sin(a1), depth_k)),
            0.011, mat_index=0, segs=6,
        )

    focus_pt = Vector((0.0, 0.0, 0.225))
    for k in range(3):
        a = math.radians(30.0 + k * 120.0)
        anchor = Vector((r_max * 0.80 * math.cos(a), r_max * 0.80 * math.sin(a), depth_k * 0.64))
        _add_strut(bm_dish, anchor, focus_pt, 0.0085, mat_index=0, segs=8)

    res_wg = bmesh.ops.create_cone(
        bm_dish, cap_ends=True, cap_tris=False, segments=16,
        radius1=0.044, radius2=0.032, depth=0.055,
        matrix=Matrix.Translation((0.0, 0.0, 0.215)),
    )
    wg_vs = set(res_wg["verts"])
    for fwg in bm_dish.faces:
        if all(v in wg_vs for v in fwg.verts):
            fwg.material_index = 0

    res_core = bmesh.ops.create_icosphere(
        bm_dish, subdivisions=2, radius=0.038, matrix=Matrix.Translation((0.0, 0.0, 0.240))
    )
    core_vset = set(res_core["verts"])
    for f_c in bm_dish.faces:
        if all(v in core_vset for v in f_c.verts):
            f_c.material_index = 2

    dish_obj = _mesh_from_bm(
        "Env_RadarDish", bm_dish, [mat_steel, mat_dish, mat_cyan], col,
        location=(0.0, 0.0, 0.14), parent=root,
    )

    b1_pos = Vector(C.STAGE["radar_dish_b1"])
    b2_pos = Vector(C.STAGE["radar_dish_b2"])
    b5_pos = Vector(C.STAGE["radar_dish_b5"])
    for f in (1, 110, 220, 221, 380, 540, 1320, 1321, 1440, 1560):
        if f <= 220:
            root.location = b1_pos
            bpos, _ = C.bomber_pose_at(float(f))
            aim = (Vector(bpos) - (b1_pos + Vector((0.0, 0.0, 0.14)))).normalized()
            dish_obj.rotation_euler = aim.to_track_quat("Z", "Y").to_euler("XYZ")
        elif f <= 540:
            root.location = b2_pos
            tz = C.lerp(2.70, 1.44, C.smootherstep(290.0, 420.0, float(f)))
            aim = (Vector((0.16, 0.12, tz)) - (b2_pos + Vector((0.0, 0.0, 0.14)))).normalized()
            dish_obj.rotation_euler = aim.to_track_quat("Z", "Y").to_euler("XYZ")
        else:
            root.location = b5_pos
            tz = C.lerp(2.50, 1.32, C.smootherstep(1375.0, 1435.0, float(f)))
            aim = (Vector((0.30, 0.08, tz)) - (b5_pos + Vector((0.0, 0.0, 0.14)))).normalized()
            dish_obj.rotation_euler = aim.to_track_quat("Z", "Y").to_euler("XYZ")
        root.keyframe_insert("location", frame=f)
        dish_obj.keyframe_insert("rotation_euler", frame=f)

    for obj_k in (root, dish_obj):
        ad = getattr(obj_k, "animation_data", None)
        if ad and ad.action:
            for fc in C.iter_action_fcurves(ad.action):
                for kp in fc.keyframe_points:
                    if int(round(kp.co.x)) in (220, 540, 1320):
                        kp.interpolation = "CONSTANT"

    for obj_v in (base_obj, dish_obj):
        _key_visibility(obj_v, [(1, 540), (1321, 1560)], total_frames)

    return root


# ==============================================================================
# 5. DRAMATIC SUNLIT MOUNTAIN RANGE HORIZON (BEATS 1 & 6)
# ==============================================================================
def _build_daylight_horizon(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Sculpted sunlit mountain peaks along the distant horizon (`Y = 3.5..14.0m`) with a low
    valley floor (`Z = 0.35m`) so no flat grey horizontal line ever cuts behind the radar scope!
    """
    mat_ridge = _mat_terrain_ridge()

    bm = bmesh.new()
    nx, ny = 32, 12
    verts: list[list[bmesh.types.BMVert]] = []
    for iy in range(ny):
        ty = iy / (ny - 1)
        y = C.lerp(2.60, 14.0, ty)
        row = []
        for ix in range(nx):
            tx = (ix / (nx - 1)) * 2.0 - 1.0
            x = tx * 13.0
            ridge_env = C.smoothstep(0.10, 0.65, ty)
            # Carve a natural low central valley behind the Beat 6 radar scope, with tall mountain peaks flanking left/right
            valley_dip = 0.35 + 0.65 * C.smoothstep(0.45, 2.20, abs(x))
            hill = ridge_env * valley_dip * (
                1.25 * math.sin(x * 0.48 + 0.7) ** 2
                + 0.72 * math.cos(x * 0.92 - 0.3) ** 2
                + 0.35 * ty
            )
            z = 0.32 + hill
            row.append(bm.verts.new((x, y, z)))
        verts.append(row)

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            f_face = bm.faces.new((verts[iy][ix], verts[iy][ix + 1], verts[iy + 1][ix + 1], verts[iy + 1][ix]))
            f_face.material_index = 0

    ridge_obj = _mesh_from_bm("Env_DaylightRidge", bm, [mat_ridge], col, smooth=True)
    _key_visibility(ridge_obj, [(1, 220), (1561, total_frames)], total_frames)
    return [ridge_obj]


# ==============================================================================
# 6. BRIGHT CINEMATIC DAYLIGHT SUN + SKY FILL + ANTI-GLARE BUNKER PRACTICALS
# ==============================================================================
def _add_area_light(
    name: str,
    col: bpy.types.Collection,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
    size: float,
    size_y: float | None = None,
    glossy_vis: bool = True,
) -> bpy.types.Object:
    ldata = bpy.data.lights.new(name, type="AREA")
    ldata.color = color
    ldata.energy = energy
    ldata.shape = "RECTANGLE" if size_y is not None else "SQUARE"
    ldata.size = size
    if size_y is not None:
        ldata.size_y = size_y
    lobj = bpy.data.objects.new(name, ldata)
    lobj.location = Vector(location)
    aim = (Vector(target) - Vector(location)).normalized()
    lobj.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler("XYZ")
    lobj.visible_glossy = glossy_vis
    col.objects.link(lobj)
    return lobj


def _build_lighting(col: bpy.types.Collection, total_frames: int) -> dict[str, bpy.types.Object]:
    """
    Crisp Golden-White Daylight SUN Key + Cool Azure Sky Rim + Anti-Glare Front Fill
    + Moody Enemy Command Room Console Practicals.
    """
    sun_data = bpy.data.lights.new("Light_Daylight_SunKey", type="SUN")
    sun_data.color = (1.00, 0.95, 0.86)
    sun_data.energy = 4.5
    sun_data.angle = math.radians(2.2)
    l_sun = bpy.data.objects.new("Light_Daylight_SunKey", sun_data)
    l_sun.location = Vector((4.2, -2.8, 6.4))
    sun_aim = (Vector((0.0, 0.2, 1.8)) - l_sun.location).normalized()
    l_sun.rotation_euler = sun_aim.to_track_quat("-Z", "Y").to_euler("XYZ")
    col.objects.link(l_sun)

    l_rim = _add_area_light(
        "Light_SkyRim_Top", col,
        location=(2.8, 1.6, 5.4), target=(0.0, 0.0, 1.95),
        color=(0.72, 0.88, 1.00), energy=680.0, size=2.6, size_y=1.8,
        glossy_vis=True,
    )
    l_fill = _add_area_light(
        "Light_SkyFill_AntiGlare", col,
        location=(-1.0, -4.6, 2.2), target=(0.0, 0.0, 1.85),
        color=(0.80, 0.90, 1.00), energy=290.0, size=3.2, size_y=2.4,
        glossy_vis=False,
    )
    l_macro = _add_area_light(
        "Light_MacroPrecision_Key", col,
        location=(0.95, -1.65, 2.85), target=(0.0, 0.0, 1.85),
        color=(0.98, 0.97, 0.94), energy=200.0, size=0.95, size_y=0.70,
        glossy_vis=True,
    )
    l_bunker = _add_area_light(
        "Light_CmdBunker_DeskGlow", col,
        location=(0.20, -1.55, 1.55), target=(0.30, -1.20, 0.95),
        color=(0.24, 0.92, 0.64), energy=28.0, size=0.95, size_y=0.50,
        glossy_vis=False,
    )

    key_frames = [1, 220, 221, 540, 541, 960, 961, 1320, 1321, 1560, 1561, total_frames]
    for f in key_frames:
        is_bunker = (f <= 220) or (f >= 1561)
        is_macro = 541 <= f <= 1320
        l_sun.data.energy = 4.8 if not is_macro else 3.4
        l_rim.data.energy = 700.0 if not is_macro else 540.0
        l_fill.data.energy = 250.0 if is_bunker else (320.0 if not is_macro else 220.0)
        l_macro.data.energy = 215.0 if is_macro else 0.0
        l_bunker.data.energy = 32.0 if is_bunker else 0.0
        for lo in (l_sun, l_rim, l_fill, l_macro, l_bunker):
            lo.data.keyframe_insert("energy", frame=f)

    return {
        "sun_key": l_sun,
        "sky_rim": l_rim,
        "front_fill": l_fill,
        "macro_key": l_macro,
        "bunker_glow": l_bunker,
    }


def build_radar_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Assemble vibrant daylight sky, 3D cumulus clouds, Enemy Radar Command Room, radar tower, horizon & lights."""
    world = _setup_world(total_frames)
    cmd_room = _build_enemy_command_room(col, total_frames)
    clouds_3d = _build_3d_cumulus_clouds(col, total_frames)
    radar_station = _build_ground_radar_station(col, total_frames)
    horizon_objs = _build_daylight_horizon(col, total_frames)
    lights = _build_lighting(col, total_frames)
    return {
        "world": world,
        "command_room": cmd_room,
        "clouds_3d": clouds_3d,
        "radar_station": radar_station,
        "horizon_objs": horizon_objs,
        "lights": lights,
    }
