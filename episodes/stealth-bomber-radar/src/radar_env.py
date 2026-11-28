"""
Environment, Daylight Sky, Enemy Radar Command Room & Lighting Rig for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Daylight Hollywood-VFX & Enemy Bunker Edition:
  1. Bright Cinematic High-Altitude Daylight Sky World (rich azure zenith -> sunlit horizon haze
     + procedural sunlit white cumulus cloud billows) so dark charcoal stealth facets pop at >8:1 contrast!
  2. Architectural Enemy Air-Defense Radar Command Room (`Env_EnemyCommandRoom`, Beats 1 & 6):
     - Dark armored steel bunker interior framing a wide unobstructed panoramic observation window
     - Full-width sloped military radar tracking console desk with backlit telemetry side-monitors,
       illuminated red/amber/green tactical switches, rotary gain dials, trackball, and a seated
       Enemy Radar Operator silhouette with military comms headset tracking the airspace!
  3. Sunlit Outdoor Military Parabolic Radar Tower (Beats 1, 2 & 5):
     - Concrete bunker pad, weathered military steel azimuth turret, hydraulic elevation actuators,
       side comms mast with aviation beacon, perforated metallic parabolic dish, 8 radial back-ribs,
       tubular rim ring, and 3-leg tripod RF waveguide feed-horn!
  4. Sunlit Distant Mountain Ridge & Cloud Horizon Deck (Beats 1, 2, 5 & 6) for true outdoor aviation depth.
  5. Crisp Golden-White Daylight SUN Key + Cool Sky Fill + Rim + Command Bunker Practicals
     (anti-glare `visible_glossy = False` on broad fill lights so radar scope stays 100% reflection-free!).

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
# 1. BRIGHT HIGH-ALTITUDE DAYLIGHT SKY WORLD + SUNLIT CUMULUS CLOUDS
# ==============================================================================
def _setup_world(total_frames: int) -> bpy.types.World:
    """
    Bright cinematic high-altitude daylight sky with warm sunlit horizon haze,
    rich aviation azure mid-sky, deep blue zenith, and layered sunlit white cumulus clouds.
    Gives >8:1 silhouette contrast for the dark charcoal B-2 Stealth Bomber!
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

    # Vertical sky gradient (Generated Z: ~0.48 horizon -> 0.58 mid-sky -> 0.85 zenith)
    sky_ramp = nt.nodes.new("ShaderNodeValToRGB")
    sky_ramp.color_ramp.elements[0].position = 0.36
    sky_ramp.color_ramp.elements[0].color = (0.34, 0.50, 0.68, 1.0)   # Lower atmospheric mist
    e_hor = sky_ramp.color_ramp.elements.new(0.49)
    e_hor.color = (0.64, 0.82, 0.98, 1.0)                               # Sunlit bright horizon haze
    e_mid = sky_ramp.color_ramp.elements.new(0.62)
    e_mid.color = (0.16, 0.46, 0.86, 1.0)                               # Crisp aviation azure sky
    sky_ramp.color_ramp.elements[-1].position = 0.86
    sky_ramp.color_ramp.elements[-1].color = (0.04, 0.18, 0.50, 1.0)    # High-altitude royal blue
    nt.links.new(sep.outputs["Z"], sky_ramp.inputs["Fac"])

    # Layered sunlit cumulus cloud billows (stretched horizontally for natural sky strata)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.6, 1.6, 4.2)
    nt.links.new(tc.outputs["Generated"], mapping.inputs["Vector"])

    noise_c = nt.nodes.new("ShaderNodeTexNoise")
    noise_c.inputs["Scale"].default_value = 3.2
    noise_c.inputs["Detail"].default_value = 8.0
    noise_c.inputs["Roughness"].default_value = 0.58
    nt.links.new(mapping.outputs["Vector"], noise_c.inputs["Vector"])

    cloud_ramp = nt.nodes.new("ShaderNodeValToRGB")
    cloud_ramp.color_ramp.elements[0].position = 0.50
    cloud_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cloud_ramp.color_ramp.elements[1].position = 0.76
    cloud_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(noise_c.outputs["Fac"], cloud_ramp.inputs["Fac"])

    # Keep clouds strongest near mid-lower sky band (Z in 0.44..0.72)
    band_lo = nt.nodes.new("ShaderNodeMapRange")
    band_lo.clamp = True
    band_lo.inputs["From Min"].default_value = 0.42
    band_lo.inputs["From Max"].default_value = 0.52
    nt.links.new(sep.outputs["Z"], band_lo.inputs["Value"])

    band_hi = nt.nodes.new("ShaderNodeMapRange")
    band_hi.clamp = True
    band_hi.inputs["From Min"].default_value = 0.76
    band_hi.inputs["From Max"].default_value = 0.56
    nt.links.new(sep.outputs["Z"], band_hi.inputs["Value"])

    band_mul = nt.nodes.new("ShaderNodeMath")
    band_mul.operation = "MULTIPLY"
    nt.links.new(band_lo.outputs["Result"], band_mul.inputs[0])
    nt.links.new(band_hi.outputs["Result"], band_mul.inputs[1])

    cloud_mask = nt.nodes.new("ShaderNodeMath")
    cloud_mask.operation = "MULTIPLY"
    nt.links.new(cloud_ramp.outputs["Color"], cloud_mask.inputs[0])
    nt.links.new(band_mul.outputs[0], cloud_mask.inputs[1])

    cloud_soft = nt.nodes.new("ShaderNodeMath")
    cloud_soft.operation = "MULTIPLY"
    cloud_soft.inputs[1].default_value = 0.72
    nt.links.new(cloud_mask.outputs[0], cloud_soft.inputs[0])

    # Blend sunlit warm-white cumulus clouds cleanly over azure sky
    mix_sky = nt.nodes.new("ShaderNodeMix")
    mix_sky.data_type = "RGBA"
    fac_s, col_a, col_b = C.get_mix_sockets(mix_sky)
    nt.links.new(cloud_soft.outputs[0], fac_s)
    nt.links.new(sky_ramp.outputs["Color"], col_a)
    col_b.default_value = (0.94, 0.96, 0.99, 1.0)

    nt.links.new(mix_sky.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.18
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
    ca.default_value = (0.48, 0.54, 0.60, 1.0)  # Sunlit brushed aluminum wire rib
    cb.default_value = (0.18, 0.22, 0.27, 1.0)  # Perforated mesh aperture

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.006
    nt.links.new(grid.outputs["Result"], bump.inputs["Height"])

    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.86
    bsdf.inputs["Roughness"].default_value = 0.28
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
    """Sunlit atmospheric mountain terrain / military ridge horizon."""
    mat = bpy.data.materials.new("Mat_Env_DaylightRidge")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.8
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac, ca, cb = C.get_mix_sockets(mix_c)
    nt.links.new(noise.outputs["Fac"], fac)
    ca.default_value = (0.14, 0.21, 0.28, 1.0)  # Atmospheric blue-slate mountain shadow
    cb.default_value = (0.28, 0.34, 0.36, 1.0)  # Sunlit dry ridge rock/scrub

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.88
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
    - Sloped military radar console desk spanning lower foreground (safely above subtitle zone)
    - Backlit side telemetry monitors, illuminated amber/green/crimson control switches, rotary knobs
    - Armored side window pillars & overhead steel ceiling bulkhead (ZERO center mullion!)
    - Seated Enemy Radar Operator silhouette (uniform shoulders + head + military headset) on right side
      watching the radar scope as the invisible stealth bomber glides right outside the window!
    """
    mat_bunker = _mat_solid("Mat_Cmd_BunkerSteel", (0.045, 0.055, 0.070), metallic=0.78, roughness=0.42, spec_ior=0.32)
    mat_desk = _mat_solid("Mat_Cmd_ConsoleDesk", (0.060, 0.072, 0.088), metallic=0.68, roughness=0.38, spec_ior=0.25)
    mat_trim = _mat_solid("Mat_Cmd_BezelTrim", (0.14, 0.17, 0.21), metallic=0.88, roughness=0.28)
    mat_scr_grn = _mat_tactical_screen("Mat_Cmd_SideScreenGrn", (0.06, 0.92, 0.56))
    mat_scr_amb = _mat_tactical_screen("Mat_Cmd_SideScreenAmb", (1.00, 0.62, 0.10))
    mat_led_grn = _mat_solid("Mat_Cmd_LedGrn", (0.08, 0.95, 0.45), 0.1, 0.25, (0.08, 0.95, 0.45), 1.75)
    mat_led_amb = _mat_solid("Mat_Cmd_LedAmb", (1.00, 0.64, 0.08), 0.1, 0.25, (1.00, 0.64, 0.08), 1.80)
    mat_led_red = _mat_solid("Mat_Cmd_LedRed", (1.00, 0.18, 0.08), 0.1, 0.25, (1.00, 0.18, 0.08), 1.85)
    mat_op_suit = _mat_solid("Mat_Cmd_OperatorSuit", (0.032, 0.042, 0.038), metallic=0.06, roughness=0.78, spec_ior=0.18)
    mat_op_skin = _mat_solid("Mat_Cmd_OperatorSkin", (0.28, 0.20, 0.16), metallic=0.02, roughness=0.58, spec_ior=0.25)

    mats = [
        mat_bunker, mat_desk, mat_trim, mat_scr_grn, mat_scr_amb,
        mat_led_grn, mat_led_amb, mat_led_red, mat_op_suit, mat_op_skin,
    ]

    room_root = bpy.data.objects.new("Env_EnemyCommandRoom_Root", None)
    col.objects.link(room_root)

    # --- A. Armored Bunker Window Frame, Side Bulkheads & Ceiling Lintel (Zero Center Mullion!) ---
    bm_room = bmesh.new()
    # Left & Right heavy angled armored window pillars framing 9:16 portrait edges cleanly in Beats 1 & 6
    _add_box(bm_room, (-1.31, -1.08, 2.12), (0.30, 0.45, 3.28), mat_index=0, bevel_w=0.020)
    _add_box(bm_room, (+1.31, -1.08, 2.12), (0.30, 0.45, 3.28), mat_index=0, bevel_w=0.020)
    # Inner chamfered metallic window jamb trim (frames portrait edges without crowding Beat 1 sky/aircraft)
    _add_box(bm_room, (-1.14, -1.02, 2.14), (0.045, 0.24, 2.40), mat_index=2, bevel_w=0.008)
    _add_box(bm_room, (+1.14, -1.02, 2.14), (0.045, 0.24, 2.40), mat_index=2, bevel_w=0.008)
    # Overhead armored bunker ceiling header beam (frames top of window cleanly)
    _add_box(bm_room, (0.0, -1.18, 3.52), (3.25, 1.35, 0.44), mat_index=0, bevel_w=0.024)
    _add_box(bm_room, (0.0, -0.96, 3.28), (2.42, 0.14, 0.055), mat_index=2, bevel_w=0.008)
    # Dim red/amber overhead tactical bunker status strips on left & right corners
    _add_box(bm_room, (-0.88, -1.05, 3.26), (0.26, 0.04, 0.024), mat_index=7)
    _add_box(bm_room, (+0.88, -1.05, 3.26), (0.26, 0.04, 0.024), mat_index=6)
    # Armored lower window sill shelf spanning behind console
    _add_box(bm_room, (0.0, -0.82, 0.94), (3.10, 0.56, 0.14), mat_index=0, bevel_w=0.015)

    # --- B. Full-Width Sloped Military Radar Tracking Console Desk ---
    # Main console cabinet base
    _add_box(bm_room, (0.0, -1.10, 0.52), (2.95, 0.88, 0.78), mat_index=1, bevel_w=0.018)
    # Angled control fascia panel facing the camera/operator (tilted 26 deg toward camera)
    _add_box(bm_room, (0.0, -1.08, 1.02), (2.82, 0.62, 0.14), mat_index=1, bevel_w=0.014, rot_deg=(-26.0, 0.0, 0.0))
    # Protective steel wrist-rest bolster bar along front console edge
    _add_box(bm_room, (0.0, -1.42, 0.88), (2.86, 0.12, 0.09), mat_index=2, bevel_w=0.020)

    # Left & Right low-profile angled tactical telemetry monitors flanking central radar scope
    # Positioned low (top Z ~ 1.21) so outdoor Beat 1 military radar tower stands 100% unobstructed!
    for side, scr_idx in ((-1.0, 3), (+1.0, 4)):
        mx = side * 0.84
        _add_box(
            bm_room, (mx, -0.94, 1.09), (0.44, 0.13, 0.24),
            mat_index=0, bevel_w=0.010, rot_deg=(-20.0, 0.0, -side * 18.0),
        )
        _add_box(
            bm_room, (mx - side * 0.014, -1.00, 1.09), (0.37, 0.02, 0.18),
            mat_index=scr_idx, rot_deg=(-20.0, 0.0, -side * 18.0),
        )

    # Tactile rows of backlit military console pushbuttons (green / amber / red) & rotary dials
    for bx_off in (-0.98, -0.88, -0.78, -0.68, -0.58, 0.56, 0.66, 0.76, 0.86, 0.96):
        led_i = 7 if abs(bx_off) > 0.90 else (6 if abs(bx_off) > 0.72 else 5)
        _add_box(bm_room, (bx_off, -1.24, 0.965), (0.046, 0.032, 0.018), mat_index=led_i, rot_deg=(-26.0, 0.0, 0.0))
        _add_box(bm_room, (bx_off, -1.32, 0.925), (0.042, 0.028, 0.015), mat_index=2, rot_deg=(-26.0, 0.0, 0.0))

    # Rotary azimuth/gain knobs on left and right console shoulders
    for kx in (-0.52, -0.44, 0.44, 0.52):
        res_k = bmesh.ops.create_cone(
            bm_room, cap_ends=True, cap_tris=False, segments=14,
            radius1=0.024, radius2=0.020, depth=0.034,
            matrix=Matrix.Translation((kx, -1.30, 0.945))
            @ Euler((math.radians(-26.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
        )
        kv = set(res_k["verts"])
        for fk in bm_room.faces:
            if all(v in kv for v in fk.verts):
                fk.material_index = 2

    room_obj = _mesh_from_bm("Env_EnemyCommandRoom", bm_room, mats, col, parent=room_root, smooth=False)
    # Visible in Beat 1 (looking past bunker console out at daylight sky) AND Beat 6 (command room payoff!)
    _key_visibility(room_obj, [(1, 220), (1561, total_frames)], total_frames)

    # --- C. Seated Enemy Radar Operator Silhouette (Beat 6 Right Foreground Edge) ---
    # Framed cleanly inside right edge (X=+0.62, Y=-1.42, Z=0.96 -> NDC X ~ +0.72..+0.84) with
    # military comms headset watching central radar scope as invisible stealth bomber glides outside!
    bm_op = bmesh.new()
    # Torso / tactical flight-jacket shoulders
    res_t = bmesh.ops.create_icosphere(
        bm_op, subdivisions=2, radius=1.0,
        matrix=Matrix.Translation((0.0, 0.0, 0.0)) @ Matrix.Diagonal((0.23, 0.14, 0.27, 1.0)),
    )
    tv = set(res_t["verts"])
    for ft in bm_op.faces:
        if all(v in tv for v in ft.verts):
            ft.material_index = 8
    # Neck + Head turned slightly left toward central radar console
    res_h = bmesh.ops.create_icosphere(
        bm_op, subdivisions=3, radius=1.0,
        matrix=Matrix.Translation((-0.02, 0.03, 0.32)) @ Matrix.Diagonal((0.096, 0.110, 0.125, 1.0)),
    )
    hv = set(res_h["verts"])
    for fh in bm_op.faces:
        if all(v in hv for v in fh.verts):
            fh.material_index = 9
    # Military tactical headset earcups + headband + boom mic
    for sx in (-1.0, +1.0):
        res_ec = bmesh.ops.create_cone(
            bm_op, cap_ends=True, cap_tris=False, segments=16,
            radius1=0.044, radius2=0.038, depth=0.036,
            matrix=Matrix.Translation((-0.02 + sx * 0.102, 0.03, 0.32))
            @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        ev = set(res_ec["verts"])
        for fe in bm_op.faces:
            if all(v in ev for v in fe.verts):
                fe.material_index = 0
    # Headset arch over crown
    for k in range(8):
        a0 = math.radians(15.0 + k * 18.75)
        a1 = math.radians(15.0 + (k + 1) * 18.75)
        p0 = Vector((-0.02 + 0.108 * math.cos(a0), 0.03, 0.32 + 0.134 * math.sin(a0)))
        p1 = Vector((-0.02 + 0.108 * math.cos(a1), 0.03, 0.32 + 0.134 * math.sin(a1)))
        _add_strut(bm_op, p0, p1, 0.012, mat_index=0, segs=8)
    # Boom microphone extending forward-left
    _add_strut(bm_op, Vector((-0.115, 0.04, 0.30)), Vector((-0.075, 0.15, 0.27)), 0.006, mat_index=2, segs=6)

    op_obj = _mesh_from_bm(
        "Env_EnemyOperator", bm_op, mats, col,
        location=(0.62, -1.42, 0.96),
        rotation=(0.0, 0.0, math.radians(28.0)),
        parent=room_root,
        smooth=True,
    )
    _key_visibility(op_obj, [(1561, total_frames)], total_frames)

    # Subtle operator head-turn animation in Beat 6 looking from scope up toward window!
    for f in (1561, 1650, 1730, total_frames):
        u = C.smootherstep(1561.0, float(total_frames), float(f))
        op_obj.rotation_euler = Euler((math.radians(C.lerp(-4.0, 6.0, u)), 0.0, math.radians(C.lerp(32.0, 20.0, u))), "XYZ")
        op_obj.keyframe_insert("rotation_euler", frame=f)

    return [room_obj, op_obj]


# ==============================================================================
# 3. OUTDOOR SUNLIT MILITARY PARABOLIC RADAR TOWER (BEATS 1, 2 & 5)
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

    # Ribbed Parabolic Reflector Dish + Outer Rim + 3-Leg Tripod Feed-Horn
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
            tz = C.lerp(2.48, 1.28, C.smootherstep(1380.0, 1440.0, float(f)))
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

    # Active in Beat 1 (outside bunker window), Beat 2 (aerial contrast) AND Beat 5 (main-scene explainer!)
    for obj_v in (base_obj, dish_obj):
        _key_visibility(obj_v, [(1, 540), (1321, 1560)], total_frames)

    return root


# ==============================================================================
# 4. SUNLIT DISTANT MOUNTAIN HORIZON & MILITARY PLATEAU (NO BLUE GRID WALL!)
# ==============================================================================
def _build_daylight_horizon(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """Sunlit atmospheric mountain ridge & military plateau visible through bunker window & aerial beats."""
    mat_ridge = _mat_terrain_ridge()
    mat_pad_ground = _mat_solid("Mat_Env_BasePlateau", (0.22, 0.25, 0.24), metallic=0.05, roughness=0.88)

    bm = bmesh.new()
    nx, ny = 24, 10
    verts: list[list[bmesh.types.BMVert]] = []
    for iy in range(ny):
        ty = iy / (ny - 1)
        y = C.lerp(0.60, 14.0, ty)
        row = []
        for ix in range(nx):
            tx = (ix / (nx - 1)) * 2.0 - 1.0
            x = tx * 14.0
            # Gentle hillside base near radar tower rising into scenic distant mountain silhouette
            hill = 0.0
            if ty > 0.22:
                ridge_env = C.smoothstep(0.22, 0.72, ty)
                hill = ridge_env * (
                    0.95 * math.sin(x * 0.42 + 0.6) ** 2
                    + 0.55 * math.cos(x * 0.85 - 0.4) ** 2
                    + 0.35 * ty
                )
            z = 0.78 - 0.18 * ty + hill
            row.append(bm.verts.new((x, y, z)))
        verts.append(row)

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            f_face = bm.faces.new((verts[iy][ix], verts[iy][ix + 1], verts[iy + 1][ix + 1], verts[iy + 1][ix]))
            f_face.material_index = 0 if iy >= 2 else 1

    ridge_obj = _mesh_from_bm("Env_DaylightRidge", bm, [mat_ridge, mat_pad_ground], col, smooth=True)
    # Visible in Beats 1 & 6 (outside panoramic bunker window); Beats 2 & 5 are high altitude above clouds
    _key_visibility(ridge_obj, [(1, 220), (1561, total_frames)], total_frames)
    return [ridge_obj]


# ==============================================================================
# 5. BRIGHT CINEMATIC DAYLIGHT SUN + SKY FILL + ANTI-GLARE BUNKER PRACTICALS
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
    Sets `visible_glossy = False` on broad front fill light so ZERO rectangular white light boxes
    ever reflect on the radar scope glass!
    """
    sun_data = bpy.data.lights.new("Light_Daylight_SunKey", type="SUN")
    sun_data.color = (1.00, 0.95, 0.86)
    sun_data.energy = 4.2
    sun_data.angle = math.radians(2.4)
    l_sun = bpy.data.objects.new("Light_Daylight_SunKey", sun_data)
    l_sun.location = Vector((4.0, -2.5, 6.5))
    sun_aim = (Vector((0.0, 0.2, 1.8)) - l_sun.location).normalized()
    l_sun.rotation_euler = sun_aim.to_track_quat("-Z", "Y").to_euler("XYZ")
    col.objects.link(l_sun)

    l_rim = _add_area_light(
        "Light_SkyRim_Top", col,
        location=(2.8, 1.6, 5.4), target=(0.0, 0.0, 1.95),
        color=(0.76, 0.90, 1.00), energy=640.0, size=2.6, size_y=1.8,
        glossy_vis=True,
    )
    l_fill = _add_area_light(
        "Light_SkyFill_AntiGlare", col,
        location=(-1.0, -4.6, 2.2), target=(0.0, 0.0, 1.85),
        color=(0.82, 0.91, 1.00), energy=280.0, size=3.2, size_y=2.4,
        glossy_vis=False,  # Prevents ugly white square reflections on radar CRT scope!
    )
    l_macro = _add_area_light(
        "Light_MacroPrecision_Key", col,
        location=(0.95, -1.65, 2.85), target=(0.0, 0.0, 1.85),
        color=(0.98, 0.97, 0.94), energy=190.0, size=0.95, size_y=0.70,
        glossy_vis=True,
    )
    # Subtle green-amber interior practical lighting the Enemy Command Room desk & operator
    l_bunker = _add_area_light(
        "Light_CmdBunker_DeskGlow", col,
        location=(0.15, -1.35, 1.65), target=(0.20, -1.10, 0.95),
        color=(0.24, 0.92, 0.64), energy=22.0, size=0.85, size_y=0.45,
        glossy_vis=False,
    )

    key_frames = [1, 220, 221, 540, 541, 960, 961, 1320, 1321, 1560, 1561, total_frames]
    for f in key_frames:
        is_bunker = (f <= 220) or (f >= 1561)
        is_macro = 541 <= f <= 1320
        l_sun.data.energy = 4.6 if not is_macro else 3.2
        l_rim.data.energy = 680.0 if not is_macro else 520.0
        l_fill.data.energy = 240.0 if is_bunker else (310.0 if not is_macro else 210.0)
        l_macro.data.energy = 210.0 if is_macro else 0.0
        l_bunker.data.energy = 26.0 if is_bunker else 0.0
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
    """Assemble daylight sky world, Enemy Radar Command Room, outdoor radar tower, mountain horizon & lights."""
    world = _setup_world(total_frames)
    cmd_room = _build_enemy_command_room(col, total_frames)
    radar_station = _build_ground_radar_station(col, total_frames)
    horizon_objs = _build_daylight_horizon(col, total_frames)
    lights = _build_lighting(col, total_frames)
    return {
        "world": world,
        "command_room": cmd_room,
        "radar_station": radar_station,
        "horizon_objs": horizon_objs,
        "lights": lights,
    }
