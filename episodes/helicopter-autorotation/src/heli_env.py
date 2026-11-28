"""
heli_env.py — Golden-Hour City Skyline, Sunset Sky Dome, Skyscraper Rooftop Helipad & V12 Lighting
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Features:
  - Procedural Golden-Hour Sunset World (`ShaderNodeTexSky` / rich amber-to-cobalt atmospheric dome)
  - 28 sunlit glass-and-concrete city skyline towers (`Env_Skyline_Towers`) with warm reflective curtain walls,
    glowing dusk office strips, aviation beacons & golden cloud banks (`Env_Sunset_Clouds`)
  - Detailed Skyscraper Rooftop Helipad (`Env_Rooftop_Helipad`, Beats 5B & 6, deck strictly at Z = 0.90 m!)
    with high-contrast yellow `H` aiming circle, 12 glowing green/amber perimeter beacons, safety net rails,
    rooftop stairwell penthouse, and animated fluttering rescue windsock
  - V12 Cinematic Lighting: warm low-elevation golden key sun (18 deg), cool cobalt sky rim bounce,
    0.42 m Softbox Cockpit Eye Catchlight (`Light_Crew_EyeCatchlight`), and pulsing crimson `ENG OUT`
    cockpit warning bounce light (`Light_Cockpit_CautionGlow`)
"""
from __future__ import annotations

import math
import random
from typing import Any

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]

import heli_contract as C


def _link(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mats: list[bpy.types.Material],
    matrix: Matrix,
    col: bpy.types.Collection,
    smooth: bool = False,
    no_shadow: bool = False,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    # Blender 5.3.0 rule: append ALL materials BEFORE bm.to_mesh(mesh)!
    for m in mats:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = matrix
    if no_shadow:
        obj.visible_shadow = False
    _link(obj, col)
    return obj


def _set_constant_hide(obj: bpy.types.Object, windows: list[tuple[int, int]], total_frames: int) -> None:
    def _visible(f: int) -> bool:
        return any(a <= f <= b for a, b in windows)

    frames = {1, total_frames}
    for a, b in windows:
        for f in (a - 1, a, b, b + 1):
            if 1 <= f <= total_frames:
                frames.add(f)

    for f in sorted(frames):
        hide = not _visible(f)
        obj.hide_render = hide
        obj.hide_viewport = hide
        obj.keyframe_insert("hide_render", frame=f)
        obj.keyframe_insert("hide_viewport", frame=f)

    ad = obj.animation_data
    if ad and ad.action:
        for fc in C.iter_action_fcurves(ad.action):
            if fc.data_path.startswith("hide_"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _make_pbr(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    coat: float = 0.0,
    emit_rgb: tuple[float, float, float] | None = None,
    emit_str: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if coat > 0.0 and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.08
    if emit_rgb is not None and emit_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    return mat


def _build_golden_hour_world() -> bpy.types.World:
    """Rich procedural golden-hour sunset sky: warm amber horizon -> peach gold -> crisp cobalt zenith."""
    world = bpy.data.worlds.new("World_GoldenHour_Skyline")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.35

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    # Vertical gradient Z in [-0.15 .. +0.65]
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = -0.10
    mr.inputs["From Max"].default_value = 0.62
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "B_SPLINE"
    cr.elements[0].position = 0.00
    cr.elements[0].color = (0.18, 0.14, 0.16, 1.0)     # warm urban haze below horizon
    cr.elements[1].position = 0.18
    cr.elements[1].color = (1.00, 0.50, 0.22, 1.0)     # radiant golden-hour sunset horizon
    e2 = cr.elements.new(0.36)
    e2.color = (0.98, 0.68, 0.42, 1.0)                 # soft peach-gold mid-sky
    e3 = cr.elements.new(0.62)
    e3.color = (0.30, 0.50, 0.78, 1.0)                 # crisp golden-hour azure
    e4 = cr.elements.new(1.00)
    e4.color = (0.11, 0.24, 0.52, 1.0)                 # deep cobalt zenith
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])

    # Warm sun-glow lobe in the sky (+X, +Y, low elevation)
    dot_sun = nt.nodes.new("ShaderNodeVectorMath")
    dot_sun.operation = "DOT_PRODUCT"
    sun_dir = Vector((0.46, 0.84, 0.28)).normalized()
    dot_sun.inputs[1].default_value = (sun_dir.x, sun_dir.y, sun_dir.z)
    nt.links.new(tc.outputs["Generated"], dot_sun.inputs[0])

    mr_sun = nt.nodes.new("ShaderNodeMapRange")
    mr_sun.clamp = True
    mr_sun.inputs["From Min"].default_value = 0.88
    mr_sun.inputs["From Max"].default_value = 0.998
    mr_sun.inputs["To Min"].default_value = 0.0
    mr_sun.inputs["To Max"].default_value = 1.0
    nt.links.new(dot_sun.outputs["Value"], mr_sun.inputs["Value"])

    pow_sun = nt.nodes.new("ShaderNodeMath")
    pow_sun.operation = "POWER"
    pow_sun.inputs[1].default_value = 3.2
    nt.links.new(mr_sun.outputs["Result"], pow_sun.inputs[0])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    fac_s, col_a, col_b = C.get_mix_sockets(mix_col)
    col_b.default_value = (1.00, 0.82, 0.50, 1.0)
    nt.links.new(pow_sun.outputs["Value"], fac_s)
    nt.links.new(ramp.outputs["Color"], col_a)
    nt.links.new(mix_col.outputs[2], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world


def _add_box_bm(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_idx: int = 0,
    bevel_w: float = 0.0,
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_w > 0.0:
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)
    rm = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ").to_matrix().to_4x4()
    bmesh.ops.transform(tmp, matrix=Matrix.Translation(Vector(center)) @ rm, verts=tmp.verts)
    vmap = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vmap[v] for v in f.verts])
        nf.material_index = mat_idx
    tmp.free()


def _build_skyline_towers(col: bpy.types.Collection) -> list[bpy.types.Object]:
    """28 golden-hour glass & steel skyscrapers framing the sunset horizon (visible in all beats)."""
    mat_glass_blue = _make_pbr("Mat_City_GlassBlue", (0.14, 0.24, 0.36), metallic=0.72, roughness=0.14, coat=0.6)
    mat_glass_gold = _make_pbr("Mat_City_GlassGold", (0.34, 0.26, 0.20), metallic=0.68, roughness=0.16, coat=0.5)
    mat_concrete = _make_pbr("Mat_City_Concrete", (0.28, 0.26, 0.25), metallic=0.05, roughness=0.72)
    mat_win_glow = _make_pbr(
        "Mat_City_WarmWin", (1.00, 0.78, 0.45), roughness=0.25,
        emit_rgb=(1.00, 0.76, 0.40), emit_str=1.25,
    )
    mat_beacon = _make_pbr(
        "Mat_City_BeaconRed", (0.95, 0.14, 0.08), roughness=0.2,
        emit_rgb=(1.00, 0.18, 0.10), emit_str=1.80,
    )

    rng = random.Random(505)
    bm = bmesh.new()

    # Foreground-mid skyline ring at Y in [+13..+42], leaving center corridor clear for helicopter flights
    tower_specs = [
        (-16.0, 22.0, 4.2, 4.2, 24.0),
        (-11.8, 18.5, 3.6, 3.4, 19.5),
        (-8.4, 24.5, 3.8, 3.8, 28.0),
        (-5.2, 28.0, 3.4, 3.4, 22.5),
        (-2.2, 34.0, 4.0, 4.0, 30.0),
        (1.6, 31.0, 3.5, 3.5, 25.5),
        (5.4, 23.5, 3.8, 3.6, 27.0),
        (9.2, 19.0, 3.6, 3.4, 21.0),
        (13.2, 24.0, 4.2, 4.0, 26.5),
        (17.5, 28.0, 4.6, 4.2, 23.0),
        (-19.5, 32.0, 4.8, 4.5, 29.0),
        (-13.6, 30.0, 3.8, 3.8, 31.5),
        (-7.0, 36.0, 4.2, 4.0, 33.0),
        (3.8, 38.0, 4.5, 4.2, 34.5),
        (11.4, 34.0, 4.0, 4.0, 30.0),
        (20.5, 36.0, 5.0, 4.6, 28.0),
        # Lower mid-rise city deck beneath the skyline so there is no empty void
        (-9.0, 14.5, 4.5, 3.2, 11.5),
        (-4.5, 16.0, 4.0, 3.2, 10.2),
        (4.8, 15.5, 4.2, 3.2, 11.8),
        (9.8, 14.5, 4.6, 3.4, 10.6),
    ]
    # Also add distant warm skyline silhouettes behind the camera (+Y negative) for Beat 1/5A reflections
    for tx, ty, sx, sy, sh in tower_specs:
        base_z = -17.0
        cz = base_z + 0.5 * sh
        mat_idx = 0 if rng.random() < 0.58 else 1
        _add_box_bm(bm, (tx, ty, cz), (sx, sy, sh), mat_idx=mat_idx)
        # Crown parapet / mechanical penthouse
        _add_box_bm(bm, (tx, ty, base_z + sh + 0.45), (sx * 0.72, sy * 0.72, 0.90), mat_idx=2)
        # Warm glowing golden-hour window bands across upper floors
        for k in range(4):
            wz = base_z + sh - 1.4 - k * 2.1
            if wz > -4.0:
                _add_box_bm(bm, (tx, ty - sy * 0.505, wz), (sx * 0.84, 0.04, 0.36), mat_idx=3)
        # Rooftop aviation red warning beacon on tallest spires
        if sh >= 24.0:
            _add_box_bm(bm, (tx, ty, base_z + sh + 1.45), (0.14, 0.14, 1.30), mat_idx=2)
            _add_box_bm(bm, (tx, ty, base_z + sh + 2.18), (0.28, 0.28, 0.28), mat_idx=4)

    # Distant golden city ground plane far below (Z = -17 m)
    _add_box_bm(bm, (0.0, 24.0, -17.4), (90.0, 70.0, 0.8), mat_idx=2)

    towers_obj = _new_bmesh_obj(
        "Env_Skyline_Towers", bm,
        [mat_glass_blue, mat_glass_gold, mat_concrete, mat_win_glow, mat_beacon],
        Matrix.Identity(4), col, smooth=False,
    )

    # Soft golden-hour cloud puffs drifting far behind skyline
    mat_cloud = _make_pbr("Mat_City_SunsetCloud", (1.00, 0.74, 0.52), roughness=0.92, emit_rgb=(1.00, 0.66, 0.38), emit_str=0.32)
    bm_c = bmesh.new()
    cloud_locs = [
        (-14.0, 44.0, 12.5, (5.5, 2.2, 1.4)),
        (-5.0, 48.0, 15.0, (6.8, 2.4, 1.6)),
        (6.5, 45.0, 13.5, (6.0, 2.2, 1.5)),
        (16.0, 46.0, 11.8, (5.2, 2.0, 1.3)),
    ]
    for cx, cy, cz, sc in cloud_locs:
        for dx, dz, rfac in ((-0.35, -0.1, 0.75), (0.0, 0.12, 1.0), (0.38, -0.06, 0.8)):
            res = bmesh.ops.create_icosphere(
                bm_c, subdivisions=2, radius=1.0,
                matrix=Matrix.Translation((cx + dx * sc[0], cy, cz + dz * sc[2]))
                @ Matrix.Diagonal((sc[0] * 0.42 * rfac, sc[1] * rfac, sc[2] * rfac, 1.0)),
            )
            vs = set(res["verts"])
            for f in bm_c.faces:
                if all(v in vs for v in f.verts):
                    f.material_index = 0
    clouds_obj = _new_bmesh_obj(
        "Env_Sunset_Clouds", bm_c, [mat_cloud], Matrix.Identity(4), col, smooth=True, no_shadow=True,
    )
    return [towers_obj, clouds_obj]


def _build_rooftop_helipad(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Skyscraper Rooftop Helipad strictly at Z = 0.90 m (visible Beats 5B & 6, f=1451..1800).
    Includes weathered concrete pad, crisp yellow `H` aiming circle, 12 glowing green/amber
    perimeter beacons, corner windsock pole + fluttering orange/white windsock, and safety rail.
    """
    hx, hy, hz = C.STAGE["helipad_center"]  # (0.08, 0.45, 0.90)
    pad_r = float(C.STAGE["helipad_radius"])

    mat_conc = _make_pbr("Mat_Pad_Concrete", (0.20, 0.21, 0.23), metallic=0.04, roughness=0.78)
    mat_dark_trim = _make_pbr("Mat_Pad_DarkSteel", (0.08, 0.09, 0.11), metallic=0.75, roughness=0.38)
    mat_yellow = _make_pbr(
        "Mat_Pad_YellowPaint", (0.96, 0.74, 0.08), metallic=0.02, roughness=0.42,
        emit_rgb=(1.00, 0.78, 0.12), emit_str=0.18,
    )
    mat_white_paint = _make_pbr("Mat_Pad_WhitePaint", (0.92, 0.93, 0.94), roughness=0.45)
    mat_green_led = _make_pbr(
        "Mat_Pad_GreenBeacon", (0.12, 0.95, 0.45), roughness=0.18,
        emit_rgb=(0.15, 1.00, 0.50), emit_str=1.75,
    )
    mat_orange_sock = _make_pbr("Mat_Pad_WindsockOrange", (0.96, 0.28, 0.05), roughness=0.58)

    bm = bmesh.new()
    # 1. Skyscraper rooftop deck block (top face strictly at Z = hz = 0.90 m!)
    _add_box_bm(bm, (hx, hy + 0.35, hz - 3.60), (6.40, 6.20, 7.20), mat_idx=0, bevel_w=0.04)
    # Steel octagonal helipad rim collar flush with deck
    res_pad = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=32,
        radius1=pad_r + 0.14, radius2=pad_r + 0.10, depth=0.08,
        matrix=Matrix.Translation((hx, hy, hz - 0.042)),
    )
    vset = set(res_pad["verts"])
    for f in bm.faces:
        if all(v in vset for v in f.verts):
            f.material_index = 1

    # 2. High-contrast Yellow Aiming Circle (segmented flat ring at Z = hz + 0.004 m)
    n_ring = 36
    r_in, r_out = 1.38, 1.54
    for i in range(n_ring):
        a0 = 2.0 * math.pi * i / n_ring
        a1 = 2.0 * math.pi * (i + 1) / n_ring
        p0 = Vector((hx + r_in * math.cos(a0), hy + r_in * math.sin(a0), hz + 0.004))
        p1 = Vector((hx + r_out * math.cos(a0), hy + r_out * math.sin(a0), hz + 0.004))
        p2 = Vector((hx + r_out * math.cos(a1), hy + r_out * math.sin(a1), hz + 0.004))
        p3 = Vector((hx + r_in * math.cos(a1), hy + r_in * math.sin(a1), hz + 0.004))
        nf = bm.faces.new([bm.verts.new(p) for p in (p0, p1, p2, p3)])
        nf.material_index = 2

    # 3. Bold Yellow "H" marking centered on helipad (Z = hz + 0.005 m)
    _add_box_bm(bm, (hx - 0.50, hy, hz + 0.005), (0.20, 1.38, 0.006), mat_idx=2)
    _add_box_bm(bm, (hx + 0.50, hy, hz + 0.005), (0.20, 1.38, 0.006), mat_idx=2)
    _add_box_bm(bm, (hx, hy, hz + 0.005), (0.86, 0.22, 0.006), mat_idx=2)

    # 4. 12 Glowing Green Flush Perimeter Helipad Beacons
    for i in range(12):
        ang = 2.0 * math.pi * i / 12.0
        bx = hx + (pad_r - 0.10) * math.cos(ang)
        by = hy + (pad_r - 0.10) * math.sin(ang)
        _add_box_bm(bm, (bx, by, hz + 0.024), (0.10, 0.10, 0.048), mat_idx=1, bevel_w=0.008)
        _add_box_bm(bm, (bx, by, hz + 0.054), (0.068, 0.068, 0.026), mat_idx=4, bevel_w=0.004)

    # 5. Rear safety parapet + HVAC stairwell tucked cleanly behind pad (+Y rear-left)
    _add_box_bm(bm, (hx - 2.15, hy + 2.65, hz + 0.55), (1.75, 1.15, 1.10), mat_idx=0, bevel_w=0.03)
    _add_box_bm(bm, (hx + 2.45, hy + 2.35, hz + 0.20), (0.85, 0.85, 0.40), mat_idx=1, bevel_w=0.02)
    # Windsock mast on far rear-right corner (never blocks hero sightline!)
    pole_x, pole_y = hx + 2.25, hy + 1.85
    _add_box_bm(bm, (pole_x, pole_y, hz + 0.90), (0.05, 0.05, 1.80), mat_idx=3)

    pad_obj = _new_bmesh_obj(
        "Env_Rooftop_Helipad", bm,
        [mat_conc, mat_dark_trim, mat_yellow, mat_white_paint, mat_green_led],
        Matrix.Identity(4), col, smooth=False,
    )
    _set_constant_hide(pad_obj, [(C.B5_EXTERIOR_START, total_frames)], total_frames)

    # Animated striped orange/white windsock reacting strongly to Beat 5B/6 rotor downwash!
    bm_ws = bmesh.new()
    n_seg = 5
    for i in range(n_seg):
        x0 = i * 0.13
        r0 = 0.095 - i * 0.011
        r1 = 0.095 - (i + 1) * 0.011
        res_w = bmesh.ops.create_cone(
            bm_ws, cap_ends=False, segments=14,
            radius1=r0, radius2=r1, depth=0.13,
            matrix=Matrix.Translation((x0 + 0.065, 0.0, 0.0))
            @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        vw = set(res_w["verts"])
        for f in bm_ws.faces:
            if all(v in vw for v in f.verts):
                f.material_index = 0 if (i % 2 == 0) else 1

    sock_obj = _new_bmesh_obj(
        "Env_Helipad_Windsock", bm_ws, [mat_orange_sock, mat_white_paint],
        Matrix.Translation((pole_x + 0.04, pole_y, hz + 1.72)), col, smooth=True,
    )
    _set_constant_hide(sock_obj, [(C.B5_EXTERIOR_START, total_frames)], total_frames)
    for f in range(C.B5_EXTERIOR_START, total_frames + 1, 8):
        wash = C.smoothstep(1460.0, 1545.0, float(f)) * (1.0 - 0.65 * C.smoothstep(1700.0, 1790.0, float(f)))
        flutter = math.sin(f * 0.42) * (10.0 + 14.0 * wash)
        droop = C.lerp(-28.0, -4.0, wash)
        sock_obj.rotation_euler = Euler((0.0, math.radians(-droop), math.radians(-22.0 + flutter)), "XYZ")
        sock_obj.keyframe_insert("rotation_euler", frame=f)

    return [pad_obj, sock_obj]


def _build_lighting(col: bpy.types.Collection, total_frames: int) -> dict[str, bpy.types.Object]:
    """Golden-hour aviation sunlight + cool sky rim + V12 0.42m Softbox Cockpit Eye Catchlight."""
    lights: dict[str, bpy.types.Object] = {}

    # 1. Warm Golden-Hour Key Sun (18 deg low elevation so white-and-red rescue hull gleams!)
    sun_d = bpy.data.lights.new("Light_GoldenSun_Data", type="SUN")
    sun_d.energy = 3.2
    sun_d.color = (1.00, 0.84, 0.64)
    sun_d.angle = math.radians(1.6)
    sun_o = bpy.data.objects.new("Light_GoldenSun", sun_d)
    sun_o.rotation_euler = Euler((math.radians(70.0), math.radians(-14.0), math.radians(-38.0)), "XYZ")
    _link(sun_o, col)
    lights["sun"] = sun_o

    # 2. Cool Cobalt Sky Rim Light (high-contrast aviation edge separation from opposite side!)
    rim_d = bpy.data.lights.new("Light_SkyRim_Data", type="AREA")
    rim_d.shape = "RECTANGLE"
    rim_d.size = 4.2
    rim_d.size_y = 2.6
    rim_d.energy = 260.0
    rim_d.color = (0.58, 0.80, 1.00)
    rim_o = bpy.data.objects.new("Light_SkyRim", rim_d)
    rim_o.location = Vector((-4.2, 3.6, 4.4))
    rim_o.rotation_euler = (Vector((0.0, 0.0, 1.9)) - rim_o.location).to_track_quat("-Z", "Y").to_euler()
    _link(rim_o, col)
    lights["sky_rim"] = rim_o

    # 3. Warm City Bounce Fill from below (illuminates helicopter belly & rooftop skids cleanly)
    fill_d = bpy.data.lights.new("Light_CityBounce_Data", type="AREA")
    fill_d.shape = "RECTANGLE"
    fill_d.size = 4.8
    fill_d.size_y = 3.2
    fill_d.energy = 115.0
    fill_d.color = (1.00, 0.72, 0.48)
    fill_o = bpy.data.objects.new("Light_CityBounce", fill_d)
    fill_o.location = Vector((1.8, -3.2, 0.2))
    fill_o.rotation_euler = (Vector((0.0, 0.3, 1.85)) - fill_o.location).to_track_quat("-Z", "Y").to_euler()
    _link(fill_o, col)
    lights["city_bounce"] = fill_o

    # 4. V12 Softbox Cockpit Eye Catchlight (0.42 m AREA, NEVER tiny POINT! Beats 1 & 5A)
    eye_d = bpy.data.lights.new("Light_Crew_EyeCatchlight_Data", type="AREA")
    eye_d.shape = "SQUARE"
    eye_d.size = 0.42
    eye_d.energy = 54.0
    eye_d.color = (1.00, 0.94, 0.86)
    if hasattr(eye_d, "diffuse_factor"):
        eye_d.diffuse_factor = 0.22
    if hasattr(eye_d, "specular_factor"):
        eye_d.specular_factor = 1.78
    eye_o = bpy.data.objects.new("Light_Crew_EyeCatchlight", eye_d)
    eye_o.location = Vector((0.00, -0.92, 1.46))
    eye_o.rotation_euler = (Vector((0.00, 0.34, float(C.CREW["pilot_eye_z"]))) - eye_o.location).to_track_quat("-Z", "Y").to_euler()
    _link(eye_o, col)
    _set_constant_hide(eye_o, [(1, 240), (1291, C.B5_COCKPIT_END)], total_frames)
    lights["eye_catchlight"] = eye_o

    # 5. Flashing Crimson ENG OUT Cockpit Caution Light (Beat 1 urgent red face bounce!)
    warn_d = bpy.data.lights.new("Light_Cockpit_CautionGlow_Data", type="AREA")
    warn_d.shape = "RECTANGLE"
    warn_d.size = 0.48
    warn_d.size_y = 0.14
    warn_d.color = (1.00, 0.12, 0.06)
    warn_o = bpy.data.objects.new("Light_Cockpit_CautionGlow", warn_d)
    warn_o.location = Vector((0.00, -0.26, 1.15))
    warn_o.rotation_euler = (Vector((0.00, 0.34, 1.38)) - warn_o.location).to_track_quat("-Z", "Y").to_euler()
    _link(warn_o, col)
    _set_constant_hide(warn_o, [(1, 240), (1291, C.B5_COCKPIT_END)], total_frames)
    for f in range(1, 241, 6):
        pulse = 0.5 + 0.5 * math.sin((f - 14) * 0.42) if f >= 14 else 0.0
        warn_d.energy = 2.0 + 18.0 * pulse
        warn_d.keyframe_insert("energy", frame=f)
    # Subdued steady amber-red instrument glow during Beat 5A final flare
    warn_d.energy = 4.5
    warn_d.keyframe_insert("energy", frame=1291)
    warn_d.keyframe_insert("energy", frame=C.B5_COCKPIT_END)
    lights["caution_glow"] = warn_o

    return lights


def build_heli_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    world = _build_golden_hour_world()
    skyline = _build_skyline_towers(col)
    helipad = _build_rooftop_helipad(col, total_frames)
    lights = _build_lighting(col, total_frames)
    return {
        "world": world,
        "skyline": skyline,
        "helipad": helipad,
        "lights": lights,
    }
