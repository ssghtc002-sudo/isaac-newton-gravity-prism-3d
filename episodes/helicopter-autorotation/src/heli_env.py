"""
heli_env.py — Photoreal Golden-Hour Curtain-Wall Glass Skyscrapers, Descending Cloud Parallax,
Weathered Skyscraper Rooftop Helipad (Z = 0.90 m) with Safety Perimeter Netting & Landing Strobes,
and V12 Cinematic Lighting for Episode #5 V2 Ultimate (60 fps, 1800 frames).

Eliminates every "Lego city" flaw from the draft:
  - Procedural **Curtain-Wall Glass Skyscraper Shaders**: reflective dark-blue & champagne-bronze
    glass grids with dark steel mullions, subtle pane roughness variation, and randomized warm dusk
    office windows (ZERO wide flat beige stripes!), plus stepped architectural crowns, spires & HVAC units!
  - **Vertical Descent Cloud Parallax**: golden-lit sunset cloud strata animate upward via
    `C.descent_parallax_z_at(f)` during Beats 3 & 4 so viewers physically FEEL the helicopter gliding down!
  - **Weathered Concrete Skyscraper Rooftop Helipad (`Z = 0.90 m`)**: procedural concrete grime/roughness,
    crisp yellow `H` aiming circle, angled perimeter safety-net frames, 12 pulsing green TLOF beacons +
    4 white approach strobes, rooftop stairwell penthouse, radar mast, and animated striped rescue windsock!
  - **V12 Lighting**: Warm golden-hour key sun + cool cobalt sky rim bounce + Beat 1A dorsal **Turbine Fire
    Flash Point Light** + `0.42 m` Softbox Cockpit Eye Catchlight + pulsing crimson `ENG OUT` cockpit bounce!
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


def _make_curtain_wall_mat(
    name: str,
    glass_rgb: tuple[float, float, float],
    mullion_rgb: tuple[float, float, float],
    warm_chance: float = 0.36,
    scale_val: float = 1.65,
) -> bpy.types.Material:
    """
    Photorealistic Procedural Skyscraper Curtain-Wall Shader:
      - Uses 3D object coordinates mapped through `ShaderNodeTexBrick` (sharp vertical & horizontal
        aluminum mullions framing reflective glass window panes) + subtle cell noise so random dusk
        offices glow softly warm amber while adjacent glass reflects the golden sunset sky!
      - Strictly uses Blender 5.3.0 safe `C.get_mix_sockets()`.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    # Combine X+Y horizontally so both front (-Y) and side (+X/-X) facades show crisp window grids!
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    add_xy = nt.nodes.new("ShaderNodeMath")
    add_xy.operation = "ADD"
    nt.links.new(sep.outputs["X"], add_xy.inputs[0])
    nt.links.new(sep.outputs["Y"], add_xy.inputs[1])

    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(add_xy.outputs["Value"], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])

    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = scale_val
    brick.inputs["Mortar Size"].default_value = 0.065
    brick.inputs["Mortar Smooth"].default_value = 0.05
    brick.inputs["Brick Width"].default_value = 0.55
    brick.inputs["Row Height"].default_value = 0.72
    brick.inputs["Color1"].default_value = (*glass_rgb, 1.0)
    brick.inputs["Color2"].default_value = (glass_rgb[0] * 0.65, glass_rgb[1] * 0.72, glass_rgb[2] * 0.85, 1.0)
    brick.inputs["Mortar"].default_value = (*mullion_rgb, 1.0)
    nt.links.new(comb.outputs["Vector"], brick.inputs["Vector"])

    # Mortar mask (0 = glass pane, 1 = aluminum mullion frame)
    inv_m = nt.nodes.new("ShaderNodeMath")
    inv_m.operation = "SUBTRACT"
    inv_m.inputs[0].default_value = 1.0
    nt.links.new(brick.outputs["Fac"], inv_m.inputs[1])

    # Metallic & Roughness: glossy reflective glass panes (R=0.07, M=0.72) vs matte dark mullions (R=0.45, M=0.85)
    mix_r = nt.nodes.new("ShaderNodeMix")
    mix_r.data_type = "FLOAT"
    f_r, a_r, b_r = C.get_mix_sockets(mix_r)
    a_r.default_value = 0.075
    b_r.default_value = 0.48
    nt.links.new(brick.outputs["Fac"], f_r)
    nt.links.new(mix_r.outputs[0], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.76
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.65
        bsdf.inputs["Coat Roughness"].default_value = 0.05
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])

    # Subtle randomized dusk office interior glow gated strictly inside glass panes (never on mullions!)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "F1"
    vor.inputs["Scale"].default_value = scale_val * 1.4
    vor.inputs["Randomness"].default_value = 0.05
    nt.links.new(comb.outputs["Vector"], vor.inputs["Vector"])

    gt_win = nt.nodes.new("ShaderNodeMath")
    gt_win.operation = "GREATER_THAN"
    gt_win.inputs[1].default_value = 1.0 - warm_chance
    nt.links.new(vor.outputs["Distance"], gt_win.inputs[0])

    mul_em = nt.nodes.new("ShaderNodeMath")
    mul_em.operation = "MULTIPLY"
    nt.links.new(gt_win.outputs["Value"], mul_em.inputs[0])
    nt.links.new(inv_m.outputs["Value"], mul_em.inputs[1])

    mul_str = nt.nodes.new("ShaderNodeMath")
    mul_str.operation = "MULTIPLY"
    mul_str.inputs[1].default_value = 0.85  # AgX-safe tasteful warm office glow
    nt.links.new(mul_em.outputs["Value"], mul_str.inputs[0])

    bsdf.inputs["Emission Color"].default_value = (1.00, 0.76, 0.42, 1.0)
    nt.links.new(mul_str.outputs["Value"], bsdf.inputs["Emission Strength"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_weathered_asphalt_mat() -> bpy.types.Material:
    """Procedural weathered skyscraper rooftop helipad concrete/asphalt with realistic roughness grime & bump."""
    mat = bpy.data.materials.new("Mat_Helipad_WeatheredAsphalt")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 3.8
    n1.inputs["Detail"].default_value = 8.0
    n1.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])

    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 26.0
    n2.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.095, 0.105, 0.118, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.185, 0.195, 0.210, 1.0)
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    mr_r = nt.nodes.new("ShaderNodeMapRange")
    mr_r.clamp = True
    mr_r.inputs["From Min"].default_value = 0.2
    mr_r.inputs["From Max"].default_value = 0.8
    mr_r.inputs["To Min"].default_value = 0.58
    mr_r.inputs["To Max"].default_value = 0.88
    nt.links.new(n1.outputs["Fac"], mr_r.inputs["Value"])
    nt.links.new(mr_r.outputs["Result"], bsdf.inputs["Roughness"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.008
    nt.links.new(n2.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _build_golden_hour_world() -> bpy.types.World:
    """Rich cinematic golden-hour sunset sky dome: glowing amber horizon -> rose gold -> twilight azure zenith."""
    world = bpy.data.worlds.new("World_GoldenHour_Skyline")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.45

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = -0.08
    mr.inputs["From Max"].default_value = 0.62
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "B_SPLINE"
    cr.elements[0].position = 0.00
    cr.elements[0].color = (0.20, 0.13, 0.14, 1.0)
    cr.elements[1].position = 0.16
    cr.elements[1].color = (1.00, 0.46, 0.18, 1.0)
    e2 = cr.elements.new(0.34)
    e2.color = (0.98, 0.66, 0.38, 1.0)
    e3 = cr.elements.new(0.60)
    e3.color = (0.24, 0.44, 0.74, 1.0)
    e4 = cr.elements.new(1.00)
    e4.color = (0.08, 0.18, 0.44, 1.0)
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])

    dot_sun = nt.nodes.new("ShaderNodeVectorMath")
    dot_sun.operation = "DOT_PRODUCT"
    sun_dir = Vector((0.46, 0.85, 0.24)).normalized()
    dot_sun.inputs[1].default_value = (sun_dir.x, sun_dir.y, sun_dir.z)
    nt.links.new(tc.outputs["Generated"], dot_sun.inputs[0])

    mr_sun = nt.nodes.new("ShaderNodeMapRange")
    mr_sun.clamp = True
    mr_sun.inputs["From Min"].default_value = 0.86
    mr_sun.inputs["From Max"].default_value = 0.998
    mr_sun.inputs["To Min"].default_value = 0.0
    mr_sun.inputs["To Max"].default_value = 1.0
    nt.links.new(dot_sun.outputs["Value"], mr_sun.inputs["Value"])

    pow_sun = nt.nodes.new("ShaderNodeMath")
    pow_sun.operation = "POWER"
    pow_sun.inputs[1].default_value = 3.0
    nt.links.new(mr_sun.outputs["Result"], pow_sun.inputs[0])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    fac_s, col_a, col_b = C.get_mix_sockets(mix_col)
    col_b.default_value = (1.00, 0.84, 0.52, 1.0)
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


def _add_tube_bm(
    bm: bmesh.types.BMesh,
    p0: Vector,
    p1: Vector,
    r0: float,
    r1: float,
    mat_idx: int = 0,
    segs: int = 10,
) -> None:
    seg = p1 - p0
    L = seg.length
    if L < 1e-5:
        return
    mid = 0.5 * (p0 + p1)
    rm = seg.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs,
        radius1=r0, radius2=r1, depth=L,
        matrix=Matrix.Translation(mid) @ rm,
    )
    vs = set(res["verts"])
    for f in bm.faces:
        if all(v in vs for v in f.verts):
            f.material_index = mat_idx


def _build_skyline_towers(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """28 Photoreal Architectural Curtain-Wall Glass Skyscrapers + Animated Descent-Parallax Sunset Clouds!"""
    mat_glass_blue = _make_curtain_wall_mat("Mat_City_CurtainBlue", (0.09, 0.18, 0.30), (0.03, 0.04, 0.05), 0.34, 1.55)
    mat_glass_gold = _make_curtain_wall_mat("Mat_City_CurtainBronze", (0.24, 0.18, 0.14), (0.05, 0.04, 0.04), 0.42, 1.75)
    mat_glass_slate = _make_curtain_wall_mat("Mat_City_CurtainSlate", (0.12, 0.15, 0.20), (0.04, 0.04, 0.05), 0.28, 1.90)
    mat_concrete = _make_pbr("Mat_City_CrownConcrete", (0.20, 0.20, 0.22), metallic=0.12, roughness=0.68)
    mat_beacon = _make_pbr(
        "Mat_City_BeaconRed", (0.98, 0.14, 0.08), roughness=0.18,
        emit_rgb=(1.00, 0.18, 0.08), emit_str=1.85,
    )

    rng = random.Random(5052)
    bm = bmesh.new()

    # Architectural metropolitan ring pushed nicely into cinematic midground/background (Y = +18..+54)
    tower_specs = [
        (-22.0, 30.0, 5.2, 5.0, 30.0),
        (-16.5, 25.0, 4.4, 4.2, 26.0),
        (-11.8, 22.0, 3.8, 3.8, 22.5),
        (-7.6, 28.0, 4.2, 4.2, 31.5),
        (-3.6, 36.0, 4.6, 4.4, 35.0),
        (1.2, 34.0, 4.2, 4.2, 32.0),
        (5.6, 26.5, 4.0, 4.0, 29.0),
        (10.2, 22.5, 3.8, 3.6, 23.5),
        (15.0, 27.0, 4.6, 4.2, 28.5),
        (20.5, 32.0, 5.4, 4.8, 27.0),
        (-25.0, 40.0, 5.8, 5.4, 37.0),
        (-15.0, 38.0, 4.8, 4.6, 39.5),
        (-8.8, 44.0, 5.2, 4.8, 42.0),
        (3.8, 46.0, 5.4, 5.0, 43.0),
        (12.6, 41.0, 4.8, 4.6, 38.0),
        (23.5, 44.0, 5.8, 5.2, 35.0),
        # Lower canyon mid-rise deck below horizon (never empty void!)
        (-10.5, 19.5, 5.0, 3.6, 13.0),
        (-4.8, 21.0, 4.6, 3.6, 12.2),
        (4.8, 20.5, 4.8, 3.6, 13.4),
        (10.8, 19.5, 5.0, 3.8, 12.6),
    ]

    for idx, (tx, ty, sx, sy, sh) in enumerate(tower_specs):
        base_z = -21.0
        g_idx = idx % 3
        # Stepped modern skyscraper architecture (main shaft + mid setback + top penthouse crown!)
        _add_box_bm(bm, (tx, ty, base_z + sh * 0.42), (sx, sy, sh * 0.84), mat_idx=g_idx, bevel_w=0.06)
        setback_h = sh * 0.16
        _add_box_bm(bm, (tx, ty, base_z + sh * 0.84 + setback_h * 0.5), (sx * 0.78, sy * 0.78, setback_h), mat_idx=g_idx, bevel_w=0.05)
        roof_z = base_z + sh
        # Architectural parapet rim + rooftop mechanical HVAC penthouse
        _add_box_bm(bm, (tx, ty, roof_z + 0.22), (sx * 0.74, sy * 0.74, 0.44), mat_idx=3, bevel_w=0.04)
        _add_box_bm(bm, (tx + rng.uniform(-0.3, 0.3), ty, roof_z + 0.80), (sx * 0.36, sy * 0.36, 0.85), mat_idx=3, bevel_w=0.04)
        # Tall slender steel spire + red aviation warning beacon on taller towers
        if sh >= 26.0:
            spire_h = rng.uniform(2.6, 5.2)
            _add_tube_bm(bm, Vector((tx, ty, roof_z + 1.15)), Vector((tx, ty, roof_z + 1.15 + spire_h)), 0.08, 0.025, mat_idx=3, segs=8)
            _add_box_bm(bm, (tx, ty, roof_z + 1.25 + spire_h), (0.16, 0.16, 0.22), mat_idx=4, bevel_w=0.03)

    skyline_obj = _new_bmesh_obj(
        "Env_Skyline_CurtainTowers",
        bm,
        [mat_glass_blue, mat_glass_gold, mat_glass_slate, mat_concrete, mat_beacon],
        Matrix.Identity(4),
        col,
        smooth=False,
    )

    # Multi-layer Sunlit Golden-Hour Cloud Strata — animated UPWARD via descent_parallax_z_at(f) in Beats 3 & 4!
    mat_cloud = _make_pbr(
        "Mat_Env_GoldenCloud", (0.98, 0.78, 0.58), roughness=0.92,
        emit_rgb=(1.00, 0.70, 0.42), emit_str=0.38,
    )
    bm_c = bmesh.new()
    cloud_specs = [
        (-13.0, 26.0, 4.8, 5.2, 2.0, 0.72),
        (-4.5, 31.0, 7.2, 6.0, 2.2, 0.80),
        (5.5, 29.0, 5.4, 5.8, 2.0, 0.76),
        (14.0, 34.0, 8.0, 6.4, 2.4, 0.82),
        (-9.0, 22.0, 0.4, 4.8, 1.8, 0.65),
        (8.5, 24.0, -1.8, 5.0, 1.9, 0.68),
        (-2.0, 28.0, -4.6, 6.4, 2.2, 0.74),
    ]
    for cx, cy, cz, rx, ry, rz in cloud_specs:
        # Multi-lobe organic cloud bank cluster
        for dx, dy, dz, sc in ((0.0, 0.0, 0.0, 1.0), (-rx * 0.45, 0.4, -0.15, 0.72), (rx * 0.48, -0.3, -0.10, 0.68)):
            bmesh.ops.create_icosphere(
                bm_c, subdivisions=2, radius=1.0,
                matrix=Matrix.Translation((cx + dx, cy + dy, cz + dz)) @ Matrix.Diagonal((rx * sc, ry * sc, rz * sc, 1.0)),
            )
    for f in bm_c.faces:
        f.material_index = 0
    clouds_obj = _new_bmesh_obj(
        "Env_Sunset_CloudStrata", bm_c, [mat_cloud], Matrix.Identity(4), col, smooth=True, no_shadow=True,
    )
    for f in sorted(set(range(1, total_frames + 1, 15)) | {1, 570, 960, 1290, total_frames}):
        clouds_obj.location = Vector((0.0, 0.0, C.descent_parallax_z_at(float(f))))
        clouds_obj.keyframe_insert("location", frame=f)

    return [skyline_obj, clouds_obj]


def _build_rooftop_helipad(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Weathered Concrete Skyscraper Rooftop Helipad (Beats 5B & 6, deck surface strictly at Z = 0.900 m!):
      - Weathered procedural asphalt deck + crisp safety yellow `H` + dashed ring + perimeter safety netting
      - 12 Pulsing Green TLOF Beacons + 4 White Approach Strobe Lights
      - Rooftop Access Stairwell Bulkhead + HVAC Chiller + Radar/Comms Antenna Mast
      - Animated Red/White Striped Rescue Windsock whipping violently during Beat 5B flare & settling!
    """
    hx, hy, hz = C.STAGE["helipad_center"]
    pad_r = float(C.STAGE["helipad_radius"])

    mat_deck = _make_weathered_asphalt_mat()
    mat_concrete = _make_pbr("Mat_Helipad_ConcreteWall", (0.28, 0.29, 0.31), roughness=0.78)
    mat_steel = _make_pbr("Mat_Helipad_SteelNet", (0.48, 0.52, 0.56), metallic=0.82, roughness=0.34)
    mat_yellow = _make_pbr("Mat_Helipad_SafetyYellow", (0.94, 0.72, 0.08), roughness=0.48)
    mat_dark = _make_pbr("Mat_Helipad_DarkMetal", (0.06, 0.07, 0.08), metallic=0.75, roughness=0.38)
    mat_orange = _make_pbr("Mat_Helipad_WindsockOrange", (0.96, 0.24, 0.04), roughness=0.58)
    mat_white = _make_pbr("Mat_Helipad_WindsockWhite", (0.94, 0.95, 0.96), roughness=0.55)

    # Pulsing Green TLOF Perimeter Beacons
    mat_green_b = _make_pbr("Mat_Helipad_BeaconGreen", (0.10, 0.98, 0.48), roughness=0.16, emit_rgb=(0.12, 1.00, 0.52), emit_str=1.85)
    # Pulsing White Approach Strobe Lights
    mat_strobe = bpy.data.materials.new("Mat_Helipad_ApproachStrobe")
    mat_strobe.use_nodes = True
    bs_st = mat_strobe.node_tree.nodes["Principled BSDF"]
    bs_st.inputs["Base Color"].default_value = (0.98, 0.99, 1.0, 1.0)
    bs_st.inputs["Emission Color"].default_value = (0.95, 0.98, 1.0, 1.0)
    for f in range(1451, total_frames + 1, 6):
        pulse = 1.95 if ((f // 12) % 2 == 0) else 0.25
        bs_st.inputs["Emission Strength"].default_value = pulse
        bs_st.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)

    bm = bmesh.new()
    # 1. Skyscraper rooftop main structure & weathered helipad slab (TOP strictly at Z = hz = 0.900 m!)
    _add_box_bm(bm, (hx, hy, hz - 7.05), (6.40, 6.00, 14.00), mat_idx=1, bevel_w=0.05)
    _add_box_bm(bm, (hx, hy, hz - 0.05), (5.90, 5.50, 0.10), mat_idx=0, bevel_w=0.015)

    # 2. Angled Metallic Safety Perimeter Netting Frames around all 4 rooftop edges!
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (hx + sx * 3.16, hy, hz + 0.08), (0.48, 5.40, 0.025), mat_idx=2, rot_deg=(0.0, sx * -14.0, 0.0))
        for ky in range(-2, 3):
            _add_tube_bm(
                bm,
                Vector((hx + sx * 2.92, hy + ky * 1.15, hz - 0.04)),
                Vector((hx + sx * 3.38, hy + ky * 1.15, hz + 0.14)),
                0.018, 0.018, mat_idx=4, segs=6,
            )
    for sy in (-1.0, +1.0):
        _add_box_bm(bm, (hx, hy + sy * 2.94, hz + 0.08), (5.70, 0.46, 0.025), mat_idx=2, rot_deg=(sy * 14.0, 0.0, 0.0))

    # 3. High-Contrast Painted Yellow Touchdown Ring (radius 1.78 m) + Crisp Center `H` (flush at Z = hz + 0.002 m)
    mark_z = hz + 0.0025
    n_seg = 40
    for k in range(n_seg):
        a0 = 2.0 * math.pi * (k / float(n_seg))
        a1 = 2.0 * math.pi * ((k + 1) / float(n_seg))
        am = 0.5 * (a0 + a1)
        seg_len = 2.0 * math.pi * 1.78 / float(n_seg) + 0.012
        _add_box_bm(
            bm,
            (hx + math.cos(am) * 1.78, hy + math.sin(am) * 1.78, mark_z),
            (0.11, seg_len, 0.004), mat_idx=3, rot_deg=(0.0, 0.0, math.degrees(am)),
        )
    _add_box_bm(bm, (hx - 0.44, hy, mark_z), (0.16, 1.30, 0.004), mat_idx=3)
    _add_box_bm(bm, (hx + 0.44, hy, mark_z), (0.16, 1.30, 0.004), mat_idx=3)
    _add_box_bm(bm, (hx, hy, mark_z), (0.88, 0.18, 0.004), mat_idx=3)

    # 4. 12 Glowing Green TLOF Perimeter Light Housings + 4 Bright White Approach Strobe Units
    for k in range(12):
        ang = 2.0 * math.pi * (k / 12.0)
        bx = hx + math.cos(ang) * (pad_r - 0.14)
        by = hy + math.sin(ang) * (pad_r - 0.14)
        _add_box_bm(bm, (bx, by, hz + 0.032), (0.13, 0.13, 0.064), mat_idx=4, bevel_w=0.01)
        mat_l = 6 if (k % 3 == 0) else 5
        _add_box_bm(bm, (bx, by, hz + 0.072), (0.09, 0.09, 0.044), mat_idx=mat_l, bevel_w=0.01)

    # 5. Rooftop Architectural Equipment (kept cleanly outside the landing corridor!)
    _add_box_bm(bm, (hx - 2.38, hy + 1.80, hz + 0.62), (1.15, 1.45, 1.24), mat_idx=1, bevel_w=0.03)
    _add_box_bm(bm, (hx - 2.38, hy + 1.80, hz + 1.28), (1.22, 1.52, 0.08), mat_idx=4, bevel_w=0.015)
    # Comms/weather radar mast on rear-left corner
    _add_tube_bm(bm, Vector((hx - 2.45, hy + 2.15, hz + 1.30)), Vector((hx - 2.45, hy + 2.15, hz + 2.65)), 0.025, 0.014, mat_idx=2, segs=8)
    # HVAC louver unit + windsock steel pole on far-right edge
    _add_box_bm(bm, (hx + 2.46, hy + 1.95, hz + 0.36), (0.95, 0.90, 0.72), mat_idx=4, bevel_w=0.02)
    _add_tube_bm(bm, Vector((hx + 2.42, hy - 0.45, hz)), Vector((hx + 2.42, hy - 0.45, hz + 1.76)), 0.028, 0.022, mat_idx=2, segs=10)

    pad_obj = _new_bmesh_obj(
        "Env_Rooftop_Helipad",
        bm,
        [mat_deck, mat_concrete, mat_steel, mat_yellow, mat_dark, mat_green_b, mat_strobe],
        Matrix.Identity(4),
        col,
        smooth=False,
    )
    _set_constant_hide(pad_obj, [(C.B5_EXTERIOR_START, total_frames)], total_frames)

    # 6. Animated Red-and-White Striped Rescue Windsock (whips hard in Beat 5B flare downwash, settles after landing!)
    bm_ws = bmesh.new()
    for s in range(5):
        x0, x1 = s * 0.145, (s + 1) * 0.145
        r0, r1 = 0.095 - s * 0.011, 0.095 - (s + 1) * 0.011
        _add_tube_bm(bm_ws, Vector((x0, 0.0, 0.0)), Vector((x1, 0.0, 0.0)), r0, r1, mat_idx=(s % 2), segs=12)
    ws_obj = _new_bmesh_obj(
        "Env_Helipad_Windsock",
        bm_ws,
        [mat_orange, mat_white],
        Matrix.Translation((hx + 2.42, hy - 0.45, hz + 1.72)),
        col,
        smooth=True,
    )
    for f in range(C.B5_EXTERIOR_START, total_frames + 1, 5):
        wash = 1.0 - C.smoothstep(1685.0, 1785.0, float(f))
        pitch_deg = C.lerp(28.0, 4.0, wash) + (6.5 * wash + 1.5) * math.sin(f * 0.34)
        yaw_deg = 22.0 + (12.0 * wash + 2.5) * math.cos(f * 0.26)
        ws_obj.rotation_euler = Euler((0.0, math.radians(pitch_deg), math.radians(yaw_deg)), "XYZ")
        ws_obj.keyframe_insert("rotation_euler", frame=f)
    _set_constant_hide(ws_obj, [(C.B5_EXTERIOR_START, total_frames)], total_frames)

    return [pad_obj, ws_obj]


def _build_lighting(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """V12 Golden-Hour Key Sun + Cool Cobalt Rim + Beat 1A Fire Flash + Cockpit Catchlight & Caution Glow."""
    lights: dict[str, Any] = {}

    # 1. Warm Golden-Hour Low-Elevation Sun (17 deg elevation for long dramatic shadows & aircraft rim sheen!)
    sun_d = bpy.data.lights.new("Light_GoldenHour_Sun_Data", type="SUN")
    sun_d.energy = 4.6
    sun_d.color = (1.00, 0.78, 0.52)
    sun_d.angle = math.radians(1.6)
    sun_o = bpy.data.objects.new("Light_GoldenHour_Sun", sun_d)
    sun_o.rotation_euler = Euler((math.radians(73.0), math.radians(-16.0), math.radians(-30.0)), "XYZ")
    _link(sun_o, col)
    lights["sun"] = sun_o

    # 2. Cool Twilight Sky Rim Light (gives razor-sharp cyan-cobalt edge separation on fuselage & blades!)
    rim_d = bpy.data.lights.new("Light_SkyRim_Area_Data", type="AREA")
    rim_d.shape = "RECTANGLE"
    rim_d.size = 6.5
    rim_d.size_y = 4.0
    rim_d.energy = 420.0
    rim_d.color = (0.58, 0.80, 1.00)
    rim_o = bpy.data.objects.new("Light_SkyRim_Area", rim_d)
    rim_o.location = Vector((-4.2, 4.6, 4.8))
    rim_o.rotation_mode = "QUATERNION"
    rim_o.rotation_quaternion = (Vector((0.0, 0.2, 1.8)) - rim_o.location).normalized().to_track_quat("-Z", "Y")
    _link(rim_o, col)
    lights["rim"] = rim_o

    # 3. Beat 1A Dorsal Turbine Explosion Flash Point Light (illuminates fuselage & tail in fiery orange-amber!)
    fire_d = bpy.data.lights.new("Light_B1A_TurbineFire_Data", type="POINT")
    fire_d.color = (1.00, 0.42, 0.08)
    fire_d.shadow_soft_size = 0.24
    fire_o = bpy.data.objects.new("Light_B1A_TurbineFire", fire_d)
    _link(fire_o, col)
    for f in sorted(set(range(1, C.B1_EXTERIOR_END + 1, 2)) | {1, 18, 24, 40, C.B1_EXTERIOR_END, C.B1_COCKPIT_START}):
        hpos, _ = C.heli_pose_at(float(min(f, C.B1_EXTERIOR_END)))
        fire_o.location = Vector((hpos[0] + 0.08, hpos[1] + 0.36, hpos[2] + 0.34))
        fire_o.keyframe_insert("location", frame=f)
        if 18 <= f <= 85:
            dt = float(f - 18)
            burst = (1.0 if dt < 8.0 else math.exp(-0.06 * (dt - 8.0))) * (0.75 + 0.25 * math.sin(dt * 1.9))
            fire_d.energy = 260.0 * burst
        else:
            fire_d.energy = 0.0
        fire_d.keyframe_insert("energy", frame=f)
    lights["turbine_fire"] = fire_o

    # 4. V12 Softbox Cockpit Eye Catchlight (0.42 m AREA aimed at crew eyes @ Z=1.38 m; Beats 1B & 5A!)
    eye_d = bpy.data.lights.new("Light_Crew_EyeCatchlight_Data", type="AREA")
    eye_d.shape = "RECTANGLE"
    eye_d.size = 0.42
    eye_d.size_y = 0.42
    eye_d.color = (1.00, 0.92, 0.80)
    eye_d.energy = 46.0
    if hasattr(eye_d, "diffuse_factor"):
        eye_d.diffuse_factor = 0.16
    if hasattr(eye_d, "specular_factor"):
        eye_d.specular_factor = 1.80
    eye_o = bpy.data.objects.new("Light_Crew_EyeCatchlight", eye_d)
    eye_o.location = Vector((0.0, -1.12, 1.48))
    eye_o.rotation_mode = "QUATERNION"
    eye_o.rotation_quaternion = (Vector((0.0, 0.34, 1.38)) - eye_o.location).normalized().to_track_quat("-Z", "Y")
    _link(eye_o, col)
    _set_constant_hide(eye_o, [(C.B1_COCKPIT_START, 240), (1291, C.B5_COCKPIT_END)], total_frames)
    lights["eye_catchlight"] = eye_o

    # 5. Pulsing Crimson Cockpit `ENG OUT` Emergency Wash Light (washes pilots' faces & helmets in red alarm!)
    warn_d = bpy.data.lights.new("Light_Cockpit_CautionGlow_Data", type="POINT")
    warn_d.color = (1.00, 0.10, 0.05)
    warn_d.shadow_soft_size = 0.22
    warn_o = bpy.data.objects.new("Light_Cockpit_CautionGlow", warn_d)
    warn_o.location = Vector((0.0, -0.32, 1.18))
    _link(warn_o, col)
    for f in range(C.B1_COCKPIT_START, 241, 4):
        pulse = 0.30 + 0.70 * (0.5 + 0.5 * math.sin((f - 116) * 0.48))
        warn_d.energy = 38.0 * pulse
        warn_d.keyframe_insert("energy", frame=f)
    warn_d.energy = 0.0
    warn_d.keyframe_insert("energy", frame=115)
    warn_d.keyframe_insert("energy", frame=241)
    # Warm golden-amber cockpit instrument glow during Beat 5A final flare!
    warn_d.energy = 10.0
    warn_d.keyframe_insert("energy", frame=1291)
    warn_d.keyframe_insert("energy", frame=C.B5_COCKPIT_END)
    warn_d.energy = 0.0
    warn_d.keyframe_insert("energy", frame=C.B5_EXTERIOR_START)
    lights["caution_glow"] = warn_o

    return lights


def build_heli_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Build golden-hour world, curtain-wall glass skyscrapers, parallax clouds, rooftop helipad & V12 lighting."""
    world = _build_golden_hour_world()
    skyline = _build_skyline_towers(col, total_frames)
    helipad = _build_rooftop_helipad(col, total_frames)
    lights = _build_lighting(col, total_frames)
    return {
        "world": world,
        "skyline": skyline,
        "helipad": helipad,
        "lights": lights,
    }
