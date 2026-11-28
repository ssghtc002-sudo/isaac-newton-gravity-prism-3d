"""
Environment & Lighting Rig for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Builds:
  1. Dark Night-Command Tactical Sky World + Atmospheric Horizon Dome (Beats 1-4, 6)
  2. Ground Military Radar Station & Rotating Parabolic Dish Emitter (Beats 1-2)
  3. Clean @zackdfilms Cyan-Blue Studio CAD Blueprint Backdrop (Z=-0.15..8.80m) &
     Extended Grid Floor (X=-7..+7m, Y=-6..+6m covering past camera) for Beat 5
  4. Multi-point Moonlight Rim, Underbelly Horizon Bounce, Macro X-Ray & Blueprint Softbox Lights

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
) -> None:
    sx, sy, sz = size
    mat = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    res = bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    verts = res["verts"]
    vset = set(verts)
    faces = [f for f in bm.faces if all(v in vset for v in f.verts)]
    for f in faces:
        f.material_index = mat_index
    if bevel_w > 0.0:
        edges = [e for e in bm.edges if e.verts[0] in vset and e.verts[1] in vset]
        bmesh.ops.bevel(bm, geom=edges, offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)


def _setup_world(total_frames: int) -> bpy.types.World:
    """Deep cinematic night-command tactical sky smoothly shifting to CAD cyan-navy in Beat 5."""
    world = bpy.data.worlds.new("World_StealthAirspace")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tex_coord.outputs["Generated"], sep.inputs["Vector"])

    # Night sky vertical horizon gradient (deep midnight navy zenith -> subtle radar-blue horizon haze)
    ramp_night = nt.nodes.new("ShaderNodeValToRGB")
    ramp_night.color_ramp.elements[0].position = 0.0
    ramp_night.color_ramp.elements[0].color = (0.008, 0.016, 0.032, 1.0)
    e_mid = ramp_night.color_ramp.elements.new(0.42)
    e_mid.color = (0.012, 0.030, 0.062, 1.0)
    ramp_night.color_ramp.elements[-1].position = 0.85
    ramp_night.color_ramp.elements[-1].color = (0.003, 0.007, 0.016, 1.0)
    nt.links.new(sep.outputs["Z"], ramp_night.inputs["Fac"])

    # Subtle moonlit atmospheric cloud texture so night sky looks rich, never flat plastic black
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.2
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.58
    nt.links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])

    cloud_ramp = nt.nodes.new("ShaderNodeValToRGB")
    cloud_ramp.color_ramp.elements[0].position = 0.42
    cloud_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cloud_ramp.color_ramp.elements[1].position = 0.78
    cloud_ramp.color_ramp.elements[1].color = (0.014, 0.036, 0.068, 1.0)
    nt.links.new(noise.outputs["Fac"], cloud_ramp.inputs["Fac"])

    mix_sky = nt.nodes.new("ShaderNodeMix")
    mix_sky.data_type = "RGBA"
    mix_sky.blend_type = "ADD"
    f_s, a_s, b_s = C.get_mix_sockets(mix_sky)
    f_s.default_value = 0.75
    nt.links.new(ramp_night.outputs["Color"], a_s)
    nt.links.new(cloud_ramp.outputs["Color"], b_s)

    # Mix with clean Studio CAD Blueprint color during Beat 5
    bp_val = nt.nodes.new("ShaderNodeValue")
    bp_val.name = "Blueprint_World_Mix"
    mix_bp = nt.nodes.new("ShaderNodeMix")
    mix_bp.data_type = "RGBA"
    f_b, a_b, b_b = C.get_mix_sockets(mix_bp)
    nt.links.new(bp_val.outputs[0], f_b)
    nt.links.new(mix_sky.outputs[2], a_b)
    b_b.default_value = (0.010, 0.048, 0.110, 1.0)

    nt.links.new(mix_bp.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.85
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    for f in range(1, total_frames + 1, 10):
        w = C.blueprint_bg_weight(float(f))
        bp_val.outputs[0].default_value = w
        bp_val.outputs[0].keyframe_insert("default_value", frame=f)
        bg.inputs["Strength"].default_value = C.lerp(0.85, 1.15, w)
        bg.inputs["Strength"].keyframe_insert("default_value", frame=f)
    bp_val.outputs[0].default_value = 0.0
    bp_val.outputs[0].keyframe_insert("default_value", frame=total_frames)
    return world


def _mat_blueprint_grid(name: str, is_floor: bool = False) -> bpy.types.Material:
    """Clean @zackdfilms Cyan-Blue CAD Blueprint Grid shader using Object-Space coordinates."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.42 if is_floor else 0.58
    bsdf.inputs["Metallic"].default_value = 0.05
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25 if is_floor else 0.08

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Create major (0.5m) & minor (0.125m) grid masks along X and (Y if floor else Z)
    ax1 = sep.outputs["X"]
    ax2 = sep.outputs["Y"] if is_floor else sep.outputs["Z"]

    def axis_lines(sock: Any, spacing: float, width: float) -> Any:
        div = nt.nodes.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = spacing
        nt.links.new(sock, div.inputs[0])

        frac = nt.nodes.new("ShaderNodeMath")
        frac.operation = "FRACT"
        nt.links.new(div.outputs[0], frac.inputs[0])

        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        nt.links.new(frac.outputs[0], sub.inputs[0])

        absv = nt.nodes.new("ShaderNodeMath")
        absv.operation = "ABSOLUTE"
        nt.links.new(sub.outputs[0], absv.inputs[0])

        # Distance from nearest grid line in [0, 0.5]
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 0.5
        nt.links.new(absv.outputs[0], inv.inputs[1])

        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value = width
        mr.inputs["From Max"].default_value = width * 0.25
        nt.links.new(inv.outputs[0], mr.inputs["Value"])
        return mr.outputs["Result"]

    maj_x = axis_lines(ax1, 0.50, 0.020)
    maj_2 = axis_lines(ax2, 0.50, 0.020)
    maj = nt.nodes.new("ShaderNodeMath")
    maj.operation = "MAXIMUM"
    nt.links.new(maj_x, maj.inputs[0])
    nt.links.new(maj_2, maj.inputs[1])

    min_x = axis_lines(ax1, 0.125, 0.022)
    min_2 = axis_lines(ax2, 0.125, 0.022)
    minor = nt.nodes.new("ShaderNodeMath")
    minor.operation = "MAXIMUM"
    nt.links.new(min_x, minor.inputs[0])
    nt.links.new(min_2, minor.inputs[1])

    minor_scaled = nt.nodes.new("ShaderNodeMath")
    minor_scaled.operation = "MULTIPLY"
    minor_scaled.inputs[1].default_value = 0.30
    nt.links.new(minor.outputs[0], minor_scaled.inputs[0])

    grid_mask = nt.nodes.new("ShaderNodeMath")
    grid_mask.operation = "MAXIMUM"
    nt.links.new(maj.outputs[0], grid_mask.inputs[0])
    nt.links.new(minor_scaled.outputs[0], grid_mask.inputs[1])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    fac, col_a, col_b = C.get_mix_sockets(mix_col)
    nt.links.new(grid_mask.outputs[0], fac)
    col_a.default_value = (0.010, 0.044, 0.105, 1.0) if not is_floor else (0.008, 0.036, 0.088, 1.0)
    col_b.default_value = (0.08, 0.72, 1.00, 1.0)

    nt.links.new(mix_col.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(mix_col.outputs[2], bsdf.inputs["Emission Color"])

    emit_mul = nt.nodes.new("ShaderNodeMath")
    emit_mul.operation = "MULTIPLY"
    emit_mul.inputs[1].default_value = 0.85 if not is_floor else 0.72
    nt.links.new(grid_mask.outputs[0], emit_mul.inputs[0])

    emit_add = nt.nodes.new("ShaderNodeMath")
    emit_add.operation = "ADD"
    emit_add.inputs[1].default_value = 0.14 if not is_floor else 0.10
    nt.links.new(emit_mul.outputs[0], emit_add.inputs[0])
    nt.links.new(emit_add.outputs[0], bsdf.inputs["Emission Strength"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mat_solid(
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.2,
    roughness: float = 0.45,
    emit_col: tuple[float, float, float] | None = None,
    emit_str: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emit_col is not None and emit_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_col, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    return mat


def _build_ground_radar_station(col: bpy.types.Collection, total_frames: int) -> bpy.types.Object:
    """Detailed military parabolic radar tower + glowing cyan feed-horn emitter for Beats 1-2."""
    mat_steel = _mat_solid("Mat_Env_RadarSteel", (0.09, 0.11, 0.14), metallic=0.82, roughness=0.38)
    mat_dish = _mat_solid("Mat_Env_RadarDish", (0.18, 0.22, 0.27), metallic=0.72, roughness=0.32)
    mat_cyan = _mat_solid(
        "Mat_Env_RadarCore", (0.06, 0.88, 1.00), metallic=0.1, roughness=0.20,
        emit_col=(0.06, 0.88, 1.00), emit_str=1.95,
    )
    mat_warn = _mat_solid(
        "Mat_Env_RadarWarn", (1.00, 0.24, 0.06), metallic=0.1, roughness=0.25,
        emit_col=(1.00, 0.24, 0.06), emit_str=1.65,
    )

    root = bpy.data.objects.new("Env_GroundRadar_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    # Pedestal base + octagonal control bunker + yoke arms
    bm_base = bmesh.new()
    bmesh.ops.create_cone(
        bm_base, cap_ends=True, cap_tris=False, segments=8,
        radius1=0.32, radius2=0.22, depth=0.36,
        matrix=Matrix.Translation((0.0, 0.0, -0.30)),
    )
    _add_box(bm_base, (0.0, 0.0, -0.06), (0.18, 0.16, 0.22), mat_index=0, bevel_w=0.012)
    _add_box(bm_base, (-0.11, 0.0, 0.06), (0.04, 0.08, 0.18), mat_index=0, bevel_w=0.006)
    _add_box(bm_base, (+0.11, 0.0, 0.06), (0.04, 0.08, 0.18), mat_index=0, bevel_w=0.006)
    # Warning beacon lights on bunker corners
    for sx in (-0.16, 0.16):
        _add_box(bm_base, (sx, -0.14, -0.14), (0.022, 0.022, 0.026), mat_index=3)
    base_obj = _mesh_from_bm("Env_RadarPedestal", bm_base, [mat_steel, mat_dish, mat_cyan, mat_warn], col, parent=root)

    # Parabolic reflector dish + radial structural ribs + central feed-horn tripod
    bm_dish = bmesh.new()
    segs = 32
    rings = 8
    r_max = 0.28
    depth_k = 0.26
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
            f_face = bm_dish.faces.new((ring_verts[ri][si], ring_verts[ri][sn], ring_verts[ri + 1][sn], ring_verts[ri + 1][si]))
            f_face.material_index = 1
    # Solidify dish rim slightly by adding back shell rim ring
    bmesh.ops.solidify(bm_dish, geom=list(bm_dish.faces), thickness=0.016)

    # Feed horn stalk & glowing cyan RF emitter dome at parabolic focus (0, 0, 0.19)
    bmesh.ops.create_cone(
        bm_dish, cap_ends=True, cap_tris=False, segments=12,
        radius1=0.016, radius2=0.012, depth=0.19,
        matrix=Matrix.Translation((0.0, 0.0, 0.095)),
    )
    res_core = bmesh.ops.create_icosphere(
        bm_dish, subdivisions=2, radius=0.036, matrix=Matrix.Translation((0.0, 0.0, 0.195))
    )
    core_vset = set(res_core["verts"])
    for f_c in bm_dish.faces:
        if all(v in core_vset for v in f_c.verts):
            f_c.material_index = 2

    dish_obj = _mesh_from_bm(
        "Env_RadarDish", bm_dish, [mat_steel, mat_dish, mat_cyan], col,
        location=(0.0, 0.0, 0.12), parent=root,
    )

    # Position & aim dish cleanly for Beat 1 (aimed up-right at gliding bomber) and Beat 2 (aimed +X at aircraft)
    b1_pos = Vector(C.STAGE["radar_dish_b1"])
    b2_pos = Vector(C.STAGE["radar_dish_b2"])
    for f in (1, 110, 220, 221, 380, 540):
        if f <= 220:
            root.location = b1_pos
            bpos, _ = C.bomber_pose_at(float(f))
            aim = (Vector(bpos) - (b1_pos + Vector((0.0, 0.0, 0.12)))).normalized()
            dish_obj.rotation_euler = aim.to_track_quat("Z", "Y").to_euler("XYZ")
        else:
            root.location = b2_pos
            # Gently sweep up/down between Conv Jet and Stealth Wing in Beat 2
            tz = C.lerp(2.68, 1.44, C.smootherstep(290.0, 420.0, float(f)))
            aim = (Vector((0.12, 0.12, tz)) - (b2_pos + Vector((0.0, 0.0, 0.12)))).normalized()
            dish_obj.rotation_euler = aim.to_track_quat("Z", "Y").to_euler("XYZ")
        root.keyframe_insert("location", frame=f)
        dish_obj.keyframe_insert("rotation_euler", frame=f)

    # Constant interpolation at Beat 1 -> Beat 2 cut
    for obj_k in (root, dish_obj):
        ad = getattr(obj_k, "animation_data", None)
        if ad and ad.action:
            for fc in C.iter_action_fcurves(ad.action):
                for kp in fc.keyframe_points:
                    if int(round(kp.co.x)) == 220:
                        kp.interpolation = "CONSTANT"

    for obj_v in (base_obj, dish_obj):
        _key_visibility(obj_v, [(1, 540)], total_frames)

    return root


def _build_blueprint_studio(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """Beat 5 @zackdfilms Cyan-Blue CAD Blueprint Wall (Z=-0.15..8.80m) & Extended Floor."""
    mat_wall = _mat_blueprint_grid("Mat_Env_BlueprintWall", is_floor=False)
    mat_floor = _mat_blueprint_grid("Mat_Env_BlueprintFloor", is_floor=True)
    mat_dark_floor = _mat_solid("Mat_Env_NightTarmac", (0.010, 0.014, 0.022), metallic=0.15, roughness=0.52)

    # Blueprint backdrop wall (tall enough to cover full 9:16 portrait sky)
    bm_w = bmesh.new()
    x0, x1 = -7.0, 7.0
    z0, z1 = float(C.STAGE["backdrop_z_min"]), float(C.STAGE["backdrop_z_max"])
    yb = float(C.STAGE["backdrop_y"])
    v1 = bm_w.verts.new((x0, yb, z0))
    v2 = bm_w.verts.new((x1, yb, z0))
    v3 = bm_w.verts.new((x1, yb, z1))
    v4 = bm_w.verts.new((x0, yb, z1))
    bm_w.faces.new((v1, v2, v3, v4)).material_index = 0
    wall_obj = _mesh_from_bm("Env_BlueprintWall", bm_w, [mat_wall], col, smooth=False)
    wall_obj.visible_shadow = False
    _key_visibility(wall_obj, [(1321, 1560)], total_frames)

    # Blueprint floor slab (covers well past camera Y=-4.5m)
    bm_f = bmesh.new()
    fz = float(C.STAGE["floor_z"])
    _add_box(bm_f, (0.0, 0.0, fz - 0.10), (14.0, 12.0, 0.20), mat_index=0)
    bp_floor = _mesh_from_bm("Env_BlueprintFloor", bm_f, [mat_floor], col, smooth=False)
    _key_visibility(bp_floor, [(1321, 1560)], total_frames)

    # Night tactical ground plane beneath radar dish in Beat 1
    bm_nf = bmesh.new()
    _add_box(bm_nf, (0.0, 0.0, fz - 0.10), (14.0, 12.0, 0.20), mat_index=0)
    night_floor = _mesh_from_bm("Env_NightGround", bm_nf, [mat_dark_floor], col, smooth=False)
    _key_visibility(night_floor, [(1, 220)], total_frames)

    return [wall_obj, bp_floor, night_floor]


def _add_area_light(
    name: str,
    col: bpy.types.Collection,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
    size: float,
    size_y: float | None = None,
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
    col.objects.link(lobj)
    return lobj


def _build_lighting(col: bpy.types.Collection, total_frames: int) -> dict[str, bpy.types.Object]:
    """Cinematographic moonlight rim + warm bounce + macro X-ray + CAD studio softbox lighting."""
    # 1. Strong Cool Cyan-Silver Moonlight Rim Key (makes obsidian flying-wing facets gleam!)
    l_rim = _add_area_light(
        "Light_MoonRim_Key", col,
        location=(2.6, 1.4, 5.2), target=(0.0, 0.0, 1.85),
        color=(0.64, 0.86, 1.00), energy=620.0, size=2.4, size_y=1.6,
    )
    # 2. Secondary Left Back-Rim Edge Light (traces left 33-deg stealth wing leading edge)
    l_rim_l = _add_area_light(
        "Light_CyanRim_Left", col,
        location=(-3.1, 0.8, 3.8), target=(0.0, 0.0, 1.90),
        color=(0.24, 0.78, 1.00), energy=420.0, size=2.0, size_y=1.4,
    )
    # 3. Subtle Front-Low Camera Fill (keeps matte-black RAM contours readable, never crushed black)
    l_fill = _add_area_light(
        "Light_FrontFill_Soft", col,
        location=(-0.8, -4.4, 1.1), target=(0.0, 0.0, 1.85),
        color=(0.76, 0.85, 0.98), energy=220.0, size=2.6, size_y=2.2,
    )
    # 4. Warm Horizon Underbelly Accent (cinematic cyan/amber complementary contrast!)
    l_warm = _add_area_light(
        "Light_HorizonBounce_Warm", col,
        location=(1.8, -2.6, 0.45), target=(0.0, 0.0, 1.85),
        color=(1.00, 0.60, 0.28), energy=145.0, size=1.8, size_y=1.0,
    )
    # 5. Macro Precision Key (Beats 3 & 4 — illuminates titanium turbines, S-ducts & 3D bumblebee)
    l_macro = _add_area_light(
        "Light_MacroPrecision_Key", col,
        location=(0.95, -1.65, 2.85), target=(0.0, 0.0, 1.85),
        color=(0.94, 0.97, 1.00), energy=120.0, size=0.85, size_y=0.65,
    )

    # Animate lighting energy cleanly per beat so every scene hits peak contrast
    key_frames = [1, 220, 221, 540, 541, 960, 961, 1320, 1321, 1560, 1561, total_frames]
    for f in key_frames:
        is_macro = 541 <= f <= 1320
        is_bp = 1321 <= f <= 1560
        l_rim.data.energy = 420.0 if is_bp else (480.0 if is_macro else 680.0)
        l_rim_l.data.energy = 380.0 if is_bp else (360.0 if is_macro else 460.0)
        l_fill.data.energy = 480.0 if is_bp else (190.0 if is_macro else 240.0)
        l_warm.data.energy = 45.0 if is_bp else (130.0 if is_macro else 160.0)
        l_macro.data.energy = 175.0 if is_macro else 0.0
        for lo in (l_rim, l_rim_l, l_fill, l_warm, l_macro):
            lo.data.keyframe_insert("energy", frame=f)

    return {
        "moon_rim": l_rim,
        "cyan_rim_l": l_rim_l,
        "front_fill": l_fill,
        "warm_bounce": l_warm,
        "macro_key": l_macro,
    }


def build_radar_env(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Assemble night airspace world, ground radar station, Beat 5 CAD blueprint studio, and lights."""
    world = _setup_world(total_frames)
    radar_station = _build_ground_radar_station(col, total_frames)
    bp_objs = _build_blueprint_studio(col, total_frames)
    lights = _build_lighting(col, total_frames)
    return {
        "world": world,
        "radar_station": radar_station,
        "blueprint_objs": bp_objs,
        "lights": lights,
    }
