"""
jet_cockpit_env.py — Supersonic Twin-Tail Fighter Jet, Cockpit Tub & Golden-Hour Sky Environment
Episode #1: "Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames)

Public API:
    build_environment(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dict keys & exact types:
    - "jet_root": bpy.types.Object (Env_Jet_Root Empty, animated via C.jet_offset)
    - "fuselage": bpy.types.Object (Env_Jet_Fuselage)
    - "wings": bpy.types.Object (Env_Jet_Wings)
    - "tail_fins": bpy.types.Object (Env_Jet_TailFins)
    - "intakes": bpy.types.Object (Env_Jet_Intakes)
    - "nozzles": bpy.types.Object (Env_Jet_Nozzles)
    - "afterburner_cones": list[bpy.types.Object] ([Env_Afterburner_L, Env_Afterburner_R])
    - "cockpit_tub": bpy.types.Object (Env_Cockpit_Tub)
    - "console_switches": bpy.types.Object (Env_Cockpit_ConsoleSwitches)
    - "instrument_panel": bpy.types.Object (Env_Cockpit_Panel)
    - "mfd_screens": bpy.types.Object (Env_Cockpit_MFDs)
    - "master_caution": bpy.types.Object (Env_Master_Caution)
    - "hud_frame": bpy.types.Object (Env_HUD_Frame)
    - "hud_glass": bpy.types.Object (Env_HUD_Glass)
    - "control_stick": bpy.types.Object (Env_Control_Stick)
    - "throttle": bpy.types.Object (Env_Throttle)
    - "rudder_pedals": bpy.types.Object (Env_Rudder_Pedals)
    - "cloud_deck": list[bpy.types.Object] ([Env_CloudDeck_Stratus, Env_CloudDeck_Cumulus])
    - "speed_clouds": list[bpy.types.Object] ([Env_SpeedCloud_01 .. Env_SpeedCloud_18])
    - "world": bpy.types.World (Env_GoldenHour_World)
    - "lights": dict[str, bpy.types.Object]
    - "xray_value_nodes": list[bpy.types.ShaderNodeValue]
    - "materials": dict[str, bpy.types.Material]
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

_THIS_FILE = Path(__file__).resolve()
_EPISODE_SRC = str(_THIS_FILE.parent)
_REPO_ROOT_SRC = str(_THIS_FILE.parents[3] / "src")
for _p in (_EPISODE_SRC, _REPO_ROOT_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402


# ============================================================================
# SHARED SCENE, COLLECTION, BMESH & ANIMATION HELPERS
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection | None = None) -> bpy.types.Object:
    target = col if col is not None else bpy.context.scene.collection
    for c in list(obj.users_collection):
        if c != target:
            c.objects.unlink(obj)
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_bsdf(bsdf: bpy.types.Node, name: str, val) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _rgba_mix_sockets(mix_node: bpy.types.Node):
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def _aim_euler(src, dst) -> Euler:
    d = Vector(dst) - Vector(src)
    if d.length < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    return d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")


def _finalize_bmesh(
    bm: bmesh.types.BMesh,
    name: str,
    col: bpy.types.Collection,
    materials: list[bpy.types.Material] | None = None,
    smooth: bool = True,
    recalc_normals: bool = True,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Convert BMesh -> Mesh -> Object with AGENTS.md Sec 3 material-slot-before-to_mesh rule."""
    if recalc_normals and len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    if materials:
        for mat in materials:
            mesh.materials.append(mat)

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
    _link_obj(obj, col)
    return obj


def _set_fcurve_interp(anim_target, interp: str = "LINEAR", prefix: str | None = None) -> None:
    ad = getattr(anim_target, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if prefix is not None and not fc.data_path.startswith(prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe hide_viewport & hide_render strictly with CONSTANT interpolation."""
    frames = {1, total_frames}
    for s, e in visible_ranges:
        if s > 1:
            frames.add(s - 1)
        frames.add(s)
        frames.add(e)
        if e < total_frames:
            frames.add(e + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)

    _set_fcurve_interp(obj, "CONSTANT", prefix="hide_")


def _add_box(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    rot_euler: tuple[float, float, float] | None = None,
    bevel_width: float = 0.0,
) -> list[bmesh.types.BMFace]:
    """Create a box (optionally rotated & beveled) inside an existing BMesh."""
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_width > 0.0:
        bmesh.ops.bevel(
            tmp,
            geom=list(tmp.edges),
            offset=bevel_width,
            segments=2,
            affect="EDGES",
            clamp_overlap=True,
        )
    if rot_euler is not None:
        bmesh.ops.rotate(tmp, matrix=Euler(rot_euler, "XYZ").to_matrix(), verts=tmp.verts)
    bmesh.ops.translate(tmp, vec=Vector(center), verts=tmp.verts)

    vert_map = {}
    for v in tmp.verts:
        vert_map[v] = bm.verts.new(v.co)
    faces = []
    for f in tmp.faces:
        nf = bm.faces.new([vert_map[v] for v in f.verts])
        nf.material_index = mat_index
        faces.append(nf)
    tmp.free()
    return faces


def _add_cyl(
    bm: bmesh.types.BMesh,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    r0: float,
    r1: float,
    segments: int = 16,
    cap_ends: bool = True,
    mat_index: int = 0,
) -> None:
    """Create a tapered cylinder/cone segment from p0 to p1 inside an existing BMesh."""
    v0 = Vector(p0)
    v1 = Vector(p1)
    dv = v1 - v0
    length = dv.length
    if length < 1e-6:
        return
    mid = (v0 + v1) * 0.5
    rot = dv.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    mat = Matrix.Translation(mid) @ rot

    tmp = bmesh.new()
    bmesh.ops.create_cone(
        tmp,
        cap_ends=cap_ends,
        cap_tris=False,
        segments=segments,
        radius1=r0,
        radius2=r1,
        depth=length,
        matrix=mat,
    )
    vert_map = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vert_map[v] for v in f.verts])
        nf.material_index = mat_index
    tmp.free()


# ============================================================================
# PROCEDURAL LOCAL-SPACE PBR, X-RAY & SKY SHADERS
# ============================================================================
def _make_xray_fuselage_material(
    name: str,
    solid_rgb: tuple[float, float, float],
    metallic: float,
    roughness: float,
    jet_root: bpy.types.Object,
    xray_value_nodes: list[bpy.types.ShaderNodeValue],
    panel_brick_scale: float = 1.0,
    xray_rim_strength: float = 1.80,
) -> bpy.types.Material:
    """
    Tactical aerospace airframe / cockpit-tub shader with local-space Object coordinates
    bound to `Env_Jet_Root`, transitioning smoothly to crystal-clear cyan X-ray glass
    during C.XRAY_FUSELAGE (f=320..1236) with mandatory LightPath.Is Shadow Ray bypass.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")

    # Local-space coordinate bound to Env_Jet_Root (AGENTS.md Sec 3)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = jet_root

    # Subtle aerospace panel lines + micro surface grain
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    # Combine XY + YZ projections cleanly by using Brick on (X+Z, Y) via Sep/Comb or 3D Voronoi
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(mp.outputs["Vector"], sep.inputs["Vector"])

    comb_panel = nt.nodes.new("ShaderNodeCombineXYZ")
    add_xz = nt.nodes.new("ShaderNodeMath")
    add_xz.operation = "ADD"
    nt.links.new(sep.outputs["X"], add_xz.inputs[0])
    nt.links.new(sep.outputs["Z"], add_xz.inputs[1])
    nt.links.new(add_xz.outputs["Value"], comb_panel.inputs["X"])
    nt.links.new(sep.outputs["Y"], comb_panel.inputs["Y"])

    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.inputs["Scale"].default_value = 1.15 * panel_brick_scale
    brick.inputs["Mortar Size"].default_value = 0.008
    brick.inputs["Mortar Smooth"].default_value = 0.25
    brick.inputs["Bias"].default_value = 0.0
    brick.inputs["Brick Width"].default_value = 1.4
    brick.inputs["Row Height"].default_value = 0.85
    brick.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    brick.inputs["Color2"].default_value = (0.94, 0.94, 0.94, 1.0)
    brick.inputs["Mortar"].default_value = (0.48, 0.48, 0.48, 1.0)
    nt.links.new(comb_panel.outputs["Vector"], brick.inputs["Vector"])

    n_grain = nt.nodes.new("ShaderNodeTexNoise")
    n_grain.inputs["Scale"].default_value = 18.0
    n_grain.inputs["Detail"].default_value = 6.0
    n_grain.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n_grain.inputs["Vector"])

    n_weather = nt.nodes.new("ShaderNodeTexNoise")
    n_weather.inputs["Scale"].default_value = 1.6
    n_weather.inputs["Detail"].default_value = 5.0
    n_weather.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], n_weather.inputs["Vector"])

    solid = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(solid, "Metallic", metallic)
    _set_bsdf(solid, "Coat Weight", 0.14)
    _set_bsdf(solid, "Coat Roughness", 0.34)

    mix_base = nt.nodes.new("ShaderNodeMix")
    mix_base.data_type = "RGBA"
    fac_b, ca_b, cb_b, out_b = _rgba_mix_sockets(mix_base)
    ca_b.default_value = (solid_rgb[0] * 0.82, solid_rgb[1] * 0.82, solid_rgb[2] * 0.84, 1.0)
    cb_b.default_value = (min(1.0, solid_rgb[0] * 1.15), min(1.0, solid_rgb[1] * 1.15), min(1.0, solid_rgb[2] * 1.16), 1.0)
    nt.links.new(n_weather.outputs["Fac"], fac_b)

    mix_panel = nt.nodes.new("ShaderNodeMix")
    mix_panel.data_type = "RGBA"
    mix_panel.blend_type = "MULTIPLY"
    fac_p, ca_p, cb_p, out_p = _rgba_mix_sockets(mix_panel)
    fac_p.default_value = 0.65
    nt.links.new(out_b, ca_p)
    nt.links.new(brick.outputs["Color"], cb_p)
    nt.links.new(out_p, solid.inputs["Base Color"])

    mr_r = nt.nodes.new("ShaderNodeMapRange")
    mr_r.inputs["To Min"].default_value = max(0.18, roughness - 0.08)
    mr_r.inputs["To Max"].default_value = min(0.88, roughness + 0.10)
    nt.links.new(n_grain.outputs["Fac"], mr_r.inputs["Value"])
    nt.links.new(mr_r.outputs["Result"], solid.inputs["Roughness"])

    bump_panel = nt.nodes.new("ShaderNodeBump")
    bump_panel.inputs["Strength"].default_value = 0.08
    bump_panel.inputs["Distance"].default_value = 0.003
    nt.links.new(brick.outputs["Fac"], bump_panel.inputs["Height"])

    bump_fine = nt.nodes.new("ShaderNodeBump")
    bump_fine.inputs["Strength"].default_value = 0.04
    bump_fine.inputs["Distance"].default_value = 0.0015
    nt.links.new(n_grain.outputs["Fac"], bump_fine.inputs["Height"])
    nt.links.new(bump_panel.outputs["Normal"], bump_fine.inputs["Normal"])
    nt.links.new(bump_fine.outputs["Normal"], solid.inputs["Normal"])

    # X-Ray Cyan Blueprint Glass Branch
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (0.95, 0.99, 1.0, 1.0)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.24

    rim_pow = nt.nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.inputs[1].default_value = 2.4
    nt.links.new(lw.outputs["Facing"], rim_pow.inputs[0])

    rim_scale = nt.nodes.new("ShaderNodeMath")
    rim_scale.operation = "MULTIPLY"
    rim_scale.inputs[1].default_value = 0.36
    nt.links.new(rim_pow.outputs["Value"], rim_scale.inputs[0])

    xray_emit = nt.nodes.new("ShaderNodeEmission")
    xray_emit.inputs["Color"].default_value = (0.12, 0.84, 1.0, 1.0)
    xray_emit.inputs["Strength"].default_value = xray_rim_strength  # AgX-safe <= 2.1

    xray_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(rim_scale.outputs["Value"], xray_mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], xray_mix.inputs[1])
    nt.links.new(xray_emit.outputs["Emission"], xray_mix.inputs[2])

    # Driven by keyframed ShaderNodeValue ("XRayFuselage")
    xray_val = nt.nodes.new("ShaderNodeValue")
    xray_val.name = "XRayFuselage"
    xray_val.outputs[0].default_value = 0.0
    xray_value_nodes.append(xray_val)

    main_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(xray_val.outputs[0], main_mix.inputs["Fac"])
    nt.links.new(solid.outputs["BSDF"], main_mix.inputs[1])
    nt.links.new(xray_mix.outputs["Shader"], main_mix.inputs[2])

    # AGENTS.md Sec 3: LightPath.Is Shadow Ray -> Transparent BSDF when X-ray active!
    lp = nt.nodes.new("ShaderNodeLightPath")
    shad_fac = nt.nodes.new("ShaderNodeMath")
    shad_fac.operation = "MULTIPLY"
    shad_fac.use_clamp = True
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_fac.inputs[0])
    nt.links.new(xray_val.outputs[0], shad_fac.inputs[1])

    final_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(shad_fac.outputs["Value"], final_mix.inputs["Fac"])
    nt.links.new(main_mix.outputs["Shader"], final_mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], final_mix.inputs[2])

    nt.links.new(final_mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_opaque_metal_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.82,
    roughness: float = 0.36,
    jet_root: bpy.types.Object | None = None,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    _set_bsdf(bsdf, "Roughness", roughness)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    if jet_root is not None:
        tc.object = jet_root
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 48.0
    n.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.0015
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_mfd_screen_mat(name: str, mode: str, jet_root: bpy.types.Object) -> bpy.types.Material:
    """Procedural emerald/cyan/amber cockpit MFD telemetry screen (AgX-safe <= 1.85)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.010, 0.018, 0.016, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.08)
    _set_bsdf(bsdf, "Coat Weight", 0.75)
    _set_bsdf(bsdf, "Coat Roughness", 0.04)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = jet_root

    wave_h = nt.nodes.new("ShaderNodeTexWave")
    wave_h.bands_direction = "Z"
    wave_h.inputs["Scale"].default_value = 26.0 if mode != "ufc" else 44.0
    wave_h.inputs["Distortion"].default_value = 0.0
    nt.links.new(tc.outputs["Object"], wave_h.inputs["Vector"])

    wave_v = nt.nodes.new("ShaderNodeTexWave")
    wave_v.bands_direction = "X"
    wave_v.inputs["Scale"].default_value = 18.0
    wave_v.inputs["Distortion"].default_value = 0.0
    nt.links.new(tc.outputs["Object"], wave_v.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.84
    cr.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cr.color_ramp.elements[1].position = 0.92
    cr.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(wave_h.outputs["Fac"], cr.inputs["Fac"])

    em_col = {
        "center": (0.12, 0.95, 0.56, 1.0),
        "left": (0.12, 0.82, 1.00, 1.0),
        "right": (1.00, 0.42, 0.08, 1.0),
        "ufc": (0.20, 1.00, 0.48, 1.0),
    }.get(mode, (0.12, 0.92, 0.60, 1.0))

    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    fac_m, ca_m, cb_m, out_m = _rgba_mix_sockets(mix_c)
    ca_m.default_value = (em_col[0] * 0.14, em_col[1] * 0.14, em_col[2] * 0.14, 1.0)
    cb_m.default_value = em_col
    nt.links.new(cr.outputs["Color"], fac_m)

    nt.links.new(out_m, bsdf.inputs["Emission Color"])
    _set_bsdf(bsdf, "Emission Strength", 1.75)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_master_caution_mat() -> tuple[bpy.types.Material, bpy.types.ShaderNodeValue]:
    """Flashing red MASTER CAUTION warning annunciator driven by CautionPulse Value node."""
    mat = bpy.data.materials.new("Mat_Env_MasterCaution")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.18, 0.01, 0.01, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.16)
    _set_bsdf(bsdf, "Emission Color", (1.0, 0.06, 0.03, 1.0))

    pulse = nt.nodes.new("ShaderNodeValue")
    pulse.name = "CautionPulse"
    pulse.outputs[0].default_value = 2.0
    nt.links.new(pulse.outputs[0], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, pulse


def _make_hud_combiner_mat() -> tuple[bpy.types.Material, bpy.types.ShaderNodeTexCoord]:
    """Emerald holographic reflective HUD combiner glass with shadow ray bypass."""
    mat = bpy.data.materials.new("Mat_Env_HUDGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (0.86, 0.99, 0.92, 1.0)

    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Color"].default_value = (0.35, 0.96, 0.68, 1.0)
    gloss.inputs["Roughness"].default_value = 0.02

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.20

    mix_glass = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Fresnel"], mix_glass.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix_glass.inputs[1])
    nt.links.new(gloss.outputs["BSDF"], mix_glass.inputs[2])

    # Procedural emerald HUD pitch ladder ticks (local-space)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 38.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.88
    cr.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cr.color_ramp.elements[1].position = 0.94
    cr.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(wave.outputs["Fac"], cr.inputs["Fac"])

    lp = nt.nodes.new("ShaderNodeLightPath")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.14, 1.0, 0.56, 1.0)

    em_gain = nt.nodes.new("ShaderNodeMath")
    em_gain.operation = "MULTIPLY"
    em_gain.inputs[1].default_value = 1.45  # AgX-safe
    nt.links.new(cr.outputs["Color"], em_gain.inputs[0])

    cam_mul = nt.nodes.new("ShaderNodeMath")
    cam_mul.operation = "MULTIPLY"
    nt.links.new(em_gain.outputs["Value"], cam_mul.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], cam_mul.inputs[1])
    nt.links.new(cam_mul.outputs["Value"], em.inputs["Strength"])

    add_hud = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(mix_glass.outputs["Shader"], add_hud.inputs[0])
    nt.links.new(em.outputs["Emission"], add_hud.inputs[1])

    shad_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_mix.inputs["Fac"])
    nt.links.new(add_hud.outputs["Shader"], shad_mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], shad_mix.inputs[2])

    nt.links.new(shad_mix.outputs["Shader"], out.inputs["Surface"])
    return mat, tc


def _make_afterburner_plume_mat(
    name: str,
    jet_root: bpy.types.Object,
    is_core: bool,
) -> tuple[bpy.types.Material, bpy.types.ShaderNodeValue]:
    """
    Open-ended additive supersonic Mach-1 afterburner shock-diamond plume (AGENTS.md Sec 3):
    AddShader(Transparent, Emission) gated by Camera Ray * (1 - Backfacing), local Y shock rings.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    add_sh = nt.nodes.new("ShaderNodeAddShader")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = jet_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Axial distance aft from nozzle exit y = -9.50 (positive along plume length)
    dist_aft = nt.nodes.new("ShaderNodeMapRange")
    dist_aft.clamp = True
    dist_aft.inputs["From Min"].default_value = -9.45
    dist_aft.inputs["From Max"].default_value = -13.60 if is_core else -14.70
    dist_aft.inputs["To Min"].default_value = 1.0
    dist_aft.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Y"], dist_aft.inputs["Value"])

    # Periodic Mach shock diamonds along local Y
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "Y"
    wave.wave_profile = "SIN"
    wave.inputs["Scale"].default_value = 1.65 if is_core else 1.15
    wave.inputs["Distortion"].default_value = 0.35
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])

    diamond_gain = nt.nodes.new("ShaderNodeMapRange")
    diamond_gain.clamp = True
    diamond_gain.inputs["From Min"].default_value = 0.0
    diamond_gain.inputs["From Max"].default_value = 1.0
    diamond_gain.inputs["To Min"].default_value = 0.42 if is_core else 0.25
    diamond_gain.inputs["To Max"].default_value = 1.0
    nt.links.new(wave.outputs["Fac"], diamond_gain.inputs["Value"])

    # Color ramp: hot blue-white nozzle -> golden-amber Mach diamonds -> violet-orange tail
    cr = nt.nodes.new("ShaderNodeValToRGB")
    if is_core:
        cr.color_ramp.elements[0].position = 0.15
        cr.color_ramp.elements[0].color = (1.0, 0.36, 0.06, 1.0)
        e_mid = cr.color_ramp.elements.new(0.62)
        e_mid.color = (1.0, 0.78, 0.34, 1.0)
        cr.color_ramp.elements[2].position = 0.92
        cr.color_ramp.elements[2].color = (0.78, 0.90, 1.0, 1.0)
    else:
        cr.color_ramp.elements[0].position = 0.15
        cr.color_ramp.elements[0].color = (0.72, 0.18, 0.42, 1.0)
        e_mid = cr.color_ramp.elements.new(0.55)
        e_mid.color = (1.0, 0.44, 0.08, 1.0)
        cr.color_ramp.elements[2].position = 0.92
        cr.color_ramp.elements[2].color = (1.0, 0.72, 0.28, 1.0)
    nt.links.new(dist_aft.outputs["Result"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], em.inputs["Color"])

    # Soft silhouette edge fade: (1 - Facing) -> actually Facing in Blender LayerWeight:
    # 0 at perpendicular grazing edge? Wait: LayerWeight.Facing = 1 - |N.V| (0 center, 1 edge!)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42
    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    inv_f.use_clamp = True
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    soft_edge = nt.nodes.new("ShaderNodeMath")
    soft_edge.operation = "POWER"
    soft_edge.inputs[1].default_value = 1.7
    nt.links.new(inv_f.outputs["Value"], soft_edge.inputs[0])

    # Camera Ray * (1 - Backfacing) gate (AGENTS.md Sec 3)
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    front = nt.nodes.new("ShaderNodeMath")
    front.operation = "SUBTRACT"
    front.inputs[0].default_value = 1.0
    front.use_clamp = True
    nt.links.new(geom.outputs["Backfacing"], front.inputs[1])

    gate = nt.nodes.new("ShaderNodeMath")
    gate.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], gate.inputs[0])
    nt.links.new(front.outputs["Value"], gate.inputs[1])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(dist_aft.outputs["Result"], m1.inputs[0])
    nt.links.new(diamond_gain.outputs["Result"], m1.inputs[1])

    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    nt.links.new(m1.outputs["Value"], m2.inputs[0])
    nt.links.new(soft_edge.outputs["Value"], m2.inputs[1])

    m3 = nt.nodes.new("ShaderNodeMath")
    m3.operation = "MULTIPLY"
    nt.links.new(m2.outputs["Value"], m3.inputs[0])
    nt.links.new(gate.outputs["Value"], m3.inputs[1])

    gain_val = nt.nodes.new("ShaderNodeValue")
    gain_val.name = "BurnerGain"
    gain_val.outputs[0].default_value = 4.6 if is_core else 1.85

    m_final = nt.nodes.new("ShaderNodeMath")
    m_final.operation = "MULTIPLY"
    nt.links.new(m3.outputs["Value"], m_final.inputs[0])
    nt.links.new(gain_val.outputs[0], m_final.inputs[1])
    nt.links.new(m_final.outputs["Value"], em.inputs["Strength"])

    nt.links.new(tr.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(em.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, gain_val


def _make_cloud_deck_mat(name: str = "Mat_Env_CloudDeck", is_cumulus: bool = False) -> tuple[bpy.types.Material, bpy.types.ShaderNodeMapping]:
    """Golden-hour sunlit high-altitude cloud sea shader (fast Cycles render, zero heavy volumes!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Roughness", 0.94)
    _set_bsdf(bsdf, "Specular IOR Level", 0.06)
    _set_bsdf(bsdf, "Subsurface Weight", 0.35)
    _set_bsdf(bsdf, "Subsurface Scale", 0.35)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    n_macro = nt.nodes.new("ShaderNodeTexNoise")
    n_macro.inputs["Scale"].default_value = 0.045 if not is_cumulus else 0.16
    n_macro.inputs["Detail"].default_value = 6.0
    n_macro.inputs["Roughness"].default_value = 0.65
    nt.links.new(mp.outputs["Vector"], n_macro.inputs["Vector"])

    n_billow = nt.nodes.new("ShaderNodeTexNoise")
    n_billow.inputs["Scale"].default_value = 0.18 if not is_cumulus else 0.52
    n_billow.inputs["Detail"].default_value = 5.0
    n_billow.inputs["Roughness"].default_value = 0.60
    nt.links.new(mp.outputs["Vector"], n_billow.inputs["Vector"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.30
    cr.color_ramp.elements[0].color = (0.38, 0.46, 0.60, 1.0)  # luminous atmospheric slate-blue cloud troughs
    e_mid = cr.color_ramp.elements.new(0.56)
    e_mid.color = (0.76, 0.74, 0.75, 1.0)
    cr.color_ramp.elements[2].position = 0.82
    cr.color_ramp.elements[2].color = (1.0, 0.90, 0.78, 1.0)  # warm golden-hour sunlit crests
    nt.links.new(n_macro.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35 if not is_cumulus else 0.42
    bump.inputs["Distance"].default_value = 1.2
    nt.links.new(n_billow.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, mp


def _make_speed_cloud_mat() -> bpy.types.Material:
    """Soft Fresnel-feathered wispy translucent speed cloud streak (visible_shadow=False)."""
    mat = bpy.data.materials.new("Mat_Env_SpeedCloud")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.96, 0.91, 0.84, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.95)
    _set_bsdf(bsdf, "Subsurface Weight", 0.30)
    _set_bsdf(bsdf, "Specular IOR Level", 0.04)

    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45

    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    inv_f.use_clamp = True
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 1.2
    n.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.use_clamp = True
    nt.links.new(inv_f.outputs["Value"], mul.inputs[0])
    nt.links.new(n.outputs["Fac"], mul.inputs[1])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.08
    mr.inputs["From Max"].default_value = 0.58
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 0.48
    nt.links.new(mul.outputs["Value"], mr.inputs["Value"])

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix_sh.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


# ============================================================================
# JET & COCKPIT GEOMETRY BUILDERS
# ============================================================================
def _loft_closed_rings(
    bm: bmesh.types.BMesh,
    rings: list[list[bmesh.types.BMVert]],
    mat_indices: list[int] | None = None,
    reverse_winding: bool = False,
) -> None:
    """Connect consecutive same-length vertex rings into quads."""
    for idx in range(len(rings) - 1):
        r0 = rings[idx]
        r1 = rings[idx + 1]
        n = len(r0)
        mi = mat_indices[idx] if mat_indices is not None else 0
        for j in range(n):
            jn = (j + 1) % n
            verts = [r0[j], r0[jn], r1[jn], r1[j]] if not reverse_winding else [r0[j], r1[j], r1[jn], r0[jn]]
            f = bm.faces.new(verts)
            f.material_index = mi


def _loft_open_u_strips(
    bm: bmesh.types.BMesh,
    strips: list[list[bmesh.types.BMVert]],
    mat_index: int = 0,
    reverse_winding: bool = False,
) -> None:
    """Connect consecutive open U-cross-section polylines (non-wrapping)."""
    for idx in range(len(strips) - 1):
        s0 = strips[idx]
        s1 = strips[idx + 1]
        for j in range(len(s0) - 1):
            verts = [s0[j], s0[j + 1], s1[j + 1], s1[j]] if not reverse_winding else [s0[j], s1[j], s1[j + 1], s0[j + 1]]
            f = bm.faces.new(verts)
            f.material_index = mat_index


def _build_fuselage(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_airframe: bpy.types.Material,
    mat_radome: bpy.types.Material,
) -> bpy.types.Object:
    """
    Build sleek supersonic twin-engine chined fighter fuselage matching C.JET:
    - nose_tip_y = 7.5, tail_end_y = -9.5, fuselage_center_z = -0.25, max_radius = 0.85
    - Clean open cockpit cutout across C.CANOPY['y_range'] = (-1.40, 1.60), |x| < 0.46, z >= 0.55
    - Explicit outward winding & zero coplanar overlap at y=1.60 and y=-1.40
    """
    bm = bmesh.new()
    y_nose = float(C.JET["nose_tip_y"])
    y_tail = float(C.JET["tail_end_y"])
    cz = float(C.JET["fuselage_center_z"])
    rmax = float(C.JET["fuselage_max_radius"])
    y_can_back, y_can_front = C.CANOPY["y_range"]
    sill_z = float(C.COCKPIT["sill_z"])
    half_w = float(C.COCKPIT["inner_half_width"])

    # --- SECTION A: Nose radome -> forward canopy sill (y = 7.50 -> 1.60) ---
    n_ring = 24
    fwd_ys = [7.46, 7.15, 6.60, 5.85, 5.10, 4.10, 3.00, 2.20, y_can_front]
    nose_tip_v = bm.verts.new((0.0, y_nose, cz - 0.08))

    fwd_rings: list[list[bmesh.types.BMVert]] = []
    for y in fwd_ys:
        u = max(0.02, min(1.0, (y_nose - y) / (y_nose - y_can_front)))
        s = math.sin(u * math.pi * 0.5) ** 0.78
        rx = 0.76 * s
        z_top = (cz - 0.08) + (sill_z - (cz - 0.08)) * (u ** 0.72)
        z_bot = (cz - 0.08) + (-0.98 - (cz - 0.08)) * (s ** 0.90)
        z_mid = 0.5 * (z_top + z_bot) - 0.04 * (1.0 - u)
        ring = []
        for j in range(n_ring):
            th = (2.0 * math.pi * j) / n_ring  # 0=+Z top, pi/2=+X right, pi=-Z bottom
            st, ct = math.sin(th), math.cos(th)
            chine_boost = 1.0 + 0.12 * (abs(st) ** 3) * min(1.0, u * 1.4)
            px = rx * st * chine_boost
            if ct >= 0.0:
                # Flatten top deck neatly to match cockpit sill width |x|<=0.50 at y_can_front
                flat_blend = 0.35 if y <= 2.3 else 0.72
                top_shape = ct ** flat_blend
                pz = z_mid + (z_top - z_mid) * top_shape
                if y <= y_can_front + 0.01 and abs(px) <= half_w + 0.05:
                    pz = sill_z
            else:
                pz = z_mid + (z_bot - z_mid) * ((-ct) ** 0.92)
            ring.append(bm.verts.new((px, y, pz)))
        fwd_rings.append(ring)

    # Cap radome tip cleanly (outward winding)
    r_first = fwd_rings[0]
    for j in range(n_ring):
        jn = (j + 1) % n_ring
        f = bm.faces.new([nose_tip_v, r_first[j], r_first[jn]])
        f.material_index = 1

    fwd_mats = [1 if y > 5.05 else 0 for y in fwd_ys[:-1]]
    _loft_closed_rings(bm, fwd_rings, fwd_mats, reverse_winding=True)
    # Single cap at y = 1.60 facing -Y (outward into cockpit footwell recess)
    bm.faces.new(list(reversed(fwd_rings[-1]))).material_index = 0

    # --- SECTION B: Open-Top U-Saddle Hull across Cockpit (y = 1.60 -> -1.40) ---
    # Meets cockpit tub cleanly at |x|=0.462, z=0.55 with ZERO inner overlap or end-caps!
    cockpit_ys = [y_can_front, 1.05, 0.50, 0.00, -0.50, -1.00, y_can_back]
    u_strips: list[list[bmesh.types.BMVert]] = []
    for y in cockpit_ys:
        t_aft = (y_can_front - y) / (y_can_front - y_can_back)
        chine_x = 0.76 + (rmax - 0.76) * t_aft
        chine_z = -0.22 - 0.04 * t_aft
        belly_z = -0.98 - 0.06 * t_aft
        pts = [
            (half_w + 0.002, y, sill_z),
            (half_w + 0.050, y, sill_z),
            (half_w + 0.145, y, 0.34),
            (chine_x, y, chine_z),
            (chine_x * 0.78, y, -0.68),
            (0.40, y, belly_z + 0.05),
            (0.0, y, belly_z),
            (-0.40, y, belly_z + 0.05),
            (-chine_x * 0.78, y, -0.68),
            (-chine_x, y, chine_z),
            (-(half_w + 0.145), y, 0.34),
            (-(half_w + 0.050), y, sill_z),
            (-(half_w + 0.002), y, sill_z),
        ]
        u_strips.append([bm.verts.new(p) for p in pts])

    # Explicit outward winding along U-channel (no end-caps -> zero coplanar overlap!)
    _loft_open_u_strips(bm, u_strips, mat_index=0, reverse_winding=True)

    # --- SECTION C: Dorsal Spine & Twin-Engine Rear Fuselage (y = -1.40 -> -9.50) ---
    n_aft = 28
    aft_ys = [y_can_back, -1.75, -2.50, -3.80, -5.40, -7.00, -8.30, -9.10, y_tail]
    aft_rings: list[list[bmesh.types.BMVert]] = []
    for y in aft_ys:
        t = (y_can_back - y) / (y_can_back - y_tail)
        rx = rmax * (0.98 - 0.14 * (t ** 2.2))
        if t < 0.12:
            z_top = 0.62 - 0.04 * (t / 0.12)
        else:
            z_top = 0.58 - 0.36 * ((t - 0.12) / 0.88) ** 1.2
        z_bot = -1.04 + 0.42 * (t ** 1.35)
        z_mid = 0.5 * (z_top + z_bot)

        ring = []
        for j in range(n_aft):
            th = (2.0 * math.pi * j) / n_aft
            st, ct = math.sin(th), math.cos(th)
            twin_bump = 0.08 * math.sin(t * math.pi) * (math.sin(th) ** 2 - 0.45 * (ct ** 2))
            chine = 1.0 + 0.10 * (abs(st) ** 3) * (1.0 - 0.5 * t)
            px = rx * st * chine
            if ct >= 0.0:
                pz = z_mid + (z_top - z_mid) * (ct ** 0.72) + twin_bump
            else:
                pz = z_mid + (z_bot - z_mid) * ((-ct) ** 0.88) - 0.5 * twin_bump
            ring.append(bm.verts.new((px, y, pz)))
        aft_rings.append(ring)

    aft_mats = [1 if y > -2.2 else 0 for y in aft_ys[:-1]]
    _loft_closed_rings(bm, aft_rings, aft_mats, reverse_winding=True)
    # Single cap at y = -1.40 facing +Y (outward) and tail cap at y = -9.50 facing -Y
    bm.faces.new(aft_rings[0]).material_index = 0
    bm.faces.new(list(reversed(aft_rings[-1]))).material_index = 1

    return _finalize_bmesh(
        bm,
        "Env_Jet_Fuselage",
        col,
        materials=[mat_airframe, mat_radome],
        smooth=True,
        recalc_normals=False,
        parent=jet_root,
    )


def _add_airfoil_wing_half(
    bm: bmesh.types.BMesh,
    side: float,
    stations: list[tuple[float, float, float, float, float]],
    mat_index: int = 0,
) -> None:
    """
    Loft a symmetric supersonic biconvex airfoil wing/fin between cross-section stations.
    Each station: (x, y_le, y_te, z, half_thickness).
    """
    n_prof = 12  # closed airfoil perimeter
    rings: list[list[bmesh.types.BMVert]] = []
    for x, y_le, y_te, z, ht in stations:
        chord = max(0.05, y_le - y_te)
        ring = []
        # Top surface LE -> TE, then bottom surface TE -> LE
        for k in range(n_prof // 2 + 1):
            u = k / (n_prof // 2)  # 0=LE, 1=TE
            y = y_le - chord * u
            thick = ht * math.sin(u * math.pi) ** 0.82
            ring.append(bm.verts.new((x, y, z + thick)))
        for k in range(1, n_prof // 2):
            u = 1.0 - k / (n_prof // 2)
            y = y_le - chord * u
            thick = ht * math.sin(u * math.pi) ** 0.82
            ring.append(bm.verts.new((x, y, z - thick)))
        rings.append(ring)

    _loft_closed_rings(bm, rings, [mat_index] * (len(rings) - 1))
    # Cap root & tip cleanly
    bm.faces.new(list(reversed(rings[0])) if side > 0 else rings[0]).material_index = mat_index
    bm.faces.new(rings[-1] if side > 0 else list(reversed(rings[-1]))).material_index = mat_index


def _build_wings_and_stabilators(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_airframe: bpy.types.Material,
    mat_radome: bpy.types.Material,
) -> bpy.types.Object:
    """Build swept delta/trapezoidal wings (C.JET), LERX strakes, tip rails & horizontal stabilators."""
    bm = bmesh.new()
    wz = float(C.JET["wing_z"])
    half_span = float(C.JET["wing_half_span"])
    y_te_root, y_le_root = C.JET["wing_y_range"]  # (-6.5, -1.5)

    for side in (-1.0, 1.0):
        # 1. Main swept trapezoidal supersonic wing
        wing_stations = [
            (side * 0.76, y_le_root, y_te_root, wz + 0.03, 0.095),
            (side * 2.10, -2.55, -6.50, wz + 0.01, 0.075),
            (side * 3.90, -3.85, -6.50, wz - 0.01, 0.052),
            (side * half_span, -4.90, -6.50, wz - 0.03, 0.032),
        ]
        _add_airfoil_wing_half(bm, side, wing_stations, mat_index=0)

        # 2. Leading-Edge Root Extension (LERX) blending intake shoulder into wing root
        lerx_stations = [
            (side * 0.72, 1.75, -1.50, wz + 0.24, 0.040),
            (side * 0.96, 0.65, -1.52, wz + 0.18, 0.035),
            (side * 1.18, -0.55, -1.55, wz + 0.12, 0.030),
            (side * 1.28, -1.45, -1.65, wz + 0.06, 0.025),
        ]
        _add_airfoil_wing_half(bm, side, lerx_stations, mat_index=0)

        # 3. Wingtip Sidewinder/AMRAAM launch rail
        _add_cyl(
            bm,
            (side * half_span, -6.40, wz - 0.03),
            (side * half_span, -3.85, wz - 0.03),
            0.036,
            0.028,
            segments=12,
            cap_ends=True,
            mat_index=1,
        )

        # 4. Horizontal stabilators (sleek swept aft tailplanes with -4 deg anhedral)
        stab_stations = [
            (side * 0.68, -6.85, -9.25, -0.26, 0.052),
            (side * 1.70, -7.85, -9.32, -0.33, 0.038),
            (side * 2.68, -8.60, -9.40, -0.40, 0.024),
        ]
        _add_airfoil_wing_half(bm, side, stab_stations, mat_index=0)

    return _finalize_bmesh(
        bm,
        "Env_Jet_Wings",
        col,
        materials=[mat_airframe, mat_radome],
        smooth=True,
        parent=jet_root,
    )


def _build_tail_fins(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_airframe: bpy.types.Material,
    mat_radome: bpy.types.Material,
) -> bpy.types.Object:
    """Build twin outward-canted (14 deg) supersonic vertical stabilizers matching C.JET."""
    bm = bmesh.new()
    y_te, y_le = C.JET["fin_y_range"]  # (-9.0, -6.0)
    top_z = float(C.JET["fin_top_z"])  # 3.2
    root_z = 0.22

    # Loft vertical airfoil sections across Z (instead of X)
    n_prof = 12
    for side in (-1.0, 1.0):
        z_levels = [
            (root_z, side * 0.56, y_le, y_te, 0.065),
            (1.10, side * 0.79, -6.70, -8.98, 0.052),
            (2.20, side * 1.06, -7.50, -8.96, 0.038),
            (top_z, side * 1.32, -8.05, -8.95, 0.024),
        ]
        rings: list[list[bmesh.types.BMVert]] = []
        for z, cx, yle, yte, ht in z_levels:
            chord = max(0.2, yle - yte)
            ring = []
            for k in range(n_prof // 2 + 1):
                u = k / (n_prof // 2)
                y = yle - chord * u
                t = ht * math.sin(u * math.pi) ** 0.80
                ring.append(bm.verts.new((cx + side * t, y, z)))
            for k in range(1, n_prof // 2):
                u = 1.0 - k / (n_prof // 2)
                y = yle - chord * u
                t = ht * math.sin(u * math.pi) ** 0.80
                ring.append(bm.verts.new((cx - side * t, y, z)))
            rings.append(ring)

        _loft_closed_rings(bm, rings, [0, 0, 1])
        bm.faces.new(rings[0] if side > 0 else list(reversed(rings[0]))).material_index = 0
        bm.faces.new(list(reversed(rings[-1])) if side > 0 else rings[-1]).material_index = 1

        # Fin-tip ECM pod fairing
        _add_cyl(
            bm,
            (side * 1.32, -9.05, top_z),
            (side * 1.32, -7.85, top_z),
            0.035,
            0.024,
            segments=10,
            cap_ends=True,
            mat_index=1,
        )

    return _finalize_bmesh(
        bm,
        "Env_Jet_TailFins",
        col,
        materials=[mat_airframe, mat_radome],
        smooth=True,
        parent=jet_root,
    )


def _build_intakes(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_airframe: bpy.types.Material,
    mat_radome: bpy.types.Material,
) -> bpy.types.Object:
    """Build twin supersonic side engine air intakes at intake_y = 1.8 with recessed turbine ducts."""
    bm = bmesh.new()
    iy = float(C.JET["intake_y"])

    for side in (-1.0, 1.0):
        # Swept caret intake duct cowling from y=1.80 back into nacelle fairing y=-3.20
        y_slices = [
            (iy, 0.72, 1.12, -0.02, -0.58),
            (iy - 0.45, 0.72, 1.15, 0.00, -0.60),
            (0.40, 0.72, 1.16, 0.02, -0.62),
            (-1.20, 0.68, 1.08, 0.02, -0.60),
            (-2.80, 0.56, 0.88, -0.02, -0.56),
            (-4.20, 0.46, 0.76, -0.06, -0.52),
        ]
        rings: list[list[bmesh.types.BMVert]] = []
        n_ring = 16
        for y, x_in, x_out, z_top, z_bot in y_slices:
            cx = side * 0.5 * (x_in + x_out)
            rx = 0.5 * (x_out - x_in)
            cz = 0.5 * (z_top + z_bot)
            rz = 0.5 * (z_top - z_bot)
            ring = []
            for j in range(n_ring):
                th = (2.0 * math.pi * j) / n_ring
                # Rounded-rectangular supersonic intake cross-section
                st, ct = math.sin(th), math.cos(th)
                sx = math.copysign(abs(st) ** 0.55, st)
                sz = math.copysign(abs(ct) ** 0.55, ct)
                ring.append(bm.verts.new((cx + rx * sx, y, cz + rz * sz)))
            rings.append(ring)

        _loft_closed_rings(bm, rings, [0] * (len(rings) - 1))
        # Recessed front intake splitter/throat depth (inset back to y=0.35 so front looks deep!)
        mouth = rings[0]
        throat = []
        cx_m = side * 0.92
        cz_m = -0.30
        for v in mouth:
            nx = cx_m + (v.co.x - cx_m) * 0.78
            nz = cz_m + (v.co.z - cz_m) * 0.78
            throat.append(bm.verts.new((nx, iy - 0.12, nz)))
        deep_face = [bm.verts.new((cx_m + (v.co.x - cx_m) * 0.62, 0.30, cz_m + (v.co.z - cz_m) * 0.62)) for v in mouth]
        _loft_closed_rings(bm, [mouth, throat, deep_face], [1, 1])
        bm.faces.new(list(reversed(deep_face))).material_index = 1
        bm.faces.new(rings[-1]).material_index = 0

    return _finalize_bmesh(
        bm,
        "Env_Jet_Intakes",
        col,
        materials=[mat_airframe, mat_radome],
        smooth=True,
        parent=jet_root,
    )


def _build_nozzles_and_afterburners(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_nozzle: bpy.types.Material,
    mat_core: bpy.types.Material,
    mat_outer: bpy.types.Material,
    burner_gain_nodes: list[bpy.types.ShaderNodeValue],
    total_frames: int,
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """
    Build twin titanium segmented exhaust nozzles at y = -9.5 plus open-ended
    additive Mach-1 afterburner cones (cap_ends=False, AGENTS.md Sec 3).
    """
    y_end = float(C.JET["tail_end_y"])  # -9.5
    cz = float(C.JET["fuselage_center_z"])  # -0.25

    bm_noz = bmesh.new()
    for sx in (-0.46, 0.46):
        # Outer shroud ring
        _add_cyl(bm_noz, (sx, y_end + 0.55, cz), (sx, y_end, cz), 0.39, 0.34, segments=20, cap_ends=False, mat_index=0)
        # Inner hot nozzle liner
        _add_cyl(bm_noz, (sx, y_end + 0.45, cz), (sx, y_end + 0.02, cz), 0.27, 0.31, segments=20, cap_ends=False, mat_index=0)
        # Recessed flameholder disk
        _add_cyl(bm_noz, (sx, y_end + 0.50, cz), (sx, y_end + 0.44, cz), 0.27, 0.27, segments=20, cap_ends=True, mat_index=0)

    nozzle_obj = _finalize_bmesh(
        bm_noz,
        "Env_Jet_Nozzles",
        col,
        materials=[mat_nozzle],
        smooth=True,
        parent=jet_root,
    )

    cones: list[bpy.types.Object] = []
    for label, sx in (("L", -0.46), ("R", 0.46)):
        bm_ab = bmesh.new()
        # Inner bright Mach-diamond core (open-ended, cap_ends=False!)
        core_len = 4.25
        rot_y = Matrix.Rotation(math.radians(90.0), 4, "X")
        mat_core_m = Matrix.Translation((sx, y_end - core_len * 0.5 + 0.08, cz)) @ rot_y
        res_c = bmesh.ops.create_cone(
            bm_ab,
            cap_ends=False,
            cap_tris=False,
            segments=24,
            radius1=0.05,
            radius2=0.28,
            depth=core_len,
            matrix=mat_core_m,
        )
        for v in res_c["verts"]:
            for f in v.link_faces:
                f.material_index = 0

        # Outer wider exhaust heat sheath (open-ended, cap_ends=False!)
        out_len = 5.35
        mat_out_m = Matrix.Translation((sx, y_end - out_len * 0.5 + 0.12, cz)) @ rot_y
        res_o = bmesh.ops.create_cone(
            bm_ab,
            cap_ends=False,
            cap_tris=False,
            segments=24,
            radius1=0.10,
            radius2=0.35,
            depth=out_len,
            matrix=mat_out_m,
        )
        for v in res_o["verts"]:
            for f in v.link_faces:
                f.material_index = 1

        cone_obj = _finalize_bmesh(
            bm_ab,
            f"Env_Afterburner_{label}",
            col,
            materials=[mat_core, mat_outer],
            smooth=True,
            recalc_normals=True,
            parent=jet_root,
        )
        cone_obj.visible_shadow = False
        cones.append(cone_obj)

    # Animate subtle supersonic afterburner flicker across 1..total_frames
    for gain_node in burner_gain_nodes:
        base_g = gain_node.outputs[0].default_value
        for f in range(1, total_frames + 1, 10):
            flicker = 1.0 + 0.08 * math.sin(f * 0.55) + 0.05 * math.cos(f * 1.15)
            # Slightly dial down unmanned jet burner late in Beat 6 dive
            fade = 1.0 - 0.35 * C.smoothstep(C.EV["rocket_burnout"], total_frames, f)
            gain_node.outputs[0].default_value = base_g * flicker * fade
            gain_node.outputs[0].keyframe_insert("default_value", frame=f)
        _set_fcurve_interp(gain_node.id_data, "LINEAR")

    return nozzle_obj, cones


def _build_cockpit_tub(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_tub_xray: bpy.types.Material,
    mat_panel: bpy.types.Material,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """
    Build cockpit tub interior matching C.COCKPIT:
    - floor_z = -0.30, sill_z = 0.55, inner_half_width = 0.46
    - Side consoles |x| = 0.31..0.44, top z = 0.255 (supports throttle at (-0.36, 0.20, 0.28))
    - Open top across y = -1.40..1.60 for canopy_fx.py
    - Rear bulkhead behind y = -1.05..-1.38 (100% clear of 15-deg rails & headbox!)
    """
    floor_z = float(C.COCKPIT["floor_z"])  # -0.30
    sill_z = float(C.COCKPIT["sill_z"])    # 0.55
    hw = float(C.COCKPIT["inner_half_width"])  # 0.46

    bm = bmesh.new()

    # 1. Cockpit floor deck (-0.30) spanning y = -1.06 .. +1.20
    _add_box(bm, (0.0, 0.07, floor_z - 0.02), (0.68, 2.26, 0.04), mat_index=0)

    # 2. Left & right lower tub side walls + sloped side console boxes (|x| = 0.31..0.44, z = -0.30..0.255)
    #    and upper cockpit inner side walls (|x| = 0.44..0.46, z = 0.255..0.55)
    for sx in (-1.0, 1.0):
        # Console shelf block
        _add_box(bm, (sx * 0.38, -0.08, 0.5 * (floor_z + 0.255)), (0.14, 1.96, 0.255 - floor_z), mat_index=0)
        # Upper cockpit side wall rising up to sill_z = 0.55
        _add_box(bm, (sx * (hw - 0.012), -0.05, 0.5 * (0.255 + sill_z)), (0.024, 2.38, sill_z - 0.255), mat_index=0)

    # 3. Angled rear cockpit bulkhead (y=-1.06 at floor -> y=-1.36 at sill; +19 deg leans aft parallel to 15-deg rails!)
    _add_box(
        bm,
        (0.0, -1.21, 0.5 * (floor_z + sill_z)),
        (hw * 1.94, 0.045, (sill_z - floor_z) * 1.08),
        mat_index=0,
        rot_euler=(math.radians(19.0), 0.0, 0.0),
    )

    # 4. Forward footwell pressure bulkhead behind pedals (y = 1.18)
    _add_box(bm, (0.0, 1.20, 0.5 * (floor_z + 0.58)), (hw * 1.90, 0.05, 0.88), mat_index=0)

    tub_obj = _finalize_bmesh(
        bm,
        "Env_Cockpit_Tub",
        col,
        materials=[mat_tub_xray],
        smooth=False,
        parent=jet_root,
    )

    # Compact console switch bezels & tactile panel lips along top of side consoles (opaque!)
    bm_sw = bmesh.new()
    for sx in (-1.0, 1.0):
        for ky in (-0.65, -0.32, 0.50, 0.74):
            _add_box(bm_sw, (sx * 0.375, ky, 0.264), (0.105, 0.22, 0.016), mat_index=0, bevel_width=0.003)
        # Small switches / knobs on right console at y=0.05..0.30 (left console y=0.20 reserved for throttle!)
        if sx > 0:
            _add_box(bm_sw, (sx * 0.375, 0.14, 0.264), (0.105, 0.28, 0.016), mat_index=0, bevel_width=0.003)
            for d_y in (0.05, 0.14, 0.23):
                _add_cyl(bm_sw, (sx * 0.37, d_y, 0.272), (sx * 0.37, d_y, 0.292), 0.012, 0.011, segments=10, mat_index=0)

    sw_obj = _finalize_bmesh(
        bm_sw,
        "Env_Cockpit_ConsoleSwitches",
        col,
        materials=[mat_panel],
        smooth=True,
        parent=jet_root,
    )
    return tub_obj, sw_obj


def _build_instrument_panel_and_alerts(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_panel: bpy.types.Material,
    mfd_mats: list[bpy.types.Material],
    mat_caution: bpy.types.Material,
    caution_pulse: bpy.types.ShaderNodeValue,
    total_frames: int,
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """
    Build forward instrument panel (panel_y=0.95, panel_z=0.18..0.62, glareshield_z=0.66),
    3 glowing tactical MFDs + UFC, and flashing red MASTER CAUTION annunciator (f=1..320).
    """
    py = float(C.COCKPIT["panel_y"])  # 0.95
    pz_lo, pz_hi = C.COCKPIT["panel_z_range"]  # (0.18, 0.62)
    gz = float(C.COCKPIT["glareshield_z"])  # 0.66

    # 1. Main panel chassis + anti-glare glareshield brow (opaque dark aerospace alloy)
    bm_p = bmesh.new()
    _add_box(bm_p, (0.0, py + 0.09, 0.5 * (pz_lo + pz_hi)), (0.78, 0.18, pz_hi - pz_lo), mat_index=0, bevel_width=0.008)
    # Anti-glare curved/sloped glareshield deck extending cleanly forward to canopy windscreen base y=1.56
    _add_box(bm_p, (0.0, py + 0.02, gz - 0.02), (0.82, 0.24, 0.04), mat_index=0, bevel_width=0.008)
    _add_box(
        bm_p,
        (0.0, 1.30, 0.60),
        (0.84, 0.54, 0.05),
        mat_index=0,
        rot_euler=(math.radians(-10.5), 0.0, 0.0),
        bevel_width=0.006,
    )
    # Recessed MFD outer bezels
    for bx, bz, bw, bh in [
        (0.0, 0.31, 0.18, 0.17),
        (-0.22, 0.35, 0.18, 0.18),
        (0.22, 0.35, 0.18, 0.18),
        (0.0, 0.51, 0.20, 0.12),
    ]:
        _add_box(bm_p, (bx, py - 0.006, bz), (bw, 0.014, bh), mat_index=0, bevel_width=0.003)

    panel_obj = _finalize_bmesh(
        bm_p,
        "Env_Cockpit_Panel",
        col,
        materials=[mat_panel],
        smooth=False,
        parent=jet_root,
    )

    # 2. Glowing MFD screens (slot 0=center, 1=left, 2=right, 3=UFC)
    bm_mfd = bmesh.new()
    _add_box(bm_mfd, (0.0, py - 0.012, 0.31), (0.155, 0.006, 0.145), mat_index=0)
    _add_box(bm_mfd, (-0.22, py - 0.012, 0.35), (0.155, 0.006, 0.155), mat_index=1)
    _add_box(bm_mfd, (0.22, py - 0.012, 0.35), (0.155, 0.006, 0.155), mat_index=2)
    _add_box(bm_mfd, (0.0, py - 0.012, 0.51), (0.175, 0.006, 0.095), mat_index=3)
    mfd_obj = _finalize_bmesh(
        bm_mfd,
        "Env_Cockpit_MFDs",
        col,
        materials=mfd_mats,
        smooth=False,
        parent=jet_root,
    )

    # 3. Flashing red MASTER CAUTION warning annunciator pads on glareshield brow
    bm_mc = bmesh.new()
    for sx in (-0.16, 0.16):
        _add_box(bm_mc, (sx, py - 0.09, gz - 0.025), (0.075, 0.018, 0.024), mat_index=0, bevel_width=0.003)
    caution_obj = _finalize_bmesh(
        bm_mc,
        "Env_Master_Caution",
        col,
        materials=[mat_caution],
        smooth=False,
        parent=jet_root,
    )

    # Animate MASTER CAUTION urgent 3.5 Hz flash across alarm_start=1..320 (AgX-safe <= 2.1!)
    for f in range(1, 330, 3):
        if f <= 320:
            s_pulse = 0.5 + 0.5 * math.sin((f - C.EV["alarm_start"]) * (2.0 * math.pi * 3.5 / C.FPS))
            val = 0.15 + 1.95 * (s_pulse ** 1.6)
        else:
            val = 0.0
        caution_pulse.outputs[0].default_value = val
        caution_pulse.outputs[0].keyframe_insert("default_value", frame=f)
    caution_pulse.outputs[0].default_value = 0.0
    caution_pulse.outputs[0].keyframe_insert("default_value", frame=335)
    caution_pulse.outputs[0].keyframe_insert("default_value", frame=total_frames)
    _set_fcurve_interp(caution_pulse.id_data, "LINEAR")

    return panel_obj, mfd_obj, caution_obj


def _build_hud(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_panel: bpy.types.Material,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Build machined HUD combiner frame + reflective emerald glass at C.COCKPIT['hud_combiner']."""
    hx, hy, hz = C.COCKPIT["hud_combiner"]  # (0.0, 0.98, 0.78)
    gz = float(C.COCKPIT["glareshield_z"])

    # Combiner frame struts rising from glareshield
    bm_f = bmesh.new()
    for sx in (-0.105, 0.105):
        _add_box(bm_f, (hx + sx, hy + 0.02, 0.5 * (gz + hz - 0.03)), (0.016, 0.045, hz - gz + 0.02), mat_index=0, bevel_width=0.002)
        _add_box(
            bm_f,
            (hx + sx, hy, hz),
            (0.012, 0.018, 0.14),
            mat_index=0,
            rot_euler=(math.radians(38.0), 0.0, 0.0),
            bevel_width=0.002,
        )
    # Top slim curved brow arch
    _add_box(bm_f, (hx, hy - 0.042, hz + 0.056), (0.22, 0.014, 0.012), mat_index=0, bevel_width=0.002)
    # Base HUD collimator lens housing
    _add_cyl(bm_f, (hx, hy + 0.02, gz - 0.01), (hx, hy + 0.02, gz + 0.02), 0.068, 0.064, segments=20, mat_index=0)

    hud_frame = _finalize_bmesh(
        bm_f,
        "Env_HUD_Frame",
        col,
        materials=[mat_panel],
        smooth=True,
        parent=jet_root,
    )

    # Combiner glass pane (material attached BEFORE bm.to_mesh per AGENTS.md Sec 3)
    hud_mat, hud_tc = _make_hud_combiner_mat()
    bm_g = bmesh.new()
    _add_box(bm_g, (0.0, 0.0, 0.0), (0.195, 0.004, 0.132), mat_index=0)
    hud_glass = _finalize_bmesh(
        bm_g,
        "Env_HUD_Glass",
        col,
        materials=[hud_mat],
        smooth=False,
        parent=jet_root,
        location=(hx, hy, hz),
        rotation=(math.radians(38.0), 0.0, 0.0),
    )
    hud_tc.object = hud_glass
    hud_glass.visible_shadow = False
    return hud_frame, hud_glass


def _build_cockpit_controls(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    mat_panel: bpy.types.Material,
    mat_grip: bpy.types.Material,
    mat_red_accent: bpy.types.Material,
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """
    Build HOTAS centre stick (0, 0.55, 0.05), left throttle (-0.36, 0.20, 0.28),
    and twin rudder pedals (x=+-0.13, y=0.80, z=-0.10).
    """
    floor_z = float(C.COCKPIT["floor_z"])  # -0.30
    sx, sy, sz = C.COCKPIT["stick"]        # (0.0, 0.55, 0.05)

    # 1. Centre Control Stick
    bm_s = bmesh.new()
    # Floor dust boot
    _add_cyl(bm_s, (sx, sy, floor_z), (sx, sy, floor_z + 0.11), 0.055, 0.025, segments=16, mat_index=1)
    # Lower alloy shaft rising toward grip
    _add_cyl(bm_s, (sx, sy, floor_z + 0.10), (sx, sy - 0.01, sz - 0.05), 0.016, 0.015, segments=14, mat_index=0)
    # Ergonomic sculpted HOTAS grip centered right at C.COCKPIT['stick']
    _add_cyl(bm_s, (sx, sy - 0.01, sz - 0.055), (sx, sy + 0.01, sz + 0.055), 0.021, 0.024, segments=16, mat_index=1)
    # Flared palm rest & top thumb control head
    _add_box(bm_s, (sx, sy - 0.012, sz - 0.045), (0.048, 0.036, 0.014), mat_index=1, bevel_width=0.003)
    _add_box(bm_s, (sx, sy + 0.006, sz + 0.062), (0.044, 0.056, 0.032), mat_index=1, bevel_width=0.006)
    # Red weapon pickle button
    _add_cyl(bm_s, (sx + 0.010, sy + 0.008, sz + 0.072), (sx + 0.010, sy - 0.020, sz + 0.072), 0.006, 0.005, segments=10, mat_index=2)
    stick_obj = _finalize_bmesh(
        bm_s,
        "Env_Control_Stick",
        col,
        materials=[mat_panel, mat_grip, mat_red_accent],
        smooth=True,
        parent=jet_root,
    )

    # 2. Left Console Throttle Quadrant & Split Grip
    tx, ty, tz = C.COCKPIT["throttle"]  # (-0.36, 0.20, 0.28)
    bm_t = bmesh.new()
    _add_box(bm_t, (tx, ty, 0.264), (0.085, 0.24, 0.022), mat_index=0, bevel_width=0.004)
    # Dual throttle arms + ergonomic cylindrical split grips centered at (tx, ty, tz)
    for ox in (-0.018, 0.018):
        _add_box(bm_t, (tx + ox, ty - 0.015, 0.272), (0.014, 0.026, 0.024), mat_index=0)
    _add_cyl(bm_t, (tx - 0.042, ty, tz), (tx + 0.042, ty, tz), 0.020, 0.020, segments=14, mat_index=1)
    throttle_obj = _finalize_bmesh(
        bm_t,
        "Env_Throttle",
        col,
        materials=[mat_panel, mat_grip],
        smooth=True,
        parent=jet_root,
    )

    # 3. Twin Rudder Pedals (x = +-0.13, y = 0.80, z = -0.10)
    ped_y = float(C.COCKPIT["pedals_y"])  # 0.80
    bm_r = bmesh.new()
    for px in (-0.13, 0.13):
        # Angled ribbed cast-aluminum pedal plate
        _add_box(
            bm_r,
            (px, ped_y + 0.01, -0.10),
            (0.095, 0.020, 0.135),
            mat_index=0,
            rot_euler=(math.radians(-18.0), 0.0, 0.0),
            bevel_width=0.003,
        )
        # Heel stirrup bar + linkage rod forward to bulkhead y=1.18
        _add_box(bm_r, (px, ped_y - 0.015, -0.165), (0.095, 0.036, 0.012), mat_index=1, bevel_width=0.002)
        _add_cyl(bm_r, (px, ped_y + 0.02, -0.09), (px, 1.18, -0.05), 0.011, 0.011, segments=10, mat_index=0)
    pedals_obj = _finalize_bmesh(
        bm_r,
        "Env_Rudder_Pedals",
        col,
        materials=[mat_panel, mat_grip],
        smooth=True,
        parent=jet_root,
    )

    return stick_obj, throttle_obj, pedals_obj


# ============================================================================
# WORLD SKY, LIGHTING RIG, 3D CLOUD DECK & KINETIC SPEED CLOUDS
# ============================================================================
def _build_world_and_lights(
    col: bpy.types.Collection,
    jet_root: bpy.types.Object,
    total_frames: int,
) -> tuple[bpy.types.World, dict[str, bpy.types.Object]]:
    """
    Build physical Nishita golden-hour high-altitude sky aligned with C.SUN_TOWARD,
    mandatory AGENTS.md Sec 1 0.42m AREA eye catchlight aimed at C.PILOT['eye_mid'],
    warm cockpit fill, cool sky rim, Beat 5-6 high-altitude fill, and flashing red alarm light.
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Env_GoldenHour_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = True
    sky.sun_size = math.radians(1.0)
    sky.sun_intensity = 0.52

    # Align Nishita elevation & rotation directly with C.SUN_TOWARD = (-0.42, 0.80, 0.43)
    sun_vec = Vector(C.SUN_TOWARD).normalized()
    sky.sun_elevation = math.asin(max(-1.0, min(1.0, sun_vec.z)))
    sky.sun_rotation = math.atan2(-sun_vec.x, -sun_vec.y)
    sky.altitude = 4200.0     # crisp high-altitude fighter cruise sky
    sky.air_density = 1.55
    sky.dust_density = 1.75   # rich golden-hour atmospheric scattering
    sky.ozone_density = 2.35  # deeper Top-Gun sapphire zenith contrast
    bg.inputs["Strength"].default_value = 0.17  # calibrated for AgX exposure=0.0

    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if scene.render.engine == "CYCLES":
        scene.cycles.transparent_max_bounces = max(14, scene.cycles.transparent_max_bounces)

    def _add_light(
        name: str,
        ltype: str,
        loc: tuple[float, float, float],
        rot,
        energy: float,
        color: tuple[float, float, float],
        size: float,
        parent: bpy.types.Object | None = None,
    ) -> bpy.types.Object:
        ld = bpy.data.lights.new(f"{name}_Data", type=ltype)
        ld.energy = energy
        ld.color = color
        if ltype == "AREA":
            ld.size = size
        elif ltype == "POINT":
            ld.shadow_soft_size = size
        obj = bpy.data.objects.new(name, ld)
        obj.location = Vector(loc)
        obj.rotation_euler = rot if isinstance(rot, Euler) else Euler(rot, "XYZ")
        obj.visible_camera = False
        if parent is not None:
            obj.parent = parent
        _link_obj(obj, col)
        return obj

    eye_mid = Vector(C.PILOT["eye_mid"])  # (0.0, 0.02, 0.81)

    # 1. AGENTS.md Sec 1 Mandatory 0.42m AREA Softbox Eye Catchlight (~2.15m distance -> pinpoint corneal glint!)
    catch_loc = (0.95, 1.85, 1.62)
    catch_light = _add_light(
        "Env_Pilot_EyeCatchlight",
        "AREA",
        catch_loc,
        _aim_euler(catch_loc, eye_mid),
        energy=46.0,
        color=(1.0, 0.96, 0.90),
        size=0.42,
    )
    if hasattr(catch_light.data, "diffuse_factor"):
        catch_light.data.diffuse_factor = 0.09
    if hasattr(catch_light.data, "specular_factor"):
        catch_light.data.specular_factor = 1.78

    # 2. Warm Cockpit Bounce Fill (soft sculpting on face, stick, seat & handle without blowing out nose/sill)
    fill_loc = (1.35, 1.45, 1.30)
    cockpit_fill = _add_light(
        "Env_Cockpit_WarmFill",
        "AREA",
        fill_loc,
        _aim_euler(fill_loc, (0.0, 0.08, 0.45)),
        energy=12.0,
        color=(1.0, 0.82, 0.58),
        size=1.65,
    )

    # 3. Cool Cerulean Sky Rim Light opposite sun (separates helmet, shoulders & sill)
    rim_loc = (1.90, -1.75, 1.95)
    sky_rim = _add_light(
        "Env_Sky_CoolRim",
        "AREA",
        rim_loc,
        _aim_euler(rim_loc, (0.0, -0.05, 0.62)),
        energy=62.0,
        color=(0.58, 0.78, 1.0),
        size=2.20,
    )

    # 4. High-Altitude Soft Sky Fill for Beats 5-6 (keeps soaring seat & parachute pilot crisp at Z=12m!)
    hi_loc = (3.8, 4.2, 14.5)
    high_alt_fill = _add_light(
        "Env_Pilot_HighAltFill",
        "AREA",
        hi_loc,
        _aim_euler(hi_loc, (0.25, -1.40, 10.5)),
        energy=82.0,
        color=(0.80, 0.90, 1.0),
        size=4.2,
    )

    # 5. Flashing Red MASTER CAUTION Cockpit Alarm Light (parented to Env_Jet_Root, f=1..320)
    alarm_light = _add_light(
        "Env_Cockpit_AlarmLight",
        "POINT",
        (0.0, 0.72, 0.60),
        (0.0, 0.0, 0.0),
        energy=26.0,
        color=(1.0, 0.06, 0.04),
        size=0.08,
        parent=jet_root,
    )
    for f in range(1, 330, 3):
        if f <= 320:
            s_pulse = 0.5 + 0.5 * math.sin((f - C.EV["alarm_start"]) * (2.0 * math.pi * 3.5 / C.FPS))
            en = 1.5 + 26.5 * (s_pulse ** 1.6)
        else:
            en = 0.0
        alarm_light.data.energy = en
        alarm_light.data.keyframe_insert("energy", frame=f)
    alarm_light.data.energy = 0.0
    alarm_light.data.keyframe_insert("energy", frame=335)
    alarm_light.data.keyframe_insert("energy", frame=total_frames)
    _set_fcurve_interp(alarm_light.data, "LINEAR")

    return world, {
        "eye_catchlight": catch_light,
        "cockpit_fill": cockpit_fill,
        "sky_rim": sky_rim,
        "high_alt_fill": high_alt_fill,
        "alarm_light": alarm_light,
    }


def _build_cloud_deck(
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Build sunlit multi-bank 3D cloud deck centered around C.CLOUD_DECK_Z = -60.0
    spanning [-2600..2600] x [-2600..2600] via signed-quadratic grid density +
    gentle outer curvature drop and 360-deg cumulus banks so Beats 5-6 have zero horizon cutoff.
    """
    deck_z = float(C.CLOUD_DECK_Z)  # -60.0
    mat_stratus, mp_stratus = _make_cloud_deck_mat("Mat_Env_CloudStratus", is_cumulus=False)
    mat_cumulus, mp_cumulus = _make_cloud_deck_mat("Mat_Env_CloudCumulus", is_cumulus=True)

    # 1. Wide horizon-spanning continuous stratus cloud sea ([-2600..2600]^2)
    bm_s = bmesh.new()
    nx, ny = 64, 64
    half_extent = 2600.0
    grid = []
    for iy in range(ny + 1):
        row = []
        v = -1.0 + 2.0 * (iy / ny)
        v_y = v * (0.22 + 0.78 * abs(v)) * half_extent
        for ix in range(nx + 1):
            u = -1.0 + 2.0 * (ix / nx)
            v_x = u * (0.22 + 0.78 * abs(u)) * half_extent
            r_norm = math.hypot(v_x, v_y) / half_extent
            ridge_atten = max(0.25, 1.0 - 0.60 * min(1.0, r_norm))
            ridge = ridge_atten * (
                2.2 * math.sin(v_x * 0.028 + v_y * 0.014)
                + 1.6 * math.cos(v_y * 0.032 - v_x * 0.018)
                + 0.8 * math.sin(v_x * 0.065 + v_y * 0.055)
            )
            drop = -28.0 * (max(0.0, r_norm - 0.55) ** 2)
            row.append(bm_s.verts.new((v_x, v_y, deck_z - 3.5 + ridge + drop)))
        grid.append(row)

    for iy in range(ny):
        for ix in range(nx):
            bm_s.faces.new([grid[iy][ix], grid[iy][ix + 1], grid[iy + 1][ix + 1], grid[iy + 1][ix]])

    stratus_obj = _finalize_bmesh(
        bm_s,
        "Env_CloudDeck_Stratus",
        col,
        materials=[mat_stratus],
        smooth=True,
        recalc_normals=False,
    )
    stratus_obj.visible_shadow = False

    # 2. Multi-bank sculpted sunlit cumulus clouds across near + outer 360-deg horizon
    #    Keep central diving-jet path (|x| < 16, y in [-15..175]) safely clear!
    rng = random.Random(42)
    bm_c = bmesh.new()
    bank_centers = []
    for i in range(60):
        if i < 36:
            cx = rng.uniform(-240.0, 240.0)
            cy = rng.uniform(-180.0, 360.0)
            base_r = rng.uniform(14.0, 26.0)
            cz = deck_z - rng.uniform(0.5, 2.8)
        else:
            ang = rng.uniform(0.0, 2.0 * math.pi)
            dist = rng.uniform(340.0, 1450.0)
            cx = math.cos(ang) * dist
            cy = math.sin(ang) * dist
            base_r = rng.uniform(42.0, 88.0)
            r_norm = dist / half_extent
            cz = deck_z - 2.0 - 28.0 * (max(0.0, r_norm - 0.55) ** 2)
        if abs(cx) < 18.0 and -20.0 < cy < 185.0:
            cx = 28.0 if cx >= 0 else -28.0
        bank_centers.append((cx, cy, cz, base_r))

    for cx, cy, cz, base_r in bank_centers:
        n_puffs = rng.randint(4, 6)
        for p_idx in range(n_puffs):
            ox = cx + rng.uniform(-1.1, 1.1) * base_r
            oy = cy + rng.uniform(-1.1, 1.1) * base_r
            oz = cz + rng.uniform(-0.8, 2.4)
            rad = base_r * rng.uniform(0.55, 0.95)
            tmp = bmesh.new()
            bmesh.ops.create_icosphere(tmp, subdivisions=2, radius=1.0)
            for v in tmp.verts:
                disp = 1.0 + 0.16 * math.sin(v.co.x * 3.1 + p_idx) * math.cos(v.co.y * 2.7)
                v.co.x = ox + v.co.x * rad * 1.25 * disp
                v.co.y = oy + v.co.y * rad * 1.45 * disp
                v.co.z = oz + v.co.z * rad * 0.30 * disp
            vmap = {v: bm_c.verts.new(v.co) for v in tmp.verts}
            for f in tmp.faces:
                bm_c.faces.new([vmap[v] for v in f.verts])
            tmp.free()

    cumulus_obj = _finalize_bmesh(
        bm_c,
        "Env_CloudDeck_Cumulus",
        col,
        materials=[mat_cumulus],
        smooth=True,
    )
    cumulus_obj.visible_shadow = False

    # Animate subtle high-altitude cloud-sea drift along -Y via shader Mapping.Location
    for mp in (mp_stratus, mp_cumulus):
        mp.inputs["Location"].default_value = (0.0, 0.0, 0.0)
        mp.inputs["Location"].keyframe_insert("default_value", frame=1)
        mp.inputs["Location"].default_value = (0.0, 260.0, 0.0)  # positive texture shift scrolls clouds -Y
        mp.inputs["Location"].keyframe_insert("default_value", frame=total_frames)
        _set_fcurve_interp(mp.id_data, "LINEAR")

    return [stratus_obj, cumulus_obj]


def _build_speed_clouds(
    col: bpy.types.Collection,
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Build 18 elongated soft-edge wispy cloud streaks (Env_SpeedCloud_01..18)
    streaking rapidly in -Y past the jet during Beats 1-4 (f=1..1236) in perimeter
    corridors clear of the cockpit camera & labels, then fading & CONSTANT-hidden >= 1237.
    """
    mat_speed = _make_speed_cloud_mat()
    rng = random.Random(108)
    streaks: list[bpy.types.Object] = []

    rail_exit = int(C.EV["rail_exit"])  # 1236
    y_spawn = 92.0
    y_despawn = -78.0
    travel_span = y_spawn - y_despawn  # 170 m

    for idx in range(18):
        # 11 on -X side (clean background for +X camera), 7 in +X outer periphery
        if idx < 11:
            px = rng.uniform(-28.0, -8.5)
            pz = rng.uniform(-12.0, 11.0)
        else:
            px = rng.uniform(11.0, 28.0)
            pz = rng.uniform(-14.0, -4.5) if (idx % 2 == 0) else rng.uniform(6.0, 14.0)

        length_y = rng.uniform(7.5, 14.5)
        width_x = rng.uniform(1.4, 2.8)
        thick_z = rng.uniform(0.45, 0.90)

        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        for v in bm.verts:
            wisp = 1.0 + 0.18 * math.sin(v.co.y * 4.0 + idx) * math.cos(v.co.x * 3.5)
            v.co.x *= width_x * 0.5 * wisp
            v.co.y *= length_y * 0.5
            v.co.z *= thick_z * 0.5 * wisp

        obj = _finalize_bmesh(
            bm,
            f"Env_SpeedCloud_{idx + 1:02d}",
            col,
            materials=[mat_speed],
            smooth=True,
            location=(px, y_spawn, pz),
        )
        obj.visible_shadow = False

        # Staggered kinetic Mach-1 passes at ~1.45 m/frame (~87 m/s) across f=1..1236
        # Reserve 2 hidden wrap frames between consecutive passes so motion blur never streaks across frame
        speed_per_f = rng.uniform(1.25, 1.65)
        pass_frames = max(60, int(round(travel_span / speed_per_f)))
        pass_cycle = pass_frames + 2
        phase_f = int((idx * pass_cycle) / 18)

        vis_ranges: list[tuple[int, int]] = []
        t_start = 1 - phase_f
        while t_start <= rail_exit:
            f0 = max(1, t_start)
            t_end = t_start + pass_frames - 1
            f1 = min(rail_exit, t_end)
            if f1 > f0 + 1:
                u0 = (f0 - t_start) / float(pass_frames - 1)
                u1 = (f1 - t_start) / float(pass_frames - 1)
                y0 = y_spawn - travel_span * u0
                y1 = y_spawn - travel_span * u1

                obj.location = Vector((px, y0, pz))
                obj.keyframe_insert("location", frame=f0)
                obj.location = Vector((px, y1, pz))
                obj.keyframe_insert("location", frame=f1)

                fade_0 = 0.05 if f0 == t_start else (1.0 - C.smoothstep(rail_exit - 24, rail_exit, f0))
                fade_1 = 0.05 if (f1 == t_end or f1 >= rail_exit - 8) else (1.0 - C.smoothstep(rail_exit - 24, rail_exit, f1))
                obj.scale = Vector((fade_0, fade_0, fade_0))
                obj.keyframe_insert("scale", frame=f0)
                if f0 + 12 < f1 - 4:
                    f_mid_in = f0 + 10
                    s_in = 1.0 - C.smoothstep(rail_exit - 24, rail_exit, f_mid_in)
                    obj.scale = Vector((s_in, s_in, s_in))
                    obj.keyframe_insert("scale", frame=f_mid_in)
                    if f1 - 12 > f_mid_in:
                        f_mid_out = f1 - 10
                        s_out = 1.0 - C.smoothstep(rail_exit - 24, rail_exit, f_mid_out)
                        obj.scale = Vector((s_out, s_out, s_out))
                        obj.keyframe_insert("scale", frame=f_mid_out)
                obj.scale = Vector((fade_1, fade_1, fade_1))
                obj.keyframe_insert("scale", frame=f1)

                vis_ranges.append((f0, f1))
            t_start += pass_cycle

        _set_fcurve_interp(obj, "LINEAR", prefix="location")
        _set_fcurve_interp(obj, "LINEAR", prefix="scale")
        _key_visibility(obj, vis_ranges, total_frames)
        streaks.append(obj)

    return streaks


# ============================================================================
# ANIMATION DRIVERS: Env_Jet_Root + XRayFuselage
# ============================================================================
def _animate_jet_root(jet_root: bpy.types.Object, total_frames: int) -> None:
    """
    Keyframe Env_Jet_Root across 1..total_frames matching C.jet_offset(f):
    - Exact (0, 0, 0) across 1..C.EV['rail_exit'] (1236)
    - Every frame 1236..total_frames with LINEAR interpolation for 0.000mm error!
    """
    rail_exit = int(C.EV["rail_exit"])  # 1236
    jet_root.location = Vector((0.0, 0.0, 0.0))
    jet_root.keyframe_insert("location", frame=1)
    jet_root.keyframe_insert("location", frame=rail_exit)

    for f in range(rail_exit + 1, total_frames + 1):
        jet_root.location = Vector(C.jet_offset(f))
        jet_root.keyframe_insert("location", frame=f)

    _set_fcurve_interp(jet_root, "LINEAR", prefix="location")


def _animate_xray_fuselage(
    xray_value_nodes: list[bpy.types.ShaderNodeValue],
    total_frames: int,
) -> None:
    """Keyframe XRayFuselage ShaderNodeValue outputs directly from C.xray_fuselage_weight(f)."""
    in0, in1, out0, out1 = C.XRAY_FUSELAGE
    key_frames = (
        [1, max(1, in0 - 1)]
        + list(range(in0, in1 + 1))
        + [max(in1 + 1, out0 - 1)]
        + list(range(out0, out1 + 1))
        + [min(total_frames, out1 + 1), total_frames]
    )
    key_frames = sorted({f for f in key_frames if 1 <= f <= total_frames})

    for node in xray_value_nodes:
        for f in key_frames:
            node.outputs[0].default_value = C.xray_fuselage_weight(f)
            node.outputs[0].keyframe_insert("default_value", frame=f)
        _set_fcurve_interp(node.id_data, "LINEAR")


# ============================================================================
# PUBLIC ENTRYPOINT
# ============================================================================
def build_environment(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict:
    """Build complete supersonic fighter jet, cockpit tub, controls, X-ray fuselage & golden-hour sky."""
    if col is None:
        col = bpy.context.scene.collection

    # 1. Jet Root Empty (origin = SRP (0,0,0), animated via C.jet_offset)
    jet_root = bpy.data.objects.new("Env_Jet_Root", None)
    jet_root.empty_display_type = "PLAIN_AXES"
    jet_root.empty_display_size = 0.65
    jet_root.location = Vector((0.0, 0.0, 0.0))
    _link_obj(jet_root, col)
    _animate_jet_root(jet_root, total_frames)

    # 2. Shared X-Ray Fuselage & Opaque Cockpit Materials
    xray_value_nodes: list[bpy.types.ShaderNodeValue] = []
    mat_airframe = _make_xray_fuselage_material(
        "Mat_Env_JetAirframe_XRay",
        solid_rgb=(0.042, 0.048, 0.058),
        metallic=0.24,
        roughness=0.60,
        jet_root=jet_root,
        xray_value_nodes=xray_value_nodes,
        panel_brick_scale=1.0,
    )
    mat_radome = _make_xray_fuselage_material(
        "Mat_Env_JetRadome_XRay",
        solid_rgb=(0.038, 0.044, 0.052),
        metallic=0.28,
        roughness=0.52,
        jet_root=jet_root,
        xray_value_nodes=xray_value_nodes,
        panel_brick_scale=1.6,
    )
    mat_tub_xray = _make_xray_fuselage_material(
        "Mat_Env_CockpitTub_XRay",
        solid_rgb=(0.024, 0.028, 0.034),
        metallic=0.35,
        roughness=0.62,
        jet_root=jet_root,
        xray_value_nodes=xray_value_nodes,
        panel_brick_scale=2.2,
        xray_rim_strength=1.55,
    )

    mat_panel = _make_opaque_metal_mat("Mat_Env_CockpitPanel", (0.022, 0.025, 0.030), metallic=0.68, roughness=0.48, jet_root=jet_root)
    mat_grip = _make_opaque_metal_mat("Mat_Env_HOTASGrip", (0.032, 0.034, 0.038), metallic=0.12, roughness=0.66, jet_root=jet_root)
    mat_nozzle = _make_opaque_metal_mat("Mat_Env_ExhaustTitanium", (0.08, 0.078, 0.085), metallic=0.92, roughness=0.34, jet_root=jet_root)
    mat_red_accent = _make_opaque_metal_mat("Mat_Env_RedPickle", (0.58, 0.04, 0.03), metallic=0.10, roughness=0.38, jet_root=jet_root)

    mfd_mats = [
        _make_mfd_screen_mat("Mat_Env_MFD_Center", "center", jet_root),
        _make_mfd_screen_mat("Mat_Env_MFD_Left", "left", jet_root),
        _make_mfd_screen_mat("Mat_Env_MFD_Right", "right", jet_root),
        _make_mfd_screen_mat("Mat_Env_MFD_UFC", "ufc", jet_root),
    ]
    mat_caution, caution_pulse = _make_master_caution_mat()

    burner_gain_nodes: list[bpy.types.ShaderNodeValue] = []
    mat_ab_core, ab_core_gain = _make_afterburner_plume_mat("Mat_Env_AfterburnerCore", jet_root, is_core=True)
    mat_ab_outer, ab_outer_gain = _make_afterburner_plume_mat("Mat_Env_AfterburnerOuter", jet_root, is_core=False)
    burner_gain_nodes.extend([ab_core_gain, ab_outer_gain])

    # 3. Jet Exterior Airframe & Propulsion (parented to Env_Jet_Root)
    fuselage = _build_fuselage(col, jet_root, mat_airframe, mat_radome)
    wings = _build_wings_and_stabilators(col, jet_root, mat_airframe, mat_radome)
    tail_fins = _build_tail_fins(col, jet_root, mat_airframe, mat_radome)
    intakes = _build_intakes(col, jet_root, mat_airframe, mat_radome)
    nozzles, afterburner_cones = _build_nozzles_and_afterburners(
        col, jet_root, mat_nozzle, mat_ab_core, mat_ab_outer, burner_gain_nodes, total_frames
    )

    # 4. Cockpit Tub, Instruments, HUD & HOTAS Controls (parented to Env_Jet_Root)
    cockpit_tub, console_switches = _build_cockpit_tub(col, jet_root, mat_tub_xray, mat_panel)
    instrument_panel, mfd_screens, master_caution = _build_instrument_panel_and_alerts(
        col, jet_root, mat_panel, mfd_mats, mat_caution, caution_pulse, total_frames
    )
    hud_frame, hud_glass = _build_hud(col, jet_root, mat_panel)
    control_stick, throttle, rudder_pedals = _build_cockpit_controls(
        col, jet_root, mat_panel, mat_grip, mat_red_accent
    )

    # 5. Animate XRayFuselage window across 320..1236
    _animate_xray_fuselage(xray_value_nodes, total_frames)

    # 6. World Sky, Studio Lights & Clouds
    world, lights = _build_world_and_lights(col, jet_root, total_frames)
    cloud_deck = _build_cloud_deck(col, total_frames)
    speed_clouds = _build_speed_clouds(col, total_frames)

    return {
        "jet_root": jet_root,
        "fuselage": fuselage,
        "wings": wings,
        "tail_fins": tail_fins,
        "intakes": intakes,
        "nozzles": nozzles,
        "afterburner_cones": afterburner_cones,
        "cockpit_tub": cockpit_tub,
        "console_switches": console_switches,
        "instrument_panel": instrument_panel,
        "mfd_screens": mfd_screens,
        "master_caution": master_caution,
        "hud_frame": hud_frame,
        "hud_glass": hud_glass,
        "control_stick": control_stick,
        "throttle": throttle,
        "rudder_pedals": rudder_pedals,
        "cloud_deck": cloud_deck,
        "speed_clouds": speed_clouds,
        "world": world,
        "lights": lights,
        "xray_value_nodes": xray_value_nodes,
        "materials": {
            "airframe_xray": mat_airframe,
            "radome_xray": mat_radome,
            "cockpit_tub_xray": mat_tub_xray,
            "panel": mat_panel,
            "grip": mat_grip,
            "nozzle": mat_nozzle,
            "master_caution": mat_caution,
            "afterburner_core": mat_ab_core,
            "afterburner_outer": mat_ab_outer,
        },
    }


# ============================================================================
# STANDALONE CPU SELF-TEST (runs only when executed directly)
# ============================================================================
if __name__ == "__main__":
    import os

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 6
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 540
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 50
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES

    env_col = bpy.data.collections.new("Env_Collection")
    scene.collection.children.link(env_col)
    env_info = build_environment(env_col, total_frames=C.TOTAL_FRAMES)

    cam_data = bpy.data.cameras.new("Env_TestCam_Data")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new("Env_TestCam", cam_data)
    _link_obj(cam_obj, env_col)
    scene.camera = cam_obj

    out_dir = "/tmp/fje_env"
    os.makedirs(out_dir, exist_ok=True)

    # Frame 100: Beat 1 cockpit front-right 3/4 view (solid jet + MFDs + MASTER CAUTION)
    # Frame 760: Beat 3 side 3/4 X-ray view (translucent cyan cockpit + internal controls)
    for test_f, cam_loc, tgt in [
        (100, (0.95, 1.85, 1.15), (0.0, 0.35, 0.35)),
        (760, (3.20, 0.85, 0.65), (0.0, 0.25, 0.05)),
    ]:
        cam_obj.location = Vector(cam_loc)
        cam_obj.rotation_euler = _aim_euler(cam_loc, tgt)
        scene.frame_set(test_f)
        scene.render.filepath = f"{out_dir}/env_f{test_f:04d}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[jet_cockpit_env self-test] Rendered frame {test_f} -> {scene.render.filepath}")
