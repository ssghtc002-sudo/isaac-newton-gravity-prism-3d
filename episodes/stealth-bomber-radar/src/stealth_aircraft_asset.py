"""
Photoreal Aerospace Stealth Bomber & Conventional Jet Assets for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Daylight Hollywood-VFX Edition:
  1. Hero 172-ft B-2 Spirit Flying-Wing Stealth Bomber (`Stealth_Bomber_Root`, 1:15 scale = 3.20m span):
     - High-resolution smooth blended lifting-body dorsal fuselage tapering cleanly into crisp 33-deg
       chined leading edges & authentic 5-chevron W-sawtooth trailing edges
     - Daylight-calibrated satin-matte military graphite-charcoal RAM carbon-composite shader with
       fine aerospace panel seams (Scale=22.0), subtle radar-tape weathering, and clearcoat sun glint
     - Flush 4-pane gold-ITO stealth cockpit windshield (Shadow Ray bypass) + dark canopy mullions
     - Contoured serrated dorsal intake cowls (using the SAME X-ray shader so Beat 3 dissolves them
       cleanly without floating bricks!) + recessed ceramic-titanium dorsal exhaust channels
     - Local-Space `ShaderNodeTexCoord.Object` X-Ray Cutaway Skin: hollow electric-cyan Fresnel
       (`AddShader(Transparent, Emission * CameraRay * (1 - Backfacing) * pow(Facing, 3.8))`)
     - Twin deep **Ribbed Serpentine S-Shaped Air Intake Ducts** + twin detailed **Multi-Stage
       Machined Titanium Turbofans** (gunmetal fan shroud casing, 20 twisted silver blades, pointed
       spinner cone, compressor stator rings — ZERO blocky orange cylinders!)
     - Active across ALL 6 beats (including Beat 5 main-scene daylight tilted-facet explainer!)
  2. Photoreal Lofted Conventional Jet (`Conv_Jet_Root`, active in Beats 2 & 5):
     - True multi-station aerodynamic lofted fuselage (pointed ogive radome nose -> cockpit brow ->
       cylindrical cabin -> tapered APU tailcone) with brushed silver aluminum + navy livery stripe
     - Contoured wrap-around dark cockpit windshield, cabin window band, tapered swept airfoil wings
       with blended winglets, realistic hollow-inlet underwing turbofans with visible metallic fan
       spinners & exhaust nozzles, and a tall 90-degree swept vertical tail fin with animated crimson
       radar-echo Fresnel flare!

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules (iter_action_fcurves, get_mix_sockets,
materials appended before bm.to_mesh, Local-Space TexCoord.Object).
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
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
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


# ==============================================================================
# MATERIALS (DAYLIGHT RAM CHARCOAL + LOCAL-SPACE X-RAY + AEROSPACE TITANIUM)
# ==============================================================================
def _mat_stealth_ram_xray(root_obj: bpy.types.Object) -> tuple[bpy.types.Material, Any, Any]:
    """
    Daylight-calibrated military satin-matte graphite-charcoal RAM carbon-composite skin with fine
    realistic aerospace panel seams (Scale=22.0, zero blocky diamonds!) and smooth Local-Space
    `ShaderNodeTexCoord.Object` hollow electric-cyan Fresnel X-ray cutaway transition in Beat 3.
    """
    mat = bpy.data.materials.new("Mat_Stealth_RAM_XRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = root_obj

    noise_micro = nt.nodes.new("ShaderNodeTexNoise")
    noise_micro.inputs["Scale"].default_value = 95.0
    noise_micro.inputs["Detail"].default_value = 8.0
    noise_micro.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], noise_micro.inputs["Vector"])

    noise_wear = nt.nodes.new("ShaderNodeTexNoise")
    noise_wear.inputs["Scale"].default_value = 5.5
    noise_wear.inputs["Detail"].default_value = 4.0
    noise_wear.inputs["Roughness"].default_value = 0.55
    nt.links.new(tc.outputs["Object"], noise_wear.inputs["Vector"])

    # Fine realistic aerospace stealth panel seams (Scale=22 instead of oversized 6.2!)
    vor_panel = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_panel.feature = "DISTANCE_TO_EDGE"
    vor_panel.inputs["Scale"].default_value = 22.0
    vor_panel.inputs["Randomness"].default_value = 0.04
    nt.links.new(tc.outputs["Object"], vor_panel.inputs["Vector"])

    panel_edge = nt.nodes.new("ShaderNodeMapRange")
    panel_edge.clamp = True
    panel_edge.inputs["From Min"].default_value = 0.008
    panel_edge.inputs["From Max"].default_value = 0.028
    nt.links.new(vor_panel.outputs["Distance"], panel_edge.inputs["Value"])

    bump_grain = nt.nodes.new("ShaderNodeBump")
    bump_grain.inputs["Strength"].default_value = 0.045
    bump_grain.inputs["Distance"].default_value = 0.002
    nt.links.new(noise_micro.outputs["Fac"], bump_grain.inputs["Height"])

    bump_panel = nt.nodes.new("ShaderNodeBump")
    bump_panel.inputs["Strength"].default_value = 0.10
    bump_panel.inputs["Distance"].default_value = 0.003
    nt.links.new(panel_edge.outputs["Result"], bump_panel.inputs["Height"])
    nt.links.new(bump_grain.outputs["Normal"], bump_panel.inputs["Normal"])

    # Authentic B-2 daylight charcoal-graphite slate (catches golden sun & blue sky bounce clearly!)
    mix_wear = nt.nodes.new("ShaderNodeMix")
    mix_wear.data_type = "RGBA"
    fw, wa, wb = C.get_mix_sockets(mix_wear)
    nt.links.new(noise_wear.outputs["Fac"], fw)
    wa.default_value = (0.064, 0.072, 0.086, 1.0)
    wb.default_value = (0.104, 0.116, 0.136, 1.0)

    mix_base = nt.nodes.new("ShaderNodeMix")
    mix_base.data_type = "RGBA"
    fb, ba, bb = C.get_mix_sockets(mix_base)
    nt.links.new(panel_edge.outputs["Result"], fb)
    ba.default_value = (0.036, 0.042, 0.052, 1.0)  # Subtle dark stealth tape seam
    nt.links.new(mix_wear.outputs[2], bb)

    rough_map = nt.nodes.new("ShaderNodeMapRange")
    rough_map.clamp = True
    rough_map.inputs["From Min"].default_value = 0.25
    rough_map.inputs["From Max"].default_value = 0.78
    rough_map.inputs["To Min"].default_value = 0.28
    rough_map.inputs["To Max"].default_value = 0.46
    nt.links.new(noise_wear.outputs["Fac"], rough_map.inputs["Value"])

    bsdf_ram = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(mix_base.outputs[2], bsdf_ram.inputs["Base Color"])
    bsdf_ram.inputs["Metallic"].default_value = 0.36
    nt.links.new(rough_map.outputs["Result"], bsdf_ram.inputs["Roughness"])
    if "Specular IOR Level" in bsdf_ram.inputs:
        bsdf_ram.inputs["Specular IOR Level"].default_value = 0.56
    if "Coat Weight" in bsdf_ram.inputs:
        bsdf_ram.inputs["Coat Weight"].default_value = 0.26
        bsdf_ram.inputs["Coat Roughness"].default_value = 0.22
    nt.links.new(bump_panel.outputs["Normal"], bsdf_ram.inputs["Normal"])

    # Local-space X-ray sweep mask along bomber chord Y (-0.72 .. +0.62)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    xray_mix_val = nt.nodes.new("ShaderNodeValue")
    xray_mix_val.name = "XRay_Skin_Mix"
    xray_mix_val.outputs[0].default_value = 0.0

    scan_y_val = nt.nodes.new("ShaderNodeValue")
    scan_y_val.name = "XRay_Scan_Y"
    scan_y_val.outputs[0].default_value = -0.80

    sub_y = nt.nodes.new("ShaderNodeMath")
    sub_y.operation = "SUBTRACT"
    nt.links.new(scan_y_val.outputs[0], sub_y.inputs[0])
    nt.links.new(sep.outputs["Y"], sub_y.inputs[1])

    sweep_mask = nt.nodes.new("ShaderNodeMapRange")
    sweep_mask.clamp = True
    sweep_mask.inputs["From Min"].default_value = -0.08
    sweep_mask.inputs["From Max"].default_value = 0.14
    nt.links.new(sub_y.outputs[0], sweep_mask.inputs["Value"])

    abs_x = nt.nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    nt.links.new(sep.outputs["X"], abs_x.inputs[0])

    bay_mask = nt.nodes.new("ShaderNodeMapRange")
    bay_mask.clamp = True
    bay_mask.inputs["From Min"].default_value = 0.80
    bay_mask.inputs["From Max"].default_value = 0.54
    nt.links.new(abs_x.outputs[0], bay_mask.inputs["Value"])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(sweep_mask.outputs["Result"], m1.inputs[0])
    nt.links.new(bay_mask.outputs["Result"], m1.inputs[1])

    eff_xray = nt.nodes.new("ShaderNodeMath")
    eff_xray.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], eff_xray.inputs[0])
    nt.links.new(xray_mix_val.outputs[0], eff_xray.inputs[1])

    abs_line = nt.nodes.new("ShaderNodeMath")
    abs_line.operation = "ABSOLUTE"
    nt.links.new(sub_y.outputs[0], abs_line.inputs[0])

    scan_ring = nt.nodes.new("ShaderNodeMapRange")
    scan_ring.clamp = True
    scan_ring.inputs["From Min"].default_value = 0.042
    scan_ring.inputs["From Max"].default_value = 0.0
    nt.links.new(abs_line.outputs[0], scan_ring.inputs["Value"])

    scan_ring_active = nt.nodes.new("ShaderNodeMath")
    scan_ring_active.operation = "MULTIPLY"
    nt.links.new(scan_ring.outputs["Result"], scan_ring_active.inputs[0])
    nt.links.new(xray_mix_val.outputs[0], scan_ring_active.inputs[1])

    # Hollow Fresnel X-Ray Shader (AGENTS.md V12: AddShader(Transparent, Emission*Cam*(1-Back)*pow(Facing,3.8)))
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45

    front_mask = nt.nodes.new("ShaderNodeMath")
    front_mask.operation = "SUBTRACT"
    front_mask.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], front_mask.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(front_mask.outputs[0], cam_front.inputs[1])

    fres_pow = nt.nodes.new("ShaderNodeMath")
    fres_pow.operation = "POWER"
    fres_pow.inputs[1].default_value = 3.8
    nt.links.new(lw.outputs["Facing"], fres_pow.inputs[0])

    fres_str = nt.nodes.new("ShaderNodeMath")
    fres_str.operation = "MULTIPLY"
    fres_str.inputs[1].default_value = 1.65
    nt.links.new(fres_pow.outputs[0], fres_str.inputs[0])

    ring_boost = nt.nodes.new("ShaderNodeMath")
    ring_boost.operation = "MULTIPLY"
    ring_boost.inputs[1].default_value = 1.95
    nt.links.new(scan_ring_active.outputs[0], ring_boost.inputs[0])

    tot_emit = nt.nodes.new("ShaderNodeMath")
    tot_emit.operation = "ADD"
    nt.links.new(fres_str.outputs[0], tot_emit.inputs[0])
    nt.links.new(ring_boost.outputs[0], tot_emit.inputs[1])

    emit_final = nt.nodes.new("ShaderNodeMath")
    emit_final.operation = "MULTIPLY"
    nt.links.new(tot_emit.outputs[0], emit_final.inputs[0])
    nt.links.new(cam_front.outputs[0], emit_final.inputs[1])

    emit_xray = nt.nodes.new("ShaderNodeEmission")
    emit_xray.inputs["Color"].default_value = (0.04, 0.84, 1.00, 1.0)
    nt.links.new(emit_final.outputs[0], emit_xray.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_xray = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_xray.inputs[0])
    nt.links.new(emit_xray.outputs["Emission"], add_xray.inputs[1])

    sh_xray = nt.nodes.new("ShaderNodeMath")
    sh_xray.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Shadow Ray"], sh_xray.inputs[0])
    nt.links.new(eff_xray.outputs[0], sh_xray.inputs[1])

    mix_main = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(eff_xray.outputs[0], mix_main.inputs["Fac"])
    nt.links.new(bsdf_ram.outputs["BSDF"], mix_main.inputs[1])
    nt.links.new(add_xray.outputs["Shader"], mix_main.inputs[2])

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(sh_xray.outputs[0], mix_sh.inputs["Fac"])
    nt.links.new(mix_main.outputs["Shader"], mix_sh.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[2])

    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, xray_mix_val, scan_y_val


def _mat_cockpit_glass() -> bpy.types.Material:
    """Gold-tinted stealth ITO cockpit glass with mandatory Shadow Ray -> Transparent BSDF bypass."""
    mat = bpy.data.materials.new("Mat_Stealth_CockpitGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.32, 0.22, 0.08, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.78
    bsdf.inputs["Roughness"].default_value = 0.05
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.22
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.80
        bsdf.inputs["Coat Roughness"].default_value = 0.02

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_sh.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _mat_sduct_xray_wall() -> bpy.types.Material:
    """Crisp translucent cyan-titanium S-duct cutaway tunnel so internal radar bounces read vividly."""
    mat = bpy.data.materials.new("Mat_Stealth_SDuctWall")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.36

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.08, 0.28, 0.42, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.68
    bsdf.inputs["Roughness"].default_value = 0.26
    bsdf.inputs["Emission Color"].default_value = (0.04, 0.78, 1.00, 1.0)

    fres = nt.nodes.new("ShaderNodeMath")
    fres.operation = "POWER"
    fres.inputs[1].default_value = 2.2
    nt.links.new(lw.outputs["Facing"], fres.inputs[0])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 1.25
    nt.links.new(fres.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    op_add = nt.nodes.new("ShaderNodeMath")
    op_add.operation = "ADD"
    op_add.inputs[1].default_value = 0.30
    nt.links.new(fres.outputs[0], op_add.inputs[0])

    op_clamp = nt.nodes.new("ShaderNodeClamp")
    op_clamp.inputs["Max"].default_value = 0.78
    nt.links.new(op_add.outputs[0], op_clamp.inputs["Value"])

    mix_t = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(op_clamp.outputs[0], mix_t.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix_t.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix_t.inputs[2])

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_sh.inputs["Fac"])
    nt.links.new(mix_t.outputs["Shader"], mix_sh.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _mat_titanium_fan() -> bpy.types.Material:
    """Bright precision-machined aerospace titanium compressor blades."""
    mat = bpy.data.materials.new("Mat_Stealth_TitaniumFan")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.82, 0.86, 0.92, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.16
    return mat


def _mat_engine_casing() -> bpy.types.Material:
    """Dark machined gunmetal-titanium turbofan engine barrel & stator shroud (zero ugly orange block!)."""
    mat = bpy.data.materials.new("Mat_Stealth_EngineCasing")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.16, 0.19, 0.24, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.90
    bsdf.inputs["Roughness"].default_value = 0.28
    return mat


def _mat_exhaust_trough() -> bpy.types.Material:
    """Heat-shielded dark sintered titanium exhaust deck with subtle muted amber core."""
    mat = bpy.data.materials.new("Mat_Stealth_ExhaustTrough")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.12, 0.13, 0.16, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.92
    bsdf.inputs["Roughness"].default_value = 0.24
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.42, 0.08, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.65
    return mat


def _mat_nav_light(name: str, color: tuple[float, float, float], strength: float = 1.90) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
    bsdf.inputs["Emission Strength"].default_value = min(2.0, strength)
    return mat


def _mat_conv_jet_echo(
    name: str,
    base_col: tuple[float, float, float],
    metallic: float = 0.82,
    roughness: float = 0.22,
) -> tuple[bpy.types.Material, Any]:
    """Aerospace metallic/painted jet material with animated blazing crimson radar-echo Fresnel glow."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base_col, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.12, 0.05, 1.0)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.44
    pow_n = nt.nodes.new("ShaderNodeMath")
    pow_n.operation = "POWER"
    pow_n.inputs[1].default_value = 1.8
    nt.links.new(lw.outputs["Facing"], pow_n.inputs[0])

    add_base = nt.nodes.new("ShaderNodeMath")
    add_base.operation = "ADD"
    add_base.inputs[1].default_value = 0.18
    nt.links.new(pow_n.outputs[0], add_base.inputs[0])

    echo_val = nt.nodes.new("ShaderNodeValue")
    echo_val.name = f"{name}_EchoGlow"
    echo_val.outputs[0].default_value = 0.0

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(add_base.outputs[0], mul.inputs[0])
    nt.links.new(echo_val.outputs[0], mul.inputs[1])
    nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, echo_val


# ==============================================================================
# 1. MUSEUM-GRADE SCULPTED B-2 SPIRIT FLYING-WING STEALTH BOMBER
# ==============================================================================
def _b2_planform_y(x_abs: float) -> tuple[float, float]:
    """
    Return `(y_lead, y_trail)` in meters along the 3.20 m wingspan.
    Nose at `(0, -0.68)`, razor-sharp 33-deg leading-edge sweep, iconic 5-chevron W-sawtooth tail!
    """
    half_span = 0.5 * float(C.BOMBER["wingspan_m"])  # 1.60 m
    x_c = min(half_span, max(0.0, x_abs))
    tan33 = math.tan(math.radians(33.0))
    y_lead = -0.68 + x_c * tan33

    knots = [
        (0.00, +0.64),
        (0.32, +0.36),
        (0.62, +0.58),
        (0.96, +0.34),
        (1.48, +0.62),
        (1.60, +0.44),
    ]
    y_trail = knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x_c <= x1 + 1e-6:
            t = (x_c - x0) / max(1e-6, x1 - x0)
            y_trail = C.lerp(y0, y1, t)
            break
    y_trail = max(y_lead + 0.06, y_trail)
    return y_lead, y_trail


def _b2_airfoil_z(x_abs: float, u_chord: float) -> tuple[float, float]:
    """Return `(z_top, z_bot)` sleek blended lifting-body flying-wing airfoil + contoured dorsal intakes."""
    half_span = 0.5 * float(C.BOMBER["wingspan_m"])
    span_t = min(1.0, x_abs / half_span)

    center_hump = math.exp(-((x_abs / 0.38) ** 2)) * 0.135
    wing_thick = C.lerp(0.058, 0.011, span_t ** 0.85)
    total_h = center_hump + wing_thick

    if u_chord <= 0.32:
        prof = (u_chord / 0.32) ** 0.65
    else:
        prof = ((1.0 - u_chord) / 0.68) ** 0.85

    z_top = total_h * prof
    z_bot = -(0.020 + 0.034 * math.exp(-((x_abs / 0.44) ** 2))) * (math.sin(math.pi * u_chord) ** 0.72)

    # Sleek cockpit canopy blister rise
    if x_abs < 0.18 and 0.12 < u_chord < 0.44:
        cx = max(0.0, 1.0 - (x_abs / 0.18) ** 2)
        cu = math.sin(math.pi * (u_chord - 0.12) / 0.32)
        z_top += 0.042 * cx * cu

    # Smoothly blended dorsal twin intake nacelle fairings (integrated right into hull — zero floating bricks!)
    if 0.17 < x_abs < 0.35 and 0.28 < u_chord < 0.74:
        nx = math.sin(math.pi * (x_abs - 0.17) / 0.18)
        nu = math.sin(math.pi * (u_chord - 0.28) / 0.46) ** 0.7
        z_top += 0.046 * nx * nu

    return z_top, z_bot


def _build_stealth_hull_mesh(
    col: bpy.types.Collection,
    root: bpy.types.Object,
    mat_ram: bpy.types.Material,
    mat_glass: bpy.types.Material,
    mat_exhaust: bpy.types.Material,
    mat_nav_red: bpy.types.Material,
    mat_nav_grn: bpy.types.Material,
) -> bpy.types.Object:
    """Build high-resolution sculpted B-2 Spirit flying-wing hull with integrated dorsal nacelles & canopy."""
    bm = bmesh.new()

    pos_xs = [
        0.00, 0.04, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.44,
        0.52, 0.62, 0.74, 0.85, 0.96, 1.12, 1.28, 1.48, 1.56, 1.60,
    ]
    xs = sorted(set([-x for x in pos_xs] + pos_xs))
    us = [
        0.00, 0.04, 0.09, 0.15, 0.21, 0.27, 0.33, 0.40,
        0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.95, 1.00,
    ]
    n_x = len(xs)
    n_u = len(us)

    top_grid: list[list[bmesh.types.BMVert]] = []
    bot_grid: list[list[bmesh.types.BMVert]] = []

    for x in xs:
        yl, yt = _b2_planform_y(abs(x))
        t_row: list[bmesh.types.BMVert] = []
        b_row: list[bmesh.types.BMVert] = []
        for ui, u in enumerate(us):
            y = C.lerp(yl, yt, u)
            zt, zb = _b2_airfoil_z(abs(x), u)
            if ui == 0 or ui == n_u - 1:
                v_edge = bm.verts.new((x, y, 0.0))
                t_row.append(v_edge)
                b_row.append(v_edge)
            else:
                t_row.append(bm.verts.new((x, y, zt)))
                b_row.append(bm.verts.new((x, y, zb)))
        top_grid.append(t_row)
        bot_grid.append(b_row)

    for ix in range(n_x - 1):
        xm = 0.5 * (xs[ix] + xs[ix + 1])
        for iu in range(n_u - 1):
            um = 0.5 * (us[iu] + us[iu + 1])
            vt = (top_grid[ix][iu], top_grid[ix + 1][iu], top_grid[ix + 1][iu + 1], top_grid[ix][iu + 1])
            if len(set(vt)) >= 3:
                try:
                    f_t = bm.faces.new(vt)
                    # Flush 4-pane gold-ITO stealth cockpit windshield (keep thin RAM center mullion!)
                    if 0.016 < abs(xm) < 0.142 and 0.145 < um < 0.295:
                        f_t.material_index = 1
                    # Flush dorsal recessed titanium exhaust trough channels behind engine nacelles
                    elif 0.21 < abs(xm) < 0.31 and 0.72 < um < 0.87:
                        f_t.material_index = 2
                    else:
                        f_t.material_index = 0
                except ValueError:
                    pass
            vb = (bot_grid[ix][iu], bot_grid[ix][iu + 1], bot_grid[ix + 1][iu + 1], bot_grid[ix + 1][iu])
            if len(set(vb)) >= 3:
                try:
                    f_b = bm.faces.new(vb)
                    f_b.material_index = 0
                except ValueError:
                    pass

    # Cap wingtips cleanly
    for ix_tip in (0, n_x - 1):
        for iu in range(1, n_u - 1):
            uniq = list(dict.fromkeys([
                top_grid[ix_tip][iu], top_grid[ix_tip][iu + 1],
                bot_grid[ix_tip][iu + 1], bot_grid[ix_tip][iu],
            ]))
            if len(uniq) >= 3:
                try:
                    bm.faces.new(uniq if ix_tip == n_x - 1 else uniq[::-1]).material_index = 0
                except ValueError:
                    pass

    # Wingtip navigation beacons (Port=-X ruby red, Starboard=+X emerald green)
    for side in (-1.0, +1.0):
        nav_m = Matrix.Translation((side * 1.585, 0.42, 0.008))
        res_nav = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.013, matrix=nav_m)
        nv = set(res_nav["verts"])
        nav_midx = 3 if side < 0 else 4
        for fn in bm.faces:
            if all(v in nv for v in fn.verts):
                fn.material_index = nav_midx

    hull_obj = _mesh_from_bm(
        "Stealth_Bomber_Hull",
        bm,
        [mat_ram, mat_glass, mat_exhaust, mat_nav_red, mat_nav_grn],
        col,
        parent=root,
        smooth=True,  # Sleek organic blended lifting body!
    )
    # Add Edge Split modifier so sharp 33-deg leading/trailing chines stay razor-crisp while dorsal hump stays silky!
    es = hull_obj.modifiers.new("StealthEdgeSplit", type="EDGE_SPLIT")
    es.split_angle = math.radians(34.0)
    return hull_obj


# ==============================================================================
# 2. RIBBED SERPENTINE S-DUCTS & MULTI-STAGE MACHINED TITANIUM TURBOFANS (BEAT 3)
# ==============================================================================
def _build_serpentine_sducts_and_turbines(
    col: bpy.types.Collection,
    root: bpy.types.Object,
    total_frames: int,
) -> dict[str, Any]:
    """
    Build twin deeply buried **ribbed serpentine S-shaped intake ducts** + twin detailed
    **multi-stage machined titanium turbofans** (dark gunmetal casing, 20 twisted titanium blades,
    pointed nose spinner, internal stator rings — ZERO ugly orange rectangles!).
    """
    mat_duct = _mat_sduct_xray_wall()
    mat_fan = _mat_titanium_fan()
    mat_casing = _mat_engine_casing()
    mat_core_glow = _mat_exhaust_trough()

    ducts: list[bpy.types.Object] = []
    fans: list[bpy.types.Object] = []

    for side in (-1.0, +1.0):
        tag = "L" if side < 0 else "R"

        pts: list[Vector] = []
        n_steps = 32
        for i in range(n_steps + 1):
            t = i / n_steps
            y = C.lerp(-0.26, +0.34, t)
            x = side * (0.24 + 0.088 * math.sin(math.pi * t) - 0.048 * math.sin(2.0 * math.pi * t))
            z = 0.068 - 0.088 * C.smootherstep(0.08, 0.72, t) + 0.024 * math.sin(2.0 * math.pi * t)
            pts.append(Vector((x, y, z)))

        bm_d = bmesh.new()
        r_duct = float(C.BOMBER["duct_radius_m"])
        n_ring = 24
        rings_v: list[list[bmesh.types.BMVert]] = []
        for i, p in enumerate(pts):
            t_vec = (pts[min(n_steps, i + 1)] - pts[max(0, i - 1)]).normalized()
            up_ref = Vector((0.0, 0.0, 1.0))
            side_v = t_vec.cross(up_ref).normalized()
            up_v = side_v.cross(t_vec).normalized()
            # Subtle structural circumferential rib swell every 4 steps
            rib = 1.06 if (i % 4 == 0 and 0 < i < n_steps) else 1.0
            rx = r_duct * rib * C.lerp(1.28, 0.98, i / n_steps)
            rz = r_duct * rib * C.lerp(0.72, 0.98, i / n_steps)
            row = []
            for j in range(n_ring):
                a = 2.0 * math.pi * j / n_ring
                pos = p + side_v * (rx * math.cos(a)) + up_v * (rz * math.sin(a))
                row.append(bm_d.verts.new(pos))
            rings_v.append(row)

        for i in range(n_steps):
            for j in range(n_ring):
                jn = (j + 1) % n_ring
                f_d = bm_d.faces.new((rings_v[i][j], rings_v[i][jn], rings_v[i + 1][jn], rings_v[i + 1][j]))
                f_d.material_index = 0

        # Sculpted metallic intake lip collar right at front mouth (pts[0])
        p0 = pts[0]
        t0 = (pts[1] - pts[0]).normalized()
        rot0 = t0.to_track_quat("Z", "Y").to_matrix().to_4x4()
        res_lip = bmesh.ops.create_cone(
            bm_d, cap_ends=False, cap_tris=False, segments=20,
            radius1=r_duct * 1.15, radius2=r_duct * 1.08, depth=0.036,
            matrix=Matrix.Translation(p0) @ rot0 @ Matrix.Diagonal((1.18, 0.72, 1.0, 1.0)),
        )
        lip_vs = set(res_lip["verts"])
        for fl in bm_d.faces:
            if all(v in lip_vs for v in fl.verts):
                fl.material_index = 1

        duct_obj = _mesh_from_bm(
            f"Stealth_SDuct_{tag}", bm_d, [mat_duct, mat_casing], col, parent=root, smooth=True
        )
        duct_obj.visible_shadow = False
        _key_visibility(duct_obj, [(541, 960)], total_frames)
        ducts.append(duct_obj)

        # --- Detailed Multi-Stage Machined Titanium Turbofan Engine ---
        fan_pos = pts[-3]
        r_fan = float(C.BOMBER["fan_radius_m"])

        # Static machined gunmetal turbofan nacelle barrel + compressor stator flanges
        bm_shroud = bmesh.new()
        rot_y90 = Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4()
        # Outer engine casing barrel (hollow open front so spinning silver blades gleam clearly!)
        res_cas = bmesh.ops.create_cone(
            bm_shroud, cap_ends=False, cap_tris=False, segments=24,
            radius1=r_fan * 1.16, radius2=r_fan * 0.96, depth=0.17,
            matrix=Matrix.Translation((0.0, 0.075, 0.0)) @ rot_y90,
        )
        bmesh.ops.solidify(bm_shroud, geom=list(bm_shroud.faces), thickness=0.008)
        for fc in bm_shroud.faces:
            fc.material_index = 0
        # Two external machined structural flange rings around engine casing
        for y_fl in (0.015, 0.095):
            res_fl = bmesh.ops.create_cone(
                bm_shroud, cap_ends=False, cap_tris=False, segments=24,
                radius1=r_fan * 1.23, radius2=r_fan * 1.23, depth=0.012,
                matrix=Matrix.Translation((0.0, y_fl, 0.0)) @ rot_y90,
            )
            fv_s = set(res_fl["verts"])
            for ff in bm_shroud.faces:
                if all(v in fv_s for v in ff.verts):
                    ff.material_index = 1
        # Subtle recessed aft turbine exhaust cone inside back of barrel
        res_ex = bmesh.ops.create_cone(
            bm_shroud, cap_ends=True, cap_tris=False, segments=18,
            radius1=r_fan * 0.62, radius2=r_fan * 0.24, depth=0.06,
            matrix=Matrix.Translation((0.0, 0.155, 0.0)) @ rot_y90,
        )
        ex_vs = set(res_ex["verts"])
        for fe in bm_shroud.faces:
            if all(v in ex_vs for v in fe.verts):
                fe.material_index = 2

        shroud_obj = _mesh_from_bm(
            f"Stealth_EngineCasing_{tag}", bm_shroud, [mat_casing, mat_fan, mat_core_glow],
            col, parent=root, location=(fan_pos.x, fan_pos.y, fan_pos.z), smooth=True,
        )
        _key_visibility(shroud_obj, [(541, 960)], total_frames)
        ducts.append(shroud_obj)

        # Spinning 20-Blade Machined Titanium Fan Rotor + Pointed Nose Spinner
        bm_f = bmesh.new()
        bmesh.ops.create_cone(
            bm_f, cap_ends=True, cap_tris=False, segments=20,
            radius1=r_fan * 0.30, radius2=0.003, depth=0.056,
            matrix=Matrix.Translation((0.0, -0.020, 0.0)) @ Euler((math.radians(-90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
        )
        bmesh.ops.create_cone(
            bm_f, cap_ends=True, cap_tris=False, segments=20,
            radius1=r_fan * 0.32, radius2=r_fan * 0.30, depth=0.045,
            matrix=Matrix.Translation((0.0, 0.015, 0.0)) @ rot_y90,
        )
        n_blades = 20
        for bi in range(n_blades):
            ang = 2.0 * math.pi * bi / n_blades
            bm_rot = Euler((0.0, ang, 0.0), "XYZ").to_matrix().to_4x4()
            twist = Euler((0.0, 0.0, math.radians(34.0)), "XYZ").to_matrix().to_4x4()
            blade_m = (
                bm_rot
                @ Matrix.Translation((0.0, 0.0, r_fan * 0.62))
                @ twist
                @ Matrix.Diagonal((0.008, 0.022, r_fan * 0.72, 1.0))
            )
            bmesh.ops.create_cube(bm_f, size=1.0, matrix=blade_m)
        for fb in bm_f.faces:
            fb.material_index = 0

        fan_obj = _mesh_from_bm(
            f"Stealth_TurbineFan_{tag}", bm_f, [mat_fan],
            col, parent=root, location=(fan_pos.x, fan_pos.y, fan_pos.z), smooth=True,
        )
        _key_visibility(fan_obj, [(541, 960)], total_frames)
        for f in range(541, 961, 10):
            fan_obj.rotation_euler = Euler((0.0, side * (f - 541) * 0.44, 0.0), "XYZ")
            fan_obj.keyframe_insert("rotation_euler", frame=f)
        fans.append(fan_obj)

    return {"ducts": ducts, "fans": fans}


# ==============================================================================
# 3. PHOTOREAL MULTI-STATION LOFTED CONVENTIONAL JET (BEATS 2 & 5!)
# ==============================================================================
def _add_lofted_wing(
    bm: bmesh.types.BMesh,
    root_pts: list[tuple[float, float, float]],
    tip_pts: list[tuple[float, float, float]],
    mat_index: int = 0,
) -> None:
    """Create a smooth swept tapered airfoil wing/fin loft between root and tip 2D/3D cross-section loops."""
    n = len(root_pts)
    v_root = [bm.verts.new(p) for p in root_pts]
    v_tip = [bm.verts.new(p) for p in tip_pts]
    for i in range(n):
        inext = (i + 1) % n
        f = bm.faces.new((v_root[i], v_root[inext], v_tip[inext], v_tip[i]))
        f.material_index = mat_index
    f_cap_r = bm.faces.new(v_root[::-1])
    f_cap_r.material_index = mat_index
    f_cap_t = bm.faces.new(v_tip)
    f_cap_t.material_index = mat_index


def _airfoil_loop(
    cx: float, cy: float, cz: float, chord: float, thick: float, is_vertical: bool = False
) -> list[tuple[float, float, float]]:
    """Return an 8-point streamlined NACA-like airfoil section around `(cx, cy, cz)`."""
    profile = [
        (-0.50, 0.00),
        (-0.34, +0.44),
        (-0.08, +0.50),
        (+0.25, +0.34),
        (+0.50, 0.00),
        (+0.25, -0.26),
        (-0.08, -0.36),
        (-0.34, -0.34),
    ]
    pts: list[tuple[float, float, float]] = []
    for u, v in profile:
        if is_vertical:
            pts.append((cx + v * thick, cy + u * chord, cz))
        else:
            pts.append((cx, cy + u * chord, cz + v * thick))
    return pts


def _build_conventional_jet(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """
    Completely rebuilt photoreal aerodynamic Conventional Jet (active in Beat 2 AND Beat 5!):
    - True smooth multi-station lofted fuselage (sharp pointed ogive nose -> sculpted cockpit brow ->
      cylindrical fuselage -> streamlined APU tailcone — ZERO backwards nose cone!)
    - Brushed aerospace aluminum hull + military/airline deep navy livery stripe + dark cockpit glass
    - Swept tapered airfoil wings with canted blended winglets
    - Realistic hollow-inlet underwing twin turbofan nacelles with machined front fan spinners
    - Tall swept 90-degree vertical tail fin & horizontal stabilizers (glows blazing crimson on radar!)
    """
    mat_silver, echo_s = _mat_conv_jet_echo("Mat_ConvJet_Silver", (0.76, 0.80, 0.85), metallic=0.84, roughness=0.22)
    mat_navy, echo_n = _mat_conv_jet_echo("Mat_ConvJet_Navy", (0.06, 0.14, 0.32), metallic=0.45, roughness=0.28)
    mat_dark = bpy.data.materials.new("Mat_ConvJet_Glass")
    mat_dark.use_nodes = True
    bsdf_d = mat_dark.node_tree.nodes["Principled BSDF"]
    bsdf_d.inputs["Base Color"].default_value = (0.014, 0.022, 0.038, 1.0)
    bsdf_d.inputs["Metallic"].default_value = 0.72
    bsdf_d.inputs["Roughness"].default_value = 0.08
    mat_fan = _mat_titanium_fan()

    root = bpy.data.objects.new("Conv_Jet_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    bm = bmesh.new()

    # 1. Smooth Multi-Station Lofted Fuselage along Y (Nose at -0.82 -> Tail at +0.78)
    stations = [
        # (y, rx, rz, z_center)
        (-0.82, 0.006, 0.006, -0.018),  # Sharp pointed ogive nose tip
        (-0.77, 0.036, 0.034, -0.016),  # Radome nose cone
        (-0.70, 0.064, 0.060, -0.010),  # Forward nose
        (-0.60, 0.084, 0.082, -0.002),  # Cockpit windshield brow
        (-0.46, 0.092, 0.094,  0.000),  # Forward fuselage full cross-section
        (-0.10, 0.092, 0.094,  0.000),  # Mid cabin
        (+0.34, 0.092, 0.094,  0.000),  # Aft cabin
        (+0.54, 0.074, 0.078,  0.008),  # Aft fuselage taper
        (+0.68, 0.044, 0.048,  0.018),  # Tailcone
        (+0.78, 0.014, 0.016,  0.024),  # APU exhaust tip
    ]
    n_seg = 28
    rings: list[list[bmesh.types.BMVert]] = []
    for y, rx, rz, zc in stations:
        row = []
        for s in range(n_seg):
            a = 2.0 * math.pi * s / n_seg
            row.append(bm.verts.new((rx * math.cos(a), y, zc + rz * math.sin(a))))
        rings.append(row)

    for si in range(len(stations) - 1):
        ym = 0.5 * (stations[si][0] + stations[si + 1][0])
        for s in range(n_seg):
            sn = (s + 1) % n_seg
            f = bm.faces.new((rings[si][s], rings[si][sn], rings[si + 1][sn], rings[si + 1][s]))
            ang_m = (s + 0.5) * 2.0 * math.pi / n_seg
            sin_a = math.sin(ang_m)
            # Wrap-around dark cockpit windshield glass on upper front nose brow!
            if -0.71 < ym < -0.58 and sin_a > 0.28 and abs(math.cos(ang_m)) < 0.88:
                f.material_index = 2
            # Sleek navy-blue livery cheatline along mid-lower fuselage flank
            elif -0.56 < ym < +0.62 and -0.22 < sin_a < 0.06:
                f.material_index = 1
            # Subtle dark cabin window stripe above navy line
            elif -0.44 < ym < +0.36 and 0.14 < sin_a < 0.28:
                f.material_index = 2
            else:
                f.material_index = 0

    # Cap nose tip & tailcone cleanly
    bm.faces.new(rings[0][::-1]).material_index = 0
    bm.faces.new(rings[-1]).material_index = 0

    # 2. Swept Tapered Airfoil Main Wings + Canted Blended Winglets + Horizontal Stabilizers
    for side in (-1.0, +1.0):
        # Main swept wing (root -> mid-span -> tip)
        root_sec = _airfoil_loop(side * 0.075, 0.02, -0.026, chord=0.34, thick=0.044)
        tip_sec = _airfoil_loop(side * 0.72, 0.26, +0.024, chord=0.11, thick=0.016)
        _add_lofted_wing(bm, root_sec if side > 0 else root_sec[::-1], tip_sec if side > 0 else tip_sec[::-1], mat_index=0)

        # Canted blended winglet rising cleanly off wingtip
        wl_base = _airfoil_loop(side * 0.72, 0.26, +0.024, chord=0.105, thick=0.014, is_vertical=True)
        wl_top = _airfoil_loop(side * 0.755, 0.31, +0.135, chord=0.058, thick=0.009, is_vertical=True)
        _add_lofted_wing(bm, wl_base, wl_top, mat_index=1)

        # Swept horizontal tail stabilizer
        h_root = _airfoil_loop(side * 0.045, 0.60, 0.028, chord=0.19, thick=0.024)
        h_tip = _airfoil_loop(side * 0.31, 0.72, 0.044, chord=0.08, thick=0.011)
        _add_lofted_wing(bm, h_root if side > 0 else h_root[::-1], h_tip if side > 0 else h_tip[::-1], mat_index=0)

        # Contoured underwing engine pylon strut
        pyl_m = Matrix.Translation((side * 0.27, 0.00, -0.056)) @ Matrix.Diagonal((0.024, 0.22, 0.072, 1.0))
        res_p = bmesh.ops.create_cube(bm, size=1.0, matrix=pyl_m)
        pv = set(res_p["verts"])
        for fp in bm.faces:
            if all(v in pv for v in fp.verts):
                fp.material_index = 0

        # Realistic Hollow-Inlet Underwing Turbofan Nacelle + Visible Front Fan Spinner + Core Exhaust!
        nx, ny, nz = side * 0.27, -0.04, -0.108
        rot_90 = Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4()
        res_nac = bmesh.ops.create_cone(
            bm, cap_ends=False, cap_tris=False, segments=24,
            radius1=0.062, radius2=0.054, depth=0.27,
            matrix=Matrix.Translation((nx, ny, nz)) @ rot_90,
        )
        nv = set(res_nac["verts"])
        n_faces = [f for f in bm.faces if all(v in nv for v in f.verts)]
        for fn in n_faces:
            fn.material_index = 0
        bmesh.ops.solidify(bm, geom=n_faces, thickness=0.008)

        # Dark shadowed intake interior + gleaming silver metallic front fan spinner cone!
        res_int = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=20,
            radius1=0.049, radius2=0.049, depth=0.016,
            matrix=Matrix.Translation((nx, ny - 0.08, nz)) @ rot_90,
        )
        iv = set(res_int["verts"])
        for fi in bm.faces:
            if all(v in iv for v in fi.verts):
                fi.material_index = 2
        res_sp = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=16,
            radius1=0.022, radius2=0.003, depth=0.052,
            matrix=Matrix.Translation((nx, ny - 0.105, nz))
            @ Euler((math.radians(-90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
        )
        spv = set(res_sp["verts"])
        for fsp in bm.faces:
            if all(v in spv for v in fsp.verts):
                fsp.material_index = 3
        # Metallic aft core exhaust nozzle
        res_nozz = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=18,
            radius1=0.036, radius2=0.022, depth=0.09,
            matrix=Matrix.Translation((nx, ny + 0.145, nz)) @ rot_90,
        )
        nzv = set(res_nozz["verts"])
        for fnz in bm.faces:
            if all(v in nzv for v in fnz.verts):
                fnz.material_index = 3

    # 3. Tall Swept 90-Degree Vertical Tail Stabilizer Fin (Major Radar Corner Reflector!)
    v_root = _airfoil_loop(0.0, 0.56, 0.065, chord=0.31, thick=0.032, is_vertical=True)
    v_top = _airfoil_loop(0.0, 0.72, 0.425, chord=0.14, thick=0.016, is_vertical=True)
    _add_lofted_wing(bm, v_root, v_top, mat_index=1)

    jet_obj = _mesh_from_bm(
        "Conv_Jet_Mesh", bm, [mat_silver, mat_navy, mat_dark, mat_fan], col, parent=root, smooth=True
    )
    es = jet_obj.modifiers.new("JetEdgeSplit", type="EDGE_SPLIT")
    es.split_angle = math.radians(38.0)

    # Visible in BOTH Beat 2 (aerial contrast) AND Beat 5 (main-scene 90-deg tail explainer!)
    _key_visibility(jet_obj, [(221, 540), (1321, 1560)], total_frames)

    cut_prev = {220, 540, 1320}
    frames = sorted(
        set(range(221, 541, 4)) | set(range(1321, 1561, 4)) | {221, 540, 1321, 1560}
    )
    for f in frames:
        pos, rot_deg = C.conv_jet_pose_at(float(f))
        root.location = Vector(pos)
        root.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)

        if f <= 540:
            flare = C.smoothstep(285.0, 322.0, float(f)) * (0.72 + 0.28 * math.sin((f - 285.0) * 0.22))
        else:
            flare = C.smoothstep(1340.0, 1372.0, float(f)) * (0.75 + 0.25 * math.sin((f - 1340.0) * 0.22))
        for ev in (echo_s, echo_n):
            ev.outputs[0].default_value = 1.55 * flare
            ev.outputs[0].keyframe_insert("default_value", frame=f)

    ad = getattr(root, "animation_data", None)
    if ad and ad.action:
        for fc in C.iter_action_fcurves(ad.action):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) in cut_prev:
                    kp.interpolation = "CONSTANT"

    return {"root": root, "mesh": jet_obj, "echo_val": echo_s}


def build_stealth_aircraft(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Assemble Hero 172-ft Stealth Bomber (all 6 beats!) + Photoreal Conventional Jet (Beats 2 & 5)."""
    root = bpy.data.objects.new("Stealth_Bomber_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    mat_ram, xray_mix_val, scan_y_val = _mat_stealth_ram_xray(root)
    mat_glass = _mat_cockpit_glass()
    mat_exhaust = _mat_exhaust_trough()
    mat_nav_red = _mat_nav_light("Mat_Stealth_NavRed", (1.00, 0.12, 0.06), 1.90)
    mat_nav_grn = _mat_nav_light("Mat_Stealth_NavGreen", (0.10, 1.00, 0.34), 1.90)

    hull_obj = _build_stealth_hull_mesh(
        col, root, mat_ram, mat_glass, mat_exhaust, mat_nav_red, mat_nav_grn
    )
    internals = _build_serpentine_sducts_and_turbines(col, root, total_frames)
    conv_jet = _build_conventional_jet(col, total_frames)

    # Bomber stays visible across ALL 6 beats (including Beat 5 main-scene daylight explainer!)
    _key_visibility(hull_obj, [(1, total_frames)], total_frames)

    cut_prev = {c - 1 for c in C.HARD_CUT_FRAMES}
    key_frames = sorted(set(range(1, total_frames + 1, 4)) | set(C.HARD_CUT_FRAMES) | cut_prev | {total_frames})
    for f in key_frames:
        pos, rot_deg = C.bomber_pose_at(float(f))
        root.location = Vector(pos)
        root.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)

        xw = C.xray_skin_weight(float(f))
        sp = C.xray_scan_progress(float(f))
        xray_mix_val.outputs[0].default_value = xw
        xray_mix_val.outputs[0].keyframe_insert("default_value", frame=f)
        scan_y_val.outputs[0].default_value = C.lerp(-0.72, +0.62, sp)
        scan_y_val.outputs[0].keyframe_insert("default_value", frame=f)

    ad = getattr(root, "animation_data", None)
    if ad and ad.action:
        for fc in C.iter_action_fcurves(ad.action):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) in cut_prev:
                    kp.interpolation = "CONSTANT"

    return {
        "bomber_root": root,
        "bomber_hull": hull_obj,
        "sducts": internals["ducts"],
        "turbines": internals["fans"],
        "conv_jet": conv_jet,
    }
