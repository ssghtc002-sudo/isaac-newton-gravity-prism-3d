"""
High-Fidelity Stealth Bomber & Conventional Jet Assets for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Builds:
  1. Hero 172-ft Flying-Wing Stealth Bomber (`Stealth_Bomber_Root`, 1:15 scale = 3.20m span):
     - Sharp 33-deg faceted diamond leading edges & authentic 5-chevron W-sawtooth trailing edge
     - Blended dorsal lifting body, flush 4-pane tinted stealth cockpit glass (with Shadow Ray bypass!)
     - Twin serrated dorsal intake nacelles & recessed V-trough titanium exhaust shields
     - Local-Space `ShaderNodeTexCoord.Object` X-Ray Cutaway Skin: hollow electric-cyan Fresnel
       (`AddShader(Transparent, Emission * CameraRay * (1 - Backfacing) * pow(Facing, 3.8))`,
       ZERO opaque center fill!) sweeping cleanly across Beat 3 (`555..955`)
     - Twin deep internal **Serpentine S-Shaped Air Intake Ducts** completely blocking direct
       line-of-sight from dorsal intake lips to the deeply buried engine fans
     - Twin buried **Spinning Titanium Turbofan Engines** (18 twisted blades + machined spinner cone)
  2. Beat 2 Contrast Conventional Jet (`Conv_Jet_Root`):
     - Cylindrical airliner/fighter fuselage, swept wings, round exposed turbofan nacelles,
       and tall 90-degree vertical tail fin with animated crimson radar-echo Fresnel flare!

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
# MATERIALS (LOCAL-SPACE X-RAY HOLLOW FRESNEL, MATTE RAM CARBON, TITANIUM, GLASS)
# ==============================================================================
def _mat_stealth_ram_xray(root_obj: bpy.types.Object) -> tuple[bpy.types.Material, Any, Any]:
    """
    Matte carbon-composite RAM skin that smoothly transitions in Beat 3 via Local-Space
    `ShaderNodeTexCoord.Object` (`Stealth_Bomber_Root`) into a **hollow electric-cyan Fresnel
    X-ray shell** (AddShader(Transparent, Emission * CameraRay * (1 - Backfacing) * pow(Facing, 3.8))).
    ZERO opaque center fill so internal S-ducts & spinning titanium blades shine vividly!
    """
    mat = bpy.data.materials.new("Mat_Stealth_RAM_XRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    # 1. Realistic matte charcoal-obsidian carbon-composite RAM shader with subtle panel bump
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = root_obj

    noise_micro = nt.nodes.new("ShaderNodeTexNoise")
    noise_micro.inputs["Scale"].default_value = 38.0
    noise_micro.inputs["Detail"].default_value = 8.0
    noise_micro.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], noise_micro.inputs["Vector"])

    vor_panel = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_panel.feature = "DISTANCE_TO_EDGE"
    vor_panel.inputs["Scale"].default_value = 5.5
    vor_panel.inputs["Randomness"].default_value = 0.08
    nt.links.new(tc.outputs["Object"], vor_panel.inputs["Vector"])

    panel_edge = nt.nodes.new("ShaderNodeMapRange")
    panel_edge.clamp = True
    panel_edge.inputs["From Min"].default_value = 0.008
    panel_edge.inputs["From Max"].default_value = 0.032
    nt.links.new(vor_panel.outputs["Distance"], panel_edge.inputs["Value"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.004
    nt.links.new(noise_micro.outputs["Fac"], bump.inputs["Height"])

    mix_base = nt.nodes.new("ShaderNodeMix")
    mix_base.data_type = "RGBA"
    fb, ba, bb = C.get_mix_sockets(mix_base)
    nt.links.new(panel_edge.outputs["Result"], fb)
    ba.default_value = (0.016, 0.020, 0.028, 1.0)   # Recessed dark radar tape seam
    bb.default_value = (0.036, 0.044, 0.058, 1.0)   # Satin charcoal graphite RAM skin

    bsdf_ram = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(mix_base.outputs[2], bsdf_ram.inputs["Base Color"])
    bsdf_ram.inputs["Metallic"].default_value = 0.28
    bsdf_ram.inputs["Roughness"].default_value = 0.34
    if "Specular IOR Level" in bsdf_ram.inputs:
        bsdf_ram.inputs["Specular IOR Level"].default_value = 0.52
    if "Coat Weight" in bsdf_ram.inputs:
        bsdf_ram.inputs["Coat Weight"].default_value = 0.18
        bsdf_ram.inputs["Coat Roughness"].default_value = 0.26
    nt.links.new(bump.outputs["Normal"], bsdf_ram.inputs["Normal"])

    # 2. Local-space X-ray mask sweeping aft along bomber local Y (-0.65 .. +0.60)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    xray_mix_val = nt.nodes.new("ShaderNodeValue")
    xray_mix_val.name = "XRay_Skin_Mix"
    xray_mix_val.outputs[0].default_value = 0.0

    scan_y_val = nt.nodes.new("ShaderNodeValue")
    scan_y_val.name = "XRay_Scan_Y"
    scan_y_val.outputs[0].default_value = -0.80

    # Smooth reveal where local Y < scan_y (sweeping from nose -0.65 toward tail +0.65)
    sub_y = nt.nodes.new("ShaderNodeMath")
    sub_y.operation = "SUBTRACT"
    nt.links.new(scan_y_val.outputs[0], sub_y.inputs[0])
    nt.links.new(sep.outputs["Y"], sub_y.inputs[1])

    sweep_mask = nt.nodes.new("ShaderNodeMapRange")
    sweep_mask.clamp = True
    sweep_mask.inputs["From Min"].default_value = -0.08
    sweep_mask.inputs["From Max"].default_value = 0.14
    nt.links.new(sub_y.outputs[0], sweep_mask.inputs["Value"])

    # Focus X-ray cutaway cleanly over center body & S-duct bays (|X| < 0.72)
    abs_x = nt.nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    nt.links.new(sep.outputs["X"], abs_x.inputs[0])

    bay_mask = nt.nodes.new("ShaderNodeMapRange")
    bay_mask.clamp = True
    bay_mask.inputs["From Min"].default_value = 0.78
    bay_mask.inputs["From Max"].default_value = 0.56
    nt.links.new(abs_x.outputs[0], bay_mask.inputs["Value"])

    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(sweep_mask.outputs["Result"], m1.inputs[0])
    nt.links.new(bay_mask.outputs["Result"], m1.inputs[1])

    eff_xray = nt.nodes.new("ShaderNodeMath")
    eff_xray.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], eff_xray.inputs[0])
    nt.links.new(xray_mix_val.outputs[0], eff_xray.inputs[1])

    # Leading glowing scanline rim right at local Y == scan_y
    abs_line = nt.nodes.new("ShaderNodeMath")
    abs_line.operation = "ABSOLUTE"
    nt.links.new(sub_y.outputs[0], abs_line.inputs[0])

    scan_ring = nt.nodes.new("ShaderNodeMapRange")
    scan_ring.clamp = True
    scan_ring.inputs["From Min"].default_value = 0.045
    scan_ring.inputs["From Max"].default_value = 0.0
    nt.links.new(abs_line.outputs[0], scan_ring.inputs["Value"])

    scan_ring_active = nt.nodes.new("ShaderNodeMath")
    scan_ring_active.operation = "MULTIPLY"
    nt.links.new(scan_ring.outputs["Result"], scan_ring_active.inputs[0])
    nt.links.new(xray_mix_val.outputs[0], scan_ring_active.inputs[1])

    # 3. Hollow Fresnel X-Ray Shader (AGENTS.md V12: AddShader(Transparent, Emission*CameraRay*(1-Back)*pow(Facing,3.8)))
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
    ring_boost.inputs[1].default_value = 2.1
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
    emit_xray.inputs["Color"].default_value = (0.06, 0.86, 1.00, 1.0)
    nt.links.new(emit_final.outputs[0], emit_xray.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_xray = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_xray.inputs[0])
    nt.links.new(emit_xray.outputs["Emission"], add_xray.inputs[1])

    # Shadow Ray bypass whenever X-ray is active so shell never casts black shadows on S-ducts!
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
    bsdf.inputs["Base Color"].default_value = (0.14, 0.11, 0.05, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.65
    bsdf.inputs["Roughness"].default_value = 0.08
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.35
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.60
        bsdf.inputs["Coat Roughness"].default_value = 0.04

    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_sh.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat


def _mat_sduct_xray_wall() -> bpy.types.Material:
    """Glowing semi-translucent amber-cyan S-duct cutaway tube so radar bounces inside read vividly."""
    mat = bpy.data.materials.new("Mat_Stealth_SDuctWall")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.04, 0.18, 0.28, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.55
    bsdf.inputs["Roughness"].default_value = 0.32
    bsdf.inputs["Emission Color"].default_value = (0.08, 0.72, 0.96, 1.0)

    fres = nt.nodes.new("ShaderNodeMath")
    fres.operation = "POWER"
    fres.inputs[1].default_value = 2.4
    nt.links.new(lw.outputs["Facing"], fres.inputs[0])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 1.15
    nt.links.new(fres.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])

    # Slightly translucent center so trapped cyan RF ray shines straight through the tube wall
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    op_add = nt.nodes.new("ShaderNodeMath")
    op_add.operation = "ADD"
    op_add.inputs[1].default_value = 0.34
    nt.links.new(fres.outputs[0], op_add.inputs[0])

    op_clamp = nt.nodes.new("ShaderNodeClamp")
    op_clamp.inputs["Max"].default_value = 0.82
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
    """Bright machined titanium turbine blades with subtle amber thermal highlight."""
    mat = bpy.data.materials.new("Mat_Stealth_TitaniumFan")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.76, 0.80, 0.86, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.18
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.62, 0.18, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.22
    return mat


def _mat_exhaust_glow() -> bpy.types.Material:
    mat = bpy.data.materials.new("Mat_Stealth_ExhaustCore")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.12, 0.14, 0.18, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.88
    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.44, 0.08, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.45
    return mat


def _mat_conv_jet_echo() -> tuple[bpy.types.Material, Any]:
    """Metallic airliner/fighter fuselage with animated blazing crimson radar-echo Fresnel glow."""
    mat = bpy.data.materials.new("Mat_ConvJet_Echo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.58, 0.63, 0.70, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.82
    bsdf.inputs["Roughness"].default_value = 0.26
    bsdf.inputs["Emission Color"].default_value = (1.00, 0.14, 0.06, 1.0)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42
    pow_n = nt.nodes.new("ShaderNodeMath")
    pow_n.operation = "POWER"
    pow_n.inputs[1].default_value = 1.8
    nt.links.new(lw.outputs["Facing"], pow_n.inputs[0])

    add_base = nt.nodes.new("ShaderNodeMath")
    add_base.operation = "ADD"
    add_base.inputs[1].default_value = 0.22
    nt.links.new(pow_n.outputs[0], add_base.inputs[0])

    echo_val = nt.nodes.new("ShaderNodeValue")
    echo_val.name = "ConvJet_Echo_Glow"
    echo_val.outputs[0].default_value = 0.0

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(add_base.outputs[0], mul.inputs[0])
    nt.links.new(echo_val.outputs[0], mul.inputs[1])
    nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, echo_val


# ==============================================================================
# PROCEDURAL B-2 FLYING-WING STEALTH BOMBER GEOMETRY
# ==============================================================================
def _b2_planform_y(x_abs: float) -> tuple[float, float]:
    """
    Return `(y_lead, y_trail)` in meters for a given `|x|` along the 3.20 m wingspan.
    Nose is at `(0, -0.68)`, leading edges sweep back cleanly at 33 deg,
    trailing edge forms the iconic B-2 5-chevron W-sawtooth!
    """
    half_span = 0.5 * float(C.BOMBER["wingspan_m"])  # 1.60 m
    x_c = min(half_span, max(0.0, x_abs))
    tan33 = math.tan(math.radians(33.0))  # 0.6494

    y_lead = -0.68 + x_c * tan33

    # Authentic B-2 sawtooth trailing edge knots (piecewise linear chevrons)
    knots = [
        (0.00, +0.64),   # Center beaver-tail apex
        (0.32, +0.36),   # Inner inward V-notch
        (0.62, +0.58),   # Mid sawtooth peak
        (0.96, +0.34),   # Outer inward notch
        (1.48, +0.62),   # Outer wing trailing corner
        (1.60, +0.44),   # Clipped stealth wingtip
    ]
    y_trail = knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x_c <= x1 + 1e-6:
            t = (x_c - x0) / max(1e-6, x1 - x0)
            y_trail = C.lerp(y0, y1, t)
            break

    # Guarantee valid chord >= 0.06 m at extreme tip
    y_trail = max(y_lead + 0.06, y_trail)
    return y_lead, y_trail


def _b2_airfoil_z(x_abs: float, u_chord: float) -> tuple[float, float]:
    """Return `(z_top, z_bot)` blended flying-wing lifting-body profile with faceted stealth chines."""
    half_span = 0.5 * float(C.BOMBER["wingspan_m"])
    span_t = min(1.0, x_abs / half_span)

    # Central fuselage lifting hump fades smoothly into sleek thin flying wing
    center_hump = math.exp(-((x_abs / 0.36) ** 2)) * 0.125
    wing_thick = C.lerp(0.055, 0.010, span_t ** 0.85)
    total_h = center_hump + wing_thick

    # Sharp faceted stealth airfoil cross-section along u_chord in [0, 1]
    # Crisp diamond-ish chine peaking at u=0.34
    if u_chord <= 0.34:
        prof = (u_chord / 0.34) ** 0.68
    else:
        prof = ((1.0 - u_chord) / 0.66) ** 0.82

    z_top = total_h * prof
    # Flatter stealth belly with subtle chine keel
    z_bot = -(0.022 + 0.032 * math.exp(-((x_abs / 0.42) ** 2))) * (math.sin(math.pi * u_chord) ** 0.72)

    # Dorsal cockpit bulge around |X| < 0.16, u in [0.16, 0.42]
    if x_abs < 0.18 and 0.14 < u_chord < 0.44:
        cx = max(0.0, 1.0 - (x_abs / 0.18) ** 2)
        cu = math.sin(math.pi * (u_chord - 0.14) / 0.30)
        z_top += 0.036 * cx * cu

    return z_top, z_bot


def _build_stealth_hull_mesh(
    col: bpy.types.Collection,
    root: bpy.types.Object,
    mat_ram: bpy.types.Material,
    mat_glass: bpy.types.Material,
    mat_exhaust: bpy.types.Material,
) -> bpy.types.Object:
    """Build crisp faceted B-2 Spirit flying-wing hull with sawtooth edges, cockpit glass & intake scoops."""
    bm = bmesh.new()

    # Exact symmetric spanwise stations including every sawtooth knot so edges stay razor-sharp!
    pos_xs = [
        0.00, 0.06, 0.12, 0.18, 0.24, 0.32, 0.42, 0.52, 0.62,
        0.74, 0.85, 0.96, 1.12, 1.28, 1.48, 1.56, 1.60,
    ]
    xs = sorted(set([-x for x in pos_xs] + pos_xs))
    us = [0.00, 0.06, 0.14, 0.22, 0.30, 0.38, 0.48, 0.60, 0.72, 0.85, 0.94, 1.00]

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
                # Weld sharp razor leading and trailing stealth edges!
                v_edge = bm.verts.new((x, y, 0.0))
                t_row.append(v_edge)
                b_row.append(v_edge)
            else:
                t_row.append(bm.verts.new((x, y, zt)))
                b_row.append(bm.verts.new((x, y, zb)))
        top_grid.append(t_row)
        bot_grid.append(b_row)

    # Create top & bottom faceted skins
    for ix in range(n_x - 1):
        xm = 0.5 * (xs[ix] + xs[ix + 1])
        for iu in range(n_u - 1):
            um = 0.5 * (us[iu] + us[iu + 1])
            # Top face
            vt = (top_grid[ix][iu], top_grid[ix + 1][iu], top_grid[ix + 1][iu + 1], top_grid[ix][iu + 1])
            if len(set(vt)) >= 3:
                try:
                    f_t = bm.faces.new(vt)
                    # Assign flush 4-pane cockpit windshield panes
                    if abs(xm) < 0.14 and 0.15 < um < 0.29:
                        f_t.material_index = 1
                    # Recessed exhaust deck troughs
                    elif 0.20 < abs(xm) < 0.36 and 0.66 < um < 0.86:
                        f_t.material_index = 2
                    else:
                        f_t.material_index = 0
                except ValueError:
                    pass
            # Bottom face (reversed winding)
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
            v_a = top_grid[ix_tip][iu]
            v_b = top_grid[ix_tip][iu + 1]
            v_c = bot_grid[ix_tip][iu + 1]
            v_d = bot_grid[ix_tip][iu]
            verts_c = [v for v in (v_a, v_b, v_c, v_d)]
            uniq = list(dict.fromkeys(verts_c))
            if len(uniq) >= 3:
                try:
                    bm.faces.new(uniq if ix_tip == n_x - 1 else uniq[::-1]).material_index = 0
                except ValueError:
                    pass

    # Add sculpted serrated dorsal intake scoops on both sides of cockpit
    for side in (-1.0, +1.0):
        scoop_mat = Matrix.Translation((side * 0.25, -0.14, 0.088)) @ Euler((0.0, side * -0.12, side * 0.10), "XYZ").to_matrix().to_4x4()
        res_s = bmesh.ops.create_cube(bm, size=1.0, matrix=scoop_mat @ Matrix.Diagonal((0.15, 0.30, 0.068, 1.0)))
        sv = set(res_s["verts"])
        for f_s in bm.faces:
            if all(v in sv for v in f_s.verts):
                f_s.material_index = 0

    hull_obj = _mesh_from_bm(
        "Stealth_Bomber_Hull", bm, [mat_ram, mat_glass, mat_exhaust],
        col, parent=root, smooth=False,  # Crisp faceted stealth creases!
    )
    return hull_obj


def _build_serpentine_sducts_and_turbines(
    col: bpy.types.Collection,
    root: bpy.types.Object,
    total_frames: int,
) -> dict[str, Any]:
    """
    Build twin deeply buried serpentine S-shaped intake ducts + twin spinning 18-blade titanium
    turbofan assemblies nestled inside the B-2 flying-wing center body.
    """
    mat_duct = _mat_sduct_xray_wall()
    mat_fan = _mat_titanium_fan()
    mat_core = _mat_exhaust_glow()

    ducts: list[bpy.types.Object] = []
    fans: list[bpy.types.Object] = []

    for side_idx, side in enumerate((-1.0, +1.0)):
        tag = "L" if side < 0 else "R"

        # 1. Swept 3D Serpentine S-Curve Tube (intake lip -> drops down & outward -> curves back to buried fan)
        pts: list[Vector] = []
        n_steps = 28
        for i in range(n_steps + 1):
            t = i / n_steps
            y = C.lerp(-0.24, +0.34, t)
            # Pronounced horizontal + vertical S-bend completely blocking straight radar line-of-sight!
            x = side * (0.24 + 0.085 * math.sin(math.pi * t) - 0.045 * math.sin(2.0 * math.pi * t))
            z = 0.068 - 0.088 * C.smootherstep(0.08, 0.72, t) + 0.024 * math.sin(2.0 * math.pi * t)
            pts.append(Vector((x, y, z)))

        bm_d = bmesh.new()
        r_duct = float(C.BOMBER["duct_radius_m"])
        n_ring = 20
        rings_v: list[list[bmesh.types.BMVert]] = []
        for i, p in enumerate(pts):
            t_vec = (pts[min(n_steps, i + 1)] - pts[max(0, i - 1)]).normalized()
            up_ref = Vector((0.0, 0.0, 1.0))
            side_v = t_vec.cross(up_ref).normalized()
            up_v = side_v.cross(t_vec).normalized()
            # Slightly elliptical stealth lip transitioning to circular turbine bore
            rx = r_duct * C.lerp(1.25, 0.96, i / n_steps)
            rz = r_duct * C.lerp(0.74, 0.96, i / n_steps)
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

        duct_obj = _mesh_from_bm(f"Stealth_SDuct_{tag}", bm_d, [mat_duct], col, parent=root, smooth=True)
        duct_obj.visible_shadow = False
        _key_visibility(duct_obj, [(541, 960)], total_frames)
        ducts.append(duct_obj)

        # 2. Buried Spinning 18-Blade Titanium Turbofan Assembly at the aft end of the S-duct
        fan_pos = pts[-3]
        bm_f = bmesh.new()
        r_fan = float(C.BOMBER["fan_radius_m"])
        # Central bullet nose spinner cone (aligned along local -Y forward into duct)
        bmesh.ops.create_cone(
            bm_f, cap_ends=True, cap_tris=False, segments=16,
            radius1=r_fan * 0.30, radius2=0.004, depth=0.048,
            matrix=Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4() @ Matrix.Translation((0.0, 0.0, 0.018)),
        )
        # Rear hot engine turbine core cylinder
        res_core = bmesh.ops.create_cone(
            bm_f, cap_ends=True, cap_tris=False, segments=18,
            radius1=r_fan * 0.88, radius2=r_fan * 0.72, depth=0.14,
            matrix=Matrix.Translation((0.0, 0.08, 0.0)) @ Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
        )
        core_vs = set(res_core["verts"])
        for fc in bm_f.faces:
            fc.material_index = 1 if all(v in core_vs for v in fc.verts) else 0

        # 18 twisted titanium compressor blades
        n_blades = 18
        for bi in range(n_blades):
            ang = 2.0 * math.pi * bi / n_blades
            bm_rot = Euler((0.0, ang, 0.0), "XYZ").to_matrix().to_4x4()
            twist = Euler((0.0, 0.0, math.radians(32.0)), "XYZ").to_matrix().to_4x4()
            blade_m = bm_rot @ Matrix.Translation((0.0, 0.0, r_fan * 0.60)) @ twist @ Matrix.Diagonal((0.012, 0.018, r_fan * 0.72, 1.0))
            res_b = bmesh.ops.create_cube(bm_f, size=1.0, matrix=blade_m)
            bv = set(res_b["verts"])
            for fb in bm_f.faces:
                if all(v in bv for v in fb.verts):
                    fb.material_index = 0

        fan_obj = _mesh_from_bm(
            f"Stealth_TurbineFan_{tag}", bm_f, [mat_fan, mat_core],
            col, parent=root, location=(fan_pos.x, fan_pos.y, fan_pos.z), smooth=True,
        )
        _key_visibility(fan_obj, [(541, 960)], total_frames)
        # High-RPM smooth rotation across Beat 3!
        for f in range(541, 961, 10):
            fan_obj.rotation_euler = Euler((0.0, side * (f - 541) * 0.42, 0.0), "XYZ")
            fan_obj.keyframe_insert("rotation_euler", frame=f)
        fans.append(fan_obj)

    return {"ducts": ducts, "fans": fans}


def _build_conventional_jet(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """Beat 2 Conventional Jet with cylindrical fuselage, round nacelles & tall 90-deg tail fin."""
    mat_echo, echo_val = _mat_conv_jet_echo()
    mat_dark = bpy.data.materials.new("Mat_ConvJet_Intake")
    mat_dark.use_nodes = True
    bsdf_d = mat_dark.node_tree.nodes["Principled BSDF"]
    bsdf_d.inputs["Base Color"].default_value = (0.04, 0.05, 0.07, 1.0)
    bsdf_d.inputs["Metallic"].default_value = 0.85
    bsdf_d.inputs["Roughness"].default_value = 0.25

    root = bpy.data.objects.new("Conv_Jet_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    bm = bmesh.new()
    # Cylindrical fuselage aligned along Y (nose at -0.74, tail at +0.74)
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.095, radius2=0.095, depth=1.08,
        matrix=Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
    )
    # Tapered nose cone
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=24,
        radius1=0.095, radius2=0.012, depth=0.34,
        matrix=Matrix.Translation((0.0, -0.71, -0.008)) @ Euler((math.radians(-90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
    )
    # Tapered tail cone
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=20,
        radius1=0.095, radius2=0.020, depth=0.30,
        matrix=Matrix.Translation((0.0, +0.69, 0.010)) @ Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
    )

    # Swept conventional main wings + tall 90-deg vertical tail fin (huge radar corner reflector!)
    for side in (-1.0, +1.0):
        # Main wing
        wm = Matrix.Translation((side * 0.38, 0.06, -0.02)) @ Euler((0.0, 0.0, side * math.radians(-22.0)), "XYZ").to_matrix().to_4x4()
        bmesh.ops.create_cube(bm, size=1.0, matrix=wm @ Matrix.Diagonal((0.72, 0.24, 0.026, 1.0)))
        # Horizontal tailplane
        hm = Matrix.Translation((side * 0.22, 0.62, 0.02)) @ Euler((0.0, 0.0, side * math.radians(-24.0)), "XYZ").to_matrix().to_4x4()
        bmesh.ops.create_cube(bm, size=1.0, matrix=hm @ Matrix.Diagonal((0.40, 0.14, 0.018, 1.0)))
        # Round exposed underwing turbofan nacelle (massive direct radar reflector!)
        bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=20,
            radius1=0.062, radius2=0.056, depth=0.28,
            matrix=Matrix.Translation((side * 0.26, -0.04, -0.09)) @ Euler((math.radians(90.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
        )

    # Tall 90-degree vertical tail fin
    vm = Matrix.Translation((0.0, 0.60, 0.22)) @ Euler((math.radians(-24.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4()
    bmesh.ops.create_cube(bm, size=1.0, matrix=vm @ Matrix.Diagonal((0.024, 0.26, 0.34, 1.0)))

    for f_face in bm.faces:
        f_face.material_index = 0

    jet_obj = _mesh_from_bm("Conv_Jet_Mesh", bm, [mat_echo, mat_dark], col, parent=root, smooth=True)
    _key_visibility(jet_obj, [(221, 540)], total_frames)

    # Animate pose & blazing crimson radar-echo flare in Beat 2
    for f in range(221, 541, 4):
        pos, rot_deg = C.conv_jet_pose_at(float(f))
        root.location = Vector(pos)
        root.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)

        # Blaze bright crimson radar echo right after f=295 -> peak at f=318..430
        flare = C.smoothstep(285.0, 322.0, float(f)) * (0.72 + 0.28 * math.sin((f - 285.0) * 0.22))
        echo_val.outputs[0].default_value = 1.65 * flare
        echo_val.outputs[0].keyframe_insert("default_value", frame=f)

    return {"root": root, "mesh": jet_obj, "echo_val": echo_val}


def build_stealth_aircraft(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Assemble Hero 172-ft Stealth Bomber (with X-ray S-ducts & turbines) + Beat 2 Conventional Jet."""
    root = bpy.data.objects.new("Stealth_Bomber_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    col.objects.link(root)

    mat_ram, xray_mix_val, scan_y_val = _mat_stealth_ram_xray(root)
    mat_glass = _mat_cockpit_glass()
    mat_exhaust = _mat_exhaust_glow()

    hull_obj = _build_stealth_hull_mesh(col, root, mat_ram, mat_glass, mat_exhaust)
    internals = _build_serpentine_sducts_and_turbines(col, root, total_frames)
    conv_jet = _build_conventional_jet(col, total_frames)

    # Bomber visible in Beats 1, 2, 3, 4, and 6 (hidden in Beat 5 pure CAD blueprint)
    _key_visibility(hull_obj, [(1, 1320), (1561, total_frames)], total_frames)

    # Keyframe Stealth Bomber trajectory & Local-Space X-Ray scan across 1..1800
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

    # Hard cuts strictly CONSTANT so bomber never streaks across frames at cut boundaries
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
