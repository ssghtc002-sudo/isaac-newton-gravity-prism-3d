"""
spine_skeleton_xray.py — Full-Body X-Ray Skeleton Rig & 28 G Axial Shockwave
for Episode #16 ("Why Jumping in a Falling Elevator WON'T Save You").

Owned by the Full-Body X-Ray Skeleton Rig Artist (`AGENTS.md` Rule 7).
Imports all timings, passenger metrics, and compression curves from
`elevator_contract.py` (never duplicates constants).

Public API:
    build_skeleton_xray(
        col: bpy.types.Collection,
        carrier: bpy.types.Object,
        rig: bpy.types.Object | None,
        total_frames: int = 1800,
    ) -> dict[str, Any]

Returns:
    {
        "root":      bpy.types.Object (`Skel_Root` Empty parented to `carrier`),
        "vertebrae": list[bpy.types.Object] (24 Mesh objects `Skel_Vert_L5`..`Skel_Vert_C1`),
        "discs":     list[bpy.types.Object] (24 Mesh objects `Skel_Disc_00_S1L5`..`Skel_Disc_23_C2C1`),
        "leg_bones": list[bpy.types.Object] (femurs, patellas, tibias/fibulas, ankle/foot clusters),
        "pelvis":    bpy.types.Object (`Skel_Pelvis` sculpted sacrum & iliac wings),
        "ribcage":   bpy.types.Object (`Skel_Ribcage` sternum, 10 rib pairs & clavicles),
        "skull":     bpy.types.Object (`Skel_Skull` sculpted cranium, orbits & mandible),
        "all_objs":  list[bpy.types.Object] (all created `Skel_*` objects),
    }
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(EPISODE_DIR)), "src")
for _p in (SCRIPT_DIR, REPO_SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elevator_contract as C  # noqa: E402

PREFIX = "Skel_"

# Visibility strictly covers Beat 4 X-ray window (f=905..1238 inclusive)
SKEL_VIS_RANGE: tuple[int, int] = (905, 1238)

# Evaluated bone-tracking window around Beat 4
EVAL_START: int = 900
EVAL_END: int = 1240
EVAL_STEP: int = 2
MILESTONE_FRAMES: tuple[int, ...] = (
    900, 905, 910, 934, 935, 938, 940, 955, 965, 970,
    985, 1005, 1020, 1060, 1120, 1185, 1205, 1210, 1235, 1236, 1238, 1240,
)

# 24 anatomical vertebrae specs:
# (label, region, rx, ry, body_h, gap_below, wing_half_x, spinous_len, spinous_drop)
VERTEBRA_SPECS: list[tuple[str, str, float, float, float, float, float, float, float]] = [
    # Lumbar (L5..L1)
    ("L5", "L", 0.0238, 0.0175, 0.0245, 0.0066, 0.039, 0.024, -0.004),
    ("L4", "L", 0.0232, 0.0170, 0.0240, 0.0065, 0.038, 0.025, -0.004),
    ("L3", "L", 0.0226, 0.0166, 0.0235, 0.0063, 0.037, 0.025, -0.005),
    ("L2", "L", 0.0220, 0.0162, 0.0230, 0.0060, 0.036, 0.024, -0.005),
    ("L1", "L", 0.0214, 0.0158, 0.0225, 0.0058, 0.034, 0.023, -0.006),
    # Thoracic (T12..T1)
    ("T12", "T", 0.0204, 0.0154, 0.0215, 0.0054, 0.032, 0.024, -0.010),
    ("T11", "T", 0.0196, 0.0150, 0.0208, 0.0052, 0.031, 0.026, -0.014),
    ("T10", "T", 0.0190, 0.0146, 0.0202, 0.0050, 0.030, 0.028, -0.016),
    ("T9",  "T", 0.0183, 0.0143, 0.0196, 0.0049, 0.029, 0.029, -0.018),
    ("T8",  "T", 0.0177, 0.0140, 0.0190, 0.0048, 0.028, 0.030, -0.020),
    ("T7",  "T", 0.0172, 0.0137, 0.0185, 0.0047, 0.028, 0.031, -0.021),
    ("T6",  "T", 0.0167, 0.0134, 0.0180, 0.0046, 0.027, 0.031, -0.021),
    ("T5",  "T", 0.0162, 0.0131, 0.0176, 0.0045, 0.026, 0.030, -0.019),
    ("T4",  "T", 0.0158, 0.0128, 0.0172, 0.0044, 0.026, 0.028, -0.018),
    ("T3",  "T", 0.0154, 0.0126, 0.0168, 0.0043, 0.025, 0.026, -0.016),
    ("T2",  "T", 0.0150, 0.0123, 0.0164, 0.0042, 0.024, 0.025, -0.013),
    ("T1",  "T", 0.0146, 0.0120, 0.0160, 0.0041, 0.024, 0.024, -0.010),
    # Cervical (C7..C1)
    ("C7", "C", 0.0138, 0.0114, 0.0146, 0.0039, 0.022, 0.023, -0.007),
    ("C6", "C", 0.0132, 0.0110, 0.0140, 0.0037, 0.021, 0.019, -0.005),
    ("C5", "C", 0.0128, 0.0107, 0.0135, 0.0035, 0.020, 0.017, -0.004),
    ("C4", "C", 0.0124, 0.0104, 0.0130, 0.0034, 0.020, 0.016, -0.004),
    ("C3", "C", 0.0120, 0.0102, 0.0126, 0.0033, 0.019, 0.015, -0.003),
    ("C2", "C", 0.0118, 0.0100, 0.0122, 0.0032, 0.019, 0.016, -0.003),
    ("C1", "C", 0.0160, 0.0125, 0.0104, 0.0034, 0.023, 0.010,  0.000),
]

SPINE_BONE_CANDIDATES: tuple[str, ...] = (
    "spine05", "spine04", "spine03", "spine02", "spine01",
    "neck01", "neck02", "neck03", "head",
)


# ---------------------------------------------------------------------------
# COLLECTION, VISIBILITY & INTERPOLATION HELPERS
# ---------------------------------------------------------------------------
def _link_to_col(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def _create_empty(
    name: str,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.04
    _link_to_col(obj, col)
    if parent is not None:
        obj.parent = parent
    return obj


def _set_constant_visibility(
    obj: bpy.types.Object,
    vis_start: int,
    vis_end: int,
) -> None:
    """Strictly visible during [vis_start, vis_end] inclusive via CONSTANT keyframes."""
    for attr in ("hide_viewport", "hide_render"):
        setattr(obj, attr, True)
        obj.keyframe_insert(data_path=attr, frame=1)
        setattr(obj, attr, False)
        obj.keyframe_insert(data_path=attr, frame=vis_start)
        setattr(obj, attr, True)
        obj.keyframe_insert(data_path=attr, frame=vis_end + 1)

    if obj.animation_data and obj.animation_data.action:
        for fc in C.iter_action_fcurves(obj.animation_data.action):
            if fc.data_path in ("hide_viewport", "hide_render"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _set_fcurve_linear_or_bezier(obj: bpy.types.Object, interp: str = "LINEAR") -> None:
    if not (obj.animation_data and obj.animation_data.action):
        return
    for fc in C.iter_action_fcurves(obj.animation_data.action):
        if fc.data_path in ("location", "rotation_quaternion", "scale"):
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def _make_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """Finalize bmesh into object with material slot preserved BEFORE bm.to_mesh."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.materials.append(mat)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    _link_to_col(obj, col)
    obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    obj.visible_shadow = False
    return obj


# ---------------------------------------------------------------------------
# LOCAL-SPACE 28 G AXIAL SHOCKWAVE X-RAY SHADERS (TexCoord.Object = carrier)
# ---------------------------------------------------------------------------
def _shockwave_curve_values(f: float) -> tuple[float, float]:
    """Return (wave_z_meters, shock_amp_0_to_1) for Beat 4 impact sweep."""
    if f < 935.0:
        return (-0.15, 0.0)
    if f <= 1020.0:
        # Sweeps upward along leg -> pelvis -> spine (z: 0.05 -> 1.65) across f=935..1020
        u = C.smootherstep(935.0, 1020.0, f)
        wave_z = C.lerp(0.05, 1.65, u ** 0.92)
        amp = C.smootherstep(934.0, 952.0, f)
        return (wave_z, amp)
    if f <= 1235.0:
        # Wave passes head, then sustained 28 G pulse cools gradually before f=1235
        wave_z = C.lerp(1.65, 1.95, C.smoothstep(1020.0, 1180.0, f))
        cool = 1.0 - C.smootherstep(1025.0, 1230.0, f)
        throb = 0.08 * math.sin((f - 1020.0) * 0.28) * cool
        amp = C.clamp(cool * 0.92 + throb, 0.0, 1.0)
        return (wave_z, amp)
    return (1.95, 0.0)


def _get_mix_sockets(mix_node: bpy.types.Node, sock_type: str) -> tuple[Any, Any, Any, Any]:
    """Blender 4.x/5.x safe socket lookup for ShaderNodeMix ('RGBA' or 'VALUE'), skipping Factor sockets."""
    # inputs[0] is Float Factor (`VALUE`), inputs[1] is Vector Factor (`VECTOR`); A/B start at inputs[2:].
    typed_in = [s for s in mix_node.inputs[2:] if s.type == sock_type]
    typed_out = [s for s in mix_node.outputs if s.type == sock_type]
    return mix_node.inputs[0], typed_in[0], typed_in[1], typed_out[0]


def _create_xray_bone_material(
    name: str,
    carrier: bpy.types.Object,
    is_disc: bool = False,
) -> bpy.types.Material:
    """
    Build 100% Local-Space (`TexCoord.Object` = `carrier`) medical X-ray bone shader.
    - Before impact (f=910..935): healthy glowing ice-cyan `(0.35, 0.88, 1.0)`, Emiss <= 1.92
    - During 28 G sweep (f=935..1020): blazing molten amber-crimson pulse sweeping upward
      along shin/knees & lumbar L5-L1 (`Emiss <= 1.95` AgX-safe!), cooling before f=1235.
    - `LightPath.Is Shadow Ray -> Transparent BSDF` prevents self-shadow blackout.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1180, 0)

    # Local-space coordinate bound directly to `carrier`
    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (-1250, 120)
    tex.object = carrier

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1040, 180)
    nt.links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # Subtle 3D trabecular / bone-density micro-variation (local-space)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-1040, -140)
    noise.inputs["Scale"].default_value = 34.0 if not is_disc else 22.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(tex.outputs["Object"], noise.inputs["Vector"])

    # Keyframed wave_z & shock_amp Value nodes
    val_wave_z = nt.nodes.new("ShaderNodeValue")
    val_wave_z.name = "WaveZ"
    val_wave_z.label = "WaveZ"
    val_wave_z.location = (-1040, 380)

    val_amp = nt.nodes.new("ShaderNodeValue")
    val_amp.name = "ShockAmp"
    val_amp.label = "ShockAmp"
    val_amp.location = (-1040, 520)

    for kf in (
        900, 910, 934, 935, 940, 950, 962, 975, 985, 998, 1010,
        1020, 1045, 1075, 1110, 1150, 1190, 1215, 1235, 1240,
    ):
        wz, amp = _shockwave_curve_values(float(kf))
        val_wave_z.outputs[0].default_value = wz
        val_wave_z.outputs[0].keyframe_insert("default_value", frame=kf)
        val_amp.outputs[0].default_value = amp
        val_amp.outputs[0].keyframe_insert("default_value", frame=kf)

    # --- Wavefront ring: front = clamp(1 - |Z - wave_z| / 0.16, 0, 1)^2 ---
    sub_zw = nt.nodes.new("ShaderNodeMath")
    sub_zw.operation = "SUBTRACT"
    sub_zw.location = (-820, 320)
    nt.links.new(sep.outputs["Z"], sub_zw.inputs[0])
    nt.links.new(val_wave_z.outputs[0], sub_zw.inputs[1])

    abs_zw = nt.nodes.new("ShaderNodeMath")
    abs_zw.operation = "ABSOLUTE"
    abs_zw.location = (-650, 320)
    nt.links.new(sub_zw.outputs["Value"], abs_zw.inputs[0])

    front_map = nt.nodes.new("ShaderNodeMapRange")
    front_map.clamp = True
    front_map.location = (-470, 320)
    front_map.inputs["From Min"].default_value = 0.0
    front_map.inputs["From Max"].default_value = 0.17
    front_map.inputs["To Min"].default_value = 1.0
    front_map.inputs["To Max"].default_value = 0.0
    nt.links.new(abs_zw.outputs["Value"], front_map.inputs["Value"])

    front_pow = nt.nodes.new("ShaderNodeMath")
    front_pow.operation = "POWER"
    front_pow.location = (-290, 320)
    front_pow.inputs[1].default_value = 1.85
    nt.links.new(front_map.outputs["Result"], front_pow.inputs[0])

    # --- Swept trail: passed = clamp((wave_z - Z) / 0.12, 0, 1) ---
    sub_wz = nt.nodes.new("ShaderNodeMath")
    sub_wz.operation = "SUBTRACT"
    sub_wz.location = (-820, 120)
    nt.links.new(val_wave_z.outputs[0], sub_wz.inputs[0])
    nt.links.new(sep.outputs["Z"], sub_wz.inputs[1])

    passed_map = nt.nodes.new("ShaderNodeMapRange")
    passed_map.clamp = True
    passed_map.location = (-650, 120)
    passed_map.inputs["From Min"].default_value = -0.02
    passed_map.inputs["From Max"].default_value = 0.12
    passed_map.inputs["To Min"].default_value = 0.0
    passed_map.inputs["To Max"].default_value = 1.0
    nt.links.new(sub_wz.outputs["Value"], passed_map.inputs["Value"])

    # --- Critical High-Stress Zone 1: Tibias & Knees (Z ~ 0.06..0.56) ---
    shin_lo = nt.nodes.new("ShaderNodeMapRange")
    shin_lo.clamp = True
    shin_lo.location = (-820, -340)
    shin_lo.inputs["From Min"].default_value = 0.03
    shin_lo.inputs["From Max"].default_value = 0.14
    shin_lo.inputs["To Min"].default_value = 0.0
    shin_lo.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], shin_lo.inputs["Value"])

    shin_hi = nt.nodes.new("ShaderNodeMapRange")
    shin_hi.clamp = True
    shin_hi.location = (-820, -540)
    shin_hi.inputs["From Min"].default_value = 0.44
    shin_hi.inputs["From Max"].default_value = 0.58
    shin_hi.inputs["To Min"].default_value = 1.0
    shin_hi.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], shin_hi.inputs["Value"])

    shin_band = nt.nodes.new("ShaderNodeMath")
    shin_band.operation = "MULTIPLY"
    shin_band.location = (-620, -440)
    nt.links.new(shin_lo.outputs["Result"], shin_band.inputs[0])
    nt.links.new(shin_hi.outputs["Result"], shin_band.inputs[1])

    # --- Critical High-Stress Zone 2: Pelvis & Lumbar L5..L1 (Z ~ 0.90..1.25) ---
    lumb_lo = nt.nodes.new("ShaderNodeMapRange")
    lumb_lo.clamp = True
    lumb_lo.location = (-820, -740)
    lumb_lo.inputs["From Min"].default_value = 0.88
    lumb_lo.inputs["From Max"].default_value = 0.98
    lumb_lo.inputs["To Min"].default_value = 0.0
    lumb_lo.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], lumb_lo.inputs["Value"])

    lumb_hi = nt.nodes.new("ShaderNodeMapRange")
    lumb_hi.clamp = True
    lumb_hi.location = (-820, -940)
    lumb_hi.inputs["From Min"].default_value = 1.16
    lumb_hi.inputs["From Max"].default_value = 1.30
    lumb_hi.inputs["To Min"].default_value = 1.0
    lumb_hi.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], lumb_hi.inputs["Value"])

    lumb_band = nt.nodes.new("ShaderNodeMath")
    lumb_band.operation = "MULTIPLY"
    lumb_band.location = (-620, -840)
    nt.links.new(lumb_lo.outputs["Result"], lumb_band.inputs[0])
    nt.links.new(lumb_hi.outputs["Result"], lumb_band.inputs[1])

    zone_max = nt.nodes.new("ShaderNodeMath")
    zone_max.operation = "MAXIMUM"
    zone_max.location = (-440, -640)
    nt.links.new(shin_band.outputs["Value"], zone_max.inputs[0])
    nt.links.new(lumb_band.outputs["Value"], zone_max.inputs[1])

    # Zone weight = base_trail + zone_boost * zone_max
    zone_scale = nt.nodes.new("ShaderNodeMath")
    zone_scale.operation = "MULTIPLY_ADD"
    zone_scale.location = (-260, -520)
    zone_scale.inputs[1].default_value = 0.78 if not is_disc else 0.92
    zone_scale.inputs[2].default_value = 0.24 if not is_disc else 0.42
    nt.links.new(zone_max.outputs["Value"], zone_scale.inputs[0])

    trail_weighted = nt.nodes.new("ShaderNodeMath")
    trail_weighted.operation = "MULTIPLY"
    trail_weighted.location = (-260, 120)
    nt.links.new(passed_map.outputs["Result"], trail_weighted.inputs[0])
    nt.links.new(zone_scale.outputs["Value"], trail_weighted.inputs[1])

    heat_raw = nt.nodes.new("ShaderNodeMath")
    heat_raw.operation = "MAXIMUM"
    heat_raw.location = (-80, 220)
    nt.links.new(front_pow.outputs["Value"], heat_raw.inputs[0])
    nt.links.new(trail_weighted.outputs["Value"], heat_raw.inputs[1])

    heat_amp = nt.nodes.new("ShaderNodeMath")
    heat_amp.operation = "MULTIPLY"
    heat_amp.use_clamp = True
    heat_amp.location = (100, 280)
    nt.links.new(heat_raw.outputs["Value"], heat_amp.inputs[0])
    nt.links.new(val_amp.outputs[0], heat_amp.inputs[1])

    # --- Color Ramp: Ice-Cyan -> Molten Amber -> Blazing Crimson ---
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.location = (-100, -180)
    lw.inputs["Blend"].default_value = 0.36

    # Healthy Cyan color (interior darker cyan -> rim (0.35, 0.88, 1.0))
    mix_cyan = nt.nodes.new("ShaderNodeMix")
    mix_cyan.data_type = "RGBA"
    mix_cyan.location = (140, -140)
    mc_fac, mc_a, mc_b, mc_out = _get_mix_sockets(mix_cyan, "RGBA")
    if is_disc:
        mc_a.default_value = (0.08, 0.72, 0.62, 1.0)
        mc_b.default_value = (0.20, 0.98, 0.84, 1.0)
    else:
        mc_a.default_value = (0.14, 0.56, 0.86, 1.0)
        mc_b.default_value = (0.35, 0.88, 1.00, 1.0)
    nt.links.new(lw.outputs["Facing"], mc_fac)

    # Molten shock color ramp (heat_amp -> deep amber-orange -> bright crimson)
    ramp_shock = nt.nodes.new("ShaderNodeValToRGB")
    ramp_shock.location = (140, 60)
    cr = ramp_shock.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (1.00, 0.42, 0.04, 1.0)
    cr.elements[1].position = 0.55
    cr.elements[1].color = (1.00, 0.18, 0.02, 1.0)
    e2 = cr.elements.new(0.95)
    e2.color = (1.00, 0.06, 0.015, 1.0) if not is_disc else (1.00, 0.10, 0.02, 1.0)
    nt.links.new(heat_amp.outputs["Value"], ramp_shock.inputs["Fac"])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.location = (420, 120)
    mcol_fac, mcol_a, mcol_b, mcol_out = _get_mix_sockets(mix_col, "RGBA")
    nt.links.new(heat_amp.outputs["Value"], mcol_fac)
    nt.links.new(mc_out, mcol_a)
    nt.links.new(ramp_shock.outputs["Color"], mcol_b)

    # Modulate gently by micro trabecular noise (0.84..1.00) so bone texture pops
    noise_map = nt.nodes.new("ShaderNodeMapRange")
    noise_map.clamp = True
    noise_map.location = (140, -340)
    noise_map.inputs["From Min"].default_value = 0.25
    noise_map.inputs["From Max"].default_value = 0.75
    noise_map.inputs["To Min"].default_value = 0.84
    noise_map.inputs["To Max"].default_value = 1.00
    nt.links.new(noise.outputs["Fac"], noise_map.inputs["Value"])

    # Emission Strength:
    # - Cyan baseline: 1.45 core .. 1.88 rim (discs: 1.55 .. 1.92; strictly <= 1.95 AgX-safe!)
    # - Shockwave peak: 1.95 (strictly <= 1.95 AgX-safe!)
    em_cyan = nt.nodes.new("ShaderNodeMapRange")
    em_cyan.clamp = True
    em_cyan.location = (380, -180)
    em_cyan.inputs["From Min"].default_value = 0.0
    em_cyan.inputs["From Max"].default_value = 1.0
    em_cyan.inputs["To Min"].default_value = 1.45 if not is_disc else 1.55
    em_cyan.inputs["To Max"].default_value = 1.88 if not is_disc else 1.92
    nt.links.new(lw.outputs["Facing"], em_cyan.inputs["Value"])

    em_mix = nt.nodes.new("ShaderNodeMix")
    em_mix.data_type = "FLOAT"
    em_mix.location = (580, -80)
    em_fac, em_a, em_b, em_out = _get_mix_sockets(em_mix, "VALUE")
    em_b.default_value = 1.95
    nt.links.new(heat_amp.outputs["Value"], em_fac)
    nt.links.new(em_cyan.outputs["Result"], em_a)

    em_mod = nt.nodes.new("ShaderNodeMath")
    em_mod.operation = "MULTIPLY"
    em_mod.use_clamp = False
    em_mod.location = (740, -140)
    nt.links.new(em_out, em_mod.inputs[0])
    nt.links.new(noise_map.outputs["Result"], em_mod.inputs[1])

    em_node = nt.nodes.new("ShaderNodeEmission")
    em_node.location = (760, 120)
    nt.links.new(mcol_out, em_node.inputs["Color"])
    nt.links.new(em_mod.outputs["Value"], em_node.inputs["Strength"])

    # X-ray shell opacity via LayerWeight (crisp cortical bone readability inside glass rim body)
    alpha_map = nt.nodes.new("ShaderNodeMapRange")
    alpha_map.clamp = True
    alpha_map.location = (520, -340)
    alpha_map.inputs["From Min"].default_value = 0.0
    alpha_map.inputs["From Max"].default_value = 1.0
    alpha_map.inputs["To Min"].default_value = 0.78 if not is_disc else 0.90
    alpha_map.inputs["To Max"].default_value = 0.98 if not is_disc else 1.00
    nt.links.new(lw.outputs["Facing"], alpha_map.inputs["Value"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (760, -300)

    mix_xray = nt.nodes.new("ShaderNodeMixShader")
    mix_xray.location = (920, 40)
    nt.links.new(alpha_map.outputs["Result"], mix_xray.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix_xray.inputs[1])
    nt.links.new(em_node.outputs["Emission"], mix_xray.inputs[2])

    # Mandatory Shadow Ray bypass (`LightPath.Is Shadow Ray -> Transparent BSDF`)
    lp = nt.nodes.new("ShaderNodeLightPath")
    lp.location = (760, 320)

    mix_shadow = nt.nodes.new("ShaderNodeMixShader")
    mix_shadow.location = (1050, 120)
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_shadow.inputs["Fac"])
    nt.links.new(mix_xray.outputs["Shader"], mix_shadow.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_shadow.inputs[2])

    nt.links.new(mix_shadow.outputs["Shader"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------
# PROCEDURAL ANATOMICAL MESH BUILDERS
# Canonical frame for ALL meshes:
#   +X = Anatomical Left, -Y = Anterior (Front), +Y = Posterior (Back), +Z = Up
# ---------------------------------------------------------------------------
def _add_tube_loft(
    bm: bmesh.types.BMesh,
    rings: list[list[Vector]],
    cap_start: bool = True,
    cap_end: bool = True,
) -> None:
    """Connect ordered vertex rings into a watertight lofted quad tube."""
    ring_verts: list[list[bmesh.types.BMVert]] = []
    for pts in rings:
        ring_verts.append([bm.verts.new(p) for p in pts])

    n_seg = len(ring_verts[0])
    for r_idx in range(len(ring_verts) - 1):
        r0 = ring_verts[r_idx]
        r1 = ring_verts[r_idx + 1]
        for i in range(n_seg):
            j = (i + 1) % n_seg
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))

    if cap_start and len(ring_verts[0]) >= 3:
        bm.faces.new(list(reversed(ring_verts[0])))
    if cap_end and len(ring_verts[-1]) >= 3:
        bm.faces.new(ring_verts[-1])


def _kidney_ring(rx: float, ry: float, z: float, n_pts: int = 20) -> list[Vector]:
    """Cross-section of an anatomical vertebral body (concave posterior spinal canal notch)."""
    pts: list[Vector] = []
    for i in range(n_pts):
        ang = 2.0 * math.pi * i / n_pts
        ca = math.cos(ang)
        sa = math.sin(ang)
        # Posterior (+Y, sa > 0) concavity for spinal cord channel
        notch = 1.0 - (0.24 * (sa ** 2) if sa > 0.0 else 0.0)
        pts.append(Vector((rx * ca, ry * sa * notch, z)))
    return pts


def _build_vertebra_mesh(
    name: str,
    spec: tuple[str, str, float, float, float, float, float, float, float],
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    label, region, rx, ry, body_h, _gap, wing_x, spin_len, spin_drop = spec
    bm = bmesh.new()
    hh = body_h * 0.5

    # 1. Vertebral body (hourglass waist + flared cortical endplate rims)
    if label == "C1":
        # Atlas C1 is an open bony ring with lateral masses instead of a thick centrum
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        for v in bm.verts:
            # Stretch into lateral-mass ring profile
            ang = math.atan2(v.co.y, v.co.x)
            ring_r = 0.0145 + 0.004 * abs(math.cos(ang))
            v.co = Vector((
                v.co.x * ring_r * 1.15,
                v.co.y * (ring_r * 0.85),
                v.co.z * hh * 1.05,
            ))
    else:
        stations = [
            (-hh, 0.96),
            (-hh * 0.72, 1.04),
            (0.0, 0.87),
            (hh * 0.72, 1.04),
            (hh, 0.96),
        ]
        rings = [_kidney_ring(rx * sc, ry * sc, z, n_pts=20) for z, sc in stations]
        _add_tube_loft(bm, rings, cap_start=True, cap_end=True)

    # 2. Neural arch (left & right pedicles + laminae wrapping behind +Y spinal canal)
    arch_r = max(0.0032, rx * 0.22)
    arch_pts = [
        Vector((-rx * 0.58, ry * 0.45, hh * 0.15)),
        Vector((-rx * 0.68, ry * 1.18, hh * 0.06)),
        Vector((-rx * 0.36, ry * 1.62, -hh * 0.05)),
        Vector((0.0, ry * 1.76, -hh * 0.10)),
        Vector((rx * 0.36, ry * 1.62, -hh * 0.05)),
        Vector((rx * 0.68, ry * 1.18, hh * 0.06)),
        Vector((rx * 0.58, ry * 0.45, hh * 0.15)),
    ]
    arch_rings: list[list[Vector]] = []
    for k, p in enumerate(arch_pts):
        t = (arch_pts[min(k + 1, len(arch_pts) - 1)] - arch_pts[max(k - 1, 0)]).normalized()
        up = Vector((0.0, 0.0, 1.0))
        side = t.cross(up).normalized()
        up2 = side.cross(t).normalized()
        ring = []
        for s in range(8):
            a = 2.0 * math.pi * s / 8
            ring.append(p + side * (arch_r * 0.85 * math.cos(a)) + up2 * (arch_r * 1.25 * math.sin(a)))
        arch_rings.append(ring)
    _add_tube_loft(bm, arch_rings, cap_start=True, cap_end=True)

    # 3. Left & Right Transverse Processes (wings)
    for side_sign in (-1.0, 1.0):
        w_rings: list[list[Vector]] = []
        w_stations = [
            (side_sign * rx * 0.62, ry * 0.92, hh * 0.12, arch_r * 0.95),
            (side_sign * (rx * 0.55 + wing_x * 0.50), ry * 1.15, hh * 0.08, arch_r * 0.78),
            (side_sign * wing_x, ry * 1.32, hh * 0.02, arch_r * 0.55),
        ]
        for wx, wy, wz, wr in w_stations:
            ring = []
            for s in range(8):
                a = 2.0 * math.pi * s / 8
                ring.append(Vector((wx, wy + wr * 0.8 * math.cos(a), wz + wr * 1.15 * math.sin(a))))
            w_rings.append(ring)
        _add_tube_loft(bm, w_rings, cap_start=True, cap_end=True)

    # 4. Posterior Spinous Process blade (+Y dorsal)
    sp_base_y = ry * 1.68
    sp_tip_y = sp_base_y + spin_len
    sp_r = max(0.0026, rx * 0.17)
    sp_stations = [
        (sp_base_y, -hh * 0.06, sp_r * 1.1, sp_r * 1.6),
        ((sp_base_y + sp_tip_y) * 0.5, spin_drop * 0.55, sp_r * 0.85, sp_r * 1.4),
        (sp_tip_y, spin_drop, sp_r * 0.62, sp_r * 1.05),
    ]
    sp_rings: list[list[Vector]] = []
    for sy, sz, sxr, szr in sp_stations:
        ring = []
        for s in range(8):
            a = 2.0 * math.pi * s / 8
            ring.append(Vector((sxr * math.cos(a), sy, sz + szr * math.sin(a))))
        sp_rings.append(ring)
    _add_tube_loft(bm, sp_rings, cap_start=True, cap_end=True)

    # 5. Superior & Inferior Articular Facet Pillars
    for side_sign in (-1.0, 1.0):
        fac = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        for v in fac["verts"]:
            v.co = Vector((
                v.co.x * (arch_r * 0.95) + side_sign * rx * 0.62,
                v.co.y * (arch_r * 0.95) + ry * 1.12,
                v.co.z * (hh * 0.95),
            ))

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_disc_mesh(
    name: str,
    rx: float,
    ry: float,
    disc_h: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    bm = bmesh.new()
    hh = disc_h * 0.5
    stations = [
        (-hh, 0.92),
        (-hh * 0.45, 1.04),
        (0.0, 1.08),
        (hh * 0.45, 1.04),
        (hh, 0.92),
    ]
    rings = [_kidney_ring(rx * sc, ry * sc, z, n_pts=20) for z, sc in stations]
    _add_tube_loft(bm, rings, cap_start=True, cap_end=True)
    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_pelvis_mesh(
    name: str,
    half_hip: float,
    sacrum_top_z: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Sculpted Sacrum + Iliac Wings + Acetabulum Sockets + Pubic/Ischial Rings.
    Origin `(0,0,0)` sits at exact midpoint between left and right hip sockets `(+-half_hip, 0, 0)`.
    """
    bm = bmesh.new()
    hh = max(0.075, half_hip)

    # 1. Sacrum wedge (5 fused segments tapering down to coccyx)
    sac_rings: list[list[Vector]] = []
    sac_stations = [
        (sacrum_top_z + 0.008, 0.044, 0.017, 0.040),
        (sacrum_top_z - 0.022, 0.048, 0.015, 0.052),
        (sacrum_top_z - 0.056, 0.036, 0.012, 0.062),
        (sacrum_top_z - 0.092, 0.022, 0.009, 0.068),
        (sacrum_top_z - 0.122, 0.010, 0.006, 0.062),
    ]
    for sz, sxr, syr, sy_off in sac_stations:
        ring = []
        for s in range(14):
            a = 2.0 * math.pi * s / 14
            ring.append(Vector((sxr * math.cos(a), sy_off + syr * math.sin(a), sz)))
        sac_rings.append(ring)
    _add_tube_loft(bm, sac_rings, cap_start=True, cap_end=True)

    # 2. Left & Right Iliac Wings, Acetabulum Sockets, and Ischiopubic Loops
    for side in (-1.0, 1.0):
        # Iliac blade loft from sacroiliac root -> sweeping flared iliac crest
        crest_pts = [
            (Vector((side * 0.040, 0.046, sacrum_top_z - 0.010)), Vector((side * 0.052, 0.024, 0.008))),
            (Vector((side * (hh * 0.88), 0.036, sacrum_top_z + 0.038)), Vector((side * (hh * 0.72), 0.010, 0.006))),
            (Vector((side * (hh * 1.24), 0.002, sacrum_top_z + 0.052)), Vector((side * (hh * 0.94), -0.006, 0.012))),
            (Vector((side * (hh * 1.22), -0.034, sacrum_top_z + 0.030)), Vector((side * (hh * 0.96), -0.018, 0.006))),
            (Vector((side * (hh * 1.06), -0.046, 0.016)), Vector((side * (hh * 0.96), -0.012, 0.000))),
        ]
        # Build flattened bone-blade cross-section at each station connecting iliac crest to acetabular root
        wing_rings: list[list[Vector]] = []
        for top_p, bot_p in crest_pts:
            mid = (top_p + bot_p) * 0.5
            span = (top_p - bot_p)
            thick = Vector((-side * 0.008, 0.005, 0.0)).normalized() * 0.0075
            ring = [
                bot_p - thick * 0.6,
                mid - thick * 0.9,
                top_p - thick * 1.1,
                top_p + thick * 1.1,
                mid + thick * 0.9,
                bot_p + thick * 0.6,
            ]
            wing_rings.append(ring)
        _add_tube_loft(bm, wing_rings, cap_start=True, cap_end=True)

        # Acetabulum socket collar directly at hip joint (side * hh, 0, 0)
        sock = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.025)
        for v in sock["verts"]:
            v.co = Vector((
                v.co.x * 0.85 + side * hh * 0.96,
                v.co.y * 1.05,
                v.co.z * 1.05,
            ))

        # Ischiopubic bony loop: acetabulum -> ischial tuberosity -> inferior ramus -> pubic symphysis -> superior ramus
        loop_path = [
            Vector((side * hh * 0.92, 0.002, -0.010)),
            Vector((side * hh * 0.78, 0.028, -0.056)),
            Vector((side * hh * 0.42, -0.010, -0.066)),
            Vector((side * 0.012, -0.052, -0.046)),
            Vector((side * hh * 0.46, -0.036, -0.016)),
            Vector((side * hh * 0.86, -0.014, -0.004)),
        ]
        loop_rings: list[list[Vector]] = []
        n_lp = len(loop_path)
        for k in range(n_lp):
            p = loop_path[k]
            t = (loop_path[(k + 1) % n_lp] - loop_path[(k - 1) % n_lp]).normalized()
            ref = Vector((0.0, 1.0, 0.0)) if abs(t.y) < 0.85 else Vector((0.0, 0.0, 1.0))
            b1 = t.cross(ref).normalized()
            b2 = t.cross(b1).normalized()
            rad = 0.0085 if k != 3 else 0.011
            ring = []
            for s in range(8):
                a = 2.0 * math.pi * s / 8
                ring.append(p + b1 * (rad * math.cos(a)) + b2 * (rad * math.sin(a)))
            loop_rings.append(ring)
        # Closed ring loft
        ring_verts = [[bm.verts.new(pt) for pt in r] for r in loop_rings]
        for k in range(n_lp):
            r0 = ring_verts[k]
            r1 = ring_verts[(k + 1) % n_lp]
            for s in range(8):
                sn = (s + 1) % 8
                bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_ribcage_mesh(
    name: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Sculpted Thoracic Ribcage (Sternum + 10 pairs of ribs + 2 Clavicles).
    Origin `(0,0,0)` anchors at mid-thoracic spine (`spine02` head, ~T4-T5).
    Dimensions fit safely inside 1.78m adult chest skin (>2.5 cm clearance).
    """
    bm = bmesh.new()

    # 1. Sternum (manubrium + gladiolus + xiphoid) along anterior chest
    st_stations = [
        (0.095, -0.072, 0.022, 0.006),   # Manubrium top
        (0.055, -0.082, 0.014, 0.006),   # Sternal angle
        (-0.010, -0.090, 0.016, 0.006),  # Mid body
        (-0.070, -0.092, 0.012, 0.005),  # Lower body
        (-0.105, -0.086, 0.005, 0.003),  # Xiphoid tip
    ]
    st_rings: list[list[Vector]] = []
    for sz, sy, sxr, syr in st_stations:
        ring = []
        for s in range(10):
            a = 2.0 * math.pi * s / 10
            ring.append(Vector((sxr * math.cos(a), sy + syr * math.sin(a), sz)))
        st_rings.append(ring)
    _add_tube_loft(bm, st_rings, cap_start=True, cap_end=True)

    # 2. 10 pairs of thoracic ribs (T1..T10)
    rib_specs = [
        # (z_post, z_ant, max_w, ant_x, ant_y, attach_sternum)
        (0.115,  0.088, 0.076, 0.018, -0.072, True),
        (0.088,  0.055, 0.095, 0.015, -0.081, True),
        (0.060,  0.022, 0.110, 0.015, -0.086, True),
        (0.032, -0.010, 0.118, 0.015, -0.089, True),
        (0.004, -0.042, 0.122, 0.014, -0.091, True),
        (-0.025, -0.072, 0.122, 0.012, -0.090, True),
        (-0.055, -0.095, 0.120, 0.022, -0.085, True),
        (-0.085, -0.116, 0.116, 0.045, -0.078, False),
        (-0.114, -0.136, 0.110, 0.066, -0.068, False),
        (-0.142, -0.156, 0.102, 0.082, -0.052, False),
    ]
    for z_post, z_ant, max_w, ant_x, ant_y, _att in rib_specs:
        for side in (-1.0, 1.0):
            pts = [
                Vector((side * 0.016, 0.020, z_post)),
                Vector((side * max_w * 0.62, 0.044, z_post - 0.004)),
                Vector((side * max_w, 0.002, (z_post + z_ant) * 0.5)),
                Vector((side * max_w * 0.78, -0.056, z_ant + 0.006)),
                Vector((side * ant_x, ant_y, z_ant)),
            ]
            # Interpolate smoothly into 9 stations along cubic-like arc
            smooth_pts: list[Vector] = []
            for t_i in range(9):
                u = t_i / 8.0
                # De Casteljau / Bernstein quartic evaluation for smooth organic rib curvature
                p0, p1, p2, p3, p4 = pts
                q = (
                    ((1 - u) ** 4) * p0
                    + 4 * ((1 - u) ** 3) * u * p1
                    + 6 * ((1 - u) ** 2) * (u ** 2) * p2
                    + 4 * (1 - u) * (u ** 3) * p3
                    + (u ** 4) * p4
                )
                smooth_pts.append(q)

            rib_rings: list[list[Vector]] = []
            for k, p in enumerate(smooth_pts):
                t = (smooth_pts[min(k + 1, 8)] - smooth_pts[max(k - 1, 0)]).normalized()
                up = Vector((0.0, 0.0, 1.0))
                out_n = t.cross(up).normalized()
                up2 = out_n.cross(t).normalized()
                ring = []
                for s in range(6):
                    a = 2.0 * math.pi * s / 6
                    ring.append(p + out_n * (0.0032 * math.cos(a)) + up2 * (0.0056 * math.sin(a)))
                rib_rings.append(ring)
            _add_tube_loft(bm, rib_rings, cap_start=True, cap_end=True)

    # 3. Left & Right Clavicles (S-curved collarbones)
    for side in (-1.0, 1.0):
        clav_pts = [
            Vector((side * 0.014, -0.072, 0.104)),
            Vector((side * 0.055, -0.078, 0.112)),
            Vector((side * 0.102, -0.048, 0.118)),
            Vector((side * 0.138, -0.014, 0.122)),
        ]
        clav_rings: list[list[Vector]] = []
        for k, p in enumerate(clav_pts):
            t = (clav_pts[min(k + 1, 3)] - clav_pts[max(k - 1, 0)]).normalized()
            up = Vector((0.0, 0.0, 1.0))
            b1 = t.cross(up).normalized()
            b2 = b1.cross(t).normalized()
            ring = []
            for s in range(8):
                a = 2.0 * math.pi * s / 8
                ring.append(p + b1 * (0.0058 * math.cos(a)) + b2 * (0.0052 * math.sin(a)))
            clav_rings.append(ring)
        _add_tube_loft(bm, clav_rings, cap_start=True, cap_end=True)

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_skull_mesh(
    name: str,
    head_len: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Anatomical Cranium + Brow + Orbits + Zygomatic Arches + Mandible Silhouette.
    Origin `(0,0,0)` sits at `head` bone head (C1/occipital junction), `+Z` up to crown (`~head_len`).
    """
    bm = bmesh.new()
    sc = max(0.75, min(1.15, head_len / 0.21))

    # 1. Cranial vault (sculpted sphere with occipital curve, temporal narrowing, brow ridge)
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co.x, v.co.y, v.co.z
        # Flatten base & narrow temples slightly
        rx = 0.067 * (0.88 if y < -0.2 else 1.0)
        ry = 0.084
        rz = 0.076 * (0.82 if z < 0.0 else 1.0)
        vx = x * rx
        vy = y * ry + 0.014
        vz = z * rz + 0.118
        # Subtle supraorbital brow ridge (-Y, z ~ 0.095)
        if y < -0.55 and -0.25 < z < 0.05:
            vy -= 0.007 * (1.0 - abs(z + 0.10) / 0.18)
        v.co = Vector((vx, vy, vz)) * sc

    # 2. Left & Right Eye Orbit Rims + Zygomatic Cheekbone Loops
    for side in (-1.0, 1.0):
        orb_pts = []
        for s in range(12):
            a = 2.0 * math.pi * s / 12
            ox = side * (0.028 + 0.016 * math.cos(a))
            oz = 0.082 + 0.014 * math.sin(a)
            oy = -0.066 + 0.009 * abs(math.cos(a))
            orb_pts.append(Vector((ox, oy, oz)) * sc)
        orb_rings: list[list[Vector]] = []
        for k in range(12):
            p = orb_pts[k]
            t = (orb_pts[(k + 1) % 12] - orb_pts[(k - 1) % 12]).normalized()
            fwd = Vector((0.0, -1.0, 0.0))
            b1 = t.cross(fwd).normalized()
            b2 = t.cross(b1).normalized()
            ring = []
            for s in range(6):
                a = 2.0 * math.pi * s / 6
                ring.append(p + b1 * (0.0042 * sc * math.cos(a)) + b2 * (0.0045 * sc * math.sin(a)))
            orb_rings.append(ring)
        ring_verts = [[bm.verts.new(pt) for pt in r] for r in orb_rings]
        for k in range(12):
            r0 = ring_verts[k]
            r1 = ring_verts[(k + 1) % 12]
            for s in range(6):
                sn = (s + 1) % 6
                bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

        # Zygomatic arch back to temporal bone
        zyg_pts = [
            Vector((side * 0.044, -0.052, 0.072)) * sc,
            Vector((side * 0.056, -0.024, 0.074)) * sc,
            Vector((side * 0.056, 0.006, 0.078)) * sc,
        ]
        zyg_rings: list[list[Vector]] = []
        for p in zyg_pts:
            ring = []
            for s in range(6):
                a = 2.0 * math.pi * s / 6
                ring.append(p + Vector((0.0035 * sc * math.cos(a), 0.0, 0.0048 * sc * math.sin(a))))
            zyg_rings.append(ring)
        _add_tube_loft(bm, zyg_rings, cap_start=True, cap_end=True)

    # 3. Nasal Bridge & Maxilla Upper Jaw Block
    max_stations = [
        (0.082, -0.068, 0.010, 0.007),
        (0.062, -0.070, 0.022, 0.012),
        (0.046, -0.068, 0.024, 0.015),
    ]
    max_rings: list[list[Vector]] = []
    for mz, my, mxr, myr in max_stations:
        ring = []
        for s in range(10):
            a = 2.0 * math.pi * s / 10
            ring.append(Vector((mxr * math.cos(a), my + myr * math.sin(a), mz)) * sc)
        max_rings.append(ring)
    _add_tube_loft(bm, max_rings, cap_start=True, cap_end=True)

    # 4. Mandible (Lower Jawbone): Left TMJ -> gonial angle -> chin -> right gonial -> right TMJ
    mand_pts = [
        (Vector((-0.046, -0.006, 0.062)) * sc, 0.0055 * sc),
        (Vector((-0.044, -0.016, 0.024)) * sc, 0.0080 * sc),
        (Vector((-0.028, -0.050, 0.016)) * sc, 0.0075 * sc),
        (Vector((0.0, -0.068, 0.014)) * sc, 0.0095 * sc),
        (Vector((0.028, -0.050, 0.016)) * sc, 0.0075 * sc),
        (Vector((0.044, -0.016, 0.024)) * sc, 0.0080 * sc),
        (Vector((0.046, -0.006, 0.062)) * sc, 0.0055 * sc),
    ]
    mand_rings: list[list[Vector]] = []
    for k, (p, mr) in enumerate(mand_pts):
        t = (mand_pts[min(k + 1, 6)][0] - mand_pts[max(k - 1, 0)][0]).normalized()
        up = Vector((0.0, 0.0, 1.0)) if abs(t.z) < 0.85 else Vector((0.0, -1.0, 0.0))
        b1 = t.cross(up).normalized()
        b2 = t.cross(b1).normalized()
        ring = []
        for s in range(8):
            a = 2.0 * math.pi * s / 8
            ring.append(p + b1 * (mr * 0.72 * math.cos(a)) + b2 * (mr * 1.15 * math.sin(a)))
        mand_rings.append(ring)
    _add_tube_loft(bm, mand_rings, cap_start=True, cap_end=True)

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_femur_mesh(
    name: str,
    side: str,
    nominal_len: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Anatomical Femur normalized along Z in `[0.0, 1.0]` (`0.0` = knee center, `1.0` = hip head).
    Scaling by `(1.0, 1.0, seg_len)` fits exact live knee-to-hip length with 0.000mm registration.
    """
    bm = bmesh.new()
    # In +X=Left space: Left hip socket is medial (-X from trochanter), Right hip socket is (+X)
    med_sign = -1.0 if side == "L" else 1.0
    lat_sign = -med_sign
    inv_z = 1.0 / max(0.25, nominal_len)

    # 1. Femoral shaft + neck + condyle flare (z in [0.02, 0.94])
    shaft_stations = [
        (0.000, 0.000, 0.002, 0.032, 0.020),   # Distal condyle base
        (0.050, 0.000, 0.002, 0.034, 0.022),   # Medial & lateral femoral condyles
        (0.120, lat_sign * 0.004, 0.000, 0.020, 0.017),
        (0.500, lat_sign * 0.020, -0.006, 0.015, 0.015),  # Mid-shaft gentle anterior bow
        (0.840, lat_sign * 0.034, 0.000, 0.017, 0.016),
        (0.935, lat_sign * 0.038, 0.002, 0.020, 0.018),   # Greater trochanter base
        (0.975, lat_sign * 0.018, 0.001, 0.014, 0.014),   # Femoral neck
        (1.000, 0.000, 0.000, 0.013, 0.013),              # Neck-head junction at hip_pos
    ]
    rings: list[list[Vector]] = []
    for sz, sx, sy, rx, ry in shaft_stations:
        ring = []
        for s in range(14):
            a = 2.0 * math.pi * s / 14
            ring.append(Vector((sx + rx * math.cos(a), sy + ry * math.sin(a), sz)))
        rings.append(ring)
    _add_tube_loft(bm, rings, cap_start=True, cap_end=True)

    # 2. Isotropic Femoral Head sphere at exact hip socket (0, 0, 1.0)
    head = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    for v in head["verts"]:
        v.co = Vector((
            v.co.x * 0.020,
            v.co.y * 0.020,
            v.co.z * (0.020 * inv_z) + 1.0,
        ))

    # 3. Greater Trochanter prominence
    troch = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    for v in troch["verts"]:
        v.co = Vector((
            v.co.x * 0.015 + lat_sign * 0.042,
            v.co.y * 0.016 + 0.004,
            v.co.z * (0.018 * inv_z) + 0.945,
        ))

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_patella_mesh(
    name: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    for v in bm.verts:
        v.co = Vector((
            v.co.x * 0.017,
            v.co.y * 0.008 - 0.032,
            v.co.z * 0.019,
        ))
    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_tibia_fibula_mesh(
    name: str,
    side: str,
    nominal_len: float,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Anatomical Tibia + Fibula pair normalized along Z in `[0.0, 1.0]` (`0.0` = ankle, `1.0` = knee).
    """
    bm = bmesh.new()
    med_sign = -1.0 if side == "L" else 1.0
    lat_sign = -med_sign
    inv_z = 1.0 / max(0.25, nominal_len)

    # 1. Tibia (thick medial shin bone with flared tibial plateau & medial malleolus)
    tib_stations = [
        (0.000, med_sign * 0.006, 0.000, 0.019, 0.016),   # Ankle mortise / medial malleolus
        (0.080, med_sign * 0.005, -0.002, 0.014, 0.013),
        (0.500, med_sign * 0.004, -0.005, 0.014, 0.015),  # Sharp anterior tibial crest
        (0.880, med_sign * 0.003, -0.003, 0.019, 0.017),
        (0.960, 0.000, 0.000, 0.033, 0.022),              # Tibial plateau
        (1.000, 0.000, 0.000, 0.031, 0.020),
    ]
    tib_rings: list[list[Vector]] = []
    for sz, sx, sy, rx, ry in tib_stations:
        ring = []
        for s in range(12):
            a = 2.0 * math.pi * s / 12
            # Subtle triangular shin ridge toward -Y
            ridge = 1.22 if math.sin(a) < -0.5 else 1.0
            ring.append(Vector((sx + rx * math.cos(a), sy + ry * math.sin(a) * ridge, sz)))
        tib_rings.append(ring)
    _add_tube_loft(bm, tib_rings, cap_start=True, cap_end=True)

    # 2. Fibula (slender lateral strut running parallel from knee head down to lateral malleolus)
    fib_stations = [
        (-0.015 * inv_z, lat_sign * 0.024, 0.006, 0.0085, 0.0085),  # Lateral malleolus
        (0.080, lat_sign * 0.024, 0.008, 0.0062, 0.0062),
        (0.500, lat_sign * 0.024, 0.010, 0.0058, 0.0058),
        (0.900, lat_sign * 0.024, 0.009, 0.0072, 0.0072),
        (0.945, lat_sign * 0.025, 0.008, 0.0095, 0.0095),           # Fibular head
    ]
    fib_rings: list[list[Vector]] = []
    for sz, sx, sy, rx, ry in fib_stations:
        ring = []
        for s in range(8):
            a = 2.0 * math.pi * s / 8
            ring.append(Vector((sx + rx * math.cos(a), sy + ry * math.sin(a), sz)))
        fib_rings.append(ring)
    _add_tube_loft(bm, fib_rings, cap_start=True, cap_end=True)

    return _make_mesh_obj(name, bm, mat, col, parent)


def _build_foot_mesh(
    name: str,
    side: str,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """
    Talus + Calcaneus (Heel) + 5 Metatarsal Rays anchored at ankle joint `(0,0,0)`.
    """
    bm = bmesh.new()
    med_sign = -1.0 if side == "L" else 1.0

    # Talus dome at ankle
    tal = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    for v in tal["verts"]:
        v.co = Vector((v.co.x * 0.018, v.co.y * 0.022, v.co.z * 0.015 - 0.012))

    # Calcaneus heel bone extending posteriorly (+Y) & downward (-Z)
    calc_stations = [
        (0.046, -0.036, 0.013, 0.014),
        (0.016, -0.032, 0.015, 0.015),
        (-0.016, -0.024, 0.017, 0.014),
    ]
    calc_rings: list[list[Vector]] = []
    for cy, cz, rx, rz in calc_stations:
        ring = []
        for s in range(8):
            a = 2.0 * math.pi * s / 8
            ring.append(Vector((rx * math.cos(a), cy, cz + rz * math.sin(a))))
        calc_rings.append(ring)
    _add_tube_loft(bm, calc_rings, cap_start=True, cap_end=True)

    # 5 Metatarsal + Toe Rays fanning forward (-Y)
    for idx in range(5):
        frac = (idx - 2.0) / 2.0  # -1 (big toe medial) .. +1 (pinky lateral)
        x_end = med_sign * (-frac * 0.026)
        x_start = med_sign * (-frac * 0.016)
        len_y = -0.145 + abs(frac + 0.5) * 0.018
        ray_stations = [
            (x_start, -0.020, -0.024, 0.0052 if idx == 0 else 0.0038),
            ((x_start + x_end) * 0.5, len_y * 0.55, -0.044, 0.0046 if idx == 0 else 0.0034),
            (x_end, len_y, -0.052, 0.0050 if idx == 0 else 0.0032),
        ]
        ray_rings: list[list[Vector]] = []
        for rx_p, ry_p, rz_p, rr in ray_stations:
            ring = []
            for s in range(6):
                a = 2.0 * math.pi * s / 6
                ring.append(Vector((rx_p + rr * math.cos(a), ry_p, rz_p + rr * math.sin(a))))
            ray_rings.append(ring)
        _add_tube_loft(bm, ray_rings, cap_start=True, cap_end=True)

    return _make_mesh_obj(name, bm, mat, col, parent)


# ---------------------------------------------------------------------------
# RIG EVALUATION & 0.000 MM BONE-TRACKING MATH
# ---------------------------------------------------------------------------
def _bone_local_character_dirs(
    rig: bpy.types.Object,
    bone_name: str,
) -> tuple[Vector, Vector]:
    """Return rest character-forward (-Y) and character-left (+X) in bone-local coordinates."""
    m_rest_inv = rig.data.bones[bone_name].matrix_local.to_3x3().inverted()
    fwd_local = (m_rest_inv @ Vector((0.0, -1.0, 0.0))).normalized()
    left_local = (m_rest_inv @ Vector((1.0, 0.0, 0.0))).normalized()
    return (fwd_local, left_local)


def _frame_from_up_back(up_axis: Vector, back_hint: Vector, left_hint: Vector | None = None) -> Quaternion:
    """
    Build right-handed quaternion aligning local `+Z = up_axis`, `+Y ~ back_hint` (posterior),
    `+X = Y x Z` (anatomical Left).
    """
    z_ax = up_axis.normalized()
    back_proj = back_hint - back_hint.dot(z_ax) * z_ax
    if back_proj.length < 1e-5 and left_hint is not None:
        left_proj = left_hint - left_hint.dot(z_ax) * z_ax
        if left_proj.length >= 1e-5:
            x_ax = left_proj.normalized()
            y_ax = z_ax.cross(x_ax).normalized()
            return Matrix((x_ax, y_ax, z_ax)).transposed().to_quaternion()
    if back_proj.length < 1e-5:
        ref = Vector((0.0, 1.0, 0.0)) if abs(z_ax.y) < 0.9 else Vector((1.0, 0.0, 0.0))
        back_proj = ref - ref.dot(z_ax) * z_ax
    y_ax = back_proj.normalized()
    x_ax = y_ax.cross(z_ax).normalized()
    y_ax = z_ax.cross(x_ax).normalized()
    return Matrix((x_ax, y_ax, z_ax)).transposed().to_quaternion()


def _catmull_rom(pts: list[Vector], cum_u: list[float], u: float) -> Vector:
    """Evaluate smooth clamped Catmull-Rom spline at normalized arc fraction `u in [0, 1]`."""
    n = len(pts)
    if n == 1:
        return pts[0].copy()
    if n == 2:
        return pts[0].lerp(pts[1], C.clamp(u, 0.0, 1.0))
    u_c = C.clamp(u, 0.0, 1.0)
    # Find segment i such that cum_u[i] <= u_c <= cum_u[i+1]
    seg = 0
    for i in range(n - 1):
        if u_c <= cum_u[i + 1] or i == n - 2:
            seg = i
            break
    span = max(1e-6, cum_u[seg + 1] - cum_u[seg])
    t = C.clamp((u_c - cum_u[seg]) / span, 0.0, 1.0)

    p1 = pts[seg]
    p2 = pts[seg + 1]
    p0 = pts[seg - 1] if seg > 0 else (p1 - (p2 - p1))
    p3 = pts[seg + 2] if seg + 2 < n else (p2 + (p2 - p1))

    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _lerp_vec_chain(vecs: list[Vector], cum_u: list[float], u: float) -> Vector:
    n = len(vecs)
    if n == 1:
        return vecs[0].normalized()
    u_c = C.clamp(u, 0.0, 1.0)
    for i in range(n - 1):
        if u_c <= cum_u[i + 1] or i == n - 2:
            span = max(1e-6, cum_u[i + 1] - cum_u[i])
            t = C.clamp((u_c - cum_u[i]) / span, 0.0, 1.0)
            return vecs[i].lerp(vecs[i + 1], t).normalized()
    return vecs[-1].normalized()


def _fallback_pose_sample(f: float) -> dict[str, Any]:
    """Analytical Beat 4 pose fallback if `rig` is None or lacks expected bones."""
    sx, sy = C.PASSENGER["stand_xy"]
    yaw = math.radians(float(C.PASSENGER["stand_yaw_deg"]))
    # +X=Left, -Y=Forward, +Z=Up rotated by stand_yaw_deg
    r_z = Matrix.Rotation(yaw, 3, "Z")
    fwd = (r_z @ Vector((0.0, -1.0, 0.0))).normalized()
    left = (r_z @ Vector((1.0, 0.0, 0.0))).normalized()
    back = -fwd

    # Subtle knee buckle & axial spinal compression across Beat 4 pit impact
    buckle = 0.0
    if 940.0 <= f <= 1040.0:
        buckle = math.sin(C.smoothstep(940.0, 990.0, f) * math.pi * 0.5) * (
            1.0 - 0.55 * C.smoothstep(990.0, 1080.0, f)
        )
    elif 1040.0 < f <= 1220.0:
        buckle = 0.45 * (1.0 - C.smoothstep(1040.0, 1220.0, f))

    comp_m = C.spine_impact_comp_cm(f) * 0.01
    pelvis_drop = buckle * 0.08 + comp_m * 0.25

    base = Vector((sx, sy, C.passenger_jump_offset_z(f)))
    hip_half = 0.095
    hip_z = float(C.PASSENGER["pelvis_z"]) - pelvis_drop
    knee_z = 0.48 - buckle * 0.03
    ankle_z = float(C.PASSENGER["ankle_z"])

    spine_pts = [
        base + Vector((0.0, 0.0, hip_z + 0.04)) + back * 0.015,
        base + Vector((0.0, 0.0, hip_z + 0.14)) + fwd * (0.01 * buckle),
        base + Vector((0.0, 0.0, hip_z + 0.30 - comp_m * 0.35)) + fwd * (0.025 * buckle),
        base + Vector((0.0, 0.0, hip_z + 0.45 - comp_m * 0.65)) + fwd * (0.035 * buckle),
        base + Vector((0.0, 0.0, hip_z + 0.54 - comp_m * 0.85)) + fwd * (0.038 * buckle),
        base + Vector((0.0, 0.0, hip_z + 0.62 - comp_m)) + fwd * (0.035 * buckle),
    ]
    head_pos = spine_pts[-1]
    head_tail = head_pos + Vector((0.0, 0.0, 0.21)) + fwd * (0.02 * buckle)

    out: dict[str, Any] = {
        "spine_pts": spine_pts,
        "spine_fwds": [fwd] * len(spine_pts),
        "spine_lefts": [left] * len(spine_pts),
        "head_pos": head_pos,
        "head_tail": head_tail,
        "head_fwd": fwd,
        "head_left": left,
        "ribs_pos": spine_pts[2],
        "ribs_fwd": fwd,
        "ribs_left": left,
        "hips": {},
        "knees": {},
        "ankles": {},
        "leg_fwds": {},
        "leg_lefts": {},
        "foot_quats": {},
    }
    for side, s_sign in (("L", 1.0), ("R", -1.0)):
        hip_p = base + left * (s_sign * hip_half) + Vector((0.0, 0.0, hip_z))
        knee_p = base + left * (s_sign * hip_half * 1.02) + fwd * (0.09 * buckle) + Vector((0.0, 0.0, knee_z))
        ank_p = base + left * (s_sign * hip_half * 1.05) + Vector((0.0, 0.0, ankle_z))
        out["hips"][side] = hip_p
        out["knees"][side] = knee_p
        out["ankles"][side] = ank_p
        out["leg_fwds"][side] = fwd
        out["leg_lefts"][side] = left
        out["foot_quats"][side] = _frame_from_up_back(Vector((0.0, 0.0, 1.0)), back, left)
    return out


def _sample_rig_pose(
    rig: bpy.types.Object | None,
    carrier: bpy.types.Object,
    bone_dirs: dict[str, tuple[Vector, Vector]],
    f: float,
    depsgraph: Any,
) -> dict[str, Any]:
    """Evaluate exact `carrier`-local joint coordinates & orientations from `Passenger_Rig`."""
    if rig is None or "spine05" not in rig.pose.bones or "head" not in rig.pose.bones:
        return _fallback_pose_sample(f)

    rig_eval = rig.evaluated_get(depsgraph) if depsgraph is not None else rig
    car_eval = carrier.evaluated_get(depsgraph) if depsgraph is not None else carrier
    car_inv = car_eval.matrix_world.inverted()
    rig_to_car = car_inv @ rig_eval.matrix_world

    def bone_mat(bname: str) -> Matrix:
        return rig_to_car @ rig_eval.pose.bones[bname].matrix

    def bone_head(bname: str) -> Vector:
        return (rig_to_car @ rig_eval.pose.bones[bname].head).copy()

    def bone_tail(bname: str) -> Vector:
        return (rig_to_car @ rig_eval.pose.bones[bname].tail).copy()

    def bone_axes(bname: str) -> tuple[Vector, Vector]:
        m3 = bone_mat(bname).to_3x3()
        fwd_l, left_l = bone_dirs[bname]
        return ((m3 @ fwd_l).normalized(), (m3 @ left_l).normalized())

    spine_pts: list[Vector] = []
    spine_fwds: list[Vector] = []
    spine_lefts: list[Vector] = []
    for bname in SPINE_BONE_CANDIDATES:
        if bname in rig_eval.pose.bones and bname in bone_dirs:
            spine_pts.append(bone_head(bname))
            fw, lf = bone_axes(bname)
            spine_fwds.append(fw)
            spine_lefts.append(lf)

    head_pos = bone_head("head")
    head_tail = bone_tail("head")
    head_fwd, head_left = bone_axes("head")

    rib_b = "spine02" if "spine02" in rig_eval.pose.bones else (
        "spine03" if "spine03" in rig_eval.pose.bones else "spine05"
    )
    ribs_pos = bone_head(rib_b)
    ribs_fwd, ribs_left = bone_axes(rib_b)

    hips: dict[str, Vector] = {}
    knees: dict[str, Vector] = {}
    ankles: dict[str, Vector] = {}
    leg_fwds: dict[str, Vector] = {}
    leg_lefts: dict[str, Vector] = {}
    foot_quats: dict[str, Quaternion] = {}

    fb = None
    for side in ("L", "R"):
        ul_b = f"upperleg01.{side}"
        ll_b = f"lowerleg01.{side}"
        ll2_b = f"lowerleg02.{side}"
        ft_b = f"foot.{side}"

        if ul_b in rig_eval.pose.bones and ll_b in rig_eval.pose.bones:
            hips[side] = bone_head(ul_b)
            knees[side] = bone_head(ll_b)
            if ft_b in rig_eval.pose.bones:
                ankles[side] = bone_head(ft_b)
                f_fwd, f_left = bone_axes(ft_b)
                f_up = (bone_head(ll_b) - bone_head(ft_b))
                # Foot rests horizontal unless lifted; derive clean foot frame from foot bone axes
                foot_up_axis = f_left.cross(-f_fwd).normalized()
                foot_quats[side] = _frame_from_up_back(foot_up_axis, -f_fwd, f_left)
            elif ll2_b in rig_eval.pose.bones:
                ankles[side] = bone_tail(ll2_b)
                f_fwd, f_left = bone_axes(ll2_b)
                foot_quats[side] = _frame_from_up_back(Vector((0.0, 0.0, 1.0)), -f_fwd, f_left)
            else:
                ankles[side] = bone_tail(ll_b)
                f_fwd, f_left = bone_axes(ll_b)
                foot_quats[side] = _frame_from_up_back(Vector((0.0, 0.0, 1.0)), -f_fwd, f_left)

            u_fwd, u_left = bone_axes(ul_b)
            l_fwd, l_left = bone_axes(ll_b)
            leg_fwds[f"{side}_U"] = u_fwd
            leg_lefts[f"{side}_U"] = u_left
            leg_fwds[f"{side}_L"] = l_fwd
            leg_lefts[f"{side}_L"] = l_left
        else:
            if fb is None:
                fb = _fallback_pose_sample(f)
            hips[side] = fb["hips"][side]
            knees[side] = fb["knees"][side]
            ankles[side] = fb["ankles"][side]
            leg_fwds[f"{side}_U"] = fb["leg_fwds"][side]
            leg_lefts[f"{side}_U"] = fb["leg_lefts"][side]
            leg_fwds[f"{side}_L"] = fb["leg_fwds"][side]
            leg_lefts[f"{side}_L"] = fb["leg_lefts"][side]
            foot_quats[side] = fb["foot_quats"][side]

    return {
        "spine_pts": spine_pts,
        "spine_fwds": spine_fwds,
        "spine_lefts": spine_lefts,
        "head_pos": head_pos,
        "head_tail": head_tail,
        "head_fwd": head_fwd,
        "head_left": head_left,
        "ribs_pos": ribs_pos,
        "ribs_fwd": ribs_fwd,
        "ribs_left": ribs_left,
        "hips": hips,
        "knees": knees,
        "ankles": ankles,
        "leg_fwds": leg_fwds,
        "leg_lefts": leg_lefts,
        "foot_quats": foot_quats,
    }


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def build_skeleton_xray(
    col: bpy.types.Collection,
    carrier: bpy.types.Object,
    rig: bpy.types.Object | None,
    total_frames: int = 1800,
) -> dict[str, Any]:
    """
    Build & animate the full-body anatomical X-ray skeleton locked inside `Passenger_Body`.
    Evaluates `Passenger_Rig` pose bones across Beat 4 (`f=900..1240`, `step=2`) in
    `carrier`-local space for 0.000mm registration with zero clipping.
    """
    scene = bpy.context.scene
    orig_frame = scene.frame_current

    # Precompute rest bone character-axis mappings if rig is present
    bone_dirs: dict[str, tuple[Vector, Vector]] = {}
    if rig is not None:
        all_candidates = list(SPINE_BONE_CANDIDATES)
        for side in ("L", "R"):
            all_candidates.extend([
                f"upperleg01.{side}", f"lowerleg01.{side}",
                f"lowerleg02.{side}", f"foot.{side}",
            ])
        for bname in all_candidates:
            if bname in rig.data.bones and bname in rig.pose.bones:
                bone_dirs[bname] = _bone_local_character_dirs(rig, bname)

    # Sample rest calibration frame (f=905) to measure true passenger proportions
    scene.frame_set(905)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rest_pose = _sample_rig_pose(rig, carrier, bone_dirs, 905.0, depsgraph)

    half_hip = max(0.075, 0.5 * (rest_pose["hips"]["L"] - rest_pose["hips"]["R"]).length)
    hip_mid_rest = 0.5 * (rest_pose["hips"]["L"] + rest_pose["hips"]["R"])
    sacrum_top_z = max(0.045, rest_pose["spine_pts"][0].z - hip_mid_rest.z)
    head_len = max(0.16, (rest_pose["head_tail"] - rest_pose["head_pos"]).length)
    femur_nom_len = {
        side: max(0.28, (rest_pose["hips"][side] - rest_pose["knees"][side]).length)
        for side in ("L", "R")
    }
    tibia_nom_len = {
        side: max(0.28, (rest_pose["knees"][side] - rest_pose["ankles"][side]).length)
        for side in ("L", "R")
    }

    # Build root Empty parented to `carrier`
    skel_root = _create_empty(f"{PREFIX}Root", col, parent=carrier)
    skel_root.rotation_mode = "QUATERNION"

    # Build 100% Local-Space (`TexCoord.Object = carrier`) X-ray shaders
    mat_bone = _create_xray_bone_material(f"{PREFIX}Mat_Bone", carrier, is_disc=False)
    mat_disc = _create_xray_bone_material(f"{PREFIX}Mat_Disc", carrier, is_disc=True)

    # Anatomical normalized height fractions `u_i` for the 24 vertebrae (`L5..C1`)
    vert_heights = [sp[4] for sp in VERTEBRA_SPECS]
    vert_gaps = [sp[5] for sp in VERTEBRA_SPECS]
    total_nom_spine = sum(vert_heights) + sum(vert_gaps)
    vert_u: list[float] = []
    cur_z = vert_gaps[0] + vert_heights[0] * 0.5
    for idx in range(len(VERTEBRA_SPECS)):
        vert_u.append(cur_z / total_nom_spine)
        if idx + 1 < len(VERTEBRA_SPECS):
            cur_z += vert_heights[idx] * 0.5 + vert_gaps[idx + 1] + vert_heights[idx + 1] * 0.5

    # 1. Build 24 Vertebrae (`L5..C1`)
    vertebrae: list[bpy.types.Object] = []
    for spec in VERTEBRA_SPECS:
        v_obj = _build_vertebra_mesh(f"{PREFIX}Vert_{spec[0]}", spec, mat_bone, col, skel_root)
        vertebrae.append(v_obj)

    # 2. Build 24 Intervertebral Discs (`S1-L5` + `L5-L4`..`C2-C1`)
    discs: list[bpy.types.Object] = []
    rest_disc_gaps: list[float] = []
    for idx in range(len(VERTEBRA_SPECS)):
        if idx == 0:
            d_name = f"{PREFIX}Disc_00_S1L5"
            rx, ry, d_gap = VERTEBRA_SPECS[0][2] * 0.96, VERTEBRA_SPECS[0][3] * 0.95, vert_gaps[0]
        else:
            prev_lbl = VERTEBRA_SPECS[idx - 1][0]
            cur_lbl = VERTEBRA_SPECS[idx][0]
            d_name = f"{PREFIX}Disc_{idx:02d}_{prev_lbl}{cur_lbl}"
            rx = 0.5 * (VERTEBRA_SPECS[idx - 1][2] + VERTEBRA_SPECS[idx][2]) * 0.95
            ry = 0.5 * (VERTEBRA_SPECS[idx - 1][3] + VERTEBRA_SPECS[idx][3]) * 0.94
            d_gap = vert_gaps[idx]
        # Slightly tuck disc ends into adjacent endplates so squash/curve never opens gaps
        d_obj = _build_disc_mesh(d_name, rx, ry, d_gap + 0.0042, mat_disc, col, skel_root)
        discs.append(d_obj)
        rest_disc_gaps.append(d_gap)

    # 3. Pelvis & Sacrum
    pelvis = _build_pelvis_mesh(f"{PREFIX}Pelvis", half_hip, sacrum_top_z, mat_bone, col, skel_root)

    # 4. Ribcage + Clavicles
    ribcage = _build_ribcage_mesh(f"{PREFIX}Ribcage", mat_bone, col, skel_root)

    # 5. Skull Cranium + Mandible
    skull = _build_skull_mesh(f"{PREFIX}Skull", head_len, mat_bone, col, skel_root)

    # 6. Lower Extremity Bones (Femurs, Patellas, Tibias/Fibulas, Feet)
    femurs: dict[str, bpy.types.Object] = {}
    patellas: dict[str, bpy.types.Object] = {}
    tibias: dict[str, bpy.types.Object] = {}
    feet: dict[str, bpy.types.Object] = {}
    for side in ("L", "R"):
        femurs[side] = _build_femur_mesh(
            f"{PREFIX}Femur_{side}", side, femur_nom_len[side], mat_bone, col, skel_root,
        )
        patellas[side] = _build_patella_mesh(
            f"{PREFIX}Patella_{side}", mat_bone, col, skel_root,
        )
        tibias[side] = _build_tibia_fibula_mesh(
            f"{PREFIX}TibiaFibula_{side}", side, tibia_nom_len[side], mat_bone, col, skel_root,
        )
        feet[side] = _build_foot_mesh(
            f"{PREFIX}Foot_{side}", side, mat_bone, col, skel_root,
        )

    leg_bones: list[bpy.types.Object] = [
        femurs["L"], femurs["R"],
        patellas["L"], patellas["R"],
        tibias["L"], tibias["R"],
        feet["L"], feet["R"],
    ]

    all_objs: list[bpy.types.Object] = [
        skel_root, pelvis, ribcage, skull,
        *vertebrae, *discs, *leg_bones,
    ]

    # Collect sorted sample frames across Beat 4 (`f=900..1240`, `step=2` + milestones)
    sample_frames = sorted(
        {f for f in range(EVAL_START, min(EVAL_END, total_frames) + 1, EVAL_STEP)}
        | {f for f in MILESTONE_FRAMES if EVAL_START <= f <= min(EVAL_END, total_frames)}
    )

    prev_quats: dict[str, Quaternion] = {}

    def _apply_xform(
        obj: bpy.types.Object,
        loc: Vector,
        quat: Quaternion,
        frame: int,
        scale: Vector | None = None,
    ) -> None:
        q = quat.copy()
        pq = prev_quats.get(obj.name)
        if pq is not None and q.dot(pq) < 0.0:
            q = -q
        prev_quats[obj.name] = q
        obj.location = loc
        obj.rotation_quaternion = q
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_quaternion", frame=frame)
        if scale is not None:
            obj.scale = scale
            obj.keyframe_insert("scale", frame=frame)

    # Rest spine arc length & vertebra-to-vertebra gaps measured at f=900
    rest_vert_pos: list[Vector] | None = None
    rest_s0_pos: Vector | None = None

    for f in sample_frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pose = _sample_rig_pose(rig, carrier, bone_dirs, float(f), depsgraph)

        # Build cumulative arc-length parameterization along evaluated spine joints
        s_pts = pose["spine_pts"]
        cum_len = [0.0]
        for k in range(len(s_pts) - 1):
            cum_len.append(cum_len[-1] + (s_pts[k + 1] - s_pts[k]).length)
        tot_len = max(1e-4, cum_len[-1])
        cum_u = [cl / tot_len for cl in cum_len]

        # Evaluate 24 vertebra positions & orientations along spine spline
        cur_vert_pos: list[Vector] = []
        cur_vert_quats: list[Quaternion] = []
        for idx, u_v in enumerate(vert_u):
            p_v = _catmull_rom(s_pts, cum_u, u_v)
            p_dn = _catmull_rom(s_pts, cum_u, max(0.0, u_v - 0.015))
            p_up = _catmull_rom(s_pts, cum_u, min(1.0, u_v + 0.015))
            tang = (p_up - p_dn)
            if tang.length < 1e-6:
                tang = Vector((0.0, 0.0, 1.0))
            fwd_v = _lerp_vec_chain(pose["spine_fwds"], cum_u, u_v)
            left_v = _lerp_vec_chain(pose["spine_lefts"], cum_u, u_v)
            q_v = _frame_from_up_back(tang, -fwd_v, left_v)
            cur_vert_pos.append(p_v)
            cur_vert_quats.append(q_v)
            _apply_xform(vertebrae[idx], p_v, q_v, f)

        if rest_vert_pos is None:
            rest_vert_pos = [p.copy() for p in cur_vert_pos]
            rest_s0_pos = s_pts[0].copy()

        # Normalized 28 G compression factor (0..1) from contract
        comp_frac = C.clamp(C.spine_impact_comp_cm(float(f)) / 3.5, 0.0, 1.0)

        # Place & squash/bulge all 24 intervertebral discs seamlessly between adjacent bones
        for idx, d_obj in enumerate(discs):
            if idx == 0:
                p_top = cur_vert_pos[0]
                p_bot = s_pts[0]
                q_d = cur_vert_quats[0]
                rest_gap = max(0.004, (rest_vert_pos[0] - rest_s0_pos).length)
            else:
                p_top = cur_vert_pos[idx]
                p_bot = cur_vert_pos[idx - 1]
                q_d = cur_vert_quats[idx - 1].slerp(cur_vert_quats[idx], 0.5)
                rest_gap = max(0.004, (rest_vert_pos[idx] - rest_vert_pos[idx - 1]).length)

            mid_p = (p_bot + p_top) * 0.5
            cur_gap = max(0.002, (p_top - p_bot).length)
            gap_ratio = C.clamp(cur_gap / rest_gap, 0.55, 1.15)
            # Lumbar discs absorb strongest axial 28 G compression
            region_w = 1.0 if idx <= 5 else (0.72 if idx <= 17 else 0.48)
            sz = C.clamp(gap_ratio * (1.0 - 0.36 * comp_frac * region_w), 0.48, 1.05)
            sxy = C.clamp(1.0 / math.sqrt(sz), 1.0, 1.34)
            _apply_xform(d_obj, mid_p, q_d, f, scale=Vector((sxy, sxy, sz)))

        # Pelvis & Sacrum anchored by live hip sockets + spine05 base
        hip_l = pose["hips"]["L"]
        hip_r = pose["hips"]["R"]
        hip_mid = 0.5 * (hip_l + hip_r)
        pelv_left = (hip_l - hip_r)
        pelv_up = (s_pts[0] - hip_mid)
        if pelv_up.length < 1e-4:
            pelv_up = Vector((0.0, 0.0, 1.0))
        pelv_fwd = _lerp_vec_chain(pose["spine_fwds"], cum_u, 0.0)
        pelv_q = _frame_from_up_back(pelv_up, -pelv_fwd, pelv_left)
        live_half_hip = max(0.06, 0.5 * pelv_left.length)
        pelv_sc = Vector((live_half_hip / half_hip, 1.0, 1.0))
        _apply_xform(pelvis, hip_mid, pelv_q, f, scale=pelv_sc)

        # Ribcage anchored to thoracic spine midpoint
        rib_q = _frame_from_up_back(
            (cur_vert_pos[14] - cur_vert_pos[7]) if len(cur_vert_pos) > 14 else Vector((0.0, 0.0, 1.0)),
            -pose["ribs_fwd"],
            pose["ribs_left"],
        )
        _apply_xform(ribcage, pose["ribs_pos"], rib_q, f)

        # Skull anchored to head bone head->tail
        head_vec = pose["head_tail"] - pose["head_pos"]
        if head_vec.length < 1e-4:
            head_vec = Vector((0.0, 0.0, 0.21))
        head_q = _frame_from_up_back(head_vec, -pose["head_fwd"], pose["head_left"])
        _apply_xform(skull, pose["head_pos"], head_q, f)

        # Left & Right Femurs, Patellas, Tibias/Fibulas, Feet
        for side in ("L", "R"):
            hip_p = pose["hips"][side]
            knee_p = pose["knees"][side]
            ank_p = pose["ankles"][side]

            fem_vec = hip_p - knee_p
            fem_len = max(0.20, fem_vec.length)
            u_fwd = pose["leg_fwds"].get(f"{side}_U", pose["leg_fwds"].get(side, Vector((0.0, -1.0, 0.0))))
            u_left = pose["leg_lefts"].get(f"{side}_U", pose["leg_lefts"].get(side, Vector((1.0, 0.0, 0.0))))
            fem_q = _frame_from_up_back(fem_vec, -u_fwd, u_left)
            _apply_xform(femurs[side], knee_p, fem_q, f, scale=Vector((1.0, 1.0, fem_len)))

            tib_vec = knee_p - ank_p
            tib_len = max(0.20, tib_vec.length)
            l_fwd = pose["leg_fwds"].get(f"{side}_L", pose["leg_fwds"].get(side, Vector((0.0, -1.0, 0.0))))
            l_left = pose["leg_lefts"].get(f"{side}_L", pose["leg_lefts"].get(side, Vector((1.0, 0.0, 0.0))))
            tib_q = _frame_from_up_back(tib_vec, -l_fwd, l_left)
            _apply_xform(tibias[side], ank_p, tib_q, f, scale=Vector((1.0, 1.0, tib_len)))

            # Patella blends femur & tibia orientations cleanly at anterior knee
            pat_q = fem_q.slerp(tib_q, 0.5)
            _apply_xform(patellas[side], knee_p, pat_q, f)

            # Foot / Ankle cluster
            _apply_xform(feet[side], ank_p, pose["foot_quats"][side], f)

    # Clean CONSTANT visibility (`True` strictly during f=905..1238, `False` everywhere else)
    for obj in all_objs:
        _set_fcurve_linear_or_bezier(obj, interp="LINEAR")
        _set_constant_visibility(obj, SKEL_VIS_RANGE[0], SKEL_VIS_RANGE[1])

    scene.frame_set(orig_frame)
    return {
        "root": skel_root,
        "vertebrae": vertebrae,
        "discs": discs,
        "leg_bones": leg_bones,
        "pelvis": pelvis,
        "ribcage": ribcage,
        "skull": skull,
        "all_objs": all_objs,
    }
