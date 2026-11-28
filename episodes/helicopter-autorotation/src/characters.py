"""
characters.py — Enclosed 1:1 Rescue Helicopter Cockpit, Live Physical 3D ENG/ROTOR RPM Gauges &
Radar Altimeter HUD, Matte Nomex Flight-Suit Aviators with 4-Point Harnesses, Zero-Crossover Controls,
and Expressive Panic-to-Focus-to-Relief Facial Acting for Episode #5 V2 Ultimate (60 fps, 1800 frames).

Strictly active ONLY during interior 1:1 cockpit scenes:
  - Beat 1B (`116..240`): Dark enclosed emergency cockpit bathed in pulsing red `ENG OUT` glare!
    * Physical 3D Dual-Tachometer Gauge shows crimson `ENG` needle plunging to `0%` while emerald
      `ROTOR` needle recovers into the green arc as Pilot slams the collective pitch lever straight down!
    * Deep panic/urgency facial acting: furrowed brows, wide darting eye saccades, tense gasping jaw,
      cabin shudder, and Co-Pilot urgently pointing downward toward the rooftop helipad!
  - Beat 5A (`1291..1450`): Label-free shallow-DOF (`f/3.6`) final flare close-up!
    * Glowing radar altimeter bar counts down (`125 -> 32 FT`) as gritty focused Pilot hauls the
      collective lever strongly UP (`f=1335..1418`), transitioning into a genuine relieved exhale
      & half-smile (`f=1418..1450`) as Co-Pilot nods in relief!

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
  - Clean matte dark-navy Nomex flight-suit cloth shader overrides any default patterned garment texture!
  - `eye_obj.visible_shadow = False`, `lash_obj.visible_shadow = False`, independent `EnhancedEye`
    node tree per actor, relaxed `-5.0 deg` upper eyelids, jaw clamped `<= 14.0 deg`.
  - `UnderClothes_Shrink` (`-0.011` DISPLACE BEFORE `ARMATURE`) including `upperarm` & `lowerarm`.
  - Materials appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)` and `C.iter_action_fcurves`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_ROOT_SRC = _REPO_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_ROOT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heli_contract as C  # noqa: E402
import mpfb_characters as mc  # noqa: E402

_HERO_BLINK_EXCL: tuple[int, ...] = (116, 130, 152, 185, 225, 1291, 1345, 1410, 1450)
_COCKPIT_WINDOWS: list[tuple[int, int]] = [(C.B1_COCKPIT_START, 240), (1291, C.B5_COCKPIT_END)]


def _link(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _key_visibility(objs: list[bpy.types.Object], windows: list[tuple[int, int]], total_frames: int) -> None:
    def _vis(f: int) -> bool:
        return any(a <= f <= b for a, b in windows)

    frames = {1, total_frames}
    for a, b in windows:
        for f in (a - 1, a, b, b + 1):
            if 1 <= f <= total_frames:
                frames.add(f)

    for obj in objs:
        for f in sorted(frames):
            h = not _vis(f)
            obj.hide_render = h
            obj.hide_viewport = h
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)
        ad = obj.animation_data
        if ad and ad.action:
            for fc in C.iter_action_fcurves(ad.action):
                if fc.data_path.startswith("hide_"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"


def _make_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.45,
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
        bsdf.inputs["Emission Strength"].default_value = min(1.90, emit_str)
    return mat


def _make_nomex_flight_suit_mat(name: str, rgb: tuple[float, float, float]) -> bpy.types.Material:
    """Authentic matte dark-navy Nomex rescue flight-suit cloth shader (ZERO shiny medieval lace!)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.88
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.16
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.25

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 95.0
    noise.inputs["Detail"].default_value = 4.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_cockpit_glass_glare_mat() -> bpy.types.Material:
    """Ultra-clean grazing-angle cockpit windshield reflection glass (never blocks faces; zero shadow!)."""
    mat = bpy.data.materials.new("Mat_Cockpit_CanopyGlare")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.18
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.10
    nt.links.new(lw.outputs["Fresnel"], mul.inputs[0])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul_cam = nt.nodes.new("ShaderNodeMath")
    mul_cam.operation = "MULTIPLY"
    nt.links.new(mul.outputs["Value"], mul_cam.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mul_cam.inputs[1])

    glossy = nt.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (0.78, 0.88, 1.00, 1.0)
    glossy.inputs["Roughness"].default_value = 0.04
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul_cam.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(glossy.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mats: list[bpy.types.Material],
    matrix: Matrix,
    col: bpy.types.Collection,
    smooth: bool = True,
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


def _add_box_bm(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_idx: int = 0,
    bevel_w: float = 0.004,
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


def _add_strut_bm(
    bm: bmesh.types.BMesh,
    p0: Vector,
    p1: Vector,
    radius: float,
    mat_idx: int = 0,
    segs: int = 10,
) -> None:
    seg = p1 - p0
    length = seg.length
    if length < 1e-5:
        return
    mid = 0.5 * (p0 + p1)
    rm = seg.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs,
        radius1=radius, radius2=radius, depth=length,
        matrix=Matrix.Translation(mid) @ rm,
    )
    vs = set(res["verts"])
    for f in bm.faces:
        if all(v in vs for v in f.verts):
            f.material_index = mat_idx


def _apply_underclothes_shrink(body: bpy.types.Object, rig: bpy.types.Object) -> None:
    """Apply UnderClothes_Shrink (-0.011 DISPLACE BEFORE ARMATURE) including upperarm & lowerarm."""
    inv_rig = rig.matrix_world.inverted()
    xf = inv_rig @ body.matrix_world
    pts = [xf @ v.co for v in body.data.vertices]
    if not pts:
        return

    z_foot = min(p.z for p in pts)
    z_crown = max(p.z for p in pts)
    scale_h = max(0.5, z_crown - z_foot) / 1.78
    z_lo = z_foot + 0.45 * scale_h
    z_hi = z_crown - 0.28 * scale_h

    excl_prefixes = ("head", "neck", "wrist", "hand", "finger", "thumb", "lowerleg", "foot", "toe")
    excl_weights: dict[int, float] = {}
    for vg in body.vertex_groups:
        vname = vg.name.lower()
        if any(vname.startswith(pref) for pref in excl_prefixes):
            idx = vg.index
            for v in body.data.vertices:
                for g in v.groups:
                    if g.group == idx and g.weight > 0.0:
                        excl_weights[v.index] = excl_weights.get(v.index, 0.0) + g.weight
                        break

    excluded_vids = {vid for vid, ew in excl_weights.items() if ew > 0.05}
    vg_shrink = body.vertex_groups.get("UnderClothes_Shrink") or body.vertex_groups.new(name="UnderClothes_Shrink")
    taper = 0.045 * scale_h
    for v, p in zip(body.data.vertices, pts):
        if v.index in excluded_vids or abs(p.x) >= 0.85 * scale_h or p.z < z_lo or p.z > z_hi:
            w = 0.0
        else:
            wx = 1.0 - C.smoothstep(0.75 * scale_h, 0.85 * scale_h, abs(p.x))
            wz_lo = C.smoothstep(z_lo, z_lo + taper, p.z)
            wz_hi = 1.0 - C.smoothstep(z_hi - taper, z_hi, p.z)
            w_excl = 1.0 - C.smoothstep(0.0, 0.05, excl_weights.get(v.index, 0.0))
            w = wx * wz_lo * wz_hi * w_excl
        if w > 0.001:
            vg_shrink.add([v.index], w, "REPLACE")

    mod = body.modifiers.new(name="UnderClothes_Shrink", type="DISPLACE")
    mod.direction = "NORMAL"
    mod.mid_level = 0.0
    mod.strength = -0.011
    mod.vertex_group = vg_shrink.name

    arm_idx = next((i for i, m in enumerate(body.modifiers) if m.type == "ARMATURE"), None)
    if arm_idx is not None:
        cur_idx = list(body.modifiers).index(mod)
        if cur_idx > arm_idx and hasattr(body.modifiers, "move"):
            body.modifiers.move(cur_idx, arm_idx)


def _enforce_v12_eyes(equipped: dict[str, bpy.types.Object]) -> None:
    """Disable eye/lash shadow blocking & ensure independent EnhancedEye node tree copy per actor."""
    for k in ("Eyes", "Eyelashes", "Eyebrows"):
        obj = equipped.get(k)
        if obj is not None:
            obj.visible_shadow = False
    eye_obj = equipped.get("Eyes")
    if eye_obj is not None and eye_obj.data:
        for mat in eye_obj.data.materials:
            if mat and mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == "GROUP" and node.node_tree is not None:
                        node.node_tree = node.node_tree.copy()


def _override_flight_suit_wardrobe(equipped: dict[str, bpy.types.Object], tag: str) -> None:
    """Replace default patterned top/pants materials with clean matte dark-navy Nomex rescue flight suit!"""
    mat_suit = _make_nomex_flight_suit_mat(f"Mat_{tag}_NomexFlightSuit", (0.045, 0.068, 0.118))
    mat_boot = _make_mat(f"Mat_{tag}_FlightBoot", (0.025, 0.028, 0.032), roughness=0.42)
    for k, m in (("Shirt", mat_suit), ("Pants", mat_suit), ("Shoes", mat_boot)):
        obj = equipped.get(k)
        if obj is not None and obj.data and hasattr(obj.data, "materials"):
            if len(obj.data.materials) == 0:
                obj.data.materials.append(m)
            else:
                for i in range(len(obj.data.materials)):
                    obj.data.materials[i] = m


def _near_hero_frame(f: int) -> bool:
    return any(abs(f - h) <= 6 for h in _HERO_BLINK_EXCL)


def _apply_actor_face(
    rig: bpy.types.Object,
    f: int,
    jaw_deg: float,
    brow_lift: float,
    smile: float,
    eye_yaw_deg: float,
    eye_pitch_deg: float,
    force_alert_lids: bool = False,
) -> None:
    jaw_clamped = C.clamp(jaw_deg, 0.0, 14.0)
    t_val = (f - 1) / float(C.FPS)
    suppress_blink = force_alert_lids or _near_hero_frame(f)
    mc.apply_facial_acting(
        rig,
        t=t_val,
        frame=f,
        jaw_open_deg=jaw_clamped,
        brow_lift=brow_lift,
        smile=smile,
        blink_period=999.0 if suppress_blink else 3.8,
        blink_phase=0.35,
        eye_yaw_deg=eye_yaw_deg,
        eye_pitch_deg=eye_pitch_deg,
        lip_narrow=0.0,
    )
    if suppress_blink:
        # Wide alarmed eyes during emergency panic (-7 deg) vs relaxed open baseline (-5 deg)!
        lid_open = -7.0 if force_alert_lids else -5.0
        mc.set_bone_char_rot(rig, "orbicularis03.L", lid_open, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", lid_open, 0.0, 0.0, frame=f)


def _pose_seated_body(
    rig: bpy.types.Object,
    torso_fwd_deg: float,
    torso_twist_deg: float,
    head_pitch_deg: float,
    head_yaw_deg: float,
    frame: int,
) -> None:
    for side in ("L", "R"):
        mc.set_bone_char_rot(rig, f"thigh01.{side}", -82.0, 0.0, 0.0, frame=frame)
        mc.set_bone_char_rot(rig, f"shin01.{side}", 76.0, 0.0, 0.0, frame=frame)
    mc.set_bone_char_rot(rig, "spine05", torso_fwd_deg * 0.25, 0.0, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine04", torso_fwd_deg * 0.25, 0.0, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine03", torso_fwd_deg * 0.20, 0.0, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine02", torso_fwd_deg * 0.15, 0.0, torso_twist_deg * 0.15, frame=frame)
    mc.set_bone_char_rot(rig, "spine01", torso_fwd_deg * 0.15, 0.0, torso_twist_deg * 0.10, frame=frame)
    mc.set_bone_char_rot(rig, "neck01", -torso_fwd_deg * 0.30 + head_pitch_deg * 0.45, 0.0, head_yaw_deg * 0.40, frame=frame)
    mc.set_bone_char_rot(rig, "head", -torso_fwd_deg * 0.25 + head_pitch_deg * 0.55, 0.0, head_yaw_deg * 0.60, frame=frame)


def _pose_arms_fk(
    rig: bpy.types.Object,
    ua_l: tuple[float, float, float],
    la_l: tuple[float, float, float],
    ua_r: tuple[float, float, float],
    la_r: tuple[float, float, float],
    hand_l: str,
    hand_r: str,
    frame: int,
) -> None:
    for tw in ("upperarm02.L", "lowerarm02.L", "upperarm02.R", "lowerarm02.R"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    mc.apply_arm_with_clavicle(rig, "L", ua_l, la_l, frame=frame)
    mc.apply_arm_with_clavicle(rig, "R", ua_r, la_r, frame=frame)
    mc.apply_hand_gesture(rig, "L", hand_l.lower(), frame=frame)
    mc.apply_hand_gesture(rig, "R", hand_r.lower(), frame=frame)


def _attach_to_bone(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    bpy.context.view_layer.update()
    world_mat = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world_mat


def _build_flight_harness_and_collar(
    name: str,
    rig: bpy.types.Object,
    col: bpy.types.Collection,
) -> bpy.types.Object:
    """
    4-Point Cockpit Safety Harness Straps + High-Collar Zip Guard + Rescue Crimson Chest Patches
    parented cleanly to `spine03` — covers any shirt neckline and gives instant rescue-aviator authenticity!
    """
    mat_strap = _make_mat(f"Mat_{name}_HarnessStrap", (0.04, 0.045, 0.052), roughness=0.72)
    mat_metal = _make_mat(f"Mat_{name}_BuckleSteel", (0.68, 0.72, 0.76), metallic=0.90, roughness=0.22)
    mat_red = _make_mat(f"Mat_{name}_RescuePatch", (0.82, 0.08, 0.06), roughness=0.45)
    mat_navy = _make_nomex_flight_suit_mat(f"Mat_{name}_CollarNomex", (0.038, 0.058, 0.102))

    bpy.context.view_layer.update()
    sc = rig.matrix_world @ rig.pose.bones["spine03"].head
    bm = bmesh.new()

    # High zipped Nomex collar band + sternum flight-suit placket clearly OUTSIDE chest mesh
    _add_box_bm(bm, (sc.x, sc.y - 0.045, sc.z + 0.168), (0.164, 0.155, 0.104), mat_idx=3, bevel_w=0.016)
    _add_box_bm(bm, (sc.x, sc.y - 0.148, sc.z + 0.078), (0.155, 0.056, 0.175), mat_idx=3, bevel_w=0.012)
    # Two shoulder harness webbing straps (-8 deg tilts top forward matching chest slope, never sinking into torso!)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sc.x + sx * 0.095, sc.y - 0.148, sc.z + 0.015), (0.046, 0.022, 0.27), mat_idx=0, bevel_w=0.004, rot_deg=(-8.0, 0.0, sx * -10.0))
        # High-visibility rescue crimson shoulder tab
        _add_box_bm(bm, (sc.x + sx * 0.135, sc.y - 0.075, sc.z + 0.138), (0.064, 0.044, 0.016), mat_idx=2, bevel_w=0.003)
    # Center circular aviation rotary quick-release harness buckle
    _add_box_bm(bm, (sc.x, sc.y - 0.170, sc.z - 0.095), (0.068, 0.026, 0.068), mat_idx=1, bevel_w=0.010)
    # Rescue chest name badge patch
    _add_box_bm(bm, (sc.x + 0.096, sc.y - 0.168, sc.z + 0.06), (0.058, 0.012, 0.026), mat_idx=2, bevel_w=0.002)

    h_obj = _new_bmesh_obj(name, bm, [mat_strap, mat_metal, mat_red, mat_navy], Matrix.Identity(4), col, smooth=True)
    _attach_to_bone(h_obj, rig, "spine03")
    return h_obj


def _build_rescue_helmet(
    name: str,
    rig: bpy.types.Object,
    col: bpy.types.Collection,
    stripe_rgb: tuple[float, float, float],
) -> bpy.types.Object:
    """White-and-red SPH-5 Rescue Flight Helmet with raised glossy tinted visor & comms boom mic."""
    mat_w = _make_mat(f"Mat_{name}_ShellWhite", (0.94, 0.95, 0.96), metallic=0.08, roughness=0.18, coat=0.72)
    mat_r = _make_mat(f"Mat_{name}_Stripe", stripe_rgb, metallic=0.14, roughness=0.22, coat=0.65)
    mat_v = _make_mat(f"Mat_{name}_Visor", (0.03, 0.05, 0.08), metallic=0.55, roughness=0.05, coat=0.95)
    mat_b = _make_mat(f"Mat_{name}_MicBlk", (0.025, 0.028, 0.032), metallic=0.28, roughness=0.42)

    bpy.context.view_layer.update()
    hc = rig.matrix_world @ rig.pose.bones["head"].head
    cz = hc.z + 0.098
    cy = hc.y + 0.006
    bm = bmesh.new()

    bmesh.ops.create_uvsphere(
        bm, u_segments=24, v_segments=16, radius=1.0,
        matrix=Matrix.Translation((hc.x, cy, cz)) @ Matrix.Diagonal((0.104, 0.116, 0.104, 1.0)),
    )
    del_faces = []
    for f in bm.faces:
        fc = f.calc_center_median()
        if fc.y < cy - 0.042 and fc.z < cz + 0.038:
            del_faces.append(f)
        elif fc.z < cz - 0.055:
            del_faces.append(f)
        else:
            f.material_index = 1 if abs(fc.x - hc.x) < 0.024 else 0
    if del_faces:
        bmesh.ops.delete(bm, geom=del_faces, context="FACES")

    # Raised curved dark glossy visor housing (catches red ENG OUT reflections!)
    _add_box_bm(bm, (hc.x, cy - 0.095, cz + 0.042), (0.144, 0.046, 0.044), mat_idx=2, bevel_w=0.012, rot_deg=(-20.0, 0.0, 0.0))

    # Padded comms earcups + flexible boom microphone curving cleanly to mouth corner (below eyes!)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (hc.x + sx * 0.100, cy - 0.004, hc.z + 0.062), (0.028, 0.066, 0.078), mat_idx=3, bevel_w=0.008)
    p0 = Vector((hc.x - 0.096, cy - 0.018, hc.z + 0.026))
    p1 = Vector((hc.x - 0.076, cy - 0.108, hc.z - 0.002))
    p2 = Vector((hc.x - 0.030, cy - 0.118, hc.z - 0.006))
    _add_strut_bm(bm, p0, p1, 0.0042, mat_idx=3, segs=8)
    _add_strut_bm(bm, p1, p2, 0.0042, mat_idx=3, segs=8)
    _add_box_bm(bm, (p2.x, p2.y, p2.z), (0.020, 0.015, 0.015), mat_idx=3, bevel_w=0.004)

    h_obj = _new_bmesh_obj(name, bm, [mat_w, mat_r, mat_v, mat_b], Matrix.Identity(4), col, smooth=True)
    _attach_to_bone(h_obj, rig, "head")
    return h_obj


def _build_cockpit_set(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """
    Enclosed Dark 1:1 Scale Modern Rescue Helicopter Cockpit (Beats 1B `116..240` & Beat 5A `1291..1450`):
      - Solid enclosed interior cabin walls, padded side doors with recessed side windows, armored rear
        bulkhead with rescue gear rail, overhead circuit-breaker switch console, and subtle angled canopy glass!
      - Physical **3D Dual-Tachometer Gauge (`ENG` crimson needle plunging to `0%` + `ROTOR` emerald needle
        staying alive in green arc!)** + glowing **Beat 5A Radar Altimeter Countdown Bar**!
      - Pulsing crimson `ENG OUT / LOW RPM` Master Caution Strip + dual glowing MFD displays!
      - Real visible **Collective Pitch Lever (`Char_Heli_CollectiveLever`)** & **Cyclic Stick** gripped cleanly!
    """
    mat_dark = _make_mat("Mat_Cockpit_DarkAlum", (0.028, 0.032, 0.038), metallic=0.62, roughness=0.40)
    mat_wall = _make_mat("Mat_Cockpit_InteriorWall", (0.048, 0.054, 0.062), metallic=0.25, roughness=0.58)
    mat_seat = _make_mat("Mat_Cockpit_SeatFabric", (0.052, 0.056, 0.064), roughness=0.86)
    mat_red = _make_mat("Mat_Cockpit_RescueRed", (0.82, 0.055, 0.045), metallic=0.14, roughness=0.26, coat=0.5)
    mat_mfd = _make_mat("Mat_Cockpit_MFDGreen", (0.06, 0.88, 0.54), roughness=0.18, emit_rgb=(0.10, 0.94, 0.58), emit_str=1.55)
    mat_cyan = _make_mat("Mat_Cockpit_AltCyan", (0.12, 0.82, 1.00), roughness=0.18, emit_rgb=(0.14, 0.88, 1.00), emit_str=1.65)
    mat_yel = _make_mat("Mat_Cockpit_AmberGrip", (0.98, 0.72, 0.08), roughness=0.28, emit_rgb=(1.00, 0.74, 0.10), emit_str=0.45)
    mat_white = _make_mat("Mat_Cockpit_DialFace", (0.015, 0.018, 0.024), roughness=0.35)
    mat_glass = _make_cockpit_glass_glare_mat()

    # Pulsing crimson `ENG OUT` annunciator material (bright urgent alarm in Beat 1B!)
    mat_warn = bpy.data.materials.new("Mat_Cockpit_EngOutWarn")
    mat_warn.use_nodes = True
    bsdf_w = mat_warn.node_tree.nodes["Principled BSDF"]
    bsdf_w.inputs["Base Color"].default_value = (0.98, 0.06, 0.03, 1.0)
    bsdf_w.inputs["Roughness"].default_value = 0.18
    bsdf_w.inputs["Emission Color"].default_value = (1.00, 0.10, 0.04, 1.0)
    for f in range(C.B1_COCKPIT_START, 241, 4):
        pulse = 0.25 + 0.75 * (0.5 + 0.5 * math.sin((f - 116) * 0.52))
        bsdf_w.inputs["Emission Strength"].default_value = 1.95 * pulse
        bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)
    bsdf_w.inputs["Emission Strength"].default_value = 0.85
    bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=1291)
    bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=C.B5_COCKPIT_END)

    bm = bmesh.new()
    # 1. Enclosed Cabin Floor, Center Avionics Pedestal & Armored Rear Cabin Bulkhead (with framed rear window!)
    _add_box_bm(bm, (0.0, 0.18, -0.04), (1.92, 1.65, 0.08), mat_idx=0, bevel_w=0.01)
    _add_box_bm(bm, (0.0, 0.02, 0.44), (0.22, 0.64, 0.72), mat_idx=0, bevel_w=0.012)
    # Armored rear bulkhead behind seats (Y = +0.86 m) — lower wall + left/right pillars + upper header
    # leaving a sleek horizontal rear cabin window at Z = 1.08..1.48 so golden skyline peeks cleanly behind crew!
    _add_box_bm(bm, (0.0, 0.86, 0.50), (1.88, 0.10, 1.08), mat_idx=1, bevel_w=0.015)
    _add_box_bm(bm, (0.0, 0.86, 1.64), (1.88, 0.10, 0.34), mat_idx=1, bevel_w=0.015)
    for sx in (-0.82, 0.0, +0.82):
        w_p = 0.24 if abs(sx) > 0.1 else 0.12
        _add_box_bm(bm, (sx, 0.86, 1.28), (w_p, 0.10, 0.44), mat_idx=1, bevel_w=0.010)
    # Rescue medical grab-handle & coiled oxygen/radio rack on rear bulkhead
    _add_box_bm(bm, (0.0, 0.80, 1.16), (0.10, 0.06, 0.24), mat_idx=3, bevel_w=0.008)

    # 2. Enclosed Left & Right Cockpit Side Doors (X = +/-0.90 m) with Armrests & Red Emergency Handles!
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sx * 0.90, 0.22, 0.48), (0.09, 1.38, 1.02), mat_idx=1, bevel_w=0.014)
        _add_box_bm(bm, (sx * 0.84, 0.20, 0.78), (0.06, 0.85, 0.08), mat_idx=2, bevel_w=0.012)
        _add_box_bm(bm, (sx * 0.84, -0.02, 0.90), (0.03, 0.12, 0.03), mat_idx=3, bevel_w=0.005)
        # Forward swept A-pillars & upper door sill frames (no center mullion blocking faces!)
        _add_strut_bm(bm, Vector((sx * 0.86, -0.44, 0.96)), Vector((sx * 0.74, 0.04, 1.70)), 0.034, mat_idx=0, segs=10)
        _add_strut_bm(bm, Vector((sx * 0.86, 0.82, 1.02)), Vector((sx * 0.78, 0.82, 1.70)), 0.034, mat_idx=0, segs=10)
        _add_strut_bm(bm, Vector((sx * 0.74, 0.04, 1.70)), Vector((sx * 0.78, 0.82, 1.70)), 0.032, mat_idx=0, segs=10)

    # 3. Overhead Avionics Roof Deck & Circuit-Breaker Switch Console (Z = 1.66..1.74 m)
    _add_box_bm(bm, (0.0, 0.36, 1.72), (1.64, 1.04, 0.10), mat_idx=0, bevel_w=0.018)
    _add_box_bm(bm, (0.0, 0.04, 1.65), (0.64, 0.28, 0.06), mat_idx=1, bevel_w=0.010)
    # Rows of overhead warning indicators & glowing backlit toggle rows
    for ix in range(-3, 4):
        _add_box_bm(bm, (ix * 0.075, -0.04, 1.615), (0.042, 0.024, 0.018), mat_idx=(5 if abs(ix) <= 1 else 6), bevel_w=0.003)

    # 4. Dual Armored Rescue Bucket Seats (widened cleanly to X = +/-0.40 m!)
    for sx in (-0.40, +0.40):
        _add_box_bm(bm, (sx, 0.34, 0.42), (0.46, 0.46, 0.10), mat_idx=2, bevel_w=0.016)
        _add_box_bm(bm, (sx, 0.56, 0.96), (0.44, 0.10, 0.88), mat_idx=2, bevel_w=0.016, rot_deg=(-7.0, 0.0, 0.0))
        _add_box_bm(bm, (sx, 0.59, 1.44), (0.26, 0.10, 0.22), mat_idx=0, bevel_w=0.012)

    # 5. Sleek Low-Profile Instrument Glare-Shield + Forward-Facing Dual MFDs + Pulsing `ENG OUT` Strip + Dial Housing!
    _add_box_bm(bm, (0.0, -0.21, 0.885), (1.58, 0.24, 0.15), mat_idx=0, bevel_w=0.016, rot_deg=(8.0, 0.0, 0.0))
    # Dual Glowing Aviation MFD Displays flush on front -Y instrument face (100% visible to camera!)
    _add_box_bm(bm, (-0.42, -0.345, 0.900), (0.28, 0.025, 0.12), mat_idx=4, bevel_w=0.005, rot_deg=(12.0, 0.0, 0.0))
    _add_box_bm(bm, (0.42, -0.345, 0.900), (0.28, 0.025, 0.12), mat_idx=4, bevel_w=0.005, rot_deg=(12.0, 0.0, 0.0))
    # Prominent Pulsing Crimson `ENG OUT / MASTER CAUTION` Annunciator Bar right on top of cowl crest!
    _add_box_bm(bm, (0.0, -0.29, 0.985), (0.36, 0.055, 0.044), mat_idx=5, bevel_w=0.008)
    # Recessed circular bezel housing for the Live Physical 3D Dual-Tachometer Gauge (facing -Y/+Z toward camera!)
    _add_box_bm(bm, (0.0, -0.340, 0.905), (0.20, 0.035, 0.14), mat_idx=7, bevel_w=0.008, rot_deg=(14.0, 0.0, 0.0))
    # Green RPM Arc & Red Warning Arc strips on the 3D Tachometer Dial face
    _add_box_bm(bm, (0.052, -0.362, 0.940), (0.052, 0.010, 0.018), mat_idx=4, bevel_w=0.002)
    _add_box_bm(bm, (-0.052, -0.362, 0.875), (0.052, 0.010, 0.018), mat_idx=5, bevel_w=0.002)

    cockpit_obj = _new_bmesh_obj(
        "Char_Heli_CockpitFrame",
        bm,
        [mat_dark, mat_wall, mat_seat, mat_red, mat_mfd, mat_warn, mat_yel, mat_white],
        Matrix.Identity(4),
        col,
        smooth=True,
    )

    # 6. Subtle Angled Side Canopy Glass Panes (adds authentic cockpit reflections without blocking faces!)
    bm_gl = bmesh.new()
    for sx in (-1.0, +1.0):
        _add_box_bm(bm_gl, (sx * 0.87, 0.34, 1.32), (0.008, 0.96, 0.62), mat_idx=0, bevel_w=0.0)
    glass_obj = _new_bmesh_obj("Char_Heli_CanopyGlass", bm_gl, [mat_glass], Matrix.Identity(4), col, smooth=False, no_shadow=True)

    # 7. LIVE ANIMATED 3D DUAL-TACHOMETER NEEDLES (`ENG` crashes 100% -> 0% in Beat 1B; `ROTOR` stays green!)
    bm_n_eng = bmesh.new()
    _add_box_bm(bm_n_eng, (0.0, -0.006, 0.028), (0.009, 0.008, 0.058), mat_idx=0, bevel_w=0.002)
    needle_eng = _new_bmesh_obj(
        "Char_Heli_Needle_ENG", bm_n_eng, [mat_warn], Matrix.Translation((0.0, -0.368, 0.905)), col, smooth=True, no_shadow=True,
    )
    bm_n_rot = bmesh.new()
    _add_box_bm(bm_n_rot, (0.0, -0.010, 0.026), (0.008, 0.008, 0.054), mat_idx=0, bevel_w=0.002)
    needle_rot = _new_bmesh_obj(
        "Char_Heli_Needle_ROTOR", bm_n_rot, [mat_mfd], Matrix.Translation((0.0, -0.372, 0.905)), col, smooth=True, no_shadow=True,
    )
    for f in sorted(set(range(C.B1_COCKPIT_START, 241, 3)) | set(range(1291, C.B5_COCKPIT_END + 1, 5)) | {116, 240, 1291, C.B5_COCKPIT_END}):
        # Map 0..100% -> +75 deg (left red zone) .. -65 deg (right green zone) around Y axis!
        eng_ang = math.radians(C.lerp(75.0, -65.0, C.engine_rpm_pct_at(float(f)) / 100.0))
        rot_ang = math.radians(C.lerp(75.0, -65.0, C.clamp(C.rotor_rpm_pct_at(float(f)) / 105.0, 0.0, 1.0)))
        needle_eng.rotation_euler = Euler((0.0, eng_ang, 0.0), "XYZ")
        needle_eng.keyframe_insert("rotation_euler", frame=f)
        needle_rot.rotation_euler = Euler((0.0, rot_ang, 0.0), "XYZ")
        needle_rot.keyframe_insert("rotation_euler", frame=f)

    # 8. Beat 5A Glowing 3D Radar Altimeter Bar (visibly drops as roof approaches from 125 ft -> 32 ft!)
    bm_alt = bmesh.new()
    _add_box_bm(bm_alt, (0.0, 0.0, 0.055), (0.026, 0.012, 0.110), mat_idx=0, bevel_w=0.003)
    alt_bar = _new_bmesh_obj(
        "Char_Heli_RadarAltBar", bm_alt, [mat_cyan], Matrix.Translation((-0.20, -0.366, 0.845)), col, smooth=True, no_shadow=True,
    )
    for f in range(1291, C.B5_COCKPIT_END + 1, 5):
        frac = C.clamp(C.radar_alt_ft_at(float(f)) / 130.0, 0.12, 1.0)
        alt_bar.scale = Vector((1.0, 1.0, frac))
        alt_bar.keyframe_insert("scale", frame=f)
    _key_visibility([alt_bar], [(1291, C.B5_COCKPIT_END)], total_frames)

    # 9. Hero Animated Collective Pitch Lever (`Char_Heli_CollectiveLever`, gripped by Pilot's LEFT hand!)
    col_pivot = bpy.data.objects.new("Char_Heli_CollectivePivot", None)
    col_pivot.location = Vector(C.STAGE["cockpit_collective_pivot"])
    _link(col_pivot, col)
    bm_lev = bmesh.new()
    _add_strut_bm(bm_lev, Vector((0.0, 0.0, 0.0)), Vector((0.0, -0.44, 0.12)), 0.017, mat_idx=0, segs=12)
    _add_strut_bm(bm_lev, Vector((0.0, -0.42, 0.115)), Vector((0.0, -0.58, 0.155)), 0.028, mat_idx=1, segs=14)
    _add_box_bm(bm_lev, (0.0, -0.57, 0.185), (0.044, 0.046, 0.036), mat_idx=2, bevel_w=0.006)
    lev_obj = _new_bmesh_obj("Char_Heli_CollectiveLever", bm_lev, [mat_dark, mat_yel, mat_red], Matrix.Identity(4), col, smooth=True)
    lev_obj.parent = col_pivot

    for f in sorted(set(range(C.B1_COCKPIT_START, 241, 3)) | set(range(1291, C.B5_COCKPIT_END + 1, 3)) | {116, 135, 178, 240, 1291, 1335, 1418, C.B5_COCKPIT_END}):
        lvl = C.collective_level_at(float(f))
        pitch_deg = C.lerp(-24.0, 22.0, lvl)
        col_pivot.rotation_euler = Euler((math.radians(pitch_deg), 0.0, 0.0), "XYZ")
        col_pivot.keyframe_insert("rotation_euler", frame=f)

    # 10. Pilot Cyclic Control Stick (between Pilot's knees at X = +0.40 m, gripped cleanly by RIGHT hand!)
    cyc_pivot = bpy.data.objects.new("Char_Heli_CyclicPivot", None)
    cyc_pivot.location = Vector(C.STAGE["cockpit_cyclic_pivot"])
    _link(cyc_pivot, col)
    bm_cyc = bmesh.new()
    _add_strut_bm(bm_cyc, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.04, 0.44)), 0.015, mat_idx=0, segs=12)
    _add_box_bm(bm_cyc, (0.0, 0.05, 0.47), (0.040, 0.058, 0.095), mat_idx=2, bevel_w=0.010)
    cyc_obj = _new_bmesh_obj("Char_Heli_CyclicStick", bm_cyc, [mat_dark, mat_yel, mat_red], Matrix.Identity(4), col, smooth=True)
    cyc_obj.parent = cyc_pivot

    for f in (116, 152, 240, 1291, 1360, 1425, C.B5_COCKPIT_END):
        cyc_pitch = -6.0 if f <= 240 else C.lerp(-4.0, 12.0, C.smootherstep(1335.0, 1420.0, float(f)))
        cyc_pivot.rotation_euler = Euler((math.radians(cyc_pitch), 0.0, 0.0), "XYZ")
        cyc_pivot.keyframe_insert("rotation_euler", frame=f)

    all_objs = [cockpit_obj, glass_obj, needle_eng, needle_rot, lev_obj, cyc_obj]
    _key_visibility(all_objs, _COCKPIT_WINDOWS, total_frames)
    return {
        "frame": cockpit_obj,
        "collective_pivot": col_pivot,
        "cyclic_pivot": cyc_pivot,
        "objs": all_objs + [alt_bar],
    }


def _animate_pilot(rig: bpy.types.Object) -> None:
    """
    Animate `Heli_Pilot` (Right Seat PIC, X=+0.40 m, ZERO arm crossover!):
      - Beat 1B (116..240): Shocked furrowed brows, wide eyes darting to crashing `ENG 0%` dial ->
        tense gasping jaw -> left hand firmly SLAMS Collective Lever DOWN (`f=135..178`), right hand on Cyclic!
      - Beat 5A (1291..1450): Laser-focused gritty determination hauling Collective UP (`f=1335..1418`),
        followed by a genuine relieved exhale & half-smile (`f=1418..1450`) as the descent arrests!
    """
    kfs = sorted(set(range(C.B1_COCKPIT_START, 241, 4)) | set(range(1291, C.B5_COCKPIT_END + 1, 4)) | {116, 135, 152, 178, 240, 1291, 1335, 1380, 1418, C.B5_COCKPIT_END})
    for f in kfs:
        ff = float(f)
        if f <= 240:
            slam_u = C.smootherstep(135.0, 178.0, ff)
            breath = 1.1 * math.sin((ff - 116.0) * 0.24)
            shudder = 2.6 * math.exp(-0.08 * max(0.0, ff - 116.0)) * math.sin((ff - 116.0) * 1.4)
            torso_fwd = C.lerp(7.0, 12.5, slam_u) + shudder * 0.6 + breath * 0.5
            # Rapid panic eye/head saccade: check ENG OUT gauge -> glance Co-Pilot -> lock onto rooftop pad!
            if ff < 148.0:
                head_yaw = C.lerp(-14.0, -18.0, C.smoothstep(116.0, 148.0, ff))
                head_pitch = 10.0
                eye_yaw = -14.0
                eye_pitch = -12.0
            else:
                head_yaw = C.lerp(-18.0, -4.0, C.smootherstep(148.0, 210.0, ff))
                head_pitch = C.lerp(10.0, 5.0, C.smootherstep(148.0, 210.0, ff))
                eye_yaw = C.lerp(-14.0, -3.0, C.smootherstep(148.0, 210.0, ff))
                eye_pitch = -6.0

            _pose_seated_body(rig, torso_fwd, -5.0, head_pitch, head_yaw, frame=f)
            # Left arm reaches down-forward onto center-right Cyclic Stick (Z~1.02); Right arm slams Collective!
            ua_l = (-14.0, 32.0, -14.0)
            la_l = (-34.0, 8.0, -22.0)
            ua_r = (C.lerp(-18.0, -10.0, slam_u), -32.0, 12.0)
            la_r = (C.lerp(-32.0, -18.0, slam_u), -6.0, 16.0)
            _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "FIST", "FIST", frame=f)

            # Deeply furrowed emergency brows + tense gasping/calling-out jaw!
            jaw = 8.5 + 2.5 * math.sin((ff - 116.0) * 0.22) if ff < 195.0 else 5.5
            brow = -0.82 if ff < 185.0 else -0.58
            _apply_actor_face(rig, f, jaw_deg=jaw, brow_lift=brow, smile=0.0, eye_yaw_deg=eye_yaw, eye_pitch_deg=eye_pitch, force_alert_lids=True)
        else:
            pull_u = C.smootherstep(1335.0, 1418.0, ff)
            relief_u = C.smootherstep(1414.0, 1448.0, ff)
            torso_fwd = C.lerp(10.0, 4.5, pull_u)
            head_pitch = C.lerp(6.5, 1.5, pull_u) + 3.0 * math.sin(math.pi * relief_u)
            head_yaw = C.lerp(-3.0, -8.0, relief_u)
            _pose_seated_body(rig, torso_fwd, -4.0, head_pitch, head_yaw, frame=f)

            # Left hand holds & flares cyclic aft; Right hand firmly HAULS collective lever UP for 30-ft flare!
            ua_l = (C.lerp(-14.0, -12.0, pull_u), 32.0, -14.0)
            la_l = (C.lerp(-34.0, -40.0, pull_u), 8.0, -22.0)
            ua_r = (C.lerp(-10.0, -24.0, pull_u), -32.0, 12.0)
            la_r = (C.lerp(-20.0, -42.0, pull_u), -6.0, 16.0)
            _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "FIST", "FIST", frame=f)

            # Gritty laser focus -> authentic relieved exhale & half-smile!
            brow = C.lerp(-0.48, 0.24, relief_u)
            smile = C.lerp(0.0, 0.46, relief_u)
            jaw = C.lerp(2.8, 1.2, relief_u)
            _apply_actor_face(rig, f, jaw_deg=jaw, brow_lift=brow, smile=smile, eye_yaw_deg=C.lerp(-3.0, -8.0, relief_u), eye_pitch_deg=-3.0, force_alert_lids=(ff < 1415.0))


def _animate_copilot(rig: bpy.types.Object) -> None:
    """
    Animate `Heli_CoPilot` (Left Seat, X=-0.40 m, ZERO arm crossover!):
      - Beat 1B (116..240): Alarmed wide-eyed glance at red `ENG OUT` panel -> shouts checklist on mic
        while right arm points clearly forward-down toward the rooftop helipad (strictly X < -0.18 m!)
        and left hand braces on left side armrest/glare-shield!
      - Beat 5A (1291..1450): Calls out `"30 feet!"` altitude drop, then gives a relieved nod & smile!
    """
    kfs = sorted(set(range(C.B1_COCKPIT_START, 241, 4)) | set(range(1291, C.B5_COCKPIT_END + 1, 4)) | {116, 145, 195, 240, 1291, 1350, 1415, C.B5_COCKPIT_END})
    for f in kfs:
        ff = float(f)
        if f <= 240:
            pt_u = C.smootherstep(122.0, 158.0, ff)
            head_yaw = C.lerp(18.0, 8.0, pt_u)
            head_pitch = C.lerp(7.0, 11.0, pt_u)
            _pose_seated_body(rig, 9.5, 6.0, head_pitch, head_yaw, frame=f)

            # Left arm tucked comfortably down onto CoPilot's side-console / instrument brace (Z~1.01)
            ua_l = (-14.0, 32.0, -12.0)
            la_l = (-34.0, 6.0, -18.0)
            # Right arm points decisively FORWARD toward rooftop helipad ahead
            ua_r = (C.lerp(-20.0, -44.0, pt_u), -24.0, C.lerp(12.0, 22.0, pt_u))
            la_r = (C.lerp(-36.0, -14.0, pt_u), 0.0, C.lerp(14.0, 10.0, pt_u))
            _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "FIST", "POINT", frame=f)

            jaw = 7.5 + 3.0 * math.sin((ff - 116.0) * 0.28)
            _apply_actor_face(rig, f, jaw_deg=jaw, brow_lift=-0.72, smile=0.0, eye_yaw_deg=12.0, eye_pitch_deg=-9.0, force_alert_lids=True)
        else:
            call_u = C.smootherstep(1320.0, 1385.0, ff)
            relief_u = C.smootherstep(1412.0, 1448.0, ff)
            head_pitch = 4.0 + 4.5 * math.sin(math.pi * relief_u)
            _pose_seated_body(rig, 7.0, 8.0, head_pitch, C.lerp(14.0, 20.0, relief_u), frame=f)

            ua_l = (-14.0, 32.0, -12.0)
            la_l = (-34.0, 6.0, -18.0)
            ua_r = (-16.0, -30.0, 14.0)
            la_r = (-36.0, -6.0, 16.0)
            _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "OPEN", "FIST", frame=f)

            jaw = C.lerp(7.0, 1.5, relief_u) if call_u > 0.1 else 2.5
            _apply_actor_face(
                rig, f,
                jaw_deg=jaw,
                brow_lift=C.lerp(-0.32, 0.28, relief_u),
                smile=C.lerp(0.0, 0.50, relief_u),
                eye_yaw_deg=14.0,
                eye_pitch_deg=-4.0,
                force_alert_lids=(ff < 1412.0),
            )


def _build_aviator_cfg(height_m: float, iris_rgb: tuple[float, float, float]) -> dict[str, Any]:
    """V12 verified MakeHuman config for rescue aviators."""
    return {
        "gender": 1.0,
        "age": 0.56,
        "muscle": 0.62,
        "weight": 0.46,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(height_m),
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.10, 0.08, 0.06),
        "hair_tint_fac": 0.70,
        "hair_value_boost": 0.72,
        "hair_roughness": 0.30,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": iris_rgb,
        "iris_minor_rgb": (iris_rgb[0] * 0.45, iris_rgb[1] * 0.45, iris_rgb[2] * 0.45),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
    }


def _spawn_aviator(
    name: str,
    col: bpy.types.Collection,
    location: tuple[float, float, float],
    yaw_deg: float,
    height_m: float,
    iris_color: tuple[float, float, float],
) -> tuple[bpy.types.Object, bpy.types.Object, dict[str, bpy.types.Object]]:
    cfg = _build_aviator_cfg(height_m=height_m, iris_rgb=iris_color)
    # Shift floor_z down -0.335 m so seated pelvis sits cleanly in bucket seat & Pilot helmet clears Z=1.65 m roof deck
    res = mc.create_realistic_scientist(
        name=name,
        floor_z=float(location[2]) - 0.335,
        start_xy=(float(location[0]), float(location[1])),
        yaw_deg=float(yaw_deg),
        cfg=cfg,
    )
    rig: bpy.types.Object = res["rig"]
    body: bpy.types.Object = res["body"]
    equipped: dict[str, bpy.types.Object] = res["equipped"]
    _link(rig, col)
    _link(body, col)
    for o in equipped.values():
        if o is not None:
            _link(o, col)
    bpy.context.view_layer.update()
    return rig, body, equipped


def build_characters(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Build enclosed 1:1 Rescue Cockpit + 2 Photoreal 4K MPFB2 Rescue Aviators in Matte Nomex Flight Suits."""
    cockpit = _build_cockpit_set(col, total_frames)

    # 1. Pilot (Right Seat PIC, X = +0.40 m, facing -Y)
    px, py, pz = C.CREW["pilot_pos"]
    pilot_rig, pilot_body, pilot_eq = _spawn_aviator(
        name="Heli_Pilot",
        col=col,
        location=(px, py, pz),
        yaw_deg=float(C.CREW["pilot_yaw_deg"]),
        height_m=1.78,
        iris_color=(0.18, 0.36, 0.54),
    )
    _enforce_v12_eyes(pilot_eq)
    _override_flight_suit_wardrobe(pilot_eq, "Pilot")
    _apply_underclothes_shrink(pilot_body, pilot_rig)
    pilot_helmet = _build_rescue_helmet("Char_Heli_PilotHelmet", pilot_rig, col, stripe_rgb=(0.86, 0.07, 0.05))
    pilot_harness = _build_flight_harness_and_collar("Char_Heli_PilotHarness", pilot_rig, col)
    _animate_pilot(pilot_rig)

    # 2. Co-Pilot (Left Seat, X = -0.40 m, facing -Y)
    cx, cy, cz = C.CREW["copilot_pos"]
    copilot_rig, copilot_body, copilot_eq = _spawn_aviator(
        name="Heli_CoPilot",
        col=col,
        location=(cx, cy, cz),
        yaw_deg=float(C.CREW["copilot_yaw_deg"]),
        height_m=1.75,
        iris_color=(0.38, 0.28, 0.14),
    )
    _enforce_v12_eyes(copilot_eq)
    _override_flight_suit_wardrobe(copilot_eq, "CoPilot")
    _apply_underclothes_shrink(copilot_body, copilot_rig)
    copilot_helmet = _build_rescue_helmet("Char_Heli_CoPilotHelmet", copilot_rig, col, stripe_rgb=(0.95, 0.72, 0.08))
    copilot_harness = _build_flight_harness_and_collar("Char_Heli_CoPilotHarness", copilot_rig, col)
    _animate_copilot(copilot_rig)

    crew_objs: list[bpy.types.Object] = [
        pilot_body, pilot_helmet, pilot_harness,
        copilot_body, copilot_helmet, copilot_harness,
    ]
    for eq in (pilot_eq, copilot_eq):
        for o in eq.values():
            if o is not None and o not in crew_objs:
                crew_objs.append(o)
    _key_visibility(crew_objs, _COCKPIT_WINDOWS, total_frames)

    return {
        "cockpit": cockpit,
        "pilot": {"rig": pilot_rig, "body": pilot_body, "helmet": pilot_helmet, "harness": pilot_harness, "equipped": pilot_eq},
        "copilot": {"rig": copilot_rig, "body": copilot_body, "helmet": copilot_helmet, "harness": copilot_harness, "equipped": copilot_eq},
    }
