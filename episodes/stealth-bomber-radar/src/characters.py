"""
characters.py — 3 Photoreal 4K MakeHuman MPFB2 Rigged Human Characters & 1:1 B-2 Cockpit Set
for Episode #4: "How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (60 fps, 1800 frames).

Cast & Cinematic Action Choreography:
  1. `Radar_Operator` (Beats 1 & 6 — Enemy Air-Defense Radar Bunker):
     - 4K MPFB2 rigged adult male in dark olive-slate air-defense tactical uniform
     - Wearable twin-earcup Military Comms Headset + Boom Mic parented cleanly to `head` bone
     - Beat 1: Frantically taps radar console switches & double-takes between the BLANK green scope
       and the sunlit B-2 outside the panoramic window!
     - Beat 6: Leans forward squinting at the single microscopic 0.001 m^2 bumblebee speck on screen,
       then jerks head up in shock as the massive B-2 looms right outside the window!
  2. `Defense_Commander` (Beats 1 & 6 — Enemy Air-Defense Radar Bunker):
     - 4K MPFB2 rigged senior officer with independent amber-hazel irises
     - Wearable Peaked Military Officer Visor Cap + Gold Rank Badge parented to `head`,
       plus Gold Officer Shoulder Epaulets parented to `spine03`
     - Beat 1: Points urgently out the armored window at the B-2 while barking orders at the operator!
     - Beat 6: Leans intently over the console scope, then whips his head up in disbelief at the bomber!
  3. `Stealth_Pilot` + 1:1 Scale B-2 Spirit Fighter Cockpit (Beat 5A, f=1321..1430):
     - 4K MPFB2 rigged USAF B-2 pilot in sage-charcoal flight suit + tactical ejection-seat harness
     - Wearable Matte-Black HGU-55 Flight Helmet + Gleaming Gold-Reflective Sun Visor +
       Oxygen Mask & Corrugated Breathing Hose parented cleanly to `head`
     - Hands gripping the animated B-2 control stick & throttle inside a sculpted 1:1 B-2 cockpit
       surround with glowing green HUD combiner glass, dual backlit tactical MFDs, angled 4-pane
       canopy frame pillars, and sunlit blue sky banking outside!

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
  - Verified MPFB2 assets (`mindfront_lusekofta.mhclo`, `mindfront_male_trousers_2.mhclo`,
    `toigo_ankle_boots_male.mhclo`, `high-poly.mhclo`, `short04.mhclo`, `eyebrow002.mhclo`,
    `eyelashes01.mhclo`, `teeth_base.mhclo`, `tongue01.mhclo`).
  - `eye_obj.visible_shadow = False`, `lash_obj.visible_shadow = False`, independent `EnhancedEye`
    node tree per actor, relaxed `-5.0 deg` upper eyelids, jaw clamped `<= 14.0 deg`.
  - `UnderClothes_Shrink` (`-0.011` DISPLACE BEFORE `ARMATURE`) including `upperarm` & `lowerarm`.
  - Materials appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
  - Blender 5.3.0 safe FCurves (`C.iter_action_fcurves`).
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

import mpfb_characters as mc  # noqa: E402
import stealth_contract as C  # noqa: E402

_HERO_BLINK_EXCL: tuple[int, ...] = (
    1, 24, 68, 135, 190, 1321, 1365, 1415, 1561, 1625, 1695, 1760, 1800,
)


# ==============================================================================
# BLENDER 5.3.0 SAFE COLLECTION, BMESH, FCURVE & BONE HELPERS
# ==============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _set_fcurve_interp(
    obj_or_data: Any,
    interp: str = "BEZIER",
    skip_prefix: str = "hide_",
    constant_frames: set[int] | None = None,
) -> None:
    ad = getattr(obj_or_data, "animation_data", None)
    if not ad or not ad.action:
        return
    const_set = constant_frames or set()
    for fc in C.iter_action_fcurves(ad.action):
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
            continue
        for kp in fc.keyframe_points:
            if int(round(kp.co.x)) in const_set:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = interp
                if interp == "BEZIER":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"


def _key_visibility_windows(
    objs: list[bpy.types.Object],
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    t_max = max(1, int(total_frames))
    frames = {1, t_max}
    for s, e in visible_ranges:
        sc = max(1, min(t_max, int(s)))
        ec = max(1, min(t_max, int(e)))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < t_max:
            frames.add(ec + 1)

    for obj in objs:
        if obj is None:
            continue
        for f in sorted(frames):
            vis = any(s <= f <= e for s, e in visible_ranges)
            obj.hide_viewport = not vis
            obj.hide_render = not vis
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        _set_fcurve_interp(obj, "CONSTANT", skip_prefix="")


def _parent_to_bone_preserve_world(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    bpy.context.view_layer.update()
    pb = rig.pose.bones[bone_name]
    tail_off = 0.0 if pb.bone.use_relative_parent else pb.length
    bone_world = rig.matrix_world @ pb.matrix @ Matrix.Translation((0.0, tail_off, 0.0))
    orig_world = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = bone_world.inverted()
    obj.matrix_world = orig_world
    bpy.context.view_layer.update()


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: bpy.types.Material | list[bpy.types.Material],
    world_xf: Matrix,
    col: bpy.types.Collection,
    smooth: bool = True,
    no_shadow: bool = False,
) -> bpy.types.Object:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = world_xf.copy()
    if no_shadow:
        obj.visible_shadow = False
    return _link_obj(obj, col)


def _make_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.10,
    roughness: float = 0.45,
    emit_rgb: tuple[float, float, float] | None = None,
    emit_str: float = 0.0,
    coat: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
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


def _add_box_bm(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    bevel_w: float = 0.004,
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_w > 0.0:
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)
    rot_m = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ").to_matrix().to_4x4()
    bmesh.ops.transform(tmp, matrix=Matrix.Translation(Vector(center)) @ rot_m, verts=tmp.verts)
    vmap = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vmap[v] for v in f.verts])
        nf.material_index = mat_index
    tmp.free()


def _add_strut_bm(
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
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs,
        radius1=radius, radius2=radius, depth=length,
        matrix=Matrix.Translation(mid) @ rot_m,
    )
    vset = set(res["verts"])
    for f in bm.faces:
        if all(v in vset for v in f.verts):
            f.material_index = mat_index


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
    z_lo = z_foot + 0.50 * scale_h
    z_hi = z_crown - 0.28 * scale_h

    excl_prefixes = (
        "head", "neck", "wrist", "hand",
        "finger", "thumb", "lowerleg", "foot", "toe",
    )
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
    weights: list[float] = [0.0] * len(body.data.vertices)
    for v, p in zip(body.data.vertices, pts):
        if v.index in excluded_vids or abs(p.x) >= 0.85 * scale_h or p.z < z_lo or p.z > z_hi:
            w = 0.0
        else:
            wx = 1.0 - C.smoothstep(0.75 * scale_h, 0.85 * scale_h, abs(p.x))
            wz_lo = C.smoothstep(z_lo, z_lo + taper, p.z)
            wz_hi = 1.0 - C.smoothstep(z_hi - taper, z_hi, p.z)
            w_excl = 1.0 - C.smoothstep(0.0, 0.05, excl_weights.get(v.index, 0.0))
            w = wx * wz_lo * wz_hi * w_excl
        weights[v.index] = w

    for vid, w in enumerate(weights):
        if w > 0.001:
            vg_shrink.add([vid], w, "REPLACE")

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
    for k in ("Eyes", "Eyelashes", "Eyebrows"):
        obj = equipped.get(k)
        if obj is not None:
            obj.visible_shadow = False


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
        blink_phase=0.45,
        eye_yaw_deg=eye_yaw_deg,
        eye_pitch_deg=eye_pitch_deg,
        lip_narrow=0.0,
    )
    if suppress_blink:
        mc.set_bone_char_rot(rig, "orbicularis03.L", -5.0, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", -5.0, 0.0, 0.0, frame=f)


def _pose_spine_head(
    rig: bpy.types.Object,
    torso_fwd_deg: float,
    torso_twist_deg: float,
    head_pitch_deg: float,
    head_yaw_deg: float,
    frame: int,
) -> None:
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
    mc.apply_hand_gesture(rig, "L", hand_l, frame=frame)
    mc.apply_hand_gesture(rig, "R", hand_r, frame=frame)


# ==============================================================================
# WEARABLE MILITARY HEADGEAR & EQUIPMENT (HEAD-BONE / SPINE-BONE PARENTED!)
# ==============================================================================
def _head_world_center(rig: bpy.types.Object) -> Vector:
    bpy.context.view_layer.update()
    pb = rig.pose.bones["head"]
    return rig.matrix_world @ pb.head


def _build_comms_headset(rig: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    """Wearable twin-earcup military radar operator comms headset + boom mic parented to `head`."""
    mat_blk = _make_mat("Mat_Gear_HeadsetBlk", (0.032, 0.036, 0.042), metallic=0.25, roughness=0.42)
    mat_pad = _make_mat("Mat_Gear_HeadsetPad", (0.018, 0.020, 0.024), metallic=0.05, roughness=0.70)
    mat_mtl = _make_mat("Mat_Gear_HeadsetMetal", (0.32, 0.36, 0.40), metallic=0.88, roughness=0.26)

    hc = _head_world_center(rig)
    # Ear height sits ~0.065m above head bone base, head faces -Y in rest space!
    ez = hc.z + 0.068
    ey = hc.y + 0.012
    bm = bmesh.new()

    for sx in (-1.0, +1.0):
        # Outer earcup shell
        res_c = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=20,
            radius1=0.042, radius2=0.034, depth=0.034,
            matrix=Matrix.Translation((hc.x + sx * 0.094, ey, ez))
            @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        vs = set(res_c["verts"])
        for f in bm.faces:
            if all(v in vs for v in f.verts):
                f.material_index = 0
        # Soft earpad cushion
        res_p = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=20,
            radius1=0.044, radius2=0.044, depth=0.014,
            matrix=Matrix.Translation((hc.x + sx * 0.078, ey, ez))
            @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
        )
        vp = set(res_p["verts"])
        for f in bm.faces:
            if all(v in vp for v in f.verts):
                f.material_index = 1

    # Curved overhead steel/padded headband arching cleanly over crown
    n_arc = 14
    prev_p = None
    for i in range(n_arc + 1):
        a = math.pi * i / n_arc
        px = hc.x + 0.098 * math.cos(a)
        pz = ez + 0.128 * math.sin(a)
        cur_p = Vector((px, ey + 0.010, pz))
        if prev_p is not None:
            _add_strut_bm(bm, prev_p, cur_p, 0.0085, mat_index=0, segs=8)
        prev_p = cur_p

    # Flexible boom mic sweeping forward (-Y) toward operator's mouth
    p_mic0 = Vector((hc.x - 0.094, ey - 0.010, ez - 0.014))
    p_mic1 = Vector((hc.x - 0.068, ey - 0.092, ez - 0.046))
    p_mic2 = Vector((hc.x - 0.022, ey - 0.124, ez - 0.052))
    _add_strut_bm(bm, p_mic0, p_mic1, 0.0042, mat_index=2, segs=8)
    _add_strut_bm(bm, p_mic1, p_mic2, 0.0040, mat_index=2, segs=8)
    res_tip = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.010, matrix=Matrix.Translation(p_mic2))
    vt = set(res_tip["verts"])
    for f in bm.faces:
        if all(v in vt for v in f.verts):
            f.material_index = 1

    obj = _new_bmesh_obj("Char_Operator_Headset", bm, [mat_blk, mat_pad, mat_mtl], Matrix.Identity(4), col, smooth=True)
    _parent_to_bone_preserve_world(obj, rig, "head")
    return obj


def _build_officer_cap_and_epaulets(rig: bpy.types.Object, col: bpy.types.Collection) -> list[bpy.types.Object]:
    """Wearable Peaked Military Officer Visor Cap (`head`) + Gold Shoulder Epaulets (`spine03`)."""
    mat_cap = _make_mat("Mat_Gear_OfficerCap", (0.042, 0.056, 0.048), metallic=0.08, roughness=0.62)
    mat_brim = _make_mat("Mat_Gear_CapVisor", (0.012, 0.014, 0.018), metallic=0.22, roughness=0.18, coat=0.50)
    mat_gold = _make_mat("Mat_Gear_OfficerGold", (0.94, 0.72, 0.16), metallic=0.88, roughness=0.22)

    hc = _head_world_center(rig)
    cz = hc.z + 0.132
    cy = hc.y + 0.012
    bm = bmesh.new()

    # Cap band around forehead
    res_b = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.095, radius2=0.100, depth=0.042,
        matrix=Matrix.Translation((hc.x, cy, cz))
        @ Euler((math.radians(-6.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((0.92, 1.08, 1.0, 1.0)),
    )
    vb = set(res_b["verts"])
    for f in bm.faces:
        if all(v in vb for v in f.verts):
            f.material_index = 1

    # Flared military peaked officer crown
    res_cr = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=28,
        radius1=0.099, radius2=0.128, depth=0.048,
        matrix=Matrix.Translation((hc.x, cy + 0.004, cz + 0.042))
        @ Euler((math.radians(-10.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((0.94, 1.12, 1.0, 1.0)),
    )
    vc = set(res_cr["verts"])
    for f in bm.faces:
        if all(v in vc for v in f.verts):
            f.material_index = 0

    # Glossy black front visor brim projecting forward (-Y!)
    _add_box_bm(
        bm, (hc.x, cy - 0.108, cz - 0.018), (0.148, 0.088, 0.010),
        mat_index=1, bevel_w=0.004, rot_deg=(18.0, 0.0, 0.0),
    )
    # Gold officer insignia badge on front crown
    _add_box_bm(bm, (hc.x, cy - 0.105, cz + 0.032), (0.032, 0.010, 0.030), mat_index=2, bevel_w=0.002)

    cap_obj = _new_bmesh_obj("Char_Commander_Cap", bm, [mat_cap, mat_brim, mat_gold], Matrix.Identity(4), col, smooth=True)
    _parent_to_bone_preserve_world(cap_obj, rig, "head")

    # Shoulder epaulets parented to spine03
    bpy.context.view_layer.update()
    sp3_w = rig.matrix_world @ rig.pose.bones["spine03"].head
    bm_ep = bmesh.new()
    for sx in (-1.0, +1.0):
        _add_box_bm(
            bm_ep, (sp3_w.x + sx * 0.165, sp3_w.y + 0.010, sp3_w.z + 0.155), (0.11, 0.056, 0.014),
            mat_index=0, bevel_w=0.003, rot_deg=(0.0, sx * 12.0, 0.0),
        )
        for dy in (-0.014, +0.014):
            _add_box_bm(
                bm_ep, (sp3_w.x + sx * 0.190, sp3_w.y + 0.010 + dy, sp3_w.z + 0.163), (0.016, 0.014, 0.007),
                mat_index=2, bevel_w=0.001, rot_deg=(0.0, sx * 12.0, 0.0),
            )
    ep_obj = _new_bmesh_obj("Char_Commander_Epaulets", bm_ep, [mat_cap, mat_brim, mat_gold], Matrix.Identity(4), col, smooth=False)
    _parent_to_bone_preserve_world(ep_obj, rig, "spine03")
    return [cap_obj, ep_obj]


def _build_pilot_helmet_and_harness(rig: bpy.types.Object, col: bpy.types.Collection) -> list[bpy.types.Object]:
    """
    Wearable Matte-Black USAF HGU-55 Stealth Flight Helmet + Reflective Gold-Tinted Sun Visor
    + Tactical Oxygen Mask & Corrugated Breathing Hose (`head` bone) + Ejection Harness (`spine03`).
    """
    mat_shell = _make_mat("Mat_Pilot_HelmetShell", (0.036, 0.042, 0.050), metallic=0.35, roughness=0.28, coat=0.35)
    mat_visor = _make_mat("Mat_Pilot_GoldVisor", (0.88, 0.64, 0.12), metallic=0.96, roughness=0.06, coat=0.80)
    mat_mask = _make_mat("Mat_Pilot_O2Mask", (0.048, 0.058, 0.052), metallic=0.14, roughness=0.55)
    mat_strap = _make_mat("Mat_Pilot_Harness", (0.085, 0.105, 0.082), metallic=0.08, roughness=0.65)
    mat_buckle = _make_mat("Mat_Pilot_Buckle", (0.48, 0.52, 0.58), metallic=0.92, roughness=0.24)

    hc = _head_world_center(rig)
    cz = hc.z + 0.095
    cy = hc.y + 0.010
    bm = bmesh.new()

    # Sleek spherical fighter helmet dome enclosing crown, back & sides (open at face -Y!)
    res_dom = bmesh.ops.create_icosphere(
        bm, subdivisions=3, radius=1.0,
        matrix=Matrix.Translation((hc.x, cy + 0.012, cz + 0.012)) @ Matrix.Diagonal((0.118, 0.126, 0.124, 1.0)),
    )
    dom_vs = set(res_dom["verts"])
    for f in list(bm.faces):
        if all(v in dom_vs for v in f.verts):
            fc = f.calc_center_median()
            # Carve out front face opening so gold visor & oxygen mask sit cleanly in front!
            if fc.y < cy - 0.060 and fc.z < cz + 0.068:
                bm.faces.remove(f)
            else:
                f.material_index = 0

    # Raised top visor housing brow ridge
    _add_box_bm(bm, (hc.x, cy - 0.038, cz + 0.104), (0.150, 0.120, 0.034), mat_index=0, bevel_w=0.012, rot_deg=(12.0, 0.0, 0.0))

    # Gleaming Curved Reflective Gold Sun Visor sweeping across pilot's eyes!
    res_vis = bmesh.ops.create_icosphere(
        bm, subdivisions=3, radius=1.0,
        matrix=Matrix.Translation((hc.x, cy - 0.008, cz + 0.012)) @ Matrix.Diagonal((0.112, 0.114, 0.088, 1.0)),
    )
    vis_vs = set(res_vis["verts"])
    for f in list(bm.faces):
        if all(v in vis_vs for v in f.verts):
            fc = f.calc_center_median()
            if fc.y < cy - 0.052 and (cz - 0.022 < fc.z < cz + 0.068):
                f.material_index = 1
            else:
                bm.faces.remove(f)

    # Sculpted USAF MBU-20/P Tactical Fighter Oxygen Mask covering nose & chin (-Y)
    res_m = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=20,
        radius1=0.052, radius2=0.034, depth=0.074,
        matrix=Matrix.Translation((hc.x, cy - 0.118, cz - 0.044))
        @ Euler((math.radians(74.0), 0.0, 0.0), "XYZ").to_matrix().to_4x4(),
    )
    m_vs = set(res_m["verts"])
    for f in bm.faces:
        if all(v in m_vs for v in f.verts):
            f.material_index = 2

    # Corrugated breathing hose looping down from oxygen mask to pilot's chest
    hose_pts = [
        Vector((hc.x, cy - 0.146, cz - 0.068)),
        Vector((hc.x + 0.028, cy - 0.168, cz - 0.118)),
        Vector((hc.x + 0.056, cy - 0.172, cz - 0.185)),
        Vector((hc.x + 0.072, cy - 0.158, cz - 0.255)),
    ]
    for p0, p1 in zip(hose_pts, hose_pts[1:]):
        _add_strut_bm(bm, p0, p1, 0.014, mat_index=2, segs=12)

    helm_obj = _new_bmesh_obj(
        "Char_Pilot_HelmetVisorMask", bm,
        [mat_shell, mat_visor, mat_mask], Matrix.Identity(4), col, smooth=True,
    )
    _parent_to_bone_preserve_world(helm_obj, rig, "head")

    # Pilot Ejection-Seat Tactical Harness Straps & Chest Quick-Release Buckle (`spine03`)
    bpy.context.view_layer.update()
    sp3_w = rig.matrix_world @ rig.pose.bones["spine03"].head
    bm_h = bmesh.new()
    for sx in (-1.0, +1.0):
        _add_box_bm(
            bm_h, (sp3_w.x + sx * 0.105, sp3_w.y - 0.138, sp3_w.z + 0.04), (0.046, 0.020, 0.34),
            mat_index=0, bevel_w=0.004, rot_deg=(-8.0, 0.0, 0.0),
        )
    # Central machined quick-release chest buckle
    _add_box_bm(bm_h, (sp3_w.x, sp3_w.y - 0.152, sp3_w.z - 0.04), (0.088, 0.022, 0.072), mat_index=1, bevel_w=0.005)
    har_obj = _new_bmesh_obj(
        "Char_Pilot_FlightHarness", bm_h, [mat_strap, mat_buckle], Matrix.Identity(4), col, smooth=False
    )
    _parent_to_bone_preserve_world(har_obj, rig, "spine03")
    return [helm_obj, har_obj]


# ==============================================================================
# 1:1 SCALE B-2 SPIRIT COCKPIT HERO SET (BEAT 5A: f=1321..1430)
# ==============================================================================
def _build_pilot_cockpit_set(col: bpy.types.Collection, total_frames: int) -> list[bpy.types.Object]:
    """
    Build a sleek 1:1 scale B-2 Spirit Stealth Bomber Flight Deck surround around `Stealth_Pilot`
    for Beat 5A (`1321..1430`):
    - High-back ACES-II military ejection seat with yellow/black pull handle
    - Matte-black angled stealth glare shield + dual glowing tactical MFDs (radar & stealth status)
    - Translucent glowing emerald-cyan Head-Up Display (HUD) combiner glass in front of pilot
    - Swept B-2 4-pane matte-charcoal armored canopy arches framing sunlit azure sky behind pilot
    - Animated flight control stick tilting cleanly with the pilot's banking input!
    """
    mat_dark = _make_mat("Mat_Cockpit_Carbon", (0.028, 0.034, 0.042), metallic=0.55, roughness=0.40)
    mat_seat = _make_mat("Mat_Cockpit_SeatPad", (0.042, 0.048, 0.056), metallic=0.08, roughness=0.68)
    mat_yel = _make_mat("Mat_Cockpit_EjectYel", (0.98, 0.72, 0.06), metallic=0.15, roughness=0.32, emit_rgb=(0.98, 0.72, 0.06), emit_str=0.35)
    mat_mfd = _make_mat("Mat_Cockpit_MFD", (0.02, 0.20, 0.14), metallic=0.10, roughness=0.24, emit_rgb=(0.06, 0.94, 0.62), emit_str=1.55)
    mat_hud_frame = _make_mat("Mat_Cockpit_HUDFrame", (0.14, 0.17, 0.21), metallic=0.88, roughness=0.25)

    # Translucent glowing emerald-cyan HUD combiner glass (AddShader, AgX-safe, shadow-bypassed!)
    mat_hud = bpy.data.materials.new("Mat_Cockpit_HUDGlass")
    mat_hud.use_nodes = True
    nt = mat_hud.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 14.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.72
    ramp.color_ramp.elements[0].color = (0.02, 0.26, 0.22, 1.0)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (0.10, 0.98, 0.72, 1.0)
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    mul_c = nt.nodes.new("ShaderNodeMath")
    mul_c.operation = "MULTIPLY"
    mul_c.inputs[1].default_value = 0.55
    nt.links.new(lp.outputs["Is Camera Ray"], mul_c.inputs[0])
    nt.links.new(mul_c.outputs[0], emit.inputs["Strength"])
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])

    # Cockpit world origin aligned around pilot seated at (0.0, 0.14, 0.42), eyes at Z=2.04
    bm = bmesh.new()
    # ACES-II Ejection seat backrest & contoured headrest behind pilot (+Y)
    _add_box_bm(bm, (0.0, 0.36, 1.82), (0.48, 0.10, 0.78), mat_index=0, bevel_w=0.015, rot_deg=(-8.0, 0.0, 0.0))
    _add_box_bm(bm, (0.0, 0.31, 1.84), (0.38, 0.05, 0.62), mat_index=1, bevel_w=0.012, rot_deg=(-8.0, 0.0, 0.0))
    _add_box_bm(bm, (0.0, 0.33, 2.26), (0.28, 0.12, 0.20), mat_index=0, bevel_w=0.014)
    # Yellow/black ejection-seat top warning rail accents
    for sx in (-0.17, +0.17):
        _add_box_bm(bm, (sx, 0.31, 2.24), (0.032, 0.06, 0.16), mat_index=2, bevel_w=0.004)

    # Front stealth glare-shield dashboard (-Y in front of pilot, low enough to never hide helmet/visor!)
    _add_box_bm(bm, (0.0, -0.36, 1.56), (1.18, 0.36, 0.18), mat_index=0, bevel_w=0.016, rot_deg=(10.0, 0.0, 0.0))
    # Dual glowing green MFDs recessed in dashboard
    for sx in (-0.28, +0.28):
        _add_box_bm(bm, (sx, -0.22, 1.54), (0.22, 0.02, 0.16), mat_index=3, rot_deg=(-22.0, 0.0, 0.0))
    # Angled B-2 canopy side-sills & swept armored canopy frame pillars (framing wide open sky behind!)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sx * 0.56, 0.04, 1.56), (0.14, 1.10, 0.16), mat_index=0, bevel_w=0.014)
        _add_strut_bm(bm, Vector((sx * 0.54, -0.34, 1.64)), Vector((sx * 0.42, 0.28, 2.52)), 0.032, mat_index=0, segs=10)
        _add_strut_bm(bm, Vector((sx * 0.54, 0.48, 1.64)), Vector((sx * 0.42, 0.28, 2.52)), 0.028, mat_index=0, segs=10)
    # Overhead B-2 canopy crown arch
    _add_strut_bm(bm, Vector((-0.42, 0.28, 2.52)), Vector((+0.42, 0.28, 2.52)), 0.032, mat_index=0, segs=10)

    # HUD combiner support rails
    for sx in (-0.13, +0.13):
        _add_strut_bm(bm, Vector((sx, -0.32, 1.65)), Vector((sx, -0.28, 1.88)), 0.008, mat_index=4, segs=8)

    cockpit_obj = _new_bmesh_obj(
        "Char_B2_CockpitFrame", bm,
        [mat_dark, mat_seat, mat_yel, mat_mfd, mat_hud_frame],
        Matrix.Identity(4), col, smooth=False,
    )
    _key_visibility_windows([cockpit_obj], [(1321, C.B5_COCKPIT_END)], total_frames)

    # Translucent glowing HUD glass pane
    bm_hud = bmesh.new()
    _add_box_bm(bm_hud, (0.0, -0.30, 1.78), (0.25, 0.006, 0.20), mat_index=0, bevel_w=0.001, rot_deg=(24.0, 0.0, 0.0))
    hud_obj = _new_bmesh_obj(
        "Char_B2_CockpitHUD", bm_hud, [mat_hud], Matrix.Identity(4), col, smooth=False, no_shadow=True
    )
    _key_visibility_windows([hud_obj], [(1321, C.B5_COCKPIT_END)], total_frames)

    # Animated flight control stick between pilot's knees
    bm_st = bmesh.new()
    _add_strut_bm(bm_st, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.02, 0.26)), 0.016, mat_index=0, segs=12)
    _add_box_bm(bm_st, (0.0, 0.02, 0.28), (0.044, 0.056, 0.092), mat_index=0, bevel_w=0.008)
    _add_box_bm(bm_st, (0.0, 0.00, 0.31), (0.022, 0.022, 0.020), mat_index=2, bevel_w=0.002)
    stick_obj = _new_bmesh_obj(
        "Char_B2_FlightStick", bm_st, [mat_dark, mat_seat, mat_yel],
        Matrix.Translation((0.0, -0.14, 1.32)), col, smooth=True,
    )
    _key_visibility_windows([stick_obj], [(1321, C.B5_COCKPIT_END)], total_frames)
    for f in range(1321, C.B5_COCKPIT_END + 1, 6):
        u = C.smootherstep(1321.0, float(C.B5_COCKPIT_END), float(f))
        bank_rad = math.radians(14.0 * math.sin(u * math.pi * 1.3))
        stick_obj.rotation_euler = Euler((math.radians(-6.0), bank_rad, 0.0), "XYZ")
        stick_obj.keyframe_insert("rotation_euler", frame=f)

    return [cockpit_obj, hud_obj, stick_obj]


# ==============================================================================
# CHARACTER BUILDERS (1: RADAR OPERATOR, 2: DEFENSE COMMANDER, 3: STEALTH PILOT)
# ==============================================================================
def _build_radar_operator(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    cfg = {
        "gender": 1.0,
        "age": 0.44,
        "muscle": 0.56,
        "weight": 0.44,
        "height": 0.55,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(C.CREW["operator_height_m"]),
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.08, 0.07, 0.06),
        "hair_tint_fac": 0.72,
        "hair_value_boost": 0.70,
        "hair_roughness": 0.30,
        "brow_tint_fac": 0.62,
        "iris_major_rgb": (0.18, 0.42, 0.62),  # Crisp steel-blue eyes
        "iris_minor_rgb": (0.07, 0.18, 0.30),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.08, 0.11, 0.09),  # Dark air-defense olive-charcoal uniform
        "shirt_tint_fac": 0.86,
        "shirt_sat_override": 0.28,
        "shirt_value_boost": 0.76,
        "shirt_roughness": 0.52,
        "shirt_sheen": 0.10,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.05, 0.06, 0.06),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.20,
        "pants_value_boost": 0.68,
        "pants_roughness": 0.54,
        "pants_sheen": 0.08,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.04, 0.04, 0.045),
        "shoe_tint_fac": 0.84,
        "shoe_sat_override": 0.12,
        "shoe_value_boost": 0.58,
        "shoe_roughness": 0.42,
        "shoe_sheen": 0.06,
    }
    res = mc.create_realistic_scientist(name="Radar_Operator", floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig, body, equipped = res["rig"], res["body"], res["equipped"]
    base_z = float(res["base_z"])

    _link_obj(rig, col)
    _link_obj(body, col)
    for o in equipped.values():
        if o is not None:
            _link_obj(o, col)
    _enforce_v12_eyes(equipped)

    rig.location = Vector((0.0, 0.0, base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.view_layer.update()

    _apply_underclothes_shrink(body, rig)
    headset_obj = _build_comms_headset(rig, col)

    kfs: set[int] = {1, 220, 1561, total_frames}
    for f in range(1, 221, 8):
        kfs.add(f)
    for f in range(1561, total_frames + 1, 8):
        kfs.add(f)

    for f in sorted(kfs):
        ff = float(f)
        if f <= 220:
            # Beat 1: Frantically scanning blank scope -> jerking head up toward the giant B-2 outside!
            ox, oy, oz_off = C.CREW["operator_b1_pos"]
            yaw = float(C.CREW["operator_b1_yaw_deg"])
            u = C.smootherstep(1.0, 220.0, ff)
            tap = math.sin(ff * 0.32)
            look_up = C.smootherstep(45.0, 115.0, ff) * (1.0 - 0.55 * C.smootherstep(145.0, 195.0, ff))
            rig.location = Vector((ox, oy, base_z + oz_off))
            rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw)), "XYZ")
            _pose_spine_head(
                rig,
                torso_fwd_deg=14.0 - 6.0 * look_up,
                torso_twist_deg=C.lerp(12.0, -8.0, u),
                head_pitch_deg=C.lerp(18.0, -24.0, look_up),
                head_yaw_deg=C.lerp(-14.0, 18.0, look_up),
                frame=f,
            )
            _pose_arms_fk(
                rig,
                ua_l=(0.0, 24.0, -56.0 + 6.0 * tap),
                la_l=(0.0, -26.0, -24.0),
                ua_r=(0.0, -28.0, 48.0 - 5.0 * tap),
                la_r=(0.0, 22.0, 28.0),
                hand_l="point" if look_up < 0.4 else "shock",
                hand_r="relaxed",
                frame=f,
            )
            _apply_actor_face(rig, f=f, jaw_deg=2.2 + 5.5 * look_up, brow_lift=0.35 + 0.45 * look_up, smile=-0.12, eye_yaw_deg=6.0, eye_pitch_deg=10.0 * look_up, force_alert_lids=True)
        else:
            # Beat 6: Leans closely over central PPI radar scope -> startled double-take up at overhead B-2!
            ox, oy, oz_off = C.CREW["operator_b6_pos"]
            yaw = float(C.CREW["operator_b6_yaw_deg"])
            u = C.smootherstep(1561.0, float(total_frames), ff)
            shock_up = C.smootherstep(1680.0, 1750.0, ff)
            rig.location = Vector((ox, oy, base_z + oz_off))
            rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw)), "XYZ")
            _pose_spine_head(
                rig,
                torso_fwd_deg=18.0 - 14.0 * shock_up,
                torso_twist_deg=C.lerp(10.0, -6.0, u),
                head_pitch_deg=C.lerp(22.0, -28.0, shock_up),
                head_yaw_deg=C.lerp(-16.0, 12.0, shock_up),
                frame=f,
            )
            _pose_arms_fk(
                rig,
                ua_l=(0.0, 22.0, -60.0),
                la_l=(0.0, -32.0, -18.0),
                ua_r=(0.0, -26.0, 46.0),
                la_r=(0.0, 20.0, 28.0),
                hand_l="point" if shock_up < 0.5 else "shock",
                hand_r="shock" if shock_up > 0.5 else "relaxed",
                frame=f,
            )
            _apply_actor_face(rig, f=f, jaw_deg=1.8 + 6.8 * shock_up, brow_lift=0.30 + 0.60 * shock_up, smile=-0.18 * shock_up, eye_yaw_deg=0.0, eye_pitch_deg=14.0 * shock_up, force_alert_lids=True)

        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

    _set_fcurve_interp(rig, interp="BEZIER", constant_frames={220})
    vis_objs = [rig, body, headset_obj] + [o for o in equipped.values() if o is not None]
    _key_visibility_windows(vis_objs, [(1, 220), (1561, total_frames)], total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "headset": headset_obj}


def _build_defense_commander(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    cfg = {
        "gender": 1.0,
        "age": 0.66,
        "muscle": 0.62,
        "weight": 0.48,
        "height": 0.62,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(C.CREW["commander_height_m"]),
        "skin_assets": ["old_caucasian_male.mhmat", "middleage_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.28, 0.28, 0.30),  # Silver-grey seasoned commander hair
        "hair_tint_fac": 0.72,
        "hair_value_boost": 0.85,
        "hair_roughness": 0.34,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": (0.44, 0.32, 0.14),  # Independent warm amber-hazel eyes
        "iris_minor_rgb": (0.18, 0.12, 0.05),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.06, 0.08, 0.07),  # Deep tactical dark-olive commander tunic
        "shirt_tint_fac": 0.88,
        "shirt_sat_override": 0.25,
        "shirt_value_boost": 0.72,
        "shirt_roughness": 0.48,
        "shirt_sheen": 0.12,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.045, 0.055, 0.052),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.18,
        "pants_value_boost": 0.66,
        "pants_roughness": 0.50,
        "pants_sheen": 0.08,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.035, 0.035, 0.040),
        "shoe_tint_fac": 0.84,
        "shoe_sat_override": 0.10,
        "shoe_value_boost": 0.56,
        "shoe_roughness": 0.38,
        "shoe_sheen": 0.06,
    }
    res = mc.create_realistic_scientist(name="Defense_Commander", floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig, body, equipped = res["rig"], res["body"], res["equipped"]
    base_z = float(res["base_z"])

    _link_obj(rig, col)
    _link_obj(body, col)
    for o in equipped.values():
        if o is not None:
            _link_obj(o, col)
    _enforce_v12_eyes(equipped)

    rig.location = Vector((0.0, 0.0, base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.view_layer.update()

    _apply_underclothes_shrink(body, rig)
    gear_objs = _build_officer_cap_and_epaulets(rig, col)

    kfs: set[int] = {1, 220, 1561, total_frames}
    for f in range(1, 221, 8):
        kfs.add(f)
    for f in range(1561, total_frames + 1, 8):
        kfs.add(f)

    for f in sorted(kfs):
        ff = float(f)
        if f <= 220:
            # Beat 1: Commander raises right arm pointing urgently out the panoramic window at the B-2!
            cx, cy, cz_off = C.CREW["commander_b1_pos"]
            yaw = float(C.CREW["commander_b1_yaw_deg"])
            u = C.smootherstep(1.0, 220.0, ff)
            pt_up = C.smootherstep(10.0, 65.0, ff)
            rig.location = Vector((cx, cy, base_z + cz_off))
            rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw)), "XYZ")
            _pose_spine_head(
                rig,
                torso_fwd_deg=6.0 + 4.0 * pt_up,
                torso_twist_deg=C.lerp(-8.0, 10.0, u),
                head_pitch_deg=C.lerp(-18.0, -10.0, u),
                head_yaw_deg=C.lerp(14.0, -8.0, u),
                frame=f,
            )
            # Right arm raised dramatically pointing toward sunlit sky window!
            _pose_arms_fk(
                rig,
                ua_l=(0.0, 44.0, -14.0),
                la_l=(0.0, 12.0, -16.0),
                ua_r=(0.0, -18.0 * pt_up - 32.0 * (1.0 - pt_up), 68.0 * pt_up + 12.0),
                la_r=(0.0, 10.0, 18.0 * pt_up),
                hand_l="relaxed",
                hand_r="point",
                frame=f,
            )
            _apply_actor_face(rig, f=f, jaw_deg=3.5 + 4.5 * pt_up, brow_lift=0.42, smile=-0.10, eye_yaw_deg=4.0, eye_pitch_deg=8.0, force_alert_lids=True)
        else:
            # Beat 6: Commander leans over radar scope squinting at tiny blip -> snaps head up at overhead bomber!
            cx, cy, cz_off = C.CREW["commander_b6_pos"]
            yaw = float(C.CREW["commander_b6_yaw_deg"])
            shock_up = C.smootherstep(1665.0, 1745.0, ff)
            rig.location = Vector((cx, cy, base_z + cz_off))
            rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw)), "XYZ")
            _pose_spine_head(
                rig,
                torso_fwd_deg=16.0 - 12.0 * shock_up,
                torso_twist_deg=C.lerp(-12.0, 8.0, shock_up),
                head_pitch_deg=C.lerp(20.0, -26.0, shock_up),
                head_yaw_deg=C.lerp(18.0, -10.0, shock_up),
                frame=f,
            )
            _pose_arms_fk(
                rig,
                ua_l=(0.0, 38.0, -26.0),
                la_l=(0.0, -18.0, -22.0),
                ua_r=(0.0, -24.0, 44.0 + 26.0 * shock_up),
                la_r=(0.0, 16.0, 24.0),
                hand_l="relaxed",
                hand_r="shock" if shock_up > 0.45 else "point",
                frame=f,
            )
            _apply_actor_face(rig, f=f, jaw_deg=2.0 + 6.5 * shock_up, brow_lift=0.25 + 0.65 * shock_up, smile=-0.15, eye_yaw_deg=0.0, eye_pitch_deg=12.0 * shock_up, force_alert_lids=True)

        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

    _set_fcurve_interp(rig, interp="BEZIER", constant_frames={220})
    vis_objs = [rig, body] + gear_objs + [o for o in equipped.values() if o is not None]
    _key_visibility_windows(vis_objs, [(1, 220), (1561, total_frames)], total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "gear": gear_objs}


def _build_stealth_pilot(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """Beat 5A (1321..1430): 4K MPFB2 USAF B-2 Stealth Pilot inside 1:1 Scale B-2 Cockpit!"""
    cfg = {
        "gender": 1.0,
        "age": 0.48,
        "muscle": 0.64,
        "weight": 0.45,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(C.CREW["pilot_height_m"]),
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.06, 0.05, 0.04),
        "hair_tint_fac": 0.80,
        "hair_value_boost": 0.60,
        "hair_roughness": 0.30,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": (0.14, 0.34, 0.54),
        "iris_minor_rgb": (0.06, 0.14, 0.24),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.11, 0.14, 0.10),  # USAF sage-green Nomex flight suit
        "shirt_tint_fac": 0.88,
        "shirt_sat_override": 0.32,
        "shirt_value_boost": 0.78,
        "shirt_roughness": 0.50,
        "shirt_sheen": 0.10,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.10, 0.13, 0.09),
        "pants_tint_fac": 0.88,
        "pants_sat_override": 0.30,
        "pants_value_boost": 0.74,
        "pants_roughness": 0.52,
        "pants_sheen": 0.08,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.04, 0.04, 0.045),
        "shoe_tint_fac": 0.84,
        "shoe_sat_override": 0.10,
        "shoe_value_boost": 0.58,
        "shoe_roughness": 0.40,
        "shoe_sheen": 0.06,
    }
    res = mc.create_realistic_scientist(name="Stealth_Pilot", floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig, body, equipped = res["rig"], res["body"], res["equipped"]
    base_z = float(res["base_z"])

    _link_obj(rig, col)
    _link_obj(body, col)
    for o in equipped.values():
        if o is not None:
            _link_obj(o, col)
    _enforce_v12_eyes(equipped)

    rig.location = Vector((0.0, 0.0, base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.view_layer.update()

    _apply_underclothes_shrink(body, rig)
    gear_objs = _build_pilot_helmet_and_harness(rig, col)
    cockpit_objs = _build_pilot_cockpit_set(col, total_frames)

    px, py, pz = C.CREW["pilot_pos"]
    yaw = float(C.CREW["pilot_yaw_deg"])

    for f in range(1321, C.B5_COCKPIT_END + 1, 6):
        u = C.smootherstep(1321.0, float(C.B5_COCKPIT_END), float(f))
        bank_sway = math.sin(u * math.pi * 1.3)
        rig.location = Vector((px, py, base_z + pz))
        rig.rotation_euler = Euler((math.radians(-4.0), math.radians(5.0 * bank_sway), math.radians(yaw)), "XYZ")
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

        # Calm veteran stealth pilot scanning HUD & banking smoothly right past SAM radar!
        _pose_spine_head(
            rig,
            torso_fwd_deg=4.0,
            torso_twist_deg=4.0 * bank_sway,
            head_pitch_deg=-3.0 + 3.0 * u,
            head_yaw_deg=C.lerp(-12.0, +10.0, u),
            frame=f,
        )
        # Right hand gripping center control stick, left hand on stealth throttle quadrant
        _pose_arms_fk(
            rig,
            ua_l=(0.0, 26.0, -52.0),
            la_l=(0.0, -34.0, -28.0),
            ua_r=(0.0, -22.0 + 5.0 * bank_sway, 56.0),
            la_r=(-6.0, 28.0, 38.0),
            hand_l="relaxed",
            hand_r="relaxed",
            frame=f,
        )
        _apply_actor_face(rig, f=f, jaw_deg=0.5, brow_lift=0.0, smile=0.08, eye_yaw_deg=0.0, eye_pitch_deg=0.0, force_alert_lids=True)

    _set_fcurve_interp(rig, interp="BEZIER")
    vis_objs = [rig, body] + gear_objs + [o for o in equipped.values() if o is not None]
    _key_visibility_windows(vis_objs, [(1321, C.B5_COCKPIT_END)], total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "gear": gear_objs, "cockpit": cockpit_objs}


def build_characters(
    col: bpy.types.Collection,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build & animate all 3 photoreal 4K MPFB2 characters + 1:1 B-2 Cockpit Set."""
    operator = _build_radar_operator(col, total_frames)
    commander = _build_defense_commander(col, total_frames)
    pilot = _build_stealth_pilot(col, total_frames)
    return {"operator": operator, "commander": commander, "pilot": pilot}
