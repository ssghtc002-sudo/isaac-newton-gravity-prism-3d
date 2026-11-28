"""
Swimmer Character Animator for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (30.0 s @ 60 fps = 1800 frames).

Public API:
    build_swimmer(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]

Returned dictionary keys & types:
    - "rig":      bpy.types.Object  (Swimmer_Rig MPFB2 163-bone armature)
    - "body":     bpy.types.Object  (Swimmer_Body 1.78m adult male basemesh)
    - "equipped": dict[str, bpy.types.Object]  (Eyes, Eyebrows, Eyelashes, Teeth, Tongue,
                                                Hair, Pants, Shirt, Shoes)
    - "cuffs":    list[bpy.types.Object]  (Fitted neoprene wrist cuffs; zero collar bib)
    - "height":   float
    - "base_z":   float
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_ROOT_SRC = _REPO_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_ROOT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bullet_contract as C  # noqa: E402
import mpfb_characters as mc  # noqa: E402

SWIMMER_HEIGHT_M: float = float(C.SWIMMER["height_m"])
_HERO_BLINK_EXCLUSION_FRAMES: tuple[int, ...] = (
    1, 60, 118, 135, 150, 165, 190, 210, 1561, 1620, 1680, 1740, 1800
)


# ============================================================================
# COLLECTION, MESH, HIERARCHY & FCURVE HELPERS (BLENDER 5.3.0 SAFE)
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _set_smooth(mesh: bpy.types.Mesh) -> None:
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()


def _set_fcurve_interp(
    obj_or_data: Any,
    interp: str = "BEZIER",
    skip_prefix: str = "hide_",
    constant_frames: set[int] | None = None,
) -> None:
    """Set FCurve interpolation safely across Blender 4.x and 5.3.0 Slotted Actions."""
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
            kf_int = int(round(kp.co.x))
            if kf_int in const_set:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = interp
                if interp == "BEZIER":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"


def _keyframe_visibility(objs: list[bpy.types.Object], total_frames: int) -> None:
    """Show swimmer ONLY in Beat 1 (1..210) and Beat 6 (1561..total_frames) via CONSTANT hide_*."""
    vis_schedule: list[tuple[int, bool]] = [(1, False)]
    if total_frames >= 211:
        vis_schedule.append((211, True))
    if total_frames >= 1561:
        vis_schedule.append((1561, False))

    for obj in objs:
        if obj is None:
            continue
        for f, hide_val in vis_schedule:
            obj.hide_viewport = hide_val
            obj.hide_render = hide_val
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        ad = getattr(obj, "animation_data", None)
        if ad and ad.action:
            for fc in C.iter_action_fcurves(ad.action):
                if fc.data_path in ("hide_viewport", "hide_render"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"


def _parent_to_bone_preserve_world(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    """Parent `obj` to `rig`'s pose bone `bone_name` preserving its exact current world transform."""
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


def _eval_mesh_verts_world(obj: bpy.types.Object) -> list[Vector]:
    """Evaluate deformed mesh vertex positions in world space."""
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    em = eo.to_mesh()
    xf = eo.matrix_world
    pts = [xf @ v.co for v in em.vertices]
    eo.to_mesh_clear()
    return pts


def _eval_mesh_bounds_world(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    pts = _eval_mesh_verts_world(obj)
    if not pts:
        z = Vector((0.0, 0.0, 0.0))
        return z, z
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material | list[bpy.types.Material],
    world_xf: Matrix,
    col: bpy.types.Collection,
    smooth: bool = True,
    recalc_normals: bool = True,
) -> bpy.types.Object:
    """Finalize `bm` into a named mesh object with material slot(s) attached BEFORE `bm.to_mesh()`."""
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    mats = mat if isinstance(mat, list) else [mat]
    for m in mats:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        _set_smooth(mesh)
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = world_xf.copy()
    _link_obj(obj, col)
    return obj


# ============================================================================
# UNDERCLOTHES SHRINK & SLEEK NEOPRENE WRIST CUFFS (ZERO FLOATING COLLAR BIB)
# ============================================================================
def _apply_underclothes_shrink(body: bpy.types.Object, rig: bpy.types.Object) -> None:
    """Apply restricted UnderClothes_Shrink (-0.011 DISPLACE BEFORE ARMATURE) including
    upperarm & lowerarm, while excluding head, neck, wrist, hand, finger, thumb, lowerleg, foot, toe."""
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

    excluded_vids: set[int] = {vid for vid, ew in excl_weights.items() if ew > 0.05}
    vg_shrink = body.vertex_groups.get("UnderClothes_Shrink") or body.vertex_groups.new(name="UnderClothes_Shrink")
    taper = 0.045 * scale_h
    weights: list[float] = [0.0] * len(body.data.vertices)
    for v, p in zip(body.data.vertices, pts):
        if v.index in excluded_vids:
            w = 0.0
        elif abs(p.x) >= 0.85 * scale_h or p.z < z_lo or p.z > z_hi:
            w = 0.0
        else:
            wx = 1.0 - C.smoothstep(0.75 * scale_h, 0.85 * scale_h, abs(p.x))
            wz_lo = C.smoothstep(z_lo, z_lo + taper, p.z)
            wz_hi = 1.0 - C.smoothstep(z_hi - taper, z_hi, p.z)
            w_excl = 1.0 - C.smoothstep(0.0, 0.05, excl_weights.get(v.index, 0.0))
            w = wx * wz_lo * wz_hi * w_excl
        weights[v.index] = w

    adj: list[list[int]] = [[] for _ in range(len(body.data.vertices))]
    for edge in body.data.edges:
        a, b = edge.vertices[0], edge.vertices[1]
        adj[a].append(b)
        adj[b].append(a)
    for _ in range(2):
        next_w = list(weights)
        for vid, nbrs in enumerate(adj):
            if vid in excluded_vids or not nbrs:
                next_w[vid] = 0.0
                continue
            if any(nb in excluded_vids for nb in nbrs) or weights[vid] < 0.99:
                avg_nb = sum(weights[nb] for nb in nbrs) / float(len(nbrs))
                next_w[vid] = 0.5 * weights[vid] + 0.5 * avg_nb
        weights = next_w

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
        if cur_idx > arm_idx:
            if hasattr(body.modifiers, "move"):
                body.modifiers.move(cur_idx, arm_idx)
            else:
                with bpy.context.temp_override(object=body, active_object=body):
                    while list(body.modifiers).index(mod) > arm_idx:
                        bpy.ops.object.modifier_move_up(modifier=mod.name)


def _filter_segment_pts(
    pts: list[Vector],
    a: Vector,
    b: Vector,
    max_perp: float,
) -> list[Vector]:
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-8:
        return []
    out: list[Vector] = []
    for p in pts:
        ap = p - a
        t = ap.dot(ab) / len_sq
        if -0.08 <= t <= 1.08:
            perp = (ap - ab * t).length
            if perp <= max_perp:
                out.append(p)
    return out


def _add_sleeve_band_from_verts(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    a: Vector,
    b: Vector,
    t_range: tuple[float, float],
    pad_m: float = 0.004,
    thick_m: float = 0.0028,
    n_rings: int = 3,
    n_seg: int = 18,
    mat_index: int = 0,
    min_r: float = 0.025,
    max_r: float = 0.036,
) -> None:
    """Create a sleek closed tubular neoprene wrist cuff fitted around segment `a -> b`."""
    axis = b - a
    if axis.length < 1e-5:
        return
    u_ax = axis.normalized()
    ref = Vector((0.0, 1.0, 0.0)) if abs(u_ax.y) < 0.85 else Vector((1.0, 0.0, 0.0))
    e1 = u_ax.cross(ref).normalized()
    e2 = u_ax.cross(e1).normalized()

    t0, t1 = t_range
    outer_rings: list[list[bmesh.types.BMVert]] = []
    inner_rings: list[list[bmesh.types.BMVert]] = []

    for r_i in range(max(2, n_rings)):
        t = C.lerp(t0, t1, r_i / float(max(1, n_rings - 1)))
        center = a + axis * t
        half_band = max(0.015, abs(t1 - t0) / float(max(1, n_rings - 1))) * 0.75
        ring_pts = [
            p for p in pts
            if abs(((p - a).dot(axis) / axis.length_squared) - t) <= half_band
        ]

        o_ring: list[bmesh.types.BMVert] = []
        i_ring: list[bmesh.types.BMVert] = []
        for s_i in range(n_seg):
            ang = (s_i / float(n_seg)) * math.tau
            rad_dir = e1 * math.cos(ang) + e2 * math.sin(ang)
            ang_cands: list[float] = []
            for p in ring_pts:
                dp = p - center
                dp_perp = dp - u_ax * dp.dot(u_ax)
                plen = dp_perp.length
                if plen > 1e-5 and (dp_perp / plen).dot(rad_dir) > 0.78:
                    ang_cands.append(plen)
            if ang_cands:
                r_base = max(ang_cands)
            elif ring_pts:
                r_base = sum(
                    ((p - center) - u_ax * (p - center).dot(u_ax)).length for p in ring_pts
                ) / float(len(ring_pts))
            else:
                r_base = 0.5 * (min_r + max_r)
            r_out = max(min_r, min(max_r, r_base + pad_m + thick_m))
            r_in = max(min_r * 0.82, r_out - thick_m)
            o_ring.append(bm.verts.new(center + rad_dir * r_out))
            i_ring.append(bm.verts.new(center + rad_dir * r_in))
        outer_rings.append(o_ring)
        inner_rings.append(i_ring)

    def _mk_face(vs: tuple[bmesh.types.BMVert, ...]) -> None:
        try:
            f = bm.faces.new(vs)
            f.material_index = mat_index
        except ValueError:
            pass

    for r_i in range(len(outer_rings) - 1):
        ro0, ro1 = outer_rings[r_i], outer_rings[r_i + 1]
        ri0, ri1 = inner_rings[r_i], inner_rings[r_i + 1]
        for s_i in range(n_seg):
            s_n = (s_i + 1) % n_seg
            _mk_face((ro0[s_i], ro0[s_n], ro1[s_n], ro1[s_i]))
            _mk_face((ri0[s_n], ri0[s_i], ri1[s_i], ri1[s_n]))

    for s_i in range(n_seg):
        s_n = (s_i + 1) % n_seg
        _mk_face((outer_rings[0][s_n], outer_rings[0][s_i], inner_rings[0][s_i], inner_rings[0][s_n]))
        _mk_face((outer_rings[-1][s_i], outer_rings[-1][s_n], inner_rings[-1][s_n], inner_rings[-1][s_i]))


def _make_neoprene_material(name: str, rgb: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.36
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.14
        if "Sheen Roughness" in bsdf.inputs:
            bsdf.inputs["Sheen Roughness"].default_value = 0.35
    return mat


def _build_wetsuit_cuffs(
    rig: bpy.types.Object,
    body: bpy.types.Object,
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """Build sleek neoprene wrist cuffs (`max_r=0.036`); zero floating collar bib."""
    body_pts = _eval_mesh_verts_world(body)
    mat_cuff = _make_neoprene_material("Swimmer_WetsuitCuff_Mat", (0.06, 0.11, 0.16))
    parts: list[bpy.types.Object] = []

    for side in ("L", "R"):
        pb_la = rig.pose.bones.get(f"lowerarm01.{side}")
        pb_wr = rig.pose.bones.get(f"wrist.{side}")
        if pb_la is None or pb_wr is None:
            continue
        la_w = rig.matrix_world @ pb_la.head
        wr_w = rig.matrix_world @ pb_wr.head
        arm_pts = _filter_segment_pts(body_pts, la_w, wr_w, max_perp=0.055)
        bm_cuff = bmesh.new()
        _add_sleeve_band_from_verts(
            bm_cuff,
            arm_pts,
            la_w,
            wr_w,
            (0.84, 0.99),
            pad_m=0.0045,
            thick_m=0.0028,
            n_rings=3,
            n_seg=18,
            mat_index=0,
            min_r=0.025,
            max_r=0.036,
        )
        cuff_obj = _new_bmesh_obj(f"Swimmer_WristCuff_{side}", bm_cuff, mat_cuff, Matrix.Identity(4), col)
        _parent_to_bone_preserve_world(cuff_obj, rig, f"lowerarm01.{side}")
        parts.append(cuff_obj)

    return parts


# ============================================================================
# UNDERWATER SWIMMING BODY, LIMBS & FACIAL ACTING HELPERS
# ============================================================================
def _pose_spine_and_head(
    rig: bpy.types.Object,
    torso_fwd_deg: float,
    torso_side_deg: float,
    torso_twist_deg: float,
    head_look_up_deg: float,
    head_yaw_deg: float,
    head_roll_deg: float,
    frame: int | None,
) -> None:
    """Pose spine (`spine05..01`), neck & head for buoyant underwater hovering.
    `torso_fwd_deg > 0` bends forward (+rx); `torso_fwd_deg < 0` arches chest/head up;
    `head_look_up_deg > 0` lifts face upward (-rx)."""
    rx_sp = float(torso_fwd_deg)
    mc.set_bone_char_rot(rig, "spine05", rx_sp * 0.24, torso_side_deg * 0.35, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine04", rx_sp * 0.22, torso_side_deg * 0.25, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine03", rx_sp * 0.20, torso_side_deg * 0.20, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine02", rx_sp * 0.18, torso_side_deg * 0.10, torso_twist_deg * 0.15, frame=frame)
    mc.set_bone_char_rot(rig, "spine01", rx_sp * 0.16, torso_side_deg * 0.10, torso_twist_deg * 0.10, frame=frame)

    neck_rx = -rx_sp * 0.30 - head_look_up_deg * 0.42
    head_rx = -rx_sp * 0.30 - head_look_up_deg * 0.58
    mc.set_bone_char_rot(rig, "neck01", neck_rx * 0.60, 0.0, head_yaw_deg * 0.32, frame=frame)
    if "neck02" in rig.pose.bones:
        mc.set_bone_char_rot(rig, "neck02", neck_rx * 0.40, 0.0, head_yaw_deg * 0.20, frame=frame)
    mc.set_bone_char_rot(rig, "head", head_rx, head_roll_deg, head_yaw_deg * 0.48, frame=frame)


def _pose_arms_fk(
    rig: bpy.types.Object,
    ua_l: tuple[float, float, float],
    la_l: tuple[float, float, float],
    ua_r: tuple[float, float, float],
    la_r: tuple[float, float, float],
    hand_l: str = "relaxed",
    hand_r: str = "relaxed",
    frame: int | None = None,
) -> None:
    """FK arm posing via `mc.apply_arm_with_clavicle` with twist bones cleanly zeroed."""
    for tw in ("upperarm02.L", "lowerarm02.L", "upperarm02.R", "lowerarm02.R"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    mc.apply_arm_with_clavicle(rig, "L", ua_l, la_l, frame=frame)
    mc.apply_arm_with_clavicle(rig, "R", ua_r, la_r, frame=frame)
    mc.apply_hand_gesture(rig, "L", hand_l, frame=frame)
    mc.apply_hand_gesture(rig, "R", hand_r, frame=frame)


def _pose_legs_swim(
    rig: bpy.types.Object,
    thigh_l_rx: float,
    shin_l_rx: float,
    foot_l_rx: float,
    thigh_r_rx: float,
    shin_r_rx: float,
    foot_r_rx: float,
    splay_deg: float = 6.0,
    frame: int | None = None,
) -> None:
    """Pose thighs, shins & dive-booted feet for buoyant underwater flutter/tread."""
    for tw in ("upperleg02.L", "lowerleg02.L", "upperleg02.R", "lowerleg02.R"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    mc.set_bone_char_rot(rig, "upperleg01.L", thigh_l_rx, -3.0, splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "lowerleg01.L", shin_l_rx, 0.0, 0.0, frame=frame)
    mc.set_bone_char_rot(rig, "foot.L", foot_l_rx, 0.0, splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "upperleg01.R", thigh_r_rx, 3.0, -splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "lowerleg01.R", shin_r_rx, 0.0, 0.0, frame=frame)
    mc.set_bone_char_rot(rig, "foot.R", foot_r_rx, 0.0, -splay_deg, frame=frame)


def _near_hero_frame(f: int) -> bool:
    return any(abs(f - h) <= 6 for h in _HERO_BLINK_EXCLUSION_FRAMES)


def _apply_swimmer_face(
    rig: bpy.types.Object,
    f: int,
    jaw_deg: float,
    brow_lift: float,
    smile: float,
    eye_yaw_deg: float,
    eye_pitch_deg: float,
    lip_narrow: float = 0.0,
    force_alert_lids: bool = False,
) -> None:
    """Apply facial acting with jaw clamped <= 9.0 deg and wide alert eyes (orbicularis03 = -5.0 deg)."""
    jaw_clamped = C.clamp(jaw_deg, 0.0, 9.0)
    t_val = (f - 1) / float(C.FPS)
    suppress_blink = force_alert_lids or _near_hero_frame(f)
    blink_period = 999.0 if suppress_blink else 3.6
    mc.apply_facial_acting(
        rig,
        t=t_val,
        frame=f,
        jaw_open_deg=jaw_clamped,
        brow_lift=brow_lift,
        smile=smile,
        blink_period=blink_period,
        blink_phase=0.85,
        eye_yaw_deg=eye_yaw_deg,
        eye_pitch_deg=eye_pitch_deg,
        lip_narrow=lip_narrow,
    )
    if suppress_blink:
        mc.set_bone_char_rot(rig, "orbicularis03.L", -5.0, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", -5.0, 0.0, 0.0, frame=f)


def _rig_loc_for_anchor(
    anchor_world: Vector,
    anchor_rig_local: Vector,
    rig_eul: Euler,
    rig_scale: float,
) -> Vector:
    """Place rig origin so `anchor_rig_local` lands cleanly at `anchor_world` under `rig_eul`."""
    r_mat = rig_eul.to_matrix()
    return anchor_world - r_mat @ (anchor_rig_local * rig_scale)


# ============================================================================
# MAIN PUBLIC ENTRYPOINT
# ============================================================================
def build_swimmer(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Build & animate the 1.78m adult male wetsuit diver visible in Beat 1 (1..210) & Beat 6 (1561..1800)."""
    b1_pos = Vector(C.SWIMMER["b1_pos"])
    b1_yaw_deg = float(C.SWIMMER["b1_yaw_deg"])
    b6_pos = Vector(C.SWIMMER["b6_pos"])
    b6_yaw_deg = float(C.SWIMMER["b6_yaw_deg"])

    # 1. Build 4K MakeHuman MPFB2 Adult Male Diver in clean closed-neck dark charcoal-navy wetsuit
    cfg = {
        "gender": 1.0,
        "age": 0.56,
        "muscle": 0.60,
        "weight": 0.44,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": SWIMMER_HEIGHT_M,
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.09, 0.08, 0.07),
        "hair_tint_fac": 0.72,
        "hair_value_boost": 0.68,
        "hair_roughness": 0.28,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": (0.14, 0.34, 0.52),
        "iris_minor_rgb": (0.05, 0.14, 0.24),
        # Closed-neck charcoal-navy wetsuit top + dark navy wetsuit pants + dark dive boots
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.06, 0.11, 0.16),
        "shirt_tint_fac": 0.86,
        "shirt_sat_override": 0.45,
        "shirt_value_boost": 0.72,
        "shirt_roughness": 0.36,
        "shirt_sheen": 0.14,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.05, 0.09, 0.14),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.42,
        "pants_value_boost": 0.70,
        "pants_roughness": 0.38,
        "pants_sheen": 0.14,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.05, 0.06, 0.08),
        "shoe_tint_fac": 0.82,
        "shoe_sat_override": 0.16,
        "shoe_value_boost": 0.62,
        "shoe_roughness": 0.38,
        "shoe_sheen": 0.08,
    }
    res = mc.create_realistic_scientist(
        name="Swimmer",
        floor_z=0.0,
        start_xy=(0.0, 0.0),
        yaw_deg=0.0,
        cfg=cfg,
    )
    rig: bpy.types.Object = res["rig"]
    body: bpy.types.Object = res["body"]
    equipped: dict[str, bpy.types.Object] = res["equipped"]
    base_z: float = float(res["base_z"])
    rig_scale = float(rig.scale.z)

    # 2. Link all generated character objects cleanly into `col`
    _link_obj(rig, col)
    _link_obj(body, col)
    for obj in equipped.values():
        if obj is not None:
            _link_obj(obj, col)

    # Enforce visible_shadow=False on Eyes, Eyelashes, Eyebrows (V12 Sec 1)
    for k in ("Eyes", "Eyelashes", "Eyebrows"):
        if k in equipped and equipped[k] is not None:
            equipped[k].visible_shadow = False

    # Reset rig to origin momentarily for clean rest-space UnderClothes_Shrink & cuff fitting
    rig.location = Vector((0.0, 0.0, 0.0))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.view_layer.update()

    # Apply UnderClothes_Shrink (-0.011 DISPLACE BEFORE ARMATURE) including upperarm & lowerarm
    _apply_underclothes_shrink(body, rig)

    # Sleek neoprene wrist cuffs only (zero floating collar bib)
    cuffs = _build_wetsuit_cuffs(rig, body, col)

    # Measure rig-local chest/torso anchor (`spine03` head) so b1_pos / b6_pos center cleanly on chest
    b_anchor = rig.data.bones.get("spine03") or rig.data.bones.get("spine04") or rig.data.bones.get("pelvis")
    anchor_rig_local = b_anchor.head_local.copy() if b_anchor is not None else Vector((0.0, 0.0, 1.15))

    # 3. Calibrate Beat 6 horizontal swim-tuck hover strictly below Z = -1.05m and clear of 3 FT & floor
    b6_yaw_rad = math.radians(b6_yaw_deg)
    b6_pitch_deg = 84.0
    b6_eul_test = Euler((math.radians(b6_pitch_deg), math.radians(2.0), b6_yaw_rad), "XYZ")
    b6_anchor_init = Vector((b6_pos.x + 0.16, b6_pos.y, b6_pos.z - 0.18))
    rig.location = _rig_loc_for_anchor(b6_anchor_init, anchor_rig_local, b6_eul_test, rig_scale)
    rig.rotation_euler = b6_eul_test
    _pose_spine_and_head(rig, -6.0, 0.0, -2.0, 38.0, -8.0, 0.0, None)
    _pose_arms_fk(rig, (-28.0, -20.0, 10.0), (-82.0, 0.0, -12.0), (-28.0, 20.0, -10.0), (-82.0, 0.0, 12.0), "relaxed", "relaxed", None)
    _pose_legs_swim(rig, -112.0, 108.0, 22.0, -108.0, 104.0, 20.0, splay_deg=6.0, frame=None)
    bpy.context.view_layer.update()

    check_objs: list[bpy.types.Object] = (
        [body]
        + [equipped[k] for k in ("Hair", "Shirt", "Pants", "Shoes") if k in equipped and equipped[k] is not None]
        + cuffs
    )
    b6_min_x = min(_eval_mesh_bounds_world(o)[0].x for o in check_objs)
    b6_max_z = max(_eval_mesh_bounds_world(o)[1].z for o in check_objs)
    b6_x_adjust = max(0.0, (C.THREE_FT_X + 0.22) - b6_min_x)
    b6_z_adjust = min(0.0, -1.065 - b6_max_z)
    b6_anchor_base = b6_anchor_init + Vector((b6_x_adjust, 0.0, b6_z_adjust))

    # 4. Build fast keyframe schedule: step=8 baseline + step=3 on Beat 1 stop reaction (115..165)
    key_frames: set[int] = set()
    b1_end = min(210, total_frames)
    for f in range(1, b1_end + 1, 8):
        key_frames.add(f)
    for f in range(115, min(165, b1_end) + 1, 3):
        key_frames.add(f)
    for f in (1, 10, 60, 115, 118, 120, 135, 150, 165, 190, 210):
        if 1 <= f <= b1_end:
            key_frames.add(f)

    if total_frames >= 1561:
        # Insert clean hidden anchor at f=1560 equal to f=1561 so BEZIER handles start flat at cut
        key_frames.add(1560)
        for f in range(1561, total_frames + 1, 8):
            key_frames.add(f)
        for f in (1561, 1615, 1620, 1680, 1740, 1800, total_frames):
            if 1561 <= f <= total_frames:
                key_frames.add(f)

    sorted_kfs = sorted(key_frames)
    b1_yaw_rad = math.radians(b1_yaw_deg)

    # 5. Animate Swimmer across Beat 1 (1..210) and Beat 6 (1561..total_frames)
    for f in sorted_kfs:
        t_s = (f - 1) / float(C.FPS)

        if f <= 210:
            # Beat 1: Upright-treading swimmer leaning slightly toward -X (3 FT) reacting to stopped bullet
            u_react = C.smootherstep(100.0, 136.0, float(f))
            u_settle = C.smootherstep(138.0, 195.0, float(f))
            bob_z = 0.014 * math.sin(t_s * 2.2)
            kick_wave = math.sin(t_s * 4.4)
            tread_wave = math.sin(t_s * 3.1)

            # Subtle recoil back (+X) & slight pitch-up when bullet screeches dead at f=135
            recoil_x = 0.040 * u_react - 0.015 * u_settle
            anchor_w = Vector((
                b1_pos.x + 0.08 + recoil_x,
                b1_pos.y + 0.010 * math.sin(t_s * 1.4),
                b1_pos.z - 0.08 + bob_z + 0.018 * u_react,
            ))
            pitch_deg = 20.0 - 4.0 * u_react + 2.0 * u_settle + 1.0 * math.cos(t_s * 1.8)
            roll_deg = -2.5 + 1.5 * math.sin(t_s * 1.5)
            rig_eul = Euler((math.radians(pitch_deg), math.radians(roll_deg), b1_yaw_rad), "XYZ")
            rig.location = _rig_loc_for_anchor(anchor_w, anchor_rig_local, rig_eul, rig_scale)
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)

            # Torso leans slightly forward toward -X; head looks squarely toward bullet at 3 FT
            torso_fwd = 6.0 - 3.5 * u_react + 1.0 * math.sin(t_s * 2.2)
            head_up = 16.0 + 4.0 * u_react - 3.0 * u_settle
            head_yaw = -6.0 + 2.5 * u_react
            _pose_spine_and_head(rig, torso_fwd, 1.5 * math.sin(t_s * 1.6), -3.0, head_up, head_yaw, 2.0, f)

            # Arms tucked comfortably near ribs/chest (calm tread -> chest/chin guard, min_x >= 1.15m)
            calm_ua_l = (-22.0 + 7.0 * tread_wave, -22.0 - 4.0 * tread_wave, 10.0)
            calm_la_l = (-84.0 - 6.0 * tread_wave, 0.0, -14.0)
            calm_ua_r = (-22.0 - 7.0 * tread_wave, 22.0 + 4.0 * tread_wave, -10.0)
            calm_la_r = (-84.0 + 6.0 * tread_wave, 0.0, 14.0)

            def_ua_l = (-44.0, -18.0, 10.0)
            def_la_l = (-104.0, -10.0, -20.0)
            def_ua_r = (-44.0, 18.0, -10.0)
            def_la_r = (-104.0, 10.0, 20.0)

            w_def = u_react * (1.0 - 0.18 * u_settle)
            ua_l = tuple(C.lerp(a, b, w_def) for a, b in zip(calm_ua_l, def_ua_l))
            la_l = tuple(C.lerp(a, b, w_def) for a, b in zip(calm_la_l, def_la_l))
            ua_r = tuple(C.lerp(a, b, w_def) for a, b in zip(calm_ua_r, def_ua_r))
            la_r = tuple(C.lerp(a, b, w_def) for a, b in zip(calm_la_r, def_la_r))
            _pose_arms_fk(
                rig,
                ua_l, la_l, ua_r, la_r,
                hand_l="shock" if w_def > 0.38 else "present",
                hand_r="shock" if w_def > 0.38 else "present",
                frame=f,
            )

            # Natural underwater treading flutter kick (knees softly bent keeping boots inside |X| <= 2.20)
            _pose_legs_swim(
                rig,
                thigh_l_rx=-36.0 - 6.0 * kick_wave,
                shin_l_rx=66.0 + 8.0 * kick_wave,
                foot_l_rx=22.0 + 5.0 * kick_wave,
                thigh_r_rx=-36.0 + 6.0 * kick_wave,
                shin_r_rx=66.0 - 8.0 * kick_wave,
                foot_r_rx=22.0 - 5.0 * kick_wave,
                splay_deg=6.5,
                frame=f,
            )

            # Alert wide eyes (orbicularis03 = -5 deg) + <= 9 deg shock gasp at bullet stop
            jaw_b1 = 1.5 + 7.0 * u_react - 2.0 * u_settle
            brow_b1 = 0.20 + 0.70 * u_react
            _apply_swimmer_face(
                rig,
                f=f,
                jaw_deg=jaw_b1,
                brow_lift=brow_b1,
                smile=-0.25 * u_react,
                eye_yaw_deg=-4.0,
                eye_pitch_deg=-3.0 * u_settle,
                lip_narrow=0.32 * u_react,
                force_alert_lids=(110 <= f <= 175),
            )

        else:
            # Beat 6 (1561..1800, plus flat hidden start key at 1560):
            # Relaxed horizontal hover strictly below Z = -1.05m watching harmless copper flakes
            f_eff = max(1561.0, float(f))
            u_drift = C.smootherstep(1561.0, float(max(1562, total_frames)), f_eff)
            t_b6 = (f_eff - 1561.0) / float(C.FPS)
            bob_z = 0.010 * math.sin(t_b6 * 1.7)
            kick_slow = math.sin(t_b6 * 2.6)
            arm_slow = math.sin(t_b6 * 1.9)

            anchor_w = Vector((
                b6_anchor_base.x - 0.015 * u_drift,
                b6_anchor_base.y + 0.015 * math.sin(t_b6 * 1.1),
                b6_anchor_base.z - 0.020 * u_drift + bob_z,
            ))
            pitch_deg = b6_pitch_deg + 1.2 * math.sin(t_b6 * 1.4)
            roll_deg = 2.0 + 1.5 * math.cos(t_b6 * 1.3)
            rig_eul = Euler((math.radians(pitch_deg), math.radians(roll_deg), b6_yaw_rad), "XYZ")
            rig.location = _rig_loc_for_anchor(anchor_w, anchor_rig_local, rig_eul, rig_scale)
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)

            # Relaxed swimmer arch watching flakes drift down past 3 FT
            _pose_spine_and_head(
                rig,
                torso_fwd_deg=-6.0 + 0.8 * math.sin(t_b6 * 1.5),
                torso_side_deg=1.5 * math.sin(t_b6 * 1.2),
                torso_twist_deg=-2.0,
                head_look_up_deg=38.0 + 2.5 * u_drift,
                head_yaw_deg=-8.0 + 3.0 * math.sin(t_b6 * 0.9),
                head_roll_deg=1.5,
                frame=f,
            )

            # Relaxed buoyancy arms (comfortably tucked down/sideways, min_x >= 1.12m)
            _pose_arms_fk(
                rig,
                (-28.0 + 4.0 * arm_slow, -20.0 - 3.0 * arm_slow, 10.0),
                (-82.0 - 4.0 * arm_slow, 0.0, -12.0),
                (-28.0 - 4.0 * arm_slow, 20.0 + 3.0 * arm_slow, -10.0),
                (-82.0 + 4.0 * arm_slow, 0.0, 12.0),
                hand_l="relaxed",
                hand_r="relaxed",
                frame=f,
            )

            # Gentle tucked swim-hover leg trail (both |X| <= 2.25 and Z >= -1.85m!)
            _pose_legs_swim(
                rig,
                thigh_l_rx=-112.0 - 4.0 * kick_slow,
                shin_l_rx=108.0 + 6.0 * kick_slow,
                foot_l_rx=22.0 + 4.0 * kick_slow,
                thigh_r_rx=-108.0 + 4.0 * kick_slow,
                shin_r_rx=104.0 - 6.0 * kick_slow,
                foot_r_rx=20.0 - 4.0 * kick_slow,
                splay_deg=6.0,
                frame=f,
            )

            # Calm relief expression watching copper scrap drift downward
            _apply_swimmer_face(
                rig,
                f=f,
                jaw_deg=1.0,
                brow_lift=0.22,
                smile=0.18,
                eye_yaw_deg=-4.0,
                eye_pitch_deg=5.0 * (1.0 - 0.5 * u_drift),
                lip_narrow=0.0,
                force_alert_lids=True,
            )

    # Set smooth BEZIER interpolation with CONSTANT snaps across hidden gap (210, 1560)
    _set_fcurve_interp(rig, interp="BEZIER", constant_frames={210, 1560})

    # 6. Keyframe CONSTANT visibility: visible ONLY in Beat 1 (1..210) & Beat 6 (1561..1800)
    all_vis_objs: list[bpy.types.Object] = (
        [rig, body]
        + [o for o in equipped.values() if o is not None]
        + cuffs
    )
    _keyframe_visibility(all_vis_objs, total_frames)

    # 7. Runtime Bounds Sanity Check across Beat 1 and Beat 6
    test_frames = [tf for tf in (1, 60, 135, 165, 210, 1561, 1620, 1700, 1800) if tf <= total_frames]
    if test_frames:
        scene = bpy.context.scene
        prev_frame = scene.frame_current
        try:
            for tf in test_frames:
                scene.frame_set(tf)
                bpy.context.view_layer.update()
                for obj in check_objs:
                    mn, mx = _eval_mesh_bounds_world(obj)
                    assert mn.x >= C.THREE_FT_X + 0.12, (
                        f"[Swimmer Bounds] f={tf} {obj.name} min_x={mn.x:.3f} clips near 3 FT ({C.THREE_FT_X:.3f})!"
                    )
                    assert max(abs(mn.x), abs(mx.x)) <= 2.62, (
                        f"[Swimmer Bounds] f={tf} {obj.name} |x|={max(abs(mn.x), abs(mx.x)):.3f} > 2.62!"
                    )
                    assert mn.z >= float(C.TANK["floor_z"]) + 0.05, (
                        f"[Swimmer Bounds] f={tf} {obj.name} min_z={mn.z:.3f} clips tank floor!"
                    )
                    if tf <= 210:
                        assert mx.z <= -0.05, (
                            f"[Swimmer Bounds] Beat 1 f={tf} {obj.name} max_z={mx.z:.3f} pokes near water_z=0!"
                        )
                    else:
                        assert mx.z <= -1.05, (
                            f"[Swimmer Bounds] Beat 6 f={tf} {obj.name} max_z={mx.z:.3f} > -1.05m!"
                        )
        finally:
            scene.frame_set(prev_frame)
            bpy.context.view_layer.update()

    return {
        "rig": rig,
        "body": body,
        "equipped": equipped,
        "cuffs": cuffs,
        "height": float(res["height"]),
        "base_z": base_z,
    }
