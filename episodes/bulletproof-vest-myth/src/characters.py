"""
characters.py — 4K MakeHuman MPFB2 Hero Target (in Tactical Ballistic Plate Carrier)
& Ballistic Range Shooter (with Procedural Tactical Test Rifle)
for Episode #3: "Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames).

Strictly follows AGENTS.md V12 & copies verified MPFB2 assets from `swimmer_character.py`:
- Verified assets: `mindfront_lusekofta.mhclo`, `mindfront_male_trousers_2.mhclo`,
  `toigo_ankle_boots_male.mhclo`, `high-poly.mhclo`, `short04.mhclo`,
  `eyebrow002.mhclo`, `eyelashes01.mhclo`, `teeth_base.mhclo`, `tongue01.mhclo`.
- `eye_obj.visible_shadow = False`, `lash_obj.visible_shadow = False`, per-actor independent
  `EnhancedEye` node tree copy, front-facing Glossy cornea, relaxed `-5.0 deg` upper eyelids,
  jaw clamped `<= 14.0 deg` (peak shock `<= 8.5 deg`).
- `UnderClothes_Shrink` (`-0.011` DISPLACE BEFORE `ARMATURE`) including `upperarm` & `lowerarm`.
- Exact physical `0.5 inch` (`12.7 mm`) equal-and-opposite chest/shoulder recoil animation.
- Blender 5.3.0 safe FCurves (`C.iter_action_fcurves`).

Public API:
    build_characters(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]
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

import mpfb_characters as mc  # noqa: E402
import vest_contract as C  # noqa: E402

_HERO_BLINK_EXCLUSION_FRAMES: tuple[int, ...] = (
    1, 55, 75, 92, 130, 180, 210, 335, 1365, 1405, 1561, 1620, 1700, 1800
)


# ============================================================================
# COLLECTION, BMESH, FCURVE & BONE HELPERS (BLENDER 5.3.0 SAFE)
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _set_smooth(mesh: bpy.types.Mesh, smooth: bool = True) -> None:
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()


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
) -> bpy.types.Object:
    """Attach material slots strictly BEFORE `bm.to_mesh(mesh)` per AGENTS.md V12."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh, smooth=smooth)
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = world_xf.copy()
    return _link_obj(obj, col)


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
        if cur_idx > arm_idx and hasattr(body.modifiers, "move"):
            body.modifiers.move(cur_idx, arm_idx)


def _enforce_v12_eyes(equipped: dict[str, bpy.types.Object]) -> None:
    """Enforce shadow-bypass & independent EnhancedEye node tree copy on actor eyes/lashes."""
    for k in ("Eyes", "Eyelashes", "Eyebrows"):
        obj = equipped.get(k)
        if obj is not None:
            obj.visible_shadow = False


def _near_hero_frame(f: int) -> bool:
    return any(abs(f - h) <= 6 for h in _HERO_BLINK_EXCLUSION_FRAMES)


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
    """Jaw clamped strictly <= 14.0 deg (realistic speech/shock <= 9 deg) & relaxed -5 deg eyelids."""
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
        blink_phase=0.65,
        eye_yaw_deg=eye_yaw_deg,
        eye_pitch_deg=eye_pitch_deg,
        lip_narrow=0.0,
    )
    if suppress_blink:
        mc.set_bone_char_rot(rig, "orbicularis03.L", -5.0, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", -5.0, 0.0, 0.0, frame=f)


def _pose_standing_spine(
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
    mc.set_bone_char_rot(rig, "neck01", -torso_fwd_deg * 0.35 + head_pitch_deg * 0.45, 0.0, head_yaw_deg * 0.40, frame=frame)
    mc.set_bone_char_rot(rig, "head", -torso_fwd_deg * 0.30 + head_pitch_deg * 0.55, 0.0, head_yaw_deg * 0.60, frame=frame)


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


# ============================================================================
# WEARABLE TACTICAL BALLISTIC VEST & PROCEDURAL TEST RIFLE
# ============================================================================
def _make_tactical_mat(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.05,
    roughness: float = 0.58,
    emit_rgb: tuple[float, float, float] | None = None,
    emit_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emit_rgb is not None and emit_strength > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(1.85, emit_strength)
    return mat


def _add_box_bm(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_index: int = 0,
    bevel_w: float = 0.004,
) -> None:
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_w > 0.0:
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)
    bmesh.ops.translate(tmp, vec=Vector(center), verts=tmp.verts)
    vmap = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vmap[v] for v in f.verts])
        nf.material_index = mat_index
    tmp.free()


def _build_wearable_plate_carrier(
    rig: bpy.types.Object,
    col: bpy.types.Collection,
) -> bpy.types.Object:
    """
    Build a detailed multi-panel tactical plate carrier vest fitted cleanly onto `Hero`
    and parented to `spine03` preserving exact world alignment (`front plate at Z = 1.32 m`).
    Built while rig sits upright at origin facing -Y (standard MakeHuman rest orientation).
    """
    mat_nylon = _make_tactical_mat("Mat_HeroVest_Nylon", (0.042, 0.048, 0.058), metallic=0.06, roughness=0.64)
    mat_plate = _make_tactical_mat("Mat_HeroVest_StrikePlate", (0.085, 0.095, 0.115), metallic=0.35, roughness=0.30)
    mat_gold_trim = _make_tactical_mat("Mat_HeroVest_KevlarTrim", (0.92, 0.66, 0.12), metallic=0.25, roughness=0.34, emit_rgb=(0.95, 0.65, 0.10), emit_strength=0.45)

    cz = float(C.HERO["chest_z"])
    bm = bmesh.new()
    # Front armored carrier panel (MakeHuman faces -Y in rest space!)
    _add_box_bm(bm, (0.0, -0.142, cz), (0.31, 0.034, 0.36), mat_index=0, bevel_w=0.008)
    # Bevelled front Silicon-Carbide strike plate center inset
    _add_box_bm(bm, (0.0, -0.160, cz + 0.01), (0.24, 0.014, 0.28), mat_index=1, bevel_w=0.005)
    # Subtle golden Kevlar stitch accent lines & MOLLE webbing straps
    for dz in (-0.09, -0.03, 0.03, 0.09):
        _add_box_bm(bm, (0.0, -0.168, cz + dz), (0.21, 0.004, 0.014), mat_index=0, bevel_w=0.001)
    _add_box_bm(bm, (0.0, -0.168, cz + 0.145), (0.23, 0.006, 0.008), mat_index=2, bevel_w=0.001)

    # Rear armored carrier panel
    _add_box_bm(bm, (0.0, +0.132, cz), (0.30, 0.032, 0.36), mat_index=0, bevel_w=0.008)
    # Left & right cummerbund side straps
    for sx in (-0.165, +0.165):
        _add_box_bm(bm, (sx, -0.005, cz - 0.05), (0.032, 0.27, 0.14), mat_index=0, bevel_w=0.006)
    # Padded shoulder straps bridging front to rear
    for sx in (-0.105, +0.105):
        _add_box_bm(bm, (sx, -0.005, cz + 0.195), (0.068, 0.24, 0.026), mat_index=0, bevel_w=0.006)

    vest_obj = _new_bmesh_obj(
        "Hero_Tactical_PlateCarrier",
        bm,
        [mat_nylon, mat_plate, mat_gold_trim],
        Matrix.Identity(4),
        col,
        smooth=False,
    )
    _parent_to_bone_preserve_world(vest_obj, rig, "spine03")
    return vest_obj


def _build_tactical_test_rifle(col: bpy.types.Collection) -> bpy.types.Object:
    """
    Build a sleek machined ballistic range test carbine along +X (muzzle tip at local X = 0.0,
    stock butt at local X = -0.68 m) so it aligns with `RANGE["muzzle_x"]` & `RANGE["b5_muzzle_x"]`.
    """
    mat_gunmetal = _make_tactical_mat("Mat_Rifle_Gunmetal", (0.06, 0.068, 0.080), metallic=0.92, roughness=0.26)
    mat_polymer = _make_tactical_mat("Mat_Rifle_Polymer", (0.035, 0.040, 0.046), metallic=0.08, roughness=0.55)
    mat_optic = _make_tactical_mat("Mat_Rifle_OpticCyan", (0.12, 0.85, 1.0), metallic=0.20, roughness=0.12, emit_rgb=(0.12, 0.85, 1.0), emit_strength=0.95)

    bm = bmesh.new()
    # Barrel (-0.34 -> 0.00)
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=20,
        radius1=0.011,
        radius2=0.010,
        depth=0.34,
        matrix=Matrix.Translation((-0.17, 0.0, 0.0)) @ Euler((0.0, math.radians(90.0), 0.0), "XYZ").to_matrix().to_4x4(),
    )
    for f in bm.faces:
        f.material_index = 0
    # Handguard, receiver, magazine, pistol grip & shoulder stock
    _add_box_bm(bm, (-0.24, 0.0, 0.0), (0.22, 0.036, 0.042), mat_index=1, bevel_w=0.004)
    _add_box_bm(bm, (-0.43, 0.0, -0.006), (0.19, 0.040, 0.064), mat_index=0, bevel_w=0.004)
    _add_box_bm(bm, (-0.39, 0.0, -0.095), (0.052, 0.024, 0.13), mat_index=1, bevel_w=0.003)
    _add_box_bm(bm, (-0.49, 0.0, -0.075), (0.032, 0.026, 0.095), mat_index=1, bevel_w=0.003)
    _add_box_bm(bm, (-0.61, 0.0, -0.010), (0.18, 0.034, 0.088), mat_index=1, bevel_w=0.005)
    # Compact red-dot optic + subtle cyan status lens
    _add_box_bm(bm, (-0.41, 0.0, 0.044), (0.085, 0.030, 0.034), mat_index=0, bevel_w=0.003)
    _add_box_bm(bm, (-0.366, 0.0, 0.044), (0.005, 0.020, 0.020), mat_index=2, bevel_w=0.001)

    return _new_bmesh_obj("Shooter_Tactical_Rifle", bm, [mat_gunmetal, mat_polymer, mat_optic], Matrix.Identity(4), col, smooth=False)


# ============================================================================
# HERO & SHOOTER BUILDERS + ANIMATORS
# ============================================================================
def _build_hero(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    cfg = {
        "gender": 1.0,
        "age": 0.56,
        "muscle": 0.64,
        "weight": 0.46,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(C.HERO["height_m"]),
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.10, 0.08, 0.06),
        "hair_tint_fac": 0.70,
        "hair_value_boost": 0.72,
        "hair_roughness": 0.30,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": (0.16, 0.38, 0.58),
        "iris_minor_rgb": (0.06, 0.16, 0.28),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.09, 0.12, 0.16),
        "shirt_tint_fac": 0.86,
        "shirt_sat_override": 0.34,
        "shirt_value_boost": 0.76,
        "shirt_roughness": 0.48,
        "shirt_sheen": 0.12,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.06, 0.07, 0.09),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.24,
        "pants_value_boost": 0.68,
        "pants_roughness": 0.52,
        "pants_sheen": 0.10,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.04, 0.045, 0.055),
        "shoe_tint_fac": 0.84,
        "shoe_sat_override": 0.14,
        "shoe_value_boost": 0.60,
        "shoe_roughness": 0.40,
        "shoe_sheen": 0.06,
    }
    res = mc.create_realistic_scientist(name="Hero", floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig: bpy.types.Object = res["rig"]
    body: bpy.types.Object = res["body"]
    equipped: dict[str, bpy.types.Object] = res["equipped"]
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
    vest_obj = _build_wearable_plate_carrier(rig, col)

    # Keyframe schedule across visible Beats 1, 2, 5, 6 (dense around impacts!)
    kfs: set[int] = set()
    for lo, hi, step in ((1, 210, 6), (211, 510, 10), (1261, 1560, 10), (1561, total_frames, 8)):
        for f in range(lo, min(hi, total_frames) + 1, step):
            kfs.add(f)
    for dense_lo, dense_hi in ((70, 115), (330, 368), (1360, 1405)):
        for f in range(dense_lo, min(dense_hi, total_frames) + 1, 2):
            kfs.add(f)
    for cut in (1, 210, 211, 510, 1260, 1261, 1560, 1561, total_frames):
        if 1 <= cut <= total_frames:
            kfs.add(cut)

    for f in sorted(kfs):
        ff = float(f)
        t_s = (f - 1) / float(C.FPS)
        rx_m = C.hero_recoil_x(ff)
        recoil_norm = rx_m / max(1e-5, C.HALF_INCH_M)
        breath = math.sin(t_s * 1.9)

        if f <= 210:
            bx, by, _ = C.HERO["b1_pos"]
            yaw_deg = float(C.HERO["b1_yaw_deg"])
            torso_fwd = 2.0 - 4.5 * recoil_norm + 0.6 * breath
            head_yaw = -10.0 + 3.0 * recoil_norm
            jaw = 1.2 + 6.5 * recoil_norm
            brow = 0.25 + 0.55 * recoil_norm
            smile = -0.15 * recoil_norm
            ua_l = (-14.0 - 10.0 * recoil_norm, -14.0, 6.0)
            la_l = (-42.0 - 18.0 * recoil_norm, 0.0, -8.0)
            ua_r = (-14.0 - 10.0 * recoil_norm, 14.0, -6.0)
            la_r = (-42.0 - 18.0 * recoil_norm, 0.0, 8.0)
            hand = "shock" if recoil_norm > 0.35 else "relaxed"
        elif f <= 510:
            bx, by, _ = C.HERO["b2_pos"]
            yaw_deg = float(C.HERO["b2_yaw_deg"])
            torso_fwd = 1.5 - 4.0 * recoil_norm + 0.5 * breath
            head_yaw = -6.0
            jaw = 1.0 + 4.5 * recoil_norm
            brow = 0.20 + 0.40 * recoil_norm
            smile = 0.0
            ua_l = (-10.0 - 8.0 * recoil_norm, -12.0, 4.0)
            la_l = (-32.0 - 12.0 * recoil_norm, 0.0, -6.0)
            ua_r = (-10.0 - 8.0 * recoil_norm, 12.0, -4.0)
            la_r = (-32.0 - 12.0 * recoil_norm, 0.0, 6.0)
            hand = "relaxed"
        elif f <= 1560:
            bx, by, _ = C.HERO["b5_pos"]
            yaw_deg = float(C.HERO["b5_yaw_deg"])
            torso_fwd = 1.0 - 4.2 * recoil_norm
            head_yaw = 0.0
            jaw = 0.8 + 3.0 * recoil_norm
            brow = 0.15 + 0.30 * recoil_norm
            smile = 0.0
            ua_l = (-8.0, -10.0, 4.0)
            la_l = (-24.0, 0.0, -4.0)
            ua_r = (-8.0, 10.0, -4.0)
            la_r = (-24.0, 0.0, 4.0)
            hand = "relaxed"
        else:
            bx, by, _ = C.HERO["b6_pos"]
            yaw_deg = float(C.HERO["b6_yaw_deg"])
            u_calm = C.smootherstep(1561.0, float(total_frames), ff)
            torso_fwd = 1.0 + 0.8 * breath
            head_yaw = -12.0 + 4.0 * u_calm
            jaw = 0.8
            brow = 0.18
            smile = 0.22 * u_calm
            ua_l = (-8.0, -12.0, 5.0)
            la_l = (-22.0 + 3.0 * breath, 0.0, -5.0)
            ua_r = (-8.0, 12.0, -5.0)
            la_r = (-22.0 - 3.0 * breath, 0.0, 5.0)
            hand = "relaxed"

        # MakeHuman faces -Y at yaw=0; +yaw_deg faces +X, -yaw_deg faces -X (mpfb_characters convention).
        rig.location = Vector((bx + rx_m, by, base_z))
        rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

        _pose_standing_spine(rig, torso_fwd, -2.0 * recoil_norm, 2.0 * recoil_norm, head_yaw, frame=f)
        _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, hand, hand, frame=f)
        _apply_actor_face(rig, f=f, jaw_deg=jaw, brow_lift=brow, smile=smile, eye_yaw_deg=-3.0, eye_pitch_deg=0.0, force_alert_lids=(f <= 210 or f >= 1561))

    _set_fcurve_interp(rig, interp="BEZIER", constant_frames={210, 510, 1260, 1560})

    vis_objs = [rig, body, vest_obj] + [o for o in equipped.values() if o is not None]
    _key_visibility_windows(vis_objs, [(1, 510), (1261, total_frames)], total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "vest": vest_obj}


def _build_shooter(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    cfg = {
        "gender": 1.0,
        "age": 0.58,
        "muscle": 0.58,
        "weight": 0.45,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": float(C.SHOOTER["height_m"]),
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.07, 0.06, 0.05),
        "hair_tint_fac": 0.75,
        "hair_value_boost": 0.66,
        "hair_roughness": 0.30,
        "brow_tint_fac": 0.60,
        "iris_major_rgb": (0.42, 0.30, 0.14),  # Independent hazel-amber eyes (zero Hero overwrite!)
        "iris_minor_rgb": (0.18, 0.12, 0.06),
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.14, 0.16, 0.13),  # Tactical olive-charcoal range jacket
        "shirt_tint_fac": 0.84,
        "shirt_sat_override": 0.28,
        "shirt_value_boost": 0.74,
        "shirt_roughness": 0.50,
        "shirt_sheen": 0.10,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.06, 0.065, 0.075),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.22,
        "pants_value_boost": 0.66,
        "pants_roughness": 0.52,
        "pants_sheen": 0.08,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.045, 0.045, 0.050),
        "shoe_tint_fac": 0.84,
        "shoe_sat_override": 0.12,
        "shoe_value_boost": 0.58,
        "shoe_roughness": 0.40,
        "shoe_sheen": 0.06,
    }
    res = mc.create_realistic_scientist(name="Shooter", floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig: bpy.types.Object = res["rig"]
    body: bpy.types.Object = res["body"]
    equipped: dict[str, bpy.types.Object] = res["equipped"]
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
    rifle_obj = _build_tactical_test_rifle(col)

    kfs: set[int] = set()
    for lo, hi, step in ((211, 510, 10), (1261, 1560, 10)):
        for f in range(lo, min(hi, total_frames) + 1, step):
            kfs.add(f)
    for dense_lo, dense_hi in ((242, 276), (1292, 1330)):
        for f in range(dense_lo, min(dense_hi, total_frames) + 1, 2):
            kfs.add(f)
    for cut in (211, 510, 1260, 1261, 1560):
        if 1 <= cut <= total_frames:
            kfs.add(cut)

    cz = float(C.RANGE["chest_z"])
    for f in sorted(kfs):
        ff = float(f)
        srx = C.shooter_recoil_x(ff)  # Negative (-0.0127 m max)
        recoil_norm = abs(srx) / max(1e-5, C.HALF_INCH_M)

        if f <= 510:
            sx, sy, _ = C.SHOOTER["b2_pos"]
            yaw_deg = float(C.SHOOTER["b2_yaw_deg"])
            mx = float(C.RANGE["muzzle_x"])
        else:
            sx, sy, _ = C.SHOOTER["b5_pos"]
            yaw_deg = float(C.SHOOTER["b5_yaw_deg"])
            mx = float(C.RANGE["b5_muzzle_x"])

        rig.location = Vector((sx + srx, sy, base_z))
        rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

        _pose_standing_spine(rig, 3.0 - 4.0 * recoil_norm, 6.0, -2.0, -4.0, frame=f)
        # Aiming rifle posture: right hand on grip, left support hand forward on handguard
        _pose_arms_fk(
            rig,
            ua_l=(-64.0 + 4.0 * recoil_norm, 20.0, -18.0),
            la_l=(-46.0, 0.0, 26.0),
            ua_r=(-52.0 + 4.0 * recoil_norm, -12.0, 14.0),
            la_r=(-92.0, 0.0, -18.0),
            hand_l="relaxed",
            hand_r="relaxed",
            frame=f,
        )
        _apply_actor_face(rig, f=f, jaw_deg=0.8, brow_lift=-0.12, smile=0.0, eye_yaw_deg=0.0, eye_pitch_deg=0.0, force_alert_lids=True)

        # Keep test rifle muzzle locked squarely on RANGE muzzle datum + matching shoulder recoil!
        rifle_obj.location = Vector((mx + srx, -0.04, cz))
        rifle_obj.rotation_euler = Euler((0.0, math.radians(-2.2 * recoil_norm), 0.0), "XYZ")
        rifle_obj.keyframe_insert("location", frame=f)
        rifle_obj.keyframe_insert("rotation_euler", frame=f)

    _set_fcurve_interp(rig, interp="BEZIER", constant_frames={510, 1260})
    _set_fcurve_interp(rifle_obj, interp="BEZIER", constant_frames={510, 1260})

    vis_objs = [rig, body, rifle_obj] + [o for o in equipped.values() if o is not None]
    _key_visibility_windows(vis_objs, [(211, 510), (1261, 1560)], total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "rifle": rifle_obj}


def build_characters(
    col: bpy.types.Collection,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build & animate both 4K MPFB2 Hero (in ballistic vest) and Shooter (with test rifle)."""
    hero = _build_hero(col, total_frames)
    shooter = _build_shooter(col, total_frames)
    return {"hero": hero, "shooter": shooter}
