"""
characters.py — 2 Photoreal 4K MakeHuman MPFB2 Rigged Rescue Aviators & 1:1 Rescue Cockpit Set
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Cast & 1:1 Cockpit Action Choreography (Beats 1 `1..240` & Beat 5A `1291..1450`):
  1. `Heli_Pilot` (Right Seat PIC, X=+0.36 m, facing -Y toward panoramic glass & camera):
     - 4K MPFB2 rigged rescue pilot in dark navy-slate flight suit + red-white rescue helmet with
       raised tinted visor (keeping photoreal eyes & catchlights 100% crisp and visible!) + comms boom mic
     - Beat 1 (`1..240`): Alarmed glance at flashing `ENG OUT` caution strip -> left hand SLAMS the
       hero **Collective Pitch Lever** straight down (`f=105..150`) while right hand stabilizes cyclic stick!
     - Beat 5A (`1291..1450`): Laser-focused `f/3.6` shallow-DOF close-up -> left hand HAULS the
       collective pitch lever strongly UP (`f=1340..1425`) while right hand pulls cyclic back for the 30-ft flare!
  2. `Heli_CoPilot` (Left Seat, X=-0.36 m, facing -Y):
     - 4K MPFB2 rigged rescue co-pilot with independent warm-hazel `EnhancedEye` node group copy
     - Beat 1: Checks overhead/instrument gauges, then points decisively downward toward the rooftop helipad!
     - Beat 5A: Calls out `"30 feet!"` on the comms mic, gripping the glare-shield as the golden skyline rises!
  3. 1:1 Rescue Helicopter Cockpit (`Char_Heli_CockpitFrame`, `Char_Heli_CollectiveLever`, `Char_Heli_CyclicStick`):
     - Curved frameless wrap-around canopy pillars (zero opaque center mullion blocking crew faces!)
     - Backlit rescue avionics console with dual digital MFDs, analog Rotor RPM dial, and pulsing crimson
       `ENG OUT` / `MASTER CAUTION` warning strip!

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
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

_HERO_BLINK_EXCL: tuple[int, ...] = (1, 14, 42, 120, 155, 220, 1291, 1350, 1410, 1450)
_COCKPIT_WINDOWS: list[tuple[int, int]] = [(1, 240), (1291, C.B5_COCKPIT_END)]


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
        mc.set_bone_char_rot(rig, "orbicularis03.L", -5.0, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", -5.0, 0.0, 0.0, frame=f)


def _pose_seated_body(
    rig: bpy.types.Object,
    torso_fwd_deg: float,
    torso_twist_deg: float,
    head_pitch_deg: float,
    head_yaw_deg: float,
    frame: int,
) -> None:
    # Seated thighs & knees neatly tucked inside rescue cockpit footwell
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
    mc.apply_hand_gesture(rig, "L", hand_l, frame=frame)
    mc.apply_hand_gesture(rig, "R", hand_r, frame=frame)


def _attach_to_bone(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    bpy.context.view_layer.update()
    world_mat = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world_mat


def _build_rescue_helmet(
    name: str,
    rig: bpy.types.Object,
    col: bpy.types.Collection,
    stripe_rgb: tuple[float, float, float],
) -> bpy.types.Object:
    """
    White-and-red SPH-5 Rescue Helicopter Flight Helmet + raised dark tinted visor (leaving
    photoreal MPFB2 eyes & softbox catchlights 100% open!) + comms boom mic, parented to `head`.
    """
    mat_w = _make_mat(f"Mat_{name}_ShellWhite", (0.94, 0.95, 0.96), metallic=0.06, roughness=0.18, coat=0.7)
    mat_r = _make_mat(f"Mat_{name}_Stripe", stripe_rgb, metallic=0.12, roughness=0.22, coat=0.6)
    mat_v = _make_mat(f"Mat_{name}_Visor", (0.04, 0.06, 0.08), metallic=0.45, roughness=0.08, coat=0.9)
    mat_b = _make_mat(f"Mat_{name}_MicBlk", (0.025, 0.028, 0.032), metallic=0.25, roughness=0.45)

    bpy.context.view_layer.update()
    hc = rig.matrix_world @ rig.pose.bones["head"].head
    # Dome centered cleanly over skull crown (rest face looks -Y)
    cz = hc.z + 0.112
    cy = hc.y + 0.022
    bm = bmesh.new()

    # Open-face rescue helmet dome (upper & rear hemisphere shell)
    res_s = bmesh.ops.create_uvsphere(
        bm, u_segments=24, v_segments=16, radius=1.0,
        matrix=Matrix.Translation((hc.x, cy, cz)) @ Matrix.Diagonal((0.106, 0.118, 0.108, 1.0)),
    )
    # Trim front face opening cleanly so eyebrows, eyes, nose & mouth stay 100% exposed!
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

    # Raised dark tinted visor housing perched cleanly over brow (Z = cz + 0.048)
    _add_box_bm(bm, (hc.x, cy - 0.095, cz + 0.044), (0.148, 0.048, 0.046), mat_idx=2, bevel_w=0.010, rot_deg=(-20.0, 0.0, 0.0))

    # Padded comms earcups + flexible boom microphone curving to mouth corner
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (hc.x + sx * 0.102, cy - 0.004, hc.z + 0.072), (0.030, 0.068, 0.082), mat_idx=3, bevel_w=0.008)
    p0 = Vector((hc.x - 0.098, cy - 0.018, hc.z + 0.060))
    p1 = Vector((hc.x - 0.075, cy - 0.118, hc.z + 0.038))
    p2 = Vector((hc.x - 0.024, cy - 0.125, hc.z + 0.036))
    _add_strut_bm(bm, p0, p1, 0.0045, mat_idx=3, segs=8)
    _add_strut_bm(bm, p1, p2, 0.0045, mat_idx=3, segs=8)
    _add_box_bm(bm, (p2.x, p2.y, p2.z), (0.022, 0.016, 0.016), mat_idx=3, bevel_w=0.004)

    h_obj = _new_bmesh_obj(
        name, bm, [mat_w, mat_r, mat_v, mat_b],
        Matrix.Identity(4), col, smooth=True,
    )
    _attach_to_bone(h_obj, rig, "head")
    return h_obj


def _build_cockpit_set(col: bpy.types.Collection, total_frames: int) -> dict[str, Any]:
    """
    1:1 Scale Rescue Helicopter Cockpit Surround (Beats 1 `1..240` & Beat 5A `1291..1450`):
      - Dual padded aviator seats behind Pilot (+0.36) & Co-Pilot (-0.36)
      - Frameless panoramic canopy side pillars & overhead console (NO center mullion blocking faces!)
      - Backlit instrument glare-shield with dual MFDs + pulsing crimson `ENG OUT` annunciator strip
      - Animated hero **Collective Pitch Lever** (`Char_Heli_CollectiveLever`) tracking `C.collective_level_at(f)`!
      - Animated center **Cyclic Stick** (`Char_Heli_CyclicStick`)
    """
    mat_dark = _make_mat("Mat_Cockpit_DarkAlum", (0.038, 0.042, 0.048), metallic=0.68, roughness=0.38)
    mat_seat = _make_mat("Mat_Cockpit_SeatFabric", (0.06, 0.065, 0.075), roughness=0.82)
    mat_white = _make_mat("Mat_Cockpit_TrimWhite", (0.86, 0.88, 0.90), metallic=0.08, roughness=0.28, coat=0.4)
    mat_red = _make_mat("Mat_Cockpit_RescueRed", (0.82, 0.055, 0.045), metallic=0.14, roughness=0.26, coat=0.5)
    mat_mfd = _make_mat(
        "Mat_Cockpit_MFDGreen", (0.08, 0.92, 0.56), roughness=0.18,
        emit_rgb=(0.12, 0.96, 0.60), emit_str=1.55,
    )
    mat_yel = _make_mat(
        "Mat_Cockpit_AmberGrip", (0.98, 0.72, 0.08), roughness=0.30,
        emit_rgb=(1.00, 0.74, 0.10), emit_str=0.35,
    )

    # Pulsing crimson `ENG OUT` / `MASTER CAUTION` annunciator strip material
    mat_warn = bpy.data.materials.new("Mat_Cockpit_EngOutWarn")
    mat_warn.use_nodes = True
    bsdf_w = mat_warn.node_tree.nodes["Principled BSDF"]
    bsdf_w.inputs["Base Color"].default_value = (0.96, 0.08, 0.04, 1.0)
    bsdf_w.inputs["Roughness"].default_value = 0.20
    bsdf_w.inputs["Emission Color"].default_value = (1.00, 0.12, 0.05, 1.0)
    for f in range(1, 241, 6):
        pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin((f - 14) * 0.45)) if f >= 14 else 0.15
        bsdf_w.inputs["Emission Strength"].default_value = 1.85 * pulse
        bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)
    bsdf_w.inputs["Emission Strength"].default_value = 1.15
    bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=1291)
    bsdf_w.inputs["Emission Strength"].keyframe_insert("default_value", frame=C.B5_COCKPIT_END)

    bm = bmesh.new()
    # Cockpit floor deck & center avionics pedestal
    _add_box_bm(bm, (0.0, 0.15, -0.04), (1.76, 1.45, 0.08), mat_idx=0, bevel_w=0.01)
    _add_box_bm(bm, (0.0, -0.02, 0.48), (0.24, 0.62, 0.76), mat_idx=0, bevel_w=0.012)

    # Dual rescue aviator bucket seats (behind Pilot +0.36 & Co-Pilot -0.36 at Y=+0.54)
    for sx in (-0.36, +0.36):
        _add_box_bm(bm, (sx, 0.34, 0.42), (0.46, 0.46, 0.10), mat_idx=1, bevel_w=0.016)
        _add_box_bm(bm, (sx, 0.54, 0.96), (0.44, 0.10, 0.88), mat_idx=1, bevel_w=0.016, rot_deg=(-7.0, 0.0, 0.0))
        _add_box_bm(bm, (sx, 0.57, 1.44), (0.26, 0.10, 0.22), mat_idx=0, bevel_w=0.012)

    # Low sculpted instrument glare-shield (-Y in front of crew, low at Z=0.96..1.10 so faces & hands stay clear!)
    _add_box_bm(bm, (0.0, -0.34, 0.96), (1.36, 0.34, 0.24), mat_idx=0, bevel_w=0.018, rot_deg=(8.0, 0.0, 0.0))
    # Dual glowing green MFD tactical flight displays
    for sx in (-0.30, +0.30):
        _add_box_bm(bm, (sx, -0.20, 0.98), (0.24, 0.02, 0.16), mat_idx=4, rot_deg=(-18.0, 0.0, 0.0))
    # Center glowing crimson `ENG OUT` / `MASTER CAUTION` annunciator strip (Z = 1.105 m!)
    _add_box_bm(bm, (0.0, -0.28, 1.105), (0.34, 0.055, 0.052), mat_idx=6, bevel_w=0.005, rot_deg=(-14.0, 0.0, 0.0))

    # Swept outer rescue canopy pillars & overhead white-and-red cockpit roof arch (ZERO center mullion!)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sx * 0.78, 0.12, 0.86), (0.12, 1.25, 0.14), mat_idx=2, bevel_w=0.014)
        _add_strut_bm(bm, Vector((sx * 0.74, -0.34, 0.98)), Vector((sx * 0.62, 0.18, 1.88)), 0.036, mat_idx=3, segs=10)
        _add_strut_bm(bm, Vector((sx * 0.74, 0.68, 0.98)), Vector((sx * 0.62, 0.56, 1.88)), 0.034, mat_idx=2, segs=10)
    # Overhead circuit-breaker eyebrow console arch (high at Z=1.88 m, clear of heads!)
    _add_box_bm(bm, (0.0, 0.36, 1.90), (1.28, 0.52, 0.10), mat_idx=0, bevel_w=0.016)
    _add_box_bm(bm, (0.0, 0.24, 1.85), (0.44, 0.18, 0.04), mat_idx=3, bevel_w=0.006)

    cockpit_obj = _new_bmesh_obj(
        "Char_Heli_CockpitFrame", bm,
        [mat_dark, mat_seat, mat_white, mat_red, mat_mfd, mat_yel, mat_warn],
        Matrix.Identity(4), col, smooth=False,
    )
    _key_visibility([cockpit_obj], _COCKPIT_WINDOWS, total_frames)

    # HERO PROP: Animated Collective Pitch Lever (slammed straight down in Beat 1, yanked UP in Beat 5A!)
    bm_col = bmesh.new()
    _add_strut_bm(bm_col, Vector((0.0, 0.0, 0.0)), Vector((0.0, -0.38, 0.10)), 0.018, mat_idx=0, segs=12)
    # High-contrast twist-throttle & yellow-black switch head box so viewers track its motion instantly!
    _add_strut_bm(bm_col, Vector((0.0, -0.38, 0.10)), Vector((0.0, -0.52, 0.13)), 0.026, mat_idx=1, segs=12)
    _add_box_bm(bm_col, (0.0, -0.55, 0.15), (0.056, 0.072, 0.056), mat_idx=2, bevel_w=0.006)
    _add_box_bm(bm_col, (0.0, -0.54, 0.185), (0.028, 0.024, 0.018), mat_idx=3, bevel_w=0.002)
    coll_obj = _new_bmesh_obj(
        "Char_Heli_CollectiveLever", bm_col, [mat_dark, mat_seat, mat_yel, mat_red],
        Matrix.Translation(Vector(C.STAGE["cockpit_collective_pivot"])), col, smooth=True,
    )
    _key_visibility([coll_obj], _COCKPIT_WINDOWS, total_frames)
    for f in list(range(1, 241, 4)) + list(range(1291, C.B5_COCKPIT_END + 1, 4)):
        lvl = C.collective_level_at(float(f))
        # lvl=1.0 -> pitched +26 deg UP, lvl=0.0 -> slammed -18 deg DOWN!
        pitch_deg = C.lerp(-18.0, 26.0, lvl)
        coll_obj.rotation_euler = Euler((math.radians(pitch_deg), 0.0, 0.0), "XYZ")
        coll_obj.keyframe_insert("rotation_euler", frame=f)

    # Animated Pilot Cyclic Control Stick (between Pilot's knees at X=+0.36)
    bm_cyc = bmesh.new()
    _add_strut_bm(bm_cyc, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.03, 0.42)), 0.015, mat_idx=0, segs=12)
    _add_box_bm(bm_cyc, (0.0, 0.04, 0.44), (0.044, 0.054, 0.095), mat_idx=1, bevel_w=0.008)
    _add_box_bm(bm_cyc, (0.0, 0.02, 0.48), (0.020, 0.020, 0.018), mat_idx=3, bevel_w=0.002)
    cyc_obj = _new_bmesh_obj(
        "Char_Heli_CyclicStick", bm_cyc, [mat_dark, mat_seat, mat_yel, mat_red],
        Matrix.Translation(Vector(C.STAGE["cockpit_cyclic_pivot"])), col, smooth=True,
    )
    _key_visibility([cyc_obj], _COCKPIT_WINDOWS, total_frames)
    for f in list(range(1, 241, 6)) + list(range(1291, C.B5_COCKPIT_END + 1, 6)):
        if f <= 240:
            # Beat 1: forward pitch-down stabilization into autorotation glide
            p_deg = C.lerp(0.0, -11.0, C.smootherstep(85.0, 165.0, float(f)))
            r_deg = 4.5 * math.sin(f * 0.14)
        else:
            # Beat 5A: decisive aft cyclic pull for 30-foot nose-up flare!
            p_deg = C.lerp(-8.0, 14.0, C.smootherstep(1335.0, 1425.0, float(f)))
            r_deg = -3.0 * math.sin((f - 1291) * 0.10)
        cyc_obj.rotation_euler = Euler((math.radians(p_deg), math.radians(r_deg), 0.0), "XYZ")
        cyc_obj.keyframe_insert("rotation_euler", frame=f)

    return {"cockpit": cockpit_obj, "collective": coll_obj, "cyclic": cyc_obj}


def _build_aviator(
    name: str,
    seat_xy: tuple[float, float],
    yaw_deg: float,
    iris_major: tuple[float, float, float],
    iris_minor: tuple[float, float, float],
    suit_rgb: tuple[float, float, float],
    stripe_rgb: tuple[float, float, float],
    is_pilot: bool,
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    cfg = {
        "gender": 1.0,
        "age": 0.42 if is_pilot else 0.36,
        "muscle": 0.58,
        "weight": 0.45,
        "height": 0.56,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": 1.78,
        "skin_assets": ["middleage_caucasian_male.mhmat", "young_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo"],
        "hair_tint_rgb": (0.09, 0.07, 0.05) if is_pilot else (0.16, 0.11, 0.07),
        "hair_tint_fac": 0.72,
        "hair_value_boost": 0.72,
        "hair_roughness": 0.32,
        "brow_tint_fac": 0.62,
        "iris_major_rgb": iris_major,
        "iris_minor_rgb": iris_minor,
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": suit_rgb,
        "shirt_tint_fac": 0.88,
        "shirt_sat_override": 0.42,
        "shirt_value_boost": 0.78,
        "shirt_roughness": 0.52,
        "shirt_sheen": 0.10,
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (suit_rgb[0] * 0.8, suit_rgb[1] * 0.8, suit_rgb[2] * 0.8),
        "pants_tint_fac": 0.86,
        "pants_sat_override": 0.35,
        "pants_value_boost": 0.70,
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
    res = mc.create_realistic_scientist(name=name, floor_z=0.0, start_xy=(0.0, 0.0), yaw_deg=0.0, cfg=cfg)
    rig, body, equipped = res["rig"], res["body"], res["equipped"]
    base_z = float(res["base_z"])

    _link(rig, col)
    _link(body, col)
    for o in equipped.values():
        if o is not None:
            _link(o, col)
    _enforce_v12_eyes(equipped)

    rig.location = Vector((0.0, 0.0, base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.view_layer.update()

    _apply_underclothes_shrink(body, rig)
    helmet_obj = _build_rescue_helmet(f"Char_{name}_Helmet", rig, col, stripe_rgb=stripe_rgb)

    # Seated vertical offset so eye height sits cleanly at C.CREW["pilot_eye_z"] (1.38 m)
    seated_z_off = -0.27
    kfs: set[int] = set(range(1, 241, 6)) | set(range(1291, C.B5_COCKPIT_END + 1, 6))
    kfs.update({1, 42, 105, 125, 150, 240, 1291, 1340, 1385, 1425, C.B5_COCKPIT_END})

    for f in sorted(kfs):
        ff = float(f)
        rig.location = Vector((seat_xy[0], seat_xy[1], base_z + seated_z_off))
        rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)

        if is_pilot:
            coll = C.collective_level_at(ff)
            if f <= 240:
                # Beat 1: Engine dies -> startled glance at ENG OUT -> left arm SLAMS collective straight down!
                slam_u = C.smootherstep(105.0, 150.0, ff)
                head_p = C.lerp(10.0, 16.0, C.smoothstep(30.0, 95.0, ff)) - 12.0 * C.smoothstep(160.0, 225.0, ff)
                head_y = C.lerp(12.0, -6.0, slam_u)
                _pose_seated_body(rig, torso_fwd_deg=8.0 + 4.0 * slam_u, torso_twist_deg=C.lerp(6.0, -4.0, slam_u), head_pitch_deg=head_p, head_yaw_deg=head_y, frame=f)
                # Left arm tracks collective lever slamming from high (-34 deg) to full down (-10 deg)!
                ua_l = (C.lerp(-14.0, -42.0, coll), 12.0, -14.0)
                la_l = (C.lerp(-35.0, -72.0, coll), 0.0, 18.0)
                # Right arm steady on cyclic control stick
                ua_r = (-32.0, -10.0, 10.0)
                la_r = (-58.0, 0.0, -14.0)
                _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "fist", "fist", frame=f)
                _apply_actor_face(rig, f, jaw_deg=4.5 * C.smoothstep(18.0, 55.0, ff), brow_lift=0.65, smile=0.0, eye_yaw_deg=head_y * 0.4, eye_pitch_deg=-6.0, force_alert_lids=True)
            else:
                # Beat 5A: Shallow-DOF final flare close-up -> Pilot HAULS collective lever UP & leans into flare!
                pull_u = C.smootherstep(1340.0, 1425.0, ff)
                _pose_seated_body(rig, torso_fwd_deg=C.lerp(10.0, 2.0, pull_u), torso_twist_deg=C.lerp(-4.0, 4.0, pull_u), head_pitch_deg=C.lerp(8.0, -4.0, pull_u), head_yaw_deg=C.lerp(-5.0, 6.0, pull_u), frame=f)
                ua_l = (C.lerp(-14.0, -48.0, coll), 10.0, -14.0)
                la_l = (C.lerp(-34.0, -78.0, coll), 0.0, 16.0)
                ua_r = (C.lerp(-34.0, -24.0, pull_u), -10.0, 10.0)
                la_r = (C.lerp(-56.0, -74.0, pull_u), 0.0, -14.0)
                _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "fist", "fist", frame=f)
                _apply_actor_face(rig, f, jaw_deg=2.0, brow_lift=0.45, smile=0.0, eye_yaw_deg=-3.0, eye_pitch_deg=-2.0, force_alert_lids=True)
        else:
            if f <= 240:
                # Co-Pilot Beat 1: Alarmed look at crimson ENG OUT strip -> points down toward rooftop helipad!
                pt_u = C.smootherstep(70.0, 145.0, ff)
                _pose_seated_body(rig, torso_fwd_deg=10.0 + 4.0 * pt_u, torso_twist_deg=C.lerp(-12.0, -18.0, pt_u), head_pitch_deg=C.lerp(8.0, 18.0, pt_u), head_yaw_deg=C.lerp(-14.0, 8.0, pt_u), frame=f)
                ua_l = (-28.0, 14.0, -10.0)
                la_l = (-52.0, 0.0, 10.0)
                ua_r = (C.lerp(-30.0, -54.0, pt_u), -14.0, 16.0)
                la_r = (C.lerp(-52.0, -24.0, pt_u), 0.0, -12.0)
                _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "open", "point", frame=f)
                _apply_actor_face(rig, f, jaw_deg=8.5 * math.sin(C.clamp((ff - 40.0) / 140.0, 0.0, 1.0) * math.pi), brow_lift=0.75, smile=0.0, eye_yaw_deg=-6.0, eye_pitch_deg=-8.0, force_alert_lids=True)
            else:
                # Co-Pilot Beat 5A: Calls out "30 feet!" over comms mic while bracing on glare-shield!
                call_u = C.smootherstep(1310.0, 1410.0, ff)
                _pose_seated_body(rig, torso_fwd_deg=11.0, torso_twist_deg=-10.0, head_pitch_deg=C.lerp(12.0, 2.0, call_u), head_yaw_deg=C.lerp(-12.0, -4.0, call_u), frame=f)
                ua_l = (-34.0, 12.0, -8.0)
                la_l = (-56.0, 0.0, 12.0)
                ua_r = (-36.0, -10.0, 10.0)
                la_r = (-60.0, 0.0, -10.0)
                _pose_arms_fk(rig, ua_l, la_l, ua_r, la_r, "open", "open", frame=f)
                _apply_actor_face(rig, f, jaw_deg=7.5 * (0.5 + 0.5 * math.sin(ff * 0.35)), brow_lift=0.55, smile=0.0, eye_yaw_deg=-5.0, eye_pitch_deg=-4.0, force_alert_lids=True)

    actor_objs = [body, helmet_obj, *(o for o in equipped.values() if o is not None)]
    _key_visibility(actor_objs, _COCKPIT_WINDOWS, total_frames)
    return {"rig": rig, "body": body, "equipped": equipped, "helmet": helmet_obj}


def build_characters(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Build 1:1 Rescue Cockpit + 2 Photoreal 4K MPFB2 Rescue Aviators (Pilot & Co-Pilot)."""
    cockpit = _build_cockpit_set(col, total_frames)
    px, py, _ = C.CREW["pilot_pos"]
    cx, cy, _ = C.CREW["copilot_pos"]

    pilot = _build_aviator(
        name="Heli_Pilot",
        seat_xy=(px, py),
        yaw_deg=-8.0,
        iris_major=(0.18, 0.44, 0.68),    # Crisp steel-blue aviator irises
        iris_minor=(0.07, 0.18, 0.30),
        suit_rgb=(0.08, 0.13, 0.22),      # Coast Guard rescue navy flight suit
        stripe_rgb=(0.84, 0.06, 0.05),    # Crimson rescue helmet stripe
        is_pilot=True,
        col=col,
        total_frames=total_frames,
    )
    copilot = _build_aviator(
        name="Heli_CoPilot",
        seat_xy=(cx, cy),
        yaw_deg=+10.0,
        iris_major=(0.44, 0.30, 0.14),    # Independent warm-hazel irises
        iris_minor=(0.18, 0.12, 0.05),
        suit_rgb=(0.10, 0.15, 0.24),
        stripe_rgb=(0.94, 0.52, 0.08),    # High-vis rescue orange-gold stripe
        is_pilot=False,
        col=col,
        total_frames=total_frames,
    )
    return {"cockpit": cockpit, "pilot": pilot, "copilot": copilot}
