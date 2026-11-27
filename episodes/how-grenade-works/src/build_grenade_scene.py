"""
Master Scene Assembler: How a Grenade Actually Works — Hollywood Action + Matrix Bullet-Time 4K X-Ray Short
Unifies all 5 specialist production modules:
  1. battlefield_env.py  — Golden-hour desert tactical outpost + 4K MakeHuman Hero Soldier & 2 Rival Sentries
  2. grenade_asset.py    — Hyper-realistic M67 cast-iron grenade + local-space X-ray cutaway + 3D pointer labels
  3. cinema_camera.py    — Dynamic 6-beat 9:16 portrait camera rig (50mm Hero -> 35mm Throw -> 60mm X-Ray -> 32mm Blast)
  4. director_shotlist   — Synchronized 720-frame (30.0s @ 24fps) continuity milestones
  5. generate_audio.py   — Studio Edge-TTS narration + 19 battlefield/mechanical Foley SFX + karaoke subtitles

Follows AGENTS.md V10 Cinema Standard (AgX color management, photoreal eyes, shadow-bypass X-ray glass,
open-ended additive optics, exact DOF focus, 100% verified 9:16 portrait framing).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent

for p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import battlefield_env  # noqa: E402
import cinema_camera  # noqa: E402
import grenade_asset  # noqa: E402

FPS = 24
TOTAL_FRAMES = 720
FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"

# Exact PullPin_Rig & chrome pull-ring geometry constants (synced with grenade_asset.py)
HINGE_Z = 0.0630
PIN_LOCAL_OFFSET = Vector((0.0, 0.0, HINGE_Z))
BITE_PT_IN_PIN = Vector((-0.0488, 0.0, -0.0018))
BITE_PT_IN_GRENADE = Vector((-0.0488, 0.0, 0.0612))
PIN_COM_LOCAL = Vector((-0.0220, 0.0, -0.0010))


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _set_bezier(obj) -> None:
    ad = getattr(obj, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if fc.data_path.startswith("hide_"):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int = TOTAL_FRAMES,
) -> None:
    """Keyframe `hide_viewport` & `hide_render` with CONSTANT interpolation."""
    all_frames = {1, total_frames}
    for s, e in visible_ranges:
        s_c = max(1, min(total_frames, int(s)))
        e_c = max(1, min(total_frames, int(e)))
        if s_c > 1:
            all_frames.add(s_c - 1)
        all_frames.add(s_c)
        all_frames.add(e_c)
        if e_c < total_frames:
            all_frames.add(e_c + 1)

    for f in sorted(all_frames):
        is_vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not is_vis
        obj.hide_render = not is_vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)

    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path.startswith("hide_"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _sample_hero_head_world_quats(
    hero_rig: bpy.types.Object | None,
    start_f: int = 1,
    end_f: int = 108,
) -> dict[int, Quaternion]:
    """Fallback evaluator for Hero's `head` pose-bone world quaternion when not pre-baked in `soldiers`."""
    quats: dict[int, Quaternion] = {}
    if hero_rig is None or hero_rig.type != "ARMATURE" or "head" not in hero_rig.pose.bones:
        return quats
    scene = bpy.context.scene
    prev_f = scene.frame_current
    pb_head = hero_rig.pose.bones["head"]
    for f in range(start_f, end_f + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        mw = hero_rig.matrix_world @ pb_head.matrix
        quats[f] = mw.to_quaternion().normalized()
    scene.frame_set(prev_f)
    bpy.context.view_layer.update()
    return quats


def _make_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def configure_render(samples: int, scale: int, device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = scale
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95

    # AgX Cinema Color Management
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    cycles = scene.cycles
    cycles.samples = samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.025
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 3
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 8
    cycles.transparent_max_bounces = 14
    cycles.volume_bounces = 0
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.sample_clamp_direct = 12.0
    cycles.sample_clamp_indirect = 5.0

    # Subtle action motion blur (clean during throw & flying spoon, sharp in bullet-time)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.28

    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device.upper() == "CPU":
        cycles.device = "CPU"
        print("[Render] Using CPU Cycles + OIDN")
        return

    chosen = None
    for backend in ([device.upper(), "OPTIX", "CUDA"] if device.upper() in {"OPTIX", "CUDA"} else ["OPTIX", "CUDA"]):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                cycles.device = "GPU"
                chosen = backend
                break
        except Exception:
            continue

    if chosen:
        print(f"[Render] Using GPU Cycles ({chosen}) + OIDN")
    else:
        cycles.device = "CPU"
        print("[Render] No GPU found -> CPU Cycles + OIDN")


def animate_grenade_world_path(
    grenade_root: bpy.types.Object,
    hero_socket_positions: dict[int, Vector],
    release_frame: int = 134,
    total_frames: int = TOTAL_FRAMES,
    hero_bite_positions: dict[int, Vector] | None = None,
    hero_mouth_positions: dict[int, Vector] | None = None,
) -> dict[int, Matrix]:
    """
    Seamlessly drives Grenade_Root from Hero Soldier's gripping right hand (f=1..134):
      - f=1..46: aims local -X pull-ring directly from socket toward Hero's exact teeth bite line
        (horizontally level in XY, gentle pitch clamped to <= 8 deg so the M67 grenade stays upright!).
      - f=18..28 (THE BITE): locks `Grenade_Root.location` onto `bite_pos - (R_g @ BITE_PT_IN_GRENADE)`
        with 0.000m precision so the chrome pull-ring outer rim sits literally between Hero's upper &
        lower front teeth (smoothly blending from/to `hero_socket_world_positions` over f=10..18 & f=28..42).
      - f=47..134: steady armed hold through Hero's Rajinikanth smirk & pin spit-flick -> athletic throw whip.
      - f=134..440: Matrix Bullet-Time mid-air X-Ray glide.
      - f=440..495: Ballistic dive into Enemy Sandbag Bunker + dirt micro-hop (f=495..508).
      - f=612..720: Wide-shot 48-shard blast scale boost.

    Returns exact world matrices `grenade_world_matrices[f]` for `f = 1..release_frame` so
    `animate_pull_pin_world_physics()` shares 0.000mm parity during Phase 1 (f=1..28).
    """
    grenade_root.rotation_mode = "XYZ"

    socket_keys = sorted(hero_socket_positions.keys()) if hero_socket_positions else []
    rel_pos = Vector(hero_socket_positions.get(release_frame, (0.24, 0.42, 1.62)))
    bt_entry_pos = Vector(cinema_camera.grenade_pos_at(152))

    bite_src = hero_bite_positions or hero_mouth_positions
    bite_keys = sorted(bite_src.keys()) if bite_src else []
    default_bite = Vector(cinema_camera.HERO_MOUTH_LOCAL)

    def _socket_at(frame: int) -> Vector:
        if hero_socket_positions:
            if frame in hero_socket_positions:
                return Vector(hero_socket_positions[frame])
            nearest = min(socket_keys, key=lambda k: abs(k - frame))
            return Vector(hero_socket_positions[nearest])
        return rel_pos.copy()

    bite_lip_offset = Vector((0.0, 0.008, -0.007))

    def _bite_at(frame: int) -> Vector:
        if bite_src:
            if frame in bite_src:
                return Vector(bite_src[frame]) + bite_lip_offset
            nearest = min(bite_keys, key=lambda k: abs(k - frame))
            return Vector(bite_src[nearest]) + bite_lip_offset
        return default_bite.copy() + bite_lip_offset

    bite_blend_in_start = 10
    bite_lock_start = 18
    bite_lock_end = 28
    bite_blend_out_end = 42
    pin_aim_end = min(46, release_frame)
    max_pitch_rad = math.radians(8.0)

    # 1. Held in Hero's right hand (f = 1..134)
    prev_rot_z: float | None = None
    last_pin_rot_y = 0.0
    last_pin_rot_z = math.radians(65.0)
    grenade_world_matrices: dict[int, Matrix] = {}

    for f in range(1, release_frame + 1):
        socket_p = _socket_at(f)

        if f <= pin_aim_end:
            bite_p = _bite_at(f)
            pin_origin = socket_p + Vector((0.0, 0.0, BITE_PT_IN_GRENADE.z))
            delta = bite_p - pin_origin
            horiz = math.hypot(delta.x, delta.y)
            if horiz > 0.015:
                # Under XYZ Euler R_z(rz) @ R_y(ry) @ R_x(0), local -X maps to:
                #   (-cos(rz)*cos(ry), -sin(rz)*cos(ry), +sin(ry))
                # Setting rz = atan2(-dy, -dx) & ry clamped to +-8 deg keeps grenade upright while
                # aiming local -X straight horizontally from socket_pos toward Hero's front teeth!
                rot_z = math.atan2(-delta.y, -delta.x)
                rot_y = max(
                    -max_pitch_rad,
                    min(max_pitch_rad, math.atan2(delta.z, max(horiz, 0.03))),
                )
            else:
                rot_z = prev_rot_z if prev_rot_z is not None else last_pin_rot_z
                rot_y = last_pin_rot_y

            if prev_rot_z is not None:
                while rot_z - prev_rot_z > math.pi:
                    rot_z -= 2.0 * math.pi
                while rot_z - prev_rot_z < -math.pi:
                    rot_z += 2.0 * math.pi
            prev_rot_z = rot_z
            last_pin_rot_y = rot_y
            last_pin_rot_z = rot_z
            tilt_x = 0.0
        else:
            # Smoothly settle from extraction axis into steady armed hold (f=47..108) & throw whip (f=109..134)
            u_settle = _smoothstep((f - pin_aim_end) / 18.0)
            u_throw = _smoothstep((f - 108) / max(1.0, float(release_frame - 108))) if f > 108 else 0.0
            tilt_x = math.radians(8.0 * u_settle + 18.0 * u_throw)
            rot_y = last_pin_rot_y * (1.0 - u_throw) + math.radians(-10.0) * u_throw
            rot_z = last_pin_rot_z

        rot_euler = Euler((tilt_x, rot_y, rot_z), "XYZ")
        r_g = rot_euler.to_matrix()

        # Exact teeth-bite contact blend: 0.0 (f<=10) -> 1.0 (f=18..28) -> 0.0 (f>=42)
        if f <= bite_blend_in_start:
            w_bite = 0.0
        elif f < bite_lock_start:
            w_bite = _smoothstep((f - bite_blend_in_start) / float(bite_lock_start - bite_blend_in_start))
        elif f <= bite_lock_end:
            w_bite = 1.0
        elif f < bite_blend_out_end:
            w_bite = 1.0 - _smoothstep((f - bite_lock_end) / float(bite_blend_out_end - bite_lock_end))
        else:
            w_bite = 0.0

        if w_bite > 0.0:
            exact_bite_loc = _bite_at(f) - (r_g @ BITE_PT_IN_GRENADE)
            loc = socket_p.lerp(exact_bite_loc, w_bite)
        else:
            loc = socket_p

        grenade_root.location = loc
        grenade_root.rotation_euler = rot_euler
        grenade_root.scale = (1.0, 1.0, 1.0)
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)
        if f in (1, release_frame):
            grenade_root.keyframe_insert("scale", frame=f)

        grenade_world_matrices[f] = Matrix.Translation(loc) @ r_g.to_4x4()

    rel_rx = math.radians(26.0)
    rel_ry = math.radians(-10.0)
    rel_rz = last_pin_rot_z
    bt_rz_base = round(rel_rz / (2.0 * math.pi)) * (2.0 * math.pi)

    # 2. Explosive release transition into Matrix Bullet-Time (f = 135..151)
    for f in range(release_frame + 1, 152):
        u = (f - release_frame) / (152.0 - release_frame)
        # Parabolic arc from exact hand release to bullet-time entry point
        p = rel_pos.lerp(bt_entry_pos, u)
        p.z += 0.22 * math.sin(math.pi * u)
        grenade_root.location = p
        # Smoothly spin into front cutaway alignment (rot_z = 0° mod 360°) facing the -Y bullet-time macro camera!
        su = _smoothstep(u)
        grenade_root.rotation_euler = Euler((
            rel_rx * (1.0 - su) + math.radians(4.0) * su,
            rel_ry * (1.0 - su),
            rel_rz * (1.0 - su) + bt_rz_base * su,
        ), "XYZ")
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 3. Matrix Bullet-Time X-Ray Glide + Ballistic Dive to Enemy Sandbags (f = 152..495)
    for f in range(152, 496):
        p = Vector(cinema_camera.grenade_pos_at(f))
        grenade_root.location = p
        if f <= 440:
            # Ultra-subtle suspended bullet-time float (keeps X-ray cutaway & 3D labels 100% locked to lens!)
            t_bt = (f - 152) / 288.0
            rx = math.radians(4.0 * math.cos(t_bt * math.pi * 1.5))
            ry = math.radians(5.0 * math.sin(t_bt * math.pi * 2.0))
            rz = bt_rz_base + math.radians(3.0 * math.sin(t_bt * math.pi))
        else:
            # Accelerating dive into enemy sandbag ground zero, stabilizing upright at landing
            u = (f - 440) / 55.0
            rx = math.radians(-12.0 * math.sin(math.pi * u))
            ry = math.radians(14.0 * math.sin(math.pi * u))
            rz = bt_rz_base
        grenade_root.rotation_euler = Euler((rx, ry, rz), "XYZ")
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 4. Crisp Ground-Zero Dirt Micro-Hop & Settle at Enemy Bunker (f = 496..611)
    land_pos = Vector(cinema_camera.BUNKER_DEFAULT_POS)
    for f in range(496, 612):
        hop = 0.0
        tilt = 0.0
        if f <= 508:
            u = (f - 495) / 13.0
            hop = 0.048 * math.sin(math.pi * u) * (1.0 - 0.35 * u)
            tilt = math.radians(14.0 * math.sin(math.pi * u))
        grenade_root.location = Vector((land_pos.x, land_pos.y, land_pos.z + hop))
        grenade_root.rotation_euler = Euler((tilt, 0.0, bt_rz_base), "XYZ")
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 5. Beat 6 Wide-Shot Detonation Scale Boost (f = 611..720)
    # Clean 2.8x hero scale at Z=0.48 makes all 48 dark iron shards & shockwave ring pop alongside bunker & sentries!
    grenade_root.scale = (1.0, 1.0, 1.0)
    grenade_root.keyframe_insert("scale", frame=611)
    grenade_root.location = Vector((land_pos.x, land_pos.y, land_pos.z))
    grenade_root.keyframe_insert("location", frame=611)

    grenade_root.location = Vector((land_pos.x, land_pos.y, 0.44))
    grenade_root.scale = (2.6, 2.6, 2.6)
    grenade_root.keyframe_insert("location", frame=624)
    grenade_root.keyframe_insert("scale", frame=624)

    grenade_root.location = Vector((land_pos.x, land_pos.y, 0.56))
    grenade_root.scale = (3.0, 3.0, 3.0)
    grenade_root.keyframe_insert("location", frame=total_frames)
    grenade_root.keyframe_insert("scale", frame=total_frames)
    _set_bezier(grenade_root)

    return grenade_world_matrices


def animate_pull_pin_world_physics(
    pin_rig: bpy.types.Object,
    grenade_world_matrices: dict[int, Matrix],
    hero_bite_positions: dict[int, Vector] | None,
    hero_head_quats: dict[int, Quaternion] | None,
    total_frames: int = TOTAL_FRAMES,
) -> None:
    """
    Exact 3-Phase World-Space `PullPin_Rig` Physics & Rajinikanth-Style Spinning Teeth Spit-Flick:
      - Clears local animation from `grenade_asset.animate_grenade()` and unparents `PullPin_Rig`
        into World Space (`parent = None`, `matrix_parent_inverse = Identity`) so its world matrix
        is controlled frame-by-frame across `f = 1..94`.
      - Phase 1 (`f = 1..28`)  — SEATED INSIDE GRENADE FUZE:
        `M_pin(f) = M_grenade(f) @ Matrix.Translation((0.0, 0.0, 0.0630))` with Euler continuity.
        At `f = 28`, captures exact head-local attachment `(rel_pos_in_head, rel_quat_in_head)` for
        a mathematically 0.000mm / 0.000° transition into Phase 2.
      - Phase 2 (`f = 29..64`) — CLAMPED IN HERO'S TEETH (RIP-OUT + RAJINIKANTH SMIRK):
        Drives `PullPin_Rig` from live `(hero_bite_positions[f], hero_head_quats[f])` with
        `loc_pin = bite_pos - (q_pin.to_matrix() @ BITE_PT_IN_PIN)` so the chrome ring outer rim
        remains 100.0% glued between Hero's upper & lower front teeth during head whip-back
        (`f=28..46`) and swagger smirk (`f=46..64`), plus a natural +4° downward gravity sag of the
        free `+X` cotter-pin shaft once it clears the fuze hole (`f=40..64`).
      - Phase 3 (`f = 65..94`) — RAJINIKANTH-STYLE SPINNING TEETH SPIT-FLICK:
        At exact chin whip-flick peak (`f = 65`), launches cleanly away from Hero's face toward
        Screen Right & camera foreground (`v0 = (-0.65, +0.85, +0.42) m/s`, `g = -4.2 m/s²`) with
        high-speed `+1200°/s` tumbling spin around Y & Z about true center-of-mass (`PIN_COM_LOCAL`),
        exiting frame cleanly before hiding `PullPin_Rig` & all children for `f >= 95`.
    """
    # 0. Clear local animation from grenade_asset.animate_grenade() & unparent cleanly into World Space
    pin_rig.animation_data_clear()
    for ch in pin_rig.children_recursive:
        ch.animation_data_clear()
    pin_rig.parent = None
    pin_rig.matrix_parent_inverse = Matrix.Identity(4)
    pin_rig.rotation_mode = "XYZ"
    pin_rig.scale = (1.0, 1.0, 1.0)

    default_bite = Vector(cinema_camera.HERO_MOUTH_LOCAL)
    bite_keys = sorted(hero_bite_positions.keys()) if hero_bite_positions else []
    head_keys = sorted(hero_head_quats.keys()) if hero_head_quats else []

    # Slight lip-seam clearance offset (-7mm Z onto exact incisal lip opening, +8mm Y outward from teeth)
    # so the chrome pull-ring sits cleanly BETWEEN his lips/teeth without touching the upper philtrum or nose!
    bite_lip_offset = Vector((0.0, 0.008, -0.007))

    def _bite_at(frame: int) -> Vector:
        if hero_bite_positions:
            if frame in hero_bite_positions:
                return Vector(hero_bite_positions[frame]) + bite_lip_offset
            nearest = min(bite_keys, key=lambda k: abs(k - frame))
            return Vector(hero_bite_positions[nearest]) + bite_lip_offset
        return default_bite.copy() + bite_lip_offset

    def _head_quat_at(frame: int) -> Quaternion:
        if hero_head_quats:
            if frame in hero_head_quats:
                return Quaternion(hero_head_quats[frame]).normalized()
            nearest = min(head_keys, key=lambda k: abs(k - frame))
            return Quaternion(hero_head_quats[nearest]).normalized()
        return Quaternion((1.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # PHASE 1 (f = 1..28): SEATED INSIDE GRENADE FUZE
    # ------------------------------------------------------------------
    pin_offset_mat = Matrix.Translation(PIN_LOCAL_OFFSET)
    prev_euler: Euler | None = None
    loc_28 = Vector((0.0, 0.0, 0.0))
    q_28 = Quaternion((1.0, 0.0, 0.0, 0.0))

    for f in range(1, 29):
        m_pin = grenade_world_matrices[f] @ pin_offset_mat
        loc = m_pin.translation.copy()
        q_pin = m_pin.to_quaternion().normalized()
        euler = q_pin.to_euler("XYZ", prev_euler) if prev_euler is not None else q_pin.to_euler("XYZ")
        prev_euler = euler.copy()

        pin_rig.location = loc
        pin_rig.rotation_euler = euler
        pin_rig.keyframe_insert("location", frame=f)
        pin_rig.keyframe_insert("rotation_euler", frame=f)

        if f == 28:
            loc_28 = loc.copy()
            q_28 = q_pin.copy()

    # Record exact head-local attachment offset at f=28 for mathematically 0.000mm transition
    bite_28 = _bite_at(28)
    q_head_28 = _head_quat_at(28)
    q_head_28_inv = q_head_28.inverted()
    rel_quat_in_head = (q_head_28_inv @ q_28).normalized()
    rel_pos_in_head = q_head_28_inv @ (loc_28 - bite_28)
    handoff_err_mm = (loc_28 - (bite_28 - (q_28.to_matrix() @ BITE_PT_IN_PIN))).length * 1000.0

    # ------------------------------------------------------------------
    # PHASE 2 (f = 29..64): CLAMPED IN HERO'S TEETH (RIP-OUT + RAJINIKANTH SMIRK!)
    # ------------------------------------------------------------------
    prev_head_q = q_head_28.copy()

    def _teeth_clamped_pose(frame: int) -> tuple[Vector, Quaternion, Euler]:
        nonlocal prev_head_q
        bite_p = _bite_at(frame)
        q_head = _head_quat_at(frame)
        if q_head.dot(prev_head_q) < 0.0:
            q_head = -q_head
        prev_head_q = q_head.copy()

        q_base = (q_head @ rel_quat_in_head).normalized()
        # Once the cotter-pin shaft clears the fuze hole (f=36..54), roll the circular chrome ring
        # flat horizontally between his teeth (+78° local X) + subtle sideways swagger angle (-26° Z)
        # + natural downward dangle (+18° Y) — iconic Rajinikanth toothpick/ring silhouette with ZERO upper-lip poke!
        sag_u = _smoothstep((frame - 36.0) / 16.0) if frame > 36 else 0.0
        q_flat = Quaternion((1.0, 0.0, 0.0), math.radians(78.0 * sag_u))
        q_dangle = Quaternion((0.0, 1.0, 0.0), math.radians(18.0 * sag_u))
        q_side = Quaternion((0.0, 0.0, 1.0), math.radians(-26.0 * sag_u))
        q_pin = (q_base @ q_side @ q_dangle @ q_flat).normalized()
        # Extra 4mm outward clearance once horizontal so chrome ring gleams cleanly outside lips
        loc_pin = bite_p + Vector((0.0, 0.004 * sag_u, 0.0)) - (q_pin.to_matrix() @ BITE_PT_IN_PIN)
        euler_pin = q_pin.to_euler("XYZ", prev_euler) if prev_euler is not None else q_pin.to_euler("XYZ")
        return loc_pin, q_pin, euler_pin

    loc_46 = loc_28.copy()
    for f in range(29, 65):
        loc_pin, _q_pin, euler_pin = _teeth_clamped_pose(f)
        prev_euler = euler_pin.copy()

        pin_rig.location = loc_pin
        pin_rig.rotation_euler = euler_pin
        pin_rig.keyframe_insert("location", frame=f)
        pin_rig.keyframe_insert("rotation_euler", frame=f)
        if f == 46:
            loc_46 = loc_pin.copy()

    # ------------------------------------------------------------------
    # PHASE 3 (f = 65..94): RAJINIKANTH-STYLE SPINNING TEETH SPIT-FLICK!!
    # ------------------------------------------------------------------
    # Exact release pose at peak of Hero's chin whip-flick (f = 65), clearing lips cleanly downward/forward
    loc_65, _q_65, euler_65 = _teeth_clamped_pose(65)
    r_65 = euler_65.to_matrix()
    p_com_65 = loc_65 + (r_65 @ PIN_COM_LOCAL) + Vector((-0.016, 0.026, -0.008))

    # Launch cleanly OUTWARD from lips (+Y toward camera) & across Screen Right (-X) — NEVER crossing nose!
    v0 = Vector((-0.94, 1.12, 0.10))
    g_vec = Vector((0.0, 0.0, -3.8))
    spin_x_rad = math.radians(840.0)
    spin_z_rad = math.radians(1350.0)

    for f in range(65, 95):
        dt = (f - 65) / float(FPS)
        # Direct continuous Euler spin (+1350 deg/s Z + 840 deg/s X): crisp glittering chrome flip!
        euler_f = Euler(
            (
                euler_65.x + spin_x_rad * dt,
                euler_65.y,
                euler_65.z + spin_z_rad * dt,
            ),
            "XYZ",
        )
        r_f = euler_f.to_matrix()
        p_com_f = p_com_65 + (v0 * dt) + (0.5 * g_vec * (dt * dt))
        loc_f = p_com_f - (r_f @ PIN_COM_LOCAL)

        pin_rig.location = loc_f
        pin_rig.rotation_euler = euler_f
        pin_rig.keyframe_insert("location", frame=f)
        pin_rig.keyframe_insert("rotation_euler", frame=f)

    pin_rig.scale = (1.0, 1.0, 1.0)
    pin_rig.keyframe_insert("scale", frame=1)
    pin_rig.keyframe_insert("scale", frame=94)
    _set_bezier(pin_rig)

    # Visible ONLY f = 1..94; 100% hidden for f >= 95 on PullPin_Rig & ALL children (never reappears in Beats 2–6!)
    for obj in [pin_rig, *pin_rig.children_recursive]:
        _key_visibility(obj, [(1, 94)], total_frames=total_frames)

    fuze_46 = (grenade_world_matrices[46] @ pin_offset_mat).translation if 46 in grenade_world_matrices else loc_28
    extract_mm = (loc_46 - fuze_46).length * 1000.0
    print(
        f"[PinPhysics] World-Space 3-Phase Ready | f=28 bite_err={handoff_err_mm:.4f}mm "
        f"| rel_pos_head=({rel_pos_in_head.x:+.4f},{rel_pos_in_head.y:+.4f},{rel_pos_in_head.z:+.4f}) "
        f"| f=46 extracted={extract_mm:.1f}mm | f=65 spit_origin=({p_com_65.x:+.3f},{p_com_65.y:+.3f},{p_com_65.z:+.3f}) "
        f"| hidden f>=95",
        flush=True,
    )


def _enforce_hidden_from_frame(
    obj: bpy.types.Object,
    hide_from: int = 600,
    total_frames: int = TOTAL_FRAMES,
) -> None:
    """
    Guarantee `obj` is hidden (`hide_viewport=True`, `hide_render=True`) for all frames `>= hide_from`
    without clobbering any earlier visibility hide keys already set by `grenade_asset.py`.
    """
    prev_f = max(1, hide_from - 1)
    for prop in ("hide_viewport", "hide_render"):
        ad = obj.animation_data
        fc = ad.action.fcurves.find(prop) if (ad and ad.action) else None
        if fc is None:
            setattr(obj, prop, False)
            obj.keyframe_insert(prop, frame=1)
            obj.keyframe_insert(prop, frame=prev_f)
        else:
            if fc.evaluate(prev_f) < 0.5:
                setattr(obj, prop, False)
                obj.keyframe_insert(prop, frame=prev_f)
            for kp in fc.keyframe_points:
                if kp.co.x >= hide_from:
                    kp.co.y = 1.0
        setattr(obj, prop, True)
        obj.keyframe_insert(prop, frame=hide_from)
        obj.keyframe_insert(prop, frame=total_frames)

    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path in ("hide_viewport", "hide_render"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def _tune_beat6_explosion_cinematography(
    env_info: dict,
    grenade: dict,
    total_frames: int = TOTAL_FRAMES,
) -> None:
    """
    Prevent Beat 6 white-out while keeping dramatic Christopher-Nolan-style high-contrast
    fiery core + dark sunlit smoke plume + crisp flying iron shards + diving sentries!
    """
    # 1. Hide handheld Beat 1 safety_pin label so no sideways 3D text floats over Hero's fingers
    pin_lbl = grenade.get("labels", {}).get("safety_pin")
    if pin_lbl is not None:
        objs_to_hide = []
        if isinstance(pin_lbl, dict):
            for v in pin_lbl.values():
                if isinstance(v, bpy.types.Object):
                    objs_to_hide.append(v)
                    objs_to_hide.extend(list(v.children_recursive))
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, bpy.types.Object):
                            objs_to_hide.append(item)
                            objs_to_hide.extend(list(item.children_recursive))
        elif isinstance(pin_lbl, bpy.types.Object):
            objs_to_hide = [pin_lbl] + list(pin_lbl.children_recursive)
        for o in set(objs_to_hide):
            o.animation_data_clear()
            o.hide_viewport = True
            o.hide_render = True

    # 1b. Defensive Beat 6 guard: guarantee pull_pin (including chrome ring/cotter children) & spoon
    #     are 100% hidden by f >= 600 so neither ever appears near the Beat 6 explosion!
    parts = grenade.get("parts", {})
    for part_key in ("pull_pin", "spoon"):
        part_obj = parts.get(part_key)
        if isinstance(part_obj, bpy.types.Object):
            for o in [part_obj, *part_obj.children_recursive]:
                _enforce_hidden_from_frame(o, hide_from=600, total_frames=total_frames)

    # 2. Clamp ALL explosion / flash / burn point lights in Beat 6 so sunlit desert & bunker stay richly exposed!
    for light_obj in bpy.data.objects:
        if light_obj.type == "LIGHT" and light_obj.data.type == "POINT":
            ld = light_obj.data
            if ld.animation_data and ld.animation_data.action:
                for fc in ld.animation_data.action.fcurves:
                    if fc.data_path == "energy":
                        for kp in fc.keyframe_points:
                            if kp.co.x >= 610:
                                kp.co.y = min(kp.co.y, 140.0)
            else:
                ld.energy = min(ld.energy, 140.0)

    # 3. Hide milky translucent beige dust-cloud spheres that obscured the diving Rival Sentries behind the bunker,
    #    keeping ONLY the sleek ground-level shockwave dust ring (darkened & crisp) + our fiery turbulent fireball + 48 iron shards!
    for m_name in ("Mat_ExplosionDust_Outer", "Mat_ExplosionDust_Lit"):
        mat = bpy.data.materials.get(m_name)
        if mat and mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    n.inputs["Base Color"].default_value = (0.14, 0.09, 0.05, 1.0)
                    if "Emission Strength" in n.inputs:
                        n.inputs["Emission Strength"].default_value = 0.0
                elif n.type == "MAP_RANGE" and n.inputs["To Max"].default_value > 0.3:
                    n.inputs["To Max"].default_value = 0.35

    fx_dict = env_info.get("explosion_fx", {})
    for fxo in fx_dict.get("all_fx", []):
        if fxo.name != "Explosion_ShockwaveDustRing":
            fxo.animation_data_clear()
            fxo.hide_viewport = True
            fxo.hide_render = True
            continue
        if fxo.animation_data and fxo.animation_data.action:
            for fc in fxo.animation_data.action.fcurves:
                if fc.data_path == "scale":
                    for kp in fc.keyframe_points:
                        kp.co.y *= 0.55

    # Boost turbulent fiery orange contrast on grenade plasma core during Beat 6 wide shot
    for pname in ("Turbulent_Plasma_Core", "Turbulent_Plasma_Fire", "Additive_Shockwave_Ring"):
        pmat = bpy.data.materials.get(pname)
        if pmat and pmat.use_nodes:
            for n in pmat.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Color"].default_value = (1.0, 0.42, 0.04, 1.0)


def build_master_scene(samples: int, scale: int, device: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("Battlefield_Env")
    col_chars = _make_collection("Soldiers")
    col_grenade = _make_collection("Hero_Grenade")
    col_cam = _make_collection("Cinema_Camera")

    # 1. Golden-Hour Desert Tactical Outpost + Enemy Sandbag Bunker + Beat 6 Explosion Plume
    env_info = battlefield_env.build_battlefield(col=col_env, total_frames=TOTAL_FRAMES)

    # 2. Real 4K MakeHuman MPFB2 Hero Soldier + 2 Rival Armored Sentries
    soldiers = battlefield_env.build_soldiers(col=col_chars, total_frames=TOTAL_FRAMES)

    # 3. Hyper-Realistic M67 Grenade + Local-Space X-Ray Cutaway + 3D Component Pointer Labels
    grenade = grenade_asset.build_grenade_assembly(
        collection=col_grenade,
        include_labels=True,
        include_fx=True,
        font_path=str(FONT_PATH),
    )

    grenade_asset.animate_grenade(
        grenade,
        pin_pull=(28, 92),
        xray_scan=(154, 186),
        spoon_fly=(244, 322),
        striker_slam=(244, 294),
        fuse_burn=(355, 498),
        pressure_swell=(499, 611),
        fracture_frame=612,
        total_frames=TOTAL_FRAMES,
        show_labels=True,
    )

    hero_bite_positions = (
        soldiers.get("hero_bite_world_positions")
        or soldiers.get("hero_mouth_world_positions")
    )
    hero_head_quats = soldiers.get("hero_head_world_quats")
    if not hero_head_quats:
        hero_rig = soldiers.get("hero", {}).get("rig")
        hero_head_quats = _sample_hero_head_world_quats(hero_rig, start_f=1, end_f=108)

    grenade_world_matrices = animate_grenade_world_path(
        grenade_root=grenade["root"],
        hero_socket_positions=soldiers["hero_socket_world_positions"],
        release_frame=soldiers["release_frame"],
        total_frames=TOTAL_FRAMES,
        hero_bite_positions=hero_bite_positions,
        hero_mouth_positions=soldiers.get("hero_mouth_world_positions"),
    )

    animate_pull_pin_world_physics(
        pin_rig=grenade["parts"]["pull_pin"],
        grenade_world_matrices=grenade_world_matrices,
        hero_bite_positions=hero_bite_positions,
        hero_head_quats=hero_head_quats,
        total_frames=TOTAL_FRAMES,
    )

    # Tune Beat 1 handheld label cleanliness & Beat 6 Hollywood explosion contrast (zero white-out!)
    _tune_beat6_explosion_cinematography(env_info, grenade, total_frames=TOTAL_FRAMES)

    mac_l = bpy.data.lights.new("Grenade_Macro_Key", type="AREA")
    mac_l.shape = "DISK"
    mac_l.size = 0.22
    mac_l.color = (0.92, 0.96, 1.0)
    mac_obj = bpy.data.objects.new("Grenade_Macro_Key", mac_l)
    mac_obj.parent = grenade["root"]
    mac_obj.location = Vector((-0.14, -0.24, 0.16))
    mac_obj.rotation_euler = (Vector((0.0, 0.0, 0.02)) - mac_obj.location).to_track_quat("-Z", "Y").to_euler()
    col_grenade.objects.link(mac_obj)
    for f, e in [(1, 0.0), (150, 0.0), (160, 1.6), (608, 1.6), (614, 0.0), (TOTAL_FRAMES, 0.0)]:
        mac_l.energy = e
        mac_l.keyframe_insert("energy", frame=f)

    # 4. Hollywood 6-Beat Cinema Camera Rig
    cinema_camera.build_cinema_camera(
        col=col_cam,
        grenade_root=grenade["root"],
        hero_pos=Vector(battlefield_env.HERO_POS),
        bunker_pos=Vector(cinema_camera.BUNKER_DEFAULT_POS),
        total_frames=TOTAL_FRAMES,
    )

    bpy.context.scene.cycles.transparent_max_bounces = 14


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Build & Render How a Grenade Works Hollywood 4K Master Scene")
    p.add_argument("--frames-dir", type=str, default="/tmp/grenade_master_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--preview", action="store_true", help="Render 6 representative beat keyframes for visual QA")
    p.add_argument("--frames", type=str, default="", help="Optional comma-separated explicit frame list")
    p.add_argument("--save-blend", type=str, default="")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_master_scene(samples=args.samples, scale=args.scale, device=args.device)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"[Master] Saved blend -> {args.save_blend}")

    os.makedirs(args.frames_dir, exist_ok=True)
    scene = bpy.context.scene

    if args.frames:
        frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    elif args.preview:
        frames = [35, 195, 294, 420, 540, 665]
    else:
        frames = list(range(args.start_frame, args.end_frame + 1))

    for f in frames:
        scene.frame_set(f)
        out_path = os.path.join(args.frames_dir, f"frame_{f:04d}.jpg")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] frame_{f:04d}.jpg", flush=True)


if __name__ == "__main__":
    main()
