"""
Master Scene Integrator for Episode #1:
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (4K 60fps, 1800 frames).

Unifies all 8 specialist 3D modules strictly from `ejection_contract.py`:
  1. jet_cockpit_env.py     — Supersonic twin-tail fighter jet, cockpit tub, X-ray fuselage & Nishita sky
  2. canopy_fx.py           — Bubble canopy, zigzag MDC detonating cord & 48 bullet-time shatter shards
  3. ejection_seat_asset.py — Martin-Baker/ACES-II seat, telescoping rails, handle, garters & rocket FX
  4. pilot_character.py     — 4K MakeHuman MPFB2 pilot, HGU-55/P helmet, Nomex suit & X-ray body shader
  5. spine_xray.py          — 24-vertebrae S-curve X-ray spine + 23 compressing/color-shifting discs
  6. parachute_asset.py     — 1.5 m gyro drogue chute & 7.6 m ring-sail main parachute
  7. cinema_camera.py       — 6-beat 9:16 portrait camera + exact DOF focus tracking
  8. labels_3d.py           — 10 AgX-safe dark-obsidian plaque 3D callouts & live telemetry gauges

Follows AGENTS.md V11 & PRODUCTION_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent

for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canopy_fx  # noqa: E402
import cinema_camera  # noqa: E402
import ejection_contract as C  # noqa: E402
import ejection_seat_asset  # noqa: E402
import jet_cockpit_env  # noqa: E402
import labels_3d  # noqa: E402
import parachute_asset  # noqa: E402
import pilot_character  # noqa: E402
import spine_xray  # noqa: E402

FPS = C.FPS
TOTAL_FRAMES = C.TOTAL_FRAMES
FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"


def _make_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def configure_render(samples: int, scale: int, device: str) -> None:
    """Configure Blender 4.2 Cycles + AgX + OIDN render settings for 1080x1920 @ 60 fps."""
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
    for look_name in ("AgX - Medium High Contrast", "AgX - Punchy", "None"):
        try:
            scene.view_settings.look = look_name
            break
        except Exception:
            continue
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

    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.25

    if device.upper() == "CPU":
        cycles.device = "CPU"
        print("[Render] Using CPU Cycles + OIDN", flush=True)
        return

    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = None
    backends = [device.upper(), "OPTIX", "CUDA"] if device.upper() in {"OPTIX", "CUDA"} else ["OPTIX", "CUDA"]
    for backend in backends:
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
        print(f"[Render] Using GPU Cycles ({chosen}) + OIDN", flush=True)
    else:
        cycles.device = "CPU"
        print("[Render] No GPU found -> CPU Cycles + OIDN", flush=True)


def _aim_eye_catchlight_at_pilot(env: dict, pilot: dict, sample_frame: int = 100) -> None:
    """Re-aim the 0.42m softbox eye catchlight at the Pilot's evaluated Beat 1 eye midpoint."""
    lights = env.get("lights", {})
    catch_obj = lights.get("eye_catchlight")
    eyes_obj = pilot.get("equipped", {}).get("Eyes")
    if catch_obj is None or eyes_obj is None:
        return

    scene = bpy.context.scene
    prev_f = scene.frame_current
    scene.frame_set(sample_frame)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    eo = eyes_obj.evaluated_get(deps)
    em = eo.to_mesh()
    if len(em.vertices) > 0:
        pts = [eo.matrix_world @ v.co for v in em.vertices]
        eye_mid = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
        d = eye_mid - catch_obj.location
        if d.length > 1e-4:
            catch_obj.rotation_euler = d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")
    eo.to_mesh_clear()
    scene.frame_set(prev_f)
    bpy.context.view_layer.update()


def build_master_scene(samples: int, scale: int, device: str, build_labels_flag: bool = True) -> dict:
    """Assemble the complete 1800-frame Episode #1 Blender scene."""
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("Env_Jet_Sky")
    col_canopy = _make_collection("Canopy_MDC")
    col_seat = _make_collection("Ejection_Seat")
    col_pilot = _make_collection("Pilot_Character")
    col_spine = _make_collection("Spine_XRay")
    col_chute = _make_collection("Parachutes")
    col_cam = _make_collection("Cinema_Camera")
    col_labels = _make_collection("Labels_3D")

    print("[Scene 1/8] Building Jet Cockpit & Golden-Hour Sky Environment...", flush=True)
    env = jet_cockpit_env.build_environment(col_env, total_frames=TOTAL_FRAMES)

    print("[Scene 2/8] Building Bubble Canopy & MDC Shatter FX...", flush=True)
    canopy = canopy_fx.build_canopy(col_canopy, env["jet_root"], total_frames=TOTAL_FRAMES)

    print("[Scene 3/8] Building Ejection Seat, Telescoping Rails & Rocket FX...", flush=True)
    seat = ejection_seat_asset.build_ejection_seat(col_seat, total_frames=TOTAL_FRAMES)
    rails_root = seat.get("rails_root")
    if rails_root is not None:
        rails_root.parent = env["jet_root"]
        rails_root.matrix_parent_inverse = Matrix.Identity(4)

    print("[Scene 4/8] Building 4K MakeHuman MPFB2 Pilot & Choreography...", flush=True)
    pilot = pilot_character.build_pilot(col_pilot, total_frames=TOTAL_FRAMES)
    _aim_eye_catchlight_at_pilot(env, pilot, sample_frame=100)

    print("[Scene 5/8] Building Anatomical 24-Vertebrae Spine X-Ray Rig...", flush=True)
    spine = spine_xray.build_spine_xray(
        col_spine,
        pilot["carrier"],
        pilot["sacrum_local"],
        pilot["c1_local"],
        total_frames=TOTAL_FRAMES,
    )

    print("[Scene 6/8] Building Drogue & 7.6m Main Ring-Sail Parachutes...", flush=True)
    chutes = parachute_asset.build_parachutes(
        col_chute,
        seat["root"],
        pilot["carrier"],
        total_frames=TOTAL_FRAMES,
    )

    print("[Scene 7/8] Building 6-Beat Cinema Camera & DOF Rig...", flush=True)
    cam = cinema_camera.build_cinema_camera(col_cam, total_frames=TOTAL_FRAMES)

    labels = {}
    if build_labels_flag:
        print("[Scene 8/8] Building 3D Dark-Plaque Labels & Live Gauges...", flush=True)
        labels = labels_3d.build_labels(
            col_labels,
            str(FONT_PATH),
            seat["root"],
            pilot["carrier"],
            env["jet_root"],
            total_frames=TOTAL_FRAMES,
        )

    # Guarantee high transparent bounce budget for layered canopy + X-ray fuselage + X-ray body + spine
    bpy.context.scene.cycles.transparent_max_bounces = max(
        14, bpy.context.scene.cycles.transparent_max_bounces
    )
    print(f"[Master] Scene built in {time.time() - t0:.1f}s!", flush=True)

    return {
        "env": env,
        "canopy": canopy,
        "seat": seat,
        "pilot": pilot,
        "spine": spine,
        "chutes": chutes,
        "camera": cam,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Build & Render Episode #1 Fighter Jet Ejection (60fps)")
    p.add_argument("--frames-dir", type=str, default="/tmp/fje_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--preview", action="store_true", help="Render C.PREVIEW_FRAMES")
    p.add_argument("--beat1-dense", action="store_true", help="Render C.BEAT1_DENSE_FRAMES")
    p.add_argument("--frames", type=str, default="", help="Comma-separated frame list")
    p.add_argument("--save-blend", type=str, default="")
    p.add_argument("--skip-labels", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_master_scene(
        samples=args.samples,
        scale=args.scale,
        device=args.device,
        build_labels_flag=not args.skip_labels,
    )

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"[Master] Saved .blend -> {args.save_blend}", flush=True)

    os.makedirs(args.frames_dir, exist_ok=True)
    scene = bpy.context.scene

    if args.frames:
        frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    elif args.preview and args.beat1_dense:
        frames = sorted(set(C.PREVIEW_FRAMES + C.BEAT1_DENSE_FRAMES))
    elif args.preview:
        frames = list(C.PREVIEW_FRAMES)
    elif args.beat1_dense:
        frames = list(C.BEAT1_DENSE_FRAMES)
    else:
        frames = list(range(args.start_frame, args.end_frame + 1))

    for f in frames:
        t_f = time.time()
        scene.frame_set(f)
        out_path = os.path.join(args.frames_dir, f"frame_{f:04d}.jpg")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] frame_{f:04d}.jpg ({time.time() - t_f:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
