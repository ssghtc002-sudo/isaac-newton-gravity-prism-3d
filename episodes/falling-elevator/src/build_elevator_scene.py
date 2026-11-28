"""
Master Scene Integrator for Episode #16:
"Why Jumping in a Falling Elevator WON'T Save You" (4K 60fps, 1800 frames).

Unifies all 6 specialist 3D modules strictly from `elevator_contract.py`:
  1. shaft_env.py           — High-rise concrete shaft, scrolling story bays, T-rails, pit buffers & lighting
  2. elevator_cab_asset.py  — Brushed-steel elevator cab, hoist cables, safety clamps, LED & Beat 6 force grid
  3. passenger_character.py — 4K MakeHuman MPFB2 passenger + 6-beat athletic jump, 28 G impact & flat-back acting
  4. spine_skeleton_xray.py — Full-body anatomical X-ray skeleton + 28 G axial shockwave & Beat 6 load glow
  5. cinema_camera.py       — 6-beat 9:16 portrait camera + exact DOF focus tracking
  6. labels_3d.py           — 10 AgX-safe dark-obsidian plaque 3D callouts & live telemetry counters

Follows AGENTS.md V11 & PRODUCTION_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent

for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elevator_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python static / CLI-help verification outside Blender
    bpy = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

if bpy is not None:
    import cinema_camera  # noqa: E402
    import elevator_cab_asset  # noqa: E402
    import labels_3d  # noqa: E402
    import passenger_character  # noqa: E402
    import shaft_env  # noqa: E402
    import spine_skeleton_xray  # noqa: E402
else:
    cinema_camera = None  # type: ignore[assignment]
    elevator_cab_asset = None  # type: ignore[assignment]
    labels_3d = None  # type: ignore[assignment]
    passenger_character = None  # type: ignore[assignment]
    shaft_env = None  # type: ignore[assignment]
    spine_skeleton_xray = None  # type: ignore[assignment]

FPS: int = C.FPS
TOTAL_FRAMES: int = C.TOTAL_FRAMES
FONT_PATH: Path = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"

DEFAULT_PREVIEW_FRAMES: tuple[int, ...] = (
    20, 150, 260,      # Beat 1: hook floor display drop + jump myth crouch
    350, 480,          # Beat 2: cable snap + 50 mph shaft plunge cutaway
    665, 750,          # Beat 3: myth jump (+0.52 m) + ceiling strike
    940, 985, 1100,    # Beat 4: 28 G pit impact + X-ray skeleton shockwave
    1280, 1465,        # Beat 5: safety rail clamps + flat-on-back survival pose
    1650, 1760,        # Beat 6: 12x floor force-distribution grid reveal
)
DEFAULT_BEAT1_DENSE_FRAMES: tuple[int, ...] = tuple(range(1, 313, 12))


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "Blender 'bpy' module is required to build or render the scene; "
            "run via blender --background --python src/build_elevator_scene.py -- ..."
        )


def _make_collection(name: str) -> Any:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def configure_render(samples: int, scale: int, device: str) -> None:
    """Configure Blender Cycles + AgX + GPU OIDN settings for 1080x1920 @ 60 fps."""
    _require_bpy()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = int(scale)
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95

    # AgX Cinema Color Management
    scene.view_settings.view_transform = "AgX"
    for look_name in ("AgX - Punchy", "None"):
        try:
            scene.view_settings.look = look_name
            break
        except Exception:
            continue
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Cycles sampling, OIDN denoising & light bounces
    cycles = scene.cycles
    cycles.samples = int(samples)
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.025
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 3
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 12
    cycles.volume_bounces = 0
    cycles.transparent_max_bounces = 16
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.sample_clamp_direct = 12.0
    cycles.sample_clamp_indirect = 5.0

    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.25

    req_dev = (device or "OPTIX").upper()
    if req_dev == "CPU":
        cycles.device = "CPU"
        if hasattr(cycles, "denoising_use_gpu"):
            try:
                cycles.denoising_use_gpu = False
            except Exception:
                pass
        print("[Render] Using CPU Cycles + OIDN", flush=True)
        return

    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = None
    backends: list[str] = []
    for candidate in ([req_dev, "OPTIX", "CUDA"] if req_dev in {"OPTIX", "CUDA"} else ["OPTIX", "CUDA"]):
        if candidate not in backends:
            backends.append(candidate)

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
        if hasattr(cycles, "denoising_use_gpu"):
            try:
                cycles.denoising_use_gpu = True
            except Exception:
                pass
        print(f"[Render] Using GPU Cycles ({chosen}) + GPU OIDN", flush=True)
    else:
        cycles.device = "CPU"
        if hasattr(cycles, "denoising_use_gpu"):
            try:
                cycles.denoising_use_gpu = False
            except Exception:
                pass
        print("[Render] No GPU found -> CPU Cycles + OIDN", flush=True)


def build_master_scene(
    samples: int = 16,
    scale: int = 100,
    device: str = "OPTIX",
    build_labels_flag: bool = True,
) -> dict[str, Any]:
    """Assemble the complete 1800-frame Episode #16 Falling Elevator Blender scene."""
    _require_bpy()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("Env_Shaft")
    col_cab = _make_collection("Elevator_Cab")
    col_passenger = _make_collection("Passenger_Character")
    col_skeleton = _make_collection("Spine_Skeleton_XRay")
    col_cam = _make_collection("Cinema_Camera")
    col_labels = _make_collection("Labels_3D")

    print("[Scene 1/6] Building High-Rise Elevator Shaft & Pit Environment...", flush=True)
    env = shaft_env.build_environment(col_env, total_frames=TOTAL_FRAMES)

    print("[Scene 2/6] Building Elevator Cab, Cables, Safety Clamps & Force Grid...", flush=True)
    cab = elevator_cab_asset.build_elevator_cab(col_cab, total_frames=TOTAL_FRAMES)

    print("[Scene 3/6] Building 4K MakeHuman MPFB2 Passenger & 6-Beat Acting...", flush=True)
    passenger = passenger_character.build_passenger(col_passenger, total_frames=TOTAL_FRAMES)

    print("[Scene 4/6] Building Full-Body Anatomical X-Ray Skeleton Rig...", flush=True)
    skeleton = spine_skeleton_xray.build_skeleton_xray(
        col_skeleton,
        passenger["carrier"],
        passenger["rig"],
        total_frames=TOTAL_FRAMES,
    )

    print("[Scene 5/6] Building 6-Beat Cinema Camera & DOF Rig...", flush=True)
    cam = cinema_camera.build_cinema_camera(col_cam, total_frames=TOTAL_FRAMES)

    labels: dict[str, Any] = {}
    if build_labels_flag:
        print("[Scene 6/6] Building 3D Dark-Obsidian Plaque Labels & Live Counters...", flush=True)
        labels = labels_3d.build_labels(
            col_labels,
            str(FONT_PATH),
            cab["root"],
            passenger["carrier"],
            total_frames=TOTAL_FRAMES,
        )

    # Guarantee high transparent bounce budget for layered cutaway shaft + cab + X-ray body + skeleton
    scene = bpy.context.scene
    scene.cycles.transparent_max_bounces = max(16, int(scene.cycles.transparent_max_bounces))
    if scene.camera is None and cam is not None:
        scene.camera = cam
    scene.frame_set(1)
    bpy.context.view_layer.update()

    print(f"[Master] Scene built in {time.time() - t0:.1f}s!", flush=True)
    return {
        "env": env,
        "cab": cab,
        "passenger": passenger,
        "skeleton": skeleton,
        "camera": cam,
        "labels": labels,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        elif bpy is None:
            argv = sys.argv[1:]
        else:
            argv = []

    p = argparse.ArgumentParser(
        description="Build & Render Episode #16 Falling Elevator Myth-Buster (60fps, 1800 frames)"
    )
    p.add_argument("--frames-dir", type=str, default="/tmp/fe_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--frames", type=str, default="", help="Comma-separated frame list (e.g. 20,350,750)")
    p.add_argument("--preview", action="store_true", help="Render representative 6-beat QA frames")
    p.add_argument("--beat1-dense", action="store_true", help="Render dense Beat 1 frames")
    p.add_argument("--save-blend", type=str, default="")
    p.add_argument("--skip-labels", action="store_true")
    return p.parse_args(argv)


def resolve_frame_list(args: argparse.Namespace) -> list[int]:
    """Resolve exact frame numbers to render from `--frames`, preset flags, or `--start-frame/--end-frame`."""
    if args.frames:
        frames: list[int] = []
        for token in args.frames.split(","):
            tok = token.strip()
            if not tok:
                continue
            f_num = int(tok)
            if not (1 <= f_num <= TOTAL_FRAMES):
                raise ValueError(f"Frame {f_num} out of range 1..{TOTAL_FRAMES}")
            frames.append(f_num)
        return frames

    preview_frames = tuple(getattr(C, "PREVIEW_FRAMES", DEFAULT_PREVIEW_FRAMES))
    beat1_frames = tuple(getattr(C, "BEAT1_DENSE_FRAMES", DEFAULT_BEAT1_DENSE_FRAMES))

    if args.preview and args.beat1_dense:
        return sorted(set(preview_frames + beat1_frames))
    if args.preview:
        return list(preview_frames)
    if args.beat1_dense:
        return list(beat1_frames)

    start_f = max(1, int(args.start_frame))
    end_f = min(TOTAL_FRAMES, int(args.end_frame))
    if end_f < start_f:
        return []
    return list(range(start_f, end_f + 1))


def main() -> None:
    args = parse_args()
    _require_bpy()

    build_master_scene(
        samples=args.samples,
        scale=args.scale,
        device=args.device,
        build_labels_flag=not args.skip_labels,
    )

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"[Master] Saved .blend -> {args.save_blend}", flush=True)

    frames = resolve_frame_list(args)
    os.makedirs(args.frames_dir, exist_ok=True)
    scene = bpy.context.scene

    for f in frames:
        t_f = time.time()
        scene.frame_set(f)
        out_path = os.path.join(args.frames_dir, f"frame_{f:04d}.jpg")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] frame_{f:04d}.jpg ({time.time() - t_f:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
