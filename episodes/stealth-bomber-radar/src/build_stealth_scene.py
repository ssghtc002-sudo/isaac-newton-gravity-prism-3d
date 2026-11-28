"""
Master Scene Integrator for Episode #4:
"How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar" (4K 60fps, 1800 frames).

Unifies all 5 specialist 3D modules strictly from `stealth_contract.py`:
  1. radar_env.py            — Night airspace world, parabolic radar station, Beat 5 CAD blueprint & lights
  2. stealth_aircraft_asset.py — 172-ft B-2 flying wing, local-space X-ray S-ducts, spinning turbines & Conv Jet
  3. radar_fx.py             — Open-ended additive radar cones, deflected rays, RAM honeycomb, 3D bumblebee & scope
  4. cinema_camera.py        — 6-beat 9:16 portrait camera & exact DOF focus tracking
  5. labels_3d.py            — 7 AgX-safe dark-obsidian plaque 3D callouts & perpendicular pointer reticles

Supports `--start-frame`, `--end-frame`, `--frame-step` (default 1, enables budget `--draft-preview`),
`--samples`, `--scale`, `--frames-dir`, `--device`, `--frames`, `--preview`, `--dense-beat1`,
`--save-blend`, and `--skip-labels`.

Follows AGENTS.md V12 (Blender 5.3.0 + @zackdfilms blueprint).
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

import stealth_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
except ImportError:
    bpy = None  # type: ignore[assignment]

if bpy is not None:
    import cinema_camera  # noqa: E402
    import labels_3d  # noqa: E402
    import radar_env  # noqa: E402
    import radar_fx  # noqa: E402
    import stealth_aircraft_asset  # noqa: E402
else:
    cinema_camera = None  # type: ignore[assignment]
    labels_3d = None  # type: ignore[assignment]
    radar_env = None  # type: ignore[assignment]
    radar_fx = None  # type: ignore[assignment]
    stealth_aircraft_asset = None  # type: ignore[assignment]

FPS: int = C.FPS
TOTAL_FRAMES: int = C.TOTAL_FRAMES
_EP_FONT = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
FONT_PATH: Path = _EP_FONT if _EP_FONT.is_file() else (REPO_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf")

DEFAULT_PREVIEW_FRAMES: tuple[int, ...] = (
    24, 90, 165,        # Beat 1: radar ping, wing deflection, 172 FT WING callout
    280, 340, 445,      # Beat 2: conventional jet crimson echo vs stealth sawtooth deflection
    575, 700, 825,      # Beat 3: local-space X-ray sweep, serpentine S-ducts & trapped RF pulse
    1025, 1110, 1235,   # Beat 4: hexagonal RAM heat absorption + 0.001 m^2 3D bumblebee reveal
    1375, 1465, 1525,   # Beat 5: CAD Blueprint 90-deg mirror echo vs 33-deg tilted stealth facet
    1625, 1735,         # Beat 6: moonlight banking pass + solitary bumblebee speck on radar scope
)
DEFAULT_BEAT1_DENSE_FRAMES: tuple[int, ...] = tuple(
    sorted(
        set(range(1, C.BEATS["B1_HOOK"][1] + 1, 10))
        | {C.EV["b1_radar_ping"], C.EV["b1_wing_deflect"], C.EV["b1_blank_scope"], C.BEATS["B1_HOOK"][1]}
    )
)


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "Blender 'bpy' module is required to build or render the scene; "
            "run via blender --background --python src/build_stealth_scene.py -- ..."
        )


def _make_collection(name: str) -> Any:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def configure_render(samples: int, scale: int, device: str) -> None:
    """Configure Blender 5.3.0 Cycles + AgX Punchy + GPU OIDN for 1080x1920 @ 60 fps."""
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

    scene.view_settings.view_transform = "AgX"
    for look_name in ("AgX - Punchy", "None"):
        try:
            scene.view_settings.look = look_name
            break
        except Exception:
            continue
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

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
    """Assemble the complete 1800-frame Episode #4 Stealth Bomber Radar Blender scene."""
    _require_bpy()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("01_Env_RadarAirspace")
    col_aircraft = _make_collection("02_Stealth_Aircraft")
    col_fx = _make_collection("03_Radar_RAM_Bumblebee_FX")
    col_cam = _make_collection("04_Cinema_Camera")
    col_labels = _make_collection("05_3D_Labels")

    print("[Scene 1/5] Building Daylight Sky, Enemy Radar Command Room, Radar Tower & Lights...", flush=True)
    env_res = radar_env.build_radar_env(col_env, TOTAL_FRAMES)

    print("[Scene 2/5] Building Sculpted 172-ft B-2 Stealth Bomber, S-Ducts, Titanium Turbines & Lofted Conv Jet...", flush=True)
    air_res = stealth_aircraft_asset.build_stealth_aircraft(col_aircraft, TOTAL_FRAMES)

    print("[Scene 3/5] Building Sleek Radar Waves, Command Console Scopes, RAM Honeycomb, Bumblebee & Beat 5 Arrows...", flush=True)
    fx_res = radar_fx.build_radar_fx(col_fx, air_res["bomber_root"], TOTAL_FRAMES)

    print("[Scene 4/5] Building 6-Beat Cinema Camera & Exact DOF Rig...", flush=True)
    cam_res = cinema_camera.build_camera(col_cam, TOTAL_FRAMES)

    labels_res: dict[str, Any] = {}
    if build_labels_flag:
        print("[Scene 5/5] Building 7 AgX-Safe 3D Dark-Obsidian Plaque Labels & Pointer Reticles...", flush=True)
        labels_res = labels_3d.build_labels(col_labels, font_path=FONT_PATH, total_frames=TOTAL_FRAMES)

    scene = bpy.context.scene
    scene.cycles.transparent_max_bounces = max(16, int(scene.cycles.transparent_max_bounces))
    scene.camera = cam_res["camera"]
    scene.frame_set(1)
    bpy.context.view_layer.update()

    print(f"[Master] Scene built in {time.time() - t0:.1f}s!", flush=True)
    return {
        "env": env_res,
        "aircraft": air_res,
        "fx": fx_res,
        "camera": cam_res,
        "labels": labels_res,
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
        description="Build & Render Episode #4 Stealth Bomber Radar (60fps, 1800 frames)"
    )
    p.add_argument("--frames-dir", type=str, default="/tmp/stealth_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--frame-step", type=int, default=1, help="Frame stride (2 for 30fps --draft-preview)")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--frames", type=str, default="", help="Comma-separated frame list")
    p.add_argument("--preview", action="store_true", help="Render representative 6-beat QA frames")
    p.add_argument("--dense-beat1", action="store_true", help="Render dense Beat 1 hook frames")
    p.add_argument("--save-blend", type=str, default="")
    p.add_argument("--skip-labels", action="store_true")
    return p.parse_args(argv)


def resolve_frame_list(args: argparse.Namespace) -> list[int]:
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

    if args.preview and args.dense_beat1:
        return sorted(set(DEFAULT_PREVIEW_FRAMES + DEFAULT_BEAT1_DENSE_FRAMES))
    if args.preview:
        return list(DEFAULT_PREVIEW_FRAMES)
    if args.dense_beat1:
        return list(DEFAULT_BEAT1_DENSE_FRAMES)

    start_f = max(1, int(args.start_frame))
    end_f = min(TOTAL_FRAMES, int(args.end_frame))
    step_f = max(1, int(args.frame_step))
    if end_f < start_f:
        return []
    return list(range(start_f, end_f + 1, step_f))


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
