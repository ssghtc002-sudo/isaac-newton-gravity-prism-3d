"""
build_heli_scene.py — Master Scene Integrator (V2 Ultimate) for Episode #5:
"Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (4K 60fps, 1800 frames).

Unifies all 6 specialist 3D modules strictly from `heli_contract.py`:
  1. heli_env.py          — Golden-hour sunset sky, curtain-wall glass skyscrapers, cloud descent parallax, weathered helipad & V12 lights
  2. characters.py        — 2 photoreal 4K MPFB2 Rescue Aviators in dark-navy Nomex suits + enclosed 1:1 cockpit w/ live crashing RPM needles
  3. helicopter_asset.py  — Modern lofted twin-engine H145/Dauphin rescue helicopter, 12-sprag X-ray clutch, coning rotor & 2.1x botanical maple seed
  4. heli_fx.py           — Beat 1A turbine fireball/sparks/smoke, Beat 2 stall halo, Beat 3 6 wide emerald ribbon arrows, flare & rooftop dust wave
  5. cinema_camera.py     — 8-segment 9:16 portrait camera (Beat 1A/1B split + label-free Beat 5A f/3.6 bokeh) & exact DOF rig
  6. labels_3d.py         — 6 AgX-safe dark-obsidian plaque 3D callouts & perpendicular pointer reticles

Supports `--start-frame`, `--end-frame`, `--frame-step` (default 1, enables budget `--draft-preview`),
`--samples`, `--scale`, `--frames-dir`, `--device`, `--frames`, `--preview`, `--dense-beat1`,
`--save-blend`, and `--skip-labels`.

Follows AGENTS.md V12 (Blender 5.3.0 Cycles OPTIX + GPU OIDN + AgX Punchy).
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

import heli_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
except ImportError:
    bpy = None  # type: ignore[assignment]

if bpy is not None:
    import characters  # noqa: E402
    import cinema_camera  # noqa: E402
    import heli_env  # noqa: E402
    import heli_fx  # noqa: E402
    import helicopter_asset  # noqa: E402
    import labels_3d  # noqa: E402
else:
    characters = None  # type: ignore[assignment]
    cinema_camera = None  # type: ignore[assignment]
    heli_env = None  # type: ignore[assignment]
    heli_fx = None  # type: ignore[assignment]
    helicopter_asset = None  # type: ignore[assignment]
    labels_3d = None  # type: ignore[assignment]

FPS: int = C.FPS
TOTAL_FRAMES: int = C.TOTAL_FRAMES
_EP_FONT = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
FONT_PATH: Path = _EP_FONT if _EP_FONT.is_file() else (REPO_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf")

DEFAULT_PREVIEW_FRAMES: tuple[int, ...] = (
    26, 55, 95,         # Beat 1A: Exterior Modern Rescue Chopper — Dorsal Turbine Fire Burst, Sparks & Black Smoke!
    125, 165, 215,      # Beat 1B: Dark Enclosed 1:1 Cockpit — Panic Crew, Crashing ENG 0% & Collective Slam!
    310, 365, 455,      # Beat 2: Local-Space Cowling X-Ray, Seized Crimson Turbine & 12 Golden Cam Sprags
    635, 765, 895,      # Beat 3: 6 Wide Sculpted Emerald 3D Ribbon Arrows & Cloud Descent Parallax
    1020, 1140, 1245,   # Beat 4: Enlarged 2.1x Botanical 3D Maple Seed vs. Modern Rescue Rotor Side-by-Side
    1375, 1495, 1555,   # Beat 5A: Enclosed Cockpit Flare Pull (f/3.6 bokeh) + Beat 5B: 30-Ft Nose-Up Flare (f/8.0)
    1635, 1715,         # Beat 6: Smooth Zero-Engine Flush Touchdown on Weathered Rooftop Helipad + Dust Wave
)
DEFAULT_BEAT1_DENSE_FRAMES: tuple[int, ...] = tuple(
    sorted(
        set(range(1, C.BEATS["B1_HOOK"][1] + 1, 10))
        | {
            C.EV["b1_engine_flameout"],
            34,
            C.B1_EXTERIOR_END,
            C.B1_COCKPIT_START,
            C.EV["b1_caution_alarm"],
            C.EV["b1_collective_slam"],
            175,
            C.BEATS["B1_HOOK"][1],
        }
    )
)


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "Blender 'bpy' module is required to build or render the scene; "
            "run via blender --background --python src/build_heli_scene.py -- ..."
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
    """Assemble the complete 1800-frame Episode #5 Helicopter Autorotation Blender scene."""
    _require_bpy()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("01_Env_GoldenSkyline_Helipad")
    col_chars = _make_collection("02_MPFB2_RescueCrew_Cockpit")
    col_heli = _make_collection("03_RescueHeli_Clutch_MapleSeed")
    col_fx = _make_collection("04_Airflow_Vortex_Flare_FX")
    col_cam = _make_collection("05_Cinema_Camera")
    col_labels = _make_collection("06_3D_Labels")

    print("[Scene 1/6] Building Golden-Hour Skyline, Sunset Sky, Rooftop Helipad (Z=0.90m) & V12 Lighting...", flush=True)
    env_res = heli_env.build_heli_env(col_env, TOTAL_FRAMES)

    print("[Scene 2/6] Building 2 Photoreal 4K MPFB2 Rescue Aviators (Pilot & Co-Pilot) + 1:1 Rescue Cockpit...", flush=True)
    char_res = characters.build_characters(col_chars, TOTAL_FRAMES)

    print("[Scene 3/6] Building White/Red Rescue Helicopter, X-Ray Sprag Clutch, Seized Turbine & 3D Maple Seed...", flush=True)
    heli_res = helicopter_asset.build_helicopter_asset(col_heli, TOTAL_FRAMES)

    print("[Scene 4/6] Building Upward Airflow Streamlines, Maple Vortices, 30-Ft Flare & Touchdown Wash FX...", flush=True)
    fx_res = heli_fx.build_heli_fx(col_fx, heli_res["heli_root"], heli_res["maple"]["root"], TOTAL_FRAMES)

    print("[Scene 5/6] Building 6-Beat Cinema Camera (Incl. Beat 5A Cockpit f/3.6 Bokeh) & Exact DOF Rig...", flush=True)
    cam_res = cinema_camera.build_camera(col_cam, TOTAL_FRAMES)

    labels_res: dict[str, Any] = {}
    if build_labels_flag:
        print("[Scene 6/6] Building 6 AgX-Safe 3D Dark-Obsidian Plaque Labels & Pointer Reticles...", flush=True)
        labels_res = labels_3d.build_labels(col_labels, font_path=FONT_PATH, total_frames=TOTAL_FRAMES)

    scene = bpy.context.scene
    scene.cycles.transparent_max_bounces = max(16, int(scene.cycles.transparent_max_bounces))
    scene.camera = cam_res["camera"]
    scene.frame_set(1)
    bpy.context.view_layer.update()

    print(f"[Master] Scene built in {time.time() - t0:.1f}s!", flush=True)
    return {
        "env": env_res,
        "characters": char_res,
        "helicopter": heli_res,
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
        description="Build & Render Episode #5 Helicopter Autorotation (60fps, 1800 frames)"
    )
    p.add_argument("--frames-dir", type=str, default="/tmp/heli_frames")
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
