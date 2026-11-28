"""
Master Scene Integrator for Episode #2:
"Why Bullets Stop Dead Just 3 Feet Underwater" (4K 60fps, 1800 frames).

Unifies all 5 specialist 3D modules strictly from `bullet_contract.py`:
  1. tank_env.py          — Crystal ballistic tank, dual-mode backdrop, 3 FT laser datum & studio lights
  2. bullet_fx.py         — Muzzle barrel, supersonic hero round, cavitation cone, X-ray & Beat 5 race
  3. swimmer_character.py — 4K MakeHuman MPFB2 wetsuit diver (Beat 1 hook & Beat 6 safe depth)
  4. cinema_camera.py     — 6-beat 9:16 portrait camera & exact DOF focus tracking
  5. labels_3d.py         — 7 AgX-safe dark-obsidian plaque 3D callouts & perpendicular pointer reticles

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

import bullet_contract as C  # noqa: E402

try:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python static / CLI-help verification outside Blender
    bpy = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

if bpy is not None:
    import bullet_fx  # noqa: E402
    import cinema_camera  # noqa: E402
    import labels_3d  # noqa: E402
    import swimmer_character  # noqa: E402
    import tank_env  # noqa: E402
else:
    bullet_fx = None  # type: ignore[assignment]
    cinema_camera = None  # type: ignore[assignment]
    labels_3d = None  # type: ignore[assignment]
    swimmer_character = None  # type: ignore[assignment]
    tank_env = None  # type: ignore[assignment]

FPS: int = C.FPS
TOTAL_FRAMES: int = C.TOTAL_FRAMES
FONT_PATH: Path = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"

DEFAULT_PREVIEW_FRAMES: tuple[int, ...] = (
    20, 135, 190,      # Beat 1: hook incoming bullet, 3 FT stop + label, swimmer shock
    265, 350, 480,     # Beat 2: muzzle fire, water entry + 800x denser, deceleration
    620, 720, 840,     # Beat 3: cavitation vapor bubble cone peak
    960, 1055, 1180,   # Beat 4: X-ray core cutaway, tumble & shatter burst
    1355, 1410, 1525,  # Beat 5: studio blueprint race (3 FT stop vs 6 FT laminar glide)
    1640, 1740,        # Beat 6: safe below 3 ft diver & settled scrap
)
DEFAULT_BEAT1_DENSE_FRAMES: tuple[int, ...] = tuple(
    sorted(set(range(1, C.BEATS["B1_HOOK"][1] + 1, 10)) | {C.EV["b1_bullet_stop"], C.BEATS["B1_HOOK"][1]})
)


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "Blender 'bpy' module is required to build or render the scene; "
            "run via blender --background --python src/build_bullets_scene.py -- ..."
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
    """Assemble the complete 1800-frame Episode #2 Underwater Bullets Blender scene."""
    _require_bpy()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("01_Env_Tank")
    col_fx = _make_collection("02_Bullets_Cavitation_FX")
    col_swimmer = _make_collection("03_Swimmer")
    col_cam = _make_collection("04_Cinema_Camera")
    col_labels = _make_collection("05_3D_Labels")

    print("[Scene 1/5] Building Ballistic Water Tank, Dual-Mode Backdrop & Studio Lights...", flush=True)
    env_res = tank_env.build_tank_env(col_env, TOTAL_FRAMES)

    print("[Scene 2/5] Building Bullets, Cavitation Vapor Cone, X-Ray & Race FX...", flush=True)
    fx_res = bullet_fx.build_bullet_fx(col_fx, TOTAL_FRAMES)

    print("[Scene 3/5] Building 4K MakeHuman MPFB2 Wetsuit Swimmer...", flush=True)
    swimmer_res = swimmer_character.build_swimmer(col_swimmer, TOTAL_FRAMES)

    print("[Scene 4/5] Building 6-Beat Cinema Camera & Exact DOF Rig...", flush=True)
    cam_res = cinema_camera.build_camera(col_cam, TOTAL_FRAMES)

    labels_res: dict[str, Any] = {}
    if build_labels_flag:
        print("[Scene 5/5] Building 3D Dark-Obsidian Plaque Labels & Pointer Reticles...", flush=True)
        labels_res = labels_3d.build_labels(col_labels, font_path=FONT_PATH, total_frames=TOTAL_FRAMES)

    # Enforce high transparent bounce budget even after all sub-builders complete
    scene = bpy.context.scene
    scene.cycles.transparent_max_bounces = max(16, int(scene.cycles.transparent_max_bounces))
    scene.camera = cam_res["camera"]
    scene.frame_set(1)
    bpy.context.view_layer.update()

    print(f"[Master] Scene built in {time.time() - t0:.1f}s!", flush=True)
    return {
        "env": env_res,
        "fx": fx_res,
        "swimmer": swimmer_res,
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
        description="Build & Render Episode #2 Underwater Bullets (60fps, 1800 frames)"
    )
    p.add_argument("--frames-dir", type=str, default="/tmp/ub_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--frames", type=str, default="", help="Comma-separated frame list (e.g. 20,135,350,1055)")
    p.add_argument("--preview", action="store_true", help="Render representative 6-beat QA frames")
    p.add_argument("--dense-beat1", action="store_true", help="Render dense Beat 1 hook frames")
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

    if args.preview and args.dense_beat1:
        return sorted(set(preview_frames + beat1_frames))
    if args.preview:
        return list(preview_frames)
    if args.dense_beat1:
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
