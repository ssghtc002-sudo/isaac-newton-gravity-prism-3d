"""
Lightweight Dual-T4 GPU Render Worker for Baked `newton_v9_baked.blend`.

Loads the pre-baked 720-frame scene in <0.8s (eliminating duplicate CPU BMesh + MPFB2
scene construction across GPU 0 & GPU 1), activates OPTIX/CUDA raytracing + FAST GPU
OpenImageDenoise (RGB_ALBEDO) on the isolated `CUDA_VISIBLE_DEVICES` GPU, and renders
the requested frame range directly to `render_frames/frame_####.jpg`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Dual-T4 Baked .blend OPTIX Turbo Render Worker")
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--scale", type=int, default=75)
    p.add_argument("--out-dir", type=str, default="render_frames")
    return p.parse_args(argv)


ARGS = parse_args()
OUT_DIR = os.path.abspath(ARGS.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.fps = ARGS.fps
scene.render.resolution_percentage = ARGS.scale
scene.render.use_persistent_data = True
scene.render.use_overwrite = False
scene.render.use_placeholder = False
scene.render.threads_mode = "FIXED"
scene.render.threads = 2
scene.use_nodes = False

scene.frame_start = max(1, ARGS.start_frame)
scene.frame_end = ARGS.end_frame

scene.cycles.samples = ARGS.samples
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.055
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_input_passes = "RGB_ALBEDO"
scene.cycles.denoising_prefilter = "NONE"
try:
    scene.cycles.denoising_use_gpu = True
except Exception:
    pass

scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.quality = 95
scene.render.filepath = os.path.join(OUT_DIR, "frame_")

gpu_enabled = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if gpu_devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                gpu_enabled = True
                print(
                    f"[Cycles Baked Worker] {backend} GPU ({len(gpu_devices)} device(s)) | "
                    f"Denoiser={scene.cycles.denoiser} (GPU RGB_ALBEDO) | "
                    f"Frames {scene.frame_start}..{scene.frame_end} ({ARGS.fps}fps, {ARGS.scale}%->1080x1920 Full HD, {ARGS.samples} spp)"
                )
                break
        except Exception:
            continue
except Exception as e:
    print(f"[Cycles Worker] GPU warning: {e}")

if not gpu_enabled:
    scene.cycles.device = "CPU"
    print("[Cycles Worker] Falling back to CPU")

t0 = time.time()
bpy.ops.render.render(animation=True)
print(
    f"[Cycles Baked Worker] Batch {scene.frame_start}..{scene.frame_end} complete in {(time.time() - t0) / 60:.2f} min!"
)
