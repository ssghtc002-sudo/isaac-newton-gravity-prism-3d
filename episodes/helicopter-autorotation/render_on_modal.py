"""
1-Command Parallel 10-GPU Cloud 4K Renderer for Modal.com (Warm Studio Image Cache Edition!)
Episode #5: Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock — 4K 60fps
Native 1080x1920 Cycles OPTIX (Blender 5.3.0) -> True 4K 2160x3840 Lanczos + AAC 320k/48kHz
Also supports budget --draft-preview (724x1286 @ 30fps, 8 spp, 900 frames, ~$0.25-$0.35).
"""
from __future__ import annotations

from pathlib import Path
import modal

APP_NAME = "helicopter-autorotation-4k-render"
REPO_URL = "https://github.com/ssghtc002-sudo/isaac-newton-gravity-prism-3d.git"
ASSET_URL = (
    "https://raw.githubusercontent.com/ssghtc002-sudo/"
    "isaac-newton-gravity-prism-3d/assets-v8/mpfb_newton_v8_4k_assets.zip"
)
EPISODE_SUBDIR = "episodes/helicopter-autorotation"
TOTAL_FRAMES = 1800
FPS = 60

app = modal.App(APP_NAME)

# Exact byte-for-byte match with cached studio image (instant warm boot on Modal!)
studio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ca-certificates",
        "curl",
        "ffmpeg",
        "fontconfig",
        "git",
        "libass9",
        "libegl1",
        "libgl1",
        "libgomp1",
        "libsm6",
        "libxfixes3",
        "libxi6",
        "libxkbcommon0",
        "libxrender1",
        "libxxf86vm1",
        "unzip",
        "xorg",
        "xz-utils",
        "zip",
    )
    .env(
        {
            "HOME": "/root",
            "CYCLES_BACKEND": "CUDA",
            "OIDN_GPU": "0",
            "BLENDER_THREADS": "4",
            "OMP_NUM_THREADS": "4",
        }
    )
    .run_commands(
        "curl -fSL --retry 3 https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz -o /tmp/blender.tar.xz "
        "&& mkdir -p /opt/blender "
        "&& tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 "
        "&& rm -f /tmp/blender.tar.xz",
        "git clone --depth 1 -q https://github.com/makehumancommunity/mpfb2.git /tmp/mpfb2 "
        "&& (cd /tmp/mpfb2/src && zip -qr /tmp/mpfb.zip mpfb) "
        "&& /opt/blender/blender -b --command extension install-file -r user_default -e /tmp/mpfb.zip "
        "&& rm -rf /tmp/mpfb2 /tmp/mpfb.zip",
    )
    .run_commands(
        f"git clone --depth 1 {REPO_URL} /opt/newton_repo "
        f"&& mkdir -p /tmp/mpfb_packs "
        f"&& curl -fSL --retry 3 '{ASSET_URL}' -o /tmp/mpfb_packs/mpfb_newton_v8_4k_assets.zip "
        "&& MPFB_PACKS_DIR=/tmp/mpfb_packs /opt/blender/blender --background --python-exit-code 1 "
        "--python /opt/newton_repo/src/install_mpfb_packs.py "
        "&& rm -rf /tmp/mpfb_packs",
    )
)


@app.function(
    image=studio_image,
    gpu=["L4", "A10G", "T4"],
    cpu=4.0,
    memory=8192,
    max_containers=10,
    timeout=3000,
    retries=1,
)
def render_chunk(
    start_frame: int,
    end_frame: int,
    samples: int = 16,
    scale: int = 100,
    frame_step: int = 1,
) -> list[tuple[int, bytes]]:
    import os
    import shutil
    import subprocess
    import time

    t0 = time.time()
    repo_root = "/opt/newton_repo"
    subprocess.run(
        "git fetch --depth 1 -q origin main && git reset --hard -q origin/main",
        cwd=repo_root,
        shell=True,
        check=True,
    )
    ep_dir = f"{repo_root}/{EPISODE_SUBDIR}"

    out_dir = f"/tmp/frames_{start_frame}_{end_frame}_{frame_step}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "/opt/blender/blender",
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        "src/build_heli_scene.py",
        "--",
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--frame-step",
        str(frame_step),
        "--samples",
        str(samples),
        "--scale",
        str(scale),
        "--frames-dir",
        out_dir,
    ]
    res = subprocess.run(cmd, cwd=ep_dir, capture_output=True, text=True)
    if res.returncode != 0:
        tail = "\n".join(res.stdout.splitlines()[-40:] + res.stderr.splitlines()[-40:])
        raise RuntimeError(f"Blender failed on frames {start_frame}..{end_frame} (step={frame_step}):\n{tail}")

    results: list[tuple[int, bytes]] = []
    for f in range(start_frame, end_frame + 1, frame_step):
        fp = Path(out_dir) / f"frame_{f:04d}.jpg"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing rendered frame: {fp}")
        results.append((f, fp.read_bytes()))

    print(
        f"[Modal GPU Worker] Finished frames {start_frame}..{end_frame} (step={frame_step}, "
        f"{len(results)} frames) in {time.time() - t0:.1f}s!"
    )
    return results


@app.function(
    image=studio_image,
    cpu=8.0,
    memory=16384,
    timeout=5400,
)
def build_full_hd_short(
    workers: int = 10,
    samples: int = 16,
    scale: int = 100,
    upscale_4k: bool = True,
    draft_preview: bool = False,
) -> tuple[bytes, str]:
    import os
    import shutil
    import subprocess
    import sys
    import time

    t0 = time.time()
    workers = max(1, min(10, workers))

    if draft_preview:
        scale = 67
        samples = 8
        frame_step = 2
        out_fps = FPS // frame_step  # 30 fps (900 frames)
    else:
        frame_step = 1
        out_fps = FPS  # 60 fps (1800 frames)

    # Build step-aligned frame list so every chunk renders exact non-overlapping frames
    all_target_frames = list(range(1, TOTAL_FRAMES + 1, frame_step))
    n_target = len(all_target_frames)
    per_worker = (n_target + workers - 1) // workers
    chunks: list[tuple[int, int, int, int, int]] = []
    for i in range(workers):
        sub = all_target_frames[i * per_worker : (i + 1) * per_worker]
        if sub:
            chunks.append((sub[0], sub[-1], samples, scale, frame_step))

    if draft_preview:
        out_label = f"Draft 720p ({scale}%, {out_fps}fps, {n_target} frames, {samples} spp)"
    else:
        out_label = "4K (2160x3840 @ 60fps)" if upscale_4k else "Full HD (1080x1920 @ 60fps)"

    print(
        f"[Modal Cloud Coordinator] Launching {len(chunks)} parallel GPUs for {n_target} frames -> {out_label}...",
        flush=True,
    )
    frames_dir = Path("/tmp/all_render_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    rendered_map: dict[int, bytes] = {}
    for chunk_out in render_chunk.starmap(chunks):
        for frame_num, jpg_bytes in chunk_out:
            rendered_map[frame_num] = jpg_bytes

    if len(rendered_map) != n_target:
        raise RuntimeError(f"Expected {n_target} frames, received {len(rendered_map)}")

    for seq_idx, frame_num in enumerate(sorted(rendered_map.keys()), start=1):
        (frames_dir / f"seq_{seq_idx:04d}.jpg").write_bytes(rendered_map[frame_num])

    print(
        f"[Modal Cloud Coordinator] All {n_target} frames rendered in {(time.time() - t0) / 60:.2f} min! "
        f"Encoding {out_label} MP4...",
        flush=True,
    )

    repo_root = "/opt/newton_repo"
    subprocess.run(
        "git fetch --depth 1 -q origin main && git reset --hard -q origin/main",
        cwd=repo_root,
        shell=True,
        check=True,
    )
    ep_dir = f"{repo_root}/{EPISODE_SUBDIR}"
    os.chdir(ep_dir)

    # Safety net: if master_audio.wav/subtitles.ass is missing OR timeline.json narration changed,
    # synthesize fresh Edge-TTS + SFX + BGM master & synced ASS karaoke automatically!
    need_audio_rebuild = (
        not Path("audio/output/master_audio.wav").is_file()
        or not Path("subtitles/subtitles.ass").is_file()
        or not Path("audio/output/voice_manifest.json").is_file()
    )
    if not need_audio_rebuild:
        import json as _json
        try:
            _tl = _json.loads(Path("config/timeline.json").read_text(encoding="utf-8"))
            _vm = _json.loads(Path("audio/output/voice_manifest.json").read_text(encoding="utf-8"))
            _tl_texts = [ln["text"] for ln in _tl.get("narration_lines", [])]
            _vm_texts = [ln["text"] for ln in _vm.get("lines", [])]
            if _tl_texts != _vm_texts:
                need_audio_rebuild = True
                print("[Modal Cloud Coordinator] Detected updated storyline in timeline.json -> rebuilding audio & subtitles...", flush=True)
        except Exception:
            need_audio_rebuild = True
    if need_audio_rebuild:
        print("[Modal Cloud Coordinator] Generating master_audio.wav & synced subtitles.ass on the fly...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy", "soundfile", "edge-tts"],
            check=True,
        )
        subprocess.run([sys.executable, "audio/generate_audio.py", "--rebuild-all"], check=True)

    if draft_preview:
        final_mp4 = Path("/tmp/helicopter_autorotation_draft_720p.mp4")
        vf_graph = (
            "[0:v]scale=724:1286:flags=lanczos,format=yuv420p,"
            "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
        )
        crf_val = "19"
        level_args: list[str] = []
    elif upscale_4k:
        final_mp4 = Path("/tmp/helicopter_autorotation_4k.mp4")
        vf_graph = (
            "[0:v]scale=2160:3840:flags=lanczos,"
            "unsharp=5:5:0.45:5:5:0.0,format=gbrp,split[base][hi];"
            "[hi]eq=brightness=-0.52:contrast=2.8,gblur=sigma=22,format=gbrp[glow];"
            "[base][glow]blend=all_mode=screen:all_opacity=0.20,"
            "colorbalance=rs=0.008:bs=0.006:rh=0.022:bh=-0.012,"
            "vignette=PI/7.0:eval=init,format=yuv420p,noise=alls=2:allf=t,"
            "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
        )
        crf_val = "15"
        level_args = ["-level:v", "5.2"]
    else:
        final_mp4 = Path("/tmp/helicopter_autorotation_1080p.mp4")
        vf_graph = (
            "[0:v]scale=1080:1920:flags=lanczos,"
            "unsharp=3:3:0.35:3:3:0.0,format=gbrp,split[base][hi];"
            "[hi]eq=brightness=-0.52:contrast=2.8,gblur=sigma=12,format=gbrp[glow];"
            "[base][glow]blend=all_mode=screen:all_opacity=0.22,"
            "colorbalance=rs=0.008:bs=0.006:rh=0.022:bh=-0.012,"
            "vignette=PI/7.0:eval=init,format=yuv420p,noise=alls=3:allf=t,"
            "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
        )
        crf_val = "15"
        level_args = []

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(out_fps),
            "-i", str(frames_dir / "seq_%04d.jpg"),
            "-i", "audio/output/master_audio.wav",
            "-filter_complex", vf_graph,
            "-map", "[vout]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-profile:v", "high", *level_args,
            "-crf", crf_val, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            str(final_mp4), "-loglevel", "warning",
        ],
        check=True,
    )

    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "compact", str(final_mp4),
        ]
    ).decode().strip()

    return final_mp4.read_bytes(), probe


@app.local_entrypoint()
def main(
    workers: int = 10,
    samples: int = 16,
    scale: int = 100,
    upscale_4k: bool = True,
    draft_preview: bool = False,
    out: str = "",
):
    import time

    t0 = time.time()
    mp4_bytes, probe = build_full_hd_short.remote(
        workers=workers,
        samples=samples,
        scale=scale,
        upscale_4k=upscale_4k,
        draft_preview=draft_preview,
    )
    if not out:
        if draft_preview:
            out = "helicopter_autorotation_draft_720p.mp4"
        elif upscale_4k:
            out = "helicopter_autorotation_4k.mp4"
        else:
            out = "helicopter_autorotation_1080p.mp4"
    out_path = Path(out).resolve()
    out_path.write_bytes(mp4_bytes)
    print(f"\nDONE in {(time.time() - t0) / 60:.2f} min! Saved {len(mp4_bytes) / (1024 * 1024):.1f} MB -> {out_path}")
    print(f"ffprobe: {probe}")
