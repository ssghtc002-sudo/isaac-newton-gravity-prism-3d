"""
1-Command Parallel Cloud GPU Renderer for Modal.com ($30 Free Credit Edition!)
Scientist #9: Isaac Newton — V9 Hollywood Mini-Film Polish (Full HD 1080x1920, 24fps, 720 frames)

Usage (from Windows PowerShell / CMD, Mac/Linux Terminal, or Google Colab CPU):
  1. pip install modal
  2. python -m modal setup
  3. python -m modal run render_on_modal.py

Spins up 10 parallel NVIDIA L4 GPUs on Modal (~90s render time, ~$0.20-$0.25 cost out of $30 credit!),
encodes the True Full HD (1080x1920) ARRI-Alexa MP4 in Modal's 10 Gbps cloud, and saves
`isaac_newton_gravity_prism_short.mp4` directly to your current folder!
"""
from __future__ import annotations

from pathlib import Path
import modal

APP_NAME = "isaac-newton-v9-fullhd-render"
REPO_URL = "https://github.com/ssghtc002-sudo/isaac-newton-gravity-prism-3d.git"
ASSET_URL = (
    "https://raw.githubusercontent.com/ssghtc002-sudo/"
    "isaac-newton-gravity-prism-3d/assets-v8/mpfb_newton_v8_4k_assets.zip"
)
TOTAL_FRAMES = 720
FPS = 24

app = modal.App(APP_NAME)

# Cached cloud container image: Blender 4.2.3 LTS + MPFB2 + Authentic 17th-Century V8 4K Cast + FFmpeg
newton_image = (
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
        # 1. Install Blender 4.2.3 LTS
        "curl -fSL --retry 3 https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz -o /tmp/blender.tar.xz "
        "&& mkdir -p /opt/blender "
        "&& tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 "
        "&& rm -f /tmp/blender.tar.xz",
        # 2. Install MakeHuman MPFB2 extension into Blender 4.2 user_default
        "git clone --depth 1 -q https://github.com/makehumancommunity/mpfb2.git /tmp/mpfb2 "
        "&& (cd /tmp/mpfb2/src && zip -qr /tmp/mpfb.zip mpfb) "
        "&& /opt/blender/blender -b --command extension install-file -r user_default -e /tmp/mpfb.zip "
        "&& rm -rf /tmp/mpfb2 /tmp/mpfb.zip",
    )
    .run_commands(
        # 3. Clone Newton V9 repository & install 4K 17th-century Newton + Halley MakeHuman assets
        f"git clone --depth 1 {REPO_URL} /opt/newton_repo "
        f"&& mkdir -p /tmp/mpfb_packs "
        f"&& curl -fSL --retry 3 '{ASSET_URL}' -o /tmp/mpfb_packs/mpfb_newton_v8_4k_assets.zip "
        "&& MPFB_PACKS_DIR=/tmp/mpfb_packs /opt/blender/blender --background --python-exit-code 1 "
        "--python /opt/newton_repo/src/install_mpfb_packs.py "
        "&& rm -rf /tmp/mpfb_packs",
    )
)


@app.function(
    image=newton_image,
    gpu=["L4", "A10G", "T4"],
    cpu=4.0,
    memory=8192,
    max_containers=10,
    timeout=1800,
    retries=1,
)
def render_chunk(
    start_frame: int,
    end_frame: int,
    samples: int = 12,
    scale: int = 75,
) -> list[tuple[int, bytes]]:
    """Renders an assigned [start_frame..end_frame] slice on a dedicated Modal GPU."""
    import os
    import shutil
    import subprocess
    import time

    t0 = time.time()
    gpu_name = "GPU"
    try:
        gpu_name = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            )
            .splitlines()[0]
            .strip()
        )
    except Exception:
        pass
    print(f"[Modal Worker] Rendering frames {start_frame}..{end_frame} on {gpu_name} ...", flush=True)

    repo_dir = "/opt/newton_repo"
    subprocess.run(
        "git fetch --depth 1 -q origin main && git reset --hard -q origin/main",
        cwd=repo_dir,
        shell=True,
        check=True,
    )

    out_dir = f"/tmp/frames_{start_frame}_{end_frame}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "/opt/blender/blender",
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        "src/build_newton_scene.py",
        "--",
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--fps",
        str(FPS),
        "--samples",
        str(samples),
        "--scale",
        str(scale),
        "--out-dir",
        out_dir,
    ]
    res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if res.returncode != 0:
        tail = "\n".join(res.stdout.splitlines()[-40:] + res.stderr.splitlines()[-40:])
        raise RuntimeError(f"Blender failed on frames {start_frame}..{end_frame}:\n{tail}")

    results: list[tuple[int, bytes]] = []
    for f in range(start_frame, end_frame + 1):
        fp = Path(out_dir) / f"frame_{f:04d}.jpg"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing rendered frame: {fp}")
        results.append((f, fp.read_bytes()))

    print(
        f"[Modal GPU Worker] Finished frames {start_frame}..{end_frame} "
        f"({len(results)} frames) in {time.time() - t0:.1f}s!"
    )
    return results


@app.function(
    image=newton_image,
    cpu=8.0,
    memory=8192,
    timeout=2400,
)
def build_full_hd_short(
    workers: int = 10,
    samples: int = 12,
    scale: int = 75,
) -> tuple[bytes, str]:
    """Coordinates parallel GPU chunks inside Modal's cloud & encodes Full HD 1080x1920 MP4."""
    import os
    import shutil
    import subprocess
    import time

    t0 = time.time()
    workers = max(1, min(10, workers))
    chunk_size = (TOTAL_FRAMES + workers - 1) // workers
    chunks = []
    for i in range(workers):
        s = i * chunk_size + 1
        e = min(TOTAL_FRAMES, (i + 1) * chunk_size)
        if s <= e:
            chunks.append((s, e, samples, scale))

    print(f"[Modal Cloud Coordinator] Launching {len(chunks)} parallel GPU containers for {TOTAL_FRAMES} frames...")
    frames_dir = Path("/tmp/all_render_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for chunk_out in render_chunk.starmap(chunks):
        for frame_num, jpg_bytes in chunk_out:
            (frames_dir / f"frame_{frame_num:04d}.jpg").write_bytes(jpg_bytes)

    rendered_count = len(list(frames_dir.glob("frame_*.jpg")))
    if rendered_count != TOTAL_FRAMES:
        raise RuntimeError(f"Expected {TOTAL_FRAMES} frames, got {rendered_count}")

    print(
        f"[Modal Cloud Coordinator] All {rendered_count}/{TOTAL_FRAMES} frames rendered in "
        f"{(time.time() - t0) / 60:.2f} min! Running Full HD 1080x1920 ARRI-Alexa FFmpeg encode..."
    )

    repo_dir = "/opt/newton_repo"
    subprocess.run(
        "git fetch --depth 1 -q origin main && git reset --hard -q origin/main",
        cwd=repo_dir,
        shell=True,
        check=True,
    )
    os.chdir(repo_dir)

    final_mp4 = Path("/tmp/isaac_newton_gravity_prism_short.mp4")
    vf_graph = (
        "[0:v]scale=1080:1920:flags=lanczos,"
        "unsharp=3:3:0.35:3:3:0.0,format=gbrp,split[base][hi];"
        "[hi]eq=brightness=-0.42:contrast=2.4,gblur=sigma=16,format=gbrp[glow];"
        "[base][glow]blend=all_mode=screen:all_opacity=0.28,"
        "colorbalance=rs=-0.012:bs=0.020:rh=0.022:bh=-0.016,"
        "vignette=PI/6.5:eval=init,format=yuv420p,noise=alls=3:allf=t,"
        "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%04d.jpg"),
            "-i",
            "audio/output/master_audio.wav",
            "-filter_complex",
            vf_graph,
            "-map",
            "[vout]",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-profile:v",
            "high",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final_mp4),
            "-loglevel",
            "warning",
        ],
        check=True,
    )

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "compact",
            str(final_mp4),
        ]
    ).decode().strip()

    return final_mp4.read_bytes(), probe


@app.local_entrypoint()
def main(
    workers: int = 10,
    samples: int = 12,
    scale: int = 75,
    out: str = "isaac_newton_gravity_prism_short.mp4",
):
    import time

    t0 = time.time()
    print(
        f"Starting Modal Parallel Cloud Render ({workers} GPUs, {TOTAL_FRAMES} frames -> Full HD 1080x1920)..."
    )
    mp4_bytes, probe = build_full_hd_short.remote(workers=workers, samples=samples, scale=scale)
    out_path = Path(out).resolve()
    out_path.write_bytes(mp4_bytes)
    print("\n=== MODAL FULL HD (1080x1920) RENDER COMPLETE! ===")
    print(probe)
    print(
        f"Saved MP4 -> {out_path} ({len(mp4_bytes) / 1e6:.2f} MB) in {(time.time() - t0) / 60:.2f} min!"
    )
