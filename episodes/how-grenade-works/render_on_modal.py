"""
1-Command Parallel 10-GPU Cloud 4K Renderer for Modal.com (Warm Studio Image Cache Edition!)
Episode: How a Grenade Actually Works — 4K Mechanical Engineering X-Ray Cutaway
Native 1080x1920 Cycles OPTIX -> True 4K 2160x3840 Lanczos ARRI-Alexa Cinema MP4
"""
from __future__ import annotations

from pathlib import Path
import modal

APP_NAME = "how-grenade-works-4k-render"
REPO_URL = "https://github.com/ssghtc002-sudo/isaac-newton-gravity-prism-3d.git"
ASSET_URL = (
    "https://raw.githubusercontent.com/ssghtc002-sudo/"
    "isaac-newton-gravity-prism-3d/assets-v8/mpfb_newton_v8_4k_assets.zip"
)
EPISODE_SUBDIR = "episodes/how-grenade-works"
TOTAL_FRAMES = 720
FPS = 24

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
    image=studio_image,
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
    samples: int = 16,
    scale: int = 100,
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

    out_dir = f"/tmp/frames_{start_frame}_{end_frame}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "/opt/blender/blender",
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        "src/build_grenade_scene.py",
        "--",
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
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
        raise RuntimeError(f"Blender failed on frames {start_frame}..{end_frame}:\n{tail}")

    results: list[tuple[int, bytes]] = []
    for f in range(start_frame, end_frame + 1):
        fp = Path(out_dir) / f"frame_{f:04d}.jpg"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing rendered frame: {fp}")
        results.append((f, fp.read_bytes()))

    print(f"[Modal GPU Worker] Finished frames {start_frame}..{end_frame} ({len(results)} frames) in {time.time() - t0:.1f}s!")
    return results


@app.function(
    image=studio_image,
    cpu=8.0,
    memory=16384,
    timeout=2400,
)
def build_full_hd_short(
    workers: int = 10,
    samples: int = 16,
    scale: int = 100,
    upscale_4k: bool = True,
) -> tuple[bytes, str]:
    import os
    import shutil
    import subprocess
    import time

    t0 = time.time()
    workers = max(1, min(10, workers))
    chunk_size = (TOTAL_FRAMES + workers - 1) // workers
    chunks = [
        (i * chunk_size + 1, min(TOTAL_FRAMES, (i + 1) * chunk_size), samples, scale)
        for i in range(workers)
        if i * chunk_size + 1 <= TOTAL_FRAMES
    ]

    out_label = "4K (2160x3840)" if upscale_4k else "Full HD (1080x1920)"
    print(f"[Modal Cloud Coordinator] Launching {len(chunks)} parallel GPUs for {TOTAL_FRAMES} frames -> {out_label}...", flush=True)
    frames_dir = Path("/tmp/all_render_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for chunk_out in render_chunk.starmap(chunks):
        for frame_num, jpg_bytes in chunk_out:
            (frames_dir / f"frame_{frame_num:04d}.jpg").write_bytes(jpg_bytes)

    print(f"[Modal Cloud Coordinator] All {TOTAL_FRAMES} frames rendered in {(time.time() - t0) / 60:.2f} min! Encoding {out_label} MP4...", flush=True)

    repo_root = "/opt/newton_repo"
    subprocess.run("git fetch --depth 1 -q origin main && git reset --hard -q origin/main", cwd=repo_root, shell=True, check=True)
    ep_dir = f"{repo_root}/{EPISODE_SUBDIR}"
    os.chdir(ep_dir)

    final_mp4 = Path("/tmp/how_grenade_works_4k.mp4")
    if upscale_4k:
        vf_graph = (
            "[0:v]scale=2160:3840:flags=lanczos,"
            "unsharp=5:5:0.45:5:5:0.0,format=gbrp,split[base][hi];"
            "[hi]eq=brightness=-0.52:contrast=2.8,gblur=sigma=22,format=gbrp[glow];"
            "[base][glow]blend=all_mode=screen:all_opacity=0.20,"
            "colorbalance=rs=0.006:bs=0.014:rh=0.018:bh=-0.010,"
            "vignette=PI/7.0:eval=init,format=yuv420p,noise=alls=2:allf=t,"
            "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
        )
        level_args = ["-level:v", "5.1"]
    else:
        vf_graph = (
            "[0:v]scale=1080:1920:flags=lanczos,"
            "unsharp=3:3:0.35:3:3:0.0,format=gbrp,split[base][hi];"
            "[hi]eq=brightness=-0.52:contrast=2.8,gblur=sigma=12,format=gbrp[glow];"
            "[base][glow]blend=all_mode=screen:all_opacity=0.22,"
            "colorbalance=rs=0.006:bs=0.014:rh=0.018:bh=-0.010,"
            "vignette=PI/7.0:eval=init,format=yuv420p,noise=alls=3:allf=t,"
            "ass=subtitles/subtitles.ass:fontsdir=subtitles/fonts[vout]"
        )
        level_args = []

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%04d.jpg"),
            "-i", "audio/output/master_audio.wav",
            "-filter_complex", vf_graph,
            "-map", "[vout]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-profile:v", "high", *level_args,
            "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
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
    out: str = "how_grenade_works_4k.mp4",
):
    import time

    t0 = time.time()
    mp4_bytes, probe = build_full_hd_short.remote(workers=workers, samples=samples, scale=scale, upscale_4k=upscale_4k)
    out_path = Path(out).resolve()
    out_path.write_bytes(mp4_bytes)
    print(f"\nDONE in {(time.time() - t0) / 60:.2f} min! Saved {len(mp4_bytes) / (1024 * 1024):.1f} MB -> {out_path}")
    print(f"ffprobe: {probe}")
