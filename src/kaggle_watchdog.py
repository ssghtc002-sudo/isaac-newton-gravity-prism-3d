"""Kaggle-safe Blender launcher: live RAM monitor + self-kill guard.

Runs a Blender command in its own process group, prints process-tree RSS and free RAM
every few seconds (flushed, so the last line survives even if Kaggle drops the session),
and kills Blender itself BEFORE the container would be OOM-killed, so the user gets a
readable log tail instead of "Draft Session off".
"""
import os
import signal
import subprocess
import time

LIMIT_GB = float(os.environ.get("BLENDER_RAM_LIMIT_GB", "22"))
MIN_FREE_GB = float(os.environ.get("BLENDER_MIN_FREE_GB", "3"))


def _tree_rss_gb(root_pid):
    kids = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append(int(p))
        except Exception:
            pass
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        stack.extend(kids.get(pid, []))
        try:
            with open(f"/proc/{pid}/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        total += int(ln.split()[1])
        except Exception:
            pass
    return total / 1048576


def _avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1048576
    return 99.0


def run(cmd, log_path, tag="Blender", interval=5, progress=None):
    """Run shell `cmd` with stdout/stderr -> log_path. Returns exit code."""
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0, peak = time.time(), 0.0
    while p.poll() is None:
        rss, avail = _tree_rss_gb(p.pid), _avail_gb()
        peak = max(peak, rss)
        extra = f" | {progress()}" if progress else ""
        print(f"\r[{tag}] {time.time()-t0:5.0f}s | RAM used {rss:5.2f} GB (peak {peak:5.2f}) | free {avail:5.1f} GB{extra}   ",
              end="", flush=True)
        if rss > LIMIT_GB or avail < MIN_FREE_GB:
            print(f"\n[WATCHDOG] RAM danger (used {rss:.1f} GB, free {avail:.1f} GB) -> stopping {tag} to save the Kaggle session!", flush=True)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            break
        time.sleep(interval)
    print(f"\n[{tag}] exit code {p.returncode} | peak RAM {peak:.2f} GB | {time.time()-t0:.0f}s", flush=True)
    log.close()
    if p.returncode != 0:
        print(f"--- {tag} LOG TAIL ---\n" + "".join(open(log_path, errors="replace").readlines()[-35:]), flush=True)
    return p.returncode
