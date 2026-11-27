"""
labels_3d.py — 3D Engineering Typography, HUD Counters & Pointer Reticles for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames).

Owned by the 3D Typography & Labels Artist.
All timings, coordinates, trajectories, and label strings are imported from
`ejection_contract.py`; all camera & NDC projections come from `cinema_camera.py`.

Standards (AGENTS.md V11 Section 2):
- Unique `Label_*` object prefix linked into `col`.
- Loaded font (`DejaVuSans-Bold.ttf`) verified against every required glyph (including `−` U+2212).
- 2-Zone brushed-face + dark obsidian extrusion shader via Object-Space normalized `|N.z| > 0.85`
  with AgX-safe emission (`1.65..1.95 <= 2.1`).
- Shallow bevel/extrusion + dark obsidian keyline + chamfered dark obsidian glass backing plaque
  with glowing cyan/gold/amber/crimson accent bars + dark-rimmed pointer leader lines & reticles.
- Deterministic multi-GPU live counters (`clock_mdc`, `g_load`, `spine_gauge`, `clock_drogue`)
  using pre-built discrete text curve variants switched via `CONSTANT` `hide_viewport`/`hide_render`
  keyframes (ZERO `frame_change_pre` handlers).

Public API:
    build_labels(
        col: bpy.types.Collection,
        font_path: str | Path,
        seat_root: bpy.types.Object | None,
        pilot_carrier: bpy.types.Object | None,
        jet_root: bpy.types.Object | None,
        total_frames: int = 1800,
    ) -> dict[str, Any]

Returned dict schema (keys -> real Blender objects / collections / metadata):
    - "root":      bpy.types.Object (`Label_Root` Empty)
    - "rigs":      dict[str, bpy.types.Object] (`Label_Rig_<key>` per C.LABELS entry)
    - "plaques":   dict[str, bpy.types.Object] (`Label_Plaque_<key>` meshes)
    - "texts":     dict[str, list[bpy.types.Object]] (all `Label_Text_<key>*` font objects)
    - "pointers":  dict[str, list[bpy.types.Object]] (`Label_Pointer*_<key>` meshes)
    - "windows":   dict[str, tuple[int, int]] (clamped visible frame ranges `(w0, w1)`)
    - "materials": dict[str, bpy.types.Material] (`Label_Mat_*` materials)
    - "all_objs":  list[bpy.types.Object] (every `Label_*` object linked into `col`)
"""
from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_SRC_DIR = EPISODE_DIR.parent.parent / "src"
for _p in (str(REPO_SRC_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402
from cinema_camera import (  # noqa: E402
    SENSOR_H_EFF_MM,
    SENSOR_W_EFF_MM,
    camera_state_at,
    project_ndc,
)

try:
    import bmesh  # type: ignore[import-not-found]
    import bpy  # type: ignore[import-not-found]
    from mathutils import Euler, Matrix, Quaternion, Vector  # type: ignore[import-not-found]
except ImportError:  # Pure-Python static verification mode outside Blender
    bmesh = None  # type: ignore[assignment]
    bpy = None  # type: ignore[assignment]
    Euler = None  # type: ignore[assignment]
    Matrix = None  # type: ignore[assignment]
    Quaternion = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

PREFIX = "Label_"
DEFAULT_FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
ASPECT_H_OVER_W = 16.0 / 9.0

# ---------------------------------------------------------------------------
# 1. CONTRACT LABEL LAYOUT & STYLE SPECIFICATIONS (NDC-normalized half_w=1.0)
# ---------------------------------------------------------------------------
_LABEL_SPECS: dict[str, dict[str, Any]] = {
    "handle": {
        "char_h_ndc": 0.034,
        "theme": "gold",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (-0.50, -0.35),   # bottom-left corner pointing to handle
        "variant_mode": "static",
    },
    "mdc": {
        "char_h_ndc": 0.029,
        "theme": "cyan",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (-0.32, -0.50),   # bottom-left edge pointing to canopy cord
        "variant_mode": "static",
    },
    "clock_mdc": {
        "char_h_ndc": 0.044,
        "theme": "amber_gold",
        "is_hud": True,
        "has_pointer": False,
        "attach_norm": (0.0, 0.0),
        "variant_mode": "clock_mdc",
    },
    "garters": {
        "char_h_ndc": 0.034,
        "theme": "gold",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (0.50, 0.05),     # right edge pointing to ankle/garter strap
        "variant_mode": "static",
    },
    "catapult": {
        "char_h_ndc": 0.033,
        "theme": "amber",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (-0.50, 0.20),    # left-top edge pointing to catapult breech
        "variant_mode": "static",
    },
    "g_load": {
        "char_h_ndc": 0.046,
        "theme": "crimson",
        "is_hud": True,
        "has_pointer": False,
        "attach_norm": (0.0, 0.0),
        "variant_mode": "g_load",
    },
    "spine_gauge": {
        "char_h_ndc": 0.036,
        "theme": "cyan",
        "is_hud": True,
        "has_pointer": True,
        "attach_norm": (-0.50, -0.25),   # bottom-left edge pointing down-left to spine
        "variant_mode": "spine_gauge",
    },
    "rocket": {
        "char_h_ndc": 0.031,
        "theme": "gold",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (0.50, 0.0),      # right edge pointing right to left-side rocket nozzle
        "variant_mode": "static",
    },
    "drogue": {
        "char_h_ndc": 0.034,
        "theme": "cyan",
        "is_hud": False,
        "has_pointer": True,
        "attach_norm": (-0.50, 0.0),     # left edge pointing to drogue canopy
        "variant_mode": "static",
    },
    "clock_drogue": {
        "char_h_ndc": 0.044,
        "theme": "amber_gold",
        "is_hud": True,
        "has_pointer": False,
        "attach_norm": (0.0, 0.0),
        "variant_mode": "clock_drogue",
    },
}


# ---------------------------------------------------------------------------
# 2. PURE-PYTHON FONT VERIFICATION, TRAJECTORIES & NDC GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def _parse_ttf_codepoints(font_path: Path) -> set[int]:
    """Read supported Unicode codepoints from a TTF/OTF SFNT `cmap` table."""
    data = font_path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"Font file truncated: {font_path}")
    num_tables = struct.unpack(">H", data[4:6])[0]
    cmap_off = None
    for i in range(num_tables):
        off = 12 + i * 16
        if data[off : off + 4] == b"cmap":
            cmap_off = struct.unpack(">I", data[off + 8 : off + 12])[0]
            break
    if cmap_off is None:
        raise ValueError(f"No `cmap` table found in font: {font_path}")

    nsub = struct.unpack(">H", data[cmap_off + 2 : cmap_off + 4])[0]
    codepoints: set[int] = set()
    for i in range(nsub):
        rec = cmap_off + 4 + i * 8
        _, _, sub_off = struct.unpack(">HHI", data[rec : rec + 8])
        st = cmap_off + sub_off
        fmt = struct.unpack(">H", data[st : st + 2])[0]
        if fmt == 4:
            seg_count = struct.unpack(">H", data[st + 6 : st + 8])[0] // 2
            ends = struct.unpack(f">{seg_count}H", data[st + 14 : st + 14 + 2 * seg_count])
            starts = struct.unpack(
                f">{seg_count}H",
                data[st + 16 + 2 * seg_count : st + 16 + 4 * seg_count],
            )
            for s, e in zip(starts, ends):
                if s != 0xFFFF:
                    codepoints.update(range(s, e + 1))
        elif fmt == 12:
            ngroups = struct.unpack(">I", data[st + 12 : st + 16])[0]
            for g in range(ngroups):
                s, e, _ = struct.unpack(">III", data[st + 16 + g * 12 : st + 28 + g * 12])
                codepoints.update(range(s, e + 1))
    return codepoints


def verify_font_glyphs(font_path: str | Path, texts: list[str] | None = None) -> set[int]:
    """Verify `font_path` exists and supports every character across all 10 labels & readouts."""
    resolved = Path(font_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required typography font missing: {resolved}")
    codepoints = _parse_ttf_codepoints(resolved)
    if texts is None:
        texts = [t for _, t, _ in C.LABELS] + ["0123456789.:−+-sGcm "]
    missing = sorted({ch for s in texts for ch in s if ch not in ("\n", "\r") and ord(ch) not in codepoints})
    if missing:
        raise ValueError(f"Font {resolved.name} missing required glyphs: {missing}")
    return codepoints


def _seat_local_to_world_pure(f: float, local_pt: C.Vec3) -> C.Vec3:
    """Transform a seat-local point into exact world space via `ejection_contract`."""
    s = C.seat_world_location(f)
    rx, ry, rz = C.seat_world_rotation_euler(f)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x, y, z = local_pt
    # Euler XYZ intrinsic/extrinsic order matching mathutils.Euler((rx, ry, rz), 'XYZ')
    y1, z1 = y * cx - z * sx, y * sx + z * cx
    x2, z2 = x * cy + z1 * sy, -x * sy + z1 * cy
    x3, y3 = x2 * cz - y1 * sz, x2 * sz + y1 * cz
    return (s[0] + x3, s[1] + y3, s[2] + z2)


def _target_world_pos(key: str, f: float) -> C.Vec3 | None:
    """Exact world-space 3D component anchor point for each pointer label."""
    if key == "handle":
        return _seat_local_to_world_pure(f, C.add3(C.handle_local(f), (0.0, 0.035, 0.0)))
    if key == "mdc":
        if f <= 300.0:
            local_pt = C.CANOPY["apex"]
        else:
            u = min(1.0, (f - 300.0) / 140.0)
            local_pt = C.lerp3(C.CANOPY["apex"], (0.0, -0.45, 1.62), u)
        return C.add3(C.jet_offset(f), local_pt)
    if key == "garters":
        return _seat_local_to_world_pure(
            f, C.lerp3(C.SEAT["garter_anchor_R"], C.ankle_local("R", f), 0.55)
        )
    if key == "catapult":
        barrel_pt = C.add3(C.RAIL_BASE, C.scale3(C.RAIL_AXIS, 0.22))
        return C.add3(C.jet_offset(f), barrel_pt)
    if key == "spine_gauge":
        return _seat_local_to_world_pure(f, (0.0, -0.085, 0.44))
    if key == "rocket":
        return _seat_local_to_world_pure(f, C.SEAT["rocket_nozzles"][0])
    if key == "drogue":
        return C.drogue_world_location(f)
    return None


def _plaque_ndc_center(key: str, f: float, st: dict[str, Any]) -> tuple[float, float]:
    """Deterministic collision-free 9:16 NDC center for each label across its active window."""
    if key == "handle":
        return (0.38, 0.20)
    if key == "clock_mdc":
        return (-0.48, 0.68)
    if key == "mdc":
        return (0.36, 0.62)
    if key == "garters":
        return (-0.44, -0.24)
    if key == "catapult":
        return (-0.22, -0.36)
    if key == "g_load":
        u = C.smootherstep(888.0, 930.0, f)
        return (C.lerp(0.48, -0.48, u), C.lerp(0.60, 0.58, u))
    if key == "spine_gauge":
        return (0.40, 0.26)
    if key == "rocket":
        noz = _seat_local_to_world_pure(f, C.SEAT["rocket_nozzles"][0])
        nx, ny, _ = project_ndc(noz, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
        return (max(-0.50, min(-0.34, nx - 0.36)), max(-0.26, min(0.06, ny - 0.04)))
    if key == "drogue":
        dr = C.drogue_world_location(f)
        nx, ny, _ = project_ndc(dr, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
        return (max(0.28, min(0.44, nx + 0.58)), max(0.36, min(0.58, ny + 0.04)))
    if key == "clock_drogue":
        return (-0.46, 0.24)
    raise KeyError(f"Unknown label key: {key}")


def _plaque_depth(key: str, st: dict[str, Any]) -> float:
    """Keep plaque crisp on/near camera DOF plane while sitting cleanly in front of foreground geometry."""
    fd = float(st["focus_dist"])
    if key in ("spine_gauge", "g_load"):
        return max(0.75, fd - 0.16)
    if key in ("handle", "mdc", "clock_mdc", "garters", "catapult"):
        return max(0.65, fd - 0.45)
    return max(0.85, fd - 0.55)


def _camera_basis_pure(
    cam_pos: C.Vec3, tgt_pos: C.Vec3
) -> tuple[C.Vec3, C.Vec3, C.Vec3]:
    """Return right-handed orthonormal (cam_right, cam_up, cam_fwd) matching `cinema_camera.project_ndc`."""
    dx = tgt_pos[0] - cam_pos[0]
    dy = tgt_pos[1] - cam_pos[1]
    dz = tgt_pos[2] - cam_pos[2]
    fl = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
    fwd = (dx / fl, dy / fl, dz / fl)
    # cross(fwd, (0, 0, 1))
    rx, ry = fwd[1], -fwd[0]
    rl = math.sqrt(rx * rx + ry * ry) or 1e-9
    right = (rx / rl, ry / rl, 0.0)
    up = (
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    )
    return right, up, fwd


def _frame_half_w_m(depth_m: float, lens_mm: float) -> float:
    return depth_m * (0.5 * SENSOR_W_EFF_MM) / lens_mm


def _unproject_ndc_pure(
    ndc_x: float, ndc_y: float, depth_m: float, st: dict[str, Any]
) -> C.Vec3:
    cp = st["cam_pos"]
    right, up, fwd = _camera_basis_pure(cp, st["tgt_pos"])
    lens_mm = float(st["lens_mm"])
    x_m = ndc_x * depth_m * (0.5 * SENSOR_W_EFF_MM) / lens_mm
    y_m = ndc_y * depth_m * (0.5 * SENSOR_H_EFF_MM) / lens_mm
    return (
        cp[0] + fwd[0] * depth_m + right[0] * x_m + up[0] * y_m,
        cp[1] + fwd[1] * depth_m + right[1] * x_m + up[1] * y_m,
        cp[2] + fwd[2] * depth_m + right[2] * x_m + up[2] * y_m,
    )


def _quantized_label_text(key: str, f: int) -> str:
    """Deterministic discrete readout string at frame `f` for static & live-counter labels."""
    spec = _LABEL_SPECS[key]
    mode = spec["variant_mode"]
    if mode == "static":
        for k, txt, _ in C.LABELS:
            if k == key:
                return txt
    if mode == "clock_mdc":
        t_q = round(C.mission_clock_s(float(f)) * 100.0) / 100.0
        return f"{t_q:.2f} s"
    if mode == "clock_drogue":
        t = C.mission_clock_s(float(f))
        if t <= 0.255:
            t_q = round(t * 100.0) / 100.0
        else:
            t_q = round(t * 50.0) / 50.0
        return f"{t_q:.2f} s"
    if mode == "g_load":
        g_int = max(1, min(16, int(round(C.g_load(float(f))))))
        return f"{g_int:2d} G"
    if mode == "spine_gauge":
        step = max(0, min(25, int(round(C.spine_compression_cm(float(f)) * 10.0))))
        return f"SPINE: \u2212{step * 0.1:.1f} cm"
    raise KeyError(f"Unsupported variant mode: {mode}")


def _build_variant_schedule(
    key: str, w0: int, w1: int
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Build ordered unique text strings and their contiguous visible frame ranges within [w0, w1]."""
    if w0 > w1:
        return [], {}
    ordered_texts: list[str] = []
    ranges_by_text: dict[str, list[tuple[int, int]]] = {}
    cur_txt: str | None = None
    run_start = w0
    for f in range(w0, w1 + 1):
        txt = _quantized_label_text(key, f)
        if txt != cur_txt:
            if cur_txt is not None:
                ranges_by_text[cur_txt].append((run_start, f - 1))
            if txt not in ranges_by_text:
                ordered_texts.append(txt)
                ranges_by_text[txt] = []
            cur_txt = txt
            run_start = f
    if cur_txt is not None:
        ranges_by_text[cur_txt].append((run_start, w1))
    return ordered_texts, ranges_by_text


def _load_ttf_advances(font_path: Path) -> dict[int, float]:
    """Read horizontal advance widths (in em units) from TTF `head`/`hhea`/`cmap`/`hmtx` tables."""
    if not font_path.is_file():
        return {}
    data = font_path.read_bytes()
    if len(data) < 12:
        return {}
    num_tables = struct.unpack(">H", data[4:6])[0]
    tables: dict[str, bytes] = {}
    for i in range(num_tables):
        off = 12 + i * 16
        tag = data[off : off + 4].decode("ascii", errors="ignore")
        t_off, t_len = struct.unpack(">II", data[off + 8 : off + 16])
        tables[tag] = data[t_off : t_off + t_len]
    if not {"head", "hhea", "cmap", "hmtx"}.issubset(tables):
        return {}

    upem = struct.unpack(">H", tables["head"][18:20])[0] or 1000
    num_hmetrics = struct.unpack(">H", tables["hhea"][34:36])[0]
    cmap = tables["cmap"]
    nsub = struct.unpack(">H", cmap[2:4])[0]
    gid_map: dict[int, int] = {}
    for i in range(nsub):
        pid, eid, sub_off = struct.unpack(">HHI", cmap[4 + i * 8 : 12 + i * 8])
        if pid == 3 and eid == 1:
            st = cmap[sub_off:]
            fmt = struct.unpack(">H", st[0:2])[0]
            if fmt == 4:
                sc = struct.unpack(">H", st[6:8])[0] // 2
                ends = struct.unpack(f">{sc}H", st[14 : 14 + 2 * sc])
                starts = struct.unpack(f">{sc}H", st[16 + 2 * sc : 16 + 4 * sc])
                deltas = struct.unpack(f">{sc}h", st[16 + 4 * sc : 16 + 6 * sc])
                roffsets = struct.unpack(f">{sc}H", st[16 + 6 * sc : 16 + 8 * sc])
                for idx in range(sc):
                    if starts[idx] == 0xFFFF:
                        continue
                    for cp in range(starts[idx], ends[idx] + 1):
                        if roffsets[idx] == 0:
                            gid = (cp + deltas[idx]) & 0xFFFF
                        else:
                            pos = 16 + 6 * sc + idx * 2 + roffsets[idx] + (cp - starts[idx]) * 2
                            gid = struct.unpack(">H", st[pos : pos + 2])[0]
                            if gid != 0:
                                gid = (gid + deltas[idx]) & 0xFFFF
                        gid_map[cp] = gid
            break

    hmtx = tables["hmtx"]
    advances: dict[int, float] = {}
    for cp, gid in gid_map.items():
        if gid < num_hmetrics:
            aw = struct.unpack(">H", hmtx[gid * 4 : gid * 4 + 2])[0]
        else:
            aw = struct.unpack(">H", hmtx[(num_hmetrics - 1) * 4 : num_hmetrics * 4 - 2])[0]
        advances[cp] = aw / float(upem)
    return advances


def _est_plaque_half_dims_ref(
    text: str,
    char_h_ndc: float,
    is_hud: bool,
    advances: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Exact/conservative reference half-dimensions in `half_w_m=1.0` plaque space."""
    size_ref = char_h_ndc * ASPECT_H_OVER_W
    lines = text.splitlines() or [text]
    if advances:
        max_em = max(
            sum(advances.get(ord(ch), 0.72) * 1.03 for ch in ln) for ln in lines
        )
        text_w = max_em * size_ref
    else:
        text_w = max(len(ln) for ln in lines) * size_ref * 0.76
    text_h = len(lines) * size_ref * 0.85
    pad_r = max(0.028, size_ref * 0.54)
    pad_l = pad_r if is_hud else (pad_r + 0.038)
    pad_y_top = max(0.020, size_ref * (0.34 if is_hud else 0.42))
    pad_y_bot = max(0.026, size_ref * (0.40 if is_hud else 0.54))
    return 0.5 * (text_w + pad_l + pad_r), 0.5 * (text_h + pad_y_top + pad_y_bot)


def verify_labels_ndc(
    total_frames: int = C.TOTAL_FRAMES,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> dict[str, Any]:
    """
    Pure-Python QA verifying all 10 labels' plaque centers, 4 corners, target anchors,
    `NDC Y >= C.LABEL_MIN_NDC_Y (-0.45)`, and zero simultaneous pairwise overlaps across `1..total_frames`.
    """
    T = max(1, int(total_frames))
    errors: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    advances = _load_ttf_advances(Path(font_path))

    windows = {k: (max(1, w0), min(T, w1)) for k, _, (w0, w1) in C.LABELS}
    ref_boxes: dict[str, tuple[float, float]] = {}
    for k, default_txt, (w0, w1) in C.LABELS:
        spec = _LABEL_SPECS[k]
        texts, _ = _build_variant_schedule(k, max(1, w0), min(T, w1))
        sample_texts = texts or [default_txt]
        max_hw = max(
            _est_plaque_half_dims_ref(t, spec["char_h_ndc"], spec["is_hud"], advances)[0]
            for t in sample_texts
        )
        max_hh_m = max(
            _est_plaque_half_dims_ref(t, spec["char_h_ndc"], spec["is_hud"], advances)[1]
            for t in sample_texts
        )
        ref_boxes[k] = (max_hw, max_hh_m / ASPECT_H_OVER_W)
        summary[k] = {"min_x": 9.0, "max_x": -9.0, "min_y": 9.0, "max_y": -9.0, "variants": len(sample_texts)}

    for f in range(1, T + 1):
        st = camera_state_at(float(f), include_motion_fx=True)
        active: list[tuple[str, float, float, float, float]] = []
        for k, (w0, w1) in windows.items():
            if not (w0 <= f <= w1):
                continue
            cx, cy = _plaque_ndc_center(k, float(f), st)
            hw, hh = ref_boxes[k]
            z_p = _plaque_depth(k, st)
            w_pos = _unproject_ndc_pure(cx, cy, z_p, st)
            right, up, _ = _camera_basis_pure(st["cam_pos"], st["tgt_pos"])
            s = _frame_half_w_m(z_p, float(st["lens_mm"]))

            # Project all 4 actual 3D plaque corners through cinema_camera.project_ndc
            x_coords: list[float] = []
            y_coords: list[float] = []
            hw_m = hw * s
            hh_m = (hh * ASPECT_H_OVER_W) * s
            for sx_c in (-1.0, 1.0):
                for sy_c in (-1.0, 1.0):
                    corner_w = (
                        w_pos[0] + right[0] * (sx_c * hw_m) + up[0] * (sy_c * hh_m),
                        w_pos[1] + right[1] * (sx_c * hw_m) + up[1] * (sy_c * hh_m),
                        w_pos[2] + right[2] * (sx_c * hw_m) + up[2] * (sy_c * hh_m),
                    )
                    cnx, cny, cz = project_ndc(corner_w, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
                    if cz <= 0.15:
                        errors.append(f"f={f} {k}: corner behind camera depth={cz:.2f}")
                    x_coords.append(cnx)
                    y_coords.append(cny)

            x0, x1 = min(x_coords), max(x_coords)
            y0, y1 = min(y_coords), max(y_coords)
            sm = summary[k]
            sm["min_x"] = min(sm["min_x"], x0)
            sm["max_x"] = max(sm["max_x"], x1)
            sm["min_y"] = min(sm["min_y"], y0)
            sm["max_y"] = max(sm["max_y"], y1)

            if x0 < -0.86 or x1 > 0.86:
                errors.append(f"f={f} {k}: X corners [{x0:+.2f}, {x1:+.2f}] out of [-0.86, +0.86]")
            if y0 < C.LABEL_MIN_NDC_Y or y1 > 0.86:
                errors.append(f"f={f} {k}: Y corners [{y0:+.2f}, {y1:+.2f}] out of [{C.LABEL_MIN_NDC_Y}, +0.86]")

            aw = _target_world_pos(k, float(f))
            if aw is not None:
                ax, ay, az = project_ndc(aw, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
                if az <= 0.15:
                    errors.append(f"f={f} {k}: target anchor depth={az:.2f}")
                if (x0 - 0.015 <= ax <= x1 + 0.015) and (y0 - 0.015 <= ay <= y1 + 0.015):
                    errors.append(f"f={f} {k}: plaque occludes own anchor ({ax:+.2f}, {ay:+.2f})")

            active.append((k, x0, x1, y0, y1))

        for i in range(len(active)):
            k1, a_x0, a_x1, a_y0, a_y1 = active[i]
            for j in range(i + 1, len(active)):
                k2, b_x0, b_x1, b_y0, b_y1 = active[j]
                pad = 0.025
                if not (a_x1 + pad < b_x0 or b_x1 + pad < a_x0 or a_y1 + pad < b_y0 or b_y1 + pad < a_y0):
                    errors.append(f"f={f}: overlap {k1} vs {k2}")

    return {"passed": len(errors) == 0, "errors": errors, "summary": summary}


# ---------------------------------------------------------------------------
# 3. BLENDER HELPERS, BMESH PRIMITIVES & 2-ZONE SHADER LIBRARY
# ---------------------------------------------------------------------------
def _link_obj(obj: "bpy.types.Object", col: "bpy.types.Collection") -> "bpy.types.Object":
    target = col if col is not None else bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def _set_interp(
    id_block: Any,
    interp: str,
    data_path_prefix: str = "",
    skip_prefix: str = "",
) -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if data_path_prefix and not fc.data_path.startswith(data_path_prefix):
            continue
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _key_visibility(
    obj: "bpy.types.Object",
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe `hide_viewport` & `hide_render` with `CONSTANT` interpolation."""
    t_max = max(1, int(total_frames))
    if not visible_ranges:
        for f in (1, t_max):
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("hide_render", frame=f)
        _set_interp(obj, "CONSTANT", "hide_")
        return

    frames = {1, t_max}
    for s, e in visible_ranges:
        sc = max(1, min(t_max, int(s)))
        ec = max(1, min(t_max, int(e)))
        if sc > 1:
            frames.add(sc - 1)
        frames.add(sc)
        frames.add(ec)
        if ec < t_max:
            frames.add(ec + 1)

    for f in sorted(frames):
        vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not vis
        obj.hide_render = not vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_interp(obj, "CONSTANT", "hide_")


def _create_mesh_obj(
    name: str,
    bm: "bmesh.types.BMesh",
    materials: list["bpy.types.Material"] | "bpy.types.Material",
    col: "bpy.types.Collection",
    parent: "bpy.types.Object | None" = None,
    location: tuple[float, float, float] | "Vector" = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | "Euler" = (0.0, 0.0, 0.0),
    smooth_all: bool | None = True,
) -> "bpy.types.Object":
    """Attach material slots BEFORE `bm.to_mesh(mesh)` per AGENTS.md Section 3."""
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for mat in mat_list:
        mesh.materials.append(mat)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    if smooth_all is not None and len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth_all] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation, "XYZ") if not isinstance(rotation, Euler) else rotation.copy()
    obj.visible_shadow = False
    if parent is not None:
        obj.parent = parent
    return _link_obj(obj, col)


def _make_rounded_plaque_bm(
    width: float,
    height: float,
    thickness: float,
    corner_r: float,
    corner_segs: int = 6,
    bevel_w: float = 0.0020,
) -> "bmesh.types.BMesh":
    """Build a chamfered rounded-rectangle backing plaque centered at (0, 0, 0) in XY."""
    bm = bmesh.new()
    hw = max(0.010, 0.5 * width)
    hh = max(0.006, 0.5 * height)
    r = min(corner_r, hw * 0.45, hh * 0.45)
    hz = 0.5 * thickness
    b = min(bevel_w, hz * 0.45, r * 0.45)

    pts_2d: list[tuple[float, float, float, float]] = []
    corners = [
        (hw - r, hh - r, 0.0, 0.5 * math.pi),
        (-hw + r, hh - r, 0.5 * math.pi, math.pi),
        (-hw + r, -hh + r, math.pi, 1.5 * math.pi),
        (hw - r, -hh + r, 1.5 * math.pi, 2.0 * math.pi),
    ]
    for cx, cy, a0, a1 in corners:
        for s in range(corner_segs + 1):
            a = C.lerp(a0, a1, s / corner_segs)
            ca, sa = math.cos(a), math.sin(a)
            pts_2d.append((cx + r * ca, cy + r * sa, ca, sa))

    n_pts = len(pts_2d)
    r_bi = [bm.verts.new((x - b * nx, y - b * ny, -hz)) for x, y, nx, ny in pts_2d]
    r_bo = [bm.verts.new((x, y, -hz + b)) for x, y, nx, ny in pts_2d]
    r_fo = [bm.verts.new((x, y, +hz - b)) for x, y, nx, ny in pts_2d]
    r_fi = [bm.verts.new((x - b * nx, y - b * ny, +hz)) for x, y, nx, ny in pts_2d]

    for ring_a, ring_b in ((r_bi, r_bo), (r_bo, r_fo), (r_fo, r_fi)):
        for i in range(n_pts):
            inext = (i + 1) % n_pts
            f = bm.faces.new((ring_a[i], ring_a[inext], ring_b[inext], ring_b[i]))
            f.smooth = True

    f_front = bm.faces.new(r_fi)
    f_front.smooth = False
    f_back = bm.faces.new(r_bi[::-1])
    f_back.smooth = False
    return bm


def _make_unit_cylinder_bm(n_seg: int = 12) -> "bmesh.types.BMesh":
    """Unit-radius cylinder spanning Z in [0.0, 1.0] for zero-rebuild dynamic pointer segments."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=n_seg,
        radius1=1.0,
        radius2=1.0,
        depth=1.0,
        matrix=Matrix.Translation((0.0, 0.0, 0.5)),
    )
    return bm


def _make_torus_bm(
    major_r: float,
    minor_r: float,
    major_seg: int = 28,
    minor_seg: int = 10,
) -> "bmesh.types.BMesh":
    """Build a closed smooth torus in the local XY plane (normal along +Z)."""
    bm = bmesh.new()
    rings: list[list[Any]] = []
    for i in range(major_seg):
        a = i * 2.0 * math.pi / major_seg
        ca, sa = math.cos(a), math.sin(a)
        ring = []
        for j in range(minor_seg):
            b = j * 2.0 * math.pi / minor_seg
            cb, sb = math.cos(b), math.sin(b)
            r = major_r + minor_r * cb
            ring.append(bm.verts.new((r * ca, r * sa, minor_r * sb)))
        rings.append(ring)

    for i in range(major_seg):
        inext = (i + 1) % major_seg
        for j in range(minor_seg):
            jnext = (j + 1) % minor_seg
            f = bm.faces.new((rings[i][j], rings[inext][j], rings[inext][jnext], rings[i][jnext]))
            f.smooth = True
    return bm


def _build_materials() -> dict[str, "bpy.types.Material"]:
    """Build AgX-safe 2-zone brushed-face text shaders, dark obsidian plaques, & glowing accents."""
    mats: dict[str, bpy.types.Material] = {}

    def make_3d_text_mat(
        name: str,
        face_col: tuple[float, float, float],
        emit_strength: float,
    ) -> "bpy.types.Material":
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")

        # Scale-invariant Object-Space |N.z| > 0.85 separation (AGENTS.md Section 2)
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        vec_t = nt.nodes.new("ShaderNodeVectorTransform")
        vec_t.vector_type = "NORMAL"
        vec_t.convert_from = "WORLD"
        vec_t.convert_to = "OBJECT"

        norm = nt.nodes.new("ShaderNodeVectorMath")
        norm.operation = "NORMALIZE"

        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        abs_z = nt.nodes.new("ShaderNodeMath")
        abs_z.operation = "ABSOLUTE"

        mask = nt.nodes.new("ShaderNodeMapRange")
        mask.clamp = True
        mask.inputs["From Min"].default_value = 0.83
        mask.inputs["From Max"].default_value = 0.92

        nt.links.new(geom.outputs["Normal"], vec_t.inputs["Vector"])
        nt.links.new(vec_t.outputs["Vector"], norm.inputs[0])
        nt.links.new(norm.outputs["Vector"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], abs_z.inputs[0])
        nt.links.new(abs_z.outputs["Value"], mask.inputs["Value"])

        face_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        face_bsdf.inputs["Base Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Metallic"].default_value = 0.10
        face_bsdf.inputs["Roughness"].default_value = 0.34
        face_bsdf.inputs["Specular IOR Level"].default_value = 0.18
        face_bsdf.inputs["Emission Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Emission Strength"].default_value = min(2.05, float(emit_strength))

        side_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        side_bsdf.inputs["Base Color"].default_value = (0.010, 0.014, 0.022, 1.0)
        side_bsdf.inputs["Metallic"].default_value = 0.78
        side_bsdf.inputs["Roughness"].default_value = 0.36

        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mask.outputs["Result"], mix.inputs["Fac"])
        nt.links.new(side_bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(face_bsdf.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        return mat

    def make_solid_mat(
        name: str,
        base_col: tuple[float, float, float],
        metallic: float,
        roughness: float,
        spec_ior: float,
        emit_col: tuple[float, float, float] | None = None,
        emit_strength: float = 0.0,
    ) -> "bpy.types.Material":
        mat = bpy.data.materials.new(f"{PREFIX}Mat_{name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*base_col, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = spec_ior
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.0
        if emit_col is not None and emit_strength > 0.0:
            bsdf.inputs["Emission Color"].default_value = (*emit_col, 1.0)
            bsdf.inputs["Emission Strength"].default_value = min(2.05, float(emit_strength))
        return mat

    # Text face materials (AgX-safe <= 2.05 emission)
    mats["text_white"] = make_3d_text_mat("Text_CrispWhite", (0.98, 0.99, 1.00), 1.65)
    mats["text_cyan"] = make_3d_text_mat("Text_Cyan", (0.14, 0.90, 1.00), 1.72)
    mats["text_gold"] = make_3d_text_mat("Text_Gold", (1.00, 0.64, 0.04), 1.82)
    mats["text_amber"] = make_3d_text_mat("Text_Amber", (1.00, 0.48, 0.03), 1.88)
    mats["text_crimson"] = make_3d_text_mat("Text_Crimson", (1.00, 0.25, 0.05), 1.92)

    # Dark obsidian glass backing plaque & glyph keyline
    mats["plaque_obsidian"] = make_solid_mat(
        "Obsidian_Plaque", (0.008, 0.012, 0.020), metallic=0.08, roughness=0.45, spec_ior=0.10
    )
    mats["text_keyline"] = make_solid_mat(
        "Obsidian_Keyline", (0.006, 0.009, 0.015), metallic=0.06, roughness=0.48, spec_ior=0.08
    )

    # Glowing trim bars & pointer leader tubes / target reticles
    mats["accent_gold"] = make_solid_mat(
        "Accent_Gold", (1.00, 0.66, 0.06), 0.15, 0.28, 0.4, (1.00, 0.66, 0.06), 1.95
    )
    mats["accent_cyan"] = make_solid_mat(
        "Accent_Cyan", (0.06, 0.84, 1.00), 0.15, 0.28, 0.4, (0.06, 0.84, 1.00), 1.95
    )
    mats["accent_amber"] = make_solid_mat(
        "Accent_Amber", (1.00, 0.44, 0.03), 0.15, 0.28, 0.4, (1.00, 0.44, 0.03), 2.00
    )
    mats["accent_amber_gold"] = make_solid_mat(
        "Accent_AmberGold", (1.00, 0.64, 0.04), 0.15, 0.28, 0.4, (1.00, 0.64, 0.04), 1.95
    )
    mats["accent_crimson"] = make_solid_mat(
        "Accent_Crimson", (1.00, 0.24, 0.04), 0.15, 0.28, 0.4, (1.00, 0.24, 0.04), 2.00
    )

    return mats


def _variant_text_material(
    key: str, text_str: str, mats: dict[str, "bpy.types.Material"]
) -> "bpy.types.Material":
    """Pick clean AgX-safe text material per label and live counter severity."""
    if key in ("clock_mdc", "clock_drogue"):
        return mats["text_gold"]
    if key == "g_load":
        g_val = int(text_str.strip().split()[0])
        if g_val >= 12:
            return mats["text_crimson"]
        if g_val >= 6:
            return mats["text_amber"]
        return mats["text_cyan"]
    if key == "spine_gauge":
        cm_str = text_str.split("\u2212")[-1].split()[0]
        cm_val = float(cm_str)
        if cm_val >= 1.8:
            return mats["text_crimson"]
        if cm_val >= 0.8:
            return mats["text_amber"]
        return mats["text_cyan"]
    return mats["text_white"]


# ---------------------------------------------------------------------------
# 4. BLENDER SCENE BUILDER & ANIMATOR PUBLIC API
# ---------------------------------------------------------------------------
def _make_plaque_group(
    key: str,
    texts: list[str],
    spec: dict[str, Any],
    font_data: "bpy.types.VectorFont",
    mats: dict[str, "bpy.types.Material"],
    col: "bpy.types.Collection",
    rig: "bpy.types.Object",
) -> dict[str, Any]:
    """
    Build `Label_PlaqueGroup_<key>` at canonical `half_w_m=1.0` reference scale,
    housing all discrete font + keyline variants, exact-fitted dark obsidian plaque,
    and glowing trim bars.
    """
    char_h_ndc = float(spec["char_h_ndc"])
    size_ref = char_h_ndc * ASPECT_H_OVER_W
    is_hud = bool(spec["is_hud"])
    accent_mat = mats[f"accent_{spec['theme']}"]

    group = bpy.data.objects.new(f"{PREFIX}PlaqueGroup_{key}", None)
    group.empty_display_type = "PLAIN_AXES"
    group.empty_display_size = 0.05
    group.rotation_mode = "QUATERNION"
    group.parent = rig
    _link_obj(group, col)

    variant_pairs: list[tuple[str, "bpy.types.Object", "bpy.types.Object"]] = []
    text_objs: list["bpy.types.Object"] = []
    keyline_objs: list["bpy.types.Object"] = []

    use_suffix = len(texts) > 1
    for idx, body in enumerate(texts):
        sfx = f"_v{idx:02d}" if use_suffix else ""
        tmat = _variant_text_material(key, body, mats)

        cdata = bpy.data.curves.new(f"{PREFIX}Text_{key}{sfx}", type="FONT")
        cdata.body = body
        cdata.font = font_data
        cdata.size = size_ref
        cdata.align_x = "CENTER"
        cdata.align_y = "CENTER"
        cdata.extrude = size_ref * 0.048
        cdata.bevel_depth = size_ref * 0.0075
        cdata.bevel_resolution = 2
        cdata.space_character = 1.03
        cdata.space_line = 1.12

        tobj = bpy.data.objects.new(f"{PREFIX}Text_{key}{sfx}", cdata)
        tobj.data.materials.append(tmat)
        tobj.parent = group
        tobj.visible_shadow = False
        _link_obj(tobj, col)

        kdata = bpy.data.curves.new(f"{PREFIX}Keyline_{key}{sfx}", type="FONT")
        kdata.body = body
        kdata.font = font_data
        kdata.size = size_ref
        kdata.align_x = "CENTER"
        kdata.align_y = "CENTER"
        kdata.extrude = size_ref * 0.032
        kdata.offset = size_ref * 0.045
        kdata.space_character = 1.03
        kdata.space_line = 1.12

        kobj = bpy.data.objects.new(f"{PREFIX}Keyline_{key}{sfx}", kdata)
        kobj.location = Vector((0.0, 0.0, -0.0045))
        kobj.data.materials.append(mats["text_keyline"])
        kobj.parent = group
        kobj.visible_shadow = False
        _link_obj(kobj, col)

        variant_pairs.append((body, tobj, kobj))
        text_objs.append(tobj)
        keyline_objs.append(kobj)

    # Evaluate union tight glyph bounds across all discrete variants
    bpy.context.view_layer.update()
    min_x, max_x = 1e9, -1e9
    min_y, max_y = 1e9, -1e9
    for _, tobj, _ in variant_pairs:
        xs = [v[0] for v in tobj.bound_box]
        ys = [v[1] for v in tobj.bound_box]
        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))

    text_w = max_x - min_x
    text_h = max_y - min_y
    if text_w < 1e-4 or text_h < 1e-4:
        est_hw, est_hh = max(
            _est_plaque_half_dims_ref(t, char_h_ndc, is_hud) for t in texts
        )
        text_w = est_hw * 1.4
        text_h = est_hh * 1.3
        min_x, max_x = -0.5 * text_w, 0.5 * text_w
        min_y, max_y = -0.5 * text_h, 0.5 * text_h

    text_cx = 0.5 * (min_x + max_x)
    text_cy = 0.5 * (min_y + max_y)

    pad_r = max(0.028, size_ref * 0.54)
    pad_l = pad_r if is_hud else (pad_r + 0.038)
    pad_y_top = max(0.020, size_ref * (0.34 if is_hud else 0.42))
    pad_y_bot = max(0.026, size_ref * (0.40 if is_hud else 0.54))

    plaque_w = text_w + pad_l + pad_r
    plaque_h = text_h + pad_y_top + pad_y_bot
    plaque_thick = 0.011 if is_hud else 0.009
    plaque_front_z = -0.012
    corner_r = min(plaque_h * 0.30, 0.026 if is_hud else 0.020)

    # Shift text slightly right when left vertical accent pill bar is present so plaque stays centered at (0,0)
    x_shift = 0.5 * (pad_l - pad_r)
    y_shift = 0.5 * (pad_y_bot - pad_y_top)
    for _, tobj, kobj in variant_pairs:
        tobj.location = Vector((x_shift - text_cx, y_shift - text_cy, 0.0))
        kobj.location = Vector((x_shift - text_cx, y_shift - text_cy, -0.0045))

    plaque_bm = _make_rounded_plaque_bm(
        width=plaque_w,
        height=plaque_h,
        thickness=plaque_thick,
        corner_r=corner_r,
        corner_segs=6,
        bevel_w=0.0020,
    )
    plaque_obj = _create_mesh_obj(
        f"{PREFIX}Plaque_{key}",
        plaque_bm,
        mats["plaque_obsidian"],
        col,
        parent=group,
        location=(0.0, 0.0, plaque_front_z - 0.5 * plaque_thick),
        smooth_all=None,
    )

    bar_objs: list["bpy.types.Object"] = []
    if is_hud:
        bot_bar_bm = _make_rounded_plaque_bm(
            width=max(0.040, plaque_w - 2.2 * corner_r),
            height=0.0082,
            thickness=0.0042,
            corner_r=0.0034,
            corner_segs=4,
            bevel_w=0.0010,
        )
        bar_objs.append(
            _create_mesh_obj(
                f"{PREFIX}PlaqueBarBot_{key}",
                bot_bar_bm,
                accent_mat,
                col,
                parent=group,
                location=(0.0, -0.5 * plaque_h + 0.010, plaque_front_z + 0.0014),
                smooth_all=None,
            )
        )
        top_bar_bm = _make_rounded_plaque_bm(
            width=max(0.030, (plaque_w - 2.2 * corner_r) * 0.56),
            height=0.0048,
            thickness=0.0036,
            corner_r=0.0022,
            corner_segs=4,
            bevel_w=0.0008,
        )
        bar_objs.append(
            _create_mesh_obj(
                f"{PREFIX}PlaqueBarTop_{key}",
                top_bar_bm,
                accent_mat,
                col,
                parent=group,
                location=(0.0, +0.5 * plaque_h - 0.0085, plaque_front_z + 0.0012),
                smooth_all=None,
            )
        )
    else:
        left_bar_bm = _make_rounded_plaque_bm(
            width=0.0076,
            height=max(0.020, plaque_h * 0.62),
            thickness=0.0040,
            corner_r=0.0032,
            corner_segs=4,
            bevel_w=0.0008,
        )
        bar_objs.append(
            _create_mesh_obj(
                f"{PREFIX}PlaqueBarLeft_{key}",
                left_bar_bm,
                accent_mat,
                col,
                parent=group,
                location=(-0.5 * plaque_w + 0.015, 0.0025, plaque_front_z + 0.0014),
                smooth_all=None,
            )
        )
        bot_trim_bm = _make_rounded_plaque_bm(
            width=max(0.040, plaque_w - 2.2 * corner_r),
            height=0.0048,
            thickness=0.0036,
            corner_r=0.0020,
            corner_segs=4,
            bevel_w=0.0008,
        )
        bar_objs.append(
            _create_mesh_obj(
                f"{PREFIX}PlaqueBarBot_{key}",
                bot_trim_bm,
                accent_mat,
                col,
                parent=group,
                location=(0.0, -0.5 * plaque_h + 0.0072, plaque_front_z + 0.0012),
                smooth_all=None,
            )
        )

    ax_n, ay_n = spec["attach_norm"]
    attach_local = Vector((ax_n * plaque_w, ay_n * plaque_h, plaque_front_z + 0.001))

    return {
        "group": group,
        "plaque": plaque_obj,
        "bars": bar_objs,
        "variant_pairs": variant_pairs,
        "text_objs": text_objs,
        "keyline_objs": keyline_objs,
        "attach_local": attach_local,
    }


def _make_pointer_objects(
    key: str,
    accent_mat: "bpy.types.Material",
    mats: dict[str, "bpy.types.Material"],
    col: "bpy.types.Collection",
    rig: "bpy.types.Object",
) -> dict[str, "bpy.types.Object"]:
    """Build dark-rimmed pointer tube, obsidian sleeve, target dot, and reticle ring."""
    line_obj = _create_mesh_obj(
        f"{PREFIX}PointerLine_{key}",
        _make_unit_cylinder_bm(n_seg=10),
        accent_mat,
        col,
        parent=rig,
    )
    line_obj.rotation_mode = "QUATERNION"

    sleeve_obj = _create_mesh_obj(
        f"{PREFIX}PointerSleeve_{key}",
        _make_unit_cylinder_bm(n_seg=10),
        mats["plaque_obsidian"],
        col,
        parent=rig,
    )
    sleeve_obj.rotation_mode = "QUATERNION"

    dot_bm = bmesh.new()
    bmesh.ops.create_icosphere(dot_bm, subdivisions=2, radius=0.011)
    dot_obj = _create_mesh_obj(
        f"{PREFIX}PointerDot_{key}", dot_bm, accent_mat, col, parent=rig
    )
    dot_obj.rotation_mode = "QUATERNION"

    ret_obj = _create_mesh_obj(
        f"{PREFIX}PointerReticle_{key}",
        _make_torus_bm(0.020, 0.0028, major_seg=28, minor_seg=10),
        accent_mat,
        col,
        parent=rig,
    )
    ret_obj.rotation_mode = "QUATERNION"

    ret_bg_obj = _create_mesh_obj(
        f"{PREFIX}PointerReticleBg_{key}",
        _make_torus_bm(0.020, 0.0048, major_seg=28, minor_seg=10),
        mats["plaque_obsidian"],
        col,
        parent=rig,
    )
    ret_bg_obj.rotation_mode = "QUATERNION"

    return {
        "line": line_obj,
        "sleeve": sleeve_obj,
        "dot": dot_obj,
        "reticle": ret_obj,
        "reticle_bg": ret_bg_obj,
    }


def _track_z_quat(seg_dir: "Vector", prev_q: "Quaternion | None") -> "Quaternion":
    """Quaternion aligning local +Z along `seg_dir` with sign continuity."""
    q = seg_dir.to_track_quat("Z", "Y")
    if prev_q is not None and prev_q.dot(q) < 0.0:
        q = -q
    return q


def build_labels(
    col: "bpy.types.Collection",
    font_path: str | Path,
    seat_root: "bpy.types.Object | None",
    pilot_carrier: "bpy.types.Object | None",
    jet_root: "bpy.types.Object | None",
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """
    Build and animate all 10 3D HUD callouts, live discrete counters, dark obsidian
    backing plaques, and dynamic pointer reticles directly from `ejection_contract.py`
    and `cinema_camera.py`.
    """
    if bpy is None:
        raise RuntimeError("build_labels() must run inside Blender (`bpy` unavailable).")

    # Keep contract references explicit (zero mutation of external roots)
    _ = (seat_root, pilot_carrier, jet_root)

    T = max(1, int(total_frames))
    resolved_font = Path(font_path) if font_path else DEFAULT_FONT_PATH
    verify_font_glyphs(resolved_font)
    font_data = bpy.data.fonts.load(str(resolved_font), check_existing=True)

    mats = _build_materials()

    root_empty = bpy.data.objects.new(f"{PREFIX}Root", None)
    root_empty.empty_display_type = "PLAIN_AXES"
    root_empty.empty_display_size = 0.08
    _link_obj(root_empty, col)

    rigs: dict[str, "bpy.types.Object"] = {}
    plaques: dict[str, "bpy.types.Object"] = {}
    texts_map: dict[str, list["bpy.types.Object"]] = {}
    pointers_map: dict[str, list["bpy.types.Object"]] = {}
    windows_map: dict[str, tuple[int, int]] = {}
    all_objs: list["bpy.types.Object"] = [root_empty]

    for key, default_txt, (raw_w0, raw_w1) in C.LABELS:
        spec = _LABEL_SPECS[key]
        w0 = max(1, int(raw_w0))
        w1 = min(T, int(raw_w1))
        active = w0 <= w1
        windows_map[key] = (w0, w1) if active else (0, -1)

        rig = bpy.data.objects.new(f"{PREFIX}Rig_{key}", None)
        rig.empty_display_type = "PLAIN_AXES"
        rig.empty_display_size = 0.05
        rig.parent = root_empty
        _link_obj(rig, col)
        rigs[key] = rig
        all_objs.append(rig)

        if active:
            unique_texts, ranges_by_text = _build_variant_schedule(key, w0, w1)
        else:
            unique_texts, ranges_by_text = [default_txt], {}

        pg = _make_plaque_group(key, unique_texts, spec, font_data, mats, col, rig)
        group = pg["group"]
        plaque_obj = pg["plaque"]
        bar_objs = pg["bars"]
        attach_local: Vector = pg["attach_local"]

        plaques[key] = plaque_obj
        texts_map[key] = pg["text_objs"]
        all_objs.extend([group, plaque_obj] + bar_objs + pg["text_objs"] + pg["keyline_objs"])

        vis_win = [(w0, w1)] if active else []
        for obj in [plaque_obj] + bar_objs:
            _key_visibility(obj, vis_win, T)
        for body, tobj, kobj in pg["variant_pairs"]:
            var_win = ranges_by_text.get(body, []) if active else []
            _key_visibility(tobj, var_win, T)
            _key_visibility(kobj, var_win, T)

        ptr_dict: dict[str, "bpy.types.Object"] | None = None
        if spec["has_pointer"]:
            ptr_dict = _make_pointer_objects(key, mats[f"accent_{spec['theme']}"], mats, col, rig)
            ptr_list = [
                ptr_dict["line"],
                ptr_dict["sleeve"],
                ptr_dict["dot"],
                ptr_dict["reticle"],
                ptr_dict["reticle_bg"],
            ]
            pointers_map[key] = ptr_list
            all_objs.extend(ptr_list)
            for pobj in ptr_list:
                _key_visibility(pobj, vis_win, T)
        else:
            pointers_map[key] = []

        if not active:
            continue

        prev_q_bill: Quaternion | None = None
        prev_q_line: Quaternion | None = None
        for f in range(w0, w1 + 1):
            st = camera_state_at(float(f), include_motion_fx=True)
            cp = Vector(st["cam_pos"])
            lens_mm = float(st["lens_mm"])

            r_p, u_p, fwd_p = _camera_basis_pure(st["cam_pos"], st["tgt_pos"])
            v_right = Vector(r_p)
            v_up = Vector(u_p)
            v_fwd = Vector(fwd_p)

            rot_m = Matrix((v_right, v_up, -v_fwd)).transposed().to_3x3()
            q_bill = rot_m.to_quaternion()
            if prev_q_bill is not None and prev_q_bill.dot(q_bill) < 0.0:
                q_bill = -q_bill
            prev_q_bill = q_bill

            nx, ny = _plaque_ndc_center(key, float(f), st)
            z_p = _plaque_depth(key, st)
            s_p = _frame_half_w_m(z_p, lens_mm)
            plaque_pos = Vector(_unproject_ndc_pure(nx, ny, z_p, st))

            group.location = plaque_pos
            group.rotation_quaternion = q_bill
            group.scale = (s_p, s_p, s_p)
            group.keyframe_insert("location", frame=f)
            group.keyframe_insert("rotation_quaternion", frame=f)
            group.keyframe_insert("scale", frame=f)

            if ptr_dict is not None:
                t_pos_tuple = _target_world_pos(key, float(f))
                assert t_pos_tuple is not None
                t_pos = Vector(t_pos_tuple)
                edge_pos = plaque_pos + rot_m @ (attach_local * s_p)

                t_depth = max(0.25, (t_pos - cp).dot(v_fwd))
                s_t = _frame_half_w_m(t_depth, lens_mm)
                s_mid = 0.5 * (s_p + s_t)

                seg = t_pos - edge_pos
                seg_len = max(1e-4, seg.length)
                seg_dir = seg / seg_len
                q_line = _track_z_quat(seg_dir, prev_q_line)
                prev_q_line = q_line

                r_core = 0.0042 * s_mid
                r_sleeve = 0.0076 * s_mid
                sleeve_push = v_fwd * (0.0052 * s_mid)

                line_obj = ptr_dict["line"]
                line_obj.location = edge_pos
                line_obj.rotation_quaternion = q_line
                line_obj.scale = (r_core, r_core, seg_len)
                line_obj.keyframe_insert("location", frame=f)
                line_obj.keyframe_insert("rotation_quaternion", frame=f)
                line_obj.keyframe_insert("scale", frame=f)

                sleeve_obj = ptr_dict["sleeve"]
                sleeve_obj.location = edge_pos + sleeve_push
                sleeve_obj.rotation_quaternion = q_line
                sleeve_obj.scale = (r_sleeve, r_sleeve, seg_len)
                sleeve_obj.keyframe_insert("location", frame=f)
                sleeve_obj.keyframe_insert("rotation_quaternion", frame=f)
                sleeve_obj.keyframe_insert("scale", frame=f)

                ret_push = v_fwd * (0.0050 * s_t)
                for robj, offset in (
                    (ptr_dict["dot"], Vector((0.0, 0.0, 0.0))),
                    (ptr_dict["reticle"], Vector((0.0, 0.0, 0.0))),
                    (ptr_dict["reticle_bg"], ret_push),
                ):
                    robj.location = t_pos + offset
                    robj.rotation_quaternion = q_bill
                    robj.scale = (s_t, s_t, s_t)
                    robj.keyframe_insert("location", frame=f)
                    robj.keyframe_insert("rotation_quaternion", frame=f)
                    robj.keyframe_insert("scale", frame=f)

        _set_interp(group, "LINEAR", skip_prefix="hide_")
        if ptr_dict is not None:
            for pobj in pointers_map[key]:
                _set_interp(pobj, "LINEAR", skip_prefix="hide_")

    return {
        "root": root_empty,
        "rigs": rigs,
        "plaques": plaques,
        "texts": texts_map,
        "pointers": pointers_map,
        "windows": windows_map,
        "materials": mats,
        "all_objs": all_objs,
    }


# ---------------------------------------------------------------------------
# 5. STANDALONE SELF-TEST
# ---------------------------------------------------------------------------
def _run_blender_selftest() -> None:
    from cinema_camera import build_cinema_camera

    out_dir = Path("/tmp/fje_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 25
    scene.render.fps = C.FPS
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES
    scene.render.image_settings.file_format = "PNG"

    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        scene.view_settings.look = "None"

    cycles = scene.cycles
    cycles.device = "CPU"
    cycles.samples = 6
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"

    world = bpy.data.worlds.new("Label_TestWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.04, 0.08, 0.16, 1.0)
    bg.inputs["Strength"].default_value = 0.8

    col_env = bpy.data.collections.new("Env_Dummy")
    col_cam = bpy.data.collections.new("Cam_Dummy")
    col_lbl = bpy.data.collections.new("Labels")
    for c in (col_env, col_cam, col_lbl):
        scene.collection.children.link(c)

    # Dummy seat, pilot carrier, and jet empties animated via contract
    seat_root = bpy.data.objects.new("Seat_Root", None)
    pilot_carrier = bpy.data.objects.new("Pilot_Carrier", None)
    jet_root = bpy.data.objects.new("Env_Jet_Root", None)
    for o in (seat_root, pilot_carrier, jet_root):
        _link_obj(o, col_env)

    for f in range(1, C.TOTAL_FRAMES + 1):
        seat_root.location = Vector(C.seat_world_location(f))
        seat_root.rotation_euler = Euler(C.seat_world_rotation_euler(f), "XYZ")
        seat_root.keyframe_insert("location", frame=f)
        seat_root.keyframe_insert("rotation_euler", frame=f)

        pilot_carrier.location = Vector(C.pilot_carrier_location(f))
        pilot_carrier.rotation_euler = Euler(C.pilot_carrier_rotation_euler(f), "XYZ")
        pilot_carrier.keyframe_insert("location", frame=f)
        pilot_carrier.keyframe_insert("rotation_euler", frame=f)

        jet_root.location = Vector(C.jet_offset(f))
        jet_root.keyframe_insert("location", frame=f)

    # Subtle key light for bevel highlights
    ld = bpy.data.lights.new("Label_TestKey", type="AREA")
    ld.energy = 80.0
    ld.size = 2.0
    lobj = bpy.data.objects.new("Label_TestKey", ld)
    lobj.location = Vector((2.2, 2.0, 3.0))
    _link_obj(lobj, col_env)

    build_cinema_camera(col_cam, total_frames=C.TOTAL_FRAMES)
    res = build_labels(
        col=col_lbl,
        font_path=DEFAULT_FONT_PATH,
        seat_root=seat_root,
        pilot_carrier=pilot_carrier,
        jet_root=jet_root,
        total_frames=C.TOTAL_FRAMES,
    )

    assert len(res["rigs"]) == len(C.LABELS)
    for obj in res["all_objs"]:
        assert obj.name.startswith(PREFIX), f"Missing Label_ prefix: {obj.name}"

    for f_test in (180, 1080):
        scene.frame_set(f_test)
        scene.render.filepath = str(out_dir / f"labels_f{f_test:04d}.png")
        bpy.ops.render.render(write_still=True)

    print(
        f"labels_3d self-test PASSED: {len(res['rigs'])} labels, "
        f"{len(res['all_objs'])} objects rendered to {out_dir}"
    )


if __name__ == "__main__":
    if bpy is not None:
        _run_blender_selftest()
    else:
        verify_font_glyphs(DEFAULT_FONT_PATH)
        qa = verify_labels_ndc(C.TOTAL_FRAMES)
        print(f"Pure-Python NDC QA passed={qa['passed']} (errors={len(qa['errors'])})")
        for k, s in qa["summary"].items():
            print(
                f"  {k:13s}: X=[{s['min_x']:+.2f}, {s['max_x']:+.2f}] "
                f"Y=[{s['min_y']:+.2f}, {s['max_y']:+.2f}] variants={s['variants']}"
            )
        if not qa["passed"]:
            for err in qa["errors"][:20]:
                print("  ERR:", err)
            sys.exit(1)
