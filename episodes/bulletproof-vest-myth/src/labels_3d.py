"""
labels_3d.py — Razor-Sharp 3D Engineering Callouts & Pointer Reticles for Episode #3
"Why Bulletproof Vests DON'T Send You Flying Backward" (60 fps, 1800 frames, 30.0 s).

Strictly enforces AGENTS.md V12 & Episode #3 PRODUCTION_CONTRACT.md:
- Imports trajectories, timings, and 7 ultra-short labels from `vest_contract.py`.
- Imports optical projection & camera/DOF state from `cinema_camera.py`.
- Places every plaque strictly on the camera DOF focus plane (`focus_dist - 0.05 m`)
  at `f/7.5..f/8.0` for zero lens-bokeh blur.
- Zero fat/blurry typography: `char_h_ndc >= 0.034`, minimal bevel (`size_ref * 0.0022`),
  shallow extrude (`size_ref * 0.022`), airy tracking (`space_character = 1.10`),
  NO bloated `Keyline` mesh over the high-contrast dark obsidian backing plaque.
- 2-zone Object-Space `|N.z| > 0.85` face shader with AgX-safe emission (`<= 1.85`).
- Sets `visible_glossy = visible_transmission = visible_diffuse = False` on all HUD objects
  so refractive glass walls never reflect upside-down text.
- Keeps every plaque strictly on the camera side of the tempered glass wall (`X < glass_wall_x - 0.15 m`).
- Perpendicular `PointerSleeve` push so angled pointers glow cleanly.
- Pure-Python safe (`__main__` runs NDC, DOF, glass-clearance & overlap QA without Blender).

Public API:
    build_labels(
        col: bpy.types.Collection,
        font_path: str | Path = DEFAULT_FONT_PATH,
        total_frames: int = C.TOTAL_FRAMES,
    ) -> dict[str, Any]
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent
REPO_SRC_DIR = REPO_ROOT / "src"
for _p in (str(REPO_SRC_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vest_contract as C  # noqa: E402
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
EPISODE_FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
REPO_FONT_PATH = REPO_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
DEFAULT_FONT_PATH = EPISODE_FONT_PATH if EPISODE_FONT_PATH.is_file() else REPO_FONT_PATH
ASPECT_H_OVER_W = 16.0 / 9.0
LABEL_MIN_NDC_Y = -0.45

# ---------------------------------------------------------------------------
# 1. CONTRACT LABEL LAYOUT & STYLE SPECIFICATIONS (7 Callouts, V12 Razor-Sharp)
# ---------------------------------------------------------------------------
_LABEL_SPECS: dict[str, dict[str, Any]] = {
    "shift_half_in": {
        "char_h_ndc": 0.036,
        "theme": "gold",
        "attach_norm": (+0.28, -0.50),
    },
    "tiny_momentum": {
        "char_h_ndc": 0.035,
        "theme": "cyan",
        "attach_norm": (+0.22, -0.50),
    },
    "kevlar_net": {
        "char_h_ndc": 0.036,
        "theme": "gold",
        "attach_norm": (+0.18, -0.50),
    },
    "ceramic_dyneema": {
        "char_h_ndc": 0.034,
        "theme": "crimson",
        "attach_norm": (0.00, -0.50),
    },
    "newton_eq": {
        "char_h_ndc": 0.038,
        "theme": "cyan",
        "attach_norm": (0.00, -0.50),
    },
    "equal_recoil": {
        "char_h_ndc": 0.035,
        "theme": "gold",
        "attach_norm": (+0.25, +0.50),
    },
    "just_bruise": {
        "char_h_ndc": 0.036,
        "theme": "emerald_cyan",
        "attach_norm": (+0.25, -0.50),
    },
}


# ---------------------------------------------------------------------------
# 2. PURE-PYTHON FONT VERIFICATION, TARGETS & NDC GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def _resolve_font_path(font_path: str | Path | None) -> Path:
    if font_path:
        p = Path(font_path)
        if p.is_file():
            return p
    if EPISODE_FONT_PATH.is_file():
        return EPISODE_FONT_PATH
    if REPO_FONT_PATH.is_file():
        return REPO_FONT_PATH
    return Path(font_path) if font_path else EPISODE_FONT_PATH


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
    """Verify `font_path` exists and supports every character across all 7 contract labels."""
    resolved = _resolve_font_path(font_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required typography font missing: {resolved}")
    codepoints = _parse_ttf_codepoints(resolved)
    if texts is None:
        texts = [t for _, t, _ in C.LABELS]
    missing = sorted(
        {ch for s in texts for ch in s if ch not in ("\n", "\r") and ord(ch) not in codepoints}
    )
    if missing:
        raise ValueError(f"Font {resolved.name} missing required glyphs: {missing}")
    return codepoints


def _load_ttf_advances(font_path: Path) -> dict[int, float]:
    """Read horizontal advance widths (in em units) from TTF `head`/`hhea`/`cmap`/`hmtx` tables."""
    resolved = _resolve_font_path(font_path)
    if not resolved.is_file():
        return {}
    data = resolved.read_bytes()
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


def _target_world_pos(key: str, f: float) -> tuple[float, float, float]:
    """Exact world-space 3D component anchor point for each of the 7 contract callouts."""
    vx = float(C.RANGE["vest_surface_x"])
    cz = float(C.RANGE["chest_z"])
    if key == "shift_half_in":
        return (vx + C.hero_recoil_x(f), 0.0, cz)
    if key == "tiny_momentum":
        return (C.pistol_bullet_x(f), 0.0, cz)
    if key == "kevlar_net":
        return (vx + 0.018 + 0.014 * C.kevlar_ripple_t(f), 0.0, cz)
    if key == "ceramic_dyneema":
        return (vx + 0.018, 0.0, cz + 0.015)
    if key == "newton_eq":
        return (float(C.RANGE["b5_shooter_x"]) - 0.28, 0.0, float(C.RANGE["b5_vector_z"]))
    if key == "equal_recoil":
        return (float(C.RANGE["b5_target_x"]) + 0.28, 0.0, float(C.RANGE["b5_vector_z"]))
    if key == "just_bruise":
        return (vx, 0.0, cz)
    raise KeyError(f"Unknown label key: {key}")


def _plaque_ndc_center(key: str, f: float, st: dict[str, Any]) -> tuple[float, float]:
    """Deterministic collision-free 9:16 NDC center for each label across its active window."""
    _ = (f, st)
    centers: dict[str, tuple[float, float]] = {
        "shift_half_in": (-0.22, +0.56),
        "tiny_momentum": (-0.20, +0.54),
        "kevlar_net": (-0.12, +0.56),
        "ceramic_dyneema": (0.00, +0.58),
        "newton_eq": (0.00, +0.46),
        "equal_recoil": (0.00, -0.26),
        "just_bruise": (-0.24, +0.56),
    }
    if key not in centers:
        raise KeyError(f"Unknown label key: {key}")
    return centers[key]


def _plaque_depth(key: str, st: dict[str, Any]) -> float:
    """Place every plaque strictly on the exact camera DOF focus plane (`focus_dist - 0.05 m`)."""
    _ = key
    return max(0.25, float(st["focus_dist"]) - 0.05)


def _camera_basis_pure(
    cam_pos: tuple[float, float, float], tgt_pos: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return right-handed orthonormal (cam_right, cam_up, cam_fwd) matching `cinema_camera.project_ndc`."""
    dx = tgt_pos[0] - cam_pos[0]
    dy = tgt_pos[1] - cam_pos[1]
    dz = tgt_pos[2] - cam_pos[2]
    fl = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
    fwd = (dx / fl, dy / fl, dz / fl)
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
) -> tuple[float, float, float]:
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


def _plaque_padding(size_ref: float) -> tuple[float, float, float, float]:
    """Generous padding around glyphs + left accent bar so no letter ever clips."""
    pad_r = max(0.046, size_ref * 0.74)
    pad_l = pad_r + 0.062
    pad_y_top = max(0.028, size_ref * 0.50)
    pad_y_bot = max(0.034, size_ref * 0.58)
    return pad_l, pad_r, pad_y_top, pad_y_bot


def _est_plaque_half_dims_ref(
    text: str,
    char_h_ndc: float,
    advances: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Conservative reference half-dimensions in `half_w_m=1.0` plaque space."""
    size_ref = char_h_ndc * ASPECT_H_OVER_W
    lines = text.splitlines() or [text]
    if advances:
        max_em = max(
            sum(advances.get(ord(ch), 0.74) * 1.10 for ch in ln) for ln in lines
        )
        text_w = max_em * size_ref
    else:
        text_w = max(len(ln) for ln in lines) * size_ref * 0.80
    text_h = len(lines) * size_ref * 0.85
    pad_l, pad_r, pad_y_top, pad_y_bot = _plaque_padding(size_ref)
    return 0.5 * (text_w + pad_l + pad_r), 0.5 * (text_h + pad_y_top + pad_y_bot)


def _hero_clearance_points(f: float) -> list[tuple[str, tuple[float, float, float]]]:
    """Key character, armor & projectile world coordinates confirming zero HUD occlusion."""
    pts: list[tuple[str, tuple[float, float, float]]] = []
    cz = float(C.RANGE["chest_z"])
    ez = float(C.RANGE["eye_z"])
    if f <= 210.0:
        hx = float(C.HERO["b1_pos"][0]) + C.hero_recoil_x(f)
        pts.extend([
            ("b1_hero_eye", (hx, 0.0, ez)),
            ("b1_hero_chest", (float(C.RANGE["vest_surface_x"]) + C.hero_recoil_x(f), 0.0, cz)),
            ("b1_bullet", C.bullet_world_pos(f)),
        ])
    elif f <= 510.0:
        hx = float(C.HERO["b2_pos"][0]) + C.hero_recoil_x(f)
        pts.extend([
            ("b2_hero_eye", (hx, 0.0, ez)),
            ("b2_hero_chest", (float(C.RANGE["vest_surface_x"]) + C.hero_recoil_x(f), 0.0, cz)),
            ("b2_bullet", C.bullet_world_pos(f)),
        ])
    elif f <= 900.0:
        pts.extend([
            ("b3_kevlar_center", (float(C.RANGE["vest_surface_x"]) + 0.018, 0.0, cz)),
            ("b3_bullet", C.bullet_world_pos(f)),
        ])
    elif f <= 1260.0:
        pts.extend([
            ("b4_ceramic_center", (float(C.RANGE["vest_surface_x"]) + 0.015, 0.0, cz)),
            ("b4_rifle_bullet", C.bullet_world_pos(f)),
        ])
    elif f <= 1560.0:
        sx5 = float(C.RANGE["b5_shooter_x"]) + C.shooter_recoil_x(f)
        tx5 = float(C.RANGE["b5_target_x"]) + C.hero_recoil_x(f)
        pts.extend([
            ("b5_shooter_eye", (sx5, 0.0, ez)),
            ("b5_target_eye", (tx5, 0.0, ez)),
            ("b5_bullet", C.bullet_world_pos(f)),
        ])
    else:
        hx = float(C.HERO["b6_pos"][0])
        pts.extend([
            ("b6_hero_eye", (hx, 0.0, ez)),
            ("b6_hero_chest", (float(C.RANGE["vest_surface_x"]), 0.0, cz)),
        ])
    return pts


def verify_labels_ndc(
    total_frames: int = C.TOTAL_FRAMES,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> dict[str, Any]:
    """
    Pure-Python QA verifying all 7 labels' plaque centers, 4 corners, target anchors,
    DOF focus-plane alignment (`|z_p - focus_dist| <= 0.12 m`), `fstop >= 7.5`,
    `NDC Y >= -0.45`, glass-wall camera-side clearance, character/bullet clearance,
    and zero simultaneous pairwise overlaps across `1..total_frames`.
    """
    T = max(1, int(total_frames))
    errors: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    advances = _load_ttf_advances(Path(font_path))

    label_texts = {k: txt for k, txt, _ in C.LABELS}
    windows = {k: (max(1, w0), min(T, w1)) for k, _, (w0, w1) in C.LABELS}
    ref_boxes: dict[str, tuple[float, float]] = {}
    for k, txt, _ in C.LABELS:
        spec = _LABEL_SPECS[k]
        if float(spec["char_h_ndc"]) < 0.034:
            errors.append(f"{k}: char_h_ndc={spec['char_h_ndc']:.4f} < 0.034")
        word_count = len([w for w in txt.replace("+", " ").replace("=", " ").split() if w])
        if word_count > 3:
            errors.append(f"{k}: label '{txt}' exceeds 3-word limit ({word_count} tokens)")
        hw, hh_m = _est_plaque_half_dims_ref(txt, spec["char_h_ndc"], advances)
        ref_boxes[k] = (hw, hh_m / ASPECT_H_OVER_W)
        summary[k] = {
            "text": txt,
            "min_x": 9.0,
            "max_x": -9.0,
            "min_y": 9.0,
            "max_y": -9.0,
            "max_x_w": -999.0,
            "max_dof_err": 0.0,
            "min_fstop": 99.0,
        }

    glass_x_limit = float(C.RANGE["glass_wall_x"]) - 0.15
    for f in range(1, T + 1):
        st = camera_state_at(float(f), include_motion_fx=True)
        has_glass_wall = f <= 510 or f >= 1561
        active: list[tuple[str, float, float, float, float]] = []
        clr_pts = [
            (pname, project_ndc(wpt, st["cam_pos"], st["tgt_pos"], st["lens_mm"]))
            for pname, wpt in _hero_clearance_points(float(f))
        ]

        for k, (w0, w1) in windows.items():
            if not (w0 <= f <= w1):
                continue
            cx, cy = _plaque_ndc_center(k, float(f), st)
            hw, hh = ref_boxes[k]
            z_p = _plaque_depth(k, st)
            dof_err = abs(z_p - float(st["focus_dist"]))
            fstop_val = float(st["fstop"])
            summary[k]["max_dof_err"] = max(summary[k]["max_dof_err"], dof_err)
            summary[k]["min_fstop"] = min(summary[k]["min_fstop"], fstop_val)
            if dof_err > 0.12:
                errors.append(f"f={f} {k}: |z_p - focus_dist|={dof_err:.3f}m > 0.12m")
            if fstop_val < 7.45:
                errors.append(f"f={f} {k}: fstop={fstop_val:.2f} < 7.5 during active label")

            w_pos = _unproject_ndc_pure(cx, cy, z_p, st)
            right, up, _ = _camera_basis_pure(st["cam_pos"], st["tgt_pos"])
            s = _frame_half_w_m(z_p, float(st["lens_mm"]))

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
                    summary[k]["max_x_w"] = max(summary[k]["max_x_w"], corner_w[0])
                    if has_glass_wall and corner_w[0] > glass_x_limit:
                        errors.append(
                            f"f={f} {k}: corner_w[0]={corner_w[0]:+.3f}m > glass_x_limit={glass_x_limit:+.3f}m"
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

            if x0 < -0.88 or x1 > 0.88:
                errors.append(f"f={f} {k}: X corners [{x0:+.2f}, {x1:+.2f}] out of [-0.88, +0.88]")
            if y0 < LABEL_MIN_NDC_Y or y1 > 0.86:
                errors.append(f"f={f} {k}: Y corners [{y0:+.2f}, {y1:+.2f}] out of [{LABEL_MIN_NDC_Y}, +0.86]")

            aw = _target_world_pos(k, float(f))
            ax, ay, az = project_ndc(aw, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
            if az <= 0.15:
                errors.append(f"f={f} {k}: target anchor depth={az:.2f}")
            if (x0 - 0.02 <= ax <= x1 + 0.02) and (y0 - 0.02 <= ay <= y1 + 0.02):
                errors.append(f"f={f} {k}: plaque occludes own anchor ({ax:+.2f}, {ay:+.2f})")

            for pname, (sx, sy, sz) in clr_pts:
                if sz > 0.15 and (x0 - 0.02 <= sx <= x1 + 0.02) and (y0 - 0.02 <= sy <= y1 + 0.02):
                    errors.append(f"f={f} {k} ({label_texts[k]}): overlaps {pname} ({sx:+.2f}, {sy:+.2f})")

            active.append((k, x0, x1, y0, y1))

        for i in range(len(active)):
            k1, a_x0, a_x1, a_y0, a_y1 = active[i]
            for j in range(i + 1, len(active)):
                k2, b_x0, b_x1, b_y0, b_y1 = active[j]
                pad = 0.030
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
    for fc in C.iter_action_fcurves(ad.action):
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
    """Attach material slots BEFORE `bm.to_mesh(mesh)` per AGENTS.md V12."""
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
    bevel_w: float = 0.0018,
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
    """Unit-radius cylinder spanning Z in [0.0, 1.0] for dynamic pointer segments."""
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
    """Build AgX-safe (`<= 1.85`) 2-zone brushed-face text shaders, obsidian plaques, & glowing accents."""
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

        # Scale-invariant Object-Space |N.z| > 0.85 separation (AGENTS.md V12 Section 2)
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
        mask.inputs["From Min"].default_value = 0.85
        mask.inputs["From Max"].default_value = 0.92

        nt.links.new(geom.outputs["Normal"], vec_t.inputs["Vector"])
        nt.links.new(vec_t.outputs["Vector"], norm.inputs[0])
        nt.links.new(norm.outputs["Vector"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], abs_z.inputs[0])
        nt.links.new(abs_z.outputs["Value"], mask.inputs["Value"])

        face_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        face_bsdf.inputs["Base Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Metallic"].default_value = 0.08
        face_bsdf.inputs["Roughness"].default_value = 0.34
        face_bsdf.inputs["Specular IOR Level"].default_value = 0.16
        face_bsdf.inputs["Emission Color"].default_value = (*face_col, 1.0)
        face_bsdf.inputs["Emission Strength"].default_value = min(1.85, max(1.35, float(emit_strength)))

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
            bsdf.inputs["Emission Strength"].default_value = min(1.85, float(emit_strength))
        return mat

    mats["text_white"] = make_3d_text_mat("Text_CrispWhite", (0.98, 0.99, 1.00), 1.62)
    mats["plaque_obsidian"] = make_solid_mat(
        "Obsidian_Plaque", (0.008, 0.012, 0.020), metallic=0.08, roughness=0.45, spec_ior=0.10
    )
    mats["accent_gold"] = make_solid_mat(
        "Accent_Gold", (1.00, 0.68, 0.08), 0.15, 0.28, 0.4, (1.00, 0.68, 0.08), 1.80
    )
    mats["accent_cyan"] = make_solid_mat(
        "Accent_Cyan", (0.06, 0.84, 1.00), 0.15, 0.28, 0.4, (0.06, 0.84, 1.00), 1.80
    )
    mats["accent_crimson"] = make_solid_mat(
        "Accent_Crimson", (1.00, 0.26, 0.06), 0.15, 0.28, 0.4, (1.00, 0.26, 0.06), 1.85
    )
    mats["accent_emerald_cyan"] = make_solid_mat(
        "Accent_EmeraldCyan", (0.08, 0.96, 0.78), 0.15, 0.28, 0.4, (0.08, 0.96, 0.78), 1.80
    )

    return mats


# ---------------------------------------------------------------------------
# 4. BLENDER SCENE BUILDER & ANIMATOR PUBLIC API
# ---------------------------------------------------------------------------
def _make_plaque_group(
    key: str,
    body: str,
    spec: dict[str, Any],
    font_data: "bpy.types.VectorFont",
    mats: dict[str, "bpy.types.Material"],
    col: "bpy.types.Collection",
    rig: "bpy.types.Object",
) -> dict[str, Any]:
    """
    Build `Label_PlaqueGroup_<key>` at canonical `half_w_m=1.0` reference scale,
    with crisp minimal-bevel 3D text (NO Keyline mesh), exact-fitted dark obsidian
    plaque, and glowing trim accent bars.
    """
    char_h_ndc = float(spec["char_h_ndc"])
    size_ref = char_h_ndc * ASPECT_H_OVER_W
    accent_mat = mats[f"accent_{spec['theme']}"]

    group = bpy.data.objects.new(f"{PREFIX}PlaqueGroup_{key}", None)
    group.empty_display_type = "PLAIN_AXES"
    group.empty_display_size = 0.05
    group.rotation_mode = "QUATERNION"
    group.parent = rig
    _link_obj(group, col)

    cdata = bpy.data.curves.new(f"{PREFIX}Text_{key}", type="FONT")
    cdata.body = body
    cdata.font = font_data
    cdata.size = size_ref
    cdata.align_x = "CENTER"
    cdata.align_y = "CENTER"
    cdata.extrude = size_ref * 0.022
    cdata.bevel_depth = size_ref * 0.0022
    cdata.bevel_resolution = 1
    cdata.space_character = 1.10
    cdata.space_line = 1.15

    tobj = bpy.data.objects.new(f"{PREFIX}Text_{key}", cdata)
    tobj.data.materials.append(mats["text_white"])
    tobj.parent = group
    tobj.visible_shadow = False
    _link_obj(tobj, col)

    bpy.context.view_layer.update()
    xs = [v[0] for v in tobj.bound_box]
    ys = [v[1] for v in tobj.bound_box]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    text_w = max_x - min_x
    text_h = max_y - min_y
    if text_w < 1e-4 or text_h < 1e-4:
        est_hw, est_hh = _est_plaque_half_dims_ref(body, char_h_ndc)
        text_w = est_hw * 1.4
        text_h = est_hh * 1.3
        min_x, max_x = -0.5 * text_w, 0.5 * text_w
        min_y, max_y = -0.5 * text_h, 0.5 * text_h

    text_cx = 0.5 * (min_x + max_x)
    text_cy = 0.5 * (min_y + max_y)

    pad_l, pad_r, pad_y_top, pad_y_bot = _plaque_padding(size_ref)
    plaque_w = text_w + pad_l + pad_r
    plaque_h = text_h + pad_y_top + pad_y_bot
    plaque_thick = 0.0085
    plaque_front_z = -0.0090
    corner_r = min(plaque_h * 0.30, 0.020)

    x_shift = 0.5 * (pad_l - pad_r)
    y_shift = 0.5 * (pad_y_bot - pad_y_top)
    tobj.location = Vector((x_shift - text_cx, y_shift - text_cy, 0.0))

    plaque_bm = _make_rounded_plaque_bm(
        width=plaque_w,
        height=plaque_h,
        thickness=plaque_thick,
        corner_r=corner_r,
        corner_segs=6,
        bevel_w=0.0018,
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

    left_bar_bm = _make_rounded_plaque_bm(
        width=0.0078,
        height=max(0.020, plaque_h * 0.62),
        thickness=0.0038,
        corner_r=0.0032,
        corner_segs=4,
        bevel_w=0.0008,
    )
    left_bar_obj = _create_mesh_obj(
        f"{PREFIX}PlaqueBarLeft_{key}",
        left_bar_bm,
        accent_mat,
        col,
        parent=group,
        location=(-0.5 * plaque_w + 0.016, 0.0020, plaque_front_z + 0.0014),
        smooth_all=None,
    )

    bot_trim_bm = _make_rounded_plaque_bm(
        width=max(0.040, plaque_w - 2.2 * corner_r),
        height=0.0046,
        thickness=0.0034,
        corner_r=0.0020,
        corner_segs=4,
        bevel_w=0.0008,
    )
    bot_trim_obj = _create_mesh_obj(
        f"{PREFIX}PlaqueBarBot_{key}",
        bot_trim_bm,
        accent_mat,
        col,
        parent=group,
        location=(0.0, -0.5 * plaque_h + 0.0072, plaque_front_z + 0.0012),
        smooth_all=None,
    )

    ax_n, ay_n = spec["attach_norm"]
    attach_local = Vector((ax_n * plaque_w, ay_n * plaque_h, plaque_front_z + 0.001))

    return {
        "group": group,
        "plaque": plaque_obj,
        "bars": [left_bar_obj, bot_trim_obj],
        "text_obj": tobj,
        "attach_local": attach_local,
    }


def _make_pointer_objects(
    key: str,
    accent_mat: "bpy.types.Material",
    mats: dict[str, "bpy.types.Material"],
    col: "bpy.types.Collection",
    rig: "bpy.types.Object",
) -> dict[str, "bpy.types.Object"]:
    """Build dark-rimmed pointer tube, perpendicular obsidian sleeve, target dot, and reticle ring."""
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
    font_path: str | Path = DEFAULT_FONT_PATH,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """
    Build and animate all 7 razor-sharp 3D callouts, dark obsidian backing plaques,
    and perpendicular-sleeve pointer reticles from `vest_contract.py` & `cinema_camera.py`.
    """
    if bpy is None:
        raise RuntimeError("build_labels() must run inside Blender (`bpy` unavailable).")

    T = max(1, int(total_frames))
    resolved_font = _resolve_font_path(font_path)
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

    for key, body_txt, (raw_w0, raw_w1) in C.LABELS:
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

        pg = _make_plaque_group(key, body_txt, spec, font_data, mats, col, rig)
        group = pg["group"]
        plaque_obj = pg["plaque"]
        bar_objs = pg["bars"]
        text_obj = pg["text_obj"]
        attach_local: Vector = pg["attach_local"]

        plaques[key] = plaque_obj
        texts_map[key] = [text_obj]
        all_objs.extend([group, plaque_obj, text_obj] + bar_objs)

        vis_win = [(w0, w1)] if active else []
        for obj in [plaque_obj, text_obj] + bar_objs:
            obj.visible_glossy = False
            obj.visible_transmission = False
            obj.visible_diffuse = False
            _key_visibility(obj, vis_win, T)

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
            pobj.visible_glossy = False
            pobj.visible_transmission = False
            pobj.visible_diffuse = False
            _key_visibility(pobj, vis_win, T)

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

            t_pos = Vector(_target_world_pos(key, float(f)))
            edge_pos = plaque_pos + rot_m @ (attach_local * s_p)

            t_depth = max(0.25, (t_pos - cp).dot(v_fwd))
            s_t = _frame_half_w_m(t_depth, lens_mm)
            s_mid = 0.5 * (s_p + s_t)

            seg = t_pos - edge_pos
            seg_len = max(1e-4, seg.length)
            seg_dir = seg / seg_len
            q_line = _track_z_quat(seg_dir, prev_q_line)
            prev_q_line = q_line

            r_core = 0.0044 * s_mid
            r_sleeve = 0.0068 * s_mid
            perp_cam = v_fwd - seg_dir * v_fwd.dot(seg_dir)
            if perp_cam.length < 1e-4:
                perp_cam = -v_up
            else:
                perp_cam.normalize()
            sleeve_push = perp_cam * (0.0110 * s_mid)

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
        for pobj in ptr_list:
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


if __name__ == "__main__":
    verify_font_glyphs(DEFAULT_FONT_PATH)
    qa = verify_labels_ndc(C.TOTAL_FRAMES)
    print(f"Pure-Python NDC & DOF QA passed={qa['passed']} (errors={len(qa['errors'])})")
    for k, s in qa["summary"].items():
        print(
            f"  {k:16s} ({s['text']:18s}): X=[{s['min_x']:+.2f}, {s['max_x']:+.2f}] "
            f"Y=[{s['min_y']:+.2f}, {s['max_y']:+.2f}] max_x_w={s['max_x_w']:+.3f}m "
            f"f/{s['min_fstop']:.1f} max_dof_err={s['max_dof_err']:.3f}m"
        )
    if not qa["passed"]:
        for err in qa["errors"][:25]:
            print("  ERR:", err)
        sys.exit(1)
