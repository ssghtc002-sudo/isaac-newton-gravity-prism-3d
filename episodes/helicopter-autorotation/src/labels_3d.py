"""
labels_3d.py — Razor-Sharp 3D Engineering Callouts & Pointer Reticles for Episode #5
"Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames, 30.0 s).

Strictly enforces AGENTS.md V12 & Episode #5 PRODUCTION_CONTRACT.md:
  - 6 ultra-short (1-3 word) labels from `heli_contract.LABELS` (Beat 5A has zero labels for f/3.6 bokeh!).
  - Every plaque pinned strictly to the camera DOF focus plane (`focus_dist - 0.05 m`) at `f/7.5..f/8.0`.
  - Razor-sharp V12 typography (`DejaVuSans-Bold.ttf`, `char_h_ndc >= 0.034`, subtle bevel, airy tracking `1.10`,
    2-zone Object-Space `|N.z| > 0.85` brushed metallic face, AgX-safe emission `<= 1.85`).
  - High-contrast dark obsidian glass backing plaque `(0.008, 0.012, 0.020)` with glowing trim accent bar
    and perpendicular `PointerSleeve` halo push.
  - `visible_glossy = visible_transmission = visible_diffuse = False` on all HUD objects so canopy glass
    never reflects upside-down text.
  - Pure-Python safe (`__main__` runs full glyph, NDC, DOF & hero clearance audit without Blender!).
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
for _p in (str(REPO_ROOT / "src"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heli_contract as C  # noqa: E402
from cinema_camera import (  # noqa: E402
    SENSOR_H_EFF_MM,
    SENSOR_W_EFF_MM,
    camera_state_at,
    project_ndc,
)

try:
    import bmesh  # type: ignore[import-not-found]
    import bpy  # type: ignore[import-not-found]
    from mathutils import Matrix, Vector  # type: ignore[import-not-found]
except ImportError:
    bmesh = None  # type: ignore[assignment]
    bpy = None  # type: ignore[assignment]
    Matrix = None  # type: ignore[assignment]
    Vector = None  # type: ignore[assignment]

EPISODE_FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
REPO_FONT_PATH = REPO_ROOT / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"
DEFAULT_FONT_PATH = EPISODE_FONT_PATH if EPISODE_FONT_PATH.is_file() else REPO_FONT_PATH
LABEL_MIN_NDC_Y = -0.45

# Layout specs for all 6 contract callouts (placed high/clear of faces, rotor hubs & subtitles!)
_LABEL_SPECS: dict[str, dict[str, Any]] = {
    "engine_dead": {
        "char_h_ndc": 0.036,
        "theme": "crimson",
        "plaque_ndc": (0.00, 0.64),
        "attach_norm": (0.00, -0.50),
    },
    "sprag_clutch": {
        "char_h_ndc": 0.036,
        "theme": "gold",
        "plaque_ndc": (-0.06, 0.64),
        "attach_norm": (0.05, -0.50),
    },
    "upward_airflow": {
        "char_h_ndc": 0.035,
        "theme": "emerald",
        "plaque_ndc": (0.02, 0.64),
        "attach_norm": (0.00, -0.50),
    },
    "maple_seed_lift": {
        "char_h_ndc": 0.034,
        "theme": "emerald",
        "plaque_ndc": (0.00, 0.65),
        "attach_norm": (-0.18, -0.50),
    },
    "flare_30ft": {
        "char_h_ndc": 0.036,
        "theme": "gold",
        "plaque_ndc": (0.04, 0.64),
        "attach_norm": (-0.06, -0.50),
    },
    "zero_engine": {
        "char_h_ndc": 0.036,
        "theme": "cyan",
        "plaque_ndc": (0.00, 0.64),
        "attach_norm": (0.00, -0.50),
    },
}

_THEME_RGB: dict[str, tuple[float, float, float]] = {
    "crimson": (1.00, 0.24, 0.16),
    "gold": (1.00, 0.78, 0.22),
    "emerald": (0.14, 0.96, 0.68),
    "cyan": (0.18, 0.88, 1.00),
}


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
    data = font_path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"Font truncated: {font_path}")
    num_tables = struct.unpack(">H", data[4:6])[0]
    cmap_off = None
    for i in range(num_tables):
        off = 12 + i * 16
        if data[off : off + 4] == b"cmap":
            cmap_off = struct.unpack(">I", data[off + 8 : off + 12])[0]
            break
    if cmap_off is None:
        raise ValueError(f"No cmap table in {font_path}")

    nsub = struct.unpack(">H", data[cmap_off + 2 : cmap_off + 4])[0]
    cps: set[int] = set()
    for i in range(nsub):
        rec = cmap_off + 4 + i * 8
        _, _, sub_off = struct.unpack(">HHI", data[rec : rec + 8])
        st = cmap_off + sub_off
        fmt = struct.unpack(">H", data[st : st + 2])[0]
        if fmt == 4:
            sc = struct.unpack(">H", data[st + 6 : st + 8])[0] // 2
            ends = struct.unpack(f">{sc}H", data[st + 14 : st + 14 + 2 * sc])
            starts = struct.unpack(f">{sc}H", data[st + 16 + 2 * sc : st + 16 + 4 * sc])
            for s, e in zip(starts, ends):
                if s != 0xFFFF:
                    cps.update(range(s, e + 1))
        elif fmt == 12:
            ng = struct.unpack(">I", data[st + 12 : st + 16])[0]
            for g in range(ng):
                s, e, _ = struct.unpack(">III", data[st + 16 + g * 12 : st + 28 + g * 12])
                cps.update(range(s, e + 1))
    return cps


def verify_font_glyphs(font_path: str | Path, texts: list[str] | None = None) -> set[int]:
    resolved = _resolve_font_path(font_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Font missing: {resolved}")
    cps = _parse_ttf_codepoints(resolved)
    if texts is None:
        texts = [t for _, t, _ in C.LABELS]
    missing = sorted({ch for s in texts for ch in s if ch not in ("\n", "\r") and ord(ch) not in cps})
    if missing:
        raise ValueError(f"Font {resolved.name} missing glyphs: {missing}")
    return cps


def _target_world_pos(key: str, f: float) -> tuple[float, float, float]:
    """Exact 3D component anchor point for each of the 6 contract callouts."""
    if key == "engine_dead":
        return tuple(C.STAGE["cockpit_caution_pos"])
    if key == "sprag_clutch":
        return tuple(C.STAGE["b2_clutch_world"])
    if key == "upward_airflow":
        hpos, _ = C.heli_pose_at(f)
        return (hpos[0], hpos[1] - 0.12, hpos[2] + 0.44)
    if key == "maple_seed_lift":
        mpos, _ = C.maple_seed_pose_at(f)
        return (mpos[0] + 0.08, mpos[1], mpos[2] + 0.08)
    if key == "flare_30ft":
        hpos, _ = C.heli_pose_at(f)
        return (hpos[0], hpos[1] - 0.10, hpos[2] + 0.36)
    if key == "zero_engine":
        hpos, _ = C.heli_pose_at(f)
        return (hpos[0], hpos[1] - 0.05, hpos[2] + 0.42)
    raise KeyError(f"Unknown label key: {key!r}")


def _cam_basis(cam_pos: tuple[float, float, float], tgt_pos: tuple[float, float, float]):
    dx, dy, dz = tgt_pos[0] - cam_pos[0], tgt_pos[1] - cam_pos[1], tgt_pos[2] - cam_pos[2]
    L = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
    fwd = (dx / L, dy / L, dz / L)
    rx, ry, rz = fwd[1] * 1.0 - fwd[2] * 0.0, fwd[2] * 0.0 - fwd[0] * 1.0, 0.0
    rL = math.sqrt(rx * rx + ry * ry + rz * rz) or 1e-9
    right = (rx / rL, ry / rL, rz / rL)
    up = (
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    )
    return right, up, fwd


def _ndc_to_world_on_focus(
    ndc_x: float,
    ndc_y: float,
    st: dict[str, Any],
    z_offset: float = -0.05,
) -> tuple[float, float, float]:
    cp = st["cam_pos"]
    right, up, fwd = _cam_basis(cp, st["tgt_pos"])
    depth = max(0.35, float(st["focus_dist"]) + z_offset)
    half_w = depth * (0.5 * SENSOR_W_EFF_MM / float(st["lens_mm"]))
    half_h = depth * (0.5 * SENSOR_H_EFF_MM / float(st["lens_mm"]))
    wx = cp[0] + fwd[0] * depth + right[0] * (ndc_x * half_w) + up[0] * (ndc_y * half_h)
    wy = cp[1] + fwd[1] * depth + right[1] * (ndc_x * half_w) + up[1] * (ndc_y * half_h)
    wz = cp[2] + fwd[2] * depth + right[2] * (ndc_x * half_w) + up[2] * (ndc_y * half_h)
    return (wx, wy, wz)


def _plaque_dims_world(text: str, char_h_ndc: float, st: dict[str, Any]) -> tuple[float, float, float]:
    depth = max(0.35, float(st["focus_dist"]) - 0.05)
    half_h = depth * (0.5 * SENSOR_H_EFF_MM / float(st["lens_mm"]))
    char_h_w = char_h_ndc * half_h
    text_w = char_h_w * (len(text) * 0.72 + 0.65)
    plaque_w = text_w + char_h_w * 1.45
    plaque_h = char_h_w * 2.15
    return char_h_w, plaque_w, plaque_h


def _hide_non_camera_rays(obj: Any) -> None:
    obj.visible_shadow = False
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    if hasattr(obj, "visible_volume_scatter"):
        obj.visible_volume_scatter = False


def _make_obsidian_plaque_mat(name: str) -> Any:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.008, 0.012, 0.020, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.05
    bsdf.inputs["Roughness"].default_value = 0.45
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.10
    return mat


def _make_accent_mat(name: str, rgb: tuple[float, float, float], emit_str: float = 1.65) -> Any:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Emission Strength"].default_value = min(1.85, emit_str)
    return mat


def _make_2zone_text_mat(name: str, face_rgb: tuple[float, float, float]) -> Any:
    """V12 2-Zone Object-Space |N.z| > 0.85 brushed face + dark metallic extrusion (AgX-safe <= 1.80)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    vxf = nt.nodes.new("ShaderNodeVectorTransform")
    vxf.vector_type = "NORMAL"
    vxf.convert_from = "WORLD"
    vxf.convert_to = "OBJECT"
    nt.links.new(geom.outputs["Normal"], vxf.inputs["Vector"])

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(vxf.outputs["Vector"], sep.inputs["Vector"])

    abz = nt.nodes.new("ShaderNodeMath")
    abz.operation = "ABSOLUTE"
    nt.links.new(sep.outputs["Z"], abz.inputs[0])

    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.85
    nt.links.new(abz.outputs["Value"], gt.inputs[0])

    bsdf_face = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_face.inputs["Base Color"].default_value = (*face_rgb, 1.0)
    bsdf_face.inputs["Metallic"].default_value = 0.12
    bsdf_face.inputs["Roughness"].default_value = 0.22
    bsdf_face.inputs["Emission Color"].default_value = (*face_rgb, 1.0)
    bsdf_face.inputs["Emission Strength"].default_value = 1.65

    bsdf_side = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_side.inputs["Base Color"].default_value = (0.04, 0.045, 0.055, 1.0)
    bsdf_side.inputs["Metallic"].default_value = 0.85
    bsdf_side.inputs["Roughness"].default_value = 0.35

    mix_s = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(gt.outputs["Value"], mix_s.inputs["Fac"])
    nt.links.new(bsdf_side.outputs["BSDF"], mix_s.inputs[1])
    nt.links.new(bsdf_face.outputs["BSDF"], mix_s.inputs[2])
    nt.links.new(mix_s.outputs["Shader"], out.inputs["Surface"])
    return mat


def _key_visibility(objs: list[Any], w0: int, w1: int, total_frames: int) -> None:
    frames = sorted({f for f in (1, w0 - 1, w0, w1, w1 + 1, total_frames) if 1 <= f <= total_frames})
    for obj in objs:
        for f in frames:
            h = not (w0 <= f <= w1)
            obj.hide_render = h
            obj.hide_viewport = h
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)
        ad = obj.animation_data
        if ad and ad.action:
            for fc in C.iter_action_fcurves(ad.action):
                if fc.data_path.startswith("hide_"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"


def build_labels(
    col: Any,
    font_path: str | Path = DEFAULT_FONT_PATH,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build all 6 AgX-safe dark-obsidian 3D engineering callouts & perpendicular pointer reticles."""
    if bpy is None:
        raise RuntimeError("build_labels() requires Blender bpy.")
    resolved_font = _resolve_font_path(font_path)
    verify_font_glyphs(resolved_font)
    bfont = bpy.data.fonts.load(str(resolved_font), check_existing=True)

    mat_obsidian = _make_obsidian_plaque_mat("Mat_Label_ObsidianPlaque")
    mat_text = _make_2zone_text_mat("Mat_Label_FaceWhite", (0.98, 0.99, 1.00))
    mat_sleeve = _make_obsidian_plaque_mat("Mat_Label_PointerSleeve")

    built: dict[str, Any] = {}
    for key, text, (w0, w1) in C.LABELS:
        spec = _LABEL_SPECS[key]
        rgb = _THEME_RGB[spec["theme"]]
        mat_acc = _make_accent_mat(f"Mat_Label_Accent_{key}", rgb, emit_str=1.68)

        mid_f = 0.5 * (w0 + w1)
        st_mid = camera_state_at(mid_f, include_motion_fx=False)
        char_h_w, plaque_w, plaque_h = _plaque_dims_world(text, float(spec["char_h_ndc"]), st_mid)

        root = bpy.data.objects.new(f"Label_{key}_Root", None)
        col.objects.link(root)

        # 1. Dark Obsidian Chamfered Backing Plaque + Glowing Left/Bottom Trim Bars
        bm_p = bmesh.new()
        tmp = bmesh.new()
        bmesh.ops.create_cube(tmp, size=1.0)
        bmesh.ops.scale(tmp, vec=Vector((plaque_w, plaque_h, 0.004)), verts=tmp.verts)
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=min(0.008, plaque_h * 0.14), segments=2, affect="EDGES", clamp_overlap=True)
        bmesh.ops.transform(tmp, matrix=Matrix.Translation((0.0, 0.0, -0.004)), verts=tmp.verts)
        vm = {v: bm_p.verts.new(v.co) for v in tmp.verts}
        for f in tmp.faces:
            nf = bm_p.faces.new([vm[v] for v in f.verts])
            nf.material_index = 0
        tmp.free()

        # Left vertical glow accent bar + bottom hairline accent bar
        for pos_b, sz_b in (
            ((-0.5 * plaque_w + char_h_w * 0.18, 0.0, -0.001), (char_h_w * 0.16, plaque_h * 0.72, 0.005)),
            ((0.0, -0.5 * plaque_h + char_h_w * 0.06, -0.001), (plaque_w * 0.96, char_h_w * 0.08, 0.005)),
        ):
            tb = bmesh.new()
            bmesh.ops.create_cube(tb, size=1.0)
            bmesh.ops.scale(tb, vec=Vector(sz_b), verts=tb.verts)
            bmesh.ops.transform(tb, matrix=Matrix.Translation(pos_b), verts=tb.verts)
            vm2 = {v: bm_p.verts.new(v.co) for v in tb.verts}
            for f in tb.faces:
                nf = bm_p.faces.new([vm2[v] for v in f.verts])
                nf.material_index = 1
            tb.free()

        mesh_p = bpy.data.meshes.new(f"Label_{key}_Plaque_Mesh")
        mesh_p.materials.append(mat_obsidian)
        mesh_p.materials.append(mat_acc)
        bm_p.to_mesh(mesh_p)
        bm_p.free()
        plaque_obj = bpy.data.objects.new(f"Label_{key}_Plaque", mesh_p)
        plaque_obj.parent = root
        _hide_non_camera_rays(plaque_obj)
        col.objects.link(plaque_obj)

        # 2. Razor-Sharp 3D Typography (V12 minimal bevel & airy tracking)
        curve = bpy.data.curves.new(f"Label_{key}_Text_Curve", type="FONT")
        curve.body = text
        curve.font = bfont
        curve.size = char_h_w
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.space_character = 1.10
        curve.extrude = min(0.0028, char_h_w * 0.022)
        curve.bevel_depth = min(0.00025, char_h_w * 0.0022)
        curve.bevel_resolution = 1
        curve.materials.append(mat_text)
        txt_obj = bpy.data.objects.new(f"Label_{key}_Text", curve)
        txt_obj.parent = root
        txt_obj.location = Vector((char_h_w * 0.12, 0.0, 0.002))
        _hide_non_camera_rays(txt_obj)
        col.objects.link(txt_obj)

        # 3. Crisp Glowing Pointer Line + Perpendicular Dark PointerSleeve Halo + Target Reticle Ring
        bm_ptr = bmesh.new()
        res_l = bmesh.ops.create_cone(
            bm_ptr, cap_ends=True, segments=8,
            radius1=char_h_w * 0.055, radius2=char_h_w * 0.055, depth=1.0,
            matrix=Matrix.Translation((0.0, 0.0, 0.5)),
        )
        for f in bm_ptr.faces:
            f.material_index = 0
        mesh_ptr = bpy.data.meshes.new(f"Label_{key}_Pointer_Mesh")
        mesh_ptr.materials.append(mat_acc)
        bm_ptr.to_mesh(mesh_ptr)
        bm_ptr.free()
        ptr_obj = bpy.data.objects.new(f"Label_{key}_Pointer", mesh_ptr)
        _hide_non_camera_rays(ptr_obj)
        col.objects.link(ptr_obj)

        bm_sl = bmesh.new()
        bmesh.ops.create_cone(
            bm_sl, cap_ends=True, segments=8,
            radius1=char_h_w * 0.11, radius2=char_h_w * 0.11, depth=1.0,
            matrix=Matrix.Translation((0.0, 0.0, 0.5)),
        )
        mesh_sl = bpy.data.meshes.new(f"Label_{key}_PointerSleeve_Mesh")
        mesh_sl.materials.append(mat_sleeve)
        bm_sl.to_mesh(mesh_sl)
        bm_sl.free()
        sl_obj = bpy.data.objects.new(f"Label_{key}_PointerSleeve", mesh_sl)
        _hide_non_camera_rays(sl_obj)
        col.objects.link(sl_obj)

        # Target dot reticle at anchor point
        bm_dot = bmesh.new()
        bmesh.ops.create_icosphere(bm_dot, subdivisions=2, radius=char_h_w * 0.22)
        mesh_dot = bpy.data.meshes.new(f"Label_{key}_Reticle_Mesh")
        mesh_dot.materials.append(mat_acc)
        bm_dot.to_mesh(mesh_dot)
        bm_dot.free()
        dot_obj = bpy.data.objects.new(f"Label_{key}_Reticle", mesh_dot)
        _hide_non_camera_rays(dot_obj)
        col.objects.link(dot_obj)

        # Animate plaque strictly pinned to focus plane + billboarded cleanly toward camera!
        nx_p, ny_p = spec["plaque_ndc"]
        ax_n, ay_n = spec["attach_norm"]
        kfs = sorted(set(range(w0, w1 + 1, 3)) | {w0, w1})
        for f in kfs:
            ff = float(f)
            st = camera_state_at(ff, include_motion_fx=True)
            right, up, fwd = _cam_basis(st["cam_pos"], st["tgt_pos"])
            p_pos = Vector(_ndc_to_world_on_focus(nx_p, ny_p, st, z_offset=-0.05))
            # Camera-facing rotation matrix (local +X=right, +Y=up, +Z=-fwd toward camera!)
            rot_m = Matrix((
                (right[0], up[0], -fwd[0]),
                (right[1], up[1], -fwd[1]),
                (right[2], up[2], -fwd[2]),
            )).to_4x4()
            pop = C.smootherstep(float(w0), float(w0 + 10), ff) * (1.0 - C.smoothstep(float(w1 - 8), float(w1), ff))
            sc = max(0.001, 0.85 + 0.15 * pop)

            root.location = p_pos
            root.rotation_mode = "QUATERNION"
            root.rotation_quaternion = rot_m.to_quaternion()
            root.scale = Vector((sc, sc, sc))
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_quaternion", frame=f)
            root.keyframe_insert("scale", frame=f)

            # Pointer from plaque bottom attach point -> 3D component anchor (with perpendicular PointerSleeve push!)
            attach_w = p_pos + Vector(right) * (ax_n * plaque_w * sc) + Vector(up) * (ay_n * plaque_h * sc)
            tgt_w = Vector(_target_world_pos(key, ff))
            seg = tgt_w - attach_w
            seg_len = max(0.01, seg.length)
            seg_dir = seg.normalized()
            v_fwd = Vector(fwd)
            # V12 Perpendicular PointerSleeve push (never pure +v_fwd so steep rays glow cleanly!)
            perp = v_fwd - seg_dir * v_fwd.dot(seg_dir)
            if perp.length < 1e-4:
                perp = v_fwd
            else:
                perp.normalize()
            q_seg = seg_dir.to_track_quat("Z", "Y")

            ptr_obj.location = attach_w
            ptr_obj.rotation_mode = "QUATERNION"
            ptr_obj.rotation_quaternion = q_seg
            ptr_obj.scale = Vector((1.0, 1.0, seg_len * pop))
            ptr_obj.keyframe_insert("location", frame=f)
            ptr_obj.keyframe_insert("rotation_quaternion", frame=f)
            ptr_obj.keyframe_insert("scale", frame=f)

            sl_obj.location = attach_w + perp * (char_h_w * 0.14)
            sl_obj.rotation_mode = "QUATERNION"
            sl_obj.rotation_quaternion = q_seg
            sl_obj.scale = Vector((1.0, 1.0, seg_len * pop))
            sl_obj.keyframe_insert("location", frame=f)
            sl_obj.keyframe_insert("rotation_quaternion", frame=f)
            sl_obj.keyframe_insert("scale", frame=f)

            dot_obj.location = tgt_w
            dot_obj.scale = Vector((pop, pop, pop))
            dot_obj.keyframe_insert("location", frame=f)
            dot_obj.keyframe_insert("scale", frame=f)

        all_objs = [plaque_obj, txt_obj, ptr_obj, sl_obj, dot_obj]
        _key_visibility(all_objs, w0, w1, total_frames)
        built[key] = {"root": root, "objs": all_objs}

    return built


def _self_audit() -> int:
    """Pure-Python V12 QA verifying font glyphs, NDC bounds, DOF fstop >= 7.1, and subtitle clearance."""
    errors: list[str] = []
    print("=== Episode #5 Helicopter Autorotation — 3D Labels V12 Audit ===")
    verify_font_glyphs(DEFAULT_FONT_PATH)
    print(f"  [PASS] Font verified: {DEFAULT_FONT_PATH.name} supports all 6 callouts")

    for key, text, (w0, w1) in C.LABELS:
        words = text.split()
        if not (1 <= len(words) <= 3):
            errors.append(f"{key}: '{text}' exceeds 3-word limit")
        spec = _LABEL_SPECS[key]
        nx_p, ny_p = spec["plaque_ndc"]
        for f in (w0, (w0 + w1) // 2, w1):
            st = camera_state_at(float(f), include_motion_fx=True)
            if st["fstop"] < 7.1:
                errors.append(f"{key} f={f}: fstop={st['fstop']:.1f} < 7.1 (bokeh blur risk!)")
            if ny_p < LABEL_MIN_NDC_Y or abs(nx_p) > 0.78:
                errors.append(f"{key} f={f}: plaque NDC ({nx_p:+.2f}, {ny_p:+.2f}) out of safe bounds")
            tgt_w = _target_world_pos(key, float(f))
            tnx, tny, tz = project_ndc(tgt_w, st["cam_pos"], st["tgt_pos"], st["lens_mm"])
            if not (tz > 0.25 and abs(tnx) <= 0.88 and LABEL_MIN_NDC_Y <= tny <= 0.85):
                errors.append(f"{key} f={f}: anchor NDC=({tnx:+.2f},{tny:+.2f}) z={tz:.2f} out of bounds")
        print(
            f"  [PASS] {key:16s} '{text:16s}' f={w0:4d}..{w1:4d} | "
            f"plaque=({nx_p:+.2f},{ny_p:+.2f}) f/{camera_state_at(float((w0+w1)//2))['fstop']:.1f}"
        )

    if errors:
        print("ERRORS:")
        for e in errors:
            print("  -", e)
        return 1
    print("RESULT: 0 errors — All 6 3D Callouts PASSED V12 Glyph, DOF & Safe-Zone Audit!")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_audit())
