"""
spine_xray.py — Anatomical 3D Spine X-Ray & 16-G Compression Rig for Episode #1
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (60 fps, 1800 frames).

Owned by the Spine X-Ray Rig Artist. Imports all timings, coordinates, X-ray windows,
and physical readouts from `ejection_contract.py` (never duplicates constants).

Public API:
    build_spine_xray(
        col: bpy.types.Collection,
        carrier: bpy.types.Object,
        sacrum_local: tuple[float, float, float],
        c1_local: tuple[float, float, float],
        total_frames: int = 1800,
    ) -> dict[str, Any]

Returned dict exact types:
    - "root":      bpy.types.Object (`Spine_Root` Empty parented to `carrier`)
    - "sacrum":    bpy.types.Object (`Spine_Sacrum` sculpted pelvis/sacrum wedge Mesh)
    - "vertebrae": list[bpy.types.Object] (24 Mesh objects `Spine_Vert_L5` .. `Spine_Vert_C1`)
    - "discs":     list[bpy.types.Object] (23 Mesh objects `Spine_Disc_01` .. `Spine_Disc_23`)
    - "force_fx":  list[bpy.types.Object] (glowing 16-G downward chevrons & shock rings `Spine_ForceArrow_*`)
    - "all_objs":  list[bpy.types.Object] (all created `Spine_*` objects)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(EPISODE_DIR)), "src")
for _p in (SCRIPT_DIR, REPO_SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ejection_contract as C  # noqa: E402

PREFIX = "Spine_"

# Visibility windows (frames) aligned with C.XRAY_BODY (895..1236) and compression (930..1180)
SPINE_VIS_RANGE: tuple[int, int] = (885, C.XRAY_BODY[3])  # 885..1236 inclusive
FORCE_FX_VIS_RANGE: tuple[int, int] = (C.EV["spine_compress_start"], 1180)  # 930..1180 inclusive

# Disc compression behavior from contract specification
DISC_TUCK_M = 0.0026          # internal endplate embed per side so discs never leave air gaps
DISC_SCALE_Z_MAX = 0.52       # vertical squash at frac=1.0
DISC_SCALE_XY_MAX = 1.28      # radial bulge at frac=1.0
DISC_RADIUS_RATIO = 0.94      # rest disc radius relative to adjacent vertebral body

# 24 anatomical vertebrae specs: (label, region, rx, ry, body_h, gap_below, wing_half_x, spinous_len, spinous_drop)
# `gap_below` for L5 is the Sacrum-L5 (L5-S1) disc gap; for every subsequent level it is the gap below that bone.
VERTEBRA_SPECS: list[tuple[str, str, float, float, float, float, float, float, float]] = [
    # Lumbar (L5..L1): heavy kidney bodies, sturdy horizontal lateral wings & hatchet spinous blades
    ("L5", "L", 0.0255, 0.0185, 0.0255, 0.0068, 0.043, 0.026, -0.004),
    ("L4", "L", 0.0248, 0.0180, 0.0250, 0.0067, 0.042, 0.027, -0.004),
    ("L3", "L", 0.0242, 0.0176, 0.0245, 0.0065, 0.041, 0.027, -0.005),
    ("L2", "L", 0.0235, 0.0172, 0.0240, 0.0062, 0.040, 0.026, -0.005),
    ("L1", "L", 0.0228, 0.0168, 0.0235, 0.0060, 0.038, 0.025, -0.006),
    # Thoracic (T12..T1): progressively tapering bodies, rib-facet wings, steep shingle-like spinous blades
    ("T12", "T", 0.0218, 0.0164, 0.0225, 0.0056, 0.035, 0.026, -0.011),
    ("T11", "T", 0.0210, 0.0160, 0.0218, 0.0054, 0.034, 0.028, -0.015),
    ("T10", "T", 0.0202, 0.0156, 0.0212, 0.0052, 0.033, 0.030, -0.018),
    ("T9",  "T", 0.0195, 0.0153, 0.0206, 0.0051, 0.032, 0.032, -0.021),
    ("T8",  "T", 0.0188, 0.0150, 0.0200, 0.0050, 0.031, 0.033, -0.023),
    ("T7",  "T", 0.0182, 0.0147, 0.0195, 0.0049, 0.031, 0.034, -0.024),
    ("T6",  "T", 0.0177, 0.0144, 0.0190, 0.0048, 0.030, 0.034, -0.024),
    ("T5",  "T", 0.0172, 0.0141, 0.0186, 0.0047, 0.029, 0.033, -0.022),
    ("T4",  "T", 0.0168, 0.0138, 0.0182, 0.0046, 0.029, 0.031, -0.020),
    ("T3",  "T", 0.0164, 0.0136, 0.0178, 0.0045, 0.028, 0.029, -0.018),
    ("T2",  "T", 0.0160, 0.0133, 0.0174, 0.0044, 0.027, 0.028, -0.015),
    ("T1",  "T", 0.0156, 0.0130, 0.0170, 0.0043, 0.027, 0.027, -0.012),
    # Cervical (C7..C1): compact bodies, C7 prominens, C2 dens peg, C1 atlas lateral-mass ring
    ("C7", "C", 0.0148, 0.0122, 0.0154, 0.0041, 0.025, 0.026, -0.008),
    ("C6", "C", 0.0142, 0.0118, 0.0148, 0.0039, 0.024, 0.021, -0.006),
    ("C5", "C", 0.0138, 0.0115, 0.0142, 0.0037, 0.023, 0.019, -0.005),
    ("C4", "C", 0.0134, 0.0112, 0.0136, 0.0036, 0.022, 0.018, -0.004),
    ("C3", "C", 0.0130, 0.0109, 0.0132, 0.0035, 0.022, 0.017, -0.004),
    ("C2", "C", 0.0128, 0.0108, 0.0128, 0.0034, 0.022, 0.018, -0.003),
    ("C1", "C", 0.0175, 0.0135, 0.0110, 0.0036, 0.025, 0.011,  0.000),
]


# ---------------------------------------------------------------------------
# 1. ANATOMICAL S-CURVE & COORDINATE MATH (Pure-Python deterministic layout)
# ---------------------------------------------------------------------------
def _s_curve_amp(u: float) -> float:
    """Sagittal S-curve offset (m) along anterior direction: lumbar +, thoracic -, cervical +."""
    return (
        -0.004 * math.sin(math.pi * u)
        + 0.006 * math.sin(2.0 * math.pi * u)
        + 0.021 * math.sin(3.0 * math.pi * u)
    )


def _s_curve_d_amp_du(u: float) -> float:
    """Derivative d(amp)/du for exact local vertebral endplate tangent alignment."""
    return (
        -0.004 * math.pi * math.cos(math.pi * u)
        + 0.006 * 2.0 * math.pi * math.cos(2.0 * math.pi * u)
        + 0.021 * 3.0 * math.pi * math.cos(3.0 * math.pi * u)
    )


def _spine_basis(
    sacrum_local: tuple[float, float, float],
    c1_local: tuple[float, float, float],
) -> tuple[Vector, Vector, float, Vector, Vector, Vector]:
    """Return (p_sac, p_c1, chord_len, spine_axis, anterior_dir, right_dir)."""
    p_sac = Vector(sacrum_local)
    p_c1 = Vector(c1_local)
    chord = p_c1 - p_sac
    chord_len = chord.length
    if chord_len < 1e-5:
        raise ValueError("sacrum_local and c1_local must be distinct points")
    axis = chord / chord_len

    ref_right = Vector((1.0, 0.0, 0.0))
    if abs(axis.dot(ref_right)) > 0.99:
        ref_right = Vector((0.0, 1.0, 0.0))
    anterior_dir = axis.cross(ref_right).normalized()
    right_dir = anterior_dir.cross(axis).normalized()
    return p_sac, p_c1, chord_len, axis, anterior_dir, right_dir


def _rest_point_and_euler(
    u: float,
    p_sac: Vector,
    chord_len: float,
    axis: Vector,
    anterior_dir: Vector,
    right_dir: Vector,
) -> tuple[Vector, Euler]:
    """Rest position along anatomical S-curve and Euler aligning local +Z to tangent, +X to right."""
    pos = p_sac + axis * (u * chord_len) + anterior_dir * _s_curve_amp(u)
    tangent = (axis * chord_len + anterior_dir * _s_curve_d_amp_du(u)).normalized()
    y_dir = tangent.cross(right_dir).normalized()
    rot_mat = Matrix((right_dir, y_dir, tangent)).transposed()
    return pos, rot_mat.to_euler("XYZ")


def _compute_normalized_layout(chord_len: float) -> list[dict[str, Any]]:
    """Scale vertebral heights & disc gaps uniformly so Sacrum(u=0)..C1_center(u=1) equals chord_len."""
    raw_steps: list[float] = []
    for idx, spec in enumerate(VERTEBRA_SPECS):
        _, _, _, _, body_h, gap_below, _, _, _ = spec
        if idx == 0:
            raw_steps.append(gap_below + 0.5 * body_h)
        else:
            prev_h = VERTEBRA_SPECS[idx - 1][4]
            raw_steps.append(0.5 * prev_h + gap_below + 0.5 * body_h)

    scale_f = chord_len / sum(raw_steps)
    layout: list[dict[str, Any]] = []
    s_center = 0.0
    for idx, spec in enumerate(VERTEBRA_SPECS):
        label, region, rx, ry, body_h, gap_below, wing_x, sp_len, sp_drop = spec
        s_center += raw_steps[idx] * scale_f
        body_h_s = body_h * scale_f
        gap_s = gap_below * scale_f
        if idx == 0:
            s_lower_endplate = 0.0
        else:
            prev_h_s = VERTEBRA_SPECS[idx - 1][4] * scale_f
            s_lower_endplate = layout[idx - 1]["s_center"] + 0.5 * prev_h_s
        s_upper_endplate = s_center - 0.5 * body_h_s
        layout.append(
            {
                "idx": float(idx + 1),
                "label": label,
                "region": region,
                "rx": rx * scale_f,
                "ry": ry * scale_f,
                "body_h": body_h_s,
                "gap_below": gap_s,
                "wing_x": wing_x * scale_f,
                "sp_len": sp_len * scale_f,
                "sp_drop": sp_drop * scale_f,
                "s_center": s_center,
                "u_center": s_center / chord_len,
                "u_lower_ep": s_lower_endplate / chord_len,
                "u_upper_ep": s_upper_endplate / chord_len,
            }
        )
    return layout


# ---------------------------------------------------------------------------
# 2. BLENDER OBJECT / KEYFRAME / BMESH HELPERS
# ---------------------------------------------------------------------------
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    if obj.name not in col.objects:
        col.objects.link(obj)


def _set_smooth(mesh: bpy.types.Mesh, smooth: bool = True) -> None:
    if hasattr(mesh, "polygons") and len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    mesh.update()


def _create_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    materials: list[bpy.types.Material] | bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] | Euler = (0.0, 0.0, 0.0),
    visible_shadow: bool = False,
    recalc_normals: bool = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mat_list = materials if isinstance(materials, list) else [materials]
    for m in mat_list:
        mesh.materials.append(m)

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    _set_smooth(mesh, True)

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = rotation.copy() if isinstance(rotation, Euler) else Euler(rotation, "XYZ")
    obj.visible_shadow = visible_shadow
    if parent is not None:
        obj.parent = parent
        obj.matrix_parent_inverse = Matrix.Identity(4)
    _link_obj(obj, col)
    return obj


def _set_fcurve_interp(id_block: Any, interp: str, data_path_prefix: str = "") -> None:
    ad = getattr(id_block, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if data_path_prefix and not fc.data_path.startswith(data_path_prefix):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interp


def _key_visibility(
    obj: bpy.types.Object,
    visible_ranges: list[tuple[int, int]],
    total_frames: int,
) -> None:
    """Keyframe hide_viewport & hide_render strictly CONSTANT across [1..total_frames]."""
    key_frames = {1, total_frames}
    for s, e in visible_ranges:
        s_c = max(1, min(total_frames, int(s)))
        e_c = max(1, min(total_frames, int(e)))
        if s_c > 1:
            key_frames.add(s_c - 1)
        key_frames.add(s_c)
        key_frames.add(e_c)
        if e_c < total_frames:
            key_frames.add(e_c + 1)

    for f in sorted(key_frames):
        is_vis = any(s <= f <= e for s, e in visible_ranges)
        obj.hide_viewport = not is_vis
        obj.hide_render = not is_vis
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    _set_fcurve_interp(obj, "CONSTANT", "hide_")


def _kidney_xy(angle: float, rx: float, ry: float, notch: float = 0.26) -> tuple[float, float]:
    """C1-continuous anatomical kidney-shaped cross section (+Y anterior, -Y posterior canal wall)."""
    cx = math.cos(angle)
    sy = math.sin(angle)
    back = max(0.0, -sy)
    indent = notch * (back ** 2.2)
    x = rx * cx * (1.0 - 0.06 * (back ** 2))
    y = ry * sy * (1.0 - indent)
    return x, y


def _add_lofted_rings(
    bm: bmesh.types.BMesh,
    rings_pts: list[list[Vector]],
    cap_bottom: bool = True,
    cap_top: bool = True,
    end_inset: float = 0.72,
) -> None:
    """Loft a sequence of equal-length point rings into closed smooth-shaded geometry."""
    n = len(rings_pts[0])
    bm_rings = [[bm.verts.new(p) for p in ring] for ring in rings_pts]

    for r0, r1 in zip(bm_rings[:-1], bm_rings[1:]):
        for k in range(n):
            kn = (k + 1) % n
            bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))

    def _cap_end(ring: list[bmesh.types.BMVert], reverse: bool) -> None:
        center_co = sum((v.co for v in ring), Vector((0.0, 0.0, 0.0))) / float(n)
        inner = [bm.verts.new(center_co.lerp(v.co, end_inset)) for v in ring]
        pole = bm.verts.new(center_co)
        for k in range(n):
            kn = (k + 1) % n
            if reverse:
                bm.faces.new((ring[kn], ring[k], inner[k], inner[kn]))
                bm.faces.new((inner[kn], inner[k], pole))
            else:
                bm.faces.new((ring[k], ring[kn], inner[kn], inner[k]))
                bm.faces.new((inner[k], inner[kn], pole))

    if cap_bottom:
        _cap_end(bm_rings[0], reverse=True)
    if cap_top:
        _cap_end(bm_rings[-1], reverse=False)


def _add_tube_along_frames(
    bm: bmesh.types.BMesh,
    stations: list[tuple[Vector, float, float]],
    n_seg: int = 10,
    cap_ends: bool = True,
) -> None:
    """Loft an elliptical cross-section tube along 3D stations using parallel-transport frames."""
    rings: list[list[Vector]] = []
    prev_tan: Vector | None = None
    binorm = Vector((1.0, 0.0, 0.0))
    norm = Vector((0.0, 1.0, 0.0))

    for idx, (pos, ra, rb) in enumerate(stations):
        if idx == 0:
            tan = (stations[1][0] - pos).normalized()
        elif idx == len(stations) - 1:
            tan = (pos - stations[-2][0]).normalized()
        else:
            tan = (stations[idx + 1][0] - stations[idx - 1][0]).normalized()
        if tan.length < 1e-6:
            tan = prev_tan or Vector((0.0, 0.0, 1.0))

        if prev_tan is None:
            up_ref = Vector((0.0, 0.0, 1.0))
            if abs(tan.dot(up_ref)) > 0.94:
                up_ref = Vector((0.0, 1.0, 0.0))
            binorm = tan.cross(up_ref).normalized()
            norm = binorm.cross(tan).normalized()
        else:
            rot_q = prev_tan.rotation_difference(tan)
            binorm = (rot_q @ binorm).normalized()
            norm = binorm.cross(tan).normalized()
        prev_tan = tan

        ring: list[Vector] = []
        for k in range(n_seg):
            ang = 2.0 * math.pi * k / float(n_seg)
            pt = pos + binorm * (ra * math.cos(ang)) + norm * (rb * math.sin(ang))
            ring.append(pt)
        rings.append(ring)

    _add_lofted_rings(bm, rings, cap_bottom=cap_ends, cap_top=cap_ends, end_inset=0.55)


# ---------------------------------------------------------------------------
# 3. PROCEDURAL ANATOMICAL MESH BUILDERS
# ---------------------------------------------------------------------------
def _build_vertebra_bm(spec: dict[str, Any]) -> bmesh.types.BMesh:
    """Sculpt a complete vertebra: waisted kidney body (or C1 atlas ring), neural arch, wings, spinous process."""
    bm = bmesh.new()
    label = str(spec["label"])
    rx = spec["rx"]
    ry = spec["ry"]
    h = spec["body_h"]
    wing_x = spec["wing_x"]
    sp_len = spec["sp_len"]
    sp_drop = spec["sp_drop"]

    if label == "C1":
        # Atlas (C1): wide lateral-mass elliptical bone ring enclosing odontoid dens, no tall solid body
        n_ring = 28
        n_prof = 10
        r_mid_x = rx * 0.86
        r_mid_y = ry * 0.86
        tube_rings: list[list[Vector]] = []
        for i in range(n_ring):
            ang = 2.0 * math.pi * i / float(n_ring)
            ca, sa = math.cos(ang), math.sin(ang)
            center = Vector((r_mid_x * ca, r_mid_y * sa, 0.0))
            rad_dir = Vector((ca, sa, 0.0)).normalized()
            # Thicker lateral masses at ±X, slender anterior/posterior arches
            lat_w = abs(ca) ** 1.4
            thick_r = 0.0030 + 0.0040 * lat_w
            thick_z = 0.5 * h * (0.68 + 0.36 * lat_w)
            prof_ring: list[Vector] = []
            for j in range(n_prof):
                pang = 2.0 * math.pi * j / float(n_prof)
                prof_ring.append(center + rad_dir * (thick_r * math.cos(pang)) + Vector((0.0, 0.0, thick_z * math.sin(pang))))
            tube_rings.append(prof_ring)
        # Stitch closed torus ring (wrap last -> first)
        bm_rings = [[bm.verts.new(p) for p in r] for r in tube_rings]
        for i in range(n_ring):
            r0 = bm_rings[i]
            r1 = bm_rings[(i + 1) % n_ring]
            for j in range(n_prof):
                jn = (j + 1) % n_prof
                bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
        # Small anterior tubercle (+Y) and posterior tubercle (-Y)
        _add_tube_along_frames(
            bm,
            [
                (Vector((0.0, r_mid_y, 0.0)), 0.0038, 0.0034),
                (Vector((0.0, r_mid_y + 0.0045, 0.0)), 0.0028, 0.0026),
                (Vector((0.0, r_mid_y + 0.0062, 0.0)), 0.0014, 0.0014),
            ],
            n_seg=8,
        )
        _add_tube_along_frames(
            bm,
            [
                (Vector((0.0, -r_mid_y, 0.0)), 0.0034, 0.0032),
                (Vector((0.0, -r_mid_y - sp_len * 0.65, 0.0)), 0.0026, 0.0024),
                (Vector((0.0, -r_mid_y - sp_len, sp_drop)), 0.0014, 0.0014),
            ],
            n_seg=8,
        )
    else:
        # 1. Waisted kidney-shaped vertebral body with flared beveled epiphyseal rims
        n_rad = 24
        z_levels = [
            (-0.50 * h, 0.90),
            (-0.42 * h, 1.03),
            (0.0,       0.89),
            (0.42 * h,  1.03),
            (0.50 * h,  0.90),
        ]
        body_rings: list[list[Vector]] = []
        for z_pos, rad_scale in z_levels:
            ring: list[Vector] = []
            for k in range(n_rad):
                ang = 2.0 * math.pi * k / float(n_rad)
                kx, ky = _kidney_xy(ang, rx * rad_scale, ry * rad_scale)
                ring.append(Vector((kx, ky, z_pos)))
            body_rings.append(ring)
        _add_lofted_rings(bm, body_rings, cap_bottom=True, cap_top=True, end_inset=0.72)

        # 2. Neural Arch (Pedicles -> Laminae -> Posterior Arch enclosing spinal foramen)
        ped_w = max(0.0032, 0.24 * rx)
        ped_h = max(0.0035, 0.34 * h)
        arch_stations = [
            (Vector((+0.64 * rx, -0.40 * ry, +0.06 * h)), ped_w * 1.15, ped_h * 1.10),
            (Vector((+0.68 * rx, -0.95 * ry, +0.08 * h)), ped_w,        ped_h),
            (Vector((+0.52 * rx, -1.48 * ry, +0.02 * h)), ped_w * 0.95, ped_h * 1.15),
            (Vector((0.0,        -1.76 * ry, -0.04 * h)), ped_w * 1.05, ped_h * 1.22),
            (Vector((-0.52 * rx, -1.48 * ry, +0.02 * h)), ped_w * 0.95, ped_h * 1.15),
            (Vector((-0.68 * rx, -0.95 * ry, +0.08 * h)), ped_w,        ped_h),
            (Vector((-0.64 * rx, -0.40 * ry, +0.06 * h)), ped_w * 1.15, ped_h * 1.10),
        ]
        _add_tube_along_frames(bm, arch_stations, n_seg=10, cap_ends=True)

        # 3. Posterior Spinous Process (-Y, sloping downward in thoracic region)
        sp_root = Vector((0.0, -1.62 * ry, -0.02 * h))
        sp_mid = Vector((0.0, -1.62 * ry - 0.58 * sp_len, 0.50 * sp_drop))
        sp_tip = Vector((0.0, -1.62 * ry - sp_len, sp_drop))
        sp_h = max(0.0036, 0.36 * h)
        sp_w = max(0.0024, 0.14 * rx)
        _add_tube_along_frames(
            bm,
            [
                (sp_root, sp_w * 1.35, sp_h * 1.15),
                (sp_mid,  sp_w * 1.05, sp_h * 0.95),
                (sp_tip,  sp_w * 0.68, sp_h * 0.58),
            ],
            n_seg=8,
            cap_ends=True,
        )

        # C2 Axis odontoid dens peg rising inside C1's hollow ring
        if label == "C2":
            _add_tube_along_frames(
                bm,
                [
                    (Vector((0.0, +0.10 * ry, +0.35 * h)), 0.0044, 0.0042),
                    (Vector((0.0, +0.12 * ry, +0.85 * h)), 0.0040, 0.0038),
                    (Vector((0.0, +0.10 * ry, +1.18 * h)), 0.0024, 0.0024),
                ],
                n_seg=10,
                cap_ends=True,
            )

    # 4. Left & Right Transverse Wing Processes (±X) for ALL 24 vertebrae
    w_rad_z = max(0.0028, 0.25 * h)
    w_rad_y = max(0.0026, 0.22 * ry)
    root_x = 0.72 * rx if label == "C1" else 0.58 * rx
    root_y = 0.0 if label == "C1" else -1.02 * ry
    tip_y = -0.18 * ry if label == "C1" else -1.30 * ry
    for sx in (-1.0, 1.0):
        _add_tube_along_frames(
            bm,
            [
                (Vector((sx * root_x,             root_y,             0.04 * h)), w_rad_y * 1.20, w_rad_z * 1.15),
                (Vector((sx * (0.52 * root_x + 0.48 * wing_x), 0.5 * (root_y + tip_y), 0.03 * h)), w_rad_y * 0.95, w_rad_z * 0.92),
                (Vector((sx * wing_x,             tip_y,              0.01 * h)), w_rad_y * 0.55, w_rad_z * 0.58),
            ],
            n_seg=8,
            cap_ends=True,
        )

    return bm


def _build_sacrum_bm(l5_spec: dict[str, Any]) -> bmesh.types.BMesh:
    """Sculpted anatomical Sacrum + Pelvis iliac wing wedge extending down from S1 plateau (z=0)."""
    bm = bmesh.new()
    rx0 = l5_spec["rx"] * 1.04
    ry0 = l5_spec["ry"] * 1.04

    # 1. Curved triangular sacral shield tapering down to coccyx (stops at z=-0.080m, well above SRP z=0)
    n_rad = 24
    sac_levels = [
        (+0.001,  0.000,  rx0,         ry0),
        (-0.014, -0.008,  rx0 * 2.10,  ry0 * 1.08),
        (-0.034, -0.019,  rx0 * 1.72,  ry0 * 0.92),
        (-0.056, -0.024,  rx0 * 0.96,  ry0 * 0.68),
        (-0.078, -0.019,  rx0 * 0.34,  ry0 * 0.36),
    ]
    rings: list[list[Vector]] = []
    for z_pos, y_shift, rx_s, ry_s in sac_levels:
        ring: list[Vector] = []
        for k in range(n_rad):
            ang = 2.0 * math.pi * k / float(n_rad)
            kx, ky = _kidney_xy(ang, rx_s, ry_s, notch=0.14)
            ring.append(Vector((kx, ky + y_shift, z_pos)))
        rings.append(ring)
    _add_lofted_rings(bm, rings, cap_bottom=True, cap_top=True, end_inset=0.70)

    # 2. Posterior median sacral crest ridge
    _add_tube_along_frames(
        bm,
        [
            (Vector((0.0, -0.022, -0.004)), 0.0045, 0.0055),
            (Vector((0.0, -0.036, -0.028)), 0.0040, 0.0050),
            (Vector((0.0, -0.036, -0.054)), 0.0028, 0.0036),
        ],
        n_seg=8,
        cap_ends=True,
    )

    # 3. Left & Right Iliac Pelvis Wedge Wings (compact sacroiliac crests framing the sacrum)
    for sx in (-1.0, 1.0):
        _add_tube_along_frames(
            bm,
            [
                (Vector((sx * 0.044, -0.010, -0.016)), 0.012, 0.009),
                (Vector((sx * 0.072, +0.002, +0.002)), 0.014, 0.009),
                (Vector((sx * 0.092, +0.016, +0.008)), 0.012, 0.007),
                (Vector((sx * 0.078, +0.004, -0.036)), 0.011, 0.007),
                (Vector((sx * 0.056, -0.012, -0.062)), 0.008, 0.006),
            ],
            n_seg=10,
            cap_ends=True,
        )

    return bm


def _build_disc_bm(rx: float, ry: float, total_h: float) -> bmesh.types.BMesh:
    """Sculpted rounded annulus-fibrosus intervertebral disc matching vertebral kidney profile."""
    bm = bmesh.new()
    n_rad = 24
    rings: list[list[Vector]] = []
    # Smooth convex outer cushion wall from -total_h/2 to +total_h/2
    prof = [
        (-0.50 * total_h, 0.80),
        (-0.34 * total_h, 0.98),
        (0.0,             1.04),
        (+0.34 * total_h, 0.98),
        (+0.50 * total_h, 0.80),
    ]
    for z_pos, r_mult in prof:
        ring: list[Vector] = []
        for k in range(n_rad):
            ang = 2.0 * math.pi * k / float(n_rad)
            kx, ky = _kidney_xy(ang, rx * r_mult, ry * r_mult, notch=0.22)
            ring.append(Vector((kx, ky, z_pos)))
        rings.append(ring)
    _add_lofted_rings(bm, rings, cap_bottom=True, cap_top=True, end_inset=0.70)
    return bm


def _build_chevron_stack_bm(count: int = 3, spacing: float = 0.072) -> bmesh.types.BMesh:
    """Build 3 stacked crisp 3D beveled V-chevrons pointing downward (-Z) for 16-G load arrows."""
    bm = bmesh.new()
    span_x = 0.019
    tip_z = -0.015
    wing_z = 0.007
    thick_y = 0.0045
    stroke_z = 0.0065

    z_offsets = [(count - 1) * 0.5 * spacing - i * spacing for i in range(count)]
    for z_off in z_offsets:
        # 6 outer profile stations across V-shape (left wing -> center apex -> right wing), top & bottom ridges
        xs = [-span_x, -0.5 * span_x, 0.0, 0.5 * span_x, span_x]
        zs = [wing_z, 0.5 * (wing_z + tip_z), tip_z, 0.5 * (wing_z + tip_z), wing_z]
        rings: list[list[Vector]] = []
        for x, z in zip(xs, zs):
            zc = z + z_off
            # Diamond/faceted 4-point cross-section so normals face forward, back, top, bottom cleanly
            rings.append(
                [
                    Vector((x,  0.0,      zc + stroke_z * 0.65)),
                    Vector((x,  thick_y,  zc)),
                    Vector((x,  0.0,      zc - stroke_z * 0.65)),
                    Vector((x, -thick_y,  zc)),
                ]
            )
        _add_lofted_rings(bm, rings, cap_bottom=True, cap_top=True, end_inset=0.4)
    return bm


def _build_open_shock_ring_bm(radius: float = 0.052, height: float = 0.016, flare: float = 0.006) -> bmesh.types.BMesh:
    """Open-ended conical energy shock ring (cap_ends=False per AGENTS.md Section 3)."""
    bm = bmesh.new()
    n_seg = 44
    levels = [
        (-0.5 * height, radius + flare),
        (0.0,           radius),
        (+0.5 * height, radius - flare * 0.5),
    ]
    rings: list[list[Vector]] = []
    for z_pos, r in levels:
        ring: list[Vector] = []
        for k in range(n_seg):
            ang = 2.0 * math.pi * k / float(n_seg)
            ring.append(Vector((r * math.cos(ang), r * math.sin(ang), z_pos)))
        rings.append(ring)
    _add_lofted_rings(bm, rings, cap_bottom=False, cap_top=False)
    return bm


# ---------------------------------------------------------------------------
# 4. CYCLES PROCEDURAL X-RAY SHADERS (Local-Space, AgX-Safe, Shadow-Free)
# ---------------------------------------------------------------------------
def _make_bone_material(
    root_empty: bpy.types.Object,
    name: str,
    is_sacrum: bool = False,
) -> bpy.types.Material:
    """Luminous ivory-bone Cycles shader with organic trabecular grain & cyan Fresnel rim."""
    mat = bpy.data.materials.new(f"{PREFIX}{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    tex_c = nt.nodes.new("ShaderNodeTexCoord")
    tex_c.object = root_empty

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 95.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    nt.links.new(tex_c.outputs["Object"], noise.inputs["Vector"])

    # Subtle warm ivory bone color variation
    col_mix = nt.nodes.new("ShaderNodeMix")
    col_mix.data_type = "RGBA"
    if is_sacrum:
        col_mix.inputs[6].default_value = (0.80, 0.76, 0.67, 1.0)
        col_mix.inputs[7].default_value = (0.91, 0.87, 0.78, 1.0)
    else:
        col_mix.inputs[6].default_value = (0.86, 0.82, 0.74, 1.0)
        col_mix.inputs[7].default_value = (0.96, 0.93, 0.85, 1.0)
    nt.links.new(noise.outputs["Fac"], col_mix.inputs["Factor"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    bsdf.inputs["Roughness"].default_value = 0.36
    bsdf.inputs["Specular IOR Level"].default_value = 0.55
    bsdf.inputs["Subsurface Weight"].default_value = 0.10
    nt.links.new(col_mix.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Subtle cyan Fresnel edge rim + gentle warm bone floor glow (stay crisp inside X-ray body!)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.28
    rim_pow = nt.nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.inputs[1].default_value = 2.8
    nt.links.new(lw.outputs["Facing"], rim_pow.inputs[0])

    em_col = nt.nodes.new("ShaderNodeMix")
    em_col.data_type = "RGBA"
    em_col.inputs[6].default_value = (0.92, 0.88, 0.78, 1.0)  # warm ivory core luminance
    em_col.inputs[7].default_value = (0.14, 0.88, 1.00, 1.0)  # subtle cyan X-ray rim
    nt.links.new(rim_pow.outputs["Value"], em_col.inputs["Factor"])

    em_mul = nt.nodes.new("ShaderNodeMath")
    em_mul.operation = "MULTIPLY"
    em_mul.inputs[1].default_value = 1.25
    nt.links.new(rim_pow.outputs["Value"], em_mul.inputs[0])

    em_add = nt.nodes.new("ShaderNodeMath")
    em_add.operation = "ADD"
    em_add.inputs[1].default_value = 0.36 if is_sacrum else 0.44
    nt.links.new(em_mul.outputs["Value"], em_add.inputs[0])

    nt.links.new(em_col.outputs[2], bsdf.inputs["Emission Color"])
    nt.links.new(em_add.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_disc_material(
    root_empty: bpy.types.Object,
    name: str = "Mat_Disc_Stress",
) -> tuple[bpy.types.Material, bpy.types.ShaderNode]:
    """Animated disc shader: cyan (0) -> amber (0.5) -> crimson (1.0), emission 1.6 -> 2.6."""
    mat = bpy.data.materials.new(f"{PREFIX}{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    frac_node = nt.nodes.new("ShaderNodeValue")
    frac_node.name = "CompressFrac"
    frac_node.outputs[0].default_value = 0.0

    # 3-stop stress ramp: cyan -> amber -> crimson-red
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    el0 = ramp.color_ramp.elements[0]
    el2 = ramp.color_ramp.elements[1]
    el1 = ramp.color_ramp.elements.new(0.5)
    el0.position = 0.0
    el0.color = (0.08, 0.85, 1.00, 1.0)
    el1.position = 0.5
    el1.color = (1.00, 0.52, 0.04, 1.0)
    el2.position = 1.0
    el2.color = (1.00, 0.12, 0.03, 1.0)
    nt.links.new(frac_node.outputs[0], ramp.inputs["Fac"])

    # Local-space subtle fibrous annulus texture
    tex_c = nt.nodes.new("ShaderNodeTexCoord")
    tex_c.object = root_empty
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 70.0
    noise.inputs["Detail"].default_value = 5.0
    nt.links.new(tex_c.outputs["Object"], noise.inputs["Vector"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])

    # Emission strength 1.6 -> 2.6 (AgX-safe) modulated lightly by Fresnel rim for 3D depth
    strength_map = nt.nodes.new("ShaderNodeMapRange")
    strength_map.clamp = True
    strength_map.inputs["From Min"].default_value = 0.0
    strength_map.inputs["From Max"].default_value = 1.0
    strength_map.inputs["To Min"].default_value = 1.60
    strength_map.inputs["To Max"].default_value = 2.60
    nt.links.new(frac_node.outputs[0], strength_map.inputs["Value"])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    rim_mod = nt.nodes.new("ShaderNodeMapRange")
    rim_mod.clamp = True
    rim_mod.inputs["From Min"].default_value = 0.0
    rim_mod.inputs["From Max"].default_value = 1.0
    rim_mod.inputs["To Min"].default_value = 0.82
    rim_mod.inputs["To Max"].default_value = 1.00
    nt.links.new(lw.outputs["Facing"], rim_mod.inputs["Value"])

    em_final = nt.nodes.new("ShaderNodeMath")
    em_final.operation = "MULTIPLY"
    nt.links.new(strength_map.outputs["Result"], em_final.inputs[0])
    nt.links.new(rim_mod.outputs["Result"], em_final.inputs[1])

    bsdf.inputs["Roughness"].default_value = 0.22
    bsdf.inputs["Specular IOR Level"].default_value = 0.65
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(em_final.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, frac_node


def _make_force_fx_material(
    name: str,
    is_ring: bool = False,
) -> tuple[bpy.types.Material, bpy.types.ShaderNode, bpy.types.ShaderNode]:
    """Open/additive amber-to-crimson force-indicator shader (`Camera Ray * (1 - Backfacing)`)."""
    mat = bpy.data.materials.new(f"{PREFIX}{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")

    intensity_node = nt.nodes.new("ShaderNodeValue")
    intensity_node.name = "ForceIntensity"
    intensity_node.outputs[0].default_value = 0.0

    stress_node = nt.nodes.new("ShaderNodeValue")
    stress_node.name = "ForceStress"
    stress_node.outputs[0].default_value = 0.0

    col_ramp = nt.nodes.new("ShaderNodeValToRGB")
    col_ramp.color_ramp.elements[0].position = 0.0
    col_ramp.color_ramp.elements[0].color = (1.00, 0.56, 0.05, 1.0)
    col_ramp.color_ramp.elements[1].position = 1.0
    col_ramp.color_ramp.elements[1].color = (1.00, 0.14, 0.03, 1.0)
    nt.links.new(stress_node.outputs[0], col_ramp.inputs["Fac"])
    nt.links.new(col_ramp.outputs["Color"], em.inputs["Color"])

    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    cam_front = nt.nodes.new("ShaderNodeMath")
    cam_front.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], cam_front.inputs[1])

    if is_ring:
        # Soft vertical fade on shock ring top/bottom rims so there is zero hard cylinder edge
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
        abs_z = nt.nodes.new("ShaderNodeMath")
        abs_z.operation = "ABSOLUTE"
        nt.links.new(sep.outputs["Z"], abs_z.inputs[0])
        edge_fade = nt.nodes.new("ShaderNodeMapRange")
        edge_fade.clamp = True
        edge_fade.inputs["From Min"].default_value = 0.0
        edge_fade.inputs["From Max"].default_value = 0.008
        edge_fade.inputs["To Min"].default_value = 1.0
        edge_fade.inputs["To Max"].default_value = 0.0
        nt.links.new(abs_z.outputs["Value"], edge_fade.inputs["Value"])

        mask_mul = nt.nodes.new("ShaderNodeMath")
        mask_mul.operation = "MULTIPLY"
        nt.links.new(cam_front.outputs["Value"], mask_mul.inputs[0])
        nt.links.new(edge_fade.outputs["Result"], mask_mul.inputs[1])
        mask_sock = mask_mul.outputs["Value"]
    else:
        mask_sock = cam_front.outputs["Value"]

    str_mul = nt.nodes.new("ShaderNodeMath")
    str_mul.operation = "MULTIPLY"
    nt.links.new(mask_sock, str_mul.inputs[0])
    nt.links.new(intensity_node.outputs[0], str_mul.inputs[1])
    nt.links.new(str_mul.outputs["Value"], em.inputs["Strength"])

    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs["BSDF"], add.inputs[0])
    nt.links.new(em.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat, intensity_node, stress_node


# ---------------------------------------------------------------------------
# 5. ANIMATION DRIVERS (Contract-exact compression + pulsing 16-G force FX)
# ---------------------------------------------------------------------------
def _animate_spine_compression(
    vertebrae: list[bpy.types.Object],
    discs: list[bpy.types.Object],
    disc_frac_node: bpy.types.ShaderNode,
    disc_mat: bpy.types.Material,
    layout: list[dict[str, Any]],
    p_sac: Vector,
    chord_len: float,
    axis: Vector,
    anterior_dir: Vector,
    right_dir: Vector,
    total_frames: int,
) -> None:
    """Keyframe exact `C.spine_compression_cm(f)` contraction across 0..0.025m & disc squash/bulge."""
    f_start = max(1, C.EV["spine_compress_start"] - 2)
    f_end = min(total_frames, C.EV["spine_compress_end"] + 2)
    anim_frames = sorted(
        {1, SPINE_VIS_RANGE[0], *range(f_start, f_end + 1), SPINE_VIS_RANGE[1], total_frames}
    )
    anim_frames = [f for f in anim_frames if 1 <= f <= total_frames]

    # Precompute rest coordinates & endplate positions
    vert_rest: list[tuple[Vector, Euler]] = [
        _rest_point_and_euler(spec["u_center"], p_sac, chord_len, axis, anterior_dir, right_dir)
        for spec in layout
    ]
    ep_lower_rest: list[Vector] = [
        _rest_point_and_euler(spec["u_lower_ep"], p_sac, chord_len, axis, anterior_dir, right_dir)[0]
        for spec in layout[:23]
    ]
    ep_upper_rest: list[Vector] = [
        _rest_point_and_euler(spec["u_upper_ep"], p_sac, chord_len, axis, anterior_dir, right_dir)[0]
        for spec in layout[:23]
    ]

    frac_sock = disc_frac_node.outputs[0]

    for f in anim_frames:
        cm = C.spine_compression_cm(f)
        frac = cm / C.SPINE_MAX_COMPRESSION_CM
        drop_m = cm * 0.01  # 0.0 -> 0.025 m

        # Shader stress fraction
        frac_sock.default_value = frac
        frac_sock.keyframe_insert("default_value", frame=f)

        # 24 Vertebrae: exact proportional downward shift `-0.025 * (i / 24)` along spine axis
        for idx, (v_obj, (p_rest, rot_rest)) in enumerate(zip(vertebrae, vert_rest), start=1):
            v_obj.location = p_rest - axis * (drop_m * (idx / 24.0))
            v_obj.rotation_euler = rot_rest
            v_obj.keyframe_insert("location", frame=f)
            v_obj.keyframe_insert("rotation_euler", frame=f)

        # 23 Intervertebral Discs: exact midpoint between adjacent compressed endplates + squash/bulge
        sx = 1.0 + (DISC_SCALE_XY_MAX - 1.0) * frac
        sz = 1.0 - (1.0 - DISC_SCALE_Z_MAX) * frac
        for d_idx, d_obj in enumerate(discs):
            p_low = ep_lower_rest[d_idx] - axis * (drop_m * (d_idx / 24.0))
            p_up = ep_upper_rest[d_idx] - axis * (drop_m * ((d_idx + 1) / 24.0))
            tan = (p_up - p_low).normalized()
            y_dir = tan.cross(right_dir).normalized()
            rot_d = Matrix((right_dir, y_dir, tan)).transposed().to_euler("XYZ")

            d_obj.location = 0.5 * (p_low + p_up)
            d_obj.rotation_euler = rot_d
            d_obj.scale = Vector((sx, sx, sz))
            d_obj.keyframe_insert("location", frame=f)
            d_obj.keyframe_insert("rotation_euler", frame=f)
            d_obj.keyframe_insert("scale", frame=f)

    for obj in (*vertebrae, *discs):
        _set_fcurve_interp(obj, "LINEAR", "location")
        _set_fcurve_interp(obj, "LINEAR", "rotation_euler")
        _set_fcurve_interp(obj, "LINEAR", "scale")
    _set_fcurve_interp(disc_mat.node_tree, "LINEAR")


def _animate_force_fx(
    chevrons: list[bpy.types.Object],
    rings: list[bpy.types.Object],
    arrow_mat: bpy.types.Material,
    arrow_int_node: bpy.types.ShaderNode,
    arrow_str_node: bpy.types.ShaderNode,
    ring_mat: bpy.types.Material,
    ring_int_node: bpy.types.ShaderNode,
    ring_str_node: bpy.types.ShaderNode,
    p_sac: Vector,
    chord_len: float,
    axis: Vector,
    anterior_dir: Vector,
    right_dir: Vector,
    total_frames: int,
) -> None:
    """Animate pulsing downward 16-G force chevrons & lumbar/cervical shock rings during 930..1180."""
    f0, f1 = FORCE_FX_VIS_RANGE
    f0_c = max(1, min(total_frames, f0))
    f1_c = max(1, min(total_frames, f1))

    # Base anchor points along spine
    p_mid_spine, rot_mid = _rest_point_and_euler(0.48, p_sac, chord_len, axis, anterior_dir, right_dir)
    p_upper_rest, rot_upper = _rest_point_and_euler(0.90, p_sac, chord_len, axis, anterior_dir, right_dir)
    p_lumbar_rest, rot_lumbar = _rest_point_and_euler(0.14, p_sac, chord_len, axis, anterior_dir, right_dir)

    chev_base_locs = [
        p_mid_spine - right_dir * 0.108 + anterior_dir * 0.015,
        p_mid_spine + right_dir * 0.108 + anterior_dir * 0.015,
    ]
    chev_base_rots = [
        (rot_mid.to_matrix() @ Matrix.Rotation(math.radians(24.0), 3, "Z")).to_euler("XYZ"),
        (rot_mid.to_matrix() @ Matrix.Rotation(math.radians(-24.0), 3, "Z")).to_euler("XYZ"),
    ]

    frames = sorted({1, f0_c, *range(f0_c, f1_c + 1, 2), f1_c, total_frames})
    frames = [f for f in frames if 1 <= f <= total_frames]

    a_int_sock = arrow_int_node.outputs[0]
    a_str_sock = arrow_str_node.outputs[0]
    r_int_sock = ring_int_node.outputs[0]
    r_str_sock = ring_str_node.outputs[0]

    for f in frames:
        env = C.smoothstep(f0_c, f0_c + 15, f) * (1.0 - C.smoothstep(f1_c - 18, f1_c, f))
        frac = C.spine_compression_cm(f) / C.SPINE_MAX_COMPRESSION_CM
        t_rel = max(0.0, (f - f0_c) / float(C.FPS))
        pulse = 0.5 - 0.5 * math.cos(2.0 * math.pi * 1.45 * t_rel)  # smooth 1.45 Hz compression surge

        # Shader intensity & color shift (amber -> high-alert crimson)
        a_int_sock.default_value = (1.15 + 1.15 * pulse) * env
        a_str_sock.default_value = frac
        r_int_sock.default_value = (0.95 + 1.05 * pulse) * env
        r_str_sock.default_value = frac
        for sock in (a_int_sock, a_str_sock, r_int_sock, r_str_sock):
            sock.keyframe_insert("default_value", frame=f)

        # Flanking downward chevron arrows surge down along -axis
        down_surge = -0.032 * pulse - 0.012 * frac
        chev_scale = 0.94 + 0.12 * pulse
        for c_obj, base_p, base_r in zip(chevrons, chev_base_locs, chev_base_rots):
            c_obj.location = base_p + axis * down_surge
            c_obj.rotation_euler = base_r
            c_obj.scale = Vector((chev_scale, chev_scale, chev_scale))
            c_obj.keyframe_insert("location", frame=f)
            c_obj.keyframe_insert("rotation_euler", frame=f)
            c_obj.keyframe_insert("scale", frame=f)

        # Upper ring rides down with cervical compression; lower ring expands radially with lumbar disc bulge
        r_upper, r_lumbar = rings[0], rings[1]
        r_upper.location = p_upper_rest - axis * (0.024 * frac + 0.018 * pulse)
        r_upper.rotation_euler = rot_upper
        u_sc = 0.88 + 0.22 * pulse
        r_upper.scale = Vector((u_sc, u_sc, 1.0))

        r_lumbar.location = p_lumbar_rest - axis * (0.004 * frac)
        r_lumbar.rotation_euler = rot_lumbar
        l_sc = 0.92 + 0.26 * frac + 0.14 * pulse
        r_lumbar.scale = Vector((l_sc, l_sc, 1.0))

        for r_obj in rings:
            r_obj.keyframe_insert("location", frame=f)
            r_obj.keyframe_insert("rotation_euler", frame=f)
            r_obj.keyframe_insert("scale", frame=f)

    for obj in (*chevrons, *rings):
        _set_fcurve_interp(obj, "LINEAR", "location")
        _set_fcurve_interp(obj, "LINEAR", "scale")
    for mat in (arrow_mat, ring_mat):
        _set_fcurve_interp(mat.node_tree, "LINEAR")


# ---------------------------------------------------------------------------
# 6. PUBLIC BUILDER ENTRY POINT
# ---------------------------------------------------------------------------
def build_spine_xray(
    col: bpy.types.Collection,
    carrier: bpy.types.Object,
    sacrum_local: tuple[float, float, float],
    c1_local: tuple[float, float, float],
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """
    Build the complete 24-vertebrae + 23-disc + Sacrum + 16-G force FX X-ray spine rig.

    Parented cleanly to `carrier` (`Pilot_Carrier`) in carrier-local coordinates spanning
    `sacrum_local` to `c1_local`, animated by `C.spine_compression_cm(f)` across Beat 4.
    """
    p_sac, _, chord_len, axis, anterior_dir, right_dir = _spine_basis(sacrum_local, c1_local)
    layout = _compute_normalized_layout(chord_len)

    # 1. Root Empty anchored at carrier origin so all children live in exact carrier-local space
    root = bpy.data.objects.new(f"{PREFIX}Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.05
    root.location = Vector((0.0, 0.0, 0.0))
    root.rotation_mode = "XYZ"
    root.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    root.parent = carrier
    root.matrix_parent_inverse = Matrix.Identity(4)
    _link_obj(root, col)
    _key_visibility(root, [SPINE_VIS_RANGE], total_frames)

    # 2. Local-space Cycles Shaders
    mat_bone = _make_bone_material(root, "Mat_Bone", is_sacrum=False)
    mat_sacrum = _make_bone_material(root, "Mat_Sacrum", is_sacrum=True)
    mat_disc, disc_frac_node = _make_disc_material(root, "Mat_Disc_Stress")
    mat_arrow, arrow_int_node, arrow_str_node = _make_force_fx_material("Mat_ForceArrow", is_ring=False)
    mat_ring, ring_int_node, ring_str_node = _make_force_fx_material("Mat_ForceRing", is_ring=True)

    # 3. Sacrum / Pelvis Wedge (`Spine_Sacrum`) anchored at `sacrum_local`
    _, sac_rot = _rest_point_and_euler(0.0, p_sac, chord_len, axis, anterior_dir, right_dir)
    bm_sac = _build_sacrum_bm(layout[0])
    sacrum_obj = _create_mesh_obj(
        f"{PREFIX}Sacrum",
        bm_sac,
        mat_sacrum,
        col,
        parent=root,
        location=p_sac,
        rotation=sac_rot,
        visible_shadow=False,
    )
    _key_visibility(sacrum_obj, [SPINE_VIS_RANGE], total_frames)

    # 4. 24 Individually Sculpted Vertebrae (`Spine_Vert_L5` .. `Spine_Vert_C1`)
    vertebrae: list[bpy.types.Object] = []
    for spec in layout:
        label = str(spec["label"])
        p_rest, rot_rest = _rest_point_and_euler(spec["u_center"], p_sac, chord_len, axis, anterior_dir, right_dir)
        bm_v = _build_vertebra_bm(spec)
        v_obj = _create_mesh_obj(
            f"{PREFIX}Vert_{label}",
            bm_v,
            mat_bone,
            col,
            parent=root,
            location=p_rest,
            rotation=rot_rest,
            visible_shadow=False,
        )
        _key_visibility(v_obj, [SPINE_VIS_RANGE], total_frames)
        vertebrae.append(v_obj)

    # 5. 23 Sculpted Intervertebral Discs (`Spine_Disc_01` .. `Spine_Disc_23`)
    # Disc 01 sits at Sacrum-L5 (L5-S1); Disc 23 sits at C3-C2 (C1-C2 is anatomical odontoid pivot).
    discs: list[bpy.types.Object] = []
    for d_idx in range(23):
        upper_spec = layout[d_idx]
        if d_idx == 0:
            disc_rx = upper_spec["rx"] * DISC_RADIUS_RATIO
            disc_ry = upper_spec["ry"] * DISC_RADIUS_RATIO
        else:
            lower_spec = layout[d_idx - 1]
            disc_rx = 0.5 * (lower_spec["rx"] + upper_spec["rx"]) * DISC_RADIUS_RATIO
            disc_ry = 0.5 * (lower_spec["ry"] + upper_spec["ry"]) * DISC_RADIUS_RATIO

        disc_h = upper_spec["gap_below"] + 2.0 * DISC_TUCK_M
        p_low, _ = _rest_point_and_euler(upper_spec["u_lower_ep"], p_sac, chord_len, axis, anterior_dir, right_dir)
        p_up, _ = _rest_point_and_euler(upper_spec["u_upper_ep"], p_sac, chord_len, axis, anterior_dir, right_dir)
        tan = (p_up - p_low).normalized()
        y_dir = tan.cross(right_dir).normalized()
        rot_d = Matrix((right_dir, y_dir, tan)).transposed().to_euler("XYZ")

        bm_d = _build_disc_bm(disc_rx, disc_ry, disc_h)
        d_obj = _create_mesh_obj(
            f"{PREFIX}Disc_{d_idx + 1:02d}",
            bm_d,
            mat_disc,
            col,
            parent=root,
            location=0.5 * (p_low + p_up),
            rotation=rot_d,
            visible_shadow=False,
        )
        _key_visibility(d_obj, [SPINE_VIS_RANGE], total_frames)
        discs.append(d_obj)

    # 6. Keyframe exact 2.5 cm spine contraction & disc compression
    _animate_spine_compression(
        vertebrae=vertebrae,
        discs=discs,
        disc_frac_node=disc_frac_node,
        disc_mat=mat_disc,
        layout=layout,
        p_sac=p_sac,
        chord_len=chord_len,
        axis=axis,
        anterior_dir=anterior_dir,
        right_dir=right_dir,
        total_frames=total_frames,
    )

    # 7. Glowing 16-G Downward Force Chevron Arrows & Shock Rings (`Spine_ForceArrow_*`)
    chevrons: list[bpy.types.Object] = []
    for side in ("L", "R"):
        bm_ch = _build_chevron_stack_bm(count=3, spacing=0.072)
        ch_obj = _create_mesh_obj(
            f"{PREFIX}ForceArrow_{side}",
            bm_ch,
            mat_arrow,
            col,
            parent=root,
            visible_shadow=False,
        )
        _key_visibility(ch_obj, [FORCE_FX_VIS_RANGE], total_frames)
        chevrons.append(ch_obj)

    rings: list[bpy.types.Object] = []
    for r_name, rad in (("Ring_Upper", 0.044), ("Ring_Lumbar", 0.056)):
        bm_rg = _build_open_shock_ring_bm(radius=rad, height=0.016, flare=0.006)
        rg_obj = _create_mesh_obj(
            f"{PREFIX}ForceArrow_{r_name}",
            bm_rg,
            mat_ring,
            col,
            parent=root,
            visible_shadow=False,
            recalc_normals=False,
        )
        _key_visibility(rg_obj, [FORCE_FX_VIS_RANGE], total_frames)
        rings.append(rg_obj)

    _animate_force_fx(
        chevrons=chevrons,
        rings=rings,
        arrow_mat=mat_arrow,
        arrow_int_node=arrow_int_node,
        arrow_str_node=arrow_str_node,
        ring_mat=mat_ring,
        ring_int_node=ring_int_node,
        ring_str_node=ring_str_node,
        p_sac=p_sac,
        chord_len=chord_len,
        axis=axis,
        anterior_dir=anterior_dir,
        right_dir=right_dir,
        total_frames=total_frames,
    )

    force_fx = [*chevrons, *rings]
    all_objs = [root, sacrum_obj, *vertebrae, *discs, *force_fx]
    return {
        "root": root,
        "sacrum": sacrum_obj,
        "vertebrae": vertebrae,
        "discs": discs,
        "force_fx": force_fx,
        "all_objs": all_objs,
    }


# ---------------------------------------------------------------------------
# 7. STANDALONE SELF-TEST (Runs only when executed directly; never imported)
# ---------------------------------------------------------------------------
def _run_standalone_test() -> None:
    out_dir = "/tmp/fje_spine"
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES

    col = bpy.data.collections.new("SpineXRayTest")
    scene.collection.children.link(col)

    carrier = bpy.data.objects.new("Pilot_Carrier", None)
    col.objects.link(carrier)

    sacrum_local = C.PILOT["sacrum"]
    c1_local = C.PILOT["c1_skull_base"]
    res = build_spine_xray(col, carrier, sacrum_local, c1_local, total_frames=C.TOTAL_FRAMES)

    # Structural & contract assertions
    assert len(res["vertebrae"]) == 24, f"Expected 24 vertebrae, got {len(res['vertebrae'])}"
    assert len(res["discs"]) == 23, f"Expected 23 discs, got {len(res['discs'])}"
    assert len(res["force_fx"]) >= 2, f"Expected >=2 force_fx, got {len(res['force_fx'])}"
    assert all(obj.name.startswith(PREFIX) for obj in res["all_objs"])

    c1_obj = res["vertebrae"][-1]
    disc01 = res["discs"][0]
    spine_axis = (Vector(c1_local) - Vector(sacrum_local)).normalized()

    scene.frame_set(884)
    assert res["sacrum"].hide_render is True, "Spine must be CONSTANT-hidden before f=885"
    scene.frame_set(920)
    assert res["sacrum"].hide_render is False, "Spine must be visible at f=920"
    assert res["force_fx"][0].hide_render is True, "Force FX must be hidden before f=930"
    c1_rest = c1_obj.location.copy()
    assert (c1_rest - Vector(c1_local)).length < 1e-4, f"C1 rest mismatch: {c1_rest}"
    assert abs(disc01.scale.z - 1.0) < 1e-4, f"Disc rest Z scale should be 1.0, got {disc01.scale.z}"

    scene.frame_set(1080)
    assert res["force_fx"][0].hide_render is False, "Force FX must be visible at f=1080"
    c1_drop = (c1_rest - c1_obj.location).dot(spine_axis)
    assert abs(c1_drop - 0.025) < 1e-4, f"Expected exact 0.025m C1 drop, got {c1_drop:.6f}"
    assert abs(disc01.scale.z - 0.52) < 1e-4, f"Expected disc scale.z=0.52, got {disc01.scale.z}"
    assert abs(disc01.scale.x - 1.28) < 1e-4, f"Expected disc scale.x=1.28, got {disc01.scale.x}"

    scene.frame_set(1237)
    assert res["sacrum"].hide_render is True, "Spine must be CONSTANT-hidden from f>=1237"

    # Studio preview world, lighting & 9:16 portrait camera
    world = bpy.data.worlds.new("SpineTestWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.015, 0.028, 0.055, 1.0)
        bg.inputs["Strength"].default_value = 0.65

    key_data = bpy.data.lights.new("TestKey", type="AREA")
    key_data.energy = 120.0
    key_data.size = 1.2
    key_obj = bpy.data.objects.new("TestKey", key_data)
    key_obj.location = Vector((1.2, 1.4, 1.1))
    key_obj.rotation_euler = (Vector((0.0, -0.09, 0.40)) - key_obj.location).to_track_quat("-Z", "Y").to_euler()
    col.objects.link(key_obj)

    cam_data = bpy.data.cameras.new("TestCam")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("TestCam", cam_data)
    cam_obj.location = Vector((0.88, 0.72, 0.44))
    cam_obj.rotation_euler = (Vector((0.0, -0.09, 0.39)) - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    col.objects.link(cam_obj)
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 6
    scene.cycles.transparent_max_bounces = 12
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 25
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        pass

    for frame in (920, 1080):
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(out_dir, f"spine_f{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[spine_xray] Standalone QA passed; rendered f=920 & f=1080 to {out_dir}")


if __name__ == "__main__":
    _run_standalone_test()
