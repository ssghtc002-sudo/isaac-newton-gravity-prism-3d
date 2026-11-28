"""
heli_fx.py — Open-Ended Additive Volumetric Airflow, Clutch Lock/Free Rings,
Maple-Seed Helical Vortices, 30-Foot Flare Cushion Wash & Rooftop Touchdown FX
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
  - Every volumetric beam, upward airflow streamline, helical vortex ribbon & flare wash ring uses
    `cap_ends=False` / open-ended strips + AgX-safe `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`
    so zero flat end-caps or opaque pastel walls ever appear!
  - Materials appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
  - Blender 5.3.0 safe FCurves (`C.iter_action_fcurves`) & `C.get_mix_sockets`.
"""
from __future__ import annotations

import math
from typing import Any

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]

import heli_contract as C


def _link(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mats: list[bpy.types.Material],
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smooth: bool = True,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    for m in mats:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    obj.rotation_mode = "XYZ"
    obj.visible_shadow = False
    if parent is not None:
        obj.parent = parent
    _link(obj, col)
    return obj


def _key_visibility(objs: list[bpy.types.Object], windows: list[tuple[int, int]], total_frames: int) -> None:
    def _vis(f: int) -> bool:
        return any(a <= f <= b for a, b in windows)

    frames = {1, total_frames}
    for a, b in windows:
        for f in (a - 1, a, b, b + 1):
            if 1 <= f <= total_frames:
                frames.add(f)

    for obj in objs:
        for f in sorted(frames):
            h = not _vis(f)
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


def _make_additive_glow_mat(
    name: str,
    rgb: tuple[float, float, float],
    strength: float = 1.45,
    fresnel_pow: float = 1.6,
) -> bpy.types.Material:
    """AgX-safe open-ended additive shader: Transparent + Emission * CameraRay * (1 - Backfacing) * soft Fresnel."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    lp = nt.nodes.new("ShaderNodeLightPath")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    inv_fac = nt.nodes.new("ShaderNodeMath")
    inv_fac.operation = "SUBTRACT"
    inv_fac.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv_fac.inputs[1])

    pow_edge = nt.nodes.new("ShaderNodeMath")
    pow_edge.operation = "POWER"
    pow_edge.inputs[1].default_value = fresnel_pow
    nt.links.new(inv_fac.outputs["Value"], pow_edge.inputs[0])

    # Keep gentle center core + soft silhouette falloff (0.28 + 0.72 * soft_edge)
    mix_f = nt.nodes.new("ShaderNodeMix")
    mix_f.data_type = "FLOAT"
    f_fac, f_a, f_b = C.get_mix_sockets(mix_f)
    f_a.default_value = 0.30
    f_b.default_value = 1.00
    nt.links.new(pow_edge.outputs["Value"], f_fac)

    mul1 = nt.nodes.new("ShaderNodeMath")
    mul1.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Camera Ray"], mul1.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], mul1.inputs[1])

    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    nt.links.new(mul1.outputs["Value"], mul2.inputs[0])
    nt.links.new(mix_f.outputs[0], mul2.inputs[1])

    mul3 = nt.nodes.new("ShaderNodeMath")
    mul3.operation = "MULTIPLY"
    mul3.inputs[1].default_value = min(1.95, strength)
    nt.links.new(mul2.outputs["Value"], mul3.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(mul3.outputs["Value"], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat


def _build_open_helix_tube_bm(
    bm: bmesh.types.BMesh,
    radius_fn,
    z_start: float,
    z_end: float,
    turns: float,
    phase_rad: float,
    tube_r: float,
    n_steps: int = 36,
    n_ring: int = 8,
    mat_idx: int = 0,
) -> None:
    """Create a smooth open-ended helical volumetric streamline tube (ZERO flat end caps!)."""
    centers: list[Vector] = []
    for i in range(n_steps + 1):
        u = i / float(n_steps)
        ang = phase_rad + turns * 2.0 * math.pi * u
        r = radius_fn(u)
        z = C.lerp(z_start, z_end, u)
        centers.append(Vector((r * math.cos(ang), r * math.sin(ang), z)))

    rings: list[list[bmesh.types.BMVert]] = []
    for i, c in enumerate(centers):
        u = i / float(n_steps)
        t_dir = (centers[min(n_steps, i + 1)] - centers[max(0, i - 1)]).normalized()
        ref = Vector((0.0, 0.0, 1.0)) if abs(t_dir.z) < 0.92 else Vector((1.0, 0.0, 0.0))
        n_vec = t_dir.cross(ref).normalized()
        b_vec = t_dir.cross(n_vec).normalized()
        # Taper cleanly to thin tip at ends so streamlines never show abrupt block cuts!
        taper = math.sin(math.pi * C.clamp(u, 0.02, 0.98)) ** 0.55
        rr = max(0.0025, tube_r * taper)
        ring = []
        for k in range(n_ring):
            a = 2.0 * math.pi * k / n_ring
            pt = c + (n_vec * math.cos(a) + b_vec * math.sin(a)) * rr
            ring.append(bm.verts.new(pt))
        rings.append(ring)

    for i in range(n_steps):
        r0, r1 = rings[i], rings[i + 1]
        for k in range(n_ring):
            kn = (k + 1) % n_ring
            nf = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
            nf.material_index = mat_idx

    # Add open-ended upward directional chevron collar near u=0.72 so upward airflow direction is unmistakable!
    idx_c = int(n_steps * 0.72)
    p_tip = centers[min(n_steps, idx_c + 2)]
    p_base = centers[max(0, idx_c - 1)]
    d_c = (p_tip - p_base).normalized()
    mid_c = 0.5 * (p_tip + p_base)
    rm_c = d_c.to_track_quat("Z", "Y").to_matrix().to_4x4()
    res_c = bmesh.ops.create_cone(
        bm, cap_ends=False, segments=10,
        radius1=tube_r * 2.6, radius2=0.002, depth=max(0.05, (p_tip - p_base).length),
        matrix=Matrix.Translation(mid_c) @ rm_c,
    )
    vs = set(res_c["verts"])
    for f in bm.faces:
        if all(v in vs for v in f.verts):
            f.material_index = mat_idx


def _build_open_ring_bm(
    bm: bmesh.types.BMesh,
    radius: float,
    height: float,
    segs: int = 40,
    mat_idx: int = 0,
) -> None:
    """Open-ended cylindrical glow band (cap_ends=False, zero flat end-cap discs)."""
    res = bmesh.ops.create_cone(
        bm, cap_ends=False, segments=segs,
        radius1=radius, radius2=radius * 1.04, depth=height,
    )
    vs = set(res["verts"])
    for f in bm.faces:
        if all(v in vs for v in f.verts):
            f.material_index = mat_idx


def build_heli_fx(
    col: bpy.types.Collection,
    heli_root: bpy.types.Object,
    maple_root: bpy.types.Object,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build all 6-beat open-ended additive FX strictly synced to `heli_contract.EV`."""
    mat_emerald = _make_additive_glow_mat("Mat_FX_EmeraldAirflow", (0.12, 0.96, 0.68), strength=1.55, fresnel_pow=1.4)
    mat_cyan = _make_additive_glow_mat("Mat_FX_CyanVortex", (0.14, 0.86, 1.00), strength=1.50, fresnel_pow=1.5)
    mat_gold = _make_additive_glow_mat("Mat_FX_GoldFlywheel", (1.00, 0.78, 0.22), strength=1.65, fresnel_pow=1.6)
    mat_crimson = _make_additive_glow_mat("Mat_FX_CrimsonLock", (1.00, 0.16, 0.08), strength=1.70, fresnel_pow=1.5)
    mat_smoke = _make_additive_glow_mat("Mat_FX_SmokeWisp", (0.76, 0.56, 0.44), strength=0.42, fresnel_pow=2.2)

    fx_objs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BEAT 1 (14..240): Turbine Failure Smoke Wisps Drifting Outside Cockpit
    # ------------------------------------------------------------------
    smoke_objs: list[bpy.types.Object] = []
    for i in range(4):
        bm_s = bmesh.new()
        res = bmesh.ops.create_icosphere(bm_s, subdivisions=2, radius=0.22 + 0.06 * i)
        # Organic turbulent displacement so smoke never looks like flat spheres!
        for v in res["verts"]:
            n = 1.0 + 0.28 * math.sin(v.co.x * 11.0 + i) * math.cos(v.co.y * 9.0)
            v.co *= n
            v.co.y *= 1.8
        so = _new_bmesh_obj(f"FX_B1_EngineSmoke_{i:02d}", bm_s, [mat_smoke], col, smooth=True)
        smoke_objs.append(so)
        _key_visibility([so], [(14, 240)], total_frames)
        for f in range(14, 241, 8):
            u = ((f - 14 + i * 24) % 96) / 96.0
            so.location = Vector((0.88 + 0.35 * u + 0.12 * math.sin(u * 5.0 + i), 1.80 + 2.40 * u, 1.52 + 0.45 * u))
            sc = 0.55 + 1.15 * u
            so.scale = Vector((sc, sc, sc * 0.75))
            so.keyframe_insert("location", frame=f)
            so.keyframe_insert("scale", frame=f)
    fx_objs["b1_smoke"] = smoke_objs

    # ------------------------------------------------------------------
    # BEAT 2 (241..570): Seized Shaft Crimson Pulse Ring + Sprag Clutch Golden Freewheel Ring
    # ------------------------------------------------------------------
    bm_lock = bmesh.new()
    _build_open_ring_bm(bm_lock, radius=0.092, height=0.036, segs=28, mat_idx=0)
    lock_ring = _new_bmesh_obj(
        "FX_B2_ShaftLockRing", bm_lock, [mat_crimson], col,
        parent=heli_root, location=C.HELI_TURBINE_LOCAL_POS, smooth=True,
    )
    lock_ring.rotation_euler = Euler((math.radians(90.0), 0.0, 0.0), "XYZ")
    _key_visibility([lock_ring], [(330, 565)], total_frames)
    for f in range(330, 566, 8):
        p = 0.92 + 0.20 * (0.5 + 0.5 * math.sin((f - 330) * 0.35))
        lock_ring.scale = Vector((p, p, 1.0))
        lock_ring.keyframe_insert("scale", frame=f)
    fx_objs["b2_lock_ring"] = lock_ring

    bm_free = bmesh.new()
    _build_open_ring_bm(bm_free, radius=0.125, height=0.048, segs=36, mat_idx=0)
    free_ring = _new_bmesh_obj(
        "FX_B2_ClutchFreeRing", bm_free, [mat_gold], col,
        parent=heli_root, location=C.HELI_CLUTCH_LOCAL_POS, smooth=True,
    )
    _key_visibility([free_ring], [(395, 568)], total_frames)
    for f in range(395, 569, 6):
        u = C.smootherstep(395.0, 445.0, float(f))
        sc = C.lerp(0.65, 1.15, u) + 0.06 * math.sin((f - 395) * 0.25)
        free_ring.scale = Vector((sc, sc, 1.0))
        free_ring.rotation_euler = Euler((0.0, 0.0, (f - 395) * 0.14), "XYZ")
        free_ring.keyframe_insert("scale", frame=f)
        free_ring.keyframe_insert("rotation_euler", frame=f)
    fx_objs["b2_free_ring"] = free_ring

    # ------------------------------------------------------------------
    # BEAT 3 (571..960): 10 Upward Emerald-Cyan Airflow Streamlines + Flywheel Kinetic Ring
    # ------------------------------------------------------------------
    bm_up = bmesh.new()
    n_streams = 10
    for i in range(n_streams):
        ph = 2.0 * math.pi * i / n_streams
        r_base = 0.36 + 0.54 * ((i % 3) / 2.0)
        _build_open_helix_tube_bm(
            bm_up,
            radius_fn=lambda u, rb=r_base: rb * (1.08 - 0.16 * math.sin(math.pi * u)),
            z_start=-0.68,
            z_end=0.92,
            turns=0.55,
            phase_rad=ph,
            tube_r=0.015,
            n_steps=28,
            n_ring=8,
            mat_idx=0 if (i % 2 == 0) else 1,
        )
    up_streams = _new_bmesh_obj(
        "FX_B3_UpwardAirflowStreams", bm_up, [mat_emerald, mat_cyan], col,
        parent=heli_root, location=(0.0, 0.02, 0.32), smooth=True,
    )
    _key_visibility([up_streams], [(585, 960)], total_frames)
    for f in range(585, 961, 4):
        ff = float(f)
        # Continuous upward surging cycle + swirl matching main rotor autorotation!
        cyc = ((ff - 585.0) % 45.0) / 45.0
        z_shift = C.lerp(-0.14, 0.16, cyc)
        grow = C.smootherstep(585.0, 625.0, ff)
        up_streams.location = Vector((0.0, 0.02, 0.32 + z_shift))
        up_streams.rotation_euler = Euler((0.0, 0.0, (ff - 585.0) * 0.065), "XYZ")
        up_streams.scale = Vector((grow, grow, grow))
        up_streams.keyframe_insert("location", frame=f)
        up_streams.keyframe_insert("rotation_euler", frame=f)
        up_streams.keyframe_insert("scale", frame=f)
    fx_objs["b3_upward_streams"] = up_streams

    # Glowing Golden-Cyan Kinetic Flywheel Ring around spinning rotor tips (Beat 3)
    bm_fw = bmesh.new()
    _build_open_ring_bm(bm_fw, radius=C.HELI_ROTOR_RADIUS + 0.04, height=0.045, segs=48, mat_idx=0)
    fw_ring = _new_bmesh_obj(
        "FX_B3_FlywheelKineticRing", bm_fw, [mat_gold], col,
        parent=heli_root, location=(0.0, 0.02, C.HELI_ROTOR_HUB_LOCAL_Z + 0.02), smooth=True,
    )
    _key_visibility([fw_ring], [(700, 960)], total_frames)
    for f in range(700, 961, 6):
        u = C.smootherstep(700.0, 760.0, float(f))
        pulse = 1.0 + 0.03 * math.sin((f - 700) * 0.25)
        fw_ring.scale = Vector((u * pulse, u * pulse, 1.0))
        fw_ring.rotation_euler = Euler((0.0, 0.0, (f - 700) * 0.18), "XYZ")
        fw_ring.keyframe_insert("scale", frame=f)
        fw_ring.keyframe_insert("rotation_euler", frame=f)
    fx_objs["b3_flywheel_ring"] = fw_ring

    # ------------------------------------------------------------------
    # BEAT 4 (961..1290): Matched Emerald Helical Vortices on Maple Seed (Left) & Rescue Rotor (Right)
    # ------------------------------------------------------------------
    bm_mv = bmesh.new()
    for i in range(4):
        _build_open_helix_tube_bm(
            bm_mv,
            radius_fn=lambda u: 0.34 + 0.12 * u,
            z_start=-0.38,
            z_end=0.38,
            turns=1.15,
            phase_rad=i * 0.5 * math.pi,
            tube_r=0.011,
            n_steps=28,
            n_ring=8,
            mat_idx=0,
        )
    maple_vortex = _new_bmesh_obj("FX_B4_MapleVortex", bm_mv, [mat_emerald], col, smooth=True)
    _key_visibility([maple_vortex], [(965, 1290)], total_frames)
    for f in range(965, 1291, 4):
        mpos, _ = C.maple_seed_pose_at(float(f))
        maple_vortex.location = Vector(mpos)
        maple_vortex.rotation_euler = Euler((0.0, 0.0, math.radians((f - 965) * 6.5)), "XYZ")
        maple_vortex.keyframe_insert("location", frame=f)
        maple_vortex.keyframe_insert("rotation_euler", frame=f)
    fx_objs["b4_maple_vortex"] = maple_vortex

    bm_hv = bmesh.new()
    for i in range(4):
        _build_open_helix_tube_bm(
            bm_hv,
            radius_fn=lambda u: 0.72 + 0.20 * u,
            z_start=-0.42,
            z_end=0.46,
            turns=0.95,
            phase_rad=i * 0.5 * math.pi,
            tube_r=0.014,
            n_steps=28,
            n_ring=8,
            mat_idx=0,
        )
    heli_vortex = _new_bmesh_obj(
        "FX_B4_HeliVortex", bm_hv, [mat_emerald], col,
        parent=heli_root, location=(0.0, 0.02, C.HELI_ROTOR_HUB_LOCAL_Z), smooth=True,
    )
    _key_visibility([heli_vortex], [(965, 1290)], total_frames)
    for f in range(965, 1291, 4):
        heli_vortex.rotation_euler = Euler((0.0, 0.0, math.radians((f - 965) * 6.5)), "XYZ")
        heli_vortex.keyframe_insert("rotation_euler", frame=f)
    fx_objs["b4_heli_vortex"] = heli_vortex

    # ------------------------------------------------------------------
    # BEAT 5B (1451..1580): Dramatic 30-Foot Flare Downwash Cushion Ring
    # ------------------------------------------------------------------
    bm_fl = bmesh.new()
    _build_open_ring_bm(bm_fl, radius=1.18, height=0.14, segs=48, mat_idx=0)
    _build_open_ring_bm(bm_fl, radius=0.78, height=0.10, segs=40, mat_idx=1)
    flare_wash = _new_bmesh_obj("FX_B5B_FlareCushionRing", bm_fl, [mat_gold, mat_cyan], col, smooth=True)
    _key_visibility([flare_wash], [(1460, 1580)], total_frames)
    for f in range(1460, 1581, 3):
        ff = float(f)
        hpos, _ = C.heli_pose_at(ff)
        cyc = ((ff - 1460.0) % 36.0) / 36.0
        flare_wash.location = Vector((hpos[0], hpos[1], C.lerp(hpos[2] + 0.15, float(C.STAGE["helipad_z"]) + 0.08, cyc)))
        sc = C.lerp(0.72, 1.48, cyc)
        flare_wash.scale = Vector((sc, sc, 1.0 - 0.35 * cyc))
        flare_wash.keyframe_insert("location", frame=f)
        flare_wash.keyframe_insert("scale", frame=f)
    fx_objs["b5b_flare_wash"] = flare_wash

    # ------------------------------------------------------------------
    # BEAT 6 (1581..1800): Gentle Rooftop Helipad Touchdown Wash Shimmer Ring
    # ------------------------------------------------------------------
    bm_td = bmesh.new()
    _build_open_ring_bm(bm_td, radius=1.35, height=0.07, segs=48, mat_idx=0)
    td_wash = _new_bmesh_obj(
        "FX_B6_TouchdownWash", bm_td, [mat_gold], col,
        location=(float(C.STAGE["helipad_center"][0]), float(C.STAGE["helipad_center"][1]), float(C.STAGE["helipad_z"]) + 0.05),
        smooth=True,
    )
    _key_visibility([td_wash], [(1645, 1775)], total_frames)
    for f in range(1645, 1776, 4):
        u = C.smootherstep(1645.0, 1765.0, float(f))
        sc = C.lerp(0.55, 1.68, u)
        td_wash.scale = Vector((sc, sc, max(0.08, 1.0 - 0.85 * u)))
        td_wash.keyframe_insert("scale", frame=f)
    fx_objs["b6_touchdown_wash"] = td_wash

    return fx_objs
