"""
heli_fx.py — Blockbuster Volumetric Engine Fire, Billowing Smoke, Clean 3D Upward Ribbon Arrows,
Hero Maple-Seed Lift Swirl, 30-Ft Flare Air-Cushion & Helipad Ground-Effect Dust Wave (V2 Ultimate)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Eliminates ALL draft clutter & obeys AGENTS.md V12 Section 3 strictly:
  - Beat 1A (`18..115`): Procedural 3D **Turbulent Engine Compressor-Stall Fireball** (displaced icosphere
    with `(1 - Facing)^p * 3D Noise` radial soft-edge falloff — ZERO flat bullseye rings!), curved 3D
    **Tapered Metallic Spark Streams**, and a **Billowing Dark Charcoal Exhaust Smoke Plume** pouring out
    of the dorsal turbine exhausts so the engine failure is 100% unmistakable in the first 1.5 seconds!
  - Beat 2 (`345..560`): Glowing crimson **Shaft Seizure Stall Halo** + sparks around the frozen turbine!
  - Beat 3 (`571..960`): Replaces messy spaghetti wires with **6 Wide, Clean Sculpted Translucent Emerald-Cyan
    3D Ribbon Arrows with Crisp Arrowheads** streaming smoothly UP through the rotor disc + sleek kinetic ring!
  - Beat 4 (`961..1290`): ONE clean golden-emerald helical ribbon trail on the enlarged Maple Seed + ONE
    matching clean trail on the Rescue Helicopter (zero visual tangle!).
  - Beat 5B & 6 (`1462..1790`): Dramatic glowing **30-Ft Flare Air-Cushion Shockwave Ring** + expanding
    golden **Rooftop Ground-Effect Dust & Downwash Wave** sweeping across the weathered concrete deck!
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


def _new_fx_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    parent: bpy.types.Object | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.materials.append(mat)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    if parent is not None:
        obj.parent = parent
    obj.visible_shadow = False
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


def _make_additive_ribbon_mat(name: str, rgb: tuple[float, float, float], strength: float = 1.65) -> bpy.types.Material:
    """Open-ended V12 additive optical shader: AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul1 = nt.nodes.new("ShaderNodeMath")
    mul1.operation = "MULTIPLY"
    nt.links.new(inv_bf.outputs["Value"], mul1.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mul1.inputs[1])

    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = min(1.90, strength)
    nt.links.new(mul1.outputs["Value"], mul2.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    nt.links.new(mul2.outputs["Value"], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_s = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_s.inputs[0])
    nt.links.new(emit.outputs["Emission"], add_s.inputs[1])
    nt.links.new(add_s.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_turbulent_fireball_mat(name: str) -> bpy.types.Material:
    """
    V12 Section 3 Procedural 3D Turbulent Fireball (ZERO flat concentric bullseyes!):
      - Local-space 3D Noise * pow(1 - Facing, 2.2) radial soft-edge falloff into fiery orange-yellow plasma!
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.2
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    pow_f = nt.nodes.new("ShaderNodeMath")
    pow_f.operation = "POWER"
    pow_f.inputs[1].default_value = 2.0
    nt.links.new(inv_f.outputs["Value"], pow_f.inputs[0])

    mul_n = nt.nodes.new("ShaderNodeMath")
    mul_n.operation = "MULTIPLY"
    nt.links.new(pow_f.outputs["Value"], mul_n.inputs[0])
    nt.links.new(noise.outputs["Fac"], mul_n.inputs[1])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul_cam = nt.nodes.new("ShaderNodeMath")
    mul_cam.operation = "MULTIPLY"
    nt.links.new(mul_n.outputs["Value"], mul_cam.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mul_cam.inputs[1])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.15
    cr.elements[0].color = (0.95, 0.14, 0.02, 1.0)
    cr.elements[1].position = 0.72
    cr.elements[1].color = (1.00, 0.78, 0.18, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    mul_str = nt.nodes.new("ShaderNodeMath")
    mul_str.operation = "MULTIPLY"
    mul_str.inputs[1].default_value = 3.2
    nt.links.new(mul_cam.outputs["Value"], mul_str.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    nt.links.new(mul_str.outputs["Value"], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_s = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_s.inputs[0])
    nt.links.new(emit.outputs["Emission"], add_s.inputs[1])
    nt.links.new(add_s.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_dark_smoke_mat(name: str, opacity: float = 0.68) -> bpy.types.Material:
    """Soft edge-faded dark charcoal engine-smoke plume shader (casts zero harsh solid shadows!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.50
    inv_f = nt.nodes.new("ShaderNodeMath")
    inv_f.operation = "SUBTRACT"
    inv_f.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs["Facing"], inv_f.inputs[1])

    pow_f = nt.nodes.new("ShaderNodeMath")
    pow_f.operation = "POWER"
    pow_f.inputs[1].default_value = 1.8
    nt.links.new(inv_f.outputs["Value"], pow_f.inputs[0])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = opacity
    nt.links.new(pow_f.outputs["Value"], mul.inputs[0])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul_cam = nt.nodes.new("ShaderNodeMath")
    mul_cam.operation = "MULTIPLY"
    nt.links.new(mul.outputs["Value"], mul_cam.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mul_cam.inputs[1])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.045, 0.048, 0.055, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.96
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul_cam.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _add_open_ribbon_tube(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    r_start: float,
    r_end: float,
    segs: int = 8,
) -> None:
    """Create a smooth open-ended tapered 3D tube (`cap_ends=False`) along polyline `pts`."""
    if len(pts) < 2:
        return
    rings: list[list[Any]] = []
    n_pts = len(pts)
    for i, p in enumerate(pts):
        u = i / float(max(1, n_pts - 1))
        r = C.lerp(r_start, r_end, u)
        if i == 0:
            tan = (pts[1] - pts[0]).normalized()
        elif i == n_pts - 1:
            tan = (pts[-1] - pts[-2]).normalized()
        else:
            tan = (pts[i + 1] - pts[i - 1]).normalized()
        up_ref = Vector((0.0, 0.0, 1.0)) if abs(tan.z) < 0.92 else Vector((1.0, 0.0, 0.0))
        right = tan.cross(up_ref).normalized()
        up = right.cross(tan).normalized()
        ring = []
        for s in range(segs):
            ang = 2.0 * math.pi * (s / float(segs))
            v = p + (right * math.cos(ang) + up * math.sin(ang)) * r
            ring.append(bm.verts.new(v))
        rings.append(ring)

    for r0, r1 in zip(rings, rings[1:]):
        for s in range(segs):
            ns = (s + 1) % segs
            bm.faces.new([r0[s], r0[ns], r1[ns], r1[s]])


def _add_wide_3d_ribbon_arrow(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    width: float,
    head_len: float = 0.16,
    head_w: float = 0.11,
) -> None:
    """
    Create a crisp, clean, double-sided sculpted 3D flat ribbon arrow with wide arrowhead
    (100% readable on vertical phone screens — ZERO messy spaghetti wireframes!).
    """
    if len(pts) < 3:
        return
    n_pts = len(pts)
    # Taper ribbon smoothly along body, then flare out for crisp arrowhead at final segment!
    body_pts = pts[:-1]
    left_v: list[Any] = []
    right_v: list[Any] = []
    for i, p in enumerate(body_pts):
        u = i / float(max(1, len(body_pts) - 1))
        if i == 0:
            tan = (body_pts[1] - body_pts[0]).normalized()
        elif i == len(body_pts) - 1:
            tan = (pts[-1] - body_pts[-1]).normalized()
        else:
            tan = (body_pts[i + 1] - body_pts[i - 1]).normalized()
        # Orient ribbon face roughly toward -Y camera so it reads wide & crisp!
        cam_hint = Vector((0.0, -1.0, 0.15)).normalized()
        side = tan.cross(cam_hint)
        if side.length < 1e-4:
            side = Vector((1.0, 0.0, 0.0))
        else:
            side.normalize()
        hw = 0.5 * width * (0.45 + 0.55 * math.sin(math.pi * min(1.0, u * 0.85 + 0.15)))
        left_v.append(bm.verts.new(p - side * hw))
        right_v.append(bm.verts.new(p + side * hw))

    for i in range(len(body_pts) - 1):
        bm.faces.new([left_v[i], right_v[i], right_v[i + 1], left_v[i + 1]])

    # Sculpted Arrowhead Triangle at tip!
    p_base = body_pts[-1]
    tip_dir = (pts[-1] - p_base).normalized()
    cam_hint = Vector((0.0, -1.0, 0.15)).normalized()
    side = tip_dir.cross(cam_hint)
    side = side.normalized() if side.length > 1e-4 else Vector((1.0, 0.0, 0.0))
    v_al = bm.verts.new(p_base - side * (0.5 * head_w))
    v_ar = bm.verts.new(p_base + side * (0.5 * head_w))
    v_tip = bm.verts.new(p_base + tip_dir * head_len)
    bm.faces.new([v_al, v_ar, v_tip])


def _add_ring_mesh(bm: bmesh.types.BMesh, radius: float, tube_r: float, z_off: float = 0.0, n_seg: int = 36) -> None:
    pts = [
        Vector((math.cos(2.0 * math.pi * (i / float(n_seg))) * radius,
                math.sin(2.0 * math.pi * (i / float(n_seg))) * radius,
                z_off))
        for i in range(n_seg + 1)
    ]
    _add_open_ribbon_tube(bm, pts, tube_r, tube_r, segs=8)


def build_heli_fx(
    col: bpy.types.Collection,
    heli_root: bpy.types.Object,
    maple_root: bpy.types.Object,
    total_frames: int = C.TOTAL_FRAMES,
) -> dict[str, Any]:
    """Build all V2 Ultimate volumetric FX (Beat 1A Fire/Smoke, Beat 2 Stall Halo, Beat 3 Ribbon Arrows, Beat 4/5B/6)."""
    mat_fire = _make_turbulent_fireball_mat("Mat_FX_TurbineFireball")
    mat_spark = _make_additive_ribbon_mat("Mat_FX_TurbineSparks", (1.00, 0.72, 0.18), strength=1.88)
    mat_smoke = _make_dark_smoke_mat("Mat_FX_DarkEngineSmoke", opacity=0.72)
    mat_emerald = _make_additive_ribbon_mat("Mat_FX_EmeraldAirflow", (0.12, 0.98, 0.68), strength=1.72)
    mat_gold = _make_additive_ribbon_mat("Mat_FX_KineticGold", (1.00, 0.78, 0.22), strength=1.68)
    mat_crimson = _make_additive_ribbon_mat("Mat_FX_SeizeCrimson", (1.00, 0.20, 0.10), strength=1.85)
    mat_cyan = _make_additive_ribbon_mat("Mat_FX_FlareCyan", (0.22, 0.90, 1.00), strength=1.62)
    mat_dust = _make_additive_ribbon_mat("Mat_FX_GoldenDustWash", (1.00, 0.78, 0.44), strength=1.18)

    fx: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. BEAT 1A (18..115): EXPLOSIVE TURBINE FIREBALL + SPARKS + BILLOWING BLACK SMOKE PLUME!
    # ------------------------------------------------------------------
    # A. Procedural 3D Displaced Turbulent Fireball erupting from dorsal turbine exhausts!
    bm_fb = bmesh.new()
    bmesh.ops.create_icosphere(bm_fb, subdivisions=3, radius=0.22)
    for v in bm_fb.verts:
        p = v.co
        disp = 1.0 + 0.30 * math.sin(p.x * 18.0 + 0.7) * math.cos(p.y * 16.0) + 0.22 * math.sin(p.z * 21.0)
        v.co = p * disp
    fire_obj = _new_fx_obj("FX_B1A_TurbineFireball", bm_fb, mat_fire, col, parent=heli_root, location=C.HELI_EXHAUST_LOCAL_POS)
    for f in sorted(set(range(18, 86, 2)) | {18, 22, 28, 45, 85}):
        dt = float(f - 18)
        grow = C.smootherstep(18.0, 26.0, float(f))
        fade = 1.0 - C.smoothstep(34.0, 82.0, float(f))
        sc = max(0.001, (0.35 + 1.45 * grow) * (0.25 + 0.75 * fade))
        fire_obj.scale = Vector((sc * 1.15, sc * 1.55, sc))
        fire_obj.rotation_euler = Euler((dt * 0.14, dt * 0.18, dt * 0.11), "XYZ")
        fire_obj.keyframe_insert("scale", frame=f)
        fire_obj.keyframe_insert("rotation_euler", frame=f)
    _key_visibility([fire_obj], [(18, 85)], total_frames)
    fx["b1a_fireball"] = fire_obj

    # B. Curved 3D Tapered Metallic Spark Streams blasting aft from the blown turbine!
    bm_sp = bmesh.new()
    for k in range(10):
        ang = (k / 10.0) * 2.0 * math.pi
        pts_s = []
        for s in range(8):
            u = s / 7.0
            sx = math.cos(ang) * (0.06 + 0.42 * u)
            sy = 0.05 + 0.95 * u
            sz = math.sin(ang) * (0.05 + 0.28 * u) - 0.18 * (u * u)
            pts_s.append(Vector((sx, sy, sz)))
        _add_open_ribbon_tube(bm_sp, pts_s, 0.011, 0.002, segs=6)
    sparks_obj = _new_fx_obj("FX_B1A_TurbineSparks", bm_sp, mat_spark, col, parent=heli_root, location=C.HELI_EXHAUST_LOCAL_POS)
    for f in (18, 26, 40, 62, 85):
        grow = C.smootherstep(18.0, 32.0, float(f))
        fade = 1.0 - C.smoothstep(44.0, 85.0, float(f))
        sc = max(0.001, grow * fade)
        sparks_obj.scale = Vector((sc, sc * 1.25, sc))
        sparks_obj.keyframe_insert("scale", frame=f)
    _key_visibility([sparks_obj], [(18, 85)], total_frames)
    fx["b1a_sparks"] = sparks_obj

    # C. Billowing Dark Charcoal Exhaust Smoke Plume (7 streaming puffs trailing aft & rising!)
    smoke_objs: list[bpy.types.Object] = []
    for k in range(7):
        bm_sm = bmesh.new()
        bmesh.ops.create_icosphere(bm_sm, subdivisions=2, radius=0.14 + 0.025 * k)
        for v in bm_sm.verts:
            p = v.co
            v.co = p * (1.0 + 0.22 * math.sin(p.x * 12.0 + k) * math.cos(p.z * 11.0))
        sm_o = _new_fx_obj(f"FX_B1A_EngineSmoke_{k:02d}", bm_sm, mat_smoke, col, parent=heli_root, location=C.HELI_EXHAUST_LOCAL_POS)
        for f in range(18, C.B1_EXTERIOR_END + 1, 3):
            phase = ((f - 18) * 0.045 + k / 7.0) % 1.0
            ex, ey, ez = C.HELI_EXHAUST_LOCAL_POS
            sm_o.location = Vector((
                ex + 0.14 * math.sin(phase * 5.0 + k),
                ey + 0.08 + phase * 1.55,
                ez + 0.04 + phase * 0.42,
            ))
            sc = max(0.01, (0.35 + 1.45 * phase) * C.smoothstep(18.0, 28.0, float(f)) * (1.0 - 0.35 * phase))
            sm_o.scale = Vector((sc, sc * 1.2, sc))
            sm_o.keyframe_insert("location", frame=f)
            sm_o.keyframe_insert("scale", frame=f)
        smoke_objs.append(sm_o)
    _key_visibility(smoke_objs, [(18, C.B1_EXTERIOR_END)], total_frames)
    fx["b1a_smoke"] = smoke_objs

    # ------------------------------------------------------------------
    # 2. BEAT 2 (345..558): SEIZED TURBINE CRIMSON STALL HALO
    # ------------------------------------------------------------------
    bm_sz = bmesh.new()
    _add_ring_mesh(bm_sz, 0.125, 0.013, 0.0, 28)
    _add_ring_mesh(bm_sz, 0.175, 0.008, 0.0, 28)
    seize_obj = _new_fx_obj("FX_B2_TurbineSeizeHalo", bm_sz, mat_crimson, col, parent=heli_root, location=C.HELI_TURBINE_LOCAL_POS)
    seize_obj.rotation_euler = Euler((math.radians(90.0), 0.0, 0.0), "XYZ")
    for f in (342, 355, 405, 535, 558):
        pulse = C.smoothstep(342.0, 355.0, float(f)) * (1.0 - C.smoothstep(535.0, 558.0, float(f)))
        sc = max(0.001, 0.85 + 0.25 * pulse)
        seize_obj.scale = Vector((sc, sc, sc))
        seize_obj.keyframe_insert("scale", frame=f)
    _key_visibility([seize_obj], [(342, 558)], total_frames)
    fx["b2_seize_halo"] = seize_obj

    # ------------------------------------------------------------------
    # 3. BEAT 3 (571..960): 6 CLEAN WIDE 3D UPWARD EMERALD RIBBON ARROWS (ZERO SPAGHETTI!)
    # ------------------------------------------------------------------
    airflow_objs: list[bpy.types.Object] = []
    # Organized symmetric columns left & right beneath the rotor blades (never tangling over the fuselage!)
    arrow_cols = [
        (-0.92, -0.10), (-0.64, 0.20), (-0.36, -0.26),
        (+0.36, -0.26), (+0.64, 0.20), (+0.92, -0.10),
    ]
    for idx, (ax, ay) in enumerate(arrow_cols):
        bm_ar = bmesh.new()
        pts_a: list[Vector] = []
        n_s = 12
        for s in range(n_s):
            u = s / float(n_s - 1)
            z = -0.82 + u * 1.48
            # Gentle aerodynamic curve bending cleanly into the spinning rotor disc!
            swirl_x = ax + 0.12 * math.sin(u * math.pi) * (1.0 if ax >= 0 else -1.0)
            swirl_y = ay - 0.10 * (u * u)
            pts_a.append(Vector((swirl_x, swirl_y, z)))
        _add_wide_3d_ribbon_arrow(bm_ar, pts_a, width=0.058, head_len=0.17, head_w=0.145)
        ar_obj = _new_fx_obj(f"FX_B3_UpwardRibbonArrow_{idx:02d}", bm_ar, mat_emerald, col, parent=heli_root)
        for f in range(571, 961, 4):
            phase = ((f - 571) * 0.022 + (idx % 3) * 0.33) % 1.0
            ar_obj.location = Vector((0.0, 0.0, -0.26 + phase * 0.56))
            vis_env = math.sin(math.pi * phase) ** 0.45
            sc = max(0.01, 0.72 + 0.32 * vis_env)
            ar_obj.scale = Vector((sc, sc, sc))
            ar_obj.keyframe_insert("location", frame=f)
            ar_obj.keyframe_insert("scale", frame=f)
        airflow_objs.append(ar_obj)
    _key_visibility(airflow_objs, [(571, 960)], total_frames)
    fx["b3_airflow_arrows"] = airflow_objs

    # Beat 3 Kinetic Flywheel Ring around rotor perimeter (charges up smoothly f=735..955!)
    bm_fw = bmesh.new()
    _add_ring_mesh(bm_fw, C.HELI_ROTOR_RADIUS * 0.97, 0.014, 0.0, 44)
    fw_obj = _new_fx_obj("FX_B3_FlywheelRing", bm_fw, mat_gold, col, parent=heli_root, location=(0.0, 0.0, C.HELI_ROTOR_HUB_LOCAL_Z + 0.015))
    for f in sorted(set(range(720, 961, 8)) | {720, 860, 960}):
        pop = C.smootherstep(720.0, 810.0, float(f)) * (1.0 - C.smoothstep(938.0, 960.0, float(f)))
        fw_obj.scale = Vector((max(0.001, pop), max(0.001, pop), max(0.001, pop)))
        fw_obj.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f)) * 0.5), "XYZ")
        fw_obj.keyframe_insert("scale", frame=f)
        fw_obj.keyframe_insert("rotation_euler", frame=f)
    _key_visibility([fw_obj], [(720, 960)], total_frames)
    fx["b3_flywheel_ring"] = fw_obj

    # ------------------------------------------------------------------
    # 4. BEAT 4 (961..1290): CLEAN MATCHING MAPLE-SEED & RESCUE-ROTOR LIFT SWIRLS!
    # ------------------------------------------------------------------
    bm_mv = bmesh.new()
    pts_mv = [
        Vector((math.cos(0.6 + (s / 24.0) * 3.8 * math.pi) * 0.58,
                math.sin(0.6 + (s / 24.0) * 3.8 * math.pi) * 0.58,
                -0.28 + (s / 24.0) * 0.58))
        for s in range(25)
    ]
    _add_open_ribbon_tube(bm_mv, pts_mv, 0.007, 0.013, segs=6)
    mv_obj = _new_fx_obj("FX_B4_MapleLiftSwirl", bm_mv, mat_gold, col, parent=maple_root)
    _key_visibility([mv_obj], [(961, 1290)], total_frames)
    fx["b4_maple_vortex"] = mv_obj

    bm_hv = bmesh.new()
    pts_hv = [
        Vector((math.cos(0.4 + (s / 28.0) * 3.4 * math.pi) * 1.16,
                math.sin(0.4 + (s / 28.0) * 3.4 * math.pi) * 1.16,
                -0.14 + (s / 28.0) * 0.62))
        for s in range(29)
    ]
    _add_open_ribbon_tube(bm_hv, pts_hv, 0.009, 0.016, segs=6)
    hv_obj = _new_fx_obj("FX_B4_HeliLiftSwirl", bm_hv, mat_emerald, col, parent=heli_root)
    for f in (961, 1125, 1290):
        hv_obj.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f)) * 0.30), "XYZ")
        hv_obj.keyframe_insert("rotation_euler", frame=f)
    _key_visibility([hv_obj], [(961, 1290)], total_frames)
    fx["b4_heli_vortex"] = hv_obj

    # ------------------------------------------------------------------
    # 5. BEAT 5B (1462..1580): 30-FOOT FLARE DOWNWASH AIR-CUSHION SHOCKWAVE RING
    # ------------------------------------------------------------------
    bm_fl = bmesh.new()
    _add_ring_mesh(bm_fl, 1.06, 0.022, 0.0, 44)
    _add_ring_mesh(bm_fl, 0.72, 0.014, -0.16, 36)
    flare_obj = _new_fx_obj("FX_B5B_FlareCushionRing", bm_fl, mat_cyan, col, parent=heli_root, location=(0.0, 0.0, -0.20))
    for f in sorted(set(range(1462, 1581, 4)) | {1462, 1522, 1580}):
        u = C.smootherstep(1462.0, 1565.0, float(f))
        sc = max(0.01, 0.55 + 0.80 * u)
        flare_obj.location = Vector((0.0, 0.0, C.lerp(-0.14, -0.58, u)))
        flare_obj.scale = Vector((sc, sc, max(0.01, 1.0 - 0.30 * u)))
        flare_obj.keyframe_insert("location", frame=f)
        flare_obj.keyframe_insert("scale", frame=f)
    _key_visibility([flare_obj], [(1462, 1580)], total_frames)
    fx["b5b_flare_cushion"] = flare_obj

    # ------------------------------------------------------------------
    # 6. BEAT 6 (1630..1785): EXPANDING GOLDEN GROUND-EFFECT ROOFTOP DUST WAVE!
    # ------------------------------------------------------------------
    hx, hy, hz = C.STAGE["helipad_center"]
    bm_td = bmesh.new()
    _add_ring_mesh(bm_td, 1.15, 0.024, 0.04, 48)
    _add_ring_mesh(bm_td, 1.55, 0.014, 0.06, 48)
    td_obj = _new_fx_obj("FX_B6_TouchdownDustWave", bm_td, mat_dust, col, parent=None, location=(hx, hy, hz + 0.02))
    for f in sorted(set(range(1630, 1786, 4)) | {1630, 1685, 1785}):
        u = C.smootherstep(1630.0, 1775.0, float(f))
        sc = max(0.01, 0.65 + 1.15 * u)
        fade = 1.0 - C.smoothstep(1715.0, 1785.0, float(f))
        td_obj.scale = Vector((sc, sc, max(0.001, fade)))
        td_obj.keyframe_insert("scale", frame=f)
    _key_visibility([td_obj], [(1630, 1785)], total_frames)
    fx["b6_touchdown_wash"] = td_obj

    return fx
