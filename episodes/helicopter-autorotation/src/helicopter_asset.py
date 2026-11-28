"""
helicopter_asset.py — Modern Twin-Engine Rescue Helicopter (Airbus H145 / Coast Guard Dauphin Silhouette),
Precision Macro X-Ray Sprag Clutch with 12 Chamfered Cam Rollers & Lock/Spin Indicators,
Motion-Blur Rotor Disc + Aerodynamic Flare Blade Coning, and Hero Botanical 3D Maple Seed (Samara)
for Episode #5 V2 Ultimate ("Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock", 60 fps).

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
  - Visible in Beat 1A (1..115), Beats 2..4 (241..1290), and Beats 5B..6 (1451..1800)!
  - Lofted multi-section modern rescue fuselage (NOT a stretched sphere!): sculpted nose radome,
    chin observation windows, upper/lower wire-strike cutters, recessed framed pilot & sliding rescue
    doors, twin dorsal turbine nacelles with meshed intakes & heat-blued exhaust nozzles, belly searchlight,
    side rescue hoist, stepped skids strictly flush at local Z = -0.340 m (`0.000 mm` flush helipad contact!),
    tapered octagonal tail boom, horizontal airfoil stabilizers with endplates, and shrouded Fenestron tail!
  - Local-Space Hollow Fresnel X-Ray Cowling (`ShaderNodeTexCoord.Object` bound to `Heli_Root`) exposing:
      * Glowing crimson-hot **Engine Turbine Shaft & Helical Bevel Gear** locking dead at f=345 + crimson stall halo
      * **Machined Steel Inner/Outer Races** + **12 Large Chamfered Golden Cam Sprags + Retaining Springs**
        visibly tilting inward to disengage at f=388..432!
      * **Gleaming Gold-Cyan Rotor Mast Drive Ring** + curved spinning motion-arc arrows freewheeling smoothly!
  - Aerodynamic **4-Blade Modern Rescue Main Rotor** (with subtle `+4.2 deg` high-G upward blade coning in Beat 5B,
    dynamic `-4 -> +14 deg` blade pitch, red/white strobe tip stripes, and translucent radial motion-blur rotor disc!).
  - Enlarged, high-detail **3D Botanical Winged Maple Samara** (`Heli_MapleSeed_Root`, Beat 4) with textured
    ribbed brown-amber seed nut + thin veined translucent golden-amber airfoil wing!
"""
from __future__ import annotations

import math
from typing import Any

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Matrix, Vector  # type: ignore[import-not-found]

import heli_contract as C

_HELI_EXTERIOR_WINDOWS: list[tuple[int, int]] = [
    (1, C.B1_EXTERIOR_END),
    (241, 1290),
    (C.B5_EXTERIOR_START, C.TOTAL_FRAMES),
]


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
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smooth: bool = True,
    no_shadow: bool = False,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    # Blender 5.3.0 rule: append ALL materials BEFORE bm.to_mesh(mesh)!
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
    obj.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
    if parent is not None:
        obj.parent = parent
    if no_shadow:
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


def _make_pbr(
    name: str,
    rgb: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.35,
    coat: float = 0.0,
    emit_rgb: tuple[float, float, float] | None = None,
    emit_str: float = 0.0,
    bump_scale: float = 0.0,
    bump_str: float = 0.04,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if coat > 0.0 and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.09
    if emit_rgb is not None and emit_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    if bump_scale > 0.0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = bump_scale
        noise.inputs["Detail"].default_value = 5.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_str
        bump.inputs["Distance"].default_value = 0.004
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_canopy_glass_mat() -> bpy.types.Material:
    """Dark aviation tinted panoramic windshield & chin glass with mandatory Shadow-Ray transparent bypass."""
    mat = bpy.data.materials.new("Mat_Heli_CanopyGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.04, 0.09, 0.14, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["IOR"].default_value = 1.50
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.68
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.90
        bsdf.inputs["Coat Roughness"].default_value = 0.02

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_cowling_xray_mat(heli_root: bpy.types.Object, total_frames: int) -> bpy.types.Material:
    """
    Local-Space Hollow Fresnel X-Ray Dorsal Cowling Shader (`AGENTS.md` V12 Section 3):
      - Driven strictly by `ShaderNodeTexCoord.Object` bound to `heli_root` (100% local space!)
      - Transitions smoothly from satin crimson rescue turbine housing (`xray=0`) to a hollow
        cyan-blue rim-only X-ray shell (`AddShader(Transparent, Emission * CameraRay * (1-Backfacing) * pow(Facing, 3.8))`)
        with ZERO opaque center fill + Shadow-Ray bypass so the 12 golden cam sprags shine vividly!
    """
    mat = bpy.data.materials.new("Mat_Heli_CowlingXRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.72, 0.045, 0.038, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.24
    bsdf.inputs["Roughness"].default_value = 0.28
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.45
        bsdf.inputs["Coat Roughness"].default_value = 0.08

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = heli_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    val_xray = nt.nodes.new("ShaderNodeValue")
    val_xray.name = "CowlingXRayFac"
    for f in (1, 240, 258, 285, 315, 340, 535, 555, 570, 571, total_frames):
        val_xray.outputs[0].default_value = C.cowling_xray_t(float(f))
        val_xray.outputs[0].keyframe_insert("default_value", frame=f)

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42
    pow_f = nt.nodes.new("ShaderNodeMath")
    pow_f.operation = "POWER"
    pow_f.inputs[1].default_value = 3.8
    nt.links.new(lw.outputs["Facing"], pow_f.inputs[0])

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    inv_bf = nt.nodes.new("ShaderNodeMath")
    inv_bf.operation = "SUBTRACT"
    inv_bf.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], inv_bf.inputs[1])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul1 = nt.nodes.new("ShaderNodeMath")
    mul1.operation = "MULTIPLY"
    nt.links.new(pow_f.outputs["Value"], mul1.inputs[0])
    nt.links.new(inv_bf.outputs["Value"], mul1.inputs[1])

    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    nt.links.new(mul1.outputs["Value"], mul2.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mul2.inputs[1])

    mul3 = nt.nodes.new("ShaderNodeMath")
    mul3.operation = "MULTIPLY"
    mul3.inputs[1].default_value = 1.50
    nt.links.new(mul2.outputs["Value"], mul3.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.14, 0.88, 1.00, 1.0)
    nt.links.new(mul3.outputs["Value"], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_xray = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_xray.inputs[0])
    nt.links.new(emit.outputs["Emission"], add_xray.inputs[1])

    mix_mode = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(val_xray.outputs[0], mix_mode.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_mode.inputs[1])
    nt.links.new(add_xray.outputs["Shader"], mix_mode.inputs[2])

    mul_sh = nt.nodes.new("ShaderNodeMath")
    mul_sh.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Shadow Ray"], mul_sh.inputs[0])
    nt.links.new(val_xray.outputs[0], mul_sh.inputs[1])

    mix_final = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul_sh.outputs["Value"], mix_final.inputs["Fac"])
    nt.links.new(mix_mode.outputs["Shader"], mix_final.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_final.inputs[2])
    nt.links.new(mix_final.outputs["Shader"], out.inputs["Surface"])
    return mat


def _make_rotor_blur_disc_mat() -> bpy.types.Material:
    """Subtle radial motion-blur rotor disc (camera-ray additive transparency with red/white tip strobe ring)."""
    mat = bpy.data.materials.new("Mat_Heli_RotorBlurDisc")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    len_n = nt.nodes.new("ShaderNodeVectorMath")
    len_n.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], len_n.inputs[0])

    # Radial mask: faint graphite sweep mid-disc + crisp glowing red/white strobe tip ring near r = 1.15..1.22
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.08
    cr.elements[0].color = (0.04, 0.045, 0.055, 1.0)
    cr.elements[1].position = 0.86
    cr.elements[1].color = (0.12, 0.14, 0.16, 1.0)
    e2 = cr.elements.new(0.93)
    e2.color = (0.95, 0.22, 0.14, 1.0)
    e3 = cr.elements.new(0.97)
    e3.color = (0.98, 0.98, 0.99, 1.0)

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = C.HELI_ROTOR_RADIUS
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    nt.links.new(len_n.outputs["Value"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])

    lp = nt.nodes.new("ShaderNodeLightPath")
    mul_a = nt.nodes.new("ShaderNodeMath")
    mul_a.operation = "MULTIPLY"
    mul_a.inputs[1].default_value = 0.18
    nt.links.new(lp.outputs["Is Camera Ray"], mul_a.inputs[0])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Emission Strength"].default_value = 0.45
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul_a.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _add_tube_bm(
    bm: bmesh.types.BMesh,
    p0: Vector,
    p1: Vector,
    r0: float,
    r1: float | None = None,
    mat_idx: int = 0,
    segs: int = 12,
    cap_ends: bool = True,
) -> None:
    if r1 is None:
        r1 = r0
    seg = p1 - p0
    L = seg.length
    if L < 1e-5:
        return
    mid = 0.5 * (p0 + p1)
    rm = seg.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    res = bmesh.ops.create_cone(
        bm, cap_ends=cap_ends, cap_tris=False, segments=segs,
        radius1=r0, radius2=r1, depth=L,
        matrix=Matrix.Translation(mid) @ rm,
    )
    vs = set(res["verts"])
    for f in bm.faces:
        if all(v in vs for v in f.verts):
            f.material_index = mat_idx


def _add_box_bm(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat_idx: int = 0,
    bevel_w: float = 0.004,
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    if bevel_w > 0.0:
        bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel_w, segments=2, affect="EDGES", clamp_overlap=True)
    rm = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ").to_matrix().to_4x4()
    bmesh.ops.transform(tmp, matrix=Matrix.Translation(Vector(center)) @ rm, verts=tmp.verts)
    vmap = {v: bm.verts.new(v.co) for v in tmp.verts}
    for f in tmp.faces:
        nf = bm.faces.new([vmap[v] for v in f.verts])
        nf.material_index = mat_idx
    tmp.free()


def _build_lofted_fuselage(
    bm: bmesh.types.BMesh,
    sections: list[tuple[float, float, float, float, float]],
    n_ring: int = 24,
) -> None:
    """
    Build a sleek modern lofted aerospace fuselage along Y (nose = -Y, tail = +Y).
    Each section: `(y, half_width_x, z_bot, z_top, squareness_p)`.
    Uses superellipse profiles (`p=2.0` round nose -> `p=3.2` flat-sided Airbus H145 cabin!)
    and automatically assigns clean two-tone rescue livery (red nose/dorsal + satin white fuselage + dark belly).
    """
    rings: list[list[Any]] = []
    for y, wx, zb, zt, sq in sections:
        zc = 0.5 * (zb + zt)
        hz = 0.5 * (zt - zb)
        ring = []
        for i in range(n_ring):
            ang = 2.0 * math.pi * (i / float(n_ring))
            ca, sa = math.cos(ang), math.sin(ang)
            # Superellipse shaping for realistic flat cabin sides + rounded upper/lower chines
            exp = 2.0 / max(1.6, sq)
            sx = math.copysign(abs(ca) ** exp, ca) * wx
            sz = math.copysign(abs(sa) ** exp, sa) * hz
            ring.append(bm.verts.new((sx, y, zc + sz)))
        rings.append(ring)

    # Nose tip cap & rear cap
    v_nose = bm.verts.new((0.0, sections[0][0] - 0.035, 0.5 * (sections[0][2] + sections[0][3])))
    v_tail = bm.verts.new((0.0, sections[-1][0] + 0.04, 0.5 * (sections[-1][2] + sections[-1][3])))

    for i in range(n_ring):
        ni = (i + 1) % n_ring
        fn = bm.faces.new([v_nose, rings[0][ni], rings[0][i]])
        fn.material_index = 2  # charcoal radome tip
        ft = bm.faces.new([v_tail, rings[-1][i], rings[-1][ni]])
        ft.material_index = 1  # crimson tail boom

    for r0, r1 in zip(rings, rings[1:]):
        for i in range(n_ring):
            ni = (i + 1) % n_ring
            f = bm.faces.new([r0[i], r0[ni], r1[ni], r1[i]])
            fc = f.calc_center_median()
            # Two-tone modern Coast Guard / H145 rescue livery:
            #   mat 0 = Satin White, mat 1 = Rescue Crimson, mat 2 = Charcoal Alloy, mat 3 = Tinted Canopy/Window Glass
            if fc.y < -0.78:
                f.material_index = 2 if fc.y < -0.92 else 1
            elif -0.78 <= fc.y <= -0.34 and fc.z > 0.04:
                # Panoramic sculpted forward windshield glass
                f.material_index = 3
            elif -0.82 <= fc.y <= -0.56 and fc.z < -0.04 and abs(fc.x) > 0.06:
                # Lower chin observation windows
                f.material_index = 3
            elif -0.26 <= fc.y <= 0.36 and abs(fc.x) > 0.24 and 0.03 <= fc.z <= 0.19:
                # Recessed side cockpit & rescue cabin windows
                f.material_index = 3
            elif fc.z < -0.135:
                # Charcoal lower belly fairing
                f.material_index = 2
            elif fc.y > 0.56 or fc.z > 0.195 or (-0.18 <= fc.y <= 0.50 and -0.06 <= fc.z <= 0.00):
                # High-visibility crimson rescue nose/stripe/tail livery!
                f.material_index = 1
            else:
                f.material_index = 0


def _build_airframe_and_skids(
    heli_root: bpy.types.Object,
    col: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> list[bpy.types.Object]:
    """Modern Twin-Engine Rescue Helicopter (Airbus H145 / Coast Guard Dauphin silhouette)."""
    bm = bmesh.new()

    # 1. Lofted multi-section fuselage (nose = -0.95 m -> rear cabin = +0.62 m -> tapered tail boom = +1.46 m)
    sections = [
        (-0.95, 0.06, -0.09, -0.02, 2.0),   # nose radome tip
        (-0.84, 0.18, -0.14, 0.03, 2.2),    # nose radar fairing
        (-0.68, 0.27, -0.17, 0.12, 2.4),    # sloped windshield base
        (-0.46, 0.33, -0.18, 0.22, 2.8),    # forward cockpit brow
        (-0.22, 0.35, -0.18, 0.25, 3.2),    # pilot doors (flat cabin sides!)
        (0.10, 0.35, -0.18, 0.25, 3.2),     # sliding rescue cabin doors
        (0.40, 0.31, -0.16, 0.24, 3.0),     # rear aft bulkhead
        (0.62, 0.20, -0.09, 0.22, 2.6),     # clamshell rear ramp taper
        (0.84, 0.11, 0.03, 0.18, 2.2),      # tail boom root
        (1.16, 0.075, 0.07, 0.165, 2.2),    # mid tail boom
        (1.44, 0.055, 0.095, 0.160, 2.2),   # Fenestron tail root
    ]
    _build_lofted_fuselage(bm, sections, n_ring=24)

    # 2. Crisp Cockpit Window Mullions, Door Frames & Grab Handles (adds instant mechanical realism!)
    for sx in (-1.0, +1.0):
        # Side windshield frame struts & door seam moldings
        _add_tube_bm(bm, Vector((sx * 0.27, -0.66, 0.04)), Vector((sx * 0.32, -0.40, 0.21)), 0.011, 0.011, mat_idx=2, segs=8)
        _add_tube_bm(bm, Vector((sx * 0.355, -0.24, -0.14)), Vector((sx * 0.355, -0.24, 0.21)), 0.008, 0.008, mat_idx=2, segs=8)
        _add_tube_bm(bm, Vector((sx * 0.355, 0.36, -0.14)), Vector((sx * 0.355, 0.36, 0.20)), 0.008, 0.008, mat_idx=2, segs=8)
        # Sliding rescue door rail track
        _add_box_bm(bm, (sx * 0.362, 0.22, 0.00), (0.014, 0.42, 0.016), mat_idx=4, bevel_w=0.002)
        # Boarding foot step on side fuselage
        _add_box_bm(bm, (sx * 0.34, 0.05, -0.22), (0.08, 0.34, 0.014), mat_idx=2, bevel_w=0.002)
    # Center slim windshield spine + twin windshield wiper arms
    _add_tube_bm(bm, Vector((0.0, -0.74, 0.03)), Vector((0.0, -0.42, 0.225)), 0.010, 0.010, mat_idx=2, segs=8)
    for sx in (-0.10, +0.10):
        _add_tube_bm(bm, Vector((sx, -0.68, 0.05)), Vector((sx * 1.4, -0.54, 0.14)), 0.004, 0.004, mat_idx=2, segs=6)

    # 3. Upper & Lower Wire-Strike Cable Cutters + Pitot Tubes + Belly Searchlight + Rescue Winch Hoist!
    _add_box_bm(bm, (0.0, -0.52, 0.27), (0.010, 0.14, 0.085), mat_idx=2, bevel_w=0.002, rot_deg=(-28.0, 0.0, 0.0))
    _add_box_bm(bm, (0.0, -0.64, -0.21), (0.010, 0.12, 0.075), mat_idx=2, bevel_w=0.002, rot_deg=(26.0, 0.0, 0.0))
    # Nose landing searchlight pod + glowing warm lens (mat_idx=6)
    _add_box_bm(bm, (0.0, -0.72, -0.185), (0.09, 0.11, 0.055), mat_idx=2, bevel_w=0.008)
    _add_tube_bm(bm, Vector((0.0, -0.77, -0.185)), Vector((0.0, -0.785, -0.185)), 0.024, 0.024, mat_idx=6, segs=12)
    # Starboard (+X) rescue winch hoist pod above sliding cabin door!
    _add_tube_bm(bm, Vector((0.40, 0.02, 0.24)), Vector((0.40, 0.22, 0.24)), 0.044, 0.040, mat_idx=4, segs=14)
    _add_box_bm(bm, (0.37, 0.10, 0.22), (0.10, 0.06, 0.04), mat_idx=2, bevel_w=0.004)
    # Belly weather radar dome + VHF whip antennas
    _add_tube_bm(bm, Vector((0.0, 0.88, 0.04)), Vector((0.0, 0.94, -0.09)), 0.004, 0.002, mat_idx=2, segs=6)
    _add_tube_bm(bm, Vector((0.0, 1.08, 0.15)), Vector((0.0, 1.14, 0.28)), 0.004, 0.002, mat_idx=2, segs=6)

    # 4. Twin Sculpted Side Turbine Air Intakes & Heat-Blued Dual Exhaust Nozzles (mat_idx=5)
    for sx in (-1.0, +1.0):
        # Forward intake scoop mesh bezel
        _add_box_bm(bm, (sx * 0.17, -0.14, 0.29), (0.11, 0.16, 0.095), mat_idx=2, bevel_w=0.012)
        # Rear angled titanium heat-blued turbine exhaust pipes (where Beat 1A fire & black smoke erupt!)
        _add_tube_bm(
            bm,
            Vector((sx * 0.14, 0.34, 0.265)),
            Vector((sx * 0.19, 0.50, 0.245)),
            0.048, 0.042, mat_idx=5, segs=14,
        )

    # 5. Horizontal Airfoil Stabilizers with Angled Endplates + Nav Lights (red port / green starboard)
    _add_box_bm(bm, (0.0, 1.08, 0.115), (0.58, 0.14, 0.018), mat_idx=1, bevel_w=0.005)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sx * 0.295, 1.09, 0.125), (0.014, 0.18, 0.13), mat_idx=0, bevel_w=0.004, rot_deg=(0.0, sx * 6.0, 0.0))
        _add_tube_bm(
            bm, Vector((sx * 0.305, 1.04, 0.15)), Vector((sx * 0.318, 1.04, 0.15)),
            0.012, 0.012, mat_idx=(7 if sx < 0 else 8), segs=10,
        )

    # 6. Modern Shrouded Fenestron Rescue Tail Fin (swept vertical airfoil + circular duct housing!)
    _add_box_bm(bm, (0.0, 1.54, 0.29), (0.046, 0.24, 0.34), mat_idx=1, bevel_w=0.012, rot_deg=(-22.0, 0.0, 0.0))
    _add_box_bm(bm, (0.0, 1.58, 0.44), (0.048, 0.18, 0.048), mat_idx=0, bevel_w=0.008, rot_deg=(-14.0, 0.0, 0.0))
    # Shroud ring (torus-like segmented collar around Fenestron fan at (0, 1.49, 0.175))
    fen_c = Vector((0.0, 1.49, 0.175))
    fen_r = 0.115
    for k in range(16):
        a0 = 2.0 * math.pi * (k / 16.0)
        a1 = 2.0 * math.pi * ((k + 1) / 16.0)
        p0 = fen_c + Vector((0.0, math.cos(a0) * fen_r, math.sin(a0) * fen_r))
        p1 = fen_c + Vector((0.0, math.cos(a1) * fen_r, math.sin(a1) * fen_r))
        _add_tube_bm(bm, p0, p1, 0.022, 0.022, mat_idx=1, segs=8)
    # Tail bumper stinger skid
    _add_tube_bm(bm, Vector((0.0, 1.42, 0.08)), Vector((0.0, 1.55, -0.02)), 0.010, 0.008, mat_idx=2, segs=8)

    # 7. Twin Stepped Steel Landing Skids — BOTTOM strictly at local Z = -0.340 m (`0.000 mm` flush contact!)
    skid_r = 0.020
    skid_center_z = C.HELI_SKID_BOTTOM_LOCAL_Z + skid_r  # -0.320 -> bottom = -0.340 m EXACT!
    for sx in (-0.37, +0.37):
        _add_tube_bm(bm, Vector((sx, -0.66, skid_center_z)), Vector((sx, 0.48, skid_center_z)), skid_r, skid_r, mat_idx=4, segs=12)
        # Curved snow/helipad upturned nose tips
        _add_tube_bm(bm, Vector((sx, -0.66, skid_center_z)), Vector((sx, -0.78, skid_center_z + 0.095)), skid_r, skid_r * 0.82, mat_idx=4, segs=10)
        # Forward & aft aerodynamic arched cross-tube struts
        _add_tube_bm(bm, Vector((sx, -0.34, skid_center_z)), Vector((sx * 0.60, -0.28, -0.14)), 0.015, 0.015, mat_idx=4, segs=10)
        _add_tube_bm(bm, Vector((sx, 0.24, skid_center_z)), Vector((sx * 0.60, 0.22, -0.13)), 0.015, 0.015, mat_idx=4, segs=10)
        # Black rubber skid wear shoes
        _add_box_bm(bm, (sx, -0.40, skid_center_z - 0.004), (0.044, 0.14, 0.028), mat_idx=2, bevel_w=0.004)
        _add_box_bm(bm, (sx, 0.30, skid_center_z - 0.004), (0.044, 0.14, 0.028), mat_idx=2, bevel_w=0.004)

    fuselage_obj = _new_bmesh_obj(
        "Heli_ModernRescue_Fuselage",
        bm,
        [
            mats["white"], mats["red"], mats["charcoal"], mats["canopy"],
            mats["steel"], mats["exhaust"], mats["searchlight"], mats["nav_red"], mats["nav_green"],
        ],
        col,
        parent=heli_root,
        smooth=True,
    )

    # 8. Separate Dorsal Twin-Turbine Cowling Shell (uses Local-Space Hollow Fresnel X-Ray Shader in Beat 2!)
    bm_cowl = bmesh.new()
    _add_box_bm(bm_cowl, (0.0, 0.08, 0.305), (0.38, 0.72, 0.175), mat_idx=0, bevel_w=0.055)
    _add_box_bm(bm_cowl, (0.0, -0.20, 0.285), (0.30, 0.24, 0.135), mat_idx=0, bevel_w=0.042, rot_deg=(16.0, 0.0, 0.0))
    _add_box_bm(bm_cowl, (0.0, 0.42, 0.285), (0.30, 0.26, 0.145), mat_idx=0, bevel_w=0.042, rot_deg=(-14.0, 0.0, 0.0))
    cowling_obj = _new_bmesh_obj(
        "Heli_DorsalCowling_XRay",
        bm_cowl,
        [mats["cowling_xray"]],
        col,
        parent=heli_root,
        smooth=True,
    )

    # 9. Spinning 8-Blade Shrouded Fenestron Tail Fan
    bm_tr = bmesh.new()
    _add_tube_bm(bm_tr, Vector((-0.024, 0.0, 0.0)), Vector((0.024, 0.0, 0.0)), 0.028, 0.028, mat_idx=0, segs=12)
    for k in range(8):
        ang = 2.0 * math.pi * (k / 8.0)
        p1 = Vector((0.0, math.cos(ang) * 0.095, math.sin(ang) * 0.095))
        _add_tube_bm(bm_tr, Vector((0.0, 0.0, 0.0)), p1, 0.011, 0.008, mat_idx=1, segs=6)
    tr_obj = _new_bmesh_obj(
        "Heli_Fenestron_Fan",
        bm_tr,
        [mats["steel"], mats["charcoal"]],
        col,
        parent=heli_root,
        location=(0.0, 1.49, 0.175),
        smooth=True,
    )
    for f in (1, C.TOTAL_FRAMES):
        tr_obj.rotation_euler = Euler((C.rotor_angle_rad_at(float(f)) * 4.2, 0.0, 0.0), "XYZ")
        tr_obj.keyframe_insert("rotation_euler", frame=f)

    return [fuselage_obj, cowling_obj, tr_obj]


def _build_xray_sprag_clutch_and_turbine(
    heli_root: bpy.types.Object,
    col: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Precision Beat 2 Macro X-Ray Mechanical Cutaway (built directly inside the dorsal deck!):
      1. Glowing crimson-hot **Engine Turbine Shaft + Helical Bevel Pinion Gear** seizing dead at f=345
      2. **Crimson Stall-Lock Ring** flashing onto the frozen turbine shaft (`f=345..555`) so viewers
         immediately see "ENGINE LOCKED DEAD!"
      3. **Machined Steel Inner Engine Race** (locks dead with turbine!) + **Gleaming Gold-Cyan Outer Rotor
         Drive Ring** (continues spinning smoothly with the main rotor mast!)
      4. **12 Large, Sharply Chamfered Golden Cam Sprags + Steel Retaining Springs** visibly tilting
         inward (`0 deg -> +36 deg`) at `f=388..432` to disengage like a bicycle freewheel!
      5. **Curved Gold-Cyan Spinning Motion-Arc Arrows** orbiting the freewheeling outer race (`f=405..555`)!
    """
    objs: list[bpy.types.Object] = []

    # 1. Seizing Engine Turbine Shaft + Helical Bevel Gear + Turbine Compressor Wheel (along +Y)
    bm_eng = bmesh.new()
    _add_tube_bm(bm_eng, Vector((0.0, -0.18, 0.0)), Vector((0.0, 0.30, 0.0)), 0.028, 0.028, mat_idx=0, segs=16)
    # Bevel gear + 16 machined teeth engaging clutch inner shaft
    _add_tube_bm(bm_eng, Vector((0.0, -0.16, 0.0)), Vector((0.0, -0.10, 0.0)), 0.068, 0.052, mat_idx=0, segs=18)
    for k in range(12):
        a = 2.0 * math.pi * (k / 12.0)
        _add_box_bm(
            bm_eng, (math.cos(a) * 0.064, -0.13, math.sin(a) * 0.064),
            (0.014, 0.042, 0.018), mat_idx=1, bevel_w=0.002,
            rot_deg=(0.0, -math.degrees(a), 0.0),
        )
    # Rear turbine compressor wheel discs
    for ty in (0.04, 0.14, 0.24):
        _add_tube_bm(bm_eng, Vector((0.0, ty - 0.022, 0.0)), Vector((0.0, ty + 0.022, 0.0)), 0.068, 0.068, mat_idx=1, segs=18)
    eng_obj = _new_bmesh_obj(
        "Heli_Engine_TurbineShaft",
        bm_eng,
        [mats["seized_crimson"], mats["steel"]],
        col,
        parent=heli_root,
        location=C.HELI_TURBINE_LOCAL_POS,
        smooth=True,
    )
    for f in sorted(set(range(241, 571, 4)) | {241, 340, 345, 355, 570}):
        eng_obj.rotation_euler = Euler((0.0, C.engine_shaft_angle_rad_at(float(f)), 0.0), "XYZ")
        eng_obj.keyframe_insert("rotation_euler", frame=f)
    objs.append(eng_obj)

    # 2. Inner Engine-Side Clutch Hub (SEIZES DEAD at f=345 with glowing crimson lock ring!)
    bm_in = bmesh.new()
    _add_tube_bm(bm_in, Vector((0.0, 0.0, -0.042)), Vector((0.0, 0.0, 0.042)), 0.072, 0.072, mat_idx=0, segs=24)
    # 6 crimson spline lock keys so viewer literally SEES the inner hub stop moving!
    for k in range(6):
        a = 2.0 * math.pi * (k / 6.0)
        _add_box_bm(bm_in, (math.cos(a) * 0.056, math.sin(a) * 0.056, 0.042), (0.022, 0.012, 0.014), mat_idx=1, bevel_w=0.002, rot_deg=(0.0, 0.0, math.degrees(a)))
    inner_race = _new_bmesh_obj(
        "Heli_Sprag_InnerRace",
        bm_in,
        [mats["seized_crimson"], mats["warn_red"]],
        col,
        parent=heli_root,
        location=C.HELI_CLUTCH_LOCAL_POS,
        smooth=True,
    )
    for f in sorted(set(range(241, 571, 4)) | {241, 345, 570}):
        inner_race.rotation_euler = Euler((0.0, 0.0, C.engine_shaft_angle_rad_at(float(f))), "XYZ")
        inner_race.keyframe_insert("rotation_euler", frame=f)
    objs.append(inner_race)

    # 3. Outer Freewheeling Rotor Drive Ring (Machined Steel + Gleaming Cyan-Gold Splines — KEEPS SPINNING!)
    bm_out = bmesh.new()
    outer_r = 0.146
    for k in range(28):
        a0 = 2.0 * math.pi * (k / 28.0)
        a1 = 2.0 * math.pi * ((k + 1) / 28.0)
        p0 = Vector((math.cos(a0) * outer_r, math.sin(a0) * outer_r, 0.0))
        p1 = Vector((math.cos(a1) * outer_r, math.sin(a1) * outer_r, 0.0))
        _add_tube_bm(bm_out, p0, p1, 0.022, 0.022, mat_idx=(0 if k % 2 == 0 else 1), segs=8)
    # Top/bottom cyan-gold freewheel accent crown teeth (makes rotation 100% unmistakable!)
    for k in range(8):
        a = 2.0 * math.pi * (k / 8.0)
        _add_box_bm(
            bm_out, (math.cos(a) * (outer_r + 0.016), math.sin(a) * (outer_r + 0.016), 0.012),
            (0.020, 0.014, 0.036), mat_idx=2, bevel_w=0.002, rot_deg=(0.0, 0.0, math.degrees(a)),
        )
    outer_race = _new_bmesh_obj(
        "Heli_Sprag_OuterRace",
        bm_out,
        [mats["clutch_gold"], mats["steel"], mats["cyan_glow"]],
        col,
        parent=heli_root,
        location=C.HELI_CLUTCH_LOCAL_POS,
        smooth=True,
    )
    for f in sorted(set(range(241, 571, 3)) | {241, 570}):
        outer_race.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f))), "XYZ")
        outer_race.keyframe_insert("rotation_euler", frame=f)
    objs.append(outer_race)

    # 4. 12 Large, Sharply Chamfered Golden Cam Sprag Rollers (visibly tilt open at f=388..432!)
    sprag_cage = bpy.data.objects.new("Heli_Sprag_CamCage", None)
    sprag_cage.parent = heli_root
    sprag_cage.location = Vector(C.HELI_CLUTCH_LOCAL_POS)
    _link(sprag_cage, col)
    for f in sorted(set(range(241, 571, 3)) | {241, 570}):
        sprag_cage.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f)) * 0.52), "XYZ")
        sprag_cage.keyframe_insert("rotation_euler", frame=f)

    n_sprags = 12
    for k in range(n_sprags):
        ang = 2.0 * math.pi * (k / float(n_sprags))
        bm_sp = bmesh.new()
        # Asymmetrical figure-8 cam sprag profile (fat upper/lower lobes + bright gold highlight!)
        _add_box_bm(bm_sp, (0.0, 0.0, 0.0), (0.022, 0.044, 0.056), mat_idx=0, bevel_w=0.005)
        _add_tube_bm(bm_sp, Vector((0.0, 0.015, -0.028)), Vector((0.0, 0.015, 0.028)), 0.012, 0.012, mat_idx=1, segs=10)
        _add_tube_bm(bm_sp, Vector((0.0, -0.015, -0.028)), Vector((0.0, -0.015, 0.028)), 0.012, 0.012, mat_idx=1, segs=10)
        sp_obj = _new_bmesh_obj(
            f"Heli_SpragCam_{k:02d}",
            bm_sp,
            [mats["clutch_gold"], mats["steel"]],
            col,
            parent=sprag_cage,
            location=(math.cos(ang) * 0.105, math.sin(ang) * 0.105, 0.0),
            smooth=True,
        )
        base_yaw = math.degrees(ang)
        for f in (241, 382, 398, 412, 428, 445, 570):
            tilt_deg = C.lerp(10.0, 44.0, C.sprag_unlock_t(float(f)))
            sp_obj.rotation_euler = Euler((0.0, 0.0, math.radians(base_yaw + tilt_deg)), "XYZ")
            sp_obj.keyframe_insert("rotation_euler", frame=f)
        objs.append(sp_obj)

    # 5. Instant-Read Curved Spinning Motion-Arc Arrows orbiting the Freewheeling Outer Race!
    bm_arc = bmesh.new()
    arc_r = 0.205
    for k_arc in range(2):
        a_start = k_arc * math.pi + 0.25
        segs = 14
        for s in range(segs):
            a0 = a_start + (s / float(segs)) * 1.85
            a1 = a_start + ((s + 1) / float(segs)) * 1.85
            p0 = Vector((math.cos(a0) * arc_r, math.sin(a0) * arc_r, 0.048))
            p1 = Vector((math.cos(a1) * arc_r, math.sin(a1) * arc_r, 0.048))
            _add_tube_bm(bm_arc, p0, p1, 0.009, 0.009, mat_idx=0, segs=8)
        # Crisp arrowhead tip pointing in direction of freewheel rotation!
        a_tip = a_start + 1.85
        tip_p = Vector((math.cos(a_tip) * arc_r, math.sin(a_tip) * arc_r, 0.048))
        tan_d = Vector((-math.sin(a_tip), math.cos(a_tip), 0.0)).normalized()
        _add_tube_bm(bm_arc, tip_p - tan_d * 0.01, tip_p + tan_d * 0.055, 0.024, 0.002, mat_idx=0, segs=8)
    arc_obj = _new_bmesh_obj(
        "Heli_Sprag_FreewheelArcs",
        bm_arc,
        [mats["cyan_glow"]],
        col,
        parent=heli_root,
        location=C.HELI_CLUTCH_LOCAL_POS,
        smooth=True,
        no_shadow=True,
    )
    for f in sorted(set(range(400, 560, 4)) | {400, 560}):
        pop = C.smootherstep(400.0, 428.0, float(f)) * (1.0 - C.smoothstep(540.0, 560.0, float(f)))
        arc_obj.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f)) * 0.85), "XYZ")
        arc_obj.scale = Vector((max(0.001, pop), max(0.001, pop), max(0.001, pop)))
        arc_obj.keyframe_insert("rotation_euler", frame=f)
        arc_obj.keyframe_insert("scale", frame=f)
    _key_visibility([arc_obj], [(400, 560)], total_frames)

    # Show precision clutch internals strictly during Beat 2 X-ray window (241..570)
    _key_visibility(objs, [(241, 570)], total_frames)
    objs.append(arc_obj)
    return objs


def _build_main_rotor_assembly(
    heli_root: bpy.types.Object,
    col: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
    total_frames: int,
) -> list[bpy.types.Object]:
    """
    Modern 4-Blade Rescue Main Rotor Hub + Animated Swashplate + Dynamic Collective Pitch (-4 -> +14 deg) +
    Aerodynamic High-G Upward Blade Coning (+4.2 deg flare!) + Translucent Radial Motion-Blur Rotor Disc!
    """
    objs: list[bpy.types.Object] = []

    # Stationary mast housing & sliding collective swashplate collar
    bm_sw = bmesh.new()
    _add_tube_bm(bm_sw, Vector((0.0, 0.0, -0.025)), Vector((0.0, 0.0, 0.025)), 0.072, 0.068, mat_idx=0, segs=18)
    _add_tube_bm(bm_sw, Vector((0.0, 0.0, -0.010)), Vector((0.0, 0.0, 0.010)), 0.095, 0.095, mat_idx=1, segs=18)
    swash_obj = _new_bmesh_obj(
        "Heli_Rotor_Swashplate",
        bm_sw,
        [mats["steel"], mats["clutch_gold"]],
        col,
        parent=heli_root,
        location=(0.0, 0.0, C.HELI_ROTOR_HUB_LOCAL_Z - 0.095),
        smooth=True,
    )
    for f in sorted(set(range(1, total_frames + 1, 10)) | {1, 115, 241, 570, 1451, 1580, 1685, total_frames}):
        dz = C.lerp(-0.024, 0.030, C.collective_level_at(float(f)))
        swash_obj.location = Vector((0.0, 0.0, C.HELI_ROTOR_HUB_LOCAL_Z - 0.095 + dz))
        swash_obj.keyframe_insert("location", frame=f)
    objs.append(swash_obj)

    # Spinning Main Rotor Mast & Hub
    rotor_spin = bpy.data.objects.new("Heli_MainRotor_Spin", None)
    rotor_spin.parent = heli_root
    rotor_spin.location = Vector((0.0, 0.0, C.HELI_ROTOR_HUB_LOCAL_Z))
    _link(rotor_spin, col)
    for f in sorted(set(range(1, total_frames + 1, 3)) | {1, total_frames}):
        rotor_spin.rotation_euler = Euler((0.0, 0.0, C.rotor_angle_rad_at(float(f))), "XYZ")
        rotor_spin.keyframe_insert("rotation_euler", frame=f)

    bm_hub = bmesh.new()
    _add_tube_bm(bm_hub, Vector((0.0, 0.0, -0.17)), Vector((0.0, 0.0, 0.035)), 0.036, 0.032, mat_idx=0, segs=16)
    _add_box_bm(bm_hub, (0.0, 0.0, 0.01), (0.24, 0.088, 0.044), mat_idx=0, bevel_w=0.010)
    _add_box_bm(bm_hub, (0.0, 0.0, 0.01), (0.088, 0.24, 0.044), mat_idx=0, bevel_w=0.010)
    _add_tube_bm(bm_hub, Vector((0.0, 0.0, 0.032)), Vector((0.0, 0.0, 0.075)), 0.068, 0.034, mat_idx=1, segs=16)
    hub_obj = _new_bmesh_obj(
        "Heli_MainRotor_Hub",
        bm_hub,
        [mats["steel"], mats["red"]],
        col,
        parent=rotor_spin,
        smooth=True,
    )
    objs.append(hub_obj)

    # 4 Articulated Airfoil Rotor Blades with BOTH Dynamic Pitch (-4 -> +14 deg) AND High-G Coning (+4.2 deg)!
    for idx in range(4):
        az_deg = idx * 90.0
        az_root = bpy.data.objects.new(f"Heli_BladeAz_{idx+1}", None)
        az_root.parent = rotor_spin
        az_root.rotation_euler = Euler((0.0, 0.0, math.radians(az_deg)), "XYZ")
        _link(az_root, col)

        bm_b = bmesh.new()
        r_len = C.HELI_ROTOR_RADIUS - 0.11
        _add_box_bm(bm_b, (0.11 + 0.5 * r_len, 0.0, 0.0), (r_len, 0.076, 0.012), mat_idx=0, bevel_w=0.003)
        # Leading-edge stainless abrasion strip + crisp high-visibility red/white strobe tip stripes!
        _add_box_bm(bm_b, (0.11 + 0.5 * r_len, -0.034, 0.0), (r_len * 0.96, 0.012, 0.013), mat_idx=3, bevel_w=0.002)
        _add_box_bm(bm_b, (C.HELI_ROTOR_RADIUS - 0.13, 0.0, 0.0), (0.085, 0.078, 0.013), mat_idx=2, bevel_w=0.002)
        _add_box_bm(bm_b, (C.HELI_ROTOR_RADIUS - 0.05, 0.0, 0.0), (0.075, 0.078, 0.013), mat_idx=1, bevel_w=0.002)
        b_obj = _new_bmesh_obj(
            f"Heli_MainBlade_{idx+1}",
            bm_b,
            [mats["charcoal"], mats["red"], mats["white"], mats["steel"]],
            col,
            parent=az_root,
            location=(0.0, 0.0, 0.010),
            smooth=True,
        )
        for f in sorted(set(range(1, total_frames + 1, 10)) | {1, 115, 241, 570, 1451, 1525, 1580, 1685, total_frames}):
            pitch_rad = math.radians(C.blade_pitch_deg_at(float(f)))
            cone_rad = -math.radians(C.blade_coning_deg_at(float(f)))  # negative Y rot lifts +X blade tip upward!
            b_obj.rotation_euler = Euler((pitch_rad, cone_rad, 0.0), "XYZ")
            b_obj.keyframe_insert("rotation_euler", frame=f)
        objs.append(b_obj)

    # Translucent Radial Motion-Blur Rotor Disc (hidden in Beat 2 X-ray & fades smoothly after touchdown!)
    bm_disc = bmesh.new()
    bmesh.ops.create_cone(
        bm_disc, cap_ends=True, cap_tris=False, segments=48,
        radius1=C.HELI_ROTOR_RADIUS, radius2=C.HELI_ROTOR_RADIUS * 0.98, depth=0.006,
    )
    for f in bm_disc.faces:
        f.material_index = 0
    disc_obj = _new_bmesh_obj(
        "Heli_Rotor_MotionBlurDisc",
        bm_disc,
        [mats["blur_disc"]],
        col,
        parent=heli_root,
        location=(0.0, 0.0, C.HELI_ROTOR_HUB_LOCAL_Z + 0.025),
        smooth=True,
        no_shadow=True,
    )
    _key_visibility([disc_obj], [(1, C.B1_EXTERIOR_END), (571, 1290), (C.B5_EXTERIOR_START, 1715)], total_frames)
    objs.append(disc_obj)

    return objs


def _build_maple_seed_samara(
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    """
    Beat 4 Enlarged, High-Detail Botanical 3D Winged Maple Seed (Samara):
      - 2.1x hero scale so viewers immediately recognize the ribbed brown-amber seed nut + thin veined
        translucent golden-amber wing blade spinning gracefully beside the modern rescue helicopter!
    """
    mat_nut = _make_pbr("Mat_Maple_SeedNut", (0.36, 0.18, 0.06), roughness=0.46, bump_scale=28.0, bump_str=0.14)
    mat_rib = _make_pbr("Mat_Maple_VeinRib", (0.58, 0.34, 0.10), roughness=0.38, emit_rgb=(0.82, 0.52, 0.14), emit_str=0.22)
    mat_wing = bpy.data.materials.new("Mat_Maple_SamaraWing")
    mat_wing.use_nodes = True
    nt = mat_wing.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.94, 0.68, 0.24, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.30
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.55
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.26
    bsdf.inputs["Emission Color"].default_value = (0.98, 0.72, 0.24, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.28

    seed_root = bpy.data.objects.new("Heli_MapleSeed_Root", None)
    _link(seed_root, col)

    bm = bmesh.new()
    # 1. Sculpted plump ribbed seed nut at the rotational pivot (enlarged hero proportions!)
    bmesh.ops.create_uvsphere(
        bm, u_segments=22, v_segments=14, radius=1.0,
        matrix=Matrix.Translation((0.0, 0.0, 0.0)) @ Matrix.Diagonal((0.115, 0.072, 0.056, 1.0)),
    )
    for f in bm.faces:
        f.material_index = 0

    # 2. Sculpted Botanical Samara Airfoil Wing (curved asymmetric silhouette + raised leading edge & radial veins!)
    n_span = 14
    wing_rings: list[tuple[Any, Any, Any, Any]] = []
    for i in range(n_span + 1):
        u = i / float(n_span)
        x = 0.08 + u * 0.62
        # Realistic curved maple-wing trailing bulge + tapered rounded tip
        chord_f = math.sin(math.pi * (0.14 + 0.84 * u)) ** 0.65
        y_le = -0.048 - 0.025 * math.sin(math.pi * u)
        y_te = 0.045 + 0.165 * chord_f
        z_camber = 0.024 * math.sin(math.pi * u)
        th = 0.007 * (1.0 - 0.55 * u)
        v_le_top = bm.verts.new((x, y_le, z_camber + th))
        v_te_top = bm.verts.new((x, y_te, z_camber * 0.35 + th * 0.4))
        v_te_bot = bm.verts.new((x, y_te, z_camber * 0.35 - th * 0.4))
        v_le_bot = bm.verts.new((x, y_le, z_camber - th))
        wing_rings.append((v_le_top, v_te_top, v_te_bot, v_le_bot))

    for r0, r1 in zip(wing_rings, wing_rings[1:]):
        for side in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a, b = side
            nf = bm.faces.new([r0[a], r0[b], r1[b], r1[a]])
            nf.material_index = 1

    # Raised thick structural leading-edge vein + 5 delicate branching wing veins!
    _add_tube_bm(bm, Vector((0.05, -0.042, 0.008)), Vector((0.67, -0.052, 0.016)), 0.016, 0.005, mat_idx=2, segs=10)
    for k in range(5):
        u = 0.18 + k * 0.15
        x0 = 0.08 + u * 0.56
        p0 = Vector((x0, -0.046, 0.018))
        p1 = Vector((x0 + 0.11, 0.08 + 0.08 * math.sin(math.pi * u), 0.010))
        _add_tube_bm(bm, p0, p1, 0.0045, 0.0022, mat_idx=2, segs=6)

    seed_mesh_obj = _new_bmesh_obj(
        "Heli_MapleSeed_Samara",
        bm,
        [mat_nut, mat_wing, mat_rib],
        col,
        parent=seed_root,
        smooth=True,
    )

    for f in sorted(set(range(961, 1291, 2)) | {961, 1290}):
        pos, rot_deg = C.maple_seed_pose_at(float(f))
        seed_root.location = Vector(pos)
        seed_root.rotation_mode = "XYZ"
        seed_root.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
        seed_root.keyframe_insert("location", frame=f)
        seed_root.keyframe_insert("rotation_euler", frame=f)

    _key_visibility([seed_mesh_obj], [(961, 1290)], total_frames)
    return {"root": seed_root, "mesh": seed_mesh_obj}


def build_helicopter_asset(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Build the V2 modern twin-engine rescue helicopter, X-ray sprag clutch, coning rotor & hero maple seed."""
    heli_root = bpy.data.objects.new("Heli_Root", None)
    heli_root.empty_display_type = "ARROWS"
    heli_root.empty_display_size = 0.35
    _link(heli_root, col)

    mats: dict[str, bpy.types.Material] = {
        "white": _make_pbr("Mat_Heli_RescueWhite", (0.93, 0.94, 0.96), metallic=0.06, roughness=0.26, coat=0.38, bump_scale=32.0, bump_str=0.025),
        "red": _make_pbr("Mat_Heli_RescueCrimson", (0.74, 0.042, 0.036), metallic=0.18, roughness=0.25, coat=0.45, bump_scale=32.0, bump_str=0.025),
        "charcoal": _make_pbr("Mat_Heli_CharcoalAlloy", (0.042, 0.046, 0.054), metallic=0.68, roughness=0.34),
        "steel": _make_pbr("Mat_Heli_MachinedSteel", (0.72, 0.75, 0.80), metallic=0.92, roughness=0.22),
        "exhaust": _make_pbr("Mat_Heli_HeatBluedExhaust", (0.16, 0.14, 0.18), metallic=0.88, roughness=0.28),
        "canopy": _make_canopy_glass_mat(),
        "cowling_xray": _make_cowling_xray_mat(heli_root, total_frames),
        "blur_disc": _make_rotor_blur_disc_mat(),
        "searchlight": _make_pbr("Mat_Heli_Searchlight", (1.00, 0.94, 0.80), roughness=0.10, emit_rgb=(1.00, 0.92, 0.76), emit_str=1.90),
        "nav_red": _make_pbr("Mat_Heli_NavRed", (1.00, 0.10, 0.06), roughness=0.15, emit_rgb=(1.00, 0.12, 0.06), emit_str=1.80),
        "nav_green": _make_pbr("Mat_Heli_NavGreen", (0.10, 0.98, 0.42), roughness=0.15, emit_rgb=(0.12, 1.00, 0.45), emit_str=1.80),
        "clutch_gold": _make_pbr("Mat_Heli_ClutchGold", (0.98, 0.74, 0.18), metallic=0.92, roughness=0.18, emit_rgb=(1.00, 0.72, 0.16), emit_str=0.42),
        "seized_crimson": _make_pbr("Mat_Heli_SeizedShaft", (0.96, 0.12, 0.06), metallic=0.60, roughness=0.26, emit_rgb=(1.00, 0.18, 0.06), emit_str=1.55),
        "warn_red": _make_pbr("Mat_Heli_LockKeyRed", (1.00, 0.08, 0.04), roughness=0.22, emit_rgb=(1.00, 0.12, 0.04), emit_str=1.85),
        "cyan_glow": _make_pbr("Mat_Heli_FreewheelCyan", (0.16, 0.92, 1.00), roughness=0.18, emit_rgb=(0.18, 0.94, 1.00), emit_str=1.75),
    }

    airframe_objs = _build_airframe_and_skids(heli_root, col, mats)
    clutch_objs = _build_xray_sprag_clutch_and_turbine(heli_root, col, mats, total_frames)
    rotor_objs = _build_main_rotor_assembly(heli_root, col, mats, total_frames)

    # Keyframe Heli_Root across all exterior flight segments (Beat 1A, Beats 2..4, Beats 5B..6)
    kfs: set[int] = set(range(1, total_frames + 1, 3)) | {1, total_frames}
    for cf in C.HARD_CUT_FRAMES:
        for df in (-1, 0, 1):
            ff = cf + df
            if 1 <= ff <= total_frames:
                kfs.add(ff)
    for f in sorted(kfs):
        pos, rot_deg = C.heli_pose_at(float(f))
        heli_root.location = Vector(pos)
        heli_root.rotation_mode = "XYZ"
        heli_root.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
        heli_root.keyframe_insert("location", frame=f)
        heli_root.keyframe_insert("rotation_euler", frame=f)

    # Mark hard-cut transitions on Heli_Root CONSTANT so motion blur never smears across cuts!
    if heli_root.animation_data and heli_root.animation_data.action:
        pre_cuts = {c - 1 for c in C.HARD_CUT_FRAMES}
        for fc in C.iter_action_fcurves(heli_root.animation_data.action):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) in pre_cuts:
                    kp.interpolation = "CONSTANT"

    # Hide exterior helicopter ONLY during interior 1:1 cockpit scenes (Beat 1B & Beat 5A)!
    _key_visibility(airframe_objs + rotor_objs[:-1], _HELI_EXTERIOR_WINDOWS, total_frames)

    maple = _build_maple_seed_samara(col, total_frames)
    return {
        "heli_root": heli_root,
        "airframe": airframe_objs,
        "clutch": clutch_objs,
        "rotor": rotor_objs,
        "maple": maple,
    }
