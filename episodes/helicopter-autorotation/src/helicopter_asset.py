"""
helicopter_asset.py — White-and-Red Rescue Twin-Blade Helicopter, Local-Space X-Ray Cowling,
Seized Turbine Shaft, Animated Freewheeling Sprag Clutch & Sculpted 3D Maple Seed (Samara)
for Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock" (60 fps, 1800 frames).

Strictly obeys AGENTS.md V12 & Blender 5.3.0 rules:
  - Crisp gloss-white & high-visibility crimson-red rescue twin-blade silhouette (1:6 scale diorama):
    * Sculpted aerodynamic fuselage, panoramic tinted bubble canopy (`Is Shadow Ray -> Transparent`),
      twin steel skids strictly flush at local Z = -0.340 m (`0.000 mm` flush contact on Z = 0.900 m helipad!)
    * Tapered tail boom, horizontal stabilizer, red-and-white vertical fin & spinning 2-blade tail rotor
  - Beat 2 Local-Space Hollow Fresnel X-Ray Cowling (`ShaderNodeTexCoord.Object` bound to `Heli_Root`,
    `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing) * pow(Facing, 3.8))`):
    * Exposes the horizontal crimson-heated **Engine Turbine Shaft** seizing dead at f=345
    * Exposes the gleaming **One-Way Freewheeling Sprag Clutch** (outer race, inner race, 12 individually
      articulated golden steel cam rollers snapping open at f=392..435 so the main rotor keeps spinning!)
    * Animated vertical **Swashplate** & **2 Airfoil Main Rotor Blades** (radius 1.15 m) pitching dynamically
      from -4 deg autorotation glide to +13 deg 30-ft flare!
  - Beat 4 Sculpted 3D Winged Maple Seed (`Heli_MapleSeed_Root`):
    * Translucent golden-amber samara wing with raised vascular leading edge & weighted seed nut spinning
      gracefully beside the descending rescue rotor!
  - Materials appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`, safe `C.get_mix_sockets()`,
    and `C.iter_action_fcurves()` for Blender 5.3.0 Slotted Actions.
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
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if coat > 0.0 and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.06
    if emit_rgb is not None and emit_str > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*emit_rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = min(2.0, emit_str)
    return mat


def _make_canopy_glass_mat() -> bpy.types.Material:
    """Panoramic tinted helicopter bubble canopy with mandatory Shadow-Ray transparent bypass."""
    mat = bpy.data.materials.new("Mat_Heli_CanopyGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.14, 0.24, 0.32, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["IOR"].default_value = 1.49
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.82
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.85
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
    Local-Space Hollow Fresnel X-Ray Cowling Shader (`AGENTS.md` V12 Section 3):
      - Driven strictly by `ShaderNodeTexCoord.Object` bound to `heli_root` (100% local space!)
      - Transitions smoothly from crisp red-and-white rescue cowling paint (`xray=0`) to a hollow
        cyan-blue rim-only X-ray shell (`AddShader(Transparent, Emission * CameraRay * (1-Backfacing) * pow(Facing, 3.8))`)
        with ZERO opaque center fill + Shadow-Ray bypass so internal golden sprag clutch shines clearly!
    """
    mat = bpy.data.materials.new("Mat_Heli_CowlingXRay")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    # Solid gloss crimson-red rescue dorsal cowling
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.78, 0.055, 0.045, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.14
    bsdf.inputs["Roughness"].default_value = 0.22
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.65
        bsdf.inputs["Coat Roughness"].default_value = 0.06

    # Local-space scan wave driven by Heli_Root Object coordinates
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = heli_root
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # Animated global X-ray factor node (0 solid -> 1 hollow Fresnel X-ray during Beat 2)
    val_xray = nt.nodes.new("ShaderNodeValue")
    val_xray.name = "CowlingXRayFac"
    for f in (1, 240, 260, 285, 315, 340, 535, 555, 570, 571, total_frames):
        val_xray.outputs[0].default_value = C.cowling_xray_t(float(f))
        val_xray.outputs[0].keyframe_insert("default_value", frame=f)

    # Local-space spatial sweep edge modulation along local Y
    mr_loc = nt.nodes.new("ShaderNodeMapRange")
    mr_loc.clamp = True
    mr_loc.inputs["From Min"].default_value = -0.42
    mr_loc.inputs["From Max"].default_value = 0.56
    mr_loc.inputs["To Min"].default_value = 0.0
    mr_loc.inputs["To Max"].default_value = 1.0
    nt.links.new(sep.outputs["Y"], mr_loc.inputs["Value"])

    # Hollow Fresnel X-ray shell: Transparent + Emission * CameraRay * (1 - Backfacing) * pow(Facing, 3.8)
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
    mul3.inputs[1].default_value = 1.55  # AgX-safe cyan rim glow
    nt.links.new(mul2.outputs["Value"], mul3.inputs[0])

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.14, 0.88, 1.00, 1.0)
    nt.links.new(mul3.outputs["Value"], emit.inputs["Strength"])

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    add_xray = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transp.outputs["BSDF"], add_xray.inputs[0])
    nt.links.new(emit.outputs["Emission"], add_xray.inputs[1])

    # Blend Solid -> Hollow X-Ray via animated CowlingXRayFac
    mix_mode = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(val_xray.outputs[0], mix_mode.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_mode.inputs[1])
    nt.links.new(add_xray.outputs["Shader"], mix_mode.inputs[2])

    # Shadow bypass when in X-ray mode so sunlight illuminates internal sprag clutch cleanly!
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


def _add_tube_bm(
    bm: bmesh.types.BMesh,
    p0: Vector,
    p1: Vector,
    r0: float,
    r1: float,
    mat_idx: int = 0,
    segs: int = 12,
) -> None:
    d = p1 - p0
    length = d.length
    if length < 1e-5:
        return
    mid = 0.5 * (p0 + p1)
    rm = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    res = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs,
        radius1=r0, radius2=r1, depth=length,
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


def _build_rescue_fuselage(
    heli_root: bpy.types.Object,
    col: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> list[bpy.types.Object]:
    """
    Sculpted White-and-Red Rescue Twin-Blade Helicopter Fuselage, Canopy, Skids & Tail Boom.
    Nose points toward -Y (forward flight); tail boom extends toward +Y.
    Skid bottoms sit STRICTLY at local Z = -0.340 m!
    """
    # 1. Lofted Aerodynamic Rescue Cabin Fuselage (gloss white with crimson rescue nose & belly stripe)
    bm = bmesh.new()
    # Cross-section stations: (y_pos, half_w, z_bot, z_top, mat_idx)
    stations = [
        (-0.88, 0.11, -0.08, 0.05, 1),   # Crimson rescue nose tip
        (-0.74, 0.21, -0.15, 0.14, 1),   # Forward crimson nose band
        (-0.52, 0.28, -0.18, 0.23, 0),   # Forward cockpit cabin (white)
        (-0.22, 0.31, -0.19, 0.26, 0),   # Main cabin doors (white)
        (0.14, 0.30, -0.18, 0.25, 0),    # Mid-aft cabin
        (0.46, 0.23, -0.14, 0.22, 1),    # Aft red rescue band
        (0.72, 0.14, -0.05, 0.16, 0),    # Tail boom transition
    ]
    n_ring = 16
    rings: list[list[bmesh.types.BMVert]] = []
    for y_pos, hw, zb, zt, _ in stations:
        zc = 0.5 * (zb + zt)
        hz = 0.5 * (zt - zb)
        ring: list[bmesh.types.BMVert] = []
        for k in range(n_ring):
            ang = 2.0 * math.pi * k / n_ring
            # Super-ellipse rounded-square helicopter cabin profile
            ca, sa = math.cos(ang), math.sin(ang)
            sx = math.copysign(abs(ca) ** 0.72, ca) * hw
            sz = zc + math.copysign(abs(sa) ** 0.78, sa) * hz
            ring.append(bm.verts.new((sx, y_pos, sz)))
        rings.append(ring)

    for s in range(len(rings) - 1):
        r0, r1 = rings[s], rings[s + 1]
        base_mat = stations[s][4]
        for k in range(n_ring):
            kn = (k + 1) % n_ring
            nf = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
            # Paint crisp crimson horizontal rescue cheat-line stripe along mid-lower cabin flanks!
            ang_mid = 2.0 * math.pi * (k + 0.5) / n_ring
            is_stripe = (-0.55 <= math.sin(ang_mid) <= -0.12) and (s in (2, 3, 4))
            nf.material_index = 1 if is_stripe else base_mat

    # Cap nose & aft transition cleanly
    f_nose = bm.faces.new(list(reversed(rings[0])))
    f_nose.material_index = 1
    f_aft = bm.faces.new(rings[-1])
    f_aft.material_index = 0

    # Raised red rescue cross emblem on both cabin sliding doors (mat_idx=1)
    for sx in (-1.0, +1.0):
        _add_box_bm(bm, (sx * 0.308, -0.04, 0.06), (0.012, 0.18, 0.055), mat_idx=1, bevel_w=0.002)
        _add_box_bm(bm, (sx * 0.308, -0.04, 0.06), (0.012, 0.055, 0.18), mat_idx=1, bevel_w=0.002)
        # Side cabin door window frames (dark metallic, mat_idx=2)
        _add_box_bm(bm, (sx * 0.304, -0.26, 0.12), (0.014, 0.24, 0.15), mat_idx=2, bevel_w=0.003)

    # Tapered Tail Boom extending +Y (0.68 -> 1.72 m) with red/white rescue bands & vertical tail fin
    _add_tube_bm(bm, Vector((0.0, 0.68, 0.08)), Vector((0.0, 1.32, 0.13)), 0.105, 0.068, mat_idx=0, segs=14)
    _add_tube_bm(bm, Vector((0.0, 1.32, 0.13)), Vector((0.0, 1.74, 0.16)), 0.068, 0.046, mat_idx=1, segs=14)
    # Horizontal tail stabilizers
    _add_box_bm(bm, (0.0, 1.34, 0.14), (0.58, 0.14, 0.022), mat_idx=1, bevel_w=0.004)
    # Swept vertical tail fin (crimson red + white tip)
    _add_box_bm(bm, (0.0, 1.70, 0.31), (0.036, 0.22, 0.36), mat_idx=1, bevel_w=0.006, rot_deg=(-22.0, 0.0, 0.0))
    _add_box_bm(bm, (0.0, 1.78, 0.45), (0.032, 0.16, 0.08), mat_idx=0, bevel_w=0.004, rot_deg=(-22.0, 0.0, 0.0))
    # Tail guard stinger
    _add_tube_bm(bm, Vector((0.0, 1.66, 0.12)), Vector((0.0, 1.76, -0.06)), 0.012, 0.008, mat_idx=2, segs=8)

    # Twin Tubular Steel Landing Skids — BOTTOM STRICTLY AT LOCAL Z = -0.340 m!
    # Tube radius = 0.020 m -> centerline at Z = -0.320 m -> bottom = -0.340 m EXACT!
    skid_z = C.HELI_SKID_BOTTOM_LOCAL_Z + 0.020  # -0.320 m
    for sx in (-0.34, +0.34):
        # Main straight skid rail
        _add_tube_bm(bm, Vector((sx, -0.58, skid_z)), Vector((sx, 0.44, skid_z)), 0.020, 0.020, mat_idx=3, segs=12)
        # Curved upturned front toe
        _add_tube_bm(bm, Vector((sx, -0.58, skid_z)), Vector((sx, -0.72, skid_z + 0.085)), 0.020, 0.017, mat_idx=3, segs=12)
        # Forward & aft cross-tube struts up to fuselage belly
        _add_tube_bm(bm, Vector((sx, -0.32, skid_z + 0.01)), Vector((sx * 0.62, -0.28, -0.16)), 0.016, 0.016, mat_idx=3, segs=10)
        _add_tube_bm(bm, Vector((sx, 0.22, skid_z + 0.01)), Vector((sx * 0.62, 0.18, -0.15)), 0.016, 0.016, mat_idx=3, segs=10)
    # Crossbars linking left & right struts
    _add_tube_bm(bm, Vector((-0.26, -0.29, -0.22)), Vector((0.26, -0.29, -0.22)), 0.014, 0.014, mat_idx=3, segs=10)
    _add_tube_bm(bm, Vector((-0.26, 0.19, -0.21)), Vector((0.26, 0.19, -0.21)), 0.014, 0.014, mat_idx=3, segs=10)

    fuselage_obj = _new_bmesh_obj(
        "Heli_Fuselage", bm,
        [mats["white"], mats["red"], mats["dark_metal"], mats["steel"]],
        col, parent=heli_root, smooth=True,
    )

    # 2. Panoramic Tinted Bubble Windshield Canopy (with shadow-ray bypass!)
    bm_g = bmesh.new()
    res_g = bmesh.ops.create_uvsphere(
        bm_g, u_segments=24, v_segments=14, radius=1.0,
        matrix=Matrix.Translation((0.0, -0.50, 0.11))
        @ Matrix.Diagonal((0.265, 0.30, 0.165, 1.0)),
    )
    vg = set(res_g["verts"])
    for f in bm_g.faces:
        if all(v in vg for v in f.verts):
            f.material_index = 0
    canopy_obj = _new_bmesh_obj(
        "Heli_CanopyGlass", bm_g, [mats["canopy_glass"]],
        col, parent=heli_root, smooth=True, no_shadow=True,
    )

    # 3. Dorsal Engine & Transmission Cowling (Local-Space Hollow Fresnel X-Ray in Beat 2!)
    bm_cw = bmesh.new()
    _add_box_bm(bm_cw, (0.0, 0.14, 0.30), (0.34, 0.78, 0.22), mat_idx=0, bevel_w=0.045)
    # Twin dark turbine exhaust nozzles protruding aft
    for sx in (-0.09, +0.09):
        _add_tube_bm(bm_cw, Vector((sx, 0.46, 0.30)), Vector((sx * 1.15, 0.62, 0.31)), 0.048, 0.040, mat_idx=1, segs=14)
    cowling_obj = _new_bmesh_obj(
        "Heli_Cowling_XRay", bm_cw, [mats["cowling_xray"], mats["dark_metal"]],
        col, parent=heli_root, smooth=True,
    )

    # 4. Spinning 2-Blade Tail Rotor on port side of vertical tail fin
    bm_tr = bmesh.new()
    _add_tube_bm(bm_tr, Vector((-0.02, 0.0, 0.0)), Vector((-0.075, 0.0, 0.0)), 0.024, 0.020, mat_idx=0, segs=12)
    for sz in (-1.0, +1.0):
        _add_box_bm(bm_tr, (-0.068, 0.0, sz * 0.12), (0.008, 0.036, 0.20), mat_idx=1, bevel_w=0.002)
        _add_box_bm(bm_tr, (-0.068, 0.0, sz * 0.20), (0.009, 0.038, 0.035), mat_idx=2, bevel_w=0.001)
    tail_rotor = _new_bmesh_obj(
        "Heli_TailRotor", bm_tr, [mats["steel"], mats["carbon"], mats["red"]],
        col, parent=heli_root, location=(-0.02, 1.72, 0.34), smooth=False,
    )
    return [fuselage_obj, canopy_obj, cowling_obj, tail_rotor]


def _build_drivetrain_and_sprag_clutch(
    heli_root: bpy.types.Object,
    col: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
    total_frames: int,
) -> dict[str, Any]:
    """
    Hero Beat 2 Internal X-Ray Mechanical Drivetrain:
      1. `Heli_Engine_TurbineShaft`: crimson-hot turbine spool & bevel gear that SEIZES DEAD at f=345!
      2. `Heli_SpragClutch_OuterRace` (coupled to dead engine, stops dead!)
      3. `Heli_SpragClutch_InnerRace` + `Heli_MainRotor_Mast` (keeps spinning freely!)
      4. 12 individually articulated golden steel **Sprag Clutch Cam Rollers** (`Heli_SpragRoller_00..11`)
         that visibly tilt/snap open at f=392..435 to decouple the dead engine from the main rotor!
      5. Animated vertical **Swashplate** + articulated hub yoke + **2 Airfoil Main Rotor Blades** (R=1.15m)
         whose collective pitch angle tracks `C.blade_pitch_deg_at(f)`!
    """
    cx, cy, cz = C.HELI_CLUTCH_LOCAL_POS  # (0.0, 0.02, 0.30)
    tx, ty, tz = C.HELI_TURBINE_LOCAL_POS  # (0.0, 0.24, 0.28)

    # 1. Seized Turbine Engine Shaft + Compressor Spool + Bevel Gear (rotates around local Y, SEIZES at f=345!)
    bm_eng = bmesh.new()
    _add_tube_bm(bm_eng, Vector((0.0, -0.14, 0.0)), Vector((0.0, 0.26, 0.0)), 0.026, 0.026, mat_idx=0, segs=16)
    # 3 turbine compressor disks with notched blades so shaft rotation & abrupt stop are unmistakable!
    for dy, rad in ((0.02, 0.068), (0.11, 0.076), (0.20, 0.072)):
        _add_tube_bm(bm_eng, Vector((0.0, dy - 0.012, 0.0)), Vector((0.0, dy + 0.012, 0.0)), rad, rad, mat_idx=1, segs=18)
        for k in range(8):
            a = 2.0 * math.pi * k / 8.0
            _add_box_bm(
                bm_eng, (math.cos(a) * rad, dy, math.sin(a) * rad),
                (0.024, 0.018, 0.010), mat_idx=0, bevel_w=0.001,
                rot_deg=(0.0, math.degrees(-a), 0.0),
            )
    # Forward bevel pinion driving the clutch outer race
    _add_tube_bm(bm_eng, Vector((0.0, -0.14, 0.0)), Vector((0.0, -0.19, 0.0)), 0.044, 0.032, mat_idx=0, segs=14)
    engine_shaft = _new_bmesh_obj(
        "Heli_Engine_TurbineShaft", bm_eng, [mats["seized_crimson"], mats["dark_metal"]],
        col, parent=heli_root, location=(tx, ty, tz), smooth=True,
    )

    # 2. Sprag Clutch Outer Race (ring gear coupled to engine — stops dead with the engine!)
    bm_or = bmesh.new()
    n_seg = 24
    r_in, r_out, h_r = 0.088, 0.112, 0.064
    for i in range(n_seg):
        a0 = 2.0 * math.pi * i / n_seg
        a1 = 2.0 * math.pi * (i + 1) / n_seg
        pts_bot = [
            (r_in * math.cos(a0), r_in * math.sin(a0), -0.5 * h_r),
            (r_out * math.cos(a0), r_out * math.sin(a0), -0.5 * h_r),
            (r_out * math.cos(a1), r_out * math.sin(a1), -0.5 * h_r),
            (r_in * math.cos(a1), r_in * math.sin(a1), -0.5 * h_r),
        ]
        pts_top = [(x, y, 0.5 * h_r) for x, y, _ in pts_bot]
        vb = [bm_or.verts.new(p) for p in pts_bot]
        vt = [bm_or.verts.new(p) for p in pts_top]
        for quad in (
            (vb[0], vb[3], vb[2], vb[1]),
            (vt[0], vt[1], vt[2], vt[3]),
            (vb[1], vb[2], vt[2], vt[1]),
            (vb[3], vb[0], vt[0], vt[3]),
        ):
            nf = bm_or.faces.new(quad)
            nf.material_index = 0
        # Gear tooth lug on outer rim so viewers clearly see outer race freeze!
        if i % 2 == 0:
            am = 0.5 * (a0 + a1)
            _add_box_bm(
                bm_or, (math.cos(am) * (r_out + 0.008), math.sin(am) * (r_out + 0.008), 0.0),
                (0.018, 0.012, h_r * 0.78), mat_idx=1, bevel_w=0.002,
                rot_deg=(0.0, 0.0, math.degrees(am)),
            )
    clutch_outer = _new_bmesh_obj(
        "Heli_SpragClutch_OuterRace", bm_or, [mats["seized_crimson"], mats["dark_metal"]],
        col, parent=heli_root, location=(cx, cy, cz), smooth=False,
    )

    # 3. Main Rotor Mast + Sprag Clutch Inner Race (parent of spinning blades & hub!)
    rotor_mast_empty = bpy.data.objects.new("Heli_MainRotor_Pivot", None)
    rotor_mast_empty.empty_display_type = "ARROWS"
    rotor_mast_empty.empty_display_size = 0.12
    rotor_mast_empty.parent = heli_root
    rotor_mast_empty.location = Vector((cx, cy, cz))
    rotor_mast_empty.rotation_mode = "XYZ"
    _link(rotor_mast_empty, col)

    bm_mast = bmesh.new()
    # Gleaming inner sprag race hub + vertical steel rotor mast up to hub yoke (Z=0.44 - 0.30 = +0.14)
    _add_tube_bm(bm_mast, Vector((0.0, 0.0, -0.04)), Vector((0.0, 0.0, 0.04)), 0.056, 0.056, mat_idx=0, segs=24)
    _add_tube_bm(bm_mast, Vector((0.0, 0.0, -0.05)), Vector((0.0, 0.0, 0.20)), 0.026, 0.024, mat_idx=1, segs=18)
    # Hub yoke & stabilizer bar at top of mast (local Z = +0.165 -> world local Z = 0.465)
    _add_box_bm(bm_mast, (0.0, 0.0, 0.155), (0.28, 0.068, 0.048), mat_idx=1, bevel_w=0.008)
    _add_tube_bm(bm_mast, Vector((0.0, 0.0, 0.18)), Vector((0.0, 0.0, 0.21)), 0.036, 0.022, mat_idx=2, segs=16)
    mast_obj = _new_bmesh_obj(
        "Heli_MainRotor_Mast", bm_mast, [mats["cyan_steel"], mats["steel"], mats["red"]],
        col, parent=rotor_mast_empty, smooth=True,
    )

    # 4. 12 Individually Articulated Golden Steel Sprag Clutch Cam Rollers (snap open at f=392..435!)
    rollers: list[bpy.types.Object] = []
    r_cam = 0.072
    for idx in range(12):
        ang = 2.0 * math.pi * idx / 12.0
        bm_r = bmesh.new()
        # Asymmetric figure-8 sprag cam lobe
        _add_box_bm(bm_r, (0.0, 0.0, 0.0), (0.026, 0.013, 0.044), mat_idx=0, bevel_w=0.003)
        robj = _new_bmesh_obj(
            f"Heli_SpragRoller_{idx:02d}", bm_r, [mats["sprag_gold"]],
            col, parent=rotor_mast_empty,
            location=(r_cam * math.cos(ang), r_cam * math.sin(ang), 0.0),
            rot_deg=(0.0, 0.0, math.degrees(ang) + 28.0),
            smooth=True,
        )
        rollers.append(robj)
        # Keyframe dramatic mechanical cam-lobe snap-open from engaged (+28 deg) to freewheeling (+66 deg)!
        for f in (241, 388, 405, 420, 438, 570):
            u_un = C.sprag_unlock_t(float(f))
            tilt = C.lerp(28.0, 68.0, u_un)
            robj.rotation_euler = Euler((0.0, 0.0, math.radians(math.degrees(ang) + tilt)), "XYZ")
            robj.keyframe_insert("rotation_euler", frame=f)

    # 5. Animated Vertical Swashplate collar sliding along mast with `collective_level_at(f)`
    bm_sw = bmesh.new()
    _add_tube_bm(bm_sw, Vector((0.0, 0.0, -0.014)), Vector((0.0, 0.0, 0.014)), 0.068, 0.068, mat_idx=0, segs=20)
    for sx in (-0.062, +0.062):
        _add_tube_bm(bm_sw, Vector((sx, 0.0, 0.012)), Vector((sx, 0.0, 0.088)), 0.006, 0.006, mat_idx=1, segs=8)
    swash_obj = _new_bmesh_obj(
        "Heli_Swashplate", bm_sw, [mats["sprag_gold"], mats["steel"]],
        col, parent=rotor_mast_empty, location=(0.0, 0.0, 0.075), smooth=True,
    )

    # 6. Two Sculpted Airfoil Main Rotor Blades (radius = 1.15 m, dynamic collective pitch!)
    blades: list[bpy.types.Object] = []
    blade_len = C.HELI_ROTOR_RADIUS - 0.11  # 1.04 m blade starting at X=+/-0.11 -> tip at 1.15 m
    for b_idx, side in enumerate((-1.0, +1.0)):
        bm_b = bmesh.new()
        # Airfoil chord cross-section extruded cleanly from root (0.0) to tip (blade_len)
        _add_box_bm(bm_b, (side * (0.5 * blade_len), 0.0, 0.0), (blade_len, 0.072, 0.012), mat_idx=0, bevel_w=0.004)
        # High-visibility rescue crimson & white tip stripes!
        _add_box_bm(bm_b, (side * (blade_len - 0.11), 0.0, 0.0), (0.09, 0.074, 0.013), mat_idx=1, bevel_w=0.002)
        _add_box_bm(bm_b, (side * (blade_len - 0.04), 0.0, 0.0), (0.055, 0.074, 0.013), mat_idx=2, bevel_w=0.002)
        b_obj = _new_bmesh_obj(
            f"Heli_MainBlade_{b_idx}", bm_b, [mats["carbon"], mats["red"], mats["white"]],
            col, parent=rotor_mast_empty, location=(side * 0.11, 0.0, 0.155), smooth=True,
        )
        blades.append(b_obj)

    # Animate engine shaft (seizes at f=345!), clutch outer race, main rotor spin, swashplate & blade pitch!
    # Sample step=4 baseline + dense step=2 around Beat 2 seizure & Beat 5B/6 flare/touchdown
    kfs: set[int] = set(range(241, total_frames + 1, 4))
    kfs.update(range(325, 445, 2))
    kfs.update(range(1451, 1740, 2))
    kfs.add(total_frames)

    for f in sorted(kfs):
        ff = float(f)
        # Main rotor mast spins smoothly around local Z (NEVER stops when engine dies!)
        r_ang = C.rotor_angle_rad_at(ff)
        rotor_mast_empty.rotation_euler = Euler((0.0, 0.0, r_ang), "XYZ")
        rotor_mast_empty.keyframe_insert("rotation_euler", frame=f)

        # Engine shaft & outer race stop DEAD at f=345 in Beat 2!
        e_ang = C.engine_shaft_angle_rad_at(ff)
        engine_shaft.rotation_euler = Euler((0.0, e_ang, 0.0), "XYZ")
        engine_shaft.keyframe_insert("rotation_euler", frame=f)
        clutch_outer.rotation_euler = Euler((0.0, 0.0, e_ang * 0.65), "XYZ")
        clutch_outer.keyframe_insert("rotation_euler", frame=f)

        # Swashplate height + collective blade pitch (-4 deg autorotation -> +13 deg flare!)
        coll = C.collective_level_at(ff)
        swash_obj.location = Vector((0.0, 0.0, C.lerp(0.058, 0.095, coll)))
        swash_obj.keyframe_insert("location", frame=f)

        pitch_rad = math.radians(C.blade_pitch_deg_at(ff))
        for b_idx, b_obj in enumerate(blades):
            sign = 1.0 if b_idx == 1 else -1.0
            b_obj.rotation_euler = Euler((sign * pitch_rad, 0.0, 0.0), "XYZ")
            b_obj.keyframe_insert("rotation_euler", frame=f)

    return {
        "engine_shaft": engine_shaft,
        "clutch_outer": clutch_outer,
        "rotor_pivot": rotor_mast_empty,
        "mast": mast_obj,
        "rollers": rollers,
        "swashplate": swash_obj,
        "blades": blades,
    }


def _build_maple_seed_samara(
    col: bpy.types.Collection,
    total_frames: int,
) -> dict[str, Any]:
    """
    Beat 4 Hero Botanical Prop: Sculpted 3D Winged Maple Seed (Samara) spinning beside the rescue rotor!
    Weighted plump brown seed nut at origin + curved translucent golden-amber veined samara blade
    extending 0.56 m with thickened aerodynamic leading edge.
    """
    root = bpy.data.objects.new("Heli_MapleSeed_Root", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 0.14
    root.rotation_mode = "XYZ"
    _link(root, col)

    # Translucent backlit golden-amber samara wing material
    mat_wing = bpy.data.materials.new("Mat_Maple_SamaraWing")
    mat_wing.use_nodes = True
    nt = mat_wing.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.86, 0.58, 0.22, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.42
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.55
    bsdf.inputs["Emission Color"].default_value = (0.95, 0.62, 0.20, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.14

    mat_nut = _make_pbr("Mat_Maple_SeedNut", (0.34, 0.19, 0.08), roughness=0.52, coat=0.18)
    mat_vein = _make_pbr("Mat_Maple_VeinEdge", (0.58, 0.34, 0.11), roughness=0.45)

    bm = bmesh.new()
    # 1. Weighted bulbous seed nut at pivot (slightly offset so center of gravity drives autorotation!)
    res_n = bmesh.ops.create_uvsphere(
        bm, u_segments=18, v_segments=12, radius=1.0,
        matrix=Matrix.Translation((-0.02, 0.0, -0.01))
        @ Matrix.Diagonal((0.068, 0.048, 0.036, 1.0)),
    )
    vn = set(res_n["verts"])
    for f in bm.faces:
        if all(v in vn for v in f.verts):
            f.material_index = 1

    # 2. Curved organic maple samara wing lofted across 14 chordwise stations (0.04 -> 0.58 m)
    n_st = 14
    prev_quad: list[bmesh.types.BMVert] | None = None
    for i in range(n_st):
        u = i / float(n_st - 1)
        x = 0.03 + u * 0.54
        # Classic maple-seed silhouette: narrow near nut, broad curved paddle at u=0.68, rounded tip
        chord = 0.042 + 0.145 * math.sin(math.pi * (u ** 0.72))
        y_lead = 0.018 * math.sin(math.pi * u)
        y_trail = y_lead - chord
        # Authentic aerodynamic twist & subtle camber
        z_lead = 0.008 * u
        z_trail = -0.022 * math.sin(math.pi * u) - 0.012 * u
        thick = 0.006 * (1.0 - 0.55 * u)
        q = [
            bm.verts.new((x, y_lead, z_lead + thick)),
            bm.verts.new((x, y_trail, z_trail + 0.4 * thick)),
            bm.verts.new((x, y_trail, z_trail - 0.4 * thick)),
            bm.verts.new((x, y_lead, z_lead - thick)),
        ]
        if prev_quad is not None:
            for k in range(4):
                kn = (k + 1) % 4
                nf = bm.faces.new((prev_quad[k], q[k], q[kn], prev_quad[kn]))
                nf.material_index = 2 if k == 3 else 0
        prev_quad = q

    # Raised thickened leading-edge vascular rib & branching veins
    _add_tube_bm(bm, Vector((0.03, 0.008, 0.002)), Vector((0.52, 0.012, 0.008)), 0.011, 0.004, mat_idx=2, segs=10)
    for vu in (0.28, 0.46, 0.64, 0.80):
        x0 = 0.03 + vu * 0.52
        _add_tube_bm(
            bm,
            Vector((x0, 0.010, 0.006)),
            Vector((x0 + 0.075, -0.085 * math.sin(math.pi * vu), -0.012)),
            0.0038, 0.0018, mat_idx=2, segs=6,
        )

    seed_obj = _new_bmesh_obj(
        "Heli_MapleSeed_Mesh", bm, [mat_wing, mat_nut, mat_vein],
        col, parent=root, smooth=True,
    )

    # Visible ONLY during Beat 4 (961..1290)
    _key_visibility([seed_obj], [(961, 1290)], total_frames)
    for f in range(961, 1291, 2):
        pos, rot = C.maple_seed_pose_at(float(f))
        root.location = Vector(pos)
        root.rotation_euler = Euler(tuple(math.radians(a) for a in rot), "XYZ")
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)

    return {"root": root, "mesh": seed_obj}


def build_helicopter_asset(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict[str, Any]:
    """Assemble the complete rescue helicopter + X-ray sprag clutch + Beat 4 3D maple seed."""
    heli_root = bpy.data.objects.new("Heli_Root", None)
    heli_root.empty_display_type = "ARROWS"
    heli_root.empty_display_size = 0.25
    heli_root.rotation_mode = "XYZ"
    _link(heli_root, col)

    mats = {
        "white": _make_pbr("Mat_Heli_RescueWhite", (0.94, 0.95, 0.96), metallic=0.06, roughness=0.20, coat=0.65),
        "red": _make_pbr("Mat_Heli_RescueRed", (0.80, 0.052, 0.042), metallic=0.12, roughness=0.22, coat=0.65),
        "dark_metal": _make_pbr("Mat_Heli_DarkMetal", (0.055, 0.062, 0.072), metallic=0.82, roughness=0.34),
        "steel": _make_pbr("Mat_Heli_Steel", (0.52, 0.56, 0.60), metallic=0.92, roughness=0.22),
        "cyan_steel": _make_pbr(
            "Mat_Heli_CyanSteel", (0.28, 0.72, 0.88), metallic=0.85, roughness=0.20,
            emit_rgb=(0.12, 0.78, 0.96), emit_str=0.35,
        ),
        "carbon": _make_pbr("Mat_Heli_CarbonBlade", (0.045, 0.048, 0.055), metallic=0.18, roughness=0.32, coat=0.30),
        "sprag_gold": _make_pbr(
            "Mat_Heli_SpragGold", (0.96, 0.74, 0.22), metallic=0.94, roughness=0.18,
            emit_rgb=(1.00, 0.76, 0.18), emit_str=0.45,
        ),
        "seized_crimson": _make_pbr(
            "Mat_Heli_SeizedCrimson", (0.88, 0.14, 0.08), metallic=0.72, roughness=0.28,
            emit_rgb=(1.00, 0.16, 0.06), emit_str=0.85,
        ),
        "canopy_glass": _make_canopy_glass_mat(),
        "cowling_xray": _make_cowling_xray_mat(heli_root, total_frames),
    }

    body_objs = _build_rescue_fuselage(heli_root, col, mats)
    tail_rotor = body_objs[-1]
    drivetrain = _build_drivetrain_and_sprag_clutch(heli_root, col, mats, total_frames)

    # Exterior rescue helicopter visible during Beats 2, 3, 4, 5B, 6 (hidden during 1:1 cockpit Beats 1 & 5A!)
    ext_windows = [(241, 1290), (C.B5_EXTERIOR_START, total_frames)]
    all_heli_meshes = [
        *body_objs,
        drivetrain["engine_shaft"],
        drivetrain["clutch_outer"],
        drivetrain["mast"],
        *drivetrain["rollers"],
        drivetrain["swashplate"],
        *drivetrain["blades"],
    ]
    _key_visibility(all_heli_meshes, ext_windows, total_frames)

    # Animate `Heli_Root` trajectory & attitude across all exterior beats + tail rotor spin!
    kfs: set[int] = set(range(241, 1291, 4)) | set(range(C.B5_EXTERIOR_START, total_frames + 1, 3))
    kfs.update({241, 570, 571, 960, 961, 1290, C.B5_EXTERIOR_START, 1580, 1581, 1685, 1698, 1712, total_frames})
    const_cuts = {571, 961, C.B5_EXTERIOR_START, 1581}

    for f in sorted(kfs):
        ff = float(f)
        pos, rot = C.heli_pose_at(ff)
        heli_root.location = Vector(pos)
        heli_root.rotation_euler = Euler(tuple(math.radians(a) for a in rot), "XYZ")
        heli_root.keyframe_insert("location", frame=f)
        heli_root.keyframe_insert("rotation_euler", frame=f)

        # Tail rotor spins in sync with main rotor transmission
        tr_ang = C.rotor_angle_rad_at(ff) * 3.4
        tail_rotor.rotation_euler = Euler((tr_ang, 0.0, 0.0), "XYZ")
        tail_rotor.keyframe_insert("rotation_euler", frame=f)

    # Enforce CONSTANT interpolation on hard-cut exterior transitions so helicopter never blurs across beats!
    ad = heli_root.animation_data
    if ad and ad.action:
        for fc in C.iter_action_fcurves(ad.action):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) in {c - 1 for c in const_cuts}:
                    kp.interpolation = "CONSTANT"

    # Build Beat 4 Sculpted 3D Maple Seed (Samara)
    maple_res = _build_maple_seed_samara(col, total_frames)

    return {
        "heli_root": heli_root,
        "body_objs": body_objs,
        "drivetrain": drivetrain,
        "maple": maple_res,
    }
