"""
Passenger Character Animator for Episode #16:
"Why Jumping in a Falling Elevator WON'T Save You" (30.0 s @ 60 fps = 1800 frames).

Public API:
    build_passenger(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dictionary keys & types:
    - "carrier":       bpy.types.Object  (Passenger_Carrier Empty synced to C.cab_world_location)
    - "rig":           bpy.types.Object  (Passenger_Rig MPFB2 163-bone armature)
    - "body":          bpy.types.Object  (Passenger_Body 1.78m adult male basemesh)
    - "equipped":      dict[str, bpy.types.Object]  (Eyes, Eyebrows, Eyelashes, Teeth, Tongue,
                                                     Hair, Pants, Shirt, Shoes)
    - "xray_nodes":    list[bpy.types.Node]  (ShaderNodeValue `XRayBody` nodes across all mats)
    - "harness_parts": list[bpy.types.Object]  (Fitted crew-neck collar/bib + tailored wrist cuffs)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_ROOT_SRC = _REPO_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_ROOT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elevator_contract as C  # noqa: E402
import mpfb_characters as mc  # noqa: E402

PASSENGER_HEIGHT_M: float = float(C.PASSENGER["height_m"])
_SPINE_SEG_BONES: tuple[str, ...] = ("spine04", "spine03", "spine02", "spine01")
_HERO_BLINK_EXCLUSION_FRAMES: tuple[int, ...] = (
    1, 110, 150, 190, 312, 450, 620, 665, 750, 795, 900, 940, 985, 1080, 1237, 1275, 1386, 1465, 1650, 1800
)


# ============================================================================
# COLLECTION, MESH, HIERARCHY & FCURVE HELPERS
# ============================================================================
def _link_obj(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)
    return obj


def _set_smooth(mesh: bpy.types.Mesh) -> None:
    if len(mesh.polygons) > 0:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()


def _set_fcurve_interp(
    obj_or_data,
    interp: str = "BEZIER",
    skip_prefix: str = "hide_",
    constant_frames: set[int] | None = None,
) -> None:
    ad = getattr(obj_or_data, "animation_data", None)
    if not ad or not ad.action:
        return
    const_set = constant_frames or set()
    for fc in C.iter_action_fcurves(ad.action):
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            continue
        for kp in fc.keyframe_points:
            kf_int = int(round(kp.co.x))
            if kf_int in const_set:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = interp
                if interp == "BEZIER":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"


def _parent_to_bone_preserve_world(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    """Parent `obj` to `rig`'s pose bone `bone_name` preserving its exact current world transform."""
    bpy.context.view_layer.update()
    pb = rig.pose.bones[bone_name]
    tail_off = 0.0 if pb.bone.use_relative_parent else pb.length
    bone_world = rig.matrix_world @ pb.matrix @ Matrix.Translation((0.0, tail_off, 0.0))
    orig_world = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = bone_world.inverted()
    obj.matrix_world = orig_world
    bpy.context.view_layer.update()


def _eval_mesh_verts_in(obj: bpy.types.Object, inv_frame: Matrix) -> list[Vector]:
    """Evaluate deformed mesh vertex positions transformed into `inv_frame` coordinate space."""
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    em = eo.to_mesh()
    xf = inv_frame @ eo.matrix_world
    pts = [xf @ v.co for v in em.vertices]
    eo.to_mesh_clear()
    return pts


def _eval_mesh_bounds_in(obj: bpy.types.Object, inv_frame: Matrix) -> tuple[Vector, Vector, Vector]:
    pts = _eval_mesh_verts_in(obj, inv_frame)
    if not pts:
        z = Vector((0.0, 0.0, 0.0))
        return z, z, z
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    ctr = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
    return mn, mx, ctr


def _new_bmesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material | list[bpy.types.Material],
    world_xf: Matrix,
    col: bpy.types.Collection,
    smooth: bool = True,
    recalc_normals: bool = True,
) -> bpy.types.Object:
    """Finalize `bm` into a named mesh object with material slot(s) attached BEFORE `bm.to_mesh()`."""
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    mats = mat if isinstance(mat, list) else [mat]
    for m in mats:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        _set_smooth(mesh)
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = world_xf.copy()
    _link_obj(obj, col)
    return obj


# ============================================================================
# UNDERCLOTHES SHRINK & PROCEDURAL COLLAR / WRIST CUFFS
# ============================================================================
def _apply_underclothes_shrink(
    body: bpy.types.Object,
    carrier: bpy.types.Object,
) -> None:
    """Apply restricted torso/pelvis/upper-thigh + upper/lower-arm negative DISPLACE (-0.011)
    BEFORE Armature so deep squats, hip bends, and folded arms never poke skin through sweater
    or chinos, while keeping head, neck, wrists, hands, knees, and calves strictly 0.0 weight."""
    inv_car = carrier.matrix_world.inverted()
    xf = inv_car @ body.matrix_world
    pts = [xf @ v.co for v in body.data.vertices]
    if not pts:
        return

    # Measure crown & foot strictly on body.data.vertices
    z_foot = min(p.z for p in pts)
    z_crown = max(p.z for p in pts)
    z_lo = z_foot + 0.50
    z_hi = z_crown - 0.28

    # Exclude any vertex weighted to head, neck, wrists, hands, lower legs, or feet
    # (upperarm & lowerarm are explicitly ALLOWED so arm skin never pokes out of sleeves)
    excl_prefixes = (
        "head", "neck", "wrist", "hand",
        "finger", "thumb", "lowerleg", "foot", "toe",
    )
    excl_weights: dict[int, float] = {}
    for vg in body.vertex_groups:
        vname = vg.name.lower()
        if any(vname.startswith(pref) for pref in excl_prefixes):
            idx = vg.index
            for v in body.data.vertices:
                for g in v.groups:
                    if g.group == idx and g.weight > 0.0:
                        excl_weights[v.index] = excl_weights.get(v.index, 0.0) + g.weight
                        break

    excluded_vids: set[int] = {vid for vid, ew in excl_weights.items() if ew > 0.05}
    vg_shrink = body.vertex_groups.get("UnderClothes_Shrink") or body.vertex_groups.new(name="UnderClothes_Shrink")
    taper = 0.045
    weights: list[float] = [0.0] * len(body.data.vertices)
    for v, p in zip(body.data.vertices, pts):
        if v.index in excluded_vids:
            w = 0.0
        elif abs(p.x) >= 0.85 or p.z < z_lo or p.z > z_hi:
            w = 0.0
        else:
            wx = 1.0 - C.smoothstep(0.75, 0.85, abs(p.x))
            wz_lo = C.smoothstep(z_lo, z_lo + taper, p.z)
            wz_hi = 1.0 - C.smoothstep(z_hi - taper, z_hi, p.z)
            w_excl = 1.0 - C.smoothstep(0.0, 0.05, excl_weights.get(v.index, 0.0))
            w = wx * wz_lo * wz_hi * w_excl
        weights[v.index] = w

    # Gentle 2-pass topological smoothing near neck/wrist boundaries (excluded_vids stay strictly 0.0)
    adj: list[list[int]] = [[] for _ in range(len(body.data.vertices))]
    for edge in body.data.edges:
        a, b = edge.vertices[0], edge.vertices[1]
        adj[a].append(b)
        adj[b].append(a)
    for _ in range(2):
        next_w = list(weights)
        for vid, nbrs in enumerate(adj):
            if vid in excluded_vids or not nbrs:
                next_w[vid] = 0.0
                continue
            if any(nb in excluded_vids for nb in nbrs) or weights[vid] < 0.99:
                avg_nb = sum(weights[nb] for nb in nbrs) / float(len(nbrs))
                next_w[vid] = 0.5 * weights[vid] + 0.5 * avg_nb
        weights = next_w

    for vid, w in enumerate(weights):
        if w > 0.001:
            vg_shrink.add([vid], w, "REPLACE")

    # Insert DISPLACE modifier strictly BEFORE Armature modifier
    mod = body.modifiers.new(name="UnderClothes_Shrink", type="DISPLACE")
    mod.direction = "NORMAL"
    mod.mid_level = 0.0
    mod.strength = -0.011
    mod.vertex_group = vg_shrink.name

    arm_idx = next((i for i, m in enumerate(body.modifiers) if m.type == "ARMATURE"), None)
    if arm_idx is not None:
        cur_idx = list(body.modifiers).index(mod)
        if cur_idx > arm_idx:
            if hasattr(body.modifiers, "move"):
                body.modifiers.move(cur_idx, arm_idx)
            else:
                with bpy.context.temp_override(object=body, active_object=body):
                    while list(body.modifiers).index(mod) > arm_idx:
                        bpy.ops.object.modifier_move_up(modifier=mod.name)


def _filter_segment_pts(
    pts: list[Vector],
    a: Vector,
    b: Vector,
    max_perp: float,
) -> list[Vector]:
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-8:
        return []
    out: list[Vector] = []
    for p in pts:
        ap = p - a
        t = ap.dot(ab) / len_sq
        if -0.08 <= t <= 1.08:
            perp = (ap - ab * t).length
            if perp <= max_perp:
                out.append(p)
    return out


def _add_sleeve_band_from_verts(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    a: Vector,
    b: Vector,
    t_range: tuple[float, float],
    pad_m: float = 0.004,
    thick_m: float = 0.003,
    n_rings: int = 4,
    n_seg: int = 20,
    mat_index: int = 0,
    min_r: float = 0.022,
    max_r: float = 0.036,
) -> None:
    """Create a sleek closed tubular cuff/sleeve band fitted around segment `a -> b`."""
    axis = b - a
    if axis.length < 1e-5:
        return
    u_ax = axis.normalized()
    ref = Vector((0.0, 1.0, 0.0)) if abs(u_ax.y) < 0.85 else Vector((1.0, 0.0, 0.0))
    e1 = u_ax.cross(ref).normalized()
    e2 = u_ax.cross(e1).normalized()

    t0, t1 = t_range
    outer_rings: list[list[bmesh.types.BMVert]] = []
    inner_rings: list[list[bmesh.types.BMVert]] = []

    for r_i in range(max(2, n_rings)):
        t = C.lerp(t0, t1, r_i / float(max(1, n_rings - 1)))
        center = a + axis * t
        half_band = max(0.015, abs(t1 - t0) / float(max(1, n_rings - 1))) * 0.75
        ring_pts = [
            p for p in pts
            if abs(((p - a).dot(axis) / axis.length_squared) - t) <= half_band
        ]

        o_ring: list[bmesh.types.BMVert] = []
        i_ring: list[bmesh.types.BMVert] = []
        for s_i in range(n_seg):
            ang = (s_i / float(n_seg)) * math.tau
            rad_dir = e1 * math.cos(ang) + e2 * math.sin(ang)
            ang_cands: list[float] = []
            for p in ring_pts:
                dp = p - center
                dp_perp = dp - u_ax * dp.dot(u_ax)
                plen = dp_perp.length
                if plen > 1e-5 and (dp_perp / plen).dot(rad_dir) > 0.78:
                    ang_cands.append(plen)
            if ang_cands:
                r_base = max(ang_cands)
            elif ring_pts:
                r_base = sum(
                    ((p - center) - u_ax * (p - center).dot(u_ax)).length for p in ring_pts
                ) / float(len(ring_pts))
            else:
                r_base = 0.5 * (min_r + max_r)
            r_out = max(min_r, min(max_r, r_base + pad_m + thick_m))
            r_in = max(min_r * 0.82, r_out - thick_m)
            o_ring.append(bm.verts.new(center + rad_dir * r_out))
            i_ring.append(bm.verts.new(center + rad_dir * r_in))
        outer_rings.append(o_ring)
        inner_rings.append(i_ring)

    def _mk_face(vs):
        try:
            f = bm.faces.new(vs)
            f.material_index = mat_index
        except ValueError:
            pass

    for r_i in range(len(outer_rings) - 1):
        ro0, ro1 = outer_rings[r_i], outer_rings[r_i + 1]
        ri0, ri1 = inner_rings[r_i], inner_rings[r_i + 1]
        for s_i in range(n_seg):
            s_n = (s_i + 1) % n_seg
            _mk_face((ro0[s_i], ro0[s_n], ro1[s_n], ro1[s_i]))
            _mk_face((ri0[s_n], ri0[s_i], ri1[s_i], ri1[s_n]))

    for s_i in range(n_seg):
        s_n = (s_i + 1) % n_seg
        _mk_face((outer_rings[0][s_n], outer_rings[0][s_i], inner_rings[0][s_i], inner_rings[0][s_n]))
        _mk_face((outer_rings[-1][s_i], outer_rings[-1][s_n], inner_rings[-1][s_n], inner_rings[-1][s_i]))


def _make_fabric_material(name: str, rgb: tuple[float, float, float], roughness: float = 0.78) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.35
        if "Sheen Roughness" in bsdf.inputs:
            bsdf.inputs["Sheen Roughness"].default_value = 0.45
    return mat


def _build_collar_and_cuffs(
    carrier: bpy.types.Object,
    rig: bpy.types.Object,
    body: bpy.types.Object,
    top_obj: bpy.types.Object | None,
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """Build subtle tapered wrist cuffs (`max_r=0.036`) matching the closed-neck sweater
    (no stiff collar bib — `mindfront_lusekofta.mhclo` has a clean closed neckline)."""
    inv_car = carrier.matrix_world.inverted()
    body_pts = _eval_mesh_verts_in(body, inv_car)
    _ = top_obj  # kept in signature for caller compatibility

    mat_cuff = _make_fabric_material("Passenger_WristCuff_Mat", (0.08, 0.10, 0.15), roughness=0.78)
    parts: list[bpy.types.Object] = []

    # Clean Tapered Wrist Cuffs (`max_r = 0.036`, zero wrist pinching)
    for side in ("L", "R"):
        pb_la = rig.pose.bones.get(f"lowerarm01.{side}")
        pb_wr = rig.pose.bones.get(f"wrist.{side}")
        if pb_la is None or pb_wr is None:
            continue
        la_c = inv_car @ (rig.matrix_world @ pb_la.head)
        wr_c = inv_car @ (rig.matrix_world @ pb_wr.head)
        arm_pts = _filter_segment_pts(body_pts, la_c, wr_c, max_perp=0.055)
        bm_cuff = bmesh.new()
        _add_sleeve_band_from_verts(
            bm_cuff,
            arm_pts,
            la_c,
            wr_c,
            (0.84, 0.99),
            pad_m=0.0045,
            thick_m=0.0028,
            n_rings=3,
            n_seg=18,
            mat_index=0,
            min_r=0.025,
            max_r=0.036,
        )
        cuff_obj = _new_bmesh_obj(f"Passenger_WristCuff_{side}", bm_cuff, mat_cuff, carrier.matrix_world, col)
        _parent_to_bone_preserve_world(cuff_obj, rig, f"lowerarm01.{side}")
        parts.append(cuff_obj)

    return parts


# ============================================================================
# X-RAY BODY & FADE-OUT SHADER PATCHING (AGENTS.MD SEC 3 COMPLIANT)
# ============================================================================
def _patch_xray_materials(
    objs: list[bpy.types.Object],
    is_fade_out: bool,
    xray_nodes: list[bpy.types.Node],
) -> None:
    """Patch materials with an `XRayBody` ShaderNodeValue (`0.0` opaque -> `1.0` translucent cyan X-ray)
    and mandatory `LightPath.Is Shadow Ray -> Transparent BSDF` bypass."""
    for obj in objs:
        if obj is None:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes or mat.node_tree is None:
                continue
            if mat.users > 1:
                mat = mat.copy()
                slot.material = mat
            nt = mat.node_tree

            # Skip if already patched
            existing = next((n for n in nt.nodes if n.name == "XRayBody"), None)
            if existing is not None:
                if existing not in xray_nodes:
                    xray_nodes.append(existing)
                continue

            out_node = next(
                (n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and getattr(n, "is_active_output", False)),
                None,
            )
            if out_node is None:
                out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if out_node is None:
                continue

            surf_in = out_node.inputs.get("Surface")
            if surf_in is None or not surf_in.is_linked:
                continue
            orig_socket = surf_in.links[0].from_socket
            nt.links.remove(surf_in.links[0])

            val = nt.nodes.new("ShaderNodeValue")
            val.name = "XRayBody"
            val.label = "XRayBody"
            val.outputs[0].default_value = 0.0

            lp = nt.nodes.new("ShaderNodeLightPath")
            tr_clear = nt.nodes.new("ShaderNodeBsdfTransparent")
            tr_clear.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

            if is_fade_out:
                # Fade Eyes/Teeth/Tongue/Hair/Brows/Lashes cleanly out during Beat 4 X-ray
                sh_boost = nt.nodes.new("ShaderNodeMath")
                sh_boost.operation = "MULTIPLY"
                sh_boost.use_clamp = True
                nt.links.new(val.outputs[0], sh_boost.inputs[0])
                nt.links.new(lp.outputs["Is Shadow Ray"], sh_boost.inputs[1])

                fac_max = nt.nodes.new("ShaderNodeMath")
                fac_max.operation = "MAXIMUM"
                fac_max.use_clamp = True
                nt.links.new(val.outputs[0], fac_max.inputs[0])
                nt.links.new(sh_boost.outputs["Value"], fac_max.inputs[1])

                mix_out = nt.nodes.new("ShaderNodeMixShader")
                nt.links.new(fac_max.outputs["Value"], mix_out.inputs["Fac"])
                nt.links.new(orig_socket, mix_out.inputs[1])
                nt.links.new(tr_clear.outputs["BSDF"], mix_out.inputs[2])
                nt.links.new(mix_out.outputs["Shader"], surf_in)
            else:
                # Hollow razor Fresnel rim via AddShader (zero opaque center fill -> crystal-clear skeleton!)
                # In Blender LayerWeight.Facing is 0.0 at camera-facing center and 1.0 at grazing edge.
                geom = nt.nodes.new("ShaderNodeNewGeometry")
                not_bf = nt.nodes.new("ShaderNodeMath")
                not_bf.operation = "SUBTRACT"
                not_bf.use_clamp = True
                not_bf.inputs[0].default_value = 1.0
                nt.links.new(geom.outputs["Backfacing"], not_bf.inputs[1])

                lw = nt.nodes.new("ShaderNodeLayerWeight")
                lw.inputs["Blend"].default_value = 0.35

                pw = nt.nodes.new("ShaderNodeMath")
                pw.operation = "POWER"
                pw.use_clamp = True
                pw.inputs[1].default_value = 3.8
                nt.links.new(lw.outputs["Facing"], pw.inputs[0])

                cam_front = nt.nodes.new("ShaderNodeMath")
                cam_front.operation = "MULTIPLY"
                cam_front.use_clamp = True
                nt.links.new(lp.outputs["Is Camera Ray"], cam_front.inputs[0])
                nt.links.new(not_bf.outputs["Value"], cam_front.inputs[1])

                rim_gate = nt.nodes.new("ShaderNodeMath")
                rim_gate.operation = "MULTIPLY"
                rim_gate.use_clamp = True
                nt.links.new(cam_front.outputs["Value"], rim_gate.inputs[0])
                nt.links.new(pw.outputs["Value"], rim_gate.inputs[1])

                rim_strength = nt.nodes.new("ShaderNodeMath")
                rim_strength.operation = "MULTIPLY"
                rim_strength.use_clamp = False
                rim_strength.inputs[1].default_value = 1.15
                nt.links.new(rim_gate.outputs["Value"], rim_strength.inputs[0])

                tr_xray = nt.nodes.new("ShaderNodeBsdfTransparent")
                tr_xray.inputs["Color"].default_value = (0.96, 0.99, 1.0, 1.0)

                em_xray = nt.nodes.new("ShaderNodeEmission")
                em_xray.inputs["Color"].default_value = (0.08, 0.68, 1.0, 1.0)
                nt.links.new(rim_strength.outputs["Value"], em_xray.inputs["Strength"])

                add_xray = nt.nodes.new("ShaderNodeAddShader")
                nt.links.new(tr_xray.outputs["BSDF"], add_xray.inputs[0])
                nt.links.new(em_xray.outputs["Emission"], add_xray.inputs[1])

                body_mix = nt.nodes.new("ShaderNodeMixShader")
                nt.links.new(val.outputs[0], body_mix.inputs["Fac"])
                nt.links.new(orig_socket, body_mix.inputs[1])
                nt.links.new(add_xray.outputs["Shader"], body_mix.inputs[2])

                # Shadow ray bypass active whenever XRayBody > 0
                sh_scale = nt.nodes.new("ShaderNodeMath")
                sh_scale.operation = "MULTIPLY"
                sh_scale.use_clamp = True
                sh_scale.inputs[1].default_value = 2.0
                nt.links.new(val.outputs[0], sh_scale.inputs[0])

                sh_gate = nt.nodes.new("ShaderNodeMath")
                sh_gate.operation = "MULTIPLY"
                sh_gate.use_clamp = True
                nt.links.new(sh_scale.outputs["Value"], sh_gate.inputs[0])
                nt.links.new(lp.outputs["Is Shadow Ray"], sh_gate.inputs[1])

                final_mix = nt.nodes.new("ShaderNodeMixShader")
                nt.links.new(sh_gate.outputs["Value"], final_mix.inputs["Fac"])
                nt.links.new(body_mix.outputs["Shader"], final_mix.inputs[1])
                nt.links.new(tr_clear.outputs["BSDF"], final_mix.inputs[2])
                nt.links.new(final_mix.outputs["Shader"], surf_in)

            xray_nodes.append(val)


# ============================================================================
# ANALYTICAL ARMATURE IK, SPINE COMPRESSION & FINGER ARTICULATION
# ============================================================================
def _pose_bone_abs_rig(
    rig: bpy.types.Object,
    bone_name: str,
    r_target_rig: Matrix,
    frame: int | None = None,
) -> None:
    """Orient `bone_name` so its armature-space orientation becomes `r_target_rig @ M_rest`."""
    pb = rig.pose.bones.get(bone_name)
    if pb is None:
        return
    pb.rotation_mode = "QUATERNION"
    m_rest = pb.bone.matrix_local.to_3x3()
    if pb.parent is not None:
        r_parent_rig = pb.parent.matrix.to_3x3() @ pb.parent.bone.matrix_local.to_3x3().inverted()
        r_rel = r_parent_rig.inverted() @ r_target_rig
    else:
        r_rel = r_target_rig
    pb.rotation_quaternion = (m_rest.inverted() @ r_rel @ m_rest).to_quaternion()
    if frame is not None:
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def _basis_from_dir_hinge(y_dir: Vector, hinge_x: Vector) -> Matrix:
    y_ax = y_dir.normalized()
    x_ax = hinge_x - y_ax * hinge_x.dot(y_ax)
    if x_ax.length < 1e-6:
        ref = Vector((1.0, 0.0, 0.0)) if abs(y_ax.x) < 0.85 else Vector((0.0, 1.0, 0.0))
        x_ax = y_ax.cross(ref)
    x_ax.normalize()
    z_ax = x_ax.cross(y_ax).normalized()
    return Matrix((x_ax, y_ax, z_ax)).transposed()


def _basis_from_dir_pole(y_dir: Vector, pole_z: Vector) -> Matrix:
    y_ax = y_dir.normalized()
    z_ax = pole_z - y_ax * pole_z.dot(y_ax)
    if z_ax.length < 1e-6:
        ref = Vector((0.0, 0.0, 1.0)) if abs(y_ax.z) < 0.85 else Vector((1.0, 0.0, 0.0))
        z_ax = ref - y_ax * ref.dot(y_ax)
    z_ax.normalize()
    x_ax = y_ax.cross(z_ax).normalized()
    return Matrix((x_ax, y_ax, z_ax)).transposed()


def _carrier_pt_to_rig(rig_rel_carrier: Matrix, pt_c: Vector | tuple[float, float, float]) -> Vector:
    return rig_rel_carrier.inverted() @ Vector(pt_c)


def _carrier_dir_to_rig(rig_rel_carrier: Matrix, dir_c: Vector | tuple[float, float, float]) -> Vector:
    return (rig_rel_carrier.to_3x3().inverted() @ Vector(dir_c)).normalized()


def _apply_finger_curl(
    rig: bpy.types.Object,
    side: str,
    curl: float,
    splay: float = 0.0,
    frame: int | None = None,
) -> None:
    """Continuous finger curl `0.0` (open) .. `1.0` (tight handrail/fist grip)."""
    side_u = "L" if side.upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0
    c = C.clamp(float(curl), 0.0, 1.0)
    sp = float(splay)

    has_f5 = f"finger5-1.{side_u}" in rig.pose.bones
    digit_indices = (2, 3, 4, 5) if has_f5 else (1, 2, 3, 4)
    for d_slot, f_idx in enumerate(digit_indices):
        base_deg = C.lerp(4.0, 68.0, c) + d_slot * 4.0
        splay_deg = (1.5 - d_slot * 1.0) * sp * side_sgn
        for seg, mult in ((1, 0.78), (2, 1.08), (3, 0.72)):
            bname = f"finger{f_idx}-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                rz_rad = math.radians(splay_deg) if seg == 1 else 0.0
                pb.rotation_euler = Euler((math.radians(base_deg * mult), 0.0, rz_rad), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)

    thm_deg = C.lerp(6.0, 42.0, c)
    if has_f5:
        for seg, mult in ((1, 0.45), (2, 0.90), (3, 0.85)):
            bname = f"finger1-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(thm_deg * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)
    for tseg, mult in (("thumb02", 0.90), ("thumb03", 1.05)):
        bname = f"{tseg}.{side_u}"
        if bname in rig.pose.bones:
            pb = rig.pose.bones[bname]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((math.radians(thm_deg * mult), 0.0, 0.0), "XYZ")
            if frame is not None:
                pb.keyframe_insert("rotation_euler", frame=frame)


def _pose_spine_and_head(
    rig: bpy.types.Object,
    torso_fwd_deg: float,
    torso_side_deg: float,
    torso_twist_deg: float,
    head_look_up_deg: float,
    head_yaw_deg: float,
    head_roll_deg: float,
    comp_cm: float,
    frame: int | None,
) -> None:
    """Pose spine (`spine05..01`), neck & head with optional `-comp_cm` spinal disc compression.
    Conventions:
      `torso_fwd_deg > 0` bends forward (`+rx`); `head_look_up_deg > 0` tilts face upward (`-rx`)."""
    rx_sp = float(torso_fwd_deg)
    mc.set_bone_char_rot(rig, "spine05", rx_sp * 0.24, torso_side_deg * 0.35, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine04", rx_sp * 0.22, torso_side_deg * 0.25, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine03", rx_sp * 0.20, torso_side_deg * 0.20, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine02", rx_sp * 0.18, torso_side_deg * 0.10, torso_twist_deg * 0.15, frame=frame)
    mc.set_bone_char_rot(rig, "spine01", rx_sp * 0.16, torso_side_deg * 0.10, torso_twist_deg * 0.10, frame=frame)

    # Counter-pitch neck & head so head holds desired gaze angle cleanly even during forward squat
    neck_rx = -rx_sp * 0.35 - head_look_up_deg * 0.34
    head_rx = -rx_sp * 0.35 - head_look_up_deg * 0.66
    mc.set_bone_char_rot(rig, "neck01", neck_rx * 0.60, 0.0, head_yaw_deg * 0.32, frame=frame)
    if "neck02" in rig.pose.bones:
        mc.set_bone_char_rot(rig, "neck02", neck_rx * 0.40, 0.0, head_yaw_deg * 0.20, frame=frame)
    mc.set_bone_char_rot(rig, "head", head_rx, head_roll_deg, head_yaw_deg * 0.48, frame=frame)

    # Axial spinal compression along each vertebra's downward axis (`0..3.5 cm` in Beat 4)
    comp_m = max(0.0, float(comp_cm)) * 0.01
    active_segs = [b for b in _SPINE_SEG_BONES if b in rig.pose.bones and not rig.data.bones[b].use_connect]
    if active_segs:
        per_seg_z = -comp_m / float(len(active_segs))
        for bname in active_segs:
            pb = rig.pose.bones[bname]
            if comp_m > 1e-6:
                pb.location = pb.bone.matrix_local.to_3x3().inverted() @ Vector((0.0, 0.0, per_seg_z))
            else:
                pb.location = Vector((0.0, 0.0, 0.0))
            if frame is not None:
                pb.keyframe_insert("location", frame=frame)


def _solve_arm_ik(
    rig: bpy.types.Object,
    rig_rel_carrier: Matrix,
    side: str,
    grip_target_c: Vector | tuple[float, float, float],
    hand_fwd_c: Vector | tuple[float, float, float],
    hand_palm_c: Vector | tuple[float, float, float],
    pole_c: Vector | tuple[float, float, float],
    clav_elev_deg: float,
    clav_protract_deg: float,
    grip_curl: float,
    frame: int | None,
) -> None:
    """Closed-loop 2-bone analytical arm + wrist IK solver calibrated to palm grip center."""
    side_u = "L" if side.upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0

    clav_ry = -side_sgn * clav_elev_deg
    clav_rz = -side_sgn * clav_protract_deg
    mc.set_bone_char_rot(rig, f"clavicle.{side_u}", 0.0, clav_ry, clav_rz, frame=frame)
    for tw in (f"upperarm02.{side_u}", f"lowerarm02.{side_u}"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    bpy.context.view_layer.update()

    b_ua = rig.data.bones[f"upperarm01.{side_u}"]
    b_la = rig.data.bones[f"lowerarm01.{side_u}"]
    b_wr = rig.data.bones[f"wrist.{side_u}"]
    b_kn = rig.data.bones.get(f"finger3-1.{side_u}")
    b_f2 = rig.data.bones.get(f"finger2-1.{side_u}")
    b_f5 = rig.data.bones.get(f"finger5-1.{side_u}")

    rest_v1 = (b_la.head_local - b_ua.head_local).normalized()
    rest_v2 = (b_wr.head_local - b_la.head_local).normalized()
    rest_pole = Vector((0.0, 1.0, 0.0))
    rest_hinge1 = rest_v1.cross(rest_pole).normalized()
    f_rest1 = _basis_from_dir_hinge(rest_v1, rest_hinge1)
    f_rest2 = _basis_from_dir_hinge(rest_v2, rest_hinge1)

    hand_fwd_rest = ((b_kn.head_local - b_wr.head_local) if b_kn else (b_wr.tail_local - b_wr.head_local)).normalized()
    palm_guess = Vector((-side_sgn, 0.0, 0.0))
    if b_f2 is not None and b_f5 is not None:
        v_idx = b_f2.head_local - b_wr.head_local
        v_pnk = b_f5.head_local - b_wr.head_local
        cand_palm = v_pnk.cross(v_idx) if side_u == "L" else v_idx.cross(v_pnk)
        if cand_palm.length > 1e-6:
            cand_palm.normalize()
            if cand_palm.dot(palm_guess) > 0.0:
                palm_guess = cand_palm
    f_rest_hand = _basis_from_dir_pole(hand_fwd_rest, palm_guess)
    grip_dist = ((b_kn.head_local - b_wr.head_local).length + 0.012) if b_kn else 0.068

    tgt_fwd_r = _carrier_dir_to_rig(rig_rel_carrier, hand_fwd_c)
    tgt_palm_r = _carrier_dir_to_rig(rig_rel_carrier, hand_palm_c)
    f_tgt_hand = _basis_from_dir_pole(tgt_fwd_r, tgt_palm_r)
    r_wrist_tgt = f_tgt_hand @ f_rest_hand.inverted()

    grip_tgt_r = _carrier_pt_to_rig(rig_rel_carrier, grip_target_c)
    pole_r = _carrier_dir_to_rig(rig_rel_carrier, pole_c)
    wrist_tgt_r = grip_tgt_r - tgt_fwd_r * grip_dist - tgt_palm_r * 0.010

    pb_ua = rig.pose.bones[f"upperarm01.{side_u}"]
    pb_kn = rig.pose.bones.get(f"finger3-1.{side_u}")
    pb_tip = rig.pose.bones.get(f"finger3-2.{side_u}")

    L1 = (b_la.head_local - b_ua.head_local).length
    L2 = (b_wr.head_local - b_la.head_local).length

    _apply_finger_curl(rig, side_u, grip_curl, splay=0.15 if grip_curl < 0.35 else 0.0, frame=frame)

    for _it in range(2):
        S = pb_ua.head.copy()
        D = wrist_tgt_r - S
        d = min(max(D.length, abs(L1 - L2) + 0.008), L1 + L2 - 0.003)
        u_dir = D.normalized() if D.length > 1e-6 else Vector((0.0, -1.0, 0.0))
        p_perp = pole_r - u_dir * pole_r.dot(u_dir)
        if p_perp.length < 1e-6:
            p_perp = Vector((side_sgn * 0.5, 0.5, -0.7)).normalized()
            p_perp = (p_perp - u_dir * p_perp.dot(u_dir)).normalized()
        else:
            p_perp.normalize()

        cos_a = max(-0.998, min(0.998, (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)))
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        E = S + u_dir * (L1 * cos_a) + p_perp * (L1 * sin_a)
        W = S + u_dir * d

        dir1 = (E - S).normalized()
        dir2 = (W - E).normalized()
        hinge_x = u_dir.cross(p_perp).normalized()

        _pose_bone_abs_rig(rig, f"upperarm01.{side_u}", _basis_from_dir_hinge(dir1, hinge_x) @ f_rest1.inverted(), frame=frame)
        bpy.context.view_layer.update()
        _pose_bone_abs_rig(rig, f"lowerarm01.{side_u}", _basis_from_dir_hinge(dir2, hinge_x) @ f_rest2.inverted(), frame=frame)
        bpy.context.view_layer.update()
        _pose_bone_abs_rig(rig, f"wrist.{side_u}", r_wrist_tgt, frame=frame)
        bpy.context.view_layer.update()

        if pb_kn is not None:
            actual_grip = pb_kn.head + (pb_tip.head - pb_kn.head) * 0.45 if pb_tip is not None else pb_kn.head
            actual_grip = actual_grip + tgt_palm_r * 0.010
            err = grip_tgt_r - actual_grip
            if err.length < 0.0015:
                break
            wrist_tgt_r = wrist_tgt_r + err * 0.90


def _solve_leg_ik(
    rig: bpy.types.Object,
    rig_rel_carrier: Matrix,
    side: str,
    ankle_target_c: Vector | tuple[float, float, float],
    foot_pitch_deg: float = 0.0,
    toe_splay_deg: float = 6.0,
    min_knee_splay_x: float = 0.115,
    frame: int | None = None,
) -> None:
    """Analytical 2-bone leg IK keeping boots grounded cleanly at target ankle coordinates."""
    side_u = "L" if side.upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0

    for tw in (f"upperleg02.{side_u}", f"lowerleg02.{side_u}"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)

    b_ul = rig.data.bones[f"upperleg01.{side_u}"]
    b_ll = rig.data.bones[f"lowerleg01.{side_u}"]
    b_ft = rig.data.bones[f"foot.{side_u}"]

    rest_v1 = (b_ll.head_local - b_ul.head_local).normalized()
    rest_v2 = (b_ft.head_local - b_ll.head_local).normalized()
    rest_pole = Vector((0.0, -1.0, 0.0))
    rest_hinge = rest_v1.cross(rest_pole).normalized()
    f_rest1 = _basis_from_dir_hinge(rest_v1, rest_hinge)
    f_rest2 = _basis_from_dir_hinge(rest_v2, rest_hinge)

    pb_ul = rig.pose.bones[f"upperleg01.{side_u}"]
    S = pb_ul.head.copy()
    W_tgt = _carrier_pt_to_rig(rig_rel_carrier, ankle_target_c)
    L1 = (b_ll.head_local - b_ul.head_local).length
    L2 = (b_ft.head_local - b_ll.head_local).length

    D = W_tgt - S
    d = min(max(D.length, abs(L1 - L2) + 0.015), L1 + L2 - 0.004)
    u_dir = D.normalized() if D.length > 1e-6 else Vector((0.0, 0.0, -1.0))

    splay_w = 0.14
    E = S
    p_perp = Vector((0.0, -1.0, 0.0))
    for _ in range(6):
        pole_r = Vector((side_sgn * splay_w, -0.92, 0.28)).normalized()
        p_perp = pole_r - u_dir * pole_r.dot(u_dir)
        if p_perp.length < 1e-6:
            p_perp = Vector((side_sgn * 0.2, -1.0, 0.0)).normalized()
        else:
            p_perp.normalize()
        cos_a = max(-0.998, min(0.998, (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)))
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        E = S + u_dir * (L1 * cos_a) + p_perp * (L1 * sin_a)
        if abs(E.x) >= min_knee_splay_x:
            break
        splay_w += 0.10

    W = S + u_dir * d
    dir1 = (E - S).normalized()
    dir2 = (W - E).normalized()
    hinge_x = u_dir.cross(p_perp).normalized()

    _pose_bone_abs_rig(rig, f"upperleg01.{side_u}", _basis_from_dir_hinge(dir1, hinge_x) @ f_rest1.inverted(), frame=frame)
    bpy.context.view_layer.update()
    _pose_bone_abs_rig(rig, f"lowerleg01.{side_u}", _basis_from_dir_hinge(dir2, hinge_x) @ f_rest2.inverted(), frame=frame)
    bpy.context.view_layer.update()

    # Exact rig-space foot orientation (0 deg pitch = boot sole parallel to rig XY floor)
    r_foot_rig = Euler(
        (math.radians(foot_pitch_deg), 0.0, math.radians(side_sgn * toe_splay_deg)),
        "XYZ",
    ).to_matrix()
    _pose_bone_abs_rig(rig, f"foot.{side_u}", r_foot_rig, frame=frame)


def _pose_arms_fk(
    rig: bpy.types.Object,
    ua_l: tuple[float, float, float],
    la_l: tuple[float, float, float],
    ua_r: tuple[float, float, float],
    la_r: tuple[float, float, float],
    hand_l: str = "relaxed",
    hand_r: str = "relaxed",
    frame: int | None = None,
) -> None:
    """FK arm posing via `mc.apply_arm_with_clavicle` with twist bones cleanly zeroed."""
    for tw in ("upperarm02.L", "lowerarm02.L", "upperarm02.R", "lowerarm02.R"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    mc.apply_arm_with_clavicle(rig, "L", ua_l, la_l, frame=frame)
    mc.apply_arm_with_clavicle(rig, "R", ua_r, la_r, frame=frame)
    mc.apply_hand_gesture(rig, "L", hand_l, frame=frame)
    mc.apply_hand_gesture(rig, "R", hand_r, frame=frame)


def _pose_legs_airborne(
    rig: bpy.types.Object,
    thigh_l_rx: float,
    shin_l_rx: float,
    foot_l_rx: float,
    thigh_r_rx: float,
    shin_r_rx: float,
    foot_r_rx: float,
    splay_deg: float = 5.0,
    frame: int | None = None,
) -> None:
    """Direct joint posing for Beat 2 weightless float & Beat 3 airborne myth jump."""
    for tw in ("upperleg02.L", "lowerleg02.L", "upperleg02.R", "lowerleg02.R"):
        if tw in rig.pose.bones:
            pb_tw = rig.pose.bones[tw]
            pb_tw.rotation_mode = "QUATERNION"
            pb_tw.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            if frame is not None:
                pb_tw.keyframe_insert("rotation_quaternion", frame=frame)
    mc.set_bone_char_rot(rig, "upperleg01.L", thigh_l_rx, 0.0, splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "lowerleg01.L", shin_l_rx, 0.0, 0.0, frame=frame)
    mc.set_bone_char_rot(rig, "foot.L", foot_l_rx, 0.0, splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "upperleg01.R", thigh_r_rx, 0.0, -splay_deg, frame=frame)
    mc.set_bone_char_rot(rig, "lowerleg01.R", shin_r_rx, 0.0, 0.0, frame=frame)
    mc.set_bone_char_rot(rig, "foot.R", foot_r_rx, 0.0, -splay_deg, frame=frame)


# ============================================================================
# FLAT-ON-BACK ORIENTATION & FACIAL ACTING HELPERS
# ============================================================================
def _flat_back_matrix() -> Matrix:
    """Build exact 3x3 cab-local orientation matrix for Beat 5b-6 flat-on-back survival pose:
    +Z_char points from pelvis/feet toward head (-0.48, 0.34); +Y_char points straight down (0,0,-1)."""
    p_head = Vector(C.PASSENGER["flat_head"])
    p_feet = Vector(C.PASSENGER["flat_feet"])
    z_char = Vector((p_head.x - p_feet.x, p_head.y - p_feet.y, 0.0)).normalized()
    y_char = Vector((0.0, 0.0, -1.0))
    x_char = y_char.cross(z_char).normalized()
    return Matrix((x_char, y_char, z_char)).transposed()


def _near_hero_frame(f: int) -> bool:
    return any(abs(f - h) <= 5 for h in _HERO_BLINK_EXCLUSION_FRAMES)


def _apply_safe_face(
    rig: bpy.types.Object,
    f: int,
    jaw_deg: float,
    brow_lift: float,
    smile: float,
    eye_yaw_deg: float,
    eye_pitch_deg: float,
    lip_narrow: float = 0.0,
    squint_override: float | None = None,
) -> None:
    """Apply `mc.apply_facial_acting` with jaw clamped <= 11.5 deg and blinks suppressed on hero frames."""
    jaw_clamped = C.clamp(jaw_deg, 0.0, 11.5)
    t_val = (f - 1) / float(C.FPS)
    # Shift blink phase away from hero frames so eyes are always open and alert on hero beats
    blink_period = 999.0 if _near_hero_frame(f) else 3.4
    mc.apply_facial_acting(
        rig,
        t=t_val,
        frame=f,
        jaw_open_deg=jaw_clamped,
        brow_lift=brow_lift,
        smile=smile,
        blink_period=blink_period,
        blink_phase=0.65,
        eye_yaw_deg=eye_yaw_deg,
        eye_pitch_deg=eye_pitch_deg,
        lip_narrow=lip_narrow,
    )
    if squint_override is not None:
        mc.set_bone_char_rot(rig, "orbicularis03.L", squint_override, 0.0, 0.0, frame=f)
        mc.set_bone_char_rot(rig, "orbicularis03.R", squint_override + 0.5, 0.0, 0.0, frame=f)


# ============================================================================
# MAIN PUBLIC ENTRYPOINT
# ============================================================================
def build_passenger(col: bpy.types.Collection, total_frames: int = 1800) -> dict:
    """Build and animate the 1.78m adult male passenger inside `Passenger_Carrier` across 1..1800."""
    # 1. Create Passenger_Carrier synced LINEAR per frame to C.cab_world_location(f)
    carrier = bpy.data.objects.new("Passenger_Carrier", None)
    carrier.empty_display_type = "PLAIN_AXES"
    carrier.empty_display_size = 0.25
    carrier.location = Vector((0.0, 0.0, 0.0))
    carrier.rotation_mode = "XYZ"
    carrier.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    _link_obj(carrier, col)

    # 2. Build 1.78m MPFB2 Adult Male at cab-local origin before parenting to carrier
    stand_xy = tuple(C.PASSENGER["stand_xy"])
    stand_yaw_deg = float(C.PASSENGER["stand_yaw_deg"])
    cfg = {
        "gender": 1.0,
        "age": 0.56,
        "muscle": 0.56,
        "weight": 0.46,
        "height": 0.58,
        "caucasian": 1.0,
        "african": 0.0,
        "asian": 0.0,
        "target_height_m": PASSENGER_HEIGHT_M,
        "skin_assets": ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"],
        "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"],
        "hair_assets": ["short04.mhclo", "short02.mhclo", "short01.mhclo", "culturalibre_hair_01.mhclo"],
        "hair_tint_rgb": (0.14, 0.11, 0.09),
        "hair_tint_fac": 0.65,
        "hair_value_boost": 0.78,
        "hair_roughness": 0.42,
        "brow_tint_fac": 0.55,
        "iris_major_rgb": (0.06, 0.16, 0.28),
        "iris_minor_rgb": (0.02, 0.05, 0.11),
        "pants_assets": ["mindfront_male_trousers_2.mhclo", "mindfront_male_trousers_1.mhclo"],
        "pants_tint_rgb": (0.12, 0.14, 0.18),
        "pants_tint_fac": 0.62,
        "pants_sat_override": 0.22,
        "pants_value_boost": 0.80,
        "pants_roughness": 0.76,
        "pants_sheen": 0.22,
        "shirt_assets": ["mindfront_lusekofta.mhclo"],
        "shirt_tint_rgb": (0.10, 0.13, 0.19),
        "shirt_tint_fac": 0.72,
        "shirt_sat_override": 0.34,
        "shirt_value_boost": 0.84,
        "shirt_roughness": 0.78,
        "shirt_sheen": 0.24,
        "shoe_assets": ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
        "shoe_tint_rgb": (0.08, 0.08, 0.09),
        "shoe_tint_fac": 0.70,
        "shoe_sat_override": 0.14,
        "shoe_value_boost": 0.68,
        "shoe_roughness": 0.45,
        "shoe_sheen": 0.12,
    }
    res = mc.create_realistic_scientist(
        name="Passenger",
        floor_z=0.0,
        start_xy=stand_xy,
        yaw_deg=stand_yaw_deg,
        cfg=cfg,
    )
    rig: bpy.types.Object = res["rig"]
    body: bpy.types.Object = res["body"]
    equipped: dict[str, bpy.types.Object] = res["equipped"]
    base_z: float = float(res["base_z"])

    # Link all generated character objects cleanly into `col`
    _link_obj(rig, col)
    _link_obj(body, col)
    for obj in equipped.values():
        if obj is not None:
            _link_obj(obj, col)

    # Enforce visible_shadow=False on Eyes, Eyelashes, Eyebrows (AGENTS.md Sec 1)
    for k in ("Eyes", "Eyelashes", "Eyebrows"):
        if k in equipped and equipped[k] is not None:
            equipped[k].visible_shadow = False

    # Parent rig to Passenger_Carrier while carrier is at world origin (rig transforms = cab-local)
    rig.parent = carrier
    rig.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()

    # Unlock spine04..01 `use_connect=False` via EditBones for Beat 4 `-3.5 cm` axial compression
    targets = [b for b in _SPINE_SEG_BONES if b in rig.data.bones and rig.data.bones[b].use_connect]
    if targets:
        prev_active = bpy.context.view_layer.objects.active
        if prev_active is not None and prev_active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        for bname in targets:
            eb = rig.data.edit_bones.get(bname)
            if eb is not None:
                eb.use_connect = False
        bpy.ops.object.mode_set(mode="OBJECT")
        if prev_active is not None and prev_active.name in bpy.context.view_layer.objects:
            bpy.context.view_layer.objects.active = prev_active

    # Apply restricted UnderClothes_Shrink (-0.011 DISPLACE) BEFORE Armature
    _apply_underclothes_shrink(body, carrier)

    # Build tapered wrist cuffs (`max_r=0.036`) matching closed-neck sweater
    extras = _build_collar_and_cuffs(carrier, rig, body, equipped.get("Shirt"), col)

    # Patch XRayBody shader nodes: only Body stays as hollow rim; clothes + facial features fade 100% transparent
    xray_nodes: list[bpy.types.Node] = []
    solid_xray_objs = [body]
    fade_xray_objs = [
        equipped[k]
        for k in ("Pants", "Shirt", "Shoes", "Eyes", "Eyebrows", "Eyelashes", "Teeth", "Tongue", "Hair")
        if k in equipped and equipped[k] is not None
    ] + extras
    _patch_xray_materials(solid_xray_objs, is_fade_out=False, xray_nodes=xray_nodes)
    _patch_xray_materials(fade_xray_objs, is_fade_out=True, xray_nodes=xray_nodes)

    # Calibrate standing rest ankle coordinates in cab-local space
    bpy.context.view_layer.update()
    inv_car0 = carrier.matrix_world.inverted()
    foot_l_rest = inv_car0 @ (rig.matrix_world @ rig.pose.bones["foot.L"].head)
    foot_r_rest = inv_car0 @ (rig.matrix_world @ rig.pose.bones["foot.R"].head)
    cal_ankle_z = 0.5 * (foot_l_rest.z + foot_r_rest.z)
    ankle_l_stand = Vector((foot_l_rest.x, foot_l_rest.y, cal_ankle_z))
    ankle_r_stand = Vector((foot_r_rest.x, foot_r_rest.y, cal_ankle_z))

    # Calibrate standing rest floor contact (+/- 5 mm)
    mn_rest, mx_rest, _ = _eval_mesh_bounds_in(body, inv_car0)
    if "Shoes" in equipped and equipped["Shoes"] is not None:
        mn_sh, _, _ = _eval_mesh_bounds_in(equipped["Shoes"], inv_car0)
        rest_min_z = min(mn_rest.z, mn_sh.z)
    else:
        rest_min_z = mn_rest.z
    if abs(rest_min_z) > 0.005:
        base_z -= rest_min_z
        ankle_l_stand.z -= rest_min_z
        ankle_r_stand.z -= rest_min_z
        rig.location = Vector((stand_xy[0], stand_xy[1], base_z))
        bpy.context.view_layer.update()

    # Precompute Beat 5a braced handrail stand geometry & Beat 5b-6 flat-on-back transform
    handrail_y = float(C.CAB["handrail_y"])
    handrail_z = float(C.CAB["handrail_z"])
    brake_stand_xy = (-0.05, 0.34)
    brake_yaw_rad = math.radians(stand_yaw_deg)
    dx_b = brake_stand_xy[0] - stand_xy[0]
    dy_b = brake_stand_xy[1] - stand_xy[1]
    ankle_l_brake = Vector((ankle_l_stand.x + dx_b, ankle_l_stand.y + dy_b, ankle_l_stand.z))
    ankle_r_brake = Vector((ankle_r_stand.x + dx_b, ankle_r_stand.y + dy_b, ankle_r_stand.z))

    r_flat_3x3 = _flat_back_matrix()
    flat_eul = r_flat_3x3.to_euler("XYZ")
    flat_pelvis_c = Vector((0.0, 0.0, float(C.PASSENGER["flat_pelvis"][2])))
    flat_feet_c = Vector(C.PASSENGER["flat_feet"])
    flat_head_c = Vector(C.PASSENGER["flat_head"])
    flat_axis = Vector((flat_head_c.x - flat_feet_c.x, flat_head_c.y - flat_feet_c.y, 0.0)).normalized()
    flat_side = Vector((flat_axis.y, -flat_axis.x, 0.0)).normalized()  # +X_char (his left)
    # Initial guess centering rig along diagonal, then exact depsgraph XY+Z centering over floor (0,0)
    pelvis_rest_h = float(C.PASSENGER["pelvis_z"])
    flat_rig_xy = Vector((flat_pelvis_c.x, flat_pelvis_c.y)) - Vector((flat_axis.x, flat_axis.y)) * pelvis_rest_h
    flat_rig_loc = Vector((flat_rig_xy.x, flat_rig_xy.y, 0.125))
    flat_ankle_base = flat_pelvis_c - flat_axis * 0.58  # Natural bent-knee crash posture keeping heels far inside cab walls

    # Calibrate exact flat_rig_loc XY center & Z flush on floor z = 0.003m via depsgraph
    rig.location = flat_rig_loc
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = flat_eul
    bpy.context.view_layer.update()
    rig_rel_flat_tmp = Matrix.Translation(flat_rig_loc) @ r_flat_3x3.to_4x4()
    _pose_spine_and_head(rig, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, None)
    flat_ankle_l = flat_ankle_base + flat_side * 0.115 + Vector((0.0, 0.0, -0.025))
    flat_ankle_r = flat_ankle_base - flat_side * 0.115 + Vector((0.0, 0.0, -0.025))
    _solve_leg_ik(
        rig, rig_rel_flat_tmp, "L",
        flat_ankle_l,
        foot_pitch_deg=18.0, toe_splay_deg=8.0, min_knee_splay_x=0.125, frame=None,
    )
    _solve_leg_ik(
        rig, rig_rel_flat_tmp, "R",
        flat_ankle_r,
        foot_pitch_deg=18.0, toe_splay_deg=8.0, min_knee_splay_x=0.125, frame=None,
    )
    # Chest & forehead protective folded arms
    _solve_arm_ik(
        rig, rig_rel_flat_tmp, "L",
        flat_pelvis_c + flat_axis * 0.36 - flat_side * 0.07 + Vector((0.0, 0.0, 0.155)),
        -flat_side * 0.85 + flat_axis * 0.35, Vector((0.0, 0.0, -1.0)),
        flat_side * 0.85 + Vector((0.0, 0.0, -0.35)),
        clav_elev_deg=6.0, clav_protract_deg=12.0, grip_curl=0.45, frame=None,
    )
    _solve_arm_ik(
        rig, rig_rel_flat_tmp, "R",
        flat_pelvis_c + flat_axis * 0.56 + flat_side * 0.04 + Vector((0.0, 0.0, 0.175)),
        flat_side * 0.75 + flat_axis * 0.45, Vector((0.0, 0.0, -1.0)),
        -flat_side * 0.85 + Vector((0.0, 0.0, -0.35)),
        clav_elev_deg=8.0, clav_protract_deg=14.0, grip_curl=0.42, frame=None,
    )
    bpy.context.view_layer.update()
    flat_check_objs = [body] + [equipped[k] for k in ("Shoes", "Pants", "Shirt") if k in equipped] + extras
    bnds_flat = [_eval_mesh_bounds_in(o, inv_car0) for o in flat_check_objs]
    flat_min_x = min(b[0].x for b in bnds_flat)
    flat_max_x = max(b[1].x for b in bnds_flat)
    flat_min_y = min(b[0].y for b in bnds_flat)
    flat_max_y = max(b[1].y for b in bnds_flat)
    flat_min_z = min(b[0].z for b in bnds_flat)
    shift_x = -0.5 * (flat_min_x + flat_max_x)
    shift_y = -0.5 * (flat_min_y + flat_max_y)
    shift_z = 0.003 - flat_min_z
    flat_rig_loc = flat_rig_loc + Vector((shift_x, shift_y, shift_z))
    flat_pelvis_c = flat_pelvis_c + Vector((shift_x, shift_y, shift_z))
    flat_ankle_l = flat_ankle_l + Vector((shift_x, shift_y, shift_z))
    flat_ankle_r = flat_ankle_r + Vector((shift_x, shift_y, shift_z))

    # Restore standing pose before starting frame loop
    rig.location = Vector((stand_xy[0], stand_xy[1], base_z))
    rig.rotation_euler = Euler((0.0, 0.0, math.radians(stand_yaw_deg)), "XYZ")
    bpy.context.view_layer.update()

    # 3. Build optimized keyframe schedule (~450 frames: 5-frame base + dense 2-frame/3-frame windows on rapid motion)
    key_frames: set[int] = set(range(1, total_frames + 1, 5))
    for dense_a, dense_b, step in (
        (110, 155, 3),
        (308, 355, 3),
        (310, 336, 2),
        (645, 815, 4),
        (652, 672, 2),
        (744, 760, 2),
        (900, 995, 3),
        (900, 944, 2),
        (1234, 1295, 4),
        (1234, 1252, 2),
        (1383, 1460, 4),
        (1383, 1404, 2),
    ):
        for f in range(dense_a, min(total_frames, dense_b) + 1, step):
            key_frames.add(f)
    for f in (1, 110, 190, 312, 313, 325, 600, 601, 620, 665, 750, 795, 900, 901, 910, 935, 940, 985,
              1205, 1235, 1236, 1237, 1275, 1384, 1385, 1386, 1465, 1530, 1531, total_frames):
        if 1 <= f <= total_frames:
            key_frames.add(f)
    sorted_kfs = sorted(key_frames)

    # Animate Passenger_Carrier LINEAR per frame (1..total_frames)
    for f in range(1, total_frames + 1):
        carrier.location = Vector(C.cab_world_location(f))
        carrier.keyframe_insert("location", frame=f)
        carrier.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
        carrier.keyframe_insert("rotation_euler", frame=f)
    _set_fcurve_interp(carrier, interp="LINEAR")

    # 4. Animate Passenger_Rig, full body, hands & face across all 6 Beats (1..1800)
    stand_yaw_rad = math.radians(stand_yaw_deg)

    for f in sorted_kfs:
        t_s = (f - 1) / float(C.FPS)
        breath = math.sin(t_s * 1.9) * 1.2

        # Drive XRayBody shader value nodes
        xw = C.xray_body_weight(f)
        for xn in xray_nodes:
            xn.outputs[0].default_value = xw
            xn.outputs[0].keyframe_insert("default_value", frame=f)

        # ------------------------------------------------------------------
        # BEAT 1 (1..312): Hook — worried look up -> alarm startle -> squat
        # ------------------------------------------------------------------
        if f <= 312:
            u_sq = C.smootherstep(190.0, 312.0, f)
            u_al = C.smootherstep(110.0, 140.0, f) * (1.0 - C.smootherstep(185.0, 215.0, f))

            rig_loc = Vector((stand_xy[0], stand_xy[1], base_z - 0.22 * u_sq))
            rig_eul = Euler((0.0, 0.0, stand_yaw_rad), "XYZ")
            rig.location = rig_loc
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
            rig_rel_c = Matrix.Translation(rig_loc) @ rig_eul.to_matrix().to_4x4()

            torso_fwd = 2.0 + 18.0 * u_sq - 4.0 * u_al + breath * 0.4
            head_up = 20.0 * (1.0 - u_sq) + 6.0 * u_sq + 4.0 * u_al
            head_yaw = 5.0 * math.sin(t_s * 1.1) * (1.0 - u_sq)
            _pose_spine_and_head(rig, torso_fwd, 1.5 * (1.0 - u_sq), -3.0 * (1.0 - u_sq), head_up, head_yaw, 0.0, 0.0, f)

            _solve_leg_ik(rig, rig_rel_c, "L", ankle_l_stand, 0.0, 7.0, 0.115, frame=f)
            _solve_leg_ik(rig, rig_rel_c, "R", ankle_r_stand, 0.0, 7.0, 0.115, frame=f)

            # Arms: calm -> startled chest recoil -> cocked back athletic squat
            ua_l_rx = C.lerp(-14.0, -48.0, u_al) * (1.0 - u_sq) + 34.0 * u_sq
            la_l_rx = C.lerp(-24.0, -78.0, u_al) * (1.0 - u_sq) - 52.0 * u_sq
            ua_r_rx = C.lerp(-10.0, -52.0, u_al) * (1.0 - u_sq) + 34.0 * u_sq
            la_r_rx = C.lerp(-18.0, -82.0, u_al) * (1.0 - u_sq) - 52.0 * u_sq
            _pose_arms_fk(
                rig,
                (ua_l_rx, -8.0 * (1.0 - u_sq) - 10.0 * u_sq, 8.0 * u_al),
                (la_l_rx, 0.0, 10.0 * (1.0 - u_sq)),
                (ua_r_rx, 8.0 * (1.0 - u_sq) + 10.0 * u_sq, -8.0 * u_al),
                (la_r_rx, 0.0, -10.0 * (1.0 - u_sq)),
                hand_l="shock" if u_al > 0.35 else "relaxed",
                hand_r="shock" if u_al > 0.35 else "relaxed",
                frame=f,
            )

            jaw_b1 = 1.5 + 7.5 * u_al + 2.0 * u_sq
            brow_b1 = -0.25 + 1.05 * u_al - 0.15 * u_sq
            _apply_safe_face(rig, f, jaw_b1, brow_b1, -0.2, head_yaw * 0.4, -10.0 * (1.0 - u_sq), lip_narrow=0.25 * u_al)

        # ------------------------------------------------------------------
        # BEAT 2 (313..600): Cable Snap & Weightless Freefall Float
        # ------------------------------------------------------------------
        elif f <= 600:
            u_fl = C.smootherstep(313.0, 360.0, f)
            u_end = C.smootherstep(565.0, 600.0, f)
            hover_w = u_fl * (1.0 - u_end)

            dz_hover = 0.052 + 0.018 * math.sin((f - 335.0) * 0.055)
            dz_pelvis = C.lerp(-0.22, dz_hover, u_fl) * (1.0 - u_end) - 0.06 * u_end
            fl_pitch = math.radians(5.5 * hover_w * math.sin((f - 325.0) * 0.042))
            fl_roll = math.radians(3.5 * hover_w * math.cos((f - 325.0) * 0.036))

            rig_loc = Vector((stand_xy[0], stand_xy[1], base_z + dz_pelvis))
            rig_eul = Euler((fl_pitch, fl_roll, stand_yaw_rad), "XYZ")
            rig.location = rig_loc
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)

            _pose_spine_and_head(
                rig,
                torso_fwd_deg=C.lerp(14.0, 4.0, u_fl),
                torso_side_deg=3.0 * math.sin(t_s * 2.1) * hover_w,
                torso_twist_deg=8.0 * math.sin(t_s * 1.6) * hover_w,
                head_look_up_deg=10.0 + 4.0 * math.sin(t_s * 2.4),
                head_yaw_deg=14.0 * math.sin(t_s * 2.0) * hover_w,
                head_roll_deg=3.5 * math.cos(t_s * 1.8) * hover_w,
                comp_cm=0.0,
                frame=f,
            )

            # Weightless floating bent knees + toes floating 0.03..0.07m above floor
            kb_fl = C.lerp(0.55, 0.28, u_fl)
            _pose_legs_airborne(
                rig,
                thigh_l_rx=-32.0 * kb_fl - 6.0 * hover_w * math.sin(t_s * 3.1),
                shin_l_rx=62.0 * kb_fl + 10.0 * hover_w * math.sin(t_s * 3.1),
                foot_l_rx=-12.0 * kb_fl + 16.0 * hover_w,
                thigh_r_rx=-28.0 * kb_fl + 6.0 * hover_w * math.sin(t_s * 3.1),
                shin_r_rx=54.0 * kb_fl - 10.0 * hover_w * math.sin(t_s * 3.1),
                foot_r_rx=-10.0 * kb_fl + 18.0 * hover_w,
                splay_deg=6.0,
                frame=f,
            )

            # Tucked defensive panic float: elbows close to torso, upright forearms guarding face/chest (|X|<=0.60, |Y|<=0.52)
            u_arm_snap = C.smootherstep(312.0, 336.0, f)
            guard_w = u_arm_snap * (1.0 - u_end)
            wiggle = math.sin(t_s * 4.2) * 8.0 * hover_w
            ua_b2_rx = C.lerp(34.0, -14.0, u_arm_snap) * (1.0 - u_end)
            la_b2_rx = C.lerp(-52.0, -104.0, u_arm_snap) * (1.0 - u_end)
            _pose_arms_fk(
                rig,
                (ua_b2_rx + wiggle, -18.0, -14.0 * guard_w),
                (la_b2_rx, -26.0 * guard_w, -26.0 * guard_w),
                (ua_b2_rx - wiggle, 18.0, 14.0 * guard_w),
                (la_b2_rx, 26.0 * guard_w, 26.0 * guard_w),
                hand_l="shock",
                hand_r="shock",
                frame=f,
            )

            _apply_safe_face(
                rig, f,
                jaw_deg=9.8 * hover_w + 4.0 * (1.0 - hover_w),
                brow_lift=0.92,
                smile=-0.45,
                eye_yaw_deg=12.0 * math.sin(t_s * 2.8),
                eye_pitch_deg=-5.0,
                lip_narrow=0.35,
            )

        # ------------------------------------------------------------------
        # BEAT 3 (601..900): THE MYTH JUMP (+0.42m, Ceiling near-miss at 2.20m)
        # ------------------------------------------------------------------
        elif f <= 900:
            # 601..655 deep crouch (-0.30m); 665..750 jump (+0.52m); 750..795 ceiling slam; 795..900 fall
            if f <= 665:
                u_cr = C.smootherstep(601.0, 648.0, f)
                u_push = C.smootherstep(650.0, 665.0, f)
                dz_pelvis = -0.30 * u_cr * (1.0 - u_push)
                rig_loc = Vector((stand_xy[0], stand_xy[1], base_z + dz_pelvis))
                rig_eul = Euler((0.0, 0.0, stand_yaw_rad), "XYZ")
                rig.location = rig_loc
                rig.rotation_euler = rig_eul
                rig.keyframe_insert("location", frame=f)
                rig.keyframe_insert("rotation_euler", frame=f)
                rig_rel_c = Matrix.Translation(rig_loc) @ rig_eul.to_matrix().to_4x4()

                _pose_spine_and_head(rig, 24.0 * u_cr * (1.0 - u_push), 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, f)
                _solve_leg_ik(rig, rig_rel_c, "L", ankle_l_stand, 0.0, 7.0, 0.125, frame=f)
                _solve_leg_ik(rig, rig_rel_c, "R", ankle_r_stand, 0.0, 7.0, 0.125, frame=f)

                _pose_arms_fk(
                    rig,
                    (36.0 * u_cr * (1.0 - u_push) - 38.0 * u_push, -18.0, 12.0 * (1.0 - u_push) - 12.0 * u_push),
                    (-58.0 * u_cr * (1.0 - u_push) - 102.0 * u_push, -24.0 * u_push, 8.0 * (1.0 - u_push) - 22.0 * u_push),
                    (36.0 * u_cr * (1.0 - u_push) - 38.0 * u_push, 18.0, -12.0 * (1.0 - u_push) + 12.0 * u_push),
                    (-58.0 * u_cr * (1.0 - u_push) - 102.0 * u_push, 24.0 * u_push, -8.0 * (1.0 - u_push) + 22.0 * u_push),
                    hand_l="relaxed", hand_r="relaxed", frame=f,
                )
                _apply_safe_face(rig, f, 4.0 + 4.0 * u_push, 0.55, -0.3, 0.0, -8.0)
            else:
                jump_z = C.passenger_jump_offset_z(f)
                u_up = C.smootherstep(665.0, 745.0, f)
                u_hit = C.smootherstep(746.0, 758.0, f) * (1.0 - C.smootherstep(785.0, 825.0, f))
                u_fall = C.smootherstep(795.0, 900.0, f)

                rig_loc = Vector((stand_xy[0], stand_xy[1], base_z + jump_z))
                recoil_pitch = math.radians(-6.5 * u_hit)
                rig_eul = Euler((recoil_pitch, 0.0, stand_yaw_rad), "XYZ")
                rig.location = rig_loc
                rig.rotation_euler = rig_eul
                rig.keyframe_insert("location", frame=f)
                rig.keyframe_insert("rotation_euler", frame=f)

                # Head snaps back in ceiling-slam recoil at f=750..795
                torso_fwd = C.lerp(-4.0, 10.0, u_fall) + 8.0 * u_hit
                head_up = 10.0 * u_up + 22.0 * u_hit - 6.0 * u_fall
                _pose_spine_and_head(rig, torso_fwd, 0.0, 0.0, head_up, 0.0, 4.0 * u_hit, 1.5 * u_hit, f)

                # Legs extend downward during jump, bend slightly preparing for floor impact by f=900
                leg_bend = 0.12 + 0.32 * u_fall
                _pose_legs_airborne(
                    rig,
                    thigh_l_rx=-12.0 * leg_bend,
                    shin_l_rx=24.0 * leg_bend,
                    foot_l_rx=24.0 * (1.0 - u_fall) + 6.0 * u_fall,
                    thigh_r_rx=-14.0 * leg_bend,
                    shin_r_rx=28.0 * leg_bend,
                    foot_r_rx=24.0 * (1.0 - u_fall) + 6.0 * u_fall,
                    splay_deg=5.0,
                    frame=f,
                )

                # Arms in upright tucked shoulder/cheek brace (|X|<=0.60, |Y|<=0.48, Z<=2.15m < 2.44m ceiling)
                ua_b3_rx = -38.0 * (1.0 - u_fall) - 22.0 * u_fall
                la_b3_rx = -102.0 * (1.0 - u_fall) - 52.0 * u_fall
                _pose_arms_fk(
                    rig,
                    (ua_b3_rx, -18.0, -12.0),
                    (la_b3_rx, -24.0 * (1.0 - u_fall), -22.0),
                    (ua_b3_rx, 18.0, 12.0),
                    (la_b3_rx, 24.0 * (1.0 - u_fall), 22.0),
                    hand_l="shock", hand_r="shock", frame=f,
                )
                squint = 11.0 if u_hit > 0.35 else None
                _apply_safe_face(
                    rig, f,
                    jaw_deg=6.5 + 3.5 * u_hit,
                    brow_lift=-0.65 if u_hit > 0.35 else 0.85,
                    smile=-0.5,
                    eye_yaw_deg=0.0,
                    eye_pitch_deg=-6.0,
                    squint_override=squint,
                )

        # ------------------------------------------------------------------
        # BEAT 4 (901..1236): 28 G Pit Impact & Full-Body X-Ray Compression
        # ------------------------------------------------------------------
        elif f <= 1236:
            u_land = C.smootherstep(901.0, 935.0, f)
            u_buck = C.smootherstep(935.0, 985.0, f)
            u_reb = C.smootherstep(985.0, 1140.0, f)
            drop_z = C.lerp(C.passenger_jump_offset_z(901.0) * (1.0 - u_land), -0.18, u_buck) + 0.03 * u_reb

            rig_loc = Vector((stand_xy[0], stand_xy[1], base_z + drop_z))
            rig_eul = Euler((0.0, 0.0, stand_yaw_rad), "XYZ")
            rig.location = rig_loc
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
            rig_rel_c = Matrix.Translation(rig_loc) @ rig_eul.to_matrix().to_4x4()

            comp_cm = C.spine_impact_comp_cm(f)
            torso_fwd = 8.0 + 10.0 * u_buck - 3.0 * u_reb
            _pose_spine_and_head(rig, torso_fwd, 0.0, 0.0, 4.0 - 6.0 * u_buck, 0.0, 0.0, comp_cm, f)

            if f < 935:
                _pose_legs_airborne(
                    rig,
                    thigh_l_rx=C.lerp(-18.0, -28.0, u_land),
                    shin_l_rx=C.lerp(34.0, 56.0, u_land),
                    foot_l_rx=C.lerp(12.0, -24.0, u_land),
                    thigh_r_rx=C.lerp(-18.0, -28.0, u_land),
                    shin_r_rx=C.lerp(34.0, 56.0, u_land),
                    foot_r_rx=C.lerp(12.0, -24.0, u_land),
                    splay_deg=6.0,
                    frame=f,
                )
            else:
                _solve_leg_ik(rig, rig_rel_c, "L", ankle_l_stand, 0.0, 8.0, 0.125, frame=f)
                _solve_leg_ik(rig, rig_rel_c, "R", ankle_r_stand, 0.0, 8.0, 0.125, frame=f)

            _pose_arms_fk(
                rig,
                (-22.0 + 18.0 * u_buck, -14.0, 6.0),
                (-54.0 - 16.0 * u_buck, 0.0, 12.0),
                (-22.0 + 18.0 * u_buck, 14.0, -6.0),
                (-54.0 - 16.0 * u_buck, 0.0, -12.0),
                hand_l="relaxed", hand_r="relaxed", frame=f,
            )
            _apply_safe_face(rig, f, 8.0 * u_buck, -0.55, -0.45, 0.0, -2.0)

        # ------------------------------------------------------------------
        # BEAT 5a (1237..1385, HARD CUT at 1237): Braced on Rear Handrail
        # ------------------------------------------------------------------
        elif f <= 1385:
            u_jolt = C.smootherstep(1268.0, 1292.0, f) * (1.0 - C.smootherstep(1305.0, 1365.0, f))
            rig_loc = Vector((brake_stand_xy[0], brake_stand_xy[1], base_z - 0.055 - 0.045 * u_jolt))
            rig_eul = Euler((math.radians(4.0), 0.0, brake_yaw_rad), "XYZ")
            rig.location = rig_loc
            rig.rotation_euler = rig_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
            rig_rel_c = Matrix.Translation(rig_loc) @ rig_eul.to_matrix().to_4x4()

            _pose_spine_and_head(rig, 4.0 + 5.0 * u_jolt, 0.0, 0.0, 8.0, -4.0, 0.0, 0.0, f)
            _solve_leg_ik(rig, rig_rel_c, "L", ankle_l_brake, 0.0, 7.0, 0.115, frame=f)
            _solve_leg_ik(rig, rig_rel_c, "R", ankle_r_brake, 0.0, 7.0, 0.115, frame=f)

            # Both hands grip rear stainless handrail (y = 0.61, z = 0.92); facing -Y means +X_char = +X_cab
            _solve_arm_ik(
                rig, rig_rel_c, "L",
                (brake_stand_xy[0] + 0.21, handrail_y, handrail_z),
                (0.0, 0.65, -0.75), (0.0, -0.75, -0.65),
                (0.45, 0.45, 0.35),
                clav_elev_deg=2.0, clav_protract_deg=-8.0, grip_curl=0.88, frame=f,
            )
            _solve_arm_ik(
                rig, rig_rel_c, "R",
                (brake_stand_xy[0] - 0.21, handrail_y, handrail_z),
                (0.0, 0.65, -0.75), (0.0, -0.75, -0.65),
                (-0.45, 0.45, 0.35),
                clav_elev_deg=2.0, clav_protract_deg=-8.0, grip_curl=0.88, frame=f,
            )
            _apply_safe_face(rig, f, 3.0 + 3.0 * u_jolt, 0.25, -0.1, -4.0, -4.0)

        # ------------------------------------------------------------------
        # BEAT 5b - BEAT 6 (1386..1800, HARD CUT at 1386): LIE FLAT ON BACK
        # ------------------------------------------------------------------
        else:
            rig.location = flat_rig_loc
            rig.rotation_euler = flat_eul
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
            rig_rel_c = Matrix.Translation(flat_rig_loc) @ r_flat_3x3.to_4x4()

            # Gentle relaxed chest breathing & slight chin tuck gazing toward ceiling/grid
            _pose_spine_and_head(rig, 1.5 + breath * 0.3, 0.0, 0.0, -10.0, 2.0 * math.sin(t_s * 0.7), 0.0, 0.0, f)

            # Knees gently bent/relaxed so boot heels rest cleanly on floor z=0.0..0.08
            _solve_leg_ik(
                rig, rig_rel_c, "L", flat_ankle_l,
                foot_pitch_deg=18.0, toe_splay_deg=8.0, min_knee_splay_x=0.125, frame=f,
            )
            _solve_leg_ik(
                rig, rig_rel_c, "R", flat_ankle_r,
                foot_pitch_deg=18.0, toe_splay_deg=8.0, min_knee_splay_x=0.125, frame=f,
            )

            # Both arms/forearms folded over chest/forehead protecting face & skull
            _solve_arm_ik(
                rig, rig_rel_c, "L",
                flat_pelvis_c + flat_axis * 0.36 - flat_side * 0.07 + Vector((0.0, 0.0, 0.155)),
                -flat_side * 0.85 + flat_axis * 0.35, Vector((0.0, 0.0, -1.0)),
                flat_side * 0.85 + Vector((0.0, 0.0, -0.35)),
                clav_elev_deg=6.0, clav_protract_deg=12.0, grip_curl=0.45, frame=f,
            )
            _solve_arm_ik(
                rig, rig_rel_c, "R",
                flat_pelvis_c + flat_axis * 0.56 + flat_side * 0.04 + Vector((0.0, 0.0, 0.175)),
                flat_side * 0.75 + flat_axis * 0.45, Vector((0.0, 0.0, -1.0)),
                -flat_side * 0.85 + Vector((0.0, 0.0, -0.35)),
                clav_elev_deg=8.0, clav_protract_deg=14.0, grip_curl=0.42, frame=f,
            )
            _apply_safe_face(rig, f, 1.5, 0.15, 0.15, 2.0 * math.sin(t_s * 0.6), 4.0)

    # Set smooth BEZIER interpolation with CONSTANT snaps at hard-cut boundaries (1236, 1385)
    cut_frames = {1236, 1385}
    _set_fcurve_interp(rig, interp="BEZIER", constant_frames=cut_frames)
    for xn in xray_nodes:
        if xn.id_data:
            _set_fcurve_interp(xn.id_data, interp="BEZIER")

    # 5. Runtime Cab Bounds Assertion across Beat 2/3 wall/ceiling clearance & Beat 5b-6 Flat-On-Back Pose
    test_frames = [tf for tf in (332, 450, 700, 750, 755, 1465, 1650) if tf <= total_frames]
    if test_frames:
        scene = bpy.context.scene
        prev_frame = scene.frame_current
        try:
            for test_f in test_frames:
                scene.frame_set(test_f)
                bpy.context.view_layer.update()
                inv_car = carrier.matrix_world.inverted()
                for obj in flat_check_objs + [equipped[k] for k in ("Hair",) if k in equipped]:
                    mn, mx, _ = _eval_mesh_bounds_in(obj, inv_car)
                    max_ax = max(abs(mn.x), abs(mx.x))
                    max_ay = max(abs(mn.y), abs(mx.y))
                    if test_f <= 900:
                        assert max_ax <= 0.78, f"[Passenger Bounds] f={test_f} {obj.name} |x|={max_ax:.3f} > 0.78!"
                        assert max_ay <= 0.67, f"[Passenger Bounds] f={test_f} {obj.name} |y|={max_ay:.3f} > 0.67!"
                        assert mx.z <= 2.38, f"[Passenger Bounds] f={test_f} {obj.name} max_z={mx.z:.3f} > 2.38!"
                    else:
                        assert max_ax <= 0.82, f"[Passenger Bounds] f={test_f} {obj.name} |x|={max_ax:.3f} > 0.82!"
                        assert max_ay <= 0.67, f"[Passenger Bounds] f={test_f} {obj.name} |y|={max_ay:.3f} > 0.67!"
                        assert mn.z >= -0.015, f"[Passenger Bounds] f={test_f} {obj.name} min_z={mn.z:.3f} < -0.015!"
        finally:
            scene.frame_set(prev_frame)
            bpy.context.view_layer.update()

    return {
        "carrier": carrier,
        "rig": rig,
        "body": body,
        "equipped": equipped,
        "xray_nodes": xray_nodes,
        "harness_parts": extras,
    }
