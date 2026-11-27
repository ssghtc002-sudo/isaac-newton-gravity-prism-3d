"""
Pilot Character Animator for Episode #1:
"Why Ejecting From a Fighter Jet Shrinks You by 1 Inch" (30.0 s @ 60 fps = 1800 frames).

Public API:
    build_pilot(col: bpy.types.Collection, total_frames: int = 1800) -> dict

Returned dictionary keys & exact Blender types:
    - "carrier":       bpy.types.Object  (Pilot_Carrier Empty animated via C.pilot_carrier_*)
    - "rig":           bpy.types.Object  (Pilot_Rig MPFB2 armature, yaw_deg=180.0 inside carrier)
    - "body":          bpy.types.Object  (Pilot_Body 1.78m MakeHuman mesh)
    - "equipped":      dict[str, bpy.types.Object]  (Pilot_Eyes, Pilot_Eyebrows, Pilot_Eyelashes,
                       Pilot_Teeth, Pilot_Tongue, Pilot_Hair, Pilot_SuitPants, Pilot_SuitTop,
                       Pilot_Boots)
    - "helmet_parts":  list[bpy.types.Object]  (Pilot_Helmet_* parented to `head` bone)
    - "harness_parts": list[bpy.types.Object]  (Pilot_Harness_* torso, pelvis & G-suit leg wraps)
    - "xray_nodes":    list[bpy.types.Node]    (ShaderNodeValue `XRayBody` nodes across all pilot mats)
    - "sacrum_local":  tuple[float, float, float]  (carrier-local calibrated sacrum rest coords)
    - "c1_local":      tuple[float, float, float]  (carrier-local calibrated C1 skull-base rest coords)
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

import ejection_contract as C  # noqa: E402
import mpfb_characters as mc  # noqa: E402

PILOT_HEIGHT_M = 1.78
_SPINE_SEG_BONES = ("spine04", "spine03", "spine02", "spine01")


# ============================================================================
# COLLECTION, MESH & HIERARCHY HELPERS
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


def _set_fcurve_interp(obj_or_data, interp: str = "BEZIER", skip_prefix: str = "hide_") -> None:
    ad = getattr(obj_or_data, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if skip_prefix and fc.data_path.startswith(skip_prefix):
            continue
        for kp in fc.keyframe_points:
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


def _eval_mesh_bounds_in(obj: bpy.types.Object, inv_frame: Matrix) -> tuple[Vector, Vector, Vector]:
    """Evaluate deformed mesh bounds transformed into `inv_frame` coordinate space."""
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    em = eo.to_mesh()
    xf = inv_frame @ eo.matrix_world
    pts = [xf @ v.co for v in em.vertices]
    eo.to_mesh_clear()
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
    recalc_normals: bool = False,
) -> bpy.types.Object:
    """Finalize `bm` into a named mesh object with material slot(s) attached BEFORE `bm.to_mesh()`."""
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mats_list = mat if isinstance(mat, (list, tuple)) else [mat]
    for m in mats_list:
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
# PROCEDURAL PILOT PBR MATERIALS & LOCAL-SPACE X-RAY BODY PATCHER
# ============================================================================
def _set_bsdf(bsdf: bpy.types.Node, name: str, val) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def _rgba_mix_sockets(mix_node: bpy.types.Node):
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def _make_gear_pbr_mat(
    name: str,
    base_rgb: tuple[float, float, float],
    roughness: float = 0.65,
    metallic: float = 0.0,
    coat: float = 0.0,
    coat_rough: float = 0.25,
    sheen: float = 0.0,
    noise_scale: float = 140.0,
    bump_strength: float = 0.10,
    weave_grid: bool = False,
) -> bpy.types.Material:
    """Create local-space procedural PBR material for Nomex suit, helmet, webbing, or hardware."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Metallic", metallic)
    _set_bsdf(bsdf, "Roughness", roughness)
    if coat > 0.0:
        _set_bsdf(bsdf, "Coat Weight", coat)
        _set_bsdf(bsdf, "Coat Roughness", coat_rough)
    if sheen > 0.0:
        _set_bsdf(bsdf, "Sheen Weight", sheen)
        _set_bsdf(bsdf, "Sheen Roughness", 0.50)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    n_micro = nt.nodes.new("ShaderNodeTexNoise")
    n_micro.inputs["Scale"].default_value = noise_scale
    n_micro.inputs["Detail"].default_value = 7.0
    nt.links.new(tc.outputs["Object"], n_micro.inputs["Vector"])

    n_macro = nt.nodes.new("ShaderNodeTexNoise")
    n_macro.inputs["Scale"].default_value = 9.0
    n_macro.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], n_macro.inputs["Vector"])

    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    fac, ca, cb, cout = _rgba_mix_sockets(mix_col)
    ca.default_value = (*base_rgb, 1.0)
    cb.default_value = (
        min(1.0, base_rgb[0] * 1.18 + 0.015),
        min(1.0, base_rgb[1] * 1.18 + 0.015),
        min(1.0, base_rgb[2] * 1.18 + 0.015),
        1.0,
    )
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.35
    mr.inputs["From Max"].default_value = 0.72
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 0.38
    nt.links.new(n_macro.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], fac)
    nt.links.new(cout, bsdf.inputs["Base Color"])

    height_sock = n_micro.outputs["Fac"]
    if weave_grid:
        wave_a = nt.nodes.new("ShaderNodeTexWave")
        wave_a.wave_type = "BANDS"
        wave_a.bands_direction = "X"
        wave_a.inputs["Scale"].default_value = 48.0
        wave_a.inputs["Distortion"].default_value = 0.4
        nt.links.new(tc.outputs["Object"], wave_a.inputs["Vector"])

        wave_b = nt.nodes.new("ShaderNodeTexWave")
        wave_b.wave_type = "BANDS"
        wave_b.bands_direction = "Y"
        wave_b.inputs["Scale"].default_value = 48.0
        wave_b.inputs["Distortion"].default_value = 0.4
        nt.links.new(tc.outputs["Object"], wave_b.inputs["Vector"])

        add_w = nt.nodes.new("ShaderNodeMath")
        add_w.operation = "MULTIPLY_ADD"
        add_w.inputs[1].default_value = 0.5
        nt.links.new(wave_a.outputs["Fac"], add_w.inputs[0])
        nt.links.new(wave_b.outputs["Fac"], add_w.inputs[2])

        blend_h = nt.nodes.new("ShaderNodeMath")
        blend_h.operation = "MULTIPLY_ADD"
        blend_h.inputs[1].default_value = 0.45
        nt.links.new(add_w.outputs["Value"], blend_h.inputs[0])
        nt.links.new(n_micro.outputs["Fac"], blend_h.inputs[2])
        height_sock = blend_h.outputs["Value"]

    if bump_strength > 0.0:
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_strength
        bump.inputs["Distance"].default_value = 0.002
        nt.links.new(height_sock, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _make_nomex_suit_mat(name: str, src_obj: bpy.types.Object | None = None) -> bpy.types.Material:
    """Create CWU-27/P Olive-Sage Nomex flight-suit ripstop shader, preserving any baked fold normal map."""
    baked_nrm_img = None
    if src_obj is not None:
        for slot in src_obj.material_slots:
            smat = slot.material
            if not smat or not smat.use_nodes or not smat.node_tree:
                continue
            for node in smat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                    fp = node.image.name.lower() + " " + node.image.filepath.lower()
                    if any(tok in fp for tok in ("nrm", "norm", "normal")):
                        baked_nrm_img = node.image
                        break
            if baked_nrm_img is not None:
                break

    mat = _make_gear_pbr_mat(
        name=name,
        base_rgb=(0.048, 0.068, 0.036),
        roughness=0.80,
        sheen=0.45,
        noise_scale=210.0,
        bump_strength=0.04 if baked_nrm_img else 0.06,
        weave_grid=True,
    )
    if baked_nrm_img is not None:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        bump = next((n for n in nt.nodes if n.type == "BUMP"), None)
        if bump is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = baked_nrm_img
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = 0.85
            nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bump.inputs["Normal"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_smoked_visor_mat() -> bpy.types.Material:
    """Dark smoked polycarbonate raised sun-visor shader with crisp high-altitude clearcoat."""
    mat = bpy.data.materials.new("Mat_Pilot_HelmetVisor")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_bsdf(bsdf, "Base Color", (0.018, 0.022, 0.028, 1.0))
    _set_bsdf(bsdf, "Metallic", 0.30)
    _set_bsdf(bsdf, "Roughness", 0.04)
    _set_bsdf(bsdf, "Coat Weight", 0.95)
    _set_bsdf(bsdf, "Coat Roughness", 0.02)
    _set_bsdf(bsdf, "Transmission Weight", 0.35)
    _set_bsdf(bsdf, "IOR", 1.58)
    # Shadow bypass for transmissive visor polycarbonate
    lp = nt.nodes.new("ShaderNodeLightPath")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.35, 0.38, 0.42, 1.0)
    mix_s = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix_s.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix_s.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix_s.inputs[2])
    nt.links.new(mix_s.outputs["Shader"], out.inputs["Surface"])
    return mat


def _tune_pilot_skin(body_obj: bpy.types.Object) -> None:
    """Subtle sculpted sun-lit tone + tight BURLEY SSS so face stays detailed under golden sunlight."""
    for slot in body_obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        mname = mat.name.lower()
        if any(skip in mname for skip in ("nail", "genital")):
            continue
        if mat.users > 1:
            mat = mat.copy()
            slot.material = mat
        is_lip = "lip" in mname
        for node in mat.node_tree.nodes:
            if node.type != "GROUP":
                continue
            if getattr(node, "node_tree", None) is not None and node.node_tree.users > 1:
                node.node_tree = node.node_tree.copy()
            if not is_lip:
                if "colorMixIn" in node.inputs and not node.inputs["colorMixIn"].is_linked:
                    node.inputs["colorMixIn"].default_value = (0.36, 0.21, 0.13, 1.0)
                if "colorMixInStrength" in node.inputs and not node.inputs["colorMixInStrength"].is_linked:
                    node.inputs["colorMixInStrength"].default_value = 0.18
                if "Brightness" in node.inputs and not node.inputs["Brightness"].is_linked:
                    node.inputs["Brightness"].default_value = -0.07
                if "Contrast" in node.inputs and not node.inputs["Contrast"].is_linked:
                    node.inputs["Contrast"].default_value = 0.07
                if "Roughness" in node.inputs and not node.inputs["Roughness"].is_linked:
                    node.inputs["Roughness"].default_value = 0.43
                if "SSS strength" in node.inputs and not node.inputs["SSS strength"].is_linked:
                    node.inputs["SSS strength"].default_value = 0.11


def _patch_material_for_xray(
    mat: bpy.types.Material,
    mode: str,
    rim_peak: float = 0.14,
    patched_ids: set[int] | None = None,
    xray_nodes: list | None = None,
) -> None:
    """Patch `mat` with a `ShaderNodeValue` (`XRayBody`) driving X-ray transition + Shadow Ray bypass.

    mode:
      - "xray_glass": blends solid -> crystal-clear cyan Fresnel rim (`rim_peak`) + shadow bypass.
      - "fade_out":   smoothly fades distracting internal features (Eyes/Teeth/Tongue/Hair/Lashes)
                      to pure Transparent BSDF before peak Beat 4 X-ray (`f=915..1215`).
    """
    if mat is None or not mat.use_nodes or not mat.node_tree:
        return
    ptr = mat.as_pointer()
    if patched_ids is not None:
        if ptr in patched_ids:
            return
        patched_ids.add(ptr)

    nt = mat.node_tree
    if nt.nodes.get("XRayBody") is not None:
        return
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is None:
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        return
    surf_in = out.inputs.get("Surface")
    if surf_in is None or not surf_in.is_linked:
        return
    orig_socket = surf_in.links[0].from_socket
    nt.links.remove(surf_in.links[0])

    val_node = nt.nodes.new("ShaderNodeValue")
    val_node.name = "XRayBody"
    val_node.label = "XRayBody"
    val_node.outputs[0].default_value = 0.0
    if xray_nodes is not None:
        xray_nodes.append(val_node)

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (0.985, 0.996, 1.0, 1.0)

    if mode == "fade_out":
        # Fade completely clear by XRayBody >= 0.45 (around f=912..1218 inside 895..1236)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value = 0.0
        mr.inputs["From Max"].default_value = 0.45
        mr.inputs["To Min"].default_value = 0.0
        mr.inputs["To Max"].default_value = 1.0
        nt.links.new(val_node.outputs[0], mr.inputs["Value"])

        mix_out = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mr.outputs["Result"], mix_out.inputs["Fac"])
        nt.links.new(orig_socket, mix_out.inputs[1])
        nt.links.new(transp.outputs["BSDF"], mix_out.inputs[2])
        nt.links.new(mix_out.outputs["Shader"], surf_in)
        return

    # mode == "xray_glass": super-clear center + thin Fresnel cyan rim + LightPath shadow bypass
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.22

    rim_pow = nt.nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.inputs[1].default_value = 2.85
    nt.links.new(lw.outputs["Facing"], rim_pow.inputs[0])

    rim_mul = nt.nodes.new("ShaderNodeMath")
    rim_mul.operation = "MULTIPLY"
    rim_mul.use_clamp = True
    rim_mul.inputs[1].default_value = rim_peak
    nt.links.new(rim_pow.outputs["Value"], rim_mul.inputs[0])

    xray_emit = nt.nodes.new("ShaderNodeEmission")
    xray_emit.inputs["Color"].default_value = (0.14, 0.85, 1.0, 1.0)
    xray_emit.inputs["Strength"].default_value = 1.65

    xray_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(rim_mul.outputs["Value"], xray_mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], xray_mix.inputs[1])
    nt.links.new(xray_emit.outputs["Emission"], xray_mix.inputs[2])

    body_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(val_node.outputs[0], body_mix.inputs["Fac"])
    nt.links.new(orig_socket, body_mix.inputs[1])
    nt.links.new(xray_mix.outputs["Shader"], body_mix.inputs[2])

    # AGENTS.md Sec 3: LightPath.Is Shadow Ray -> Transparent BSDF when X-ray is active
    lp = nt.nodes.new("ShaderNodeLightPath")
    shad_ramp = nt.nodes.new("ShaderNodeMapRange")
    shad_ramp.clamp = True
    shad_ramp.inputs["From Min"].default_value = 0.05
    shad_ramp.inputs["From Max"].default_value = 0.35
    nt.links.new(val_node.outputs[0], shad_ramp.inputs["Value"])

    shad_fac = nt.nodes.new("ShaderNodeMath")
    shad_fac.operation = "MULTIPLY"
    shad_fac.use_clamp = True
    nt.links.new(lp.outputs["Is Shadow Ray"], shad_fac.inputs[0])
    nt.links.new(shad_ramp.outputs["Result"], shad_fac.inputs[1])

    final_mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(shad_fac.outputs["Value"], final_mix.inputs["Fac"])
    nt.links.new(body_mix.outputs["Shader"], final_mix.inputs[1])
    nt.links.new(transp.outputs["BSDF"], final_mix.inputs[2])
    nt.links.new(final_mix.outputs["Shader"], surf_in)


def _animate_xray_nodes(xray_nodes: list, total_frames: int) -> None:
    key_frames = sorted({
        1, 894, 895, 902, 910, 918, 925, 930,
        1200, 1207, 1215, 1223, 1230, 1236, 1237, total_frames,
    })
    seen_trees: set[int] = set()
    for vn in xray_nodes:
        out_sock = vn.outputs[0]
        for f in key_frames:
            if 1 <= f <= total_frames:
                out_sock.default_value = C.xray_body_weight(f)
                out_sock.keyframe_insert("default_value", frame=f)
        nt = vn.id_data
        nt_ptr = nt.as_pointer()
        if nt_ptr not in seen_trees:
            seen_trees.add(nt_ptr)
            _set_fcurve_interp(nt, "LINEAR", skip_prefix="")


# ============================================================================
# MPFB2 CHARACTER PREP: HEIGHT, SLEEVE CUFFS, HAIR TUCK & RENAME
# ============================================================================
def _normalize_pilot_height(actor: dict, target_height_m: float = PILOT_HEIGHT_M) -> None:
    """Scale `Pilot_Rig` uniformly so evaluated Body + Boots stature equals `target_height_m`."""
    rig = actor["rig"]
    body = actor["body"]
    shoes = actor["equipped"].get("Shoes")

    bpy.context.view_layer.update()
    b_min, b_max, _ = _eval_mesh_bounds_in(body, Matrix.Identity(4))
    s_min = _eval_mesh_bounds_in(shoes, Matrix.Identity(4))[0] if shoes is not None else b_min
    cur_min_z = min(b_min.z, s_min.z)
    cur_h = max(0.5, b_max.z - cur_min_z)

    rig.scale = rig.scale * (target_height_m / cur_h)
    bpy.context.view_layer.update()

    b_min2, b_max2, _ = _eval_mesh_bounds_in(body, Matrix.Identity(4))
    s_min2 = _eval_mesh_bounds_in(shoes, Matrix.Identity(4))[0] if shoes is not None else b_min2
    cur_min_z2 = min(b_min2.z, s_min2.z)
    rig.location.z += (0.0 - cur_min_z2)
    bpy.context.view_layer.update()

    actor["height"] = b_max2.z - cur_min_z2
    actor["base_z"] = rig.location.z


def _cinch_shirt_cuffs(rig: bpy.types.Object, shirt_obj: bpy.types.Object | None) -> None:
    """Taper loose boho wrist cuffs of `elvs_male_boho_top1` into crisp fitted flight-suit cuffs."""
    if shirt_obj is None or shirt_obj.type != "MESH":
        return
    bpy.context.view_layer.update()
    rig_from_shirt = rig.matrix_world.inverted() @ shirt_obj.matrix_world
    shirt_from_rig = rig_from_shirt.inverted()
    mesh = shirt_obj.data

    for side in ("L", "R"):
        pb_w = rig.pose.bones.get(f"wrist.{side}")
        pb_la2 = rig.pose.bones.get(f"lowerarm02.{side}") or rig.pose.bones.get(f"lowerarm01.{side}")
        if pb_w is None or pb_la2 is None:
            continue
        w_head = pb_w.head.copy()
        arm_dir = (w_head - pb_la2.head).normalized()
        for v in mesh.vertices:
            p_rig = rig_from_shirt @ v.co
            rel = p_rig - w_head
            proj = rel.dot(arm_dir)
            if -0.115 < proj < 0.075:
                rad_vec = rel - arm_dir * proj
                rad = rad_vec.length
                if 1e-5 < rad < 0.135:
                    u = max(0.0, min(1.0, (proj + 0.115) / 0.095))
                    w = u * u * (3.0 - 2.0 * u)
                    max_r = 0.034
                    new_rad = (rad * (1.0 - w) + max_r * w) if rad > max_r else rad
                    if proj > -0.045:
                        t_p = min(1.0, (proj + 0.045) / 0.120)
                        new_proj = -0.045 + 0.025 * t_p  # Monotonically ends <= -0.020m clear of wrist
                    else:
                        new_proj = proj
                    p_new = w_head + arm_dir * new_proj + rad_vec * (new_rad / rad)
                    v.co = shirt_from_rig @ p_new
    mesh.update()


def _tuck_shirt_hem(rig: bpy.types.Object, shirt_obj: bpy.types.Object | None) -> None:
    """Pull loose lower tunic skirt of `elvs_male_boho_top1` inward so flaps never hang into crotch crease."""
    if shirt_obj is None or shirt_obj.type != "MESH":
        return
    b_sac = rig.data.bones.get("spine05") or rig.data.bones.get("spine04")
    if b_sac is None:
        return
    bpy.context.view_layer.update()
    rig_from_shirt = rig.matrix_world.inverted() @ shirt_obj.matrix_world
    shirt_from_rig = rig_from_shirt.inverted()
    z_cut = b_sac.head_local.z + 0.06
    mesh = shirt_obj.data

    for v in mesh.vertices:
        p_rig = rig_from_shirt @ v.co
        if p_rig.z < z_cut:
            u = min(1.0, (z_cut - p_rig.z) / 0.05)
            w = u * u * (3.0 - 2.0 * u)
            s = 1.0 - 0.15 * w  # Smoothly reaches x *= 0.85, y *= 0.85 on lower tunic skirt
            p_rig.x *= s
            p_rig.y *= s
            v.co = shirt_from_rig @ p_rig
    mesh.update()


def _create_underclothes_shrink_mod(rig: bpy.types.Object, body: bpy.types.Object) -> bpy.types.Modifier:
    """Create `UnderClothes` vertex group & `DISPLACE` (`strength=-0.014`) at modifier index 0 BEFORE Armature.

    Restricted strictly to lower torso / pelvis / upper-thigh crease (`abs(x) < 0.24`,
    `z_hips_lo <= z <= z_hips_hi`) so arms, forearms, wrists, hands, chest, calves, neck & head stay
    at 0.0 weight (100% normal anatomical thickness, zero twig wrists).
    """
    bpy.context.view_layer.update()
    rig_from_body = rig.matrix_world.inverted() @ body.matrix_world

    body_pts_rig = [rig_from_body @ v.co for v in body.data.vertices]
    z_crown = max(p.z for p in body_pts_rig)
    z_sole = min(p.z for p in body_pts_rig)
    z_foot_cut = z_sole + 0.06
    z_hips_lo = z_foot_cut + 0.45  # ~mid-thigh / above knee
    z_hips_hi = z_crown - 0.62     # ~mid-ribcage / below pectorals

    vg = body.vertex_groups.get("UnderClothes") or body.vertex_groups.new(name="UnderClothes")
    max_weighted_z = -1e9
    min_weighted_z = 1e9
    max_weighted_x = 0.0
    for v, p in zip(body.data.vertices, body_pts_rig):
        ax = abs(p.x)
        if ax >= 0.24 or p.z <= z_hips_lo or p.z >= z_hips_hi:
            continue

        w = 1.0
        # Smooth vertical falloff at bottom (mid-thigh) and top (mid-ribcage)
        if p.z < z_hips_lo + 0.06:
            u = (p.z - z_hips_lo) / 0.06
            w *= u * u * (3.0 - 2.0 * u)
        if p.z > z_hips_hi - 0.06:
            u = (z_hips_hi - p.z) / 0.06
            w *= u * u * (3.0 - 2.0 * u)
        # Smooth lateral falloff at |x| = 0.20..0.24 so arms/elbows never receive shrink weight
        if ax > 0.20:
            u = (0.24 - ax) / 0.04
            w *= u * u * (3.0 - 2.0 * u)

        if w > 1e-4:
            vg.add([v.index], float(w), "REPLACE")
            max_weighted_z = max(max_weighted_z, p.z)
            min_weighted_z = min(min_weighted_z, p.z)
            max_weighted_x = max(max_weighted_x, ax)

    assert max_weighted_z <= z_hips_hi and min_weighted_z >= z_hips_lo, (
        f"UnderClothes leaked outside hip/pelvis box [{z_hips_lo:.3f}, {z_hips_hi:.3f}]: [{min_weighted_z:.3f}, {max_weighted_z:.3f}]"
    )
    assert max_weighted_x < 0.24, f"UnderClothes leaked beyond |x|<0.24: {max_weighted_x:.3f}"

    disp = body.modifiers.new(name="UnderClothes_Shrink", type="DISPLACE")
    disp.direction = "NORMAL"
    disp.space = "LOCAL"
    disp.mid_level = 0.0
    disp.strength = -0.014
    disp.vertex_group = "UnderClothes"
    disp.show_viewport = False
    disp.show_render = False
    idx = len(body.modifiers) - 1
    if idx > 0:
        body.modifiers.move(idx, 0)
    return disp


def _tuck_hair_under_helmet(rig: bpy.types.Object, hair_obj: bpy.types.Object | None, eye_z_rig: float) -> None:
    """Pull crown/parietal hair cards inward slightly so zero hair pokes through HGU-55/P helmet shell."""
    if hair_obj is None or hair_obj.type != "MESH":
        return
    bpy.context.view_layer.update()
    rig_from_hair = rig.matrix_world.inverted() @ hair_obj.matrix_world
    hair_from_rig = rig_from_hair.inverted()
    head_pb = rig.pose.bones.get("head")
    center_rig = Vector((0.0, 0.01, eye_z_rig + 0.02)) if head_pb is None else Vector((0.0, 0.01, head_pb.head.z + 0.07))

    mesh = hair_obj.data
    for v in mesh.vertices:
        p_rig = rig_from_hair @ v.co
        if p_rig.z > eye_z_rig + 0.015:
            dv = p_rig - center_rig
            if dv.length > 0.05:
                # Keep front fringe/sideburns near hairline visible, tuck covered upper/back skull hair inward
                if p_rig.y > -0.08 or p_rig.z > eye_z_rig + 0.045:
                    p_rig = center_rig + dv * 0.86
                    v.co = hair_from_rig @ p_rig
    mesh.update()


def _unlock_spine_translation(rig: bpy.types.Object) -> None:
    """Disable `use_connect` on thoracic/lumbar spine bones so `-2.5 cm` disc compression translates cleanly."""
    targets = [b for b in _SPINE_SEG_BONES if b in rig.data.bones]
    if not any(rig.data.bones[b].use_connect for b in targets):
        return
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


def _rename_pilot_hierarchy(actor: dict) -> None:
    """Enforce `Pilot_*` prefix on every MPFB2 object and mesh datablock."""
    rig = actor["rig"]
    body = actor["body"]
    rig.name = "Pilot_Rig"
    rig.data.name = "Pilot_Rig_Armature"
    body.name = "Pilot_Body"
    body.data.name = "Pilot_Body_Mesh"

    slot_map = {
        "Eyes": "Pilot_Eyes",
        "Eyebrows": "Pilot_Eyebrows",
        "Eyelashes": "Pilot_Eyelashes",
        "Teeth": "Pilot_Teeth",
        "Tongue": "Pilot_Tongue",
        "Hair": "Pilot_Hair",
        "Pants": "Pilot_SuitPants",
        "Shirt": "Pilot_SuitTop",
        "Shoes": "Pilot_Boots",
    }
    for slot, obj in actor["equipped"].items():
        if obj is None:
            continue
        obj.name = slot_map.get(slot, f"Pilot_{slot}")
        if obj.data is not None:
            obj.data.name = f"{obj.name}_Mesh"


# ============================================================================
# FIGHTER PILOT GEAR: HGU-55/P HELMET + FLIGHT HARNESS + G-SUIT WRAPS
# ============================================================================
def _build_pilot_helmet(
    rig: bpy.types.Object,
    body: bpy.types.Object,
    equipped: dict,
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
) -> list[bpy.types.Object]:
    """Build sleek HGU-55/P Flight Helmet with raised smoked visor, bayonet receivers & boom mic on `head`."""
    bpy.context.view_layer.update()
    inv_rig = rig.matrix_world.inverted()

    _, _, eye_ctr_rig = _eval_mesh_bounds_in(equipped["Eyes"], inv_rig) if "Eyes" in equipped else (None, None, Vector((0.0, -0.10, 1.66)))
    _, _, teeth_ctr_rig = _eval_mesh_bounds_in(equipped["Teeth"], inv_rig) if "Teeth" in equipped else (None, None, Vector((0.0, -0.09, 1.59)))

    deps = bpy.context.evaluated_depsgraph_get()
    eb = body.evaluated_get(deps)
    em = eb.to_mesh()
    xf = inv_rig @ eb.matrix_world
    head_top_z = max((xf @ v.co).z for v in em.vertices)
    all_head = [xf @ v.co for v in em.vertices if (xf @ v.co).z > head_top_z - 0.28]
    skull_pts = [p for p in all_head if p.z > head_top_z - 0.155]
    jaw_pts = [p for p in all_head if abs(p.x) < 0.115 and p.z <= eye_ctr_rig.z + 0.02]
    eb.to_mesh_clear()

    sx_min, sx_max = min(p.x for p in skull_pts), max(p.x for p in skull_pts)
    sy_min, sy_max = min(p.y for p in skull_pts), max(p.y for p in skull_pts)  # -Y front, +Y rear
    sz_max = max(p.z for p in skull_pts)
    cx, cy = 0.5 * (sx_min + sx_max), 0.5 * (sy_min + sy_max)
    rx, ry = 0.5 * (sx_max - sx_min) + 0.014, 0.5 * (sy_max - sy_min) + 0.015

    brow_z = eye_ctr_rig.z + 0.022
    crown_z = sz_max + 0.028
    helm_cz = brow_z + 0.008
    helm_h = crown_z - brow_z

    # 1. HGU-55/P Shell (high brow clearance, sculpted ear domes, deep occipital drop)
    bm_sh = bmesh.new()
    nu, nv = 40, 18
    rings = []
    for j in range(nv + 1):
        vf = j / nv
        phi = vf * (math.pi * 0.5)
        cp, sp = math.cos(phi), math.sin(phi)
        row = []
        for i in range(nu):
            th = (i / nu) * math.tau
            ct, st = math.cos(th), math.sin(th)
            # HGU-55/P covers ears fully (drop at sides), high brow arch (-st > 0), deep rear nape (+st > 0)
            ear_drop = -0.036 * (ct ** 4)
            nape_drop = -0.064 * max(0.0, st) ** 1.35
            brow_arch = 0.016 * max(0.0, -st) ** 1.8
            rim_z_mod = (ear_drop + nape_drop + brow_arch) * ((1.0 - vf) ** 1.45)
            # Signature HGU-55/P lateral ear-cup bulge
            ear_bulge = 1.0 + 0.13 * (ct ** 6) * math.sin(vf * math.pi * 0.65)
            rear_flare = 1.0 + 0.04 * ((1.0 - vf) ** 2.6) * max(0.0, st)
            px = cx + ct * cp * rx * ear_bulge
            py = cy + st * cp * ry * rear_flare
            pz = helm_cz + sp * (crown_z - helm_cz) + rim_z_mod
            row.append(bm_sh.verts.new((px, py, pz)))
        rings.append(row)
    for j in range(nv):
        for i in range(nu):
            ni = (i + 1) % nu
            bm_sh.faces.new((rings[j][i], rings[j][ni], rings[j + 1][ni], rings[j + 1][i]))
    bmesh.ops.remove_doubles(bm_sh, verts=bm_sh.verts, dist=0.001)
    shell = _new_bmesh_obj("Pilot_Helmet_Shell", bm_sh, mats["helmet_shell"], rig.matrix_world, col)
    sol = shell.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.009
    sol.offset = 1.0

    # 2. Upper Visor Housing Cover (matte carbon bungee cover on forehead crown)
    bm_vc = bmesh.new()
    vu, vv = 24, 10
    v_rings = []
    for j in range(vv + 1):
        vf = 0.24 + 0.72 * (j / vv)
        phi = vf * (math.pi * 0.5)
        cp, sp = math.cos(phi), math.sin(phi)
        row = []
        for i in range(vu + 1):
            ang = ((i / vu) * 2.0 - 1.0) * math.radians(66.0)
            ct = math.sin(ang)     # +-X
            st = -math.cos(ang)    # -Y front
            px = cx + ct * cp * (rx + 0.015)
            py = cy + st * cp * (ry + 0.017)
            pz = helm_cz + sp * (crown_z - helm_cz) + 0.011
            row.append(bm_vc.verts.new((px, py, pz)))
        v_rings.append(row)
    for j in range(vv):
        for i in range(vu):
            bm_vc.faces.new((v_rings[j][i], v_rings[j][i + 1], v_rings[j + 1][i + 1], v_rings[j + 1][i]))
    visor_cover = _new_bmesh_obj("Pilot_Helmet_VisorCover", bm_vc, mats["helmet_dark"], rig.matrix_world, col)
    vsol = visor_cover.modifiers.new("Solidify", "SOLIDIFY")
    vsol.thickness = 0.005
    vsol.offset = 1.0

    # 3. RAISED Smoked Sun Visor tucked cleanly above brow (eyes & brows 100% unobstructed!)
    bm_vis = bmesh.new()
    g_u, g_v = 24, 6
    l_grid = []
    vis_z0 = brow_z + 0.036  # Strictly above eye_ctr + 0.058m!
    vis_rx = rx * 1.09
    vis_ry = ry * 1.11
    for j in range(g_v + 1):
        vf = j / g_v
        for i in range(g_u + 1):
            uf = (i / g_u) * 2.0 - 1.0
            ang = uf * math.radians(60.0)
            # Signature pilot visor nose cutout on lower edge
            nose_cut = 0.013 * math.exp(-(uf * 5.2) ** 2) * (1.0 - vf)
            tilt_back = vf * 0.022
            vx = cx + math.sin(ang) * vis_rx
            vy = cy - math.cos(ang) * (vis_ry - tilt_back) - 0.004
            vz = vis_z0 + vf * 0.044 + nose_cut
            l_grid.append(bm_vis.verts.new((vx, vy, vz)))
    for j in range(g_v):
        for i in range(g_u):
            idx0 = j * (g_u + 1) + i
            bm_vis.faces.new((l_grid[idx0], l_grid[idx0 + 1], l_grid[idx0 + g_u + 2], l_grid[idx0 + g_u + 1]))
    visor = _new_bmesh_obj("Pilot_Helmet_Visor", bm_vis, mats["helmet_visor"], rig.matrix_world, col)
    v_sol = visor.modifiers.new("Solidify", "SOLIDIFY")
    v_sol.thickness = 0.0035

    # 4. Hardware: Side Bayonet Receivers, Center Visor Slider Knob & Sleek Right-Cheek Boom Mic
    bm_hw = bmesh.new()
    # Center visor detent slider
    r_knob = bmesh.ops.create_cube(bm_hw, size=1.0)
    bmesh.ops.scale(bm_hw, vec=(0.020, 0.012, 0.024), verts=r_knob["verts"])
    bmesh.ops.translate(bm_hw, vec=(cx, cy - ry - 0.018, helm_cz + helm_h * 0.62), verts=r_knob["verts"])
    # Side bayonet receivers (+X left cheek, -X right cheek)
    for sgn in (-1.0, 1.0):
        rb = bmesh.ops.create_cube(bm_hw, size=1.0)
        bmesh.ops.scale(bm_hw, vec=(0.014, 0.042, 0.026), verts=rb["verts"])
        bmesh.ops.translate(bm_hw, vec=(cx + sgn * (rx * 1.04), cy - 0.045, eye_ctr_rig.z - 0.025), verts=rb["verts"])

    # Sleek boom microphone along pilot's RIGHT cheek (-X_r) stopping clear beside mouth corner
    mic_pts = [
        Vector((cx - rx * 1.05, cy - 0.045, eye_ctr_rig.z - 0.032)),
        Vector((cx - rx * 0.88, teeth_ctr_rig.y - 0.018, teeth_ctr_rig.z - 0.018)),
        Vector((cx - 0.056, teeth_ctr_rig.y - 0.032, teeth_ctr_rig.z - 0.008)),
    ]
    for a, b in zip(mic_pts, mic_pts[1:]):
        seg_v = b - a
        rc = bmesh.ops.create_cone(
            bm_hw, cap_ends=True, cap_tris=False, segments=8,
            radius1=0.0032, radius2=0.0028, depth=seg_v.length,
        )
        rot_q = Vector((0, 0, 1)).rotation_difference(seg_v.normalized())
        bmesh.ops.rotate(bm_hw, cent=(0, 0, 0), matrix=rot_q.to_matrix(), verts=rc["verts"])
        bmesh.ops.translate(bm_hw, vec=0.5 * (a + b), verts=rc["verts"])
    # Small oval mic windscreen tip clear to the right of mouth
    rm_tip = bmesh.ops.create_uvsphere(bm_hw, u_segments=12, v_segments=8, radius=0.0085)
    bmesh.ops.scale(bm_hw, vec=(1.2, 0.9, 0.9), verts=rm_tip["verts"])
    bmesh.ops.translate(bm_hw, vec=mic_pts[-1], verts=rm_tip["verts"])
    hardware = _new_bmesh_obj("Pilot_Helmet_Hardware", bm_hw, mats["hardware"], rig.matrix_world, col)

    # 5. Snug Chin Strap hugging under jawline (measured strictly against jaw vertices)
    bm_st = bmesh.new()
    chin_y = teeth_ctr_rig.y + 0.024
    jaw_z_floor = teeth_ctr_rig.z - 0.056
    jaw_only = [p for p in jaw_pts if p.z >= jaw_z_floor - 0.006 and p.y < cy + 0.025]
    under_ctr = [
        p for p in jaw_only
        if abs(p.x - cx) < 0.018 and abs(p.y - chin_y) < 0.022 and p.z < teeth_ctr_rig.z - 0.022
    ]
    chin_z = max(jaw_z_floor, (min(p.z for p in under_ctr) - 0.002) if under_ctr else (teeth_ctr_rig.z - 0.052))
    cs_pts: list[Vector] = []
    for k in range(23):
        u = (k / 22.0) * 2.0 - 1.0
        au = abs(u)
        px = cx + u * (rx * 0.92)
        pz = chin_z + (helm_cz - 0.015 - chin_z) * (au ** 2.25)
        py = chin_y * (1.0 - au * au) + (cy + 0.012) * (au * au)
        if 0.0 <= au <= 0.85 and jaw_only:
            if au < 0.45:
                near_u = [p for p in jaw_only if abs(p.x - px) < 0.014 and abs(p.y - py) < 0.018 and p.z < teeth_ctr_rig.z - 0.018]
                if near_u:
                    pz = max(jaw_z_floor, min(p.z for p in near_u) - 0.002)
            else:
                sgn = 1.0 if u > 0.0 else -1.0
                near_s = [p for p in jaw_only if (p.x - cx) * sgn > 0.015 and abs(p.y - py) < 0.018 and abs(p.z - pz) < 0.016]
                if near_s:
                    outer_x = max((p.x - cx) * sgn for p in near_s)
                    px = cx + sgn * max(abs(px - cx), outer_x + 0.0028)
        cs_pts.append(Vector((px, py, pz)))
    for _ in range(2):
        cs_pts = [cs_pts[0]] + [cs_pts[k - 1] * 0.22 + cs_pts[k] * 0.56 + cs_pts[k + 1] * 0.22 for k in range(1, len(cs_pts) - 1)] + [cs_pts[-1]]
    prev_pair = None
    for pt in cs_pts:
        v1 = bm_st.verts.new((pt.x, pt.y - 0.005, pt.z))
        v2 = bm_st.verts.new((pt.x, pt.y + 0.005, pt.z))
        if prev_pair is not None:
            bm_st.faces.new((prev_pair[0], v1, v2, prev_pair[1]))
        prev_pair = (v1, v2)
    straps = _new_bmesh_obj("Pilot_Helmet_Straps", bm_st, mats["webbing"], rig.matrix_world, col)
    ssol = straps.modifiers.new("Solidify", "SOLIDIFY")
    ssol.thickness = 0.0028

    parts = [shell, visor_cover, visor, hardware, straps]
    for p in parts:
        _parent_to_bone_preserve_world(p, rig, "head")
    return parts


def _sweep_ribbon_in_bm(
    bm: bmesh.types.BMesh,
    pts: list[Vector],
    half_w: float,
    thickness: float,
    side_axis: Vector,
) -> None:
    """Sweep a closed flat webbing strap along 3D `pts` with width along `side_axis` and outward normal."""
    rings = []
    n = len(pts)
    for i, p in enumerate(pts):
        p_prev = pts[max(0, i - 1)]
        p_next = pts[min(n - 1, i + 1)]
        tang = (p_next - p_prev)
        if tang.length < 1e-6:
            tang = Vector((0.0, 0.0, 1.0))
        tang.normalize()
        w_ax = (side_axis - tang * side_axis.dot(tang))
        if w_ax.length < 1e-6:
            w_ax = Vector((1.0, 0.0, 0.0))
        w_ax.normalize()
        n_ax = tang.cross(w_ax).normalized()
        c_in = p - n_ax * (thickness * 0.30)
        c_out = p + n_ax * (thickness * 0.70)
        v0 = bm.verts.new(c_in - w_ax * half_w)
        v1 = bm.verts.new(c_in + w_ax * half_w)
        v2 = bm.verts.new(c_out + w_ax * half_w)
        v3 = bm.verts.new(c_out - w_ax * half_w)
        rings.append((v0, v1, v2, v3))

    for k in range(len(rings) - 1):
        a0, a1, a2, a3 = rings[k]
        b0, b1, b2, b3 = rings[k + 1]
        bm.faces.new((a0, a1, b1, b0))
        bm.faces.new((a1, a2, b2, b1))
        bm.faces.new((a2, a3, b3, b2))
        bm.faces.new((a3, a0, b0, b3))
    bm.faces.new((rings[0][3], rings[0][2], rings[0][1], rings[0][0]))
    bm.faces.new((rings[-1][0], rings[-1][1], rings[-1][2], rings[-1][3]))


def _filter_segment_pts(
    pts: list[Vector],
    p0: Vector,
    p1: Vector,
    max_perp: float,
) -> list[Vector]:
    """Filter evaluated points strictly within cylindrical segment `p0 -> p1` (`0 <= proj <= len`, `perp < max_perp`)."""
    seg = p1 - p0
    seg_len = seg.length
    if seg_len < 1e-5:
        return []
    axis = seg / seg_len
    out: list[Vector] = []
    for p in pts:
        rel = p - p0
        proj = rel.dot(axis)
        if 0.0 <= proj <= seg_len:
            if (rel - axis * proj).length < max_perp:
                out.append(p)
    return out


def _add_sleeve_band_from_verts(
    bm: bmesh.types.BMesh,
    verts_bone: list[Vector],
    p0: Vector,
    p1: Vector,
    u_fracs: tuple[float, float],
    pad_m: float,
    thick_m: float,
    n_seg: int = 20,
    n_rings: int = 3,
    mat_index: int = 0,
    min_r: float = 0.032,
    max_r: float = 0.088,
) -> None:
    """Wrap a fitted G-suit bladder/cuff cleanly around evaluated limb mesh vertices along `p0 -> p1`."""
    axis = (p1 - p0).normalized()
    length = (p1 - p0).length
    ref = Vector((1.0, 0.0, 0.0)) if abs(axis.x) < 0.85 else Vector((0.0, 1.0, 0.0))
    ax_u = (ref - axis * ref.dot(axis)).normalized()
    ax_v = axis.cross(ax_u).normalized()

    def ring_radii(proj_lo: float, proj_hi: float) -> list[float]:
        band_pts = [p for p in verts_bone if proj_lo <= (p - p0).dot(axis) <= proj_hi]
        fallback_r = min(max_r, max(min_r, 0.042 + pad_m))
        radii = [fallback_r] * n_seg
        if not band_pts:
            return radii
        perps = []
        for pt in band_pts:
            rel = pt - p0
            perp = rel - axis * rel.dot(axis)
            plen = perp.length
            if plen > 1e-4:
                perps.append((perp / plen, plen))
        for i in range(n_seg):
            ang = (i / n_seg) * math.tau
            dir_r = ax_u * math.cos(ang) + ax_v * math.sin(ang)
            best_tight = 0.0
            best_wide = 0.0
            for p_unit, plen in perps:
                dp = p_unit.dot(dir_r)
                if dp > 0.88:
                    best_tight = max(best_tight, plen)
                elif dp > 0.72:
                    best_wide = max(best_wide, plen)
            raw_r = best_tight if best_tight > 0.0 else (best_wide if best_wide > 0.0 else min_r)
            radii[i] = min(max_r, max(min_r, raw_r + pad_m))
        for _ in range(2):
            radii = [
                0.25 * radii[(i - 1) % n_seg] + 0.50 * radii[i] + 0.25 * radii[(i + 1) % n_seg]
                for i in range(n_seg)
            ]
        return radii

    d0 = u_fracs[0] * length
    d1 = u_fracs[1] * length
    n_rings = max(2, int(n_rings))
    d_vals = [d0 + (d1 - d0) * (k / float(n_rings - 1)) for k in range(n_rings)]
    rad_rings = [ring_radii(dk - 0.022, dk + 0.022) for dk in d_vals]

    def make_loop(dist_d: float, rads: list[float], off_r: float) -> list:
        ctr = p0 + axis * dist_d
        return [
            bm.verts.new(
                ctr + (ax_u * math.cos((i / n_seg) * math.tau) + ax_v * math.sin((i / n_seg) * math.tau)) * (rads[i] + off_r)
            )
            for i in range(n_seg)
        ]

    in_rings = [make_loop(dk, rk, 0.0) for dk, rk in zip(d_vals, rad_rings)]
    out_rings = [make_loop(dk, rk, thick_m) for dk, rk in zip(d_vals, rad_rings)]

    def add_face(verts) -> None:
        f = bm.faces.new(verts)
        f.material_index = mat_index

    for k in range(n_rings - 1):
        for i in range(n_seg):
            ni = (i + 1) % n_seg
            add_face((out_rings[k][i], out_rings[k][ni], out_rings[k + 1][ni], out_rings[k + 1][i]))
            add_face((in_rings[k + 1][i], in_rings[k + 1][ni], in_rings[k][ni], in_rings[k][i]))
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        add_face((in_rings[0][i], in_rings[0][ni], out_rings[0][ni], out_rings[0][i]))
        add_face((out_rings[-1][i], out_rings[-1][ni], in_rings[-1][ni], in_rings[-1][i]))


def _build_collar_and_bib(
    carrier: bpy.types.Object,
    rig: bpy.types.Object,
    top_pts: list[Vector],
    body_pts: list[Vector],
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
) -> bpy.types.Object:
    """Build seamless CWU-27/P Nomex mandarin collar (anchored to skull `hd_c.z`), narrow flush
    chest bib tucked 1.5 cm inside collar (`z_bib_hi = z_c0 + 0.015`) & central zipper placket."""
    inv_car = carrier.matrix_world.inverted()
    pb_s1 = rig.pose.bones.get("spine01")
    pb_n1 = rig.pose.bones.get("neck01") or pb_s1
    pb_hd = rig.pose.bones.get("head") or pb_n1

    s1_c = inv_car @ (rig.matrix_world @ pb_s1.head) if pb_s1 is not None else Vector((0.0, -0.12, 0.48))
    n1_c = inv_car @ (rig.matrix_world @ pb_n1.head) if pb_n1 is not None else Vector((0.0, -0.15, 0.59))
    hd_c = inv_car @ (rig.matrix_world @ pb_hd.head) if pb_hd is not None else Vector((0.0, -0.16, 0.77))

    ref_pts = body_pts if body_pts else top_pts

    z_c0 = hd_c.z - 0.205
    z_c1 = hd_c.z - 0.130
    z_bib_lo = max(0.38, min(0.42, s1_c.z - 0.08))
    z_bib_hi = z_c0 + 0.015  # Tucks cleanly 1.5 cm UP INSIDE collar so zero shelf or throat gap!

    bm = bmesh.new()

    # 1. Narrow Flush Chest V-Bib (`z_bib_lo .. z_bib_hi`) with softly tapered side/bottom edges
    nu, nv = 16, 10
    y_back_guard = min(-0.14, n1_c.y) - 0.03
    half_w = 0.050
    x_grid: list[list[float]] = []
    y_grid: list[list[float]] = []
    z_grid: list[list[float]] = []

    for j in range(nv + 1):
        vf = j / float(nv)
        zj = z_bib_lo + vf * (z_bib_hi - z_bib_lo)
        row_b = [p for p in ref_pts if abs(p.z - zj) < 0.024 and p.y > y_back_guard]
        row_t = [p for p in top_pts if abs(p.z - zj) < 0.024 and p.y > y_back_guard]
        xs, ys, zs = [], [], []
        for i in range(nu + 1):
            uf = (i / float(nu)) * 2.0 - 1.0
            xi = uf * half_w
            near_b = [p.y for p in row_b if abs(p.x - xi) < 0.022]
            if not near_b:
                near_b = [p.y for p in row_b if abs(p.x - xi) < 0.045]
            y_b = max(near_b) if near_b else (0.080 - 0.140 * vf - 0.25 * (xi * xi))
            near_t = [p.y for p in row_t if abs(p.x - xi) < 0.022]
            y_t_center = max(near_t) if near_t else y_b
            y_surf = max(y_b, min(y_t_center, y_b + 0.022))
            xs.append(xi)
            ys.append(y_surf)
            zs.append(zj)
        x_grid.append(xs)
        y_grid.append(ys)
        z_grid.append(zs)

    # 2 passes of gentle 2D Laplacian smoothing (clamped >= y_raw so bib never dips below shirt/skin)
    y_raw = [list(r) for r in y_grid]
    for _ in range(2):
        y_new = [list(r) for r in y_grid]
        for j in range(nv + 1):
            for i in range(nu + 1):
                acc = y_grid[j][i] * 2.0
                wsum = 2.0
                if i > 0:
                    acc += y_grid[j][i - 1]
                    wsum += 1.0
                if i < nu:
                    acc += y_grid[j][i + 1]
                    wsum += 1.0
                if j > 0:
                    acc += y_grid[j - 1][i]
                    wsum += 1.0
                if j < nv:
                    acc += y_grid[j + 1][i]
                    wsum += 1.0
                y_new[j][i] = max(y_raw[j][i], acc / wsum)
        y_grid = y_new

    bib_in = []
    bib_out = []
    for j in range(nv + 1):
        vf = j / float(nv)
        v_bot = max(0.0, (0.16 - vf) / 0.16)
        row_in, row_out = [], []
        for i in range(nu + 1):
            u_abs = abs((i / float(nu)) * 2.0 - 1.0)
            side_edge = max(0.0, (u_abs - 0.62) / 0.38)
            edge_t = max(side_edge, v_bot)
            edge_s = edge_t * edge_t * (3.0 - 2.0 * edge_t)
            # Center cleanly proud (+0.0045), side & bottom rims gently tucked inward (flank_in = -0.002)
            y_off_out = 0.0045 * (1.0 - edge_s) + (-0.0020) * edge_s
            row_in.append(bm.verts.new((x_grid[j][i], y_grid[j][i] - 0.0032, z_grid[j][i])))
            row_out.append(bm.verts.new((x_grid[j][i], y_grid[j][i] + y_off_out, z_grid[j][i])))
        bib_in.append(row_in)
        bib_out.append(row_out)
    for j in range(nv):
        for i in range(nu):
            bm.faces.new((bib_out[j][i], bib_out[j][i + 1], bib_out[j + 1][i + 1], bib_out[j + 1][i])).material_index = 0
            bm.faces.new((bib_in[j + 1][i], bib_in[j + 1][i + 1], bib_in[j][i + 1], bib_in[j][i])).material_index = 0
    for i in range(nu):
        bm.faces.new((bib_in[0][i], bib_in[0][i + 1], bib_out[0][i + 1], bib_out[0][i])).material_index = 0
        bm.faces.new((bib_out[nv][i], bib_out[nv][i + 1], bib_in[nv][i + 1], bib_in[nv][i])).material_index = 0
    for j in range(nv):
        bm.faces.new((bib_out[j][0], bib_out[j + 1][0], bib_in[j + 1][0], bib_in[j][0])).material_index = 0
        bm.faces.new((bib_in[j][nu], bib_in[j + 1][nu], bib_out[j + 1][nu], bib_out[j][nu])).material_index = 0

    # 2. Central CWU-27/P Zipper Placket & Gunmetal Zipper Slider (`material_index=0` placket, `1` slider)
    i_mid = nu // 2
    placket_pts = [Vector((0.0, y_grid[j][i_mid] + 0.0052, z_grid[j][i_mid])) for j in range(1, nv + 1)]
    _sweep_ribbon_in_bm(bm, placket_pts, half_w=0.011, thickness=0.0036, side_axis=Vector((1.0, 0.0, 0.0)))
    r_zip = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.009, 0.007, 0.016), verts=r_zip["verts"])
    bmesh.ops.translate(bm, vec=Vector((0.0, y_grid[nv - 1][i_mid] + 0.009, z_c0 - 0.002)), verts=r_zip["verts"])
    zip_vset = set(r_zip["verts"])
    for f in bm.faces:
        if all(v in zip_vset for v in f.verts):
            f.material_index = 1

    # 3. 360-deg CWU-27/P Mandarin Flight-Suit Collar (`z_c0 = hd_c.z - 0.205 .. z_c1 = hd_c.z - 0.130`)
    nu_c, nv_c = 28, 5
    col_in_rings = []
    col_out_rings = []
    for j in range(nv_c + 1):
        zj = z_c0 + (j / float(nv_c)) * (z_c1 - z_c0)
        ns = [p for p in ref_pts if abs(p.z - zj) < 0.020 and abs(p.x) < 0.095]
        ts = [p for p in top_pts if abs(p.z - zj) < 0.020 and abs(p.x) < 0.105]
        y_ctr = (0.5 * (min(p.y for p in ns) + max(p.y for p in ns))) if ns else (n1_c.y + 0.012)
        radii = []
        for i in range(nu_c):
            th = (i / float(nu_c)) * math.tau
            dx, dy = math.cos(th), math.sin(th)
            r_s = 0.056
            for pt in ns:
                rx, ry = pt.x, pt.y - y_ctr
                rl = math.hypot(rx, ry)
                if rl > 1e-4 and (rx * dx + ry * dy) / rl > 0.85:
                    r_s = max(r_s, rl)
            r_t = r_s
            for pt in ts:
                rx, ry = pt.x, pt.y - y_ctr
                rl = math.hypot(rx, ry)
                if rl > 1e-4 and (rx * dx + ry * dy) / rl > 0.85:
                    r_t = max(r_t, min(rl, r_s + 0.012))
            radii.append(max(r_s, r_t) + 0.0052)
        for _ in range(2):
            radii = [
                0.25 * radii[(i - 1) % nu_c] + 0.50 * radii[i] + 0.25 * radii[(i + 1) % nu_c]
                for i in range(nu_c)
            ]
        rin_row, rout_row = [], []
        for i in range(nu_c):
            th = (i / float(nu_c)) * math.tau
            dx, dy = math.cos(th), math.sin(th)
            ri = radii[i]
            ro = ri + 0.0046
            rin_row.append(bm.verts.new((dx * ri, y_ctr + dy * ri, zj)))
            rout_row.append(bm.verts.new((dx * ro, y_ctr + dy * ro, zj)))
        col_in_rings.append(rin_row)
        col_out_rings.append(rout_row)

    for j in range(nv_c):
        for i in range(nu_c):
            ni = (i + 1) % nu_c
            bm.faces.new((col_out_rings[j][i], col_out_rings[j][ni], col_out_rings[j + 1][ni], col_out_rings[j + 1][i])).material_index = 0
            bm.faces.new((col_in_rings[j + 1][i], col_in_rings[j + 1][ni], col_in_rings[j][ni], col_in_rings[j][i])).material_index = 0
    for i in range(nu_c):
        ni = (i + 1) % nu_c
        bm.faces.new((col_in_rings[0][i], col_in_rings[0][ni], col_out_rings[0][ni], col_out_rings[0][i])).material_index = 0
        bm.faces.new((col_out_rings[-1][i], col_out_rings[-1][ni], col_in_rings[-1][ni], col_in_rings[-1][i])).material_index = 0

    collar_obj = _new_bmesh_obj(
        "Pilot_Harness_CollarBib",
        bm,
        [mats["undershirt"], mats["hardware"]],
        carrier.matrix_world,
        col,
        recalc_normals=True,
    )
    _parent_to_bone_preserve_world(collar_obj, rig, "spine01")
    return collar_obj


def _build_pilot_harness_and_gsuit(
    carrier: bpy.types.Object,
    rig: bpy.types.Object,
    suit_top: bpy.types.Object | None,
    suit_pants: bpy.types.Object | None,
    mats: dict[str, bpy.types.Material],
    col: bpy.types.Collection,
    body: bpy.types.Object | None = None,
) -> list[bpy.types.Object]:
    """Build seated flight harness webbing, CWU-27/P collar/bib, Koch riser D-rings & sleek G-suit wraps."""
    bpy.context.view_layer.update()
    inv_car = carrier.matrix_world.inverted()
    deps = bpy.context.evaluated_depsgraph_get()

    def eval_verts_carrier(obj: bpy.types.Object | None) -> list[Vector]:
        if obj is None:
            return []
        eo = obj.evaluated_get(deps)
        em = eo.to_mesh()
        xf = inv_car @ eo.matrix_world
        pts = [xf @ v.co for v in em.vertices]
        eo.to_mesh_clear()
        return pts

    top_pts = eval_verts_carrier(suit_top)
    pants_pts = eval_verts_carrier(suit_pants)
    body_pts = eval_verts_carrier(body)

    # 1. Torso Harness + Shoulder Koch Riser D-Rings at exact C.RISER_ATTACH_L/R (parented to `spine01`)
    bm_ch = bmesh.new()
    for side, riser_c in (("L", C.RISER_ATTACH_L), ("R", C.RISER_ATTACH_R)):
        sx = -1.0 if side == "L" else 1.0
        rx, ry, rz = riser_c
        # Measure front/back chest surface near riser X so webbing hugs Nomex flight suit
        band = [p for p in top_pts if abs(p.x - sx * 0.125) < 0.045 and 0.25 <= p.z <= 0.56]
        y_front = (max(p.y for p in band) + 0.005) if band else 0.06
        y_back = (min(p.y for p in band) - 0.005) if band else -0.17

        # Shoulder webbing path draping from upper back -> Koch D-ring -> front chest -> lower ribcage
        strap_pts = [
            Vector((sx * 0.135, y_back - 0.006, 0.34)),
            Vector((sx * 0.155, y_back + 0.015, 0.48)),
            Vector((rx, ry, rz - 0.012)),
            Vector((sx * 0.140, y_front * 0.65, 0.53)),
            Vector((sx * 0.125, y_front + 0.008, 0.43)),
            Vector((sx * 0.115, y_front + 0.004, 0.28)),
        ]
        _sweep_ribbon_in_bm(bm_ch, strap_pts, half_w=0.023, thickness=0.008, side_axis=Vector((1.0, 0.0, 0.0)))

        # Metallic Koch Riser Hardware + D-Ring centered cleanly at C.RISER_ATTACH_L/R
        rk = bmesh.ops.create_cube(bm_ch, size=1.0)
        bmesh.ops.scale(bm_ch, vec=(0.048, 0.032, 0.018), verts=rk["verts"])
        bmesh.ops.translate(bm_ch, vec=(rx, ry, rz - 0.014), verts=rk["verts"])

        # D-ring loop rising to z = rz + 0.008 (keeps riser attachment clear even after -2.5cm compression!)
        d_pts = [
            Vector((rx - 0.018, ry, rz - 0.008)),
            Vector((rx - 0.015, ry, rz + 0.010)),
            Vector((rx + 0.015, ry, rz + 0.010)),
            Vector((rx + 0.018, ry, rz - 0.008)),
        ]
        _sweep_ribbon_in_bm(bm_ch, d_pts, half_w=0.005, thickness=0.006, side_axis=Vector((0.0, 1.0, 0.0)))

    # Horizontal sternum chest strap connecting left & right harness straps
    chest_band = [p for p in top_pts if abs(p.x) < 0.13 and 0.36 <= p.z <= 0.44]
    y_st = (max(p.y for p in chest_band) + 0.014) if chest_band else 0.085
    sternum_pts = [
        Vector((-0.125, y_st - 0.010, 0.40)),
        Vector((-0.055, y_st + 0.003, 0.40)),
        Vector((0.055, y_st + 0.003, 0.40)),
        Vector((0.125, y_st - 0.010, 0.40)),
    ]
    _sweep_ribbon_in_bm(bm_ch, sternum_pts, half_w=0.018, thickness=0.008, side_axis=Vector((0.0, 0.0, 1.0)))
    # Center chest quick-release buckle
    rb_c = bmesh.ops.create_cube(bm_ch, size=1.0)
    bmesh.ops.scale(bm_ch, vec=(0.046, 0.016, 0.042), verts=rb_c["verts"])
    bmesh.ops.translate(bm_ch, vec=(0.0, y_st + 0.006, 0.40), verts=rb_c["verts"])

    bmesh.ops.bevel(bm_ch, geom=list(bm_ch.edges), offset=0.002, segments=1, affect="EDGES")
    chest_obj = _new_bmesh_obj("Pilot_Harness_Chest", bm_ch, mats["webbing"], carrier.matrix_world, col)
    _parent_to_bone_preserve_world(chest_obj, rig, "spine01")

    # 1b. CWU-27/P Nomex Collar & Dark Undershirt Bib / Zipper Placket (closes V-neck chest gap)
    collar_obj = _build_collar_and_bib(carrier, rig, top_pts, body_pts, mats, col)

    # 2. Pelvic Waist Belt & Lap Harness Hardware (restricted strictly to true pelvis/hip band, never thighs!)
    bm_pel = bmesh.new()
    waist_band = [
        p for p in (*top_pts, *pants_pts)
        if abs(p.x) < 0.20 and -0.22 <= p.y <= 0.13 and 0.14 <= p.z <= 0.25
    ]
    w_rx = min(0.172, max(0.140, (max(abs(p.x) for p in waist_band) + 0.008) if waist_band else 0.160))
    w_yf = min(0.132, max(0.085, (max(p.y for p in waist_band) + 0.008) if waist_band else 0.120))
    w_yb = max(-0.145, min(-0.085, (min(p.y for p in waist_band) - 0.006) if waist_band else -0.130))
    belt_pts = [
        Vector((math.cos((k / 24.0) * math.tau) * w_rx, 0.5 * (w_yf + w_yb) + math.sin((k / 24.0) * math.tau) * 0.5 * (w_yf - w_yb), 0.205))
        for k in range(25)
    ]
    _sweep_ribbon_in_bm(bm_pel, belt_pts, half_w=0.022, thickness=0.009, side_axis=Vector((0.0, 0.0, 1.0)))
    rb_w = bmesh.ops.create_cube(bm_pel, size=1.0)
    bmesh.ops.scale(bm_pel, vec=(0.056, 0.018, 0.048), verts=rb_w["verts"])
    bmesh.ops.translate(bm_pel, vec=(0.0, w_yf + 0.005, 0.205), verts=rb_w["verts"])
    pelvis_obj = _new_bmesh_obj("Pilot_Harness_Pelvis", bm_pel, mats["webbing"], carrier.matrix_world, col)
    _parent_to_bone_preserve_world(pelvis_obj, rig, "spine05")

    # 3. Sleek G-Suit Thigh & Calf Wraps + Dark Ankle Leg-Restraint Garter Straps (anatomical pre-shrink `body_pts`!)
    src_leg_pts = body_pts if body_pts else pants_pts
    parts = [chest_obj, collar_obj, pelvis_obj]
    for side in ("L", "R"):
        pb_hip = rig.pose.bones[f"upperleg01.{side}"]
        pb_knee = rig.pose.bones[f"lowerleg01.{side}"]
        pb_foot = rig.pose.bones[f"foot.{side}"]
        hip_c = inv_car @ (rig.matrix_world @ pb_hip.head)
        knee_c = inv_car @ (rig.matrix_world @ pb_knee.head)
        ankle_c = inv_car @ (rig.matrix_world @ pb_foot.head)

        # Thigh G-suit wrap (0.34..0.72 along thigh, radially filtered & clamped <= 0.088m)
        th_pts = _filter_segment_pts(src_leg_pts, hip_c, knee_c, max_perp=0.11)
        bm_th = bmesh.new()
        _add_sleeve_band_from_verts(
            bm_th, th_pts, hip_c, knee_c, (0.34, 0.72),
            pad_m=0.007, thick_m=0.003, mat_index=0, min_r=0.046, max_r=0.088,
        )
        th_obj = _new_bmesh_obj(f"Pilot_Harness_GSuit_Thigh_{side}", bm_th, mats["gsuit"], carrier.matrix_world, col)
        _parent_to_bone_preserve_world(th_obj, rig, f"upperleg01.{side}")
        parts.append(th_obj)

        # Shin G-suit calf wrap (0.24..0.64, <= 0.062m) + sleek dark webbing ankle garter strap (0.84..0.93, <= 0.048m)
        sh_pts = _filter_segment_pts(src_leg_pts, knee_c, ankle_c, max_perp=0.075)
        bm_sh = bmesh.new()
        _add_sleeve_band_from_verts(
            bm_sh, sh_pts, knee_c, ankle_c, (0.24, 0.64),
            pad_m=0.006, thick_m=0.003, mat_index=0, min_r=0.038, max_r=0.062,
        )
        _add_sleeve_band_from_verts(
            bm_sh, sh_pts, knee_c, ankle_c, (0.84, 0.93),
            pad_m=0.006, thick_m=0.0045, n_rings=2, mat_index=1, min_r=0.032, max_r=0.048,
        )
        sh_obj = _new_bmesh_obj(
            f"Pilot_Harness_GSuit_Shin_{side}",
            bm_sh,
            [mats["gsuit"], mats["garter_dark"]],
            carrier.matrix_world,
            col,
        )
        _parent_to_bone_preserve_world(sh_obj, rig, f"lowerleg01.{side}")
        parts.append(sh_obj)

    return parts


# ============================================================================
# EXACT ARMATURE SOLVERS: ROTATION, SPINE COMPRESSION & 2-BONE ANALYTICAL IK
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
    """Build right-handed 3x3 orthonormal matrix with Y = `y_dir` and X = `hinge_x`."""
    y_ax = y_dir.normalized()
    x_ax = (hinge_x - y_ax * hinge_x.dot(y_ax))
    if x_ax.length < 1e-6:
        ref = Vector((1.0, 0.0, 0.0)) if abs(y_ax.x) < 0.85 else Vector((0.0, 1.0, 0.0))
        x_ax = y_ax.cross(ref)
    x_ax.normalize()
    z_ax = x_ax.cross(y_ax).normalized()
    return Matrix((x_ax, y_ax, z_ax)).transposed()


def _basis_from_dir_pole(y_dir: Vector, pole_z: Vector) -> Matrix:
    """Build right-handed 3x3 orthonormal matrix with Y = `y_dir` and Z aligned toward `pole_z`."""
    y_ax = y_dir.normalized()
    z_ax = (pole_z - y_ax * pole_z.dot(y_ax))
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


def _apply_spine_and_head_pose(
    rig: bpy.types.Object,
    rig_rel_carrier: Matrix,
    lean_aft_deg: float,
    torso_twist_deg: float,
    torso_side_deg: float,
    head_pitch_deg: float,
    head_yaw_deg: float,
    head_roll_deg: float,
    comp_cm: float,
    frame: int | None,
) -> None:
    """Pose spine (`spine05..01`), neck & head with exact `-comp_cm` spinal disc compression."""
    # In Rig-local char space: -rx leans aft (+Y_r / -Y_c), +rx leans forward (-Y_r / +Y_c)
    rx_total = -lean_aft_deg
    mc.set_bone_char_rot(rig, "spine05", rx_total * 0.24, torso_side_deg * 0.35, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine04", rx_total * 0.22, torso_side_deg * 0.25, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine03", rx_total * 0.20, torso_side_deg * 0.20, torso_twist_deg * 0.25, frame=frame)
    mc.set_bone_char_rot(rig, "spine02", rx_total * 0.18, torso_side_deg * 0.10, torso_twist_deg * 0.15, frame=frame)
    mc.set_bone_char_rot(rig, "spine01", rx_total * 0.16, torso_side_deg * 0.10, torso_twist_deg * 0.10, frame=frame)

    # Counter-pitch neck forward gently so head remains ergonomic inside the tilted seat
    neck_rx = lean_aft_deg * 0.38 + head_pitch_deg * 0.32
    head_rx = lean_aft_deg * 0.32 + head_pitch_deg * 0.68
    mc.set_bone_char_rot(rig, "neck01", neck_rx * 0.60, 0.0, head_yaw_deg * 0.32, frame=frame)
    if "neck02" in rig.pose.bones:
        mc.set_bone_char_rot(rig, "neck02", neck_rx * 0.40, 0.0, head_yaw_deg * 0.20, frame=frame)
    mc.set_bone_char_rot(rig, "head", head_rx, head_roll_deg, head_yaw_deg * 0.48, frame=frame)

    # Apply exact spinal compression along (sacrum - c1) axis matching Spine_XRay contraction
    bpy.context.view_layer.update()
    comp_m = max(0.0, float(comp_cm)) * 0.01
    down_c = (Vector(C.PILOT["sacrum"]) - Vector(C.PILOT["c1_skull_base"])).normalized()
    down_rig = _carrier_dir_to_rig(rig_rel_carrier, down_c)

    active_segs = [b for b in _SPINE_SEG_BONES if b in rig.pose.bones and not rig.data.bones[b].use_connect]
    if active_segs:
        per_seg = down_rig * (comp_m / float(len(active_segs)))
        for bname in active_segs:
            pb = rig.pose.bones[bname]
            if pb.parent is not None:
                r_parent_rig = pb.parent.matrix.to_3x3() @ pb.parent.bone.matrix_local.to_3x3().inverted()
            else:
                r_parent_rig = Matrix.Identity(3)
            pb.location = (r_parent_rig @ pb.bone.matrix_local.to_3x3()).inverted() @ per_seg
            if frame is not None:
                pb.keyframe_insert("location", frame=frame)
            bpy.context.view_layer.update()


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
    """Closed-loop 2-bone analytical arm + wrist IK solver calibrated to palm grip center (< 1 mm error)."""
    side_u = "L" if side.upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0  # +1 = L (+X_r), -1 = R (-X_r)

    # Set clavicle elevation & forward protraction first (-side_sgn * protract pulls shoulder toward -Y_r forward)
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
    rest_pole = Vector((0.0, 1.0, 0.0))  # elbow bends backward (+Y_r)
    rest_hinge1 = rest_v1.cross(rest_pole).normalized()
    f_rest1 = _basis_from_dir_hinge(rest_v1, rest_hinge1)
    f_rest2 = _basis_from_dir_hinge(rest_v2, rest_hinge1)

    # Rest hand frame on wrist.{side_u} derived from knuckle plane
    hand_fwd_rest = ((b_kn.head_local - b_wr.head_local) if b_kn else (b_wr.tail_local - b_wr.head_local)).normalized()
    palm_guess = Vector((-side_sgn, 0.0, 0.0))
    if b_f2 is not None and b_f5 is not None:
        v_idx = b_f2.head_local - b_wr.head_local
        v_pnk = b_f5.head_local - b_wr.head_local
        cand_palm = (v_pnk.cross(v_idx) if side_u == "L" else v_idx.cross(v_pnk))
        if cand_palm.length > 1e-6:
            cand_palm.normalize()
            if cand_palm.dot(palm_guess) > 0.0:
                palm_guess = cand_palm
    f_rest_hand = _basis_from_dir_pole(hand_fwd_rest, palm_guess)
    grip_dist = ((b_kn.head_local - b_wr.head_local).length + 0.014) if b_kn else 0.070

    tgt_fwd_r = _carrier_dir_to_rig(rig_rel_carrier, hand_fwd_c)
    tgt_palm_r = _carrier_dir_to_rig(rig_rel_carrier, hand_palm_c)
    f_tgt_hand = _basis_from_dir_pole(tgt_fwd_r, tgt_palm_r)
    r_wrist_tgt = f_tgt_hand @ f_rest_hand.inverted()

    grip_tgt_r = _carrier_pt_to_rig(rig_rel_carrier, grip_target_c)
    pole_r = _carrier_dir_to_rig(rig_rel_carrier, pole_c)
    wrist_tgt_r = grip_tgt_r - tgt_fwd_r * grip_dist - tgt_palm_r * 0.012

    pb_ua = rig.pose.bones[f"upperarm01.{side_u}"]
    pb_kn = rig.pose.bones.get(f"finger3-1.{side_u}")
    pb_tip = rig.pose.bones.get(f"finger3-2.{side_u}")

    L1 = (b_la.head_local - b_ua.head_local).length
    L2 = (b_wr.head_local - b_la.head_local).length

    # Apply fingers first so closed-loop grip center measurement uses real curled finger cup!
    _apply_finger_curl(rig, side_u, grip_curl, frame=frame)

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

        r_ua = _basis_from_dir_hinge(dir1, hinge_x) @ f_rest1.inverted()
        _pose_bone_abs_rig(rig, f"upperarm01.{side_u}", r_ua, frame=frame)
        bpy.context.view_layer.update()

        r_la = _basis_from_dir_hinge(dir2, hinge_x) @ f_rest2.inverted()
        _pose_bone_abs_rig(rig, f"lowerarm01.{side_u}", r_la, frame=frame)
        bpy.context.view_layer.update()

        _pose_bone_abs_rig(rig, f"wrist.{side_u}", r_wrist_tgt, frame=frame)
        bpy.context.view_layer.update()

        if pb_kn is not None:
            actual_grip = pb_kn.head + (pb_tip.head - pb_kn.head) * 0.45 if pb_tip is not None else pb_kn.head
            actual_grip = actual_grip + tgt_palm_r * 0.012
            err = grip_tgt_r - actual_grip
            if err.length < 0.0012:
                break
            wrist_tgt_r = wrist_tgt_r + err * 0.90


def _solve_leg_ik(
    rig: bpy.types.Object,
    rig_rel_carrier: Matrix,
    side: str,
    ankle_target_c: Vector | tuple[float, float, float],
    foot_fwd_c: Vector | tuple[float, float, float],
    min_knee_splay_x: float = 0.155,
    frame: int | None = None,
) -> None:
    """Analytical 2-bone leg IK enforcing outward knee splay (`|x_knee| >= min_knee_splay_x`) so thighs
    never clip the yellow-black ejection handle."""
    side_u = "L" if side.upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0  # +1 = L (+X_r), -1 = R (-X_r)

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
    rest_pole = Vector((0.0, -1.0, 0.0))  # knee points forward (-Y_r)
    rest_hinge = rest_v1.cross(rest_pole).normalized()
    f_rest1 = _basis_from_dir_hinge(rest_v1, rest_hinge)
    f_rest2 = _basis_from_dir_hinge(rest_v2, rest_hinge)

    foot_fwd_rest = (b_ft.tail_local - b_ft.head_local).normalized()
    f_rest_foot = _basis_from_dir_pole(foot_fwd_rest, Vector((0.0, 0.0, 1.0)))

    pb_ul = rig.pose.bones[f"upperleg01.{side_u}"]
    S = pb_ul.head.copy()
    W_tgt = _carrier_pt_to_rig(rig_rel_carrier, ankle_target_c)
    L1 = (b_ll.head_local - b_ul.head_local).length
    L2 = (b_ft.head_local - b_ll.head_local).length

    D = W_tgt - S
    d = min(max(D.length, abs(L1 - L2) + 0.015), L1 + L2 - 0.004)
    u_dir = D.normalized()

    # Start with forward-upward knee pole with gentle outward splay; bump splay if needed
    splay_w = 0.24
    E = S
    p_perp = Vector((0.0, -1.0, 0.0))
    for _ in range(6):
        pole_r = Vector((side_sgn * splay_w, -0.90, 0.36)).normalized()
        p_perp = (pole_r - u_dir * pole_r.dot(u_dir)).normalized()
        cos_a = max(-0.998, min(0.998, (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)))
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        E = S + u_dir * (L1 * cos_a) + p_perp * (L1 * sin_a)
        if abs(E.x) >= min_knee_splay_x:
            break
        splay_w += 0.14

    W = S + u_dir * d
    dir1 = (E - S).normalized()
    dir2 = (W - E).normalized()
    hinge_x = u_dir.cross(p_perp).normalized()

    _pose_bone_abs_rig(rig, f"upperleg01.{side_u}", _basis_from_dir_hinge(dir1, hinge_x) @ f_rest1.inverted(), frame=frame)
    bpy.context.view_layer.update()
    _pose_bone_abs_rig(rig, f"lowerleg01.{side_u}", _basis_from_dir_hinge(dir2, hinge_x) @ f_rest2.inverted(), frame=frame)
    bpy.context.view_layer.update()

    # Orient foot along `foot_fwd_c` with natural outward toe splay
    fwd_r = _carrier_dir_to_rig(rig_rel_carrier, foot_fwd_c)
    up_r = Vector((0.0, 0.0, 1.0))
    _pose_bone_abs_rig(rig, f"foot.{side_u}", _basis_from_dir_pole(fwd_r, up_r) @ f_rest_foot.inverted(), frame=frame)
    bpy.context.view_layer.update()


def _apply_finger_curl(rig: bpy.types.Object, side_u: str, curl: float, frame: int | None = None) -> None:
    """Blend smoothly between open ready fingers (`curl=0`) and tight ejection-handle/stick grip (`curl=1`)."""
    g = max(0.0, min(1.0, float(curl)))
    if g < 0.18:
        mc.apply_hand_gesture(rig, side_u, "relaxed", frame=frame)
        return

    has_f5 = f"finger5-1.{side_u}" in rig.pose.bones
    digits = (2, 3, 4, 5) if has_f5 else (1, 2, 3, 4)
    curls_open = (12.0, 15.0, 19.0, 23.0)
    curls_closed = (54.0, 60.0, 64.0, 68.0)

    for d_i, f_idx in enumerate(digits):
        c_deg = curls_open[d_i] + (curls_closed[d_i] - curls_open[d_i]) * g
        for seg, mult in ((1, 0.80), (2, 1.05), (3, 0.72)):
            pb = rig.pose.bones.get(f"finger{f_idx}-{seg}.{side_u}")
            if pb is not None:
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(c_deg * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)

    thm_deg = 12.0 + 34.0 * g
    if has_f5:
        for seg, mult in ((1, 0.42), (2, 0.90), (3, 0.82)):
            pb = rig.pose.bones.get(f"finger1-{seg}.{side_u}")
            if pb is not None:
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(thm_deg * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)


# ============================================================================
# SEATED ALIGNMENT & FULL 6-BEAT CHOREOGRAPHY (FRAMES 1..1800)
# ============================================================================
def _calibrate_seated_placement(
    carrier: bpy.types.Object,
    rig: bpy.types.Object,
    suit_pants: bpy.types.Object | None,
) -> tuple[Matrix, tuple[float, float, float], tuple[float, float, float]]:
    """Seat `Pilot_Rig` at 15 deg aft inside `Pilot_Carrier` so sacrum/c1/eye/shoulders match C.PILOT."""
    rig.parent = carrier
    rig.matrix_parent_inverse = Matrix.Identity(4)
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, math.pi), "XYZ")
    rig.location = Vector((0.0, 0.0, -0.82))
    bpy.context.view_layer.update()

    # Apply 15-deg aft seated spine pose
    tmp_rel = Matrix.Translation(rig.location) @ rig.rotation_euler.to_matrix().to_4x4()
    _apply_spine_and_head_pose(rig, tmp_rel, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, frame=None)
    bpy.context.view_layer.update()

    # Measure `spine05.head` (sacrum) in carrier-local and snap to C.PILOT["sacrum"]
    pb_sac = rig.pose.bones["spine05"]
    sac_c = carrier.matrix_world.inverted() @ (rig.matrix_world @ pb_sac.head)
    delta_sac = Vector(C.PILOT["sacrum"]) - sac_c
    rig.location = rig.location + delta_sac
    bpy.context.view_layer.update()

    # Pose seated legs onto rudder pedals (`f=1`) FIRST so calves/ankles swing forward out of pelvic glute band
    rig_rel_carrier = Matrix.Translation(rig.location) @ rig.rotation_euler.to_matrix().to_4x4()
    for side in ("L", "R"):
        _solve_leg_ik(rig, rig_rel_carrier, side, C.ankle_local(side, 1), (0.0, 0.90, 0.44), min_knee_splay_x=0.175, frame=None)
    bpy.context.view_layer.update()

    # Check trouser glute cushion contact strictly within true pelvic glute band (`|x|<0.16, -0.16<y<0.12, z>-0.12`)
    if suit_pants is not None:
        deps = bpy.context.evaluated_depsgraph_get()
        ep = suit_pants.evaluated_get(deps)
        pm = ep.to_mesh()
        xf = carrier.matrix_world.inverted() @ ep.matrix_world
        pts_c = [xf @ v.co for v in pm.vertices]
        ep.to_mesh_clear()
        glute_zs = [
            p.z for p in pts_c
            if abs(p.x) < 0.16 and -0.16 < p.y < 0.12 and p.z > -0.12
        ]
        if glute_zs:
            g_min_z = min(glute_zs)
            delta_z = 0.0
            if g_min_z < -0.012:
                delta_z = -0.008 - g_min_z
            elif g_min_z > 0.018:
                delta_z = 0.004 - g_min_z
            delta_z = max(-0.04, min(0.04, delta_z))
            if abs(delta_z) > 1e-5:
                rig.location.z += delta_z
                bpy.context.view_layer.update()

    rig_rel_carrier = Matrix.Translation(rig.location) @ rig.rotation_euler.to_matrix().to_4x4()

    # Re-solve seated legs after glute cushion contact adjustment
    for side in ("L", "R"):
        _solve_leg_ik(rig, rig_rel_carrier, side, C.ankle_local(side, 1), (0.0, 0.90, 0.44), min_knee_splay_x=0.175, frame=None)

    pb_c1 = rig.pose.bones.get("head") or rig.pose.bones.get("neck02") or rig.pose.bones.get("neck01") or rig.pose.bones["spine01"]
    sac_final = carrier.matrix_world.inverted() @ (rig.matrix_world @ pb_sac.head)
    c1_final = carrier.matrix_world.inverted() @ (rig.matrix_world @ pb_c1.head)
    return rig_rel_carrier, (float(sac_final.x), float(sac_final.y), float(sac_final.z)), (float(c1_final.x), float(c1_final.y), float(c1_final.z))


def _animate_pilot_carrier(carrier: bpy.types.Object, total_frames: int) -> None:
    """Keyframe `Pilot_Carrier` on every frame with LINEAR interpolation for 0.000mm contract accuracy."""
    carrier.rotation_mode = "XYZ"
    for f in range(1, total_frames + 1):
        carrier.location = Vector(C.pilot_carrier_location(f))
        carrier.rotation_euler = Euler(C.pilot_carrier_rotation_euler(f), "XYZ")
        carrier.keyframe_insert("location", frame=f)
        carrier.keyframe_insert("rotation_euler", frame=f)
    _set_fcurve_interp(carrier, "LINEAR", skip_prefix="")


def _animate_pilot_choreography(
    rig: bpy.types.Object,
    rig_rel_carrier: Matrix,
    total_frames: int,
) -> None:
    """Animate full 6-beat pilot choreography across `1..total_frames`: controls -> handle dive ->
    violent yank -> brace -> canopy flinch -> garter reel-back -> 16G spine compression -> risers."""
    h_pulled = C.handle_local(C.EV["yank_end"])
    throttle_pos = C.COCKPIT["throttle"]
    stick_pos = C.COCKPIT["stick"]

    # Hand pose definitions: (grip_c, fwd_c, palm_c, pole_c, clav_elev, clav_protract, curl)
    def handle_hand(side: str, h_pt: tuple[float, float, float], brace: bool = False):
        sx = -1.0 if side == "L" else 1.0
        grip = (h_pt[0] + 0.036 * sx, h_pt[1], h_pt[2])
        fwd = (-0.25 * sx, 0.85, -0.45) if not brace else (-0.20 * sx, 0.75, -0.62)
        palm = (-0.30 * sx, -0.78, 0.55)
        pole = (0.30 * sx, -0.45, -0.84) if brace else (0.60 * sx, -0.55, -0.58)
        return grip, fwd, palm, pole

    def pose_key(
        f: int,
        lean_aft: float,
        twist: float,
        head_p: float,
        head_y: float,
        arm_l: tuple,
        arm_r: tuple,
        ankle_l: tuple[float, float, float],
        ankle_r: tuple[float, float, float],
        foot_fwd: tuple[float, float, float],
        jaw_deg: float = 2.0,
        brow: float = -0.35,
        eye_y: float = 0.0,
        eye_p: float = 0.0,
        lip_n: float = 0.0,
        knee_splay: float = 0.175,
    ) -> None:
        t = (f - 1) / float(C.FPS)
        comp_cm = C.spine_compression_cm(f)
        _apply_spine_and_head_pose(rig, rig_rel_carrier, lean_aft, twist, 0.0, head_p, head_y, 0.0, comp_cm, frame=f)
        _solve_arm_ik(rig, rig_rel_carrier, "L", *arm_l, frame=f)
        _solve_arm_ik(rig, rig_rel_carrier, "R", *arm_r, frame=f)
        _solve_leg_ik(rig, rig_rel_carrier, "L", ankle_l, foot_fwd, min_knee_splay_x=knee_splay, frame=f)
        _solve_leg_ik(rig, rig_rel_carrier, "R", ankle_r, foot_fwd, min_knee_splay_x=knee_splay, frame=f)
        mc.apply_facial_acting(
            rig,
            t=t,
            frame=f,
            jaw_open_deg=min(12.0, max(0.0, jaw_deg)),
            brow_lift=brow,
            smile=0.0,
            blink_period=1000.0,
            blink_phase=10.0,
            eye_yaw_deg=eye_y,
            eye_pitch_deg=eye_p,
            lip_narrow=lip_n,
        )

    ped_l, ped_r = C.ankle_local("L", 1), C.ankle_local("R", 1)
    foot_ped = (0.0, 0.90, 0.44)
    foot_reeled = (0.0, 0.85, -0.52)
    foot_hang = (0.0, 0.82, -0.57)

    # Beat 1a: f=1..59 — Panicked cockpit scan, Left hand on throttle, Right hand on centre stick
    ctrl_l = (throttle_pos, (-0.25, 0.92, -0.28), (0.35, -0.25, -0.90), (-0.75, -0.45, -0.48), 2.0, 10.0, 0.72)
    ctrl_r = (stick_pos, (-0.15, 0.85, -0.50), (-0.92, -0.35, -0.15), (0.65, -0.50, -0.55), 4.0, 22.0, 0.88)
    for f, hy, hp, ey, ep, jw in [
        (1, -6.0, 4.0, -4.0, -2.0, 2.2),
        (28, 8.0, 6.0, 5.0, -4.0, 3.0),
        (59, 0.0, 7.0, 0.0, -5.0, 3.8),
    ]:
        # Upright cockpit reach (2.5 deg aft, +10 deg right-forward twist) so stick at y=0.55 is comfortably reached
        pose_key(f, 2.5, 10.0, hp, hy, ctrl_l, ctrl_r, ped_l, ped_r, foot_ped, jaw_deg=jw, brow=-0.50, eye_y=ey, eye_p=ep)

    # Beat 1b: f=60..130 — Both hands sweep cleanly OVER thighs and drop into crotch/handle channel
    reach_waypoints = [
        (85, 6.5, 13.0, (-0.14, 0.42, 0.28), (0.14, 0.42, 0.28), 0.30, -8.0),
        (108, 9.0, 15.0, (-0.07, 0.38, 0.16), (0.07, 0.38, 0.16), 0.45, -10.0),
    ]
    for f, laft, hp, gl, gr, crl, ep in reach_waypoints:
        al = (gl, (0.25, 0.82, -0.50), (0.35, -0.55, -0.75), (-0.70, -0.45, -0.55), 3.0, 14.0, crl)
        ar = (gr, (-0.25, 0.82, -0.50), (-0.35, -0.55, -0.75), (0.70, -0.45, -0.55), 3.0, 14.0, crl)
        pose_key(f, laft, 0.0, hp, 0.0, al, ar, ped_l, ped_r, foot_ped, jaw_deg=4.5, brow=-0.62, eye_p=ep)

    # Grip closed on handle (f=130..150)
    for f, laft, hp, jw, ep in [(130, 9.0, 12.0, 4.8, -8.0), (148, 9.5, 8.0, 5.8, -3.0), (150, 9.5, 6.0, 6.5, 0.0)]:
        gl, fl, pl, pol = handle_hand("L", C.handle_local(f), brace=False)
        gr, fr, pr, por = handle_hand("R", C.handle_local(f), brace=False)
        pose_key(
            f, laft, 0.0, hp, 0.0,
            (gl, fl, pl, pol, 4.0, 15.0, 0.96),
            (gr, fr, pr, por, 4.0, 15.0, 0.96),
            ped_l, ped_r, foot_ped,
            jaw_deg=jw, brow=-0.76, eye_p=ep, lip_n=0.25,
        )

    # Beat 1c: f=150..168 — VIOLENT upward handle yank (z: 0.06 -> 0.24)
    for f in (153, 156, 159, 162, 165, 168):
        u = C.smootherstep(C.EV["yank_start"], C.EV["yank_end"], f)
        h_pt = C.handle_local(f)
        gl, fl, pl, pol = handle_hand("L", h_pt, brace=(u > 0.65))
        gr, fr, pr, por = handle_hand("R", h_pt, brace=(u > 0.65))
        elev = 4.0 + 4.5 * math.sin(u * math.pi)
        pose_key(
            f, C.lerp(9.5, 14.5, u), 0.0, C.lerp(6.0, -5.0, u), 0.0,
            (gl, fl, pl, pol, elev, C.lerp(15.0, 8.0, u), 0.98),
            (gr, fr, pr, por, elev, C.lerp(15.0, 8.0, u), 0.98),
            ped_l, ped_r, foot_ped,
            jaw_deg=7.4, brow=-0.85, eye_p=1.5, lip_n=0.35,
        )

    # Beat 1d + Beat 2: Brace against seat/headbox (f=176..609) + Canopy Shatter flinch at f=300
    gl_b, fl_b, pl_b, pol_b = handle_hand("L", h_pulled, brace=True)
    gr_b, fr_b, pr_b, por_b = handle_hand("R", h_pulled, brace=True)
    brace_l = (gl_b, fl_b, pl_b, pol_b, 3.0, 7.0, 0.98)
    brace_r = (gr_b, fr_b, pr_b, por_b, 3.0, 7.0, 0.98)

    for f, hp, jw, br, ep in [
        (176, -8.0, 4.5, -0.75, 1.0),
        (230, -8.5, 3.5, -0.65, 2.0),
        (282, -7.5, 4.0, -0.60, 6.0),   # MDC spark
        (302, -9.5, 7.8, 0.45, 7.0),    # Canopy detonation startled shock
        (350, -8.0, 4.8, 0.25, 4.0),
        (430, -8.0, 3.8, -0.40, 2.0),
        (520, -8.2, 3.5, -0.50, 1.0),
        (589, -8.2, 4.0, -0.60, 0.0),
    ]:
        pose_key(f, 15.0, 0.0, hp, 0.0, brace_l, brace_r, ped_l, ped_r, foot_ped, jaw_deg=jw, brow=br, eye_p=ep)

    # Beat 3: Leg-restraint garters reel boots cleanly back (f=610..650), then 16G catapult ignition (f=700)
    for f in (610, 620, 630, 640, 650):
        u = C.smootherstep(C.EV["garter_start"], C.EV["garter_end"], f)
        al_k = C.ankle_local("L", f)
        ar_k = C.ankle_local("R", f)
        f_fwd = C.lerp3(foot_ped, foot_reeled, u)
        pose_key(f, 15.0, 0.0, -8.0, 0.0, brace_l, brace_r, al_k, ar_k, f_fwd, jaw_deg=5.2, brow=-0.72, eye_p=-2.0)

    re_l, re_r = C.ankle_local("L", C.EV["garter_end"]), C.ankle_local("R", C.EV["garter_end"])
    for f, hp, jw, br in [
        (700, -8.5, 6.5, -0.82),
        (740, -7.0, 7.5, -0.88),
        (815, -7.2, 7.2, -0.85),
        (888, -7.5, 6.8, -0.80),
    ]:
        pose_key(f, 15.0, 0.0, hp, 0.0, brace_l, brace_r, re_l, re_r, foot_reeled, jaw_deg=jw, brow=br, lip_n=0.20)

    # Beat 4: f=889..1236 — Spine X-ray (-2.5 cm compression across f=930..1080; hands stay locked on handle!)
    for f in (930, 955, 980, 1005, 1030, 1055, 1080, 1140, 1200, 1236):
        u_c = C.spine_compression_cm(f) / C.SPINE_MAX_COMPRESSION_CM
        pose_key(
            f, 15.0, 0.0, -7.5 + 2.0 * u_c, 0.0,
            (gl_b, fl_b, pl_b, pol_b, 3.0 + 3.0 * u_c, 7.0, 0.98),
            (gr_b, fr_b, pr_b, por_b, 3.0 + 3.0 * u_c, 7.0, 0.98),
            re_l, re_r, foot_reeled,
            jaw_deg=7.2, brow=-0.85, lip_n=0.25,
        )

    # Beat 5: f=1237..1530 — Rocket motor + drogue stabilization wobble
    for f, hy, hp in [(1250, 2.0, -7.0), (1310, -2.5, -5.5), (1365, 1.5, -6.5), (1400, -1.5, -7.5), (1460, 1.0, -6.0), (1530, 0.0, -5.0)]:
        pose_key(
            f, 15.0, hy * 0.4, hp, hy,
            (gl_b, fl_b, pl_b, pol_b, 4.5, 7.0, 0.98),
            (gr_b, fr_b, pr_b, por_b, 4.5, 7.0, 0.98),
            re_l, re_r, foot_reeled,
            jaw_deg=5.2, brow=-0.65,
        )

    # Beat 6: f=1545..1800 — Seat separation -> release handle -> reach shoulder risers & relaxed leg hang!
    pose_key(
        1545, 15.0, 0.0, -5.0, 0.0,
        (gl_b, fl_b, pl_b, pol_b, 4.0, 7.0, 0.85),
        (gr_b, fr_b, pr_b, por_b, 4.0, 7.0, 0.85),
        re_l, re_r, foot_reeled,
        jaw_deg=5.5, brow=-0.50,
    )
    # Release outward briefly (f=1558) as seat drops below
    rel_l = ((-0.16, 0.25, 0.34), (-0.30, 0.70, 0.65), (0.60, -0.50, -0.60), (-0.75, -0.35, -0.55), 4.0, 6.0, 0.30)
    rel_r = ((0.16, 0.25, 0.34), (0.30, 0.70, 0.65), (-0.60, -0.50, -0.60), (0.75, -0.35, -0.55), 4.0, 6.0, 0.30)
    pose_key(
        1558, 13.0, 0.0, -2.0, 0.0,
        rel_l, rel_r,
        (-0.14, 0.28, -0.36), (0.14, 0.28, -0.36), foot_hang,
        jaw_deg=7.5, brow=0.35, knee_splay=0.145,
    )

    # Grab shoulder parachute risers above C.RISER_ATTACH_L/R = (±0.17, -0.10, 0.60)
    riser_l = ((-0.175, -0.075, 0.68), (0.30, -0.25, 0.90), (0.92, -0.30, -0.20), (-0.50, 0.45, -0.74), 7.5, 4.0, 0.92)
    riser_r = ((0.175, -0.075, 0.68), (-0.30, -0.25, 0.90), (-0.92, -0.30, -0.20), (0.50, 0.45, -0.74), 7.5, 4.0, 0.92)
    hang_l, hang_r = (-0.14, 0.20, -0.58), (0.14, 0.20, -0.58)
    for f, laft, hp, hy, jw, br, ep in [
        (1585, 11.0, 0.0, 0.0, 5.5, 0.20, 4.0),
        (1650, 10.0, -3.0, -3.0, 3.8, -0.15, 6.0),   # main chute full blossom!
        (1720, 10.0, 2.0, 5.0, 2.0, -0.25, 2.0),
        (total_frames, 10.0, 1.0, -2.0, 1.8, -0.25, 1.0),
    ]:
        pose_key(f, laft, 0.0, hp, hy, riser_l, riser_r, hang_l, hang_r, foot_hang, jaw_deg=jw, brow=br, eye_p=ep, knee_splay=0.145)

    # Intentional crisp ~150ms natural human blinks away from QA hero frames (lids stay at -5.0 deg everywhere else)
    for f_open0, f_shut, f_open1 in ((40, 44, 49), (301, 305, 310), (1380, 1384, 1389), (1672, 1676, 1682)):
        if f_open1 <= total_frames:
            for bf, lid_deg in ((f_open0, -5.0), (f_shut, 27.5), (f_open1, -5.0)):
                mc.set_bone_char_rot(rig, "orbicularis03.L", lid_deg, 0.0, 0.0, frame=bf)
                mc.set_bone_char_rot(rig, "orbicularis03.R", lid_deg, 0.0, 0.0, frame=bf + 1 if lid_deg > 0 else bf)

    _set_fcurve_interp(rig, "BEZIER", skip_prefix="hide_")


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================
def build_pilot(col: bpy.types.Collection, total_frames: int = C.TOTAL_FRAMES) -> dict:
    """Build & animate the 1.78m MakeHuman MPFB2 Fighter Pilot inside `Pilot_Carrier` (`1..total_frames`)."""
    scene = bpy.context.scene
    prev_frame = scene.frame_current
    scene.frame_set(1)

    # 1. Create Pilot_Carrier root Empty (world trajectory animated from ejection_contract.py)
    carrier = bpy.data.objects.new("Pilot_Carrier", None)
    carrier.empty_display_type = "PLAIN_AXES"
    carrier.empty_display_size = 0.22
    carrier.rotation_mode = "XYZ"
    _link_obj(carrier, col)

    # 2. Create 1.78m Adult Fighter Pilot via MPFB2 (`yaw_deg=180.0`: -Y_rig -> +Y_carrier forward, +X_rig -> -X_carrier Left)
    cfg = {
        "age": 0.58,
        "gender": 1.0,
        "muscle": 0.70,
        "weight": 0.48,
        "height": 0.60,
        "target_height_m": PILOT_HEIGHT_M,
        "skin_assets": ["middleage_caucasian_male.mhmat"],
        "iris_major_rgb": (0.055, 0.135, 0.240),
        "iris_minor_rgb": (0.015, 0.040, 0.085),
        "eyebrow_assets": ["eyebrow002.mhclo"],
        "hair_assets": ["short04.mhclo"],
        "hair_tint_rgb": (0.20, 0.14, 0.10),
        "hair_tint_fac": 0.68,
        "brow_tint_rgb": (0.18, 0.12, 0.09),
        "brow_tint_fac": 0.65,
        "pants_assets": ["mindfront_male_trousers_2.mhclo"],
        "shirt_assets": ["elvs_male_boho_top1.mhclo"],
        "shoe_assets": ["toigo_ankle_boots_male.mhclo"],
        "shoe_sat_override": 0.15,
        "shoe_value_boost": 0.40,
        "shoe_tint_rgb": (0.12, 0.11, 0.10),
        "shoe_tint_fac": 0.85,
        "shoe_roughness": 0.44,
    }
    actor = mc.create_realistic_scientist(
        name="Pilot",
        floor_z=0.0,
        start_xy=(0.0, 0.0),
        yaw_deg=180.0,
        cfg=cfg,
    )
    _normalize_pilot_height(actor, target_height_m=PILOT_HEIGHT_M)
    _tune_pilot_skin(actor["body"])
    _rename_pilot_hierarchy(actor)

    rig: bpy.types.Object = actor["rig"]
    body: bpy.types.Object = actor["body"]
    equipped: dict[str, bpy.types.Object] = actor["equipped"]

    # AGENTS.md Sec 1: Eyes & Eyelashes must NEVER cast shadow rays
    for pk in ("Eyes", "Eyelashes"):
        if pk in equipped and equipped[pk] is not None:
            equipped[pk].visible_shadow = False

    # Move all MPFB2 objects cleanly into target collection `col`
    for obj in [rig, body, *equipped.values()]:
        if obj is not None:
            _link_obj(obj, col)

    # Ensure `spine04..01` can translate cleanly for `-2.5 cm` Beat 4 spinal disc compression
    _unlock_spine_translation(rig)

    # 3. Cinch loose boho cuffs, tuck lower tunic skirt, apply Nomex suit material & shrink under-clothes body
    _cinch_shirt_cuffs(rig, equipped.get("Shirt"))
    _tuck_shirt_hem(rig, equipped.get("Shirt"))
    under_disp = _create_underclothes_shrink_mod(rig, body)
    _, _, eye_ctr_rig = (
        _eval_mesh_bounds_in(equipped["Eyes"], rig.matrix_world.inverted())
        if "Eyes" in equipped
        else (None, None, Vector((0.0, -0.10, 1.66)))
    )
    _tuck_hair_under_helmet(rig, equipped.get("Hair"), eye_ctr_rig.z)

    mat_suit_top = _make_nomex_suit_mat("Mat_Pilot_NomexTop", equipped.get("Shirt"))
    mat_suit_pants = _make_nomex_suit_mat("Mat_Pilot_NomexPants", equipped.get("Pants"))
    for slot_key, nmat in (("Shirt", mat_suit_top), ("Pants", mat_suit_pants)):
        sobj = equipped.get(slot_key)
        if sobj is not None and len(sobj.material_slots) > 0:
            sobj.material_slots[0].material = nmat

    # 4. Pilot Gear Materials
    gear_mats = {
        "helmet_shell": _make_gear_pbr_mat("Mat_Pilot_HelmetShell", (0.085, 0.098, 0.112), roughness=0.26, metallic=0.18, coat=0.55, coat_rough=0.12, noise_scale=85.0, bump_strength=0.04),
        "helmet_dark": _make_gear_pbr_mat("Mat_Pilot_HelmetDark", (0.032, 0.035, 0.038), roughness=0.52, noise_scale=160.0, bump_strength=0.08),
        "helmet_visor": _make_smoked_visor_mat(),
        "hardware": _make_gear_pbr_mat("Mat_Pilot_Hardware", (0.045, 0.050, 0.055), roughness=0.35, metallic=0.82, noise_scale=120.0, bump_strength=0.05),
        "webbing": _make_gear_pbr_mat("Mat_Pilot_HarnessWebbing", (0.125, 0.118, 0.085), roughness=0.84, sheen=0.35, noise_scale=220.0, bump_strength=0.14, weave_grid=True),
        "gsuit": _make_gear_pbr_mat("Mat_Pilot_GSuit", (0.036, 0.052, 0.028), roughness=0.78, sheen=0.30, noise_scale=190.0, bump_strength=0.11, weave_grid=True),
        "undershirt": _make_gear_pbr_mat("Mat_Pilot_UndershirtCollar", (0.044, 0.062, 0.034), roughness=0.82, sheen=0.40, noise_scale=210.0, bump_strength=0.06, weave_grid=True),
        "garter_dark": _make_gear_pbr_mat("Mat_Pilot_GarterStrap", (0.022, 0.024, 0.026), roughness=0.82, sheen=0.25, noise_scale=220.0, bump_strength=0.12, weave_grid=True),
    }

    # 5. Build HGU-55/P Flight Helmet in rest pose (parented to `head` bone)
    helmet_parts = _build_pilot_helmet(rig, body, equipped, gear_mats, col)

    # 6. Seat Pilot_Rig inside Pilot_Carrier (15 deg aft recline, sacrum/c1 aligned to C.PILOT)
    rig_rel_carrier, sacrum_local, c1_local = _calibrate_seated_placement(carrier, rig, equipped.get("Pants"))
    sac_dz = abs(sacrum_local[2] - C.PILOT["sacrum"][2])
    c1_dz = abs(c1_local[2] - C.PILOT["c1_skull_base"][2])
    _, _, eye_c_dbg = (
        _eval_mesh_bounds_in(equipped["Eyes"], carrier.matrix_world.inverted())
        if "Eyes" in equipped
        else (None, None, Vector(C.PILOT["eye_mid"]))
    )
    print(
        f"[Pilot Calibration] sacrum_local=({sacrum_local[0]:.4f}, {sacrum_local[1]:.4f}, {sacrum_local[2]:.4f}) "
        f"(|dz|={sac_dz:.4f}m), c1_local=({c1_local[0]:.4f}, {c1_local[1]:.4f}, {c1_local[2]:.4f}) "
        f"(|dz|={c1_dz:.4f}m), eye_mid_c=({eye_c_dbg.x:.4f}, {eye_c_dbg.y:.4f}, {eye_c_dbg.z:.4f})",
        flush=True,
    )
    assert sac_dz <= 0.042, f"Pilot sacrum_local misaligned: {sacrum_local} vs {C.PILOT['sacrum']}"
    # Anatomical 1.78m MakeHuman head.head (C1 skull base) sits ~0.657m above sacrum along 15deg spine (~0.777m vs 0.70m neck01 ref)
    assert c1_dz <= 0.095, f"Pilot c1_local misaligned: {c1_local} vs {C.PILOT['c1_skull_base']}"

    # 7. Build Seated Parachute Harness Webbing, CWU-27/P Collar/Bib, Shoulder Riser D-Rings & G-Suit Leg Wraps
    #    (measured against pre-shrink `body`!)
    harness_parts = _build_pilot_harness_and_gsuit(
        carrier, rig, equipped.get("Shirt"), equipped.get("Pants"), gear_mats, col, body=body
    )

    # Enable -1.4cm pelvic/hip under-clothing shrink (modifier index 0 BEFORE Armature) prior to choreography
    under_disp.show_viewport = True
    under_disp.show_render = True
    arm_mod_indices = [i for i, m in enumerate(body.modifiers) if m.type == "ARMATURE"]
    assert len(body.modifiers) > 0 and body.modifiers[0].type == "DISPLACE" and body.modifiers[0].vertex_group == "UnderClothes"
    assert not arm_mod_indices or arm_mod_indices[0] > 0, f"DISPLACE must precede Armature: {[m.type for m in body.modifiers]}"
    bpy.context.view_layer.update()

    # 8. Animate Pilot_Carrier (world trajectory) + Full 6-Beat Pilot Choreography
    _animate_pilot_carrier(carrier, total_frames)
    _animate_pilot_choreography(rig, rig_rel_carrier, total_frames)

    # 9. Patch ALL Pilot Materials for Beat 4 X-Ray Window (C.xray_body_weight(f), f=895..1236)
    xray_nodes: list = []
    patched_ids: set[int] = set()

    for slot in body.material_slots:
        _patch_material_for_xray(slot.material, "xray_glass", rim_peak=0.14, patched_ids=patched_ids, xray_nodes=xray_nodes)

    for key, mode, rim in (
        ("Shirt", "xray_glass", 0.14),
        ("Pants", "xray_glass", 0.14),
        ("Shoes", "xray_glass", 0.11),
        ("Eyes", "fade_out", 0.0),
        ("Eyelashes", "fade_out", 0.0),
        ("Teeth", "fade_out", 0.0),
        ("Tongue", "fade_out", 0.0),
        ("Hair", "fade_out", 0.0),
        ("Eyebrows", "fade_out", 0.0),
    ):
        obj = equipped.get(key)
        if obj is not None:
            for slot in obj.material_slots:
                _patch_material_for_xray(slot.material, mode, rim_peak=rim, patched_ids=patched_ids, xray_nodes=xray_nodes)

    for obj in (*helmet_parts, *harness_parts):
        for slot in obj.material_slots:
            _patch_material_for_xray(slot.material, "xray_glass", rim_peak=0.11, patched_ids=patched_ids, xray_nodes=xray_nodes)

    _animate_xray_nodes(xray_nodes, total_frames)

    scene.frame_set(prev_frame)
    bpy.context.view_layer.update()

    return {
        "carrier": carrier,
        "rig": rig,
        "body": body,
        "equipped": equipped,
        "helmet_parts": helmet_parts,
        "harness_parts": harness_parts,
        "xray_nodes": xray_nodes,
        "sacrum_local": sacrum_local,
        "c1_local": c1_local,
    }


# ============================================================================
# STANDALONE SELF-TEST (factory reset, builds pilot, renders f=155 & f=1080)
# ============================================================================
def _run_standalone_verification() -> None:
    out_dir = Path("/tmp/fje_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = C.TOTAL_FRAMES
    scene.render.fps = C.FPS

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 6
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 25
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        pass

    world = bpy.data.worlds.new("Test_World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.16, 0.24, 0.38, 1.0)
        bg.inputs["Strength"].default_value = 0.75

    pilot_col = bpy.data.collections.new("Pilot_Col")
    scene.collection.children.link(pilot_col)
    res = build_pilot(pilot_col, total_frames=C.TOTAL_FRAMES)

    # Diagnostic readout of contract alignment & prefix compliance
    all_objs = [o.name for o in pilot_col.objects]
    bad = [n for n in all_objs if not n.startswith("Pilot_")]
    assert not bad, f"Non-Pilot_ object names found: {bad}"
    print(f"[Pilot QA] Built {len(all_objs)} Pilot_ objects, {len(res['xray_nodes'])} XRayBody nodes")
    print(f"[Pilot QA] Calibrated sacrum={res['sacrum_local']}, c1={res['c1_local']}")

    carrier = res["carrier"]
    rig = res["rig"]

    # Add temporary test key/catchlight + handle proxy (outside Pilot_Col)
    def aim_quat(src: Vector, dst: Vector) -> Euler:
        return (dst - src).normalized().to_track_quat("-Z", "Y").to_euler("XYZ")

    key_ld = bpy.data.lights.new("Test_Sun", type="SUN")
    key_ld.energy = 3.2
    key_obj = bpy.data.objects.new("Test_Sun", key_ld)
    key_obj.rotation_euler = aim_quat(Vector((-2.0, 4.0, 3.0)), Vector((0.0, 0.0, 0.5)))
    scene.collection.objects.link(key_obj)

    catch_ld = bpy.data.lights.new("Test_Catchlight", type="AREA")
    catch_ld.energy = 52.0
    catch_ld.size = 0.42
    catch_obj = bpy.data.objects.new("Test_Catchlight", catch_ld)
    catch_obj.parent = carrier
    catch_obj.location = Vector((0.35, 1.15, 0.90))
    catch_obj.rotation_euler = aim_quat(catch_obj.location, Vector((0.0, 0.02, 0.81)))
    scene.collection.objects.link(catch_obj)

    cam_d = bpy.data.cameras.new("Test_Cam")
    cam_d.lens = 45.0
    cam = bpy.data.objects.new("Test_Cam", cam_d)
    cam.parent = carrier
    scene.collection.objects.link(cam)
    scene.camera = cam

    for f, fname, cam_pos, tgt_pos in [
        (155, "pilot_f0155.png", Vector((0.68, 1.55, 0.62)), Vector((0.0, 0.12, 0.44))),
        (1080, "pilot_f1080.png", Vector((0.92, 1.48, 0.55)), Vector((0.0, -0.05, 0.42))),
    ]:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        cam.location = cam_pos
        cam.rotation_euler = aim_quat(cam_pos, tgt_pos)
        pb_c1 = rig.pose.bones.get("head") or rig.pose.bones.get("neck02") or rig.pose.bones.get("neck01") or rig.pose.bones["spine01"]
        c1_c = carrier.matrix_world.inverted() @ (rig.matrix_world @ pb_c1.head)
        print(f"[Pilot QA f={f}] XRayBody={C.xray_body_weight(f):.3f}, c1_local=({c1_c.x:.4f}, {c1_c.y:.4f}, {c1_c.z:.4f})")
        scene.render.filepath = str(out_dir / fname)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    _run_standalone_verification()
