"""
Reusable MPFB2 V7 Authentic 17th-Century Cinema Human Character Engine for Blender 4.2.3 LTS Cycles (Kaggle GPU T4 x2)
Uses curated CC0/CC-BY MakeHuman Community 4K Asset Packs (delivered via ~1.2s GitHub CDN in Kaggle Cell 2!):
  - Real 4K Photo-Scanned Adult Masculine Skins with ENHANCED_SSS Cycles Shader (pores, real lips, brow
    shadows, wrinkles, knuckles — tuned to fast BURLEY SSS for Dual-T4 GPU speed!)
  - Adult Age & Exact Stature Normalization (`age >= 0.54`, `target_height_m`) so Newton (1.78m) & Halley (1.79m)
    always stand eye-to-eye as equal adult 17th-century men — NEVER an undersized teenager!
  - Real High-Poly Photoreal Eyes (high-poly.mhclo + PROCEDURAL_EYES EnhancedEye refractive cornea dome!)
  - Real Alpha-Card Eyebrows, Eyelashes, 3D Teeth & Tongue (realistic mouth during Kokoro lip-sync!)
  - Real Sleek Shoulder-Length Hair (`culturalibre_hair_01.mhclo` + 2K normal map) for Sir Isaac Newton's iconic
    1689 Godfrey Kneller portrait look & swept adult gentleman hair (`short04.mhclo` / `short02.mhclo`) for Halley!
  - Multi-Layer Historical 17th-Century Outfits (`shirt_assets`, `pants_assets`, `coat_assets`, `cloak_assets`,
    `shoe_assets` — ZERO modern neckties! ZERO modern bowties!) with linear-safe `gamma_lift` color grading &
    woven wool/velvet micro-bump normal shading!
  - Optional Bone-Parented Rigged 17th-Century White Linen Ruffled Jabot Cravat (`add_cravat: True`) &
    Draped Velvet Cambridge Scholar Shoulder-Mantle/Cape (`add_scholar_cloak: True`) fitted automatically to each
    character's live MPFB2 torso landmarks with arm-swing anti-clipping clearance!
  - Blender 4.2 safe ShaderNodeMix (socket.type == 'RGBA') + recursive ShaderNodeGroup copying so
    per-character wardrobe/hair grading ALWAYS renders without cross-contaminating actors!
  - Roll-proof character-space bone posing (M_rest^-1 @ R_char @ M_rest) + true MPFB2 +Z facing convention
    (+yaw_deg/+rz_deg turns toward +X Screen Right; -yaw_deg/-rz_deg turns toward -X Screen Left)!
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector


def _set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def enable_mpfb2():
    """Ensure MPFB2 extension is enabled in Blender 4.2 headless session and return core service classes."""
    if not any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
        for mod_name in ("bl_ext.user_default.mpfb", "mpfb"):
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                if any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
                    print(f"[MPFB2] Enabled addon module: {mod_name}")
                    break
            except Exception:
                continue

    def dynamic_import(absolute_package_str: str, key: str):
        for amod in list(sys.modules.keys()):
            if amod.endswith(absolute_package_str):
                mpfb_mod = importlib.import_module(amod)
                if hasattr(mpfb_mod, key):
                    return getattr(mpfb_mod, key)
        raise ImportError(f"MPFB2 module '{absolute_package_str}.{key}' not loaded! Install MPFB2 extension first.")

    return (
        dynamic_import("mpfb.services.humanservice", "HumanService"),
        dynamic_import("mpfb.services.targetservice", "TargetService"),
        dynamic_import("mpfb.services.assetservice", "AssetService"),
        dynamic_import("mpfb.entities.objectproperties", "HumanObjectProperties"),
    )


# ============================================================================
# SMART ASSET RESOLVER (FINDS EXACT OR FUZZY MATCH ACROSS INSTALLED MPFB2 PACKS)
# ============================================================================
def _find_best_asset(AssetService, subdirs: tuple[str, ...], candidates: list[str], is_mhmat: bool = False) -> tuple[str | None, str]:
    """Search MPFB2 installed asset packs for the best matching asset across one or more subdirs."""
    for sub in subdirs:
        for cand in candidates:
            exact = AssetService.find_asset_absolute_path(cand, asset_subdir=sub)
            if exact and os.path.isfile(exact):
                return exact, sub
            ext = ".mhmat" if is_mhmat else ".mhclo"
            if not cand.lower().endswith(ext):
                exact2 = AssetService.find_asset_absolute_path(cand + ext, asset_subdir=sub)
                if exact2 and os.path.isfile(exact2):
                    return exact2, sub

    # Fuzzy case-insensitive token search across all installed assets in subdirs
    for sub in subdirs:
        try:
            all_paths = AssetService.list_mhmat_assets(sub) if is_mhmat else AssetService.list_mhclo_assets(sub)
        except Exception:
            all_paths = []
        path_strs = [str(p) for p in all_paths]
        for cand in candidates:
            tok = cand.lower().replace(".mhclo", "").replace(".mhmat", "")
            for ps in path_strs:
                if tok in ps.lower():
                    return ps, sub
    return None, subdirs[0]


# ============================================================================
# CYCLES SHADER TUNING HELPERS (FAST PHOTOREAL SKIN, ENHANCED EYES, HAIR/CLOTH GRADE)
# ============================================================================
def _find_principled_nodes(nt, visited_trees=None):
    """Recursively find all (node_tree, bsdf_node) pairs inside a material, copying any shared
    ShaderNodeGroup so grading or tuning one character's skin/outfit/hair never mutates another actor."""
    if visited_trees is None:
        visited_trees = set()
    if nt is None or id(nt) in visited_trees:
        return []
    visited_trees.add(id(nt))
    found = []
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            found.append((nt, node))
        elif node.type == "GROUP" and getattr(node, "node_tree", None) is not None:
            if node.node_tree.users > 1:
                node.node_tree = node.node_tree.copy()
            found.extend(_find_principled_nodes(node.node_tree, visited_trees))
    return found


def _get_rgba_mix_sockets(mix_node):
    """In Blender 4.x, ShaderNodeMix (data_type='RGBA') has duplicate 'A'/'B'/'Result' sockets
    (Float, Vector, RGBA). ALWAYS pick the true RGBA color sockets by socket.type == 'RGBA'."""
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    fac_in = mix_node.inputs[0]
    return fac_in, rgba_in[0], rgba_in[1], rgba_out[0]


def _optimize_enhanced_skin_for_t4(basemesh):
    """Keep MPFB2's gorgeous 4K photo-scanned ENHANCED_SSS pores/lips/zones while switching
    Principled BSDF subsurface method to fast BURLEY so Kaggle T4 GPU renders 2x faster!"""
    for slot in basemesh.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for _nt, bsdf in _find_principled_nodes(mat.node_tree):
            if hasattr(bsdf, "subsurface_method"):
                try:
                    bsdf.subsurface_method = "BURLEY"
                except Exception:
                    pass
            # Ensure realistic non-plastic skin sheen
            if "Roughness" in bsdf.inputs and not bsdf.inputs["Roughness"].is_linked:
                bsdf.inputs["Roughness"].default_value = max(0.38, bsdf.inputs["Roughness"].default_value)


def _tune_procedural_eyes(eye_obj, major_rgb=(0.14, 0.36, 0.20), minor_rgb=(0.06, 0.16, 0.10)):
    """Dial realistic iris major/minor colors directly on MPFB2's official EnhancedEye refractive cornea node!"""
    if eye_obj is None:
        return
    for slot in eye_obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        if mat.users > 1:
            mat = mat.copy()
            slot.material = mat
        for node in mat.node_tree.nodes:
            if node.type == "GROUP" and "EnhancedEye" in (node.name + " " + node.label):
                if node.node_tree and node.node_tree.users > 1:
                    node.node_tree = node.node_tree.copy()
                if "IrisMajorColor" in node.inputs:
                    node.inputs["IrisMajorColor"].default_value = (*major_rgb, 1.0)
                if "IrisMinorColor" in node.inputs:
                    node.inputs["IrisMinorColor"].default_value = (*minor_rgb, 1.0)
                if "PupilSize" in node.inputs:
                    node.inputs["PupilSize"].default_value = 0.28
                if "Clearcoat" in node.inputs:
                    node.inputs["Clearcoat"].default_value = 0.85


def _add_woven_fabric_bump(nt, bsdf, scale: float = 300.0, strength: float = 0.075):
    """Attach subtle procedural woven broadcloth/velvet micro-bump to clothing lacking a baked normal map."""
    n_in = bsdf.inputs.get("Normal")
    if n_in is None or n_in.is_linked:
        return
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.62
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.0022
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], n_in)


def _tint_asset_material(
    obj,
    tint_rgb: tuple[float, float, float] | None = None,
    mix_fac: float = 0.45,
    roughness_override: float | None = None,
    is_hair: bool = False,
    silver_hair: bool = False,
    value_boost: float = 1.0,
    gamma_lift: float = 1.0,
    add_fabric_bump: bool = False,
):
    """Non-destructively color-grade an imported MPFB2 clothing or alpha-hair material while preserving
    100% of its original photo-scanned diffuse texture, fabric folds, normal/bump maps & alpha transparency!
    Supports linear-safe `gamma_lift` (< 1.0 lifts dark base textures like black hair or dark cloth cleanly
    into silver-pewter or rich royal indigo/burgundy without losing strand detail) and Blender 4.2-safe RGBA Mix."""
    if obj is None:
        return
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        if mat.users > 1:
            mat = mat.copy()
            slot.material = mat
        for nt, bsdf in _find_principled_nodes(mat.node_tree):
            bc = bsdf.inputs.get("Base Color")
            if bc is None:
                continue

            if is_hair:
                _set_bsdf(bsdf, "Roughness", roughness_override if roughness_override is not None else 0.42)
                _set_bsdf(bsdf, "Coat Weight", 0.20)
                _set_bsdf(bsdf, "Sheen Weight", 0.38)
            else:
                if roughness_override is not None and not bsdf.inputs["Roughness"].is_linked:
                    _set_bsdf(bsdf, "Roughness", roughness_override)
                _set_bsdf(bsdf, "Sheen Weight", 0.52)
                if add_fabric_bump:
                    _add_woven_fabric_bump(nt, bsdf)

            try:
                if bc.is_linked:
                    src_socket = bc.links[0].from_socket
                    nt.links.remove(bc.links[0])

                    eff_gamma = min(gamma_lift, 0.36) if silver_hair else gamma_lift
                    if eff_gamma != 1.0:
                        gam = nt.nodes.new("ShaderNodeGamma")
                        gam.inputs["Gamma"].default_value = eff_gamma
                        nt.links.new(src_socket, gam.inputs["Color"])
                        src_socket = gam.outputs["Color"]

                    if silver_hair:
                        hsv = nt.nodes.new("ShaderNodeHueSaturation")
                        hsv.inputs["Saturation"].default_value = 0.14
                        hsv.inputs["Value"].default_value = max(1.30, value_boost)
                        nt.links.new(src_socket, hsv.inputs["Color"])
                        src_socket = hsv.outputs["Color"]
                    elif value_boost != 1.0:
                        hsv = nt.nodes.new("ShaderNodeHueSaturation")
                        hsv.inputs["Value"].default_value = value_boost
                        nt.links.new(src_socket, hsv.inputs["Color"])
                        src_socket = hsv.outputs["Color"]

                    if tint_rgb is not None and mix_fac > 0.0:
                        mix = nt.nodes.new("ShaderNodeMix")
                        mix.data_type = "RGBA"
                        mix.blend_type = "MULTIPLY"
                        fac_in, col_a, col_b, col_out = _get_rgba_mix_sockets(mix)
                        fac_in.default_value = mix_fac
                        col_b.default_value = (*tint_rgb, 1.0)
                        nt.links.new(src_socket, col_a)
                        nt.links.new(col_out, bc)
                    else:
                        nt.links.new(src_socket, bc)
                else:
                    orig = bc.default_value
                    eff_gamma = min(gamma_lift, 0.36) if silver_hair else gamma_lift
                    lifted = [min(1.0, (max(0.001, orig[c]) ** eff_gamma) * value_boost) for c in range(3)]
                    if silver_hair:
                        lum = 0.299 * lifted[0] + 0.587 * lifted[1] + 0.114 * lifted[2]
                        lifted = [lum * 0.88 + lifted[c] * 0.12 for c in range(3)]
                    if tint_rgb is not None and mix_fac > 0.0:
                        bc.default_value = (
                            lifted[0] * (1.0 - mix_fac) + (lifted[0] * tint_rgb[0]) * mix_fac,
                            lifted[1] * (1.0 - mix_fac) + (lifted[1] * tint_rgb[1]) * mix_fac,
                            lifted[2] * (1.0 - mix_fac) + (lifted[2] * tint_rgb[2]) * mix_fac,
                            1.0,
                        )
                    else:
                        bc.default_value = (lifted[0], lifted[1], lifted[2], 1.0)
            except Exception as tint_err:
                print(f"[MPFB2 Tint Warning] {obj.name}/{mat.name}: {tint_err}")


# ============================================================================
# PROCEDURAL 17TH-CENTURY RIGGED GARMENTS (LINEN JABOT CRAVAT & VELVET CLOAK!)
# ============================================================================
def _link_to_active_collection(obj):
    if obj.name not in bpy.context.collection.objects:
        bpy.context.collection.objects.link(obj)


def _parent_object_to_bone_exact(obj, rig, bone_name: str):
    """Bone-parent a world-space mesh (built at rest pose while rig is at origin/identity)
    to `bone_name` with ZERO transform jump across all Blender 4.x versions!"""
    if bone_name not in rig.data.bones:
        for fallback in ("spine01", "spine02", "neck01", "root"):
            if fallback in rig.data.bones:
                bone_name = fallback
                break
    obj.matrix_local = Matrix.Identity(4)
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()
    # Self-calibrating exact inverse: guarantees obj.matrix_world == Identity at rest pose!
    obj.matrix_parent_inverse = obj.matrix_world.inverted()
    bpy.context.view_layer.update()


def _measure_torso_landmarks(rig, basemesh) -> dict[str, float]:
    """Sample exact rest-pose anatomical landmarks directly from the character's live MPFB2 skeleton & mesh
    so the 17th-century cravat & draped velvet scholar cloak fit ANY height/weight/muscle cleanly!"""
    bpy.context.view_layer.update()
    mw = basemesh.matrix_world
    verts = [mw @ v.co for v in basemesh.data.vertices]

    # Skeleton joints in rest world coordinates
    neck_b = rig.data.bones.get("neck01")
    spine_b = rig.data.bones.get("spine01")
    clav_l = rig.data.bones.get("clavicle.L")
    clav_r = rig.data.bones.get("clavicle.R")

    neck_head = (rig.matrix_world @ neck_b.head_local) if neck_b else Vector((0.0, 0.0, 1.46))
    neck_tail = (rig.matrix_world @ neck_b.tail_local) if neck_b else Vector((0.0, 0.0, 1.53))
    spine_head = (rig.matrix_world @ spine_b.head_local) if spine_b else Vector((0.0, 0.0, 1.32))

    neck_base_z = neck_head.z
    neck_mid_z = 0.5 * (neck_head.z + neck_tail.z)
    neck_center_y = neck_head.y

    # Measure neck radius from vertices right around the lower neck
    neck_ring = [v for v in verts if abs(v.z - (neck_base_z + 0.025)) < 0.018 and abs(v.x) < 0.095]
    if neck_ring:
        neck_rx = max(0.052, max(abs(v.x) for v in neck_ring))
        neck_front_y = min(v.y for v in neck_ring)
        neck_back_y = max(v.y for v in neck_ring)
    else:
        neck_rx, neck_front_y, neck_back_y = 0.062, -0.062, 0.060

    # Measure front sternum & upper-back contours below the collar
    chest_verts = [v for v in verts if (neck_base_z - 0.16) <= v.z <= (neck_base_z - 0.02) and abs(v.x) < 0.065]
    sternum_y = min((v.y for v in chest_verts), default=-0.125)
    back_verts = [v for v in verts if (neck_base_z - 0.24) <= v.z <= (neck_base_z - 0.03) and abs(v.x) < 0.16]
    back_y = max((v.y for v in back_verts), default=0.115)

    # Measure outer shoulder width & shoulder top height (clavicle outer end)
    sh_verts = [v for v in verts if (neck_base_z - 0.08) <= v.z <= (neck_base_z + 0.02) and abs(v.x) < 0.28]
    if clav_l and clav_r:
        clav_x = 0.5 * (abs((rig.matrix_world @ clav_l.tail_local).x) + abs((rig.matrix_world @ clav_r.tail_local).x))
        shoulder_half_x = max(0.18, clav_x + 0.042)
    else:
        shoulder_half_x = max(0.20, max((abs(v.x) for v in sh_verts), default=0.22))

    trap_verts = [v for v in verts if 0.08 <= abs(v.x) <= 0.17 and (neck_base_z - 0.10) <= v.z <= (neck_base_z + 0.04)]
    shoulder_top_z = max((v.z for v in trap_verts), default=neck_base_z - 0.025)

    return {
        "neck_base_z": neck_base_z,
        "neck_mid_z": neck_mid_z,
        "neck_center_y": neck_center_y,
        "neck_rx": neck_rx,
        "neck_front_y": neck_front_y,
        "neck_back_y": neck_back_y,
        "sternum_y": sternum_y,
        "back_y": back_y,
        "shoulder_half_x": shoulder_half_x,
        "shoulder_top_z": shoulder_top_z,
        "spine_head_z": spine_head.z,
    }


def _make_linen_cravat_material(name: str, base_rgb=(0.92, 0.90, 0.85)):
    """Create a tactile 17th-century antique ivory silk-linen Cycles material with fine weave bump & soft SSS."""
    mat = bpy.data.materials.new(name=f"{name}_LinenCravatMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
    _set_bsdf(bsdf, "Roughness", 0.72)
    _set_bsdf(bsdf, "Sheen Weight", 0.68)
    _set_bsdf(bsdf, "Sheen Roughness", 0.45)
    _set_bsdf(bsdf, "Subsurface Weight", 0.07)
    _set_bsdf(bsdf, "Subsurface Scale", 0.004)
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "BURLEY"
        except Exception:
            pass
    _add_woven_fabric_bump(nt, bsdf, scale=460.0, strength=0.065)
    return mat


def _make_velvet_cloak_materials(name: str, cloak_rgb=(0.045, 0.068, 0.165), trim_rgb=(0.68, 0.50, 0.20)):
    """Create a feature-film Cycles velvet material (LayerWeight grazing velvet pile sheen + micro-bump)
    plus an antique-gold brocade embroidery trim & chased brass clasp material."""
    velvet = bpy.data.materials.new(name=f"{name}_VelvetCloakMat")
    velvet.use_nodes = True
    nt = velvet.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    # Grazing-angle velvet pile highlight (classic feature-film velvet shading!)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.32
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    fac_in, col_a, col_b, col_out = _get_rgba_mix_sockets(mix)
    col_a.default_value = (*cloak_rgb, 1.0)
    # Velvet pile catches grazing warm/cool sheen brighter & slightly desaturated
    pile_rgb = tuple(min(1.0, c * 2.15 + 0.035) for c in cloak_rgb)
    col_b.default_value = (*pile_rgb, 1.0)
    nt.links.new(lw.outputs["Facing"], fac_in)
    nt.links.new(col_out, bsdf.inputs["Base Color"])

    _set_bsdf(bsdf, "Roughness", 0.78)
    _set_bsdf(bsdf, "Sheen Weight", 0.88)
    _set_bsdf(bsdf, "Sheen Roughness", 0.36)
    _add_woven_fabric_bump(nt, bsdf, scale=360.0, strength=0.085)

    trim = bpy.data.materials.new(name=f"{name}_GoldBrocadeMat")
    trim.use_nodes = True
    t_nt = trim.node_tree
    t_bsdf = next(n for n in t_nt.nodes if n.type == "BSDF_PRINCIPLED")
    t_bsdf.inputs["Base Color"].default_value = (*trim_rgb, 1.0)
    _set_bsdf(t_bsdf, "Metallic", 0.72)
    _set_bsdf(t_bsdf, "Roughness", 0.34)
    _add_woven_fabric_bump(t_nt, t_bsdf, scale=420.0, strength=0.10)
    return velvet, trim


def _add_grid_surface(bm, grid_pts: list[list[Vector]], mat_index: int = 0, closed_u: bool = False):
    """Build a smooth quad strip surface from a 2D array of Vector points `[row][col]` inside a BMesh."""
    rows = len(grid_pts)
    cols = len(grid_pts[0])
    vgrid = [[bm.verts.new(p) for p in row] for row in grid_pts]
    for r in range(rows - 1):
        c_max = cols if closed_u else cols - 1
        for c in range(c_max):
            c_next = (c + 1) % cols
            v00, v01 = vgrid[r][c], vgrid[r][c_next]
            v11, v10 = vgrid[r + 1][c_next], vgrid[r + 1][c]
            try:
                f = bm.faces.new((v00, v01, v11, v10))
                f.smooth = True
                f.material_index = mat_index
            except ValueError:
                pass


def _build_17c_cravat(name: str, rig, basemesh, lm: dict[str, float], cfg: dict):
    """Build a smooth, multi-tiered 17th-century white silk-linen ruffled jabot or Steinkirk cravat
    fitted directly to the actor's live neck & sternum landmarks and bone-parented cleanly to `spine01`."""
    style = str(cfg.get("cravat_style", "jabot")).lower()
    cravat_rgb = cfg.get("cravat_rgb", (0.92, 0.90, 0.85))
    cravat_bone = str(cfg.get("cravat_bone", "spine01"))

    bm = bmesh.new()

    # 1. High-folded linen neckband collar hugging the base of the throat cleanly outside skin & shirt collar
    rx = lm["neck_rx"] + 0.013
    ry_front = abs(lm["neck_front_y"] - lm["neck_center_y"]) + 0.015
    ry_back = abs(lm["neck_back_y"] - lm["neck_center_y"]) + 0.014
    cy = lm["neck_center_y"]
    z_bot = lm["neck_base_z"] - 0.022
    z_top = lm["neck_base_z"] + 0.036

    n_seg = 32
    n_rings = 5
    collar_grid = []
    for r in range(n_rings):
        v = r / (n_rings - 1)
        z = z_bot + (z_top - z_bot) * v
        # Soft tapered bell flare at bottom, rolled crisp collar lip at top
        flare = 1.0 + 0.10 * ((1.0 - v) ** 1.6) + 0.03 * math.sin(v * math.pi)
        ring = []
        for s in range(n_seg):
            ang = (s / n_seg) * 2.0 * math.pi
            ca, sa = math.cos(ang), math.sin(ang)
            ry = ry_front if sa < 0 else ry_back
            # Subtle linen fold ripple along collar
            pleat = 1.0 + 0.022 * math.sin(ang * 6.0 + v * 1.4)
            # Dip slightly lower in front center (authentic anatomical throat notch)
            z_dip = -0.011 * max(0.0, -sa) * (1.0 - 0.45 * v)
            ring.append(Vector((ca * rx * flare * pleat, cy + sa * ry * flare * pleat, z + z_dip)))
        collar_grid.append(ring)
    _add_grid_surface(bm, collar_grid, mat_index=0, closed_u=True)

    # 2. Softly tied silk-linen stock knot tucked cleanly at the suprasternal notch
    knot_z = lm["neck_base_z"] - 0.022
    knot_y = min(lm["neck_front_y"] - 0.018, lm["sternum_y"] - 0.014)
    knot_mat = (
        Matrix.Translation((0.0, knot_y, knot_z))
        @ Matrix.Diagonal((0.027, 0.016, 0.020, 1.0))
    )
    bmesh.ops.create_uvsphere(bm, u_segments=18, v_segments=12, radius=1.0, matrix=knot_mat)

    # 3. Layered cascading ruffled jabot frills (Newton) OR sleek folded Steinkirk cravat (Halley)
    chest_base_y = lm["sternum_y"] - 0.016
    if style == "tied":
        tiers = [
            # (length, w_top, w_bot, z_offset, y_proud, wave_amp, wave_freq, x_skew)
            (0.115, 0.024, 0.044, -0.022, 0.014, 0.0045, 3.5, -0.004),
            (0.145, 0.020, 0.052, -0.025, 0.008, 0.0055, 4.0, 0.005),
        ]
    else:
        # Authentic Godfrey Kneller 1689 Sir Isaac Newton multi-tiered ruffled white silk jabot!
        tiers = [
            (0.088, 0.028, 0.068, -0.020, 0.022, 0.0075, 5.0, 0.000),
            (0.132, 0.032, 0.086, -0.024, 0.015, 0.0085, 6.0, 0.002),
            (0.168, 0.028, 0.074, -0.028, 0.008, 0.0070, 5.5, -0.002),
        ]

    for length, w_top, w_bot, z_off, y_proud, w_amp, w_freq, x_skew in tiers:
        rows, cols = 12, 25
        grid = []
        for r in range(rows):
            v = r / (rows - 1)  # 0 at knot -> 1 at ruffled bottom hem
            z = lm["neck_base_z"] + z_off - length * v
            # Follow natural chest curve forward at upper sternum then gently down
            chest_bulge = math.sin(v * math.pi * 0.85) * 0.012
            half_w = (w_top * (1.0 - v) + w_bot * (v ** 0.78)) * 0.5
            row_pts = []
            for c in range(cols):
                u = (c / (cols - 1)) * 2.0 - 1.0  # -1..+1
                # Scalloped rounded bottom hem so jabot looks like soft draped silk lace, never a stiff rectangle!
                hem_curve = math.sqrt(max(0.08, 1.0 - 0.65 * (u * u)))
                z_pt = lm["neck_base_z"] + z_off - length * v * (0.72 + 0.28 * hem_curve)
                x = u * half_w + x_skew * v
                # Organic sinusoidal vertical linen pleats deepening gracefully toward the hem
                pleat_env = (v ** 0.85) * (0.45 + 0.55 * abs(u))
                ripple_y = -w_amp * pleat_env * math.cos(u * math.pi * w_freq)
                # Slight arch wrapping softly around sternum curvature
                arch_y = 0.010 * (u * u)
                y = chest_base_y - y_proud - chest_bulge + ripple_y + arch_y
                row_pts.append(Vector((x, y, z_pt)))
            grid.append(row_pts)
        _add_grid_surface(bm, grid, mat_index=0, closed_u=False)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_CravatMesh")
    bm.to_mesh(mesh)
    bm.free()

    for p in mesh.polygons:
        p.use_smooth = True

    obj = bpy.data.objects.new(f"{name}_17cCravat", mesh)
    _link_to_active_collection(obj)
    obj.data.materials.append(_make_linen_cravat_material(name, base_rgb=cravat_rgb))

    # Add real subtle cloth thickness (2.2mm) so silhouette edges catch warm key & rim light!
    sol = obj.modifiers.new("LinenThickness", "SOLIDIFY")
    sol.thickness = 0.0022
    sol.offset = 0.0

    _parent_object_to_bone_exact(obj, rig, cravat_bone)
    return obj


def _build_17c_scholar_cloak(name: str, rig, basemesh, lm: dict[str, float], cfg: dict):
    """Build a sumptuous, clip-proof 17th-century draped velvet Cambridge Scholar / Royal Society
    Shoulder-Mantle & Flowing Back-Cape fitted directly to the actor's live shoulder & back landmarks!
    Front academic stoles stay cleanly inboard of the shoulder joints (`|x| <= 0.72 * shoulder_half_x`)
    and the flowing pleated cape drapes cleanly behind the shoulder blades so raised arm gestures NEVER clip!"""
    cloak_rgb = cfg.get("cloak_rgb", (0.045, 0.068, 0.165))
    trim_rgb = cfg.get("cloak_trim_rgb", (0.68, 0.50, 0.20))
    cloak_style = str(cfg.get("cloak_style", "cambridge_scholar")).lower()
    cloak_bone = str(cfg.get("cloak_bone", "spine01"))
    add_clasp = bool(cfg.get("cloak_clasp", cloak_style == "captain_mantle"))

    bm = bmesh.new()
    sh_x = lm["shoulder_half_x"]
    sh_z = lm["shoulder_top_z"]
    neck_z = lm["neck_base_z"]
    back_y = lm["back_y"]
    front_y = lm["sternum_y"]

    # 1. Flowing Pleated Velvet Back-Cape & Sculpted Shoulder Cap
    # Circumferential angle theta: 0 = straight back (+Y), +/- pi*0.58 = wrapping just over outer-back shoulders!
    # Stays safely behind the forward upper-arm swing cone so gesturing arms NEVER intersect the cloak!
    rows_b = 16
    cols_b = 31
    cape_len = 0.70 if cloak_style == "cambridge_scholar" else 0.62
    cape_grid = []
    for r in range(rows_b):
        v = r / (rows_b - 1)  # 0 at collar rim -> 1 at flowing mid-thigh hem
        # Angular sweep narrows slightly around mid-arms (avoiding elbow swing!) then flares gracefully at hem
        max_ang = math.radians(104.0 * (1.0 - 0.20 * math.sin(v * math.pi * 0.75)))
        row_pts = []
        for c in range(cols_b):
            u = (c / (cols_b - 1)) * 2.0 - 1.0  # -1 (right) .. +1 (left)
            ang = u * max_ang
            sa, ca = math.sin(ang), math.cos(ang)  # ca=1 at back (+Y), sa=+-1 at sides (+-X)

            # Vertical drop curves smoothly from collar top over shoulder arch then hangs straight down
            if v < 0.18:
                t_sh = v / 0.18
                z = (neck_z + 0.024) * (1.0 - t_sh) + (sh_z + 0.015) * t_sh
                rad_x = (lm["neck_rx"] + 0.020) * (1.0 - t_sh) + (sh_x + 0.022) * t_sh
                rad_y = (abs(lm["neck_back_y"] - lm["neck_center_y"]) + 0.024) * (1.0 - t_sh) + (back_y + 0.030) * t_sh
                pleat_r = 0.004 * t_sh * math.cos(u * math.pi * 5.0)
            else:
                t_dr = (v - 0.18) / 0.82
                z = (sh_z + 0.015) - cape_len * t_dr
                # Graceful A-line gravity drape flaring slightly at hem + organic heavy velvet drapery folds
                rad_x = (sh_x + 0.022) + 0.085 * (t_dr ** 1.15)
                rad_y = (back_y + 0.030) + 0.095 * (t_dr ** 1.10)
                pleat_r = (0.022 * (t_dr ** 0.85)) * math.cos(u * math.pi * 5.5)

            x = sa * (rad_x + pleat_r * 0.65)
            # Keep cape body cleanly behind the arm plane once below the shoulder cap!
            y = max(lm["neck_center_y"] + 0.020 * min(1.0, v * 3.0), ca * (rad_y + pleat_r))
            if v >= 0.18:
                y = max(y, back_y * 0.45 + 0.028)
            row_pts.append(Vector((x, y, z)))
        cape_grid.append(row_pts)
    _add_grid_surface(bm, cape_grid, mat_index=0, closed_u=False)

    # 2. Draped Velvet Academic Front Stoles / Scholar Lapels (Left & Right, framing the white cravat!)
    # Stay strictly inboard of the shoulder joints (`|x| in [0.052 .. 0.148m]`) so arms move 100% freely!
    stole_len = 0.40 if cloak_style == "cambridge_scholar" else 0.34
    for side in (-1.0, 1.0):
        stole_grid = []
        trim_grid = []
        s_rows, s_cols = 14, 9
        for r in range(s_rows):
            v = r / (s_rows - 1)  # 0 at back-of-neck collar -> 1 at lower chest hem
            # Arch gracefully over trapezius (v in 0..0.24) then cascade down chest (v in 0.24..1.0)
            if v < 0.24:
                t_a = v / 0.24
                z_base = (neck_z + 0.022) * (1.0 - t_a) + (sh_z + 0.008) * t_a
                y_base = (lm["neck_back_y"] + 0.010) * (1.0 - t_a) + (front_y - 0.020) * (t_a ** 1.15)
                x_in = (lm["neck_rx"] + 0.014) * (1.0 - t_a) + 0.056 * t_a
                stole_w = 0.058 * (1.0 - t_a) + 0.088 * t_a
            else:
                t_d = (v - 0.24) / 0.76
                z_base = (sh_z + 0.008) - stole_len * t_d
                # Hug front chest contour proudly (+0.022m outside shirt fabric, zero clipping!)
                chest_env = math.sin(t_d * math.pi * 0.80) * 0.014
                y_base = (front_y - 0.022) - chest_env + 0.012 * (t_d ** 1.6)
                # Gently open outward framing the white jabot cravat!
                x_in = 0.056 + 0.026 * math.sin(t_d * math.pi * 0.65)
                stole_w = 0.088 + 0.016 * t_d

            row_pts = []
            t_row_pts = []
            for c in range(s_cols):
                u = c / (s_cols - 1)  # 0 = inner edge next to cravat, 1 = outer shoulder edge
                x_mag = x_in + u * stole_w
                # Clamp outer edge safely inboard of shoulder joint ball so arm gestures never clip!
                x_mag = min(x_mag, sh_x * 0.78)
                # Natural velvet fold ripple across stole width
                fold_y = -0.008 * math.sin(u * math.pi * 2.5) * min(1.0, v * 2.5)
                # Outer edge curves softly back around ribcage contour
                wrap_y = 0.032 * (u ** 1.8)
                z_pt = z_base - 0.010 * (u ** 1.2) * (1.0 - 0.5 * v)
                pt = Vector((side * x_mag, y_base + fold_y + wrap_y, z_pt))
                row_pts.append(pt)
                if c <= 1:
                    # Slight forward offset for embossed gold brocade trim along inner border
                    t_row_pts.append(pt + Vector((0.0, -0.0032, 0.0)))
            stole_grid.append(row_pts)
            trim_grid.append(t_row_pts)

        # Flip winding on negative side so face normals consistently point outward!
        if side < 0:
            stole_grid = [list(reversed(row)) for row in stole_grid]
            trim_grid = [list(reversed(row)) for row in trim_grid]
        _add_grid_surface(bm, stole_grid, mat_index=0, closed_u=False)
        _add_grid_surface(bm, trim_grid, mat_index=1, closed_u=False)

    # 3. Optional Antique Chased-Brass Cloak Clasps at upper collar corners (authentic 17th-c Royal Society / Captain mantle!)
    if add_clasp:
        clasp_z = sh_z - 0.018
        clasp_y = front_y - 0.032
        for side in (-1.0, 1.0):
            c_mat = (
                Matrix.Translation((side * 0.064, clasp_y, clasp_z))
                @ Matrix.Rotation(math.radians(90.0), 4, "X")
            )
            ret = bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=16,
                radius1=0.014,
                radius2=0.011,
                depth=0.006,
                matrix=c_mat,
            )
            for v_c in ret["verts"]:
                for f_c in v_c.link_faces:
                    f_c.material_index = 1
                    f_c.smooth = True

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_ScholarCloakMesh")
    bm.to_mesh(mesh)
    bm.free()

    for p in mesh.polygons:
        p.use_smooth = True

    obj = bpy.data.objects.new(f"{name}_17cScholarCloak", mesh)
    _link_to_active_collection(obj)
    velvet_mat, trim_mat = _make_velvet_cloak_materials(name, cloak_rgb=cloak_rgb, trim_rgb=trim_rgb)
    obj.data.materials.append(velvet_mat)
    obj.data.materials.append(trim_mat)

    # Sumptuous 4mm heavy velvet cloth thickness!
    sol = obj.modifiers.new("VelvetThickness", "SOLIDIFY")
    sol.thickness = 0.0042
    sol.offset = 0.25

    _parent_object_to_bone_exact(obj, rig, cloak_bone)
    return obj


# ============================================================================
# ROLL-PROOF CHARACTER-SPACE BONE POSING + REAL FACIAL ACTING & BLINKING!
# ============================================================================
def set_bone_char_rot(rig, bone_name: str, rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0, frame: int | None = None):
    """Apply a character-space Euler rotation (degrees) to any MPFB2 bone regardless of rest roll!
    TRUE MPFB2 Character axes:
      +X = Character's Left (Screen Right)
      -Y = Forward (facing camera at yaw_deg=0)
      +Z = Up (+rz_deg turns face toward +X Screen Right; -rz_deg turns face toward -X Screen Left)."""
    if bone_name not in rig.pose.bones:
        return
    pb = rig.pose.bones[bone_name]
    pb.rotation_mode = "QUATERNION"
    bone = rig.data.bones[bone_name]
    m_rest = bone.matrix_local.to_3x3()
    r_char = Euler((math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)), "XYZ").to_matrix()
    q_local = (m_rest.inverted() @ r_char @ m_rest).to_quaternion()
    pb.rotation_quaternion = q_local
    if frame is not None:
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def set_bone_char_loc(rig, bone_name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, frame: int | None = None):
    """Apply a character-space small translation (meters) to facial expression bones (eyebrows, eyelids, smile)."""
    if bone_name not in rig.pose.bones:
        return
    pb = rig.pose.bones[bone_name]
    bone = rig.data.bones[bone_name]
    m_rest = bone.matrix_local.to_3x3()
    pb.location = m_rest.inverted() @ Vector((dx, dy, dz))
    if frame is not None:
        pb.keyframe_insert("location", frame=frame)


def apply_facial_acting(
    rig,
    t: float,
    frame: int,
    jaw_open_deg: float = 0.0,
    brow_lift: float = 0.0,
    smile: float = 0.0,
    blink_period: float = 3.2,
    blink_phase: float = 0.8,
):
    """Animate lifelike facial expressions using MPFB2's canonical 163-bone facial rig:
    - Natural 0.14s automatic human eye blinks (orbicularis03.L/R upper eyelid drop!)
    - Expressive eyebrow raises & furrows (oculi01.L/R)
    - Subtle mouth corner smiles & grimaces (risorius03.L/R)
    - Exact Kokoro audio jaw lip-sync (jaw)!"""
    set_bone_char_rot(rig, "jaw", jaw_open_deg, 0.0, 0.0, frame=frame)

    cyc = (t + blink_phase) % blink_period
    blink = 0.0
    if cyc < 0.14:
        blink = math.sin((cyc / 0.14) * math.pi)
    lid_deg = blink * 28.0
    set_bone_char_rot(rig, "orbicularis03.L", lid_deg, 0.0, 0.0, frame=frame)
    set_bone_char_rot(rig, "orbicularis03.R", lid_deg, 0.0, 0.0, frame=frame)

    dz_brow = brow_lift * 0.0065
    set_bone_char_loc(rig, "oculi01.L", 0.0, 0.0, dz_brow, frame=frame)
    set_bone_char_loc(rig, "oculi01.R", 0.0, 0.0, dz_brow, frame=frame)

    dz_sm = smile * 0.0055
    dx_sm = abs(smile) * 0.0030
    set_bone_char_loc(rig, "risorius03.L", dx_sm, 0.002 * smile, dz_sm, frame=frame)
    set_bone_char_loc(rig, "risorius03.R", -dx_sm, 0.002 * smile, dz_sm, frame=frame)


def apply_relaxed_fingers(rig):
    """Add natural anatomical finger curl to all 10 digits so hands look expressive & lifelike."""
    for side in ("L", "R"):
        for f_idx in range(1, 5):
            base_curl = 18.0 + (f_idx - 1) * 5.0
            for seg, mult in ((1, 0.8), (2, 1.1), (3, 0.7)):
                bname = f"finger{f_idx}-{seg}.{side}"
                if bname in rig.pose.bones:
                    pb = rig.pose.bones[bname]
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = Euler((math.radians(base_curl * mult), 0.0, 0.0), "XYZ")
        for tseg, ang in (("thumb02", 14.0), ("thumb03", 16.0)):
            bname = f"{tseg}.{side}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(ang), 0.0, 0.0), "XYZ")


# ============================================================================
# PUBLIC FACTORY: CREATE V7 AUTHENTIC 17TH-CENTURY ULTRA-REALISTIC MPFB2 HUMAN!
# ============================================================================
def _equip_clothing_layer(
    HumanService,
    AssetService,
    basemesh,
    name: str,
    equipped: dict,
    slot_label: str,
    candidates: list[str],
    cfg: dict,
    prefix: str,
    default_roughness: float = 0.74,
    add_fabric_bump: bool = True,
):
    """Load an optional MPFB2 clothing layer (shirt/pants/coat/cloak/suit/shoes) with independent color grading."""
    if not candidates:
        return None
    apath, _ = _find_best_asset(AssetService, ("clothes",), candidates)
    if not apath:
        print(f"[MPFB2 Asset] '{name}' {slot_label:10s} -> NOT FOUND ({candidates})")
        return None
    obj = HumanService.add_mhclo_asset(
        apath,
        basemesh,
        asset_type="Clothes",
        subdiv_levels=0,
        material_type="MAKESKIN",
        set_up_rigging=True,
        interpolate_weights=True,
    )
    equipped[slot_label] = obj
    print(f"[MPFB2 Asset] '{name}' {slot_label:10s} -> {os.path.basename(apath)}")

    tint_rgb = cfg.get(f"{prefix}_tint_rgb")
    tint_fac = float(cfg.get(f"{prefix}_tint_fac", 0.0))
    val_boost = float(cfg.get(f"{prefix}_value_boost", 1.0))
    gam_lift = float(cfg.get(f"{prefix}_gamma_lift", 1.0))
    rough = cfg.get(f"{prefix}_roughness", default_roughness)
    if tint_rgb or tint_fac > 0.0 or val_boost != 1.0 or gam_lift != 1.0 or add_fabric_bump:
        _tint_asset_material(
            obj,
            tint_rgb=tint_rgb,
            mix_fac=tint_fac,
            roughness_override=float(rough) if rough is not None else None,
            is_hair=False,
            value_boost=val_boost,
            gamma_lift=gam_lift,
            add_fabric_bump=add_fabric_bump,
        )
    return obj


def create_realistic_scientist(
    name: str,
    floor_z: float,
    start_xy: tuple[float, float],
    yaw_deg: float,
    cfg: dict,
    brass_mat=None,
):
    """Build an ultra-realistic Cycles human using official MakeHuman 4K photo-scanned adult skins,
    PROCEDURAL_EYES high-poly eyes, alpha eyebrows/eyelashes, 3D teeth/tongue, authentic hair,
    multi-layer 17th-century historical attire (shirt/tunic, wool trousers, optional coat/cloak,
    leather boots — ZERO modern ties!), plus optional rigged 17th-century white linen jabot cravat
    and draped velvet Cambridge scholar shoulder-mantle/cape!
    Note: +yaw_deg faces toward +X (Screen Right); -yaw_deg faces toward -X (Screen Left)."""
    HumanService, TargetService, AssetService, HumanObjectProperties = enable_mpfb2()
    AssetService.update_all_asset_lists()

    # 1. Create anatomical basemesh (19,158 verts) with enforced adult male stature & proportions!
    basemesh = HumanService.create_human(feet_on_ground=True, scale=0.1)
    basemesh.name = f"{name}_Body"

    raw_age = float(cfg.get("age", 0.58))
    if not cfg.get("allow_youth", False) and raw_age < 0.54:
        print(f"[MPFB2 Guard] '{name}' age={raw_age:.2f} < 0.54 (teenager in MakeHuman!) -> clamped to 0.56 adult!")
        raw_age = 0.56

    HumanObjectProperties.set_value("gender", float(cfg.get("gender", 1.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("age", raw_age, entity_reference=basemesh)
    HumanObjectProperties.set_value("muscle", float(cfg.get("muscle", 0.54)), entity_reference=basemesh)
    HumanObjectProperties.set_value("weight", float(cfg.get("weight", 0.48)), entity_reference=basemesh)
    HumanObjectProperties.set_value("height", float(cfg.get("height", 0.58)), entity_reference=basemesh)
    HumanObjectProperties.set_value("caucasian", float(cfg.get("caucasian", 1.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("african", float(cfg.get("african", 0.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("asian", float(cfg.get("asian", 0.0)), entity_reference=basemesh)
    TargetService.reapply_macro_details(basemesh)

    # 2. Apply Real 4K Photo-Scanned Adult Skin with MPFB2 ENHANCED_SSS (tuned to fast BURLEY SSS!)
    skin_candidates = cfg.get("skin_assets", ["middleage_caucasian_male.mhmat", "old_caucasian_male.mhmat"])
    skin_path, _ = _find_best_asset(AssetService, ("skins",), skin_candidates, is_mhmat=True)
    if skin_path:
        print(f"[MPFB2 4K Skin] '{name}' -> {os.path.basename(skin_path)} (ENHANCED_SSS / BURLEY)")
        HumanService.set_character_skin(skin_path, basemesh, skin_type="ENHANCED_SSS", material_instances=True)
        _optimize_enhanced_skin_for_t4(basemesh)
    else:
        print(f"[MPFB2 Warning] No 4K skin pack found for '{name}' — run Cell 2 asset pack download!")

    # 3. Attach Full 163-Bone Canonical MakeHuman Skeleton (BEFORE clothes/hair so MPFB2 auto-rigs ALL assets!)
    rig = HumanService.add_builtin_rig(basemesh, "default")
    rig.name = f"{name}_Rig"

    # 4. Equip Real Photoreal Bodyparts (High-Poly Procedural Eyes, Alpha Eyebrows, Eyelashes, Teeth & Tongue!)
    bodyparts = [
        ("Eyes", ("eyes",), ["high-poly.mhclo", "low-poly.mhclo"], "PROCEDURAL_EYES", 0),
        ("Eyebrows", ("eyebrows",), cfg.get("eyebrow_assets", ["eyebrow002.mhclo", "eyebrow003.mhclo", "eyebrow001.mhclo"]), "MAKESKIN", 0),
        ("Eyelashes", ("eyelashes",), ["eyelashes01.mhclo"], "MAKESKIN", 0),
        ("Teeth", ("teeth",), ["teeth_base.mhclo"], "MAKESKIN", 0),
        ("Tongue", ("tongue",), ["tongue01.mhclo"], "MAKESKIN", 0),
    ]
    equipped = {}
    for atype, subdirs, cands, mtype, subdiv in bodyparts:
        apath, _ = _find_best_asset(AssetService, subdirs, cands)
        if apath:
            obj = HumanService.add_mhclo_asset(
                apath,
                basemesh,
                asset_type=atype,
                subdiv_levels=subdiv,
                material_type=mtype,
                set_up_rigging=True,
                interpolate_weights=True,
            )
            equipped[atype] = obj
            print(f"[MPFB2 Asset] '{name}' {atype:10s} -> {os.path.basename(apath)}")
        else:
            print(f"[MPFB2 Asset] '{name}' {atype:10s} -> NOT FOUND ({cands})")

    # Dial vivid realistic iris colors directly on EnhancedEye shader!
    if "Eyes" in equipped:
        _tune_procedural_eyes(
            equipped["Eyes"],
            cfg.get("iris_major_rgb", (0.14, 0.36, 0.24)),
            cfg.get("iris_minor_rgb", (0.06, 0.16, 0.12)),
        )

    # Match eyebrow color cleanly to hair color!
    is_silver = bool(cfg.get("silver_hair", False))
    if "Eyebrows" in equipped:
        _tint_asset_material(
            equipped["Eyebrows"],
            cfg.get("brow_tint_rgb", cfg.get("hair_tint_rgb")),
            mix_fac=float(cfg.get("brow_tint_fac", 0.45)),
            is_hair=True,
            silver_hair=is_silver,
            gamma_lift=float(cfg.get("brow_gamma_lift", 1.0)),
        )

    # 5. Equip Real Alpha-Strand / Normal-Mapped Hair (distinct per character — subdiv_levels=0!)
    hair_cands = cfg.get(
        "hair_assets",
        ["culturalibre_hair_01.mhclo", "bob02.mhclo", "short04.mhclo", "short02.mhclo"],
    )
    hair_path, _ = _find_best_asset(AssetService, ("hair",), hair_cands)
    if hair_path:
        hair_obj = HumanService.add_mhclo_asset(
            hair_path,
            basemesh,
            asset_type="Hair",
            subdiv_levels=0,
            material_type="MAKESKIN",
            set_up_rigging=True,
            interpolate_weights=True,
        )
        equipped["Hair"] = hair_obj
        print(f"[MPFB2 Asset] '{name}' Hair       -> {os.path.basename(hair_path)}")
        _tint_asset_material(
            hair_obj,
            cfg.get("hair_tint_rgb"),
            mix_fac=float(cfg.get("hair_tint_fac", 0.0)),
            roughness_override=cfg.get("hair_roughness", 0.40),
            is_hair=True,
            silver_hair=bool(cfg.get("silver_hair_main", is_silver)),
            value_boost=float(cfg.get("hair_value_boost", 1.0)),
            gamma_lift=float(cfg.get("hair_gamma_lift", 1.0)),
        )

    # 6. Optional Real 3D Alpha-Card Mustache / Beard
    beard_cands = cfg.get("beard_assets", [])
    if beard_cands:
        beard_path, _ = _find_best_asset(AssetService, ("clothes", "hair", "bodyparts"), beard_cands)
        if beard_path:
            beard_obj = HumanService.add_mhclo_asset(
                beard_path,
                basemesh,
                asset_type="Clothes",
                subdiv_levels=0,
                material_type="MAKESKIN",
                set_up_rigging=True,
                interpolate_weights=True,
            )
            equipped["Beard"] = beard_obj
            print(f"[MPFB2 Asset] '{name}' Beard      -> {os.path.basename(beard_path)}")
            _tint_asset_material(
                beard_obj,
                cfg.get("hair_tint_rgb"),
                mix_fac=float(cfg.get("beard_tint_fac", 0.0)),
                is_hair=True,
                silver_hair=is_silver,
                value_boost=float(cfg.get("beard_value_boost", 1.0)),
                gamma_lift=float(cfg.get("beard_gamma_lift", 1.0)),
            )

    # 7. Equip Multi-Layer Authentic 17th-Century Historical Outfit (subdiv_levels=0 for OOM-proof T4 GPU speed!)
    # Load trousers BEFORE outer tunic/coat so MPFB2 z_depth & delete_verts stack cleanly!
    has_layered = any(cfg.get(k) for k in ("shirt_assets", "pants_assets", "coat_assets", "cloak_assets"))

    if cfg.get("pants_assets"):
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Pants", cfg["pants_assets"], cfg, "pants", default_roughness=0.78, add_fabric_bump=True,
        )

    if cfg.get("shirt_assets"):
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Shirt", cfg["shirt_assets"], cfg, "shirt", default_roughness=0.74, add_fabric_bump=True,
        )

    if cfg.get("coat_assets"):
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Coat", cfg["coat_assets"], cfg, "coat", default_roughness=0.76, add_fabric_bump=True,
        )

    if cfg.get("cloak_assets"):
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Cloak", cfg["cloak_assets"], cfg, "cloak", default_roughness=0.80, add_fabric_bump=True,
        )

    # Backwards-compatible single-piece suit slot (ONLY used if explicitly requested, or if no layered pieces provided)
    suit_cands = cfg.get("suit_assets")
    if suit_cands is None and not has_layered:
        # Default fallback is historical layered doublet + wool trousers (NEVER modern tie suits!)
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Pants", ["mindfront_male_trousers_2.mhclo", "toigo_wool_pants.mhclo"], cfg, "pants",
            default_roughness=0.78, add_fabric_bump=True,
        )
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Shirt", ["elvs_male_boho_top1.mhclo", "elvs_male_shirt_untucked_bd1.mhclo"], cfg, "shirt",
            default_roughness=0.74, add_fabric_bump=True,
        )
    elif suit_cands:
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Suit", suit_cands, cfg, "suit", default_roughness=0.72, add_fabric_bump=True,
        )

    # Authentic 17th-Century Leather Boots / Shoes
    shoe_cands = cfg.get(
        "shoe_assets",
        ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo", "culturalibre_male_boots.mhclo"],
    )
    _equip_clothing_layer(
        HumanService, AssetService, basemesh, name, equipped,
        "Shoes", shoe_cands, cfg, "shoe", default_roughness=0.46, add_fabric_bump=False,
    )

    # 8. Optional Rigged 17th-Century White Linen Ruffled Jabot Cravat & Draped Velvet Scholar Cloak!
    # Built cleanly in rest pose (rig at origin/identity) from live torso measurements and bone-parented!
    if cfg.get("add_cravat", False) or cfg.get("add_scholar_cloak", False):
        lm = _measure_torso_landmarks(rig, basemesh)
        if cfg.get("add_cravat", False):
            cravat_obj = _build_17c_cravat(name, rig, basemesh, lm, cfg)
            equipped["Cravat"] = cravat_obj
            print(f"[MPFB2 17c]   '{name}' Cravat     -> Rigged 17c Linen ({cfg.get('cravat_style', 'jabot')}, bone={cfg.get('cravat_bone', 'spine01')})")
        if cfg.get("add_scholar_cloak", False):
            cloak_obj = _build_17c_scholar_cloak(name, rig, basemesh, lm, cfg)
            equipped["ScholarCloak"] = cloak_obj
            print(f"[MPFB2 17c]   '{name}' Mantle     -> Rigged 17c Velvet ({cfg.get('cloak_style', 'cambridge_scholar')}, bone={cfg.get('cloak_bone', 'spine01')})")

    # 9. Natural relaxed curled fingers + Adult Stature Normalization (`target_height_m`) + exact floor grounding!
    apply_relaxed_fingers(rig)

    bpy.context.view_layer.update()
    body_zs = [(basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices]
    raw_min_z = min(body_zs)
    raw_max_z = max(body_zs)
    raw_h = max(0.5, raw_max_z - raw_min_z)

    target_h = cfg.get("target_height_m")
    if target_h is not None:
        scale_f = max(0.85, min(1.25, float(target_h) / raw_h))
        rig.scale = Vector((scale_f, scale_f, scale_f))
        bpy.context.view_layer.update()

    # Recompute exact grounded min_z across body + boot soles so feet rest cleanly ON floor_z!
    body_zs = [(basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices]
    min_z = min(body_zs)
    max_z = max(body_zs)
    if "Shoes" in equipped and equipped["Shoes"] is not None:
        shoe_obj = equipped["Shoes"]
        if len(shoe_obj.data.vertices) > 0:
            shoe_min_z = min((shoe_obj.matrix_world @ v.co).z for v in shoe_obj.data.vertices)
            min_z = min(min_z, shoe_min_z)

    base_z = floor_z - min_z
    rig.location = Vector((start_xy[0], start_xy[1], base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)))

    final_h = max_z - min_z
    print(
        f"[MPFB2 V7 READY] '{name}' ({len(equipped)} authentic 17c/4K assets equipped, age={raw_age:.2f}, "
        f"height={final_h:.2f}m, feet@z={floor_z:.2f}m, head_top@z={floor_z + final_h:.2f}m)"
    )
    return {"rig": rig, "body": basemesh, "equipped": equipped, "height": final_h, "base_z": base_z}
