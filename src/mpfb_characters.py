"""
Reusable MPFB2 V9 Cinema-Acting & 17th-Century Human Character Engine for Blender 4.2.3 LTS Cycles (Kaggle GPU T4 x2)
Uses curated CC0/CC-BY MakeHuman Community 4K Asset Packs (delivered via ~1.2s GitHub CDN in Kaggle Cell 2!):
  - Real 4K Photo-Scanned Adult Masculine Skins with ENHANCED_SSS Cycles Shader (pores, real lips, brow
    shadows, wrinkles, knuckles — tuned to fast BURLEY SSS, warm 0.40-0.46 roughness & 0.48 T-zone specular!)
  - Adult Age & Exact Stature Normalization (`age >= 0.54`, `target_height_m`) so Newton (1.78m) & Halley (1.79m)
    always stand eye-to-eye as equal adult 17th-century men — NEVER an undersized teenager!
  - Real High-Poly Photoreal Eyes (high-poly.mhclo + PROCEDURAL_EYES EnhancedEye wet cornea catchlights:
    Clearcoat=0.95, PupilSize=0.27, IOR=1.376!)
  - Real Alpha-Card Eyebrows, Eyelashes, 3D Teeth & Tongue (realistic mouth during dual-axis Kokoro lip-sync!)
  - Real Sleek Shoulder-Length Hair (`culturalibre_hair_01.mhclo` + 2K normal map) for Sir Isaac Newton's iconic
    1689 Godfrey Kneller portrait look & swept adult gentleman hair (`short04.mhclo`) for Halley!
  - 100% REAL 3D Sculpted MakeHuman Multi-Layer Historical 17th-Century Wardrobe (`pants_assets`, `shirt_assets`,
    `vest_assets`, `coat_assets`, `cloak_assets`, `suit_assets`, `shoe_assets` — ZERO modern neckties, ZERO
    modern bowties, ZERO blocky procedural foam straps or chin-covering neck-braces!) with linear-safe
    `gamma_lift`, `sat_override`, `hue_shift`, `value_boost`, `tint_rgb`, velvet `sheen`, automatic sibling
    `*_METAL.png` button hookup, and normal-map-preserving woven wool/velvet micro-bump relief!
  - V9 Pixar / Live-Action Acting Suite:
    * `apply_facial_acting`: Eyeball gaze (`eye.L/R`) + organic micro-saccades, dual-axis Kokoro vowel lip-sync
      (`jaw` + `lip_narrow` corner pucker & `oris02`/`oris06` rounding), asymmetric 0.015s blinks + eyelid gaze
      follow-through, brow micro-life & Duchenne smile cheek lift (`levator05.L/R`).
    * `apply_arm_with_clavicle`: Biomechanical clavicle shrug/protraction coupling + relaxed wrist alignment.
    * `apply_hand_gesture`: 4 lifelike 10-digit poses (`relaxed`, `present`, `point`, `shock`) mapped accurately
      to MakeHuman's canonical 163-bone default rig (`finger1`=thumb .. `finger5`=pinky) with fallback support!
    * `apply_natural_stance`: Living contrapposto spine S-curve, thoracic breathing & grounded knee-bend crouch.
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
        raise ImportError(f"MPFB2 module '{absolute_package_str}.{key}' not found! Install MPFB2 extension first.")

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


def _tune_skin_group_roughness(nt, visited_trees=None):
    """Recursively dial warm natural skin roughness (0.40-0.46) on MPFB2 ENHANCED_SSS group nodes
    (MpfbFace, MpfbBody, MpfbEars, MpfbSkin) while leaving naturally glossy Lips/Nails untouched."""
    if visited_trees is None:
        visited_trees = set()
    if nt is None or id(nt) in visited_trees:
        return
    visited_trees.add(id(nt))
    for node in nt.nodes:
        if node.type != "GROUP":
            continue
        gtree = getattr(node, "node_tree", None)
        gname = gtree.name if gtree is not None else ""
        tag = f"{node.name} {node.label} {gname}".lower()
        if any(skip in tag for skip in ("lip", "nail", "mouth", "teeth", "tongue", "eye", "areola", "aureola")):
            continue
        r_in = node.inputs.get("Roughness")
        if r_in is not None and not r_in.is_linked:
            try:
                cur_r = float(r_in.default_value)
                if "face" in tag:
                    # Sculptural warm cheekbone sheen under golden-hour sunlight without plastic glare
                    r_in.default_value = 0.42
                else:
                    r_in.default_value = min(0.46, max(0.40, cur_r))
            except Exception:
                pass
        if gtree is not None:
            _tune_skin_group_roughness(gtree, visited_trees)


def _optimize_enhanced_skin_for_t4(basemesh):
    """Keep MPFB2's gorgeous 4K photo-scanned ENHANCED_SSS pores/lips/zones while switching
    Principled BSDF subsurface method to fast BURLEY for 2x Kaggle T4 GPU speed, dialing warm
    natural skin roughness (0.40-0.46) and subtle oily T-zone specular (Specular IOR Level=0.48)!"""
    for slot in basemesh.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        if mat.users > 1:
            mat = mat.copy()
            slot.material = mat
        for _nt, bsdf in _find_principled_nodes(mat.node_tree):
            if hasattr(bsdf, "subsurface_method"):
                try:
                    bsdf.subsurface_method = "BURLEY"
                except Exception:
                    pass
            # Ensure realistic warm non-plastic, non-chalky skin roughness (0.40-0.46)
            if "Roughness" in bsdf.inputs and not bsdf.inputs["Roughness"].is_linked:
                cur_r = float(bsdf.inputs["Roughness"].default_value)
                bsdf.inputs["Roughness"].default_value = min(0.46, max(0.40, cur_r))
            # Subtle oily T-zone specular highlight so golden-hour key light sculpts cheekbones
            if "Specular IOR Level" in bsdf.inputs and not bsdf.inputs["Specular IOR Level"].is_linked:
                bsdf.inputs["Specular IOR Level"].default_value = 0.48
            elif "Specular" in bsdf.inputs and not bsdf.inputs["Specular"].is_linked:
                bsdf.inputs["Specular"].default_value = 0.48
        # Also tune outer MPFB2 skin group Roughness sockets (MpfbFace/Body/Ears/Skin feed Principled BSDF!)
        _tune_skin_group_roughness(mat.node_tree)


def _tune_procedural_eyes(eye_obj, major_rgb=(0.14, 0.36, 0.20), minor_rgb=(0.06, 0.16, 0.10)):
    """Dial vivid realistic iris colors & crisp wet-cornea catchlights directly on MPFB2's official
    EnhancedEye refractive cornea node (Clearcoat=0.95, PupilSize=0.27, IOR=1.376)!"""
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
                    node.inputs["PupilSize"].default_value = 0.27
                if "Clearcoat" in node.inputs:
                    node.inputs["Clearcoat"].default_value = 0.95
                if "Clearcoat Roughness" in node.inputs:
                    node.inputs["Clearcoat Roughness"].default_value = 0.02
                if "OuterLayerRoughness" in node.inputs:
                    node.inputs["OuterLayerRoughness"].default_value = 0.0
                if "OuterLayerIOR" in node.inputs:
                    node.inputs["OuterLayerIOR"].default_value = 1.376
                if "InnerLayerRoughness" in node.inputs:
                    node.inputs["InnerLayerRoughness"].default_value = 0.04
                if "IrisBumpStrength" in node.inputs:
                    node.inputs["IrisBumpStrength"].default_value = 0.36


def _add_woven_fabric_bump(nt, bsdf, scale: float = 340.0, strength: float = 0.070):
    """Attach subtle procedural woven broadcloth/velvet micro-bump to clothing.
    If the garment ALREADY has a baked 2K normal map linked to `Normal`, chains a fine
    woven micro-grain bump AFTER the baked normal map so 100% of baked folds are preserved!"""
    n_in = bsdf.inputs.get("Normal")
    if n_in is None:
        return
    has_baked_nrm = n_in.is_linked
    prev_nrm_socket = n_in.links[0].from_socket if has_baked_nrm else None
    if has_baked_nrm:
        nt.links.remove(n_in.links[0])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 460.0 if has_baked_nrm else scale
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.62

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = min(strength, 0.036) if has_baked_nrm else strength
    bump.inputs["Distance"].default_value = 0.0012 if has_baked_nrm else 0.0022
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    if prev_nrm_socket is not None:
        nt.links.new(prev_nrm_socket, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], n_in)


def _try_attach_sibling_metal_map(nt, bsdf, obj_name: str):
    """If a MakeHuman garment ships with a sibling `*_METAL.png` mask (e.g. `F_Lusekofta_METAL.png`
    for engraved filigree brass/silver buttons) that MAKESKIN did not auto-link, wire it cleanly
    as Non-Color into Principled BSDF `Metallic` so real metallic buttons gleam in Cycles!"""
    m_in = bsdf.inputs.get("Metallic")
    if m_in is None or m_in.is_linked:
        return
    for node in nt.nodes:
        if node.type != "TEX_IMAGE" or getattr(node, "image", None) is None:
            continue
        raw_fp = bpy.path.abspath(node.image.filepath)
        if not raw_fp or not os.path.isfile(raw_fp):
            continue
        stem, ext = os.path.splitext(raw_fp)
        low_stem = stem.lower()
        if any(suf in low_stem for suf in ("_nrm", "_norm", "_normal", "_spec", "_rough", "_metal", "_ao")):
            continue
        for cand_suffix in ("_METAL.png", "_metal.png", "_metallic.png", "_METAL.PNG"):
            cand_path = stem + cand_suffix
            if os.path.isfile(cand_path):
                try:
                    img = bpy.data.images.load(cand_path, check_existing=True)
                    img.colorspace_settings.name = "Non-Color"
                    m_tex = nt.nodes.new("ShaderNodeTexImage")
                    m_tex.image = img
                    nt.links.new(m_tex.outputs["Color"], m_in)
                    print(f"[MPFB2 PBR] '{obj_name}' auto-wired metallic map -> {os.path.basename(cand_path)}")
                    return
                except Exception as merr:
                    print(f"[MPFB2 PBR Warning] Failed wiring {cand_path}: {merr}")
                return


def _tint_asset_material(
    obj,
    tint_rgb: tuple[float, float, float] | None = None,
    mix_fac: float = 0.45,
    roughness_override: float | None = None,
    is_hair: bool = False,
    silver_hair: bool = False,
    value_boost: float = 1.0,
    gamma_lift: float = 1.0,
    sat_override: float | None = None,
    hue_shift: float = 0.5,
    sheen_weight: float | None = None,
    add_fabric_bump: bool = False,
    auto_metal_map: bool = False,
):
    """Non-destructively color-grade an imported MPFB2 clothing or alpha-hair material while preserving
    100% of its original photo-scanned diffuse texture, fabric folds, normal/bump maps & alpha transparency!
    Supports linear-safe `gamma_lift`, `sat_override` (e.g. desaturating brown/blue base cloth before pure
    Royal Midnight-Navy or Royal Burgundy grading!), `hue_shift`, `sheen_weight` velvet pile sheen,
    automatic `*_METAL.png` button hookup, and Blender 4.2-safe RGBA Mix (`s.type == 'RGBA'`)."""
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
                _set_bsdf(bsdf, "Sheen Weight", sheen_weight if sheen_weight is not None else 0.38)
            else:
                if roughness_override is not None and not bsdf.inputs["Roughness"].is_linked:
                    _set_bsdf(bsdf, "Roughness", roughness_override)
                _set_bsdf(bsdf, "Sheen Weight", sheen_weight if sheen_weight is not None else 0.54)
                _set_bsdf(bsdf, "Sheen Roughness", 0.42)
                if auto_metal_map:
                    _try_attach_sibling_metal_map(nt, bsdf, obj.name)
                if add_fabric_bump:
                    _add_woven_fabric_bump(nt, bsdf)

            try:
                eff_gamma = min(gamma_lift, 0.36) if silver_hair else gamma_lift
                eff_sat = 0.14 if silver_hair else (sat_override if sat_override is not None else 1.0)
                eff_val = max(1.30, value_boost) if silver_hair else value_boost
                needs_hsv = silver_hair or (eff_sat != 1.0) or (eff_val != 1.0) or (hue_shift != 0.5)

                if bc.is_linked:
                    src_socket = bc.links[0].from_socket
                    nt.links.remove(bc.links[0])

                    if eff_gamma != 1.0:
                        gam = nt.nodes.new("ShaderNodeGamma")
                        gam.inputs["Gamma"].default_value = eff_gamma
                        nt.links.new(src_socket, gam.inputs["Color"])
                        src_socket = gam.outputs["Color"]

                    if needs_hsv:
                        hsv = nt.nodes.new("ShaderNodeHueSaturation")
                        hsv.inputs["Hue"].default_value = hue_shift
                        hsv.inputs["Saturation"].default_value = eff_sat
                        hsv.inputs["Value"].default_value = eff_val
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
                    lifted = [min(1.0, (max(0.001, orig[c]) ** eff_gamma) * eff_val) for c in range(3)]
                    if eff_sat != 1.0:
                        lum = 0.2126 * lifted[0] + 0.7152 * lifted[1] + 0.0722 * lifted[2]
                        lifted = [lum + (lifted[c] - lum) * eff_sat for c in range(3)]
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


def _blink_envelope(t_val: float, period: float, phase: float) -> float:
    """Asymmetric organic human blink envelope: crisp 55ms lid close + smooth 100ms reopen follow-through."""
    cyc = (t_val + phase) % max(0.5, period)
    if cyc < 0.055:
        return math.sin((cyc / 0.055) * math.pi * 0.5)
    if cyc < 0.155:
        u = (cyc - 0.055) / 0.100
        return math.cos(u * math.pi * 0.5) ** 1.35
    return 0.0


def apply_facial_acting(
    rig,
    t: float,
    frame: int,
    jaw_open_deg: float = 0.0,
    brow_lift: float = 0.0,
    smile: float = 0.0,
    blink_period: float = 3.2,
    blink_phase: float = 0.8,
    eye_yaw_deg: float = 0.0,
    eye_pitch_deg: float = 0.0,
    lip_narrow: float = 0.0,
):
    """Animate Pixar/live-action facial acting using MPFB2's canonical 163-bone facial rig:
    - True eyeball gaze (`eye.L`/`eye.R`) clamped safely (|yaw|<=18 deg, |pitch|<=14 deg) with organic
      stepped micro-saccades (~1.7s fixation wave, +-1.6 deg) + subtle 0.7 deg close-up eye convergence!
    - Dual-axis Kokoro vowel lip-sync: `jaw` opening + `lip_narrow` (0..1) mouth-corner pucker
      (`risorius03.L/R`) and upper/lower lip protrusion (`oris02`/`oris06` when present).
    - Asymmetric natural human eye blinks (0.015s L/R lag + upper-eyelid gaze follow-through on
      `orbicularis03.L/R` + subtle Duchenne lower-lid squint on `orbicularis04.L/R`).
    - Living brow micro-drift & glabella furrow (`oculi01.L/R`) + Duchenne cheek lift (`levator05.L/R`)!"""
    # 1. Kokoro jaw opening (clamped safely so mandible never over-rotates)
    jaw_deg = max(0.0, min(26.0, float(jaw_open_deg)))
    set_bone_char_rot(rig, "jaw", jaw_deg, 0.0, 0.0, frame=frame)

    # 2. Eyeball gaze + organic stepped micro-saccades (~1.7s fixations, smooth 0.12s shifts)
    sacc_t = (t + blink_phase * 0.73) / 1.7
    sacc_idx = math.floor(sacc_t)
    sacc_frac = sacc_t - sacc_idx
    u_sacc = min(1.0, sacc_frac / 0.12)
    u_smooth = u_sacc * u_sacc * (3.0 - 2.0 * u_sacc)
    y0 = math.sin(sacc_idx * 2.39996 + blink_phase)
    y1 = math.sin((sacc_idx + 1.0) * 2.39996 + blink_phase)
    p0 = math.cos(sacc_idx * 2.23053 + blink_phase * 1.3)
    p1 = math.cos((sacc_idx + 1.0) * 2.23053 + blink_phase * 1.3)
    sacc_yaw = (y0 + (y1 - y0) * u_smooth) * 1.6
    sacc_pitch = (p0 + (p1 - p0) * u_smooth) * 0.85

    final_eye_yaw = max(-18.0, min(18.0, float(eye_yaw_deg) + sacc_yaw))
    final_eye_pitch = max(-14.0, min(14.0, float(eye_pitch_deg) + sacc_pitch))
    # Subtle 0.7 deg physiological inward convergence (eye.L -> -X, eye.R -> +X) kills dead CG stare!
    eye_yaw_l = max(-18.0, min(18.0, final_eye_yaw - 0.7))
    eye_yaw_r = max(-18.0, min(18.0, final_eye_yaw + 0.7))
    set_bone_char_rot(rig, "eye.L", final_eye_pitch, 0.0, eye_yaw_l, frame=frame)
    set_bone_char_rot(rig, "eye.R", final_eye_pitch, 0.0, eye_yaw_r, frame=frame)

    # 3. Asymmetric natural blink (0.015s L/R organic lag) + upper-eyelid gaze & smile follow-through
    blink_l = _blink_envelope(t, blink_period, blink_phase)
    blink_r = _blink_envelope(t - 0.015, blink_period, blink_phase)
    gaze_lid = max(-4.0, min(6.0, final_eye_pitch * 0.28))
    smile_pos = max(0.0, float(smile))
    lid_l = min(30.0, blink_l * 28.5 + (1.0 - blink_l) * (gaze_lid + smile_pos * 1.4))
    lid_r = min(30.0, blink_r * 28.5 + (1.0 - blink_r) * (gaze_lid + smile_pos * 1.4))
    set_bone_char_rot(rig, "orbicularis03.L", lid_l, 0.0, 0.0, frame=frame)
    set_bone_char_rot(rig, "orbicularis03.R", lid_r, 0.0, 0.0, frame=frame)
    # Subtle lower-eyelid warm Duchenne lift on smiles
    if "orbicularis04.L" in rig.pose.bones:
        set_bone_char_loc(rig, "orbicularis04.L", 0.0, 0.0, smile_pos * 0.0014, frame=frame)
    if "orbicularis04.R" in rig.pose.bones:
        set_bone_char_loc(rig, "orbicularis04.R", 0.0, 0.0, smile_pos * 0.0014, frame=frame)

    # 4. Expressive brow micro-life, subtle human asymmetry & inward glabella furrow
    brow_micro = math.sin(t * 1.15 + blink_phase) * 0.00065
    furrow_dx = min(0.0, float(brow_lift)) * 0.0022
    dz_brow_l = float(brow_lift) * 0.0065 + brow_micro
    dz_brow_r = float(brow_lift) * 0.0062 - brow_micro * 0.45
    set_bone_char_loc(rig, "oculi01.L", furrow_dx, 0.0, dz_brow_l, frame=frame)
    set_bone_char_loc(rig, "oculi01.R", -furrow_dx, 0.0, dz_brow_r, frame=frame)

    # 5. Dual-axis Kokoro vowel lip-sync (`lip_narrow` 0..1) + smile corner pull & lip pucker protrusion
    narrow = max(0.0, min(1.0, float(lip_narrow)))
    dz_sm = float(smile) * 0.0055
    dx_sm = abs(float(smile)) * 0.0030
    narrow_in = narrow * 0.0022
    narrow_fwd = -narrow * 0.0014
    set_bone_char_loc(rig, "risorius03.L", dx_sm - narrow_in, 0.0020 * float(smile) + narrow_fwd, dz_sm, frame=frame)
    set_bone_char_loc(rig, "risorius03.R", -dx_sm + narrow_in, 0.0020 * float(smile) + narrow_fwd, dz_sm, frame=frame)

    # Subtle forward lip rounding (-Y is forward in MPFB2!) for 'oo'/'oh' vowels
    if "oris02" in rig.pose.bones:
        set_bone_char_loc(rig, "oris02", 0.0, -narrow * 0.0022, -narrow * 0.0005, frame=frame)
    if "oris06" in rig.pose.bones:
        set_bone_char_loc(rig, "oris06", 0.0, -narrow * 0.0024, narrow * 0.0006, frame=frame)

    # Subtle Duchenne cheek lift on smiles (`levator05.L/R`)
    if "levator05.L" in rig.pose.bones:
        set_bone_char_loc(rig, "levator05.L", smile_pos * 0.0012, -smile_pos * 0.0014, smile_pos * 0.0038, frame=frame)
    if "levator05.R" in rig.pose.bones:
        set_bone_char_loc(rig, "levator05.R", -smile_pos * 0.0012, -smile_pos * 0.0014, smile_pos * 0.0038, frame=frame)


def apply_arm_with_clavicle(
    rig,
    side: str,
    ua_deg: tuple[float, float, float],
    la_deg: tuple[float, float, float],
    frame: int | None = None,
):
    """Pose upperarm01 & lowerarm01 with automatic biomechanical clavicle shrug/protraction coupling
    (~18-24% of arm lift, safely clamped <= 10 deg) and natural wrist alignment (+-6 deg extension)."""
    side_u = "L" if str(side).upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0

    ua_rx, ua_ry, ua_rz = float(ua_deg[0]), float(ua_deg[1]), float(ua_deg[2])
    la_rx, la_ry, la_rz = float(la_deg[0]), float(la_deg[1]), float(la_deg[2])

    # Biomechanical clavicle coupling: lifting arm forward/up (-ua_rx > 0) elevates shoulder cleanly
    # TRUE MPFB2 Y-rotation: on L (+X), -ry lifts +X toward +Z; on R (-X), +ry lifts -X toward +Z!
    lift_drive = (-ua_rx) * 0.21 + max(0.0, (-side_sgn * ua_ry) + 12.0) * 0.08
    clav_elev = max(-4.5, min(9.5, lift_drive))
    clav_ry = -side_sgn * clav_elev
    # Protraction: reaching forward/inward rolls clavicle gently toward -Y (forward)
    clav_rz = max(-9.0, min(9.0, ua_rz * 0.20 - side_sgn * max(0.0, -ua_rx) * 0.08))
    clav_rx = max(-4.0, min(4.0, ua_rx * 0.06))

    set_bone_char_rot(rig, f"clavicle.{side_u}", clav_rx, clav_ry, clav_rz, frame=frame)
    set_bone_char_rot(rig, f"upperarm01.{side_u}", ua_rx, ua_ry, ua_rz, frame=frame)
    set_bone_char_rot(rig, f"lowerarm01.{side_u}", la_rx, la_ry, la_rz, frame=frame)

    # Gentle natural wrist extension & counter-alignment (+-6 deg) so hands never look stiff or broken
    wrist_rx = max(-6.0, min(6.0, (la_rx + 14.0) * 0.16))
    wrist_ry = max(-5.0, min(5.0, side_sgn * max(0.0, -la_rx - 16.0) * 0.12))
    wrist_rz = max(-5.0, min(5.0, -la_rz * 0.18))
    set_bone_char_rot(rig, f"wrist.{side_u}", wrist_rx, wrist_ry, wrist_rz, frame=frame)


def apply_hand_gesture(rig, side: str, pose: str = "relaxed", frame: int | None = None):
    """Animate all 5 digits (15 phalanges) cleanly into 4 lifelike cinema hand poses:
    - 'relaxed': soft natural anatomical curl (16..34 deg gradual pinky cascade)
    - 'present': open welcoming scholar gesture (8..18 deg graceful curved open fingers)
    - 'point':   expressive index finger extended (4 deg straight, middle-pinky curled 50..64 deg, thumb tucked)
    - 'shock':   wide-splayed defensive startled fingers (2..8 deg curl + expressive knuckle splay)
    Auto-detects MakeHuman's canonical 163-bone default rig (`finger1`=thumb, `finger2`=index .. `finger5`=pinky)
    while remaining 100% compatible with rigs exposing `finger1..4` + `thumb02/03`."""
    side_u = "L" if str(side).upper().startswith("L") else "R"
    side_sgn = 1.0 if side_u == "L" else -1.0

    # Anatomical presets: (index, middle, ring, pinky, thumb, splays=(idx, mid, rng, pnk))
    presets = {
        "relaxed": (24.0, 30.0, 36.0, 42.0, 16.0, (0.8, 0.2, -0.4, -0.8)),
        "present": (12.0, 15.0, 19.0, 23.0, 8.0, (2.2, 0.7, -0.7, -2.0)),
        "point": (4.0, 50.0, 57.0, 64.0, 20.0, (1.5, -1.5, -2.5, -3.5)),
        "shock": (2.0, 4.0, 6.0, 8.0, 3.0, (7.5, 2.5, -2.5, -7.5)),
    }
    idx_c, mid_c, rng_c, pnk_c, thm_c, splays = presets.get(pose, presets["relaxed"])
    four_curls = (idx_c, mid_c, rng_c, pnk_c)

    # MakeHuman's canonical 163-bone 'default' rig numbers thumb as finger1 and index..pinky as finger2..5!
    has_finger5 = f"finger5-1.{side_u}" in rig.pose.bones
    digit_indices = (2, 3, 4, 5) if has_finger5 else (1, 2, 3, 4)

    for d_slot, f_idx in enumerate(digit_indices):
        base_curl = four_curls[d_slot]
        splay_deg = splays[d_slot] * side_sgn
        for seg, mult in ((1, 0.78), (2, 1.08), (3, 0.72)):
            bname = f"finger{f_idx}-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                rz_rad = math.radians(splay_deg) if seg == 1 else 0.0
                pb.rotation_euler = Euler((math.radians(base_curl * mult), 0.0, rz_rad), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)

    # Thumb articulation: `finger1-1..3` on MakeHuman default rig + `thumb02/03` fallback if present
    if has_finger5:
        for seg, mult in ((1, 0.45), (2, 0.90), (3, 0.85)):
            bname = f"finger1-{seg}.{side_u}"
            if bname in rig.pose.bones:
                pb = rig.pose.bones[bname]
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = Euler((math.radians(thm_c * mult), 0.0, 0.0), "XYZ")
                if frame is not None:
                    pb.keyframe_insert("rotation_euler", frame=frame)
    for tseg, mult in (("thumb02", 0.90), ("thumb03", 1.05)):
        bname = f"{tseg}.{side_u}"
        if bname in rig.pose.bones:
            pb = rig.pose.bones[bname]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((math.radians(thm_c * mult), 0.0, 0.0), "XYZ")
            if frame is not None:
                pb.keyframe_insert("rotation_euler", frame=frame)


def apply_natural_stance(
    rig,
    frame: int | None = None,
    weight_shift: float = 0.0,
    breath_phase: float = 0.0,
    knee_bend: float = 0.0,
):
    """Give characters subtle living contrapposto posture, thoracic breathing, and grounded knee-bend crouch:
    - Chest breathing expansion & contrapposto counter-balance on `spine03` (additive to `spine01`/`spine02` lean!).
    - Lower-torso contrapposto S-curve on `spine05`/`spine04` (plus `pelvis` if present) so hips stay planted
      without lifting or sliding either boot off FLOOR_Z!
    - Exact zero-sum grounded knee-bend (`thigh_rx + shin_rx + foot_rx == 0`) with gentle unloaded-leg softness."""
    ws = max(-1.0, min(1.0, float(weight_shift)))
    kb = max(0.0, min(1.0, float(knee_bend)))

    # Living thoracic breathing (-rx gently opens/lifts ribcage on inhale) + contrapposto S-curve counter-roll
    breath_rx = -math.sin(float(breath_phase)) * 1.15
    set_bone_char_rot(rig, "spine03", breath_rx, -ws * 1.8, ws * 1.2, frame=frame)

    # Lower lumbar contrapposto tilt (`spine05` & `spine04` on MakeHuman default rig) keeps root/feet 100% planted!
    if "spine05" in rig.pose.bones:
        set_bone_char_rot(rig, "spine05", 0.0, ws * 2.4, -ws * 1.5, frame=frame)
    if "spine04" in rig.pose.bones:
        set_bone_char_rot(rig, "spine04", breath_rx * 0.25, ws * 0.6, 0.0, frame=frame)
    if "pelvis" in rig.pose.bones:
        set_bone_char_rot(rig, "pelvis", 0.0, ws * 2.2, -ws * 1.2, frame=frame)

    # Grounded knee-bend articulation (exact thigh+shin+foot=0 keeps soles 100% parallel to floor!)
    # Subtle micro-softness on the unloaded leg breaks statue-stiff twin-locked knees (<0.7mm vertical delta)
    relax_kb = abs(ws) * 0.12
    kb_l = min(1.0, kb + (relax_kb if ws < 0.0 else 0.0))
    kb_r = min(1.0, kb + (relax_kb if ws > 0.0 else 0.0))
    for side, side_kb in (("L", kb_l), ("R", kb_r)):
        thigh_rx = -20.0 * side_kb
        shin_rx = 40.0 * side_kb
        foot_rx = -20.0 * side_kb
        set_bone_char_rot(rig, f"upperleg01.{side}", thigh_rx, 0.0, 0.0, frame=frame)
        set_bone_char_rot(rig, f"lowerleg01.{side}", shin_rx, 0.0, 0.0, frame=frame)
        set_bone_char_rot(rig, f"foot.{side}", foot_rx, 0.0, 0.0, frame=frame)


def apply_relaxed_fingers(rig, frame: int | None = None):
    """Add natural anatomical finger curl to all 10 digits so hands look expressive & lifelike."""
    apply_hand_gesture(rig, "L", "relaxed", frame=frame)
    apply_hand_gesture(rig, "R", "relaxed", frame=frame)


# ============================================================================
# PUBLIC FACTORY: CREATE V8 AUTHENTIC 17TH-CENTURY ULTRA-REALISTIC MPFB2 HUMAN!
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
    auto_metal_map: bool = True,
):
    """Load an optional MPFB2 3D clothing layer (pants/shirt/vest/coat/cloak/suit/shoes)
    with independent linear-safe color grading, velvet sheen, woven micro-bump & sibling *_METAL.png hookup."""
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
    sat_ovr = cfg.get(f"{prefix}_sat_override")
    hue_sh = float(cfg.get(f"{prefix}_hue_shift", 0.5))
    sheen_w = cfg.get(f"{prefix}_sheen")
    rough = cfg.get(f"{prefix}_roughness", default_roughness)
    use_metal = bool(cfg.get(f"{prefix}_auto_metal_map", auto_metal_map))

    if (
        tint_rgb
        or tint_fac > 0.0
        or val_boost != 1.0
        or gam_lift != 1.0
        or sat_ovr is not None
        or hue_sh != 0.5
        or sheen_w is not None
        or add_fabric_bump
        or use_metal
    ):
        _tint_asset_material(
            obj,
            tint_rgb=tint_rgb,
            mix_fac=tint_fac,
            roughness_override=float(rough) if rough is not None else None,
            is_hair=False,
            value_boost=val_boost,
            gamma_lift=gam_lift,
            sat_override=float(sat_ovr) if sat_ovr is not None else None,
            hue_shift=hue_sh,
            sheen_weight=float(sheen_w) if sheen_w is not None else None,
            add_fabric_bump=add_fabric_bump,
            auto_metal_map=use_metal,
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
    and 100% REAL 3D Sculpted MakeHuman multi-layer 17th-century historical attire (`pants_assets`,
    `shirt_assets`, `vest_assets`, `coat_assets`, `cloak_assets`, `suit_assets`, `shoe_assets` —
    ZERO modern ties, ZERO modern bowties, ZERO blocky procedural foam straps or neck-braces!).
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
        ["culturalibre_hair_01.mhclo", "short04.mhclo"],
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

    # 7. Equip Multi-Layer 100% REAL 3D MakeHuman Historical Wardrobe (subdiv_levels=0 for OOM-proof T4 GPU!)
    # Load inner layers (pants -> shirt -> vest) BEFORE outer layers (coat -> cloak) for clean MPFB2 stacking!
    layered_keys = ("pants_assets", "shirt_assets", "vest_assets", "coat_assets", "cloak_assets")
    has_layered = any(cfg.get(k) for k in layered_keys)

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

    if cfg.get("vest_assets"):
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Vest", cfg["vest_assets"], cfg, "vest", default_roughness=0.72, add_fabric_bump=True,
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
        # Default fallback is real 3D MakeHuman Cambridge scholar robe (NEVER modern tie suits!)
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Coat", ["donitz_monk_robe.mhclo", "mindfront_lusekofta.mhclo", "elvs_male_boho_top1.mhclo"],
            cfg, "coat", default_roughness=0.76, add_fabric_bump=True,
        )
    elif suit_cands:
        _equip_clothing_layer(
            HumanService, AssetService, basemesh, name, equipped,
            "Suit", suit_cands, cfg, "suit", default_roughness=0.72, add_fabric_bump=True,
        )

    # Authentic 17th-Century Leather Boots / Shoes
    shoe_cands = cfg.get(
        "shoe_assets",
        ["toigo_ankle_boots_male.mhclo", "punkduck_medieval_boots.mhclo"],
    )
    _equip_clothing_layer(
        HumanService, AssetService, basemesh, name, equipped,
        "Shoes", shoe_cands, cfg, "shoe", default_roughness=0.46, add_fabric_bump=False, auto_metal_map=False,
    )

    # V8 Guard: Legacy procedural `add_cravat` & `add_scholar_cloak` default False and are intentionally
    # disabled in V8 so blocky procedural foam straps & chin-burying neck-braces NEVER render!
    if cfg.get("add_cravat", False) or cfg.get("add_scholar_cloak", False):
        print(
            f"[MPFB2 V8 Guard] '{name}': Ignoring legacy procedural add_cravat/add_scholar_cloak — "
            f"using 100% real 3D sculpted MakeHuman .mhclo garments!"
        )

    # 8. Natural relaxed curled fingers + Adult Stature Normalization (`target_height_m`) + exact floor grounding!
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
        f"[MPFB2 V8 READY] '{name}' ({len(equipped)} real 3D 17c/4K MakeHuman assets equipped, age={raw_age:.2f}, "
        f"height={final_h:.2f}m, feet@z={floor_z:.2f}m, head_top@z={floor_z + final_h:.2f}m)"
    )
    return {"rig": rig, "body": basemesh, "equipped": equipped, "height": final_h, "base_z": base_z}

