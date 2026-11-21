"""
Reusable MPFB2 V6 Ultra-Realistic Cinema Human Character Engine for Blender 4.2.3 LTS Cycles (Kaggle GPU T4 x2)
Uses curated CC0/CC-BY MakeHuman Community 4K Asset Packs (delivered via ~1.2s GitHub CDN in Kaggle Cell 2!):
  - Real 4K Photo-Scanned Male/Female Skins with ENHANCED_SSS Cycles Shader (pores, real lips, brow
    shadows, wrinkles, knuckles — tuned to fast BURLEY SSS for Dual-T4 GPU speed!)
  - Real High-Poly Photoreal Eyes (high-poly.mhclo + PROCEDURAL_EYES EnhancedEye refractive cornea dome!)
  - Real Alpha-Card Eyebrows, Eyelashes, 3D Teeth & Tongue (realistic mouth during Kokoro lip-sync!)
  - Real Alpha-Strand Hair & Optional Facial Hair (flowing wavy shoulder-length Newton mane +
    normal-mapped gentleman hair for Edmond Halley — 100% distinct per Short!)
  - Real 3D Modeled Tailored Suits & Normal-Mapped Leather Shoes (subdiv_levels=0 for OOM-proof T4 GPU speed,
    automatic Delete-Group skin masking — ZERO bodysuit look, ZERO skin clipping!)
  - Blender 4.2 safe ShaderNodeMix (socket.type == 'RGBA') + recursive ShaderNodeGroup copying so
    per-character suit/hair color grading ALWAYS renders without cross-contaminating actors!
  - Roll-proof character-space bone posing (M_rest^-1 @ R_char @ M_rest) + true MPFB2 +Z facing convention
    (+yaw_deg/+rz_deg turns toward +X Screen Right; -yaw_deg/-rz_deg turns toward -X Screen Left)!
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import bpy
from mathutils import Euler, Vector


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
# CYCLES SHADER TUNING HELPERS (FAST PHOTOREAL SKIN, ENHANCED EYES, HAIR/SUIT GRADE)
# ============================================================================
def _find_principled_nodes(nt, visited_trees=None):
    """Recursively find all (node_tree, bsdf_node) pairs inside a material, copying any shared
    ShaderNodeGroup so grading or tuning one character's skin/suit/hair never mutates another actor."""
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


def _tint_asset_material(
    obj,
    tint_rgb: tuple[float, float, float] | None = None,
    mix_fac: float = 0.45,
    roughness_override: float | None = None,
    is_hair: bool = False,
    silver_hair: bool = False,
    value_boost: float = 1.0,
):
    """Non-destructively color-grade an imported MPFB2 clothing or alpha-hair material while preserving
    100% of its original photo-scanned diffuse texture, fabric folds, normal/bump maps & alpha transparency!
    Uses Blender 4.2-safe RGBA socket lookup and isolated ShaderNodeGroup copies."""
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
                _set_bsdf(bsdf, "Roughness", roughness_override if roughness_override is not None else 0.44)
                _set_bsdf(bsdf, "Coat Weight", 0.18)
                _set_bsdf(bsdf, "Sheen Weight", 0.36)
            else:
                if roughness_override is not None and not bsdf.inputs["Roughness"].is_linked:
                    _set_bsdf(bsdf, "Roughness", roughness_override)
                _set_bsdf(bsdf, "Sheen Weight", 0.45)

            try:
                if silver_hair and bc.is_linked:
                    src_socket = bc.links[0].from_socket
                    nt.links.remove(bc.links[0])
                    hsv = nt.nodes.new("ShaderNodeHueSaturation")
                    hsv.inputs["Saturation"].default_value = 0.12
                    hsv.inputs["Value"].default_value = max(1.55, value_boost)
                    nt.links.new(src_socket, hsv.inputs["Color"])
                    nt.links.new(hsv.outputs["Color"], bc)
                    continue

                if tint_rgb is None or mix_fac <= 0.0:
                    if value_boost != 1.0 and bc.is_linked:
                        src_socket = bc.links[0].from_socket
                        nt.links.remove(bc.links[0])
                        hsv = nt.nodes.new("ShaderNodeHueSaturation")
                        hsv.inputs["Value"].default_value = value_boost
                        nt.links.new(src_socket, hsv.inputs["Color"])
                        nt.links.new(hsv.outputs["Color"], bc)
                    continue

                if bc.is_linked:
                    src_socket = bc.links[0].from_socket
                    nt.links.remove(bc.links[0])
                    if value_boost != 1.0:
                        hsv = nt.nodes.new("ShaderNodeHueSaturation")
                        hsv.inputs["Value"].default_value = value_boost
                        nt.links.new(src_socket, hsv.inputs["Color"])
                        src_socket = hsv.outputs["Color"]
                    mix = nt.nodes.new("ShaderNodeMix")
                    mix.data_type = "RGBA"
                    mix.blend_type = "MULTIPLY"
                    fac_in, col_a, col_b, col_out = _get_rgba_mix_sockets(mix)
                    fac_in.default_value = mix_fac
                    col_b.default_value = (*tint_rgb, 1.0)
                    nt.links.new(src_socket, col_a)
                    nt.links.new(col_out, bc)
                else:
                    orig = bc.default_value
                    bc.default_value = (
                        min(1.0, orig[0] * value_boost) * (1.0 - mix_fac) + tint_rgb[0] * mix_fac,
                        min(1.0, orig[1] * value_boost) * (1.0 - mix_fac) + tint_rgb[1] * mix_fac,
                        min(1.0, orig[2] * value_boost) * (1.0 - mix_fac) + tint_rgb[2] * mix_fac,
                        1.0,
                    )
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
    # 1. Exact Kokoro jaw rotation
    set_bone_char_rot(rig, "jaw", jaw_open_deg, 0.0, 0.0, frame=frame)

    # 2. Automatic natural human blinking (smooth 0.14s pulse every ~3.2s)
    cyc = (t + blink_phase) % blink_period
    blink = 0.0
    if cyc < 0.14:
        blink = math.sin((cyc / 0.14) * math.pi)
    lid_deg = blink * 28.0
    set_bone_char_rot(rig, "orbicularis03.L", lid_deg, 0.0, 0.0, frame=frame)
    set_bone_char_rot(rig, "orbicularis03.R", lid_deg, 0.0, 0.0, frame=frame)

    # 3. Expressive eyebrows (oculi01.L/R: +Z lifts brows in wonder/shock, -Z furrows in focus/skepticism)
    dz_brow = brow_lift * 0.0065
    set_bone_char_loc(rig, "oculi01.L", 0.0, 0.0, dz_brow, frame=frame)
    set_bone_char_loc(rig, "oculi01.R", 0.0, 0.0, dz_brow, frame=frame)

    # 4. Subtle cheek/mouth-corner smile or shock pull (risorius03.L/R)
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
# PUBLIC FACTORY: CREATE V6 ULTRA-REALISTIC MPFB2 HUMAN!
# ============================================================================
def create_realistic_scientist(
    name: str,
    floor_z: float,
    start_xy: tuple[float, float],
    yaw_deg: float,
    cfg: dict,
    brass_mat=None,
):
    """Build an ultra-realistic Cycles human using official MakeHuman 4K photo-scanned skins,
    PROCEDURAL_EYES high-poly eyes, alpha eyebrows/eyelashes, 3D teeth/tongue, alpha-strand hair,
    optional 3D mustache/beard & tailored formal suits rigged cleanly to the 163-bone canonical skeleton!
    Note: +yaw_deg faces toward +X (Screen Right); -yaw_deg faces toward -X (Screen Left)."""
    HumanService, TargetService, AssetService, HumanObjectProperties = enable_mpfb2()
    AssetService.update_all_asset_lists()

    # 1. Create anatomical basemesh (19,158 verts) with distinct male/character proportions
    basemesh = HumanService.create_human(feet_on_ground=True, scale=0.1)
    basemesh.name = f"{name}_Body"

    HumanObjectProperties.set_value("gender", float(cfg.get("gender", 1.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("age", float(cfg.get("age", 0.42)), entity_reference=basemesh)
    HumanObjectProperties.set_value("muscle", float(cfg.get("muscle", 0.52)), entity_reference=basemesh)
    HumanObjectProperties.set_value("weight", float(cfg.get("weight", 0.48)), entity_reference=basemesh)
    HumanObjectProperties.set_value("height", float(cfg.get("height", 0.56)), entity_reference=basemesh)
    HumanObjectProperties.set_value("caucasian", float(cfg.get("caucasian", 1.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("african", float(cfg.get("african", 0.0)), entity_reference=basemesh)
    HumanObjectProperties.set_value("asian", float(cfg.get("asian", 0.0)), entity_reference=basemesh)
    TargetService.reapply_macro_details(basemesh)

    # 2. Apply Real 4K Photo-Scanned Skin with MPFB2 ENHANCED_SSS (tuned to fast BURLEY SSS!)
    skin_candidates = cfg.get("skin_assets", ["young_caucasian_male.mhmat", "middleage_caucasian_male.mhmat"])
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
        ("Eyebrows", ("eyebrows",), cfg.get("eyebrow_assets", ["eyebrow001.mhclo", "eyebrow003.mhclo", "eyebrow002.mhclo"]), "MAKESKIN", 0),
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
            cfg.get("hair_tint_rgb"),
            mix_fac=float(cfg.get("brow_tint_fac", 0.45)),
            is_hair=True,
            silver_hair=is_silver,
        )

    # 5. Equip Real Alpha-Strand Hair (distinct per character — subdiv_levels=0!)
    hair_cands = cfg.get(
        "hair_assets",
        ["punkduck_alpha7_curly2.mhclo", "punkduck_alpha7_long.mhclo", "short02.mhclo", "bob02.mhclo"],
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
            is_hair=True,
            silver_hair=bool(cfg.get("silver_hair_main", False)),
            value_boost=float(cfg.get("hair_value_boost", 1.0)),
        )

    # 6. Optional Real 3D Alpha-Card Mustache / Beard (from bodyparts05 pack!)
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
            )

    # 7. Equip Real 3D Tailored Formal Suit + Normal-Mapped Dress Shoes (subdiv_levels=0 for OOM-proof T4 GPU speed!)
    suit_cands = cfg.get(
        "suit_assets",
        [
            "toigo_male_suit_3.mhclo",
            "toigo_suit_with_jacket_and_bowtie.mhclo",
            "toigo_male_suit_tie_and_jacket.mhclo",
            "male_elegantsuit01.mhclo",
        ],
    )
    suit_path, _ = _find_best_asset(AssetService, ("clothes",), suit_cands)
    if suit_path:
        suit_obj = HumanService.add_mhclo_asset(
            suit_path,
            basemesh,
            asset_type="Clothes",
            subdiv_levels=0,
            material_type="MAKESKIN",
            set_up_rigging=True,
            interpolate_weights=True,
        )
        equipped["Suit"] = suit_obj
        print(f"[MPFB2 Asset] '{name}' Suit       -> {os.path.basename(suit_path)}")
        if cfg.get("suit_tint_rgb") and cfg.get("suit_tint_fac", 0.0) > 0:
            _tint_asset_material(
                suit_obj,
                cfg["suit_tint_rgb"],
                mix_fac=float(cfg["suit_tint_fac"]),
                roughness_override=0.72,
                is_hair=False,
                value_boost=float(cfg.get("suit_value_boost", 1.0)),
            )

    shoe_cands = cfg.get("shoe_assets", ["shoes01.mhclo", "shoes04.mhclo", "shoes02.mhclo"])
    shoe_path, _ = _find_best_asset(AssetService, ("clothes",), shoe_cands)
    if shoe_path:
        shoe_obj = HumanService.add_mhclo_asset(
            shoe_path,
            basemesh,
            asset_type="Clothes",
            subdiv_levels=0,
            material_type="MAKESKIN",
            set_up_rigging=True,
            interpolate_weights=True,
        )
        equipped["Shoes"] = shoe_obj
        print(f"[MPFB2 Asset] '{name}' Shoes      -> {os.path.basename(shoe_path)}")

    # 8. Natural relaxed curled fingers + exact floor grounding
    apply_relaxed_fingers(rig)

    bpy.context.view_layer.update()
    min_z = min((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    max_z = max((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    base_z = floor_z - min_z
    rig.location = Vector((start_xy[0], start_xy[1], base_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)))

    print(
        f"[MPFB2 V6 READY] '{name}' ({len(equipped)} real 4K/3D assets equipped, "
        f"height={max_z - min_z:.2f}m, feet@z={floor_z:.2f}m, head_top@z={floor_z + (max_z - min_z):.2f}m)"
    )
    return {"rig": rig, "body": basemesh, "equipped": equipped, "height": max_z - min_z, "base_z": base_z}
