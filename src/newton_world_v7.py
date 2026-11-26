"""
V9 Cinema Photoreal Outdoor Woolsthorpe Orchard & Cosmos Environment Module for Scientist #9: Isaac Newton
==========================================================================================================

100% outdoor-to-cosmos Cycles photoreal environment (ZERO Play-Doh canopy spheres, ZERO indoor brick
room, ZERO cartoon green mortar lines, ZERO blown-out grey daylight sky, ZERO black moon ball!).

Features:
  1. Animated Physical Cycles Sky: Calibrated NISHITA Golden-Hour Amber-to-Sapphire Sunset (Beats 1-4:
     5.8° sun elevation, ozone=3.8, Strength=0.24) blending smoothly into a Deep-Sapphire & Starlight
     Milky Way Nebula Veil Twilight Cosmos (Beats 5-6, t = 19.8..22.2s).
  2. 160 True 3D Random Spherical Stars (warm-gold & sapphire-white, random.Random(1687)) +
     True 3D Sculpted Crescent Moon (pure smooth bmesh lunar crescent, ZERO black occluder!).
  3. Rolling Woolsthorpe Meadow Grass Terrain, Weathered Cotswold Honey-Buff & Taupe-Grey Flagstone
     Courtyard (exact Z = 0.000, dark charcoal-umber recessed mortar, corner-only damp moss, subtle
     dew-sheen specular pockets 0.26..0.86 roughness, Cycles AO), Carved Cotswold Stone Garden
     Balustrade (18 turned stone urn balusters + wrought-iron flickering twilight garden lantern on
     middle pier X=+1.85), Climbing Dark-Emerald English Ivy (~220 matte sculpted leaves, 1 fast
     mesh, visible_shadow=False), 3 Layered Atmospheric Rolling Hill Ridges & 140 Grass/Wildflower Clumps.
  4. Monumental 3.5m-tall Sculpted Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0) with deeply
     fissured AO gnarled bark, 5 sweeping organic primary boughs (Hero Overhanging Bough terminating at
     exact HERO_APPLE_ATTACH = (-0.52, 0.12, 1.94)), 50+ slender woody branches & sub-twigs, ~2,600
     individual sculpted 6-vert matte-botanical leaves (ZERO blob spheres!), shaped cleanly left/low
     of Beat 6's starry sky title zone, 10 hanging heirloom apples + 3 fallen heirloom apples at roots.
  5. Shaded Left Cotswold Stone Manor Garden Loggia / Optical Alcove (X in [-3.75, -1.80]) with
     semicircular carved stone arch, walnut optical shutter with physical brass-rimmed pinhole at exact
     PRISM_PINHOLE = (-3.38, 0.06, 1.54), recessed warm candlelit 17th-century dark-oak leaded cottage
     casement window, and dark limestone chiaroscuro back wall for Beat 3's prism.
  6. Three Sculpted Cotswold Stone Garden Plinths (APPLE_PLINTH, PRINCIPIA_PLINTH, CRADLE_PLINTH)
     with flush inlaid carved antique-bronze classical laurel accent rings & Cycles AO crevice depth,
     plus open flagstone anchors (PRISM_ANCHOR, ORBIT_ANCHOR).
  7. Volumetric Golden-Hour God-Rays + 45 Drifting Golden Pollen Motes + 18 Tumbling Golden-Hour
     Orchard Leaves (Beats 1-4) -> 24 Pulsing Warm-Gold Twilight Fireflies (Beats 5-6).
  8. Hollywood Golden-Hour Chiaroscuro Sun & Rembrandt Lighting Rig animating smoothly to Moonlit
     Twilight with warm-amber flickering lantern counterpoint.
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector


# ============================================================================
# 1. PUBLIC WORLD CONSTANTS & EXACT V7 ANCHOR COORDINATES
# ============================================================================
FLOOR_Z = 0.0
HERO_APPLE_ATTACH = Vector((-0.52, 0.12, 1.94))
APPLE_PLINTH = Vector((-0.52, 0.12, 0.88))
PRINCIPIA_PLINTH = Vector((0.55, 0.08, 0.90))
PRISM_ANCHOR = Vector((-2.85, 0.06, 0.0))
PRISM_PINHOLE = Vector((-3.38, 0.06, 1.54))
CRADLE_PLINTH = Vector((2.88, 0.08, 0.90))
ORBIT_ANCHOR = Vector((0.64, 0.14, 0.0))

# Convenience aliases matching spec wording
PRISM_FLOOR = PRISM_ANCHOR
ORBIT_FLOOR = ORBIT_ANCHOR
TREE_BASE = Vector((-0.18, 1.35, 0.0))
CRESCENT_MOON_POS = Vector((5.20, 19.50, 8.90))


# ============================================================================
# 2. MATH & BLENDER 4.2 SHADER SAFETY HELPERS
# ============================================================================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    u = clamp01((x - e0) / max(1e-6, e1 - e0))
    return u * u * (3.0 - 2.0 * u)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def set_bsdf(bsdf, name: str, val) -> None:
    """Blender 4.2 safe Principled BSDF socket assignment."""
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def get_rgba_mix_sockets(mix_node):
    """Blender 4.2 safe RGBA socket lookup for ShaderNodeMix (data_type='RGBA')."""
    rgba_in = [s for s in mix_node.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix_node.outputs if s.type == "RGBA"]
    return mix_node.inputs[0], rgba_in[0], rgba_in[1], rgba_out[0]


def _link_obj(scene, obj) -> None:
    if obj.name not in scene.collection.objects:
        scene.collection.objects.link(obj)


def _finalize_bmesh(
    scene,
    name: str,
    bm: bmesh.types.BMesh,
    mats=None,
    loc=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0),
    smooth: bool = True,
    edge_split_deg: float | None = None,
):
    """Convert bmesh to a linked Blender Object cleanly with outward normals & optional edge-split."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    obj.rotation_euler = Euler(rot, "XYZ")
    _link_obj(scene, obj)

    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    if edge_split_deg is not None:
        es = obj.modifiers.new("SmoothSplit", "EDGE_SPLIT")
        es.split_angle = math.radians(edge_split_deg)

    if mats is not None:
        if isinstance(mats, (list, tuple)):
            for m in mats:
                obj.data.materials.append(m)
        else:
            obj.data.materials.append(mats)
    return obj


# ============================================================================
# 3. PROCEDURAL CYCLES PBR MATERIALS (COTSWOLD ORCHARD & COSMOS PALETTE)
# ============================================================================
def _make_cotswold_stone_mat(name: str = "CotswoldCarvedLimestone", dark: bool = False):
    """Weathered 17th-century Cotswold limestone with Cycles Ambient Occlusion crevice shading & crisp carved grain."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_macro.inputs["Scale"].default_value = 4.6
    ns_macro.inputs["Detail"].default_value = 9.0
    ns_macro.inputs["Roughness"].default_value = 0.66

    ns_fine = nt.nodes.new("ShaderNodeTexNoise")
    ns_fine.inputs["Scale"].default_value = 26.0
    ns_fine.inputs["Detail"].default_value = 10.0
    ns_fine.inputs["Roughness"].default_value = 0.70

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    if dark:
        rmp.color_ramp.elements[0].position = 0.22
        rmp.color_ramp.elements[0].color = (0.020, 0.018, 0.022, 1.0)
        e_mid = rmp.color_ramp.elements.new(0.55)
        e_mid.color = (0.040, 0.036, 0.034, 1.0)
        rmp.color_ramp.elements[-1].position = 0.86
        rmp.color_ramp.elements[-1].color = (0.066, 0.058, 0.050, 1.0)
    else:
        rmp.color_ramp.elements[0].position = 0.20
        rmp.color_ramp.elements[0].color = (0.205, 0.162, 0.108, 1.0)
        e_mid = rmp.color_ramp.elements.new(0.55)
        e_mid.color = (0.315, 0.258, 0.178, 1.0)
        rmp.color_ramp.elements[-1].position = 0.86
        rmp.color_ramp.elements[-1].color = (0.415, 0.348, 0.248, 1.0)

    # Subtle weathered garden lichen/damp moss undertone in shaded pockets
    ns_moss = nt.nodes.new("ShaderNodeTexNoise")
    ns_moss.inputs["Scale"].default_value = 2.6
    ns_moss.inputs["Detail"].default_value = 6.0
    moss_rmp = nt.nodes.new("ShaderNodeValToRGB")
    moss_rmp.color_ramp.elements[0].position = 0.66
    moss_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    moss_rmp.color_ramp.elements[1].position = 0.88
    m_v = 0.22 if not dark else 0.10
    moss_rmp.color_ramp.elements[1].color = (m_v, m_v, m_v, 1.0)

    mix_moss = nt.nodes.new("ShaderNodeMix")
    mix_moss.data_type = "RGBA"
    mix_moss.blend_type = "MIX"
    mm_fac, mm_a, mm_b, mm_out = get_rgba_mix_sockets(mix_moss)
    mm_b.default_value = (0.032, 0.054, 0.020, 1.0) if not dark else (0.015, 0.026, 0.014, 1.0)

    # Photorealistic Cycles Ambient Occlusion darkening flutes, plinth moldings & urn crevices
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.only_local = True
    ao.inputs["Distance"].default_value = 0.45
    ao_rmp = nt.nodes.new("ShaderNodeValToRGB")
    ao_rmp.color_ramp.elements[0].position = 0.12
    ao_rmp.color_ramp.elements[0].color = (0.42, 0.40, 0.37, 1.0)
    ao_rmp.color_ramp.elements[1].position = 0.94
    ao_rmp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mao_fac, mao_a, mao_b, mao_out = get_rgba_mix_sockets(mix_ao)
    mao_fac.default_value = 1.0

    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.clamp = True
    r_map.inputs["From Min"].default_value = 0.20
    r_map.inputs["From Max"].default_value = 0.82
    r_map.inputs["To Min"].default_value = 0.58 if not dark else 0.72
    r_map.inputs["To Max"].default_value = 0.86 if not dark else 0.92

    bmp_fine = nt.nodes.new("ShaderNodeBump")
    bmp_fine.inputs["Strength"].default_value = 0.18
    bmp_fine.inputs["Distance"].default_value = 0.008
    bmp_macro = nt.nodes.new("ShaderNodeBump")
    bmp_macro.inputs["Strength"].default_value = 0.28
    bmp_macro.inputs["Distance"].default_value = 0.018

    nt.links.new(tc.outputs["Object"], ns_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_fine.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_moss.inputs["Vector"])
    nt.links.new(ns_macro.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(ns_moss.outputs["Fac"], moss_rmp.inputs["Fac"])
    nt.links.new(moss_rmp.outputs["Color"], mm_fac)
    nt.links.new(rmp.outputs["Color"], mm_a)
    nt.links.new(ao.outputs["AO"], ao_rmp.inputs["Fac"])
    nt.links.new(mm_out, mao_a)
    nt.links.new(ao_rmp.outputs["Color"], mao_b)
    nt.links.new(mao_out, bsdf.inputs["Base Color"])
    nt.links.new(ns_fine.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns_fine.outputs["Fac"], bmp_fine.inputs["Height"])
    nt.links.new(ns_macro.outputs["Fac"], bmp_macro.inputs["Height"])
    nt.links.new(bmp_fine.outputs["Normal"], bmp_macro.inputs["Normal"])
    nt.links.new(bmp_macro.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_cotswold_ashlar_mat(name: str = "CotswoldManorAshlar", dark: bool = False):
    """Weathered Cotswold manor ashlar stone blocks with mortar relief & Cycles AO crevice depth."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (math.pi * 0.5, 0.0, 0.0)
    mp.inputs["Scale"].default_value = (1.1, 1.1, 1.1)

    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.5
    if dark:
        br.inputs["Color1"].default_value = (0.040, 0.036, 0.034, 1.0)
        br.inputs["Color2"].default_value = (0.024, 0.022, 0.024, 1.0)
        br.inputs["Mortar"].default_value = (0.009, 0.009, 0.010, 1.0)
    else:
        br.inputs["Color1"].default_value = (0.31, 0.245, 0.165, 1.0)
        br.inputs["Color2"].default_value = (0.22, 0.175, 0.120, 1.0)
        br.inputs["Mortar"].default_value = (0.065, 0.056, 0.045, 1.0)
    br.inputs["Scale"].default_value = 2.6
    br.inputs["Mortar Size"].default_value = 0.018
    br.inputs["Mortar Smooth"].default_value = 0.18
    br.inputs["Brick Width"].default_value = 0.65
    br.inputs["Row Height"].default_value = 0.28

    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 9.0
    ns.inputs["Detail"].default_value = 10.0
    ns.inputs["Roughness"].default_value = 0.68

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.58, 0.54, 0.48, 1.0)
    rmp.color_ramp.elements[1].position = 0.82
    rmp.color_ramp.elements[1].color = (1.0, 0.96, 0.90, 1.0)

    mix_w = nt.nodes.new("ShaderNodeMix")
    mix_w.data_type = "RGBA"
    mix_w.blend_type = "MULTIPLY"
    mw_fac, mw_a, mw_b, mw_out = get_rgba_mix_sockets(mix_w)
    mw_fac.default_value = 0.42

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.only_local = True
    ao.inputs["Distance"].default_value = 0.45
    ao_rmp = nt.nodes.new("ShaderNodeValToRGB")
    ao_rmp.color_ramp.elements[0].position = 0.10
    ao_rmp.color_ramp.elements[0].color = (0.42, 0.40, 0.38, 1.0)
    ao_rmp.color_ramp.elements[1].position = 0.94
    ao_rmp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mao_fac, mao_a, mao_b, mao_out = get_rgba_mix_sockets(mix_ao)
    mao_fac.default_value = 1.0

    bmp_g = nt.nodes.new("ShaderNodeBump")
    bmp_g.inputs["Strength"].default_value = 0.22
    bmp_m = nt.nodes.new("ShaderNodeBump")
    bmp_m.inputs["Strength"].default_value = 0.38 if not dark else 0.24

    set_bsdf(bsdf, "Roughness", 0.78 if not dark else 0.86)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(br.outputs["Color"], mw_a)
    nt.links.new(rmp.outputs["Color"], mw_b)
    nt.links.new(ao.outputs["AO"], ao_rmp.inputs["Fac"])
    nt.links.new(mw_out, mao_a)
    nt.links.new(ao_rmp.outputs["Color"], mao_b)
    nt.links.new(mao_out, bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp_g.inputs["Height"])
    nt.links.new(br.outputs["Fac"], bmp_m.inputs["Height"])
    nt.links.new(bmp_g.outputs["Normal"], bmp_m.inputs["Normal"])
    nt.links.new(bmp_m.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_mossy_flagstone_mat(name: str = "CotswoldMossyFlagstone"):
    """Weathered Cotswold honey-buff & warm taupe-grey flagstones with dark charcoal-umber mortar, corner-only damp moss & AO."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Hand-laid organic slight warp so antique limestone slab joints never look like a CGI grid
    ns_warp = nt.nodes.new("ShaderNodeTexNoise")
    ns_warp.inputs["Scale"].default_value = 2.1
    ns_warp.inputs["Detail"].default_value = 4.5

    vec_scale = nt.nodes.new("ShaderNodeVectorMath")
    vec_scale.operation = "SCALE"
    vec_scale.inputs["Scale"].default_value = 0.052

    vec_add = nt.nodes.new("ShaderNodeVectorMath")
    vec_add.operation = "ADD"

    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.44
    br.inputs["Color1"].default_value = (0.265, 0.212, 0.148, 1.0)  # Weathered Cotswold honey-buff limestone
    br.inputs["Color2"].default_value = (0.172, 0.156, 0.132, 1.0)  # Aged warm taupe-grey flagstone
    br.inputs["Mortar"].default_value = (0.032, 0.030, 0.026, 1.0)  # Dark earthy charcoal-umber recessed mortar!
    br.inputs["Bias"].default_value = -0.08
    br.inputs["Scale"].default_value = 2.05
    br.inputs["Mortar Size"].default_value = 0.020
    br.inputs["Mortar Smooth"].default_value = 0.20
    br.inputs["Brick Width"].default_value = 0.74
    br.inputs["Row Height"].default_value = 0.50

    # Stone grain & mineral weathering across individual slabs
    ns_stone = nt.nodes.new("ShaderNodeTexNoise")
    ns_stone.inputs["Scale"].default_value = 11.5
    ns_stone.inputs["Detail"].default_value = 10.0
    ns_stone.inputs["Roughness"].default_value = 0.68

    ns_fine = nt.nodes.new("ShaderNodeTexNoise")
    ns_fine.inputs["Scale"].default_value = 32.0
    ns_fine.inputs["Detail"].default_value = 8.0
    ns_fine.inputs["Roughness"].default_value = 0.72

    rmp_tint = nt.nodes.new("ShaderNodeValToRGB")
    rmp_tint.color_ramp.elements[0].position = 0.20
    rmp_tint.color_ramp.elements[0].color = (0.64, 0.60, 0.54, 1.0)
    rmp_tint.color_ramp.elements[1].position = 0.84
    rmp_tint.color_ramp.elements[1].color = (1.06, 1.00, 0.92, 1.0)

    mix_tint = nt.nodes.new("ShaderNodeMix")
    mix_tint.data_type = "RGBA"
    mix_tint.blend_type = "MULTIPLY"
    mt_fac, mt_a, mt_b, mt_out = get_rgba_mix_sockets(mix_tint)
    mt_fac.default_value = 0.48

    # Subtle irregular dark-moss patches ONLY in damp corners (low-frequency macro mask * patchy detail)
    ns_damp_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_damp_macro.inputs["Scale"].default_value = 1.15
    ns_damp_macro.inputs["Detail"].default_value = 5.0
    ns_damp_macro.inputs["Roughness"].default_value = 0.55

    ns_moss_detail = nt.nodes.new("ShaderNodeTexNoise")
    ns_moss_detail.inputs["Scale"].default_value = 4.8
    ns_moss_detail.inputs["Detail"].default_value = 7.0

    moss_mul = nt.nodes.new("ShaderNodeMath")
    moss_mul.operation = "MULTIPLY"

    moss_rmp = nt.nodes.new("ShaderNodeValToRGB")
    moss_rmp.color_ramp.elements[0].position = 0.42
    moss_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    moss_rmp.color_ramp.elements[1].position = 0.66
    moss_rmp.color_ramp.elements[1].color = (0.42, 0.42, 0.42, 1.0)

    mix_moss = nt.nodes.new("ShaderNodeMix")
    mix_moss.data_type = "RGBA"
    mix_moss.blend_type = "MIX"
    mm_fac, mm_a, mm_b, mm_out = get_rgba_mix_sockets(mix_moss)
    mm_b.default_value = (0.024, 0.048, 0.015, 1.0)  # Deep earthy damp olive moss (NEVER bright neon green!)

    # Cycles Ambient Occlusion grounding plinths, balustrade base & boots into the flagstones
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.inputs["Distance"].default_value = 0.35
    ao_rmp = nt.nodes.new("ShaderNodeValToRGB")
    ao_rmp.color_ramp.elements[0].position = 0.14
    ao_rmp.color_ramp.elements[0].color = (0.36, 0.34, 0.31, 1.0)
    ao_rmp.color_ramp.elements[1].position = 0.94
    ao_rmp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mao_fac, mao_a, mao_b, mao_out = get_rgba_mix_sockets(mix_ao)
    mao_fac.default_value = 0.92

    # V9 Dew-Sheen Specular Pockets: dry weathered limestone (0.54..0.86) modulated by low-frequency
    # morning/evening dampness pockets (0.26 subtle golden-sunlit sheen patches, strictly off mortar seams!)
    rough_mix = nt.nodes.new("ShaderNodeMath")
    rough_mix.operation = "MULTIPLY"
    rough_mix.inputs[1].default_value = 0.55

    rough_add = nt.nodes.new("ShaderNodeMath")
    rough_add.operation = "ADD"

    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.clamp = True
    r_map.inputs["From Min"].default_value = 0.15
    r_map.inputs["From Max"].default_value = 1.35
    r_map.inputs["To Min"].default_value = 0.54
    r_map.inputs["To Max"].default_value = 0.86

    ns_dew = nt.nodes.new("ShaderNodeTexNoise")
    ns_dew.inputs["Scale"].default_value = 0.85
    ns_dew.inputs["Detail"].default_value = 5.0
    ns_dew.inputs["Roughness"].default_value = 0.58

    dew_mask = nt.nodes.new("ShaderNodeMapRange")
    dew_mask.clamp = True
    dew_mask.interpolation_type = "SMOOTHSTEP"
    dew_mask.inputs["From Min"].default_value = 0.46
    dew_mask.inputs["From Max"].default_value = 0.72
    dew_mask.inputs["To Min"].default_value = 0.0
    dew_mask.inputs["To Max"].default_value = 1.0

    # Keep recessed mortar seams matte (br.Fac == 1.0 in mortar -> mortar_inv == 0.0)
    mortar_inv = nt.nodes.new("ShaderNodeMath")
    mortar_inv.operation = "SUBTRACT"
    mortar_inv.inputs[0].default_value = 1.0
    mortar_inv.use_clamp = True

    dew_on_slab = nt.nodes.new("ShaderNodeMath")
    dew_on_slab.operation = "MULTIPLY"

    # r_final = r_dry - dew_on_slab * (r_dry - 0.26) -> exact [0.26 .. 0.86] range!
    r_minus_dew = nt.nodes.new("ShaderNodeMath")
    r_minus_dew.operation = "SUBTRACT"
    r_minus_dew.inputs[1].default_value = 0.26

    dew_drop = nt.nodes.new("ShaderNodeMath")
    dew_drop.operation = "MULTIPLY"

    r_final = nt.nodes.new("ShaderNodeMath")
    r_final.operation = "SUBTRACT"
    r_final.use_clamp = True

    dew_coat = nt.nodes.new("ShaderNodeMath")
    dew_coat.operation = "MULTIPLY"
    dew_coat.inputs[1].default_value = 0.26
    set_bsdf(bsdf, "Coat Roughness", 0.18)

    # Strong stone grain bump (0.26) + crisp recessed mortar relief (0.48)
    bmp_fine = nt.nodes.new("ShaderNodeBump")
    bmp_fine.inputs["Strength"].default_value = 0.12
    bmp_fine.inputs["Distance"].default_value = 0.006
    bmp_grain = nt.nodes.new("ShaderNodeBump")
    bmp_grain.inputs["Strength"].default_value = 0.26
    bmp_grain.inputs["Distance"].default_value = 0.015
    bmp_seam = nt.nodes.new("ShaderNodeBump")
    bmp_seam.inputs["Strength"].default_value = 0.48
    bmp_seam.inputs["Distance"].default_value = 0.026
    bmp_seam.invert = True

    nt.links.new(tc.outputs["Object"], ns_warp.inputs["Vector"])
    nt.links.new(ns_warp.outputs["Color"], vec_scale.inputs[0])
    nt.links.new(tc.outputs["Object"], vec_add.inputs[0])
    nt.links.new(vec_scale.outputs["Vector"], vec_add.inputs[1])
    nt.links.new(vec_add.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_stone.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_fine.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_dew.inputs["Vector"])
    nt.links.new(ns_stone.outputs["Fac"], rmp_tint.inputs["Fac"])
    nt.links.new(br.outputs["Color"], mt_a)
    nt.links.new(rmp_tint.outputs["Color"], mt_b)
    nt.links.new(tc.outputs["Object"], ns_damp_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_moss_detail.inputs["Vector"])
    nt.links.new(ns_damp_macro.outputs["Fac"], moss_mul.inputs[0])
    nt.links.new(ns_moss_detail.outputs["Fac"], moss_mul.inputs[1])
    nt.links.new(moss_mul.outputs["Value"], moss_rmp.inputs["Fac"])
    nt.links.new(moss_rmp.outputs["Color"], mm_fac)
    nt.links.new(mt_out, mm_a)
    nt.links.new(ao.outputs["AO"], ao_rmp.inputs["Fac"])
    nt.links.new(mm_out, mao_a)
    nt.links.new(ao_rmp.outputs["Color"], mao_b)
    nt.links.new(mao_out, bsdf.inputs["Base Color"])
    nt.links.new(ns_stone.outputs["Fac"], rough_mix.inputs[0])
    nt.links.new(br.outputs["Fac"], rough_add.inputs[0])
    nt.links.new(rough_mix.outputs["Value"], rough_add.inputs[1])
    nt.links.new(rough_add.outputs["Value"], r_map.inputs["Value"])
    nt.links.new(ns_dew.outputs["Fac"], dew_mask.inputs["Value"])
    nt.links.new(br.outputs["Fac"], mortar_inv.inputs[1])
    nt.links.new(dew_mask.outputs["Result"], dew_on_slab.inputs[0])
    nt.links.new(mortar_inv.outputs["Value"], dew_on_slab.inputs[1])
    nt.links.new(r_map.outputs["Result"], r_minus_dew.inputs[0])
    nt.links.new(dew_on_slab.outputs["Value"], dew_drop.inputs[0])
    nt.links.new(r_minus_dew.outputs["Value"], dew_drop.inputs[1])
    nt.links.new(r_map.outputs["Result"], r_final.inputs[0])
    nt.links.new(dew_drop.outputs["Value"], r_final.inputs[1])
    nt.links.new(r_final.outputs["Value"], bsdf.inputs["Roughness"])
    nt.links.new(dew_on_slab.outputs["Value"], dew_coat.inputs[0])
    if "Coat Weight" in bsdf.inputs:
        nt.links.new(dew_coat.outputs["Value"], bsdf.inputs["Coat Weight"])
    nt.links.new(ns_fine.outputs["Fac"], bmp_fine.inputs["Height"])
    nt.links.new(ns_stone.outputs["Fac"], bmp_grain.inputs["Height"])
    nt.links.new(br.outputs["Fac"], bmp_seam.inputs["Height"])
    nt.links.new(bmp_fine.outputs["Normal"], bmp_grain.inputs["Normal"])
    nt.links.new(bmp_grain.outputs["Normal"], bmp_seam.inputs["Normal"])
    nt.links.new(bmp_seam.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_meadow_grass_mat(name: str = "WoolsthorpeMeadowGrass", distant: bool = False, haze_mix: float = 0.0):
    """Natural botanical Woolsthorpe meadow grass (or layered golden-hour atmospheric misty Lincolnshire hills)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Macro pasture patchwork & tree-line hedgerow variation
    ns_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_macro.inputs["Scale"].default_value = 0.48 if distant else 1.45
    ns_macro.inputs["Detail"].default_value = 8.0
    ns_macro.inputs["Roughness"].default_value = 0.62

    # Fine grass-blade + hedgerow Voronoi/Noise
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 3.2 if distant else 42.0
    ns_blade = nt.nodes.new("ShaderNodeTexNoise")
    ns_blade.inputs["Scale"].default_value = 12.0 if distant else 58.0
    ns_blade.inputs["Detail"].default_value = 10.0
    ns_blade.inputs["Roughness"].default_value = 0.72

    rmp_macro = nt.nodes.new("ShaderNodeValToRGB")
    cr = rmp_macro.color_ramp
    if distant:
        cr.elements[0].position = 0.22
        cr.elements[0].color = (0.022, 0.052, 0.020, 1.0)  # Shaded distant woodland hedgerow
        e1 = cr.elements.new(0.54)
        e1.color = (0.058, 0.095, 0.032, 1.0)  # Sun-kissed olive Lincolnshire pasture
        cr.elements[-1].position = 0.84
        cr.elements[-1].color = (0.135, 0.145, 0.052, 1.0)  # Golden-hour dry meadow ridge
    else:
        cr.elements[0].position = 0.22
        cr.elements[0].color = (0.020, 0.068, 0.015, 1.0)  # Deep natural English orchard meadow
        e1 = cr.elements.new(0.54)
        e1.color = (0.048, 0.125, 0.022, 1.0)  # Rich natural olive-emerald grass
        cr.elements[-1].position = 0.84
        cr.elements[-1].color = (0.118, 0.175, 0.034, 1.0)  # Warm golden-clover sunlit pasture

    rmp_blade = nt.nodes.new("ShaderNodeValToRGB")
    rmp_blade.color_ramp.elements[0].position = 0.18
    rmp_blade.color_ramp.elements[0].color = (0.64, 0.68, 0.58, 1.0)
    rmp_blade.color_ramp.elements[1].position = 0.84
    rmp_blade.color_ramp.elements[1].color = (1.12, 1.08, 0.90, 1.0)

    mix_b = nt.nodes.new("ShaderNodeMix")
    mix_b.data_type = "RGBA"
    mix_b.blend_type = "MULTIPLY"
    mb_fac, mb_a, mb_b, mb_out = get_rgba_mix_sockets(mix_b)
    mb_fac.default_value = 0.46

    final_col_out = mb_out
    if haze_mix > 0.0:
        # Layered atmospheric perspective: warm golden-hour valley mist transitioning to soft dusty haze!
        sep_z = nt.nodes.new("ShaderNodeSeparateXYZ")
        alt_rmp = nt.nodes.new("ShaderNodeValToRGB")
        alt_rmp.color_ramp.elements[0].position = 0.05
        alt_rmp.color_ramp.elements[0].color = (0.24, 0.20, 0.15, 1.0)  # Warm golden-hour valley mist
        alt_rmp.color_ramp.elements[1].position = 0.85
        alt_rmp.color_ramp.elements[1].color = (0.15, 0.17, 0.20, 1.0)  # Cool distant horizon atmospheric haze

        mix_haze = nt.nodes.new("ShaderNodeMix")
        mix_haze.data_type = "RGBA"
        mix_haze.blend_type = "MIX"
        mh_fac, mh_a, mh_b, mh_out = get_rgba_mix_sockets(mix_haze)
        mh_fac.default_value = clamp01(haze_mix)
        nt.links.new(tc.outputs["Generated"], sep_z.inputs["Vector"])
        nt.links.new(sep_z.outputs["Z"], alt_rmp.inputs["Fac"])
        nt.links.new(mb_out, mh_a)
        nt.links.new(alt_rmp.outputs["Color"], mh_b)
        final_col_out = mh_out

    bmp_vor = nt.nodes.new("ShaderNodeBump")
    bmp_vor.inputs["Strength"].default_value = 0.22 if distant else 0.28
    bmp_bl = nt.nodes.new("ShaderNodeBump")
    bmp_bl.inputs["Strength"].default_value = 0.24 if distant else 0.36

    set_bsdf(bsdf, "Roughness", 0.88 if not distant else 0.94)
    set_bsdf(bsdf, "Specular IOR Level", 0.18)

    nt.links.new(tc.outputs["Object"], ns_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_blade.inputs["Vector"])
    nt.links.new(ns_macro.outputs["Fac"], rmp_macro.inputs["Fac"])
    nt.links.new(ns_blade.outputs["Fac"], rmp_blade.inputs["Fac"])
    nt.links.new(rmp_macro.outputs["Color"], mb_a)
    nt.links.new(rmp_blade.outputs["Color"], mb_b)
    nt.links.new(final_col_out, bsdf.inputs["Base Color"])
    nt.links.new(vor.outputs["Distance"], bmp_vor.inputs["Height"])
    nt.links.new(ns_blade.outputs["Fac"], bmp_bl.inputs["Height"])
    nt.links.new(bmp_vor.outputs["Normal"], bmp_bl.inputs["Normal"])
    nt.links.new(bmp_bl.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_gnarled_bark_mat(name: str = "WoolsthorpeGnarledBark"):
    """Ancient gnarled apple-tree bark: deep vertical fissures (Z=0.35, X/Y=5.5, Bump=0.62), espresso crevices & AO."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (5.5, 5.5, 0.35)  # Deeply vertically stretched ancient bark fissures!

    ns_bark = nt.nodes.new("ShaderNodeTexNoise")
    ns_bark.inputs["Scale"].default_value = 6.2
    ns_bark.inputs["Detail"].default_value = 12.0
    ns_bark.inputs["Roughness"].default_value = 0.78
    ns_bark.inputs["Distortion"].default_value = 1.45

    ns_ridge = nt.nodes.new("ShaderNodeTexNoise")
    ns_ridge.inputs["Scale"].default_value = 18.0
    ns_ridge.inputs["Detail"].default_value = 8.0
    ns_ridge.inputs["Roughness"].default_value = 0.72

    rmp_bark = nt.nodes.new("ShaderNodeValToRGB")
    rmp_bark.color_ramp.elements[0].position = 0.24
    rmp_bark.color_ramp.elements[0].color = (0.014, 0.009, 0.006, 1.0)  # Deep espresso-umber bark fissures
    e_mid = rmp_bark.color_ramp.elements.new(0.56)
    e_mid.color = (0.058, 0.038, 0.022, 1.0)  # Aged dark English orchard bark
    rmp_bark.color_ramp.elements[-1].position = 0.86
    rmp_bark.color_ramp.elements[-1].color = (0.118, 0.082, 0.050, 1.0)  # Weathered dry-wood ridge highlights

    # Subtle natural orchard lichen patches on gnarled trunk
    ns_lich = nt.nodes.new("ShaderNodeTexNoise")
    ns_lich.inputs["Scale"].default_value = 4.2
    ns_lich.inputs["Detail"].default_value = 7.0
    lich_rmp = nt.nodes.new("ShaderNodeValToRGB")
    lich_rmp.color_ramp.elements[0].position = 0.68
    lich_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    lich_rmp.color_ramp.elements[1].position = 0.86
    lich_rmp.color_ramp.elements[1].color = (0.38, 0.38, 0.38, 1.0)

    mix_l = nt.nodes.new("ShaderNodeMix")
    mix_l.data_type = "RGBA"
    mix_l.blend_type = "MIX"
    ml_fac, ml_a, ml_b, ml_out = get_rgba_mix_sockets(mix_l)
    ml_b.default_value = (0.072, 0.098, 0.054, 1.0)  # Muted pale sage English orchard lichen

    # Cycles Ambient Occlusion deepening root flares, trunk crevices & branch forks
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.only_local = True
    ao.inputs["Distance"].default_value = 0.30
    ao_rmp = nt.nodes.new("ShaderNodeValToRGB")
    ao_rmp.color_ramp.elements[0].position = 0.14
    ao_rmp.color_ramp.elements[0].color = (0.34, 0.30, 0.27, 1.0)
    ao_rmp.color_ramp.elements[1].position = 0.92
    ao_rmp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mao_fac, mao_a, mao_b, mao_out = get_rgba_mix_sockets(mix_ao)
    mao_fac.default_value = 1.0

    bmp_fine = nt.nodes.new("ShaderNodeBump")
    bmp_fine.inputs["Strength"].default_value = 0.22
    bmp_fine.inputs["Distance"].default_value = 0.010

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.62
    bmp.inputs["Distance"].default_value = 0.028

    set_bsdf(bsdf, "Roughness", 0.90)
    set_bsdf(bsdf, "Specular IOR Level", 0.22)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns_bark.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns_ridge.inputs["Vector"])
    nt.links.new(ns_bark.outputs["Fac"], rmp_bark.inputs["Fac"])
    nt.links.new(tc.outputs["Object"], ns_lich.inputs["Vector"])
    nt.links.new(ns_lich.outputs["Fac"], lich_rmp.inputs["Fac"])
    nt.links.new(lich_rmp.outputs["Color"], ml_fac)
    nt.links.new(rmp_bark.outputs["Color"], ml_a)
    nt.links.new(ao.outputs["AO"], ao_rmp.inputs["Fac"])
    nt.links.new(ml_out, mao_a)
    nt.links.new(ao_rmp.outputs["Color"], mao_b)
    nt.links.new(mao_out, bsdf.inputs["Base Color"])
    nt.links.new(ns_ridge.outputs["Fac"], bmp_fine.inputs["Height"])
    nt.links.new(ns_bark.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp_fine.outputs["Normal"], bmp.inputs["Normal"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_translucent_leaf_mat(name: str = "WoolsthorpeSunlitCanopyLeaf"):
    """Photoreal matte-botanical apple leaf shader (ZERO plastic shine, ZERO ray-splitting Translucent BSDF!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 3.2
    ns.inputs["Detail"].default_value = 6.0
    ns.inputs["Roughness"].default_value = 0.58

    ns_vein = nt.nodes.new("ShaderNodeTexNoise")
    ns_vein.inputs["Scale"].default_value = 26.0
    ns_vein.inputs["Detail"].default_value = 5.0
    ns_vein.inputs["Roughness"].default_value = 0.65

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.18
    rmp.color_ramp.elements[0].color = (0.018, 0.082, 0.014, 1.0)  # Deep forest shadow olive-emerald
    e1 = rmp.color_ramp.elements.new(0.54)
    e1.color = (0.055, 0.175, 0.024, 1.0)  # Rich natural botanical mid emerald
    rmp.color_ramp.elements[-1].position = 0.86
    rmp.color_ramp.elements[-1].color = (0.155, 0.245, 0.036, 1.0)  # Sunlit warm golden-green autumn tips!

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.15
    bmp.inputs["Distance"].default_value = 0.004

    set_bsdf(bsdf, "Roughness", 0.56)
    set_bsdf(bsdf, "Specular IOR Level", 0.30)
    set_bsdf(bsdf, "Coat Weight", 0.04)
    set_bsdf(bsdf, "Coat Roughness", 0.42)
    set_bsdf(bsdf, "Sheen Weight", 0.12)
    set_bsdf(bsdf, "Subsurface Weight", 0.20)
    set_bsdf(bsdf, "Subsurface Scale", 0.012)
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "BURLEY"
        except Exception:
            pass

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_vein.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns_vein.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_ruby_apple_mat(name: str = "WoolsthorpeCanopyApple"):
    """Glossy ruby-gold mottled Woolsthorpe Flower-of-Kent heirloom apple skin."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.8
    ns.inputs["Detail"].default_value = 7.0
    ns.inputs["Distortion"].default_value = 0.75

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.26
    rmp.color_ramp.elements[0].color = (0.48, 0.018, 0.014, 1.0)
    e1 = rmp.color_ramp.elements.new(0.60)
    e1.color = (0.78, 0.058, 0.024, 1.0)
    rmp.color_ramp.elements[-1].position = 0.84
    rmp.color_ramp.elements[-1].color = (0.88, 0.54, 0.085, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.032
    set_bsdf(bsdf, "Roughness", 0.16)
    set_bsdf(bsdf, "Coat Weight", 0.90)
    set_bsdf(bsdf, "Coat Roughness", 0.05)
    set_bsdf(bsdf, "Subsurface Weight", 0.08)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_oak_walnut_mat(name: str = "ManorDarkWalnutOak", dark: bool = True):
    """Aged English walnut/oak timber with wood-grain roughness variation & warm wax sheen."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.6, 1.6, 5.2)

    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.5
    ns.inputs["Detail"].default_value = 9.0
    ns.inputs["Distortion"].default_value = 1.6

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.24
    rmp.color_ramp.elements[0].color = (0.032, 0.016, 0.008, 1.0) if dark else (0.088, 0.048, 0.020, 1.0)
    rmp.color_ramp.elements[1].position = 0.80
    rmp.color_ramp.elements[1].color = (0.095, 0.050, 0.022, 1.0) if dark else (0.215, 0.122, 0.052, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.14
    set_bsdf(bsdf, "Roughness", 0.44 if dark else 0.34)
    set_bsdf(bsdf, "Coat Weight", 0.22)
    set_bsdf(bsdf, "Coat Roughness", 0.18)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_simple_pbr_mat(name: str, rgb, rough: float = 0.5, metallic: float = 0.0, bump_str: float = 0.0, coat: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Roughness", rough)
    set_bsdf(bsdf, "Metallic", metallic)
    if coat > 0.0:
        set_bsdf(bsdf, "Coat Weight", coat)
    if bump_str > 0.0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        ns = nt.nodes.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = 42.0
        ns.inputs["Detail"].default_value = 6.0
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = bump_str
        nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
        nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_emissive_mat(name: str, rgb, strength: float = 4.0):
    """Returns (mat, color_socket, strength_socket) for direct animation."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat, em.inputs["Color"], em.inputs["Strength"]


def _make_crescent_moon_mat(name: str = "WoolsthorpePearlescentCrescentMoon"):
    """Soft pearlescent sculpted lunar crescent with subtle lunar maria shading & animatable fade-in."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    mix_sh.inputs["Fac"].default_value = 0.0  # Animated 0.0 -> 1.0 in Beats 5-6!
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 3.1

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 3.8
    ns.inputs["Detail"].default_value = 7.0
    ns.inputs["Roughness"].default_value = 0.62

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.28
    rmp.color_ramp.elements[0].color = (0.78, 0.84, 0.96, 1.0)  # Soft silver-blue lunar maria
    rmp.color_ramp.elements[1].position = 0.76
    rmp.color_ramp.elements[1].color = (1.00, 0.97, 0.88, 1.0)  # Luminous warm pearl highland

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], em.inputs["Color"])
    nt.links.new(tr.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(em.outputs["Emission"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, mix_sh.inputs["Fac"]


def _make_godray_mat(name: str = "WoolsthorpeGoldenGodRay"):
    """Feathered volumetric golden sunbeam shaft (LayerWeight.Facing^2.6 + vertical fade + animatable opacity)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.80, 0.40, 1.0)
    em.inputs["Strength"].default_value = 0.78

    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.50
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.6

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    fade_rmp = nt.nodes.new("ShaderNodeValToRGB")
    fade_rmp.color_ramp.elements[0].position = 0.03
    fade_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    e_mid = fade_rmp.color_ramp.elements.new(0.48)
    e_mid.color = (0.20, 0.20, 0.20, 1.0)
    fade_rmp.color_ramp.elements[-1].position = 0.97
    fade_rmp.color_ramp.elements[-1].color = (0.02, 0.02, 0.02, 1.0)

    mul_geo = nt.nodes.new("ShaderNodeMath")
    mul_geo.operation = "MULTIPLY"

    mul_anim = nt.nodes.new("ShaderNodeMath")
    mul_anim.operation = "MULTIPLY"
    mul_anim.inputs[1].default_value = 1.0  # Animated 1.0 -> 0.0 as golden hour fades to twilight!

    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs["Value"], pw.inputs[0])
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], fade_rmp.inputs["Fac"])
    nt.links.new(pw.outputs["Value"], mul_geo.inputs[0])
    nt.links.new(fade_rmp.outputs["Color"], mul_geo.inputs[1])
    nt.links.new(mul_geo.outputs["Value"], mul_anim.inputs[0])
    nt.links.new(mul_anim.outputs["Value"], mix_sh.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(em.outputs["Emission"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return mat, mul_anim.inputs[1]


def _make_antique_bronze_mat(name: str = "AgedGardenBronze"):
    """Museum-grade carved 17th-century antique bronze with warm metallic highlights & AO verdigris crevices."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 18.0
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Roughness"].default_value = 0.65

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.only_local = True
    ao.inputs["Distance"].default_value = 0.14

    rmp_pat = nt.nodes.new("ShaderNodeValToRGB")
    rmp_pat.color_ramp.elements[0].position = 0.25
    rmp_pat.color_ramp.elements[0].color = (0.14, 0.16, 0.10, 1.0)  # Subtle aged verdigris crevice patina
    e_mid = rmp_pat.color_ramp.elements.new(0.62)
    e_mid.color = (0.28, 0.19, 0.095, 1.0)  # Aged classical garden bronze
    rmp_pat.color_ramp.elements[-1].position = 0.90
    rmp_pat.color_ramp.elements[-1].color = (0.40, 0.27, 0.12, 1.0)  # Burnished laurel-ridge highlights

    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.clamp = True
    r_map.inputs["From Min"].default_value = 0.20
    r_map.inputs["From Max"].default_value = 0.85
    r_map.inputs["To Min"].default_value = 0.24
    r_map.inputs["To Max"].default_value = 0.44

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.055
    bmp.inputs["Distance"].default_value = 0.004

    set_bsdf(bsdf, "Metallic", 0.90)
    set_bsdf(bsdf, "Coat Weight", 0.12)
    set_bsdf(bsdf, "Coat Roughness", 0.22)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ao.outputs["AO"], rmp_pat.inputs["Fac"])
    nt.links.new(rmp_pat.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_wrought_iron_mat(name: str = "ManorBlackenedWroughtIron"):
    """Hand-forged 17th-century blackened wrought iron with subtle hammered patina for lantern & leaded cames."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 28.0
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Roughness"].default_value = 0.68

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.018, 0.018, 0.022, 1.0)
    rmp.color_ramp.elements[1].position = 0.82
    rmp.color_ramp.elements[1].color = (0.048, 0.044, 0.046, 1.0)

    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.clamp = True
    r_map.inputs["From Min"].default_value = 0.20
    r_map.inputs["From Max"].default_value = 0.82
    r_map.inputs["To Min"].default_value = 0.36
    r_map.inputs["To Max"].default_value = 0.56

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.075
    bmp.inputs["Distance"].default_value = 0.004

    set_bsdf(bsdf, "Metallic", 0.86)
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_english_ivy_leaf_mat(name: str = "CotswoldClimbingEnglishIvy"):
    """Rich dark-emerald matte-botanical English ivy leaf shader (ZERO Translucent BSDF!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 4.6
    ns.inputs["Detail"].default_value = 6.0
    ns.inputs["Roughness"].default_value = 0.58

    ns_vein = nt.nodes.new("ShaderNodeTexNoise")
    ns_vein.inputs["Scale"].default_value = 32.0
    ns_vein.inputs["Detail"].default_value = 5.0

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.20
    rmp.color_ramp.elements[0].color = (0.010, 0.046, 0.014, 1.0)  # Deep shadow evergreen ivy
    e1 = rmp.color_ramp.elements.new(0.56)
    e1.color = (0.024, 0.088, 0.022, 1.0)  # Classic dark-emerald English ivy
    rmp.color_ramp.elements[-1].position = 0.88
    rmp.color_ramp.elements[-1].color = (0.056, 0.132, 0.028, 1.0)  # Sun-kissed young ivy shoot tips

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.12
    bmp.inputs["Distance"].default_value = 0.003

    set_bsdf(bsdf, "Roughness", 0.48)
    set_bsdf(bsdf, "Specular IOR Level", 0.34)
    set_bsdf(bsdf, "Coat Weight", 0.10)
    set_bsdf(bsdf, "Coat Roughness", 0.36)
    set_bsdf(bsdf, "Subsurface Weight", 0.12)
    set_bsdf(bsdf, "Subsurface Scale", 0.008)
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "BURLEY"
        except Exception:
            pass

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_vein.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns_vein.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_drifting_autumn_leaf_mat(name: str = "GoldenHourDriftingOrchardLeaf"):
    """Sunlit matte gold-emerald falling orchard leaf shader (ZERO Translucent BSDF!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 8.5
    ns.inputs["Detail"].default_value = 5.0

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.24
    rmp.color_ramp.elements[0].color = (0.095, 0.185, 0.022, 1.0)  # Warm orchard olive-emerald
    e1 = rmp.color_ramp.elements.new(0.58)
    e1.color = (0.245, 0.235, 0.030, 1.0)  # Golden-hour honey-olive
    rmp.color_ramp.elements[-1].position = 0.86
    rmp.color_ramp.elements[-1].color = (0.420, 0.230, 0.032, 1.0)  # Sun-kissed amber autumn tip

    set_bsdf(bsdf, "Roughness", 0.54)
    set_bsdf(bsdf, "Specular IOR Level", 0.28)
    set_bsdf(bsdf, "Subsurface Weight", 0.16)
    set_bsdf(bsdf, "Subsurface Scale", 0.010)
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "BURLEY"
        except Exception:
            pass

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _make_candlelit_glass_mat(name: str, base_strength: float = 1.8):
    """Warm candlelit hand-blown frosted-amber crown glass with animatable Emission Strength."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 9.5
    ns.inputs["Detail"].default_value = 6.0
    ns.inputs["Distortion"].default_value = 0.65

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.95, 0.44, 0.10, 1.0)  # Deep warm beeswax-amber edge
    e_mid = rmp.color_ramp.elements.new(0.58)
    e_mid.color = (1.00, 0.66, 0.22, 1.0)  # Cozy honey-gold candlelight
    rmp.color_ramp.elements[-1].position = 0.88
    rmp.color_ramp.elements[-1].color = (1.00, 0.86, 0.48, 1.0)  # Luminous warm hearth core

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.06
    bmp.inputs["Distance"].default_value = 0.004

    set_bsdf(bsdf, "Roughness", 0.28)
    set_bsdf(bsdf, "Specular IOR Level", 0.55)
    set_bsdf(bsdf, "Coat Weight", 0.35)
    set_bsdf(bsdf, "Coat Roughness", 0.14)
    set_bsdf(bsdf, "Emission Strength", base_strength)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(rmp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat, bsdf.inputs["Emission Strength"]


# ============================================================================
# 4. BMESH PROCEDURAL GEOMETRY BUILDERS (ZERO BLOCKY PRIMITIVES!)
# ============================================================================
def _catmull_rom_sample(pts_r, subdivs: int = 3):
    """Smoothly upsample [(x, y, z, r), ...] via centripetal-safe Catmull-Rom spline passing through endpoints."""
    n = len(pts_r)
    if n < 2:
        return list(pts_r)
    out = []
    for i in range(n - 1):
        p0 = pts_r[max(0, i - 1)]
        p1 = pts_r[i]
        p2 = pts_r[i + 1]
        p3 = pts_r[min(n - 1, i + 2)]
        steps = subdivs if i < n - 2 else subdivs + 1
        for s_i in range(steps):
            t = s_i / float(subdivs)
            t2 = t * t
            t3 = t2 * t
            vals = []
            for k in range(4):
                v = 0.5 * (
                    (2.0 * p1[k])
                    + (-p0[k] + p2[k]) * t
                    + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                    + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3
                )
                vals.append(v)
            vals[3] = max(0.002, vals[3])
            out.append(tuple(vals))
    return out


def _bmesh_add_tapered_tube(
    bm: bmesh.types.BMesh,
    pts_r,
    radial_segs: int = 12,
    subdivs: int = 3,
    bark_ridge: float = 0.0,
    cap_start: bool = True,
    cap_end: bool = True,
) -> None:
    """Append a smooth parallel-transport tapered organic spline tube directly into bm."""
    samples = _catmull_rom_sample(pts_r, subdivs=subdivs)
    if len(samples) < 2:
        return

    coords = [Vector((p[0], p[1], p[2])) for p in samples]
    radii = [p[3] for p in samples]
    m = len(coords)

    # Compute tangents + parallel-transport frames (zero gimbal twist!)
    tangents = []
    for i in range(m):
        if i == 0:
            t_vec = coords[1] - coords[0]
        elif i == m - 1:
            t_vec = coords[-1] - coords[-2]
        else:
            t_vec = coords[i + 1] - coords[i - 1]
        tangents.append(t_vec.normalized() if t_vec.length > 1e-6 else Vector((0.0, 0.0, 1.0)))

    ref = Vector((1.0, 0.0, 0.0)) if abs(tangents[0].x) < 0.85 else Vector((0.0, 1.0, 0.0))
    n_vec = (ref - tangents[0] * ref.dot(tangents[0])).normalized()
    b_vec = tangents[0].cross(n_vec).normalized()

    rings = []
    arc_len = 0.0
    for i in range(m):
        if i > 0:
            arc_len += (coords[i] - coords[i - 1]).length
            t_cur = tangents[i]
            n_proj = n_vec - t_cur * n_vec.dot(t_cur)
            if n_proj.length < 1e-6:
                alt = Vector((1.0, 0.0, 0.0)) if abs(t_cur.x) < 0.85 else Vector((0.0, 1.0, 0.0))
                n_vec = (alt - t_cur * alt.dot(t_cur)).normalized()
            else:
                n_vec = n_proj.normalized()
            b_vec = t_cur.cross(n_vec).normalized()

        c = coords[i]
        r = radii[i]
        ring = []
        for s_i in range(radial_segs):
            ang = (s_i / float(radial_segs)) * math.pi * 2.0
            ridge = 1.0 + bark_ridge * math.sin(5.0 * ang + arc_len * 3.8)
            rr = r * ridge
            pt = c + (n_vec * math.cos(ang) + b_vec * math.sin(ang)) * rr
            ring.append(bm.verts.new(pt))
        rings.append(ring)

    for i in range(m - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for s_i in range(radial_segs):
            sn = (s_i + 1) % radial_segs
            bm.faces.new((r0[s_i], r0[sn], r1[sn], r1[s_i]))

    if cap_start:
        bm.faces.new(list(reversed(rings[0])))
    if cap_end:
        bm.faces.new(rings[-1])


def _bmesh_add_lathe(
    bm: bmesh.types.BMesh,
    profile_rz,
    center: Vector,
    segs: int = 16,
    angle_offset: float = 0.0,
    cap_bot: bool = True,
    cap_top: bool = True,
    flute_count: int = 0,
    flute_depth: float = 0.0,
    flute_z_range=None,
) -> None:
    """Append a revolved classical stone urn/column/plinth profile into bm."""
    rings = []
    for r_raw, z_local in profile_rz:
        r = max(1e-4, r_raw)
        in_flute = (
            flute_count > 0
            and flute_depth > 0.0
            and flute_z_range is not None
            and (flute_z_range[0] <= z_local <= flute_z_range[1])
        )
        ring = []
        for s_i in range(segs):
            ang = angle_offset + (s_i / float(segs)) * math.pi * 2.0
            rr = r - (flute_depth * max(0.0, math.cos(flute_count * ang)) if in_flute else 0.0)
            ring.append(bm.verts.new((center.x + rr * math.cos(ang), center.y + rr * math.sin(ang), center.z + z_local)))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for s_i in range(segs):
            sn = (s_i + 1) % segs
            bm.faces.new((r0[s_i], r0[sn], r1[sn], r1[s_i]))

    if cap_bot:
        bm.faces.new(list(reversed(rings[0])))
    if cap_top:
        bm.faces.new(rings[-1])


def _bmesh_add_box(bm: bmesh.types.BMesh, center, size, bevel_w: float = 0.0) -> None:
    """Append a clean axis-aligned (optionally beveled) box directly in world coordinates."""
    cx, cy, cz = center
    sx, sy, sz = size
    mat = Matrix.Translation((cx, cy, cz)) @ Matrix.Diagonal((sx, sy, sz, 1.0))
    ret = bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    if bevel_w > 0.0:
        verts = ret["verts"]
        vert_set = set(verts)
        edges = [e for e in bm.edges if e.verts[0] in vert_set and e.verts[1] in vert_set]
        try:
            bmesh.ops.bevel(
                bm,
                geom=edges,
                offset=bevel_w,
                offset_type="OFFSET",
                segments=2,
                profile=0.5,
                affect="EDGES",
                clamp_overlap=True,
            )
        except Exception:
            pass


def _bmesh_add_uv_sphere(
    bm: bmesh.types.BMesh,
    center: Vector,
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 1.0,
    segs: int = 12,
    rings: int = 8,
    ruffle_amp: float = 0.0,
    ruffle_freq: float = 4.5,
) -> None:
    """Pure-bmesh closed quad/tri ellipsoid with optional organic botanical surface ruffle (Blender 3.x/4.x safe)."""
    v_bot = bm.verts.new((center.x, center.y, center.z - rz))
    v_top = bm.verts.new((center.x, center.y, center.z + rz))
    lat_rings = []
    for ri in range(1, rings):
        phi = -math.pi * 0.5 + (ri / float(rings)) * math.pi
        cp, sp = math.cos(phi), math.sin(phi)
        ring = []
        for si in range(segs):
            th = (si / float(segs)) * math.pi * 2.0
            dx = cp * math.cos(th) * rx
            dy = cp * math.sin(th) * ry
            dz = sp * rz
            px, py, pz = center.x + dx, center.y + dy, center.z + dz
            if ruffle_amp > 0.0:
                rf = 1.0 + ruffle_amp * math.sin(px * ruffle_freq + pz * (ruffle_freq * 0.85)) * math.cos(
                    py * (ruffle_freq * 0.92) - px * (ruffle_freq * 0.55)
                )
                px, py, pz = center.x + dx * rf, center.y + dy * rf, center.z + dz * rf
            ring.append(bm.verts.new((px, py, pz)))
        lat_rings.append(ring)

    # Bottom cap
    r0 = lat_rings[0]
    for si in range(segs):
        sn = (si + 1) % segs
        bm.faces.new((v_bot, r0[sn], r0[si]))
    # Latitude quads
    for ri in range(len(lat_rings) - 1):
        ra = lat_rings[ri]
        rb = lat_rings[ri + 1]
        for si in range(segs):
            sn = (si + 1) % segs
            bm.faces.new((ra[si], ra[sn], rb[sn], rb[si]))
    # Top cap
    rl = lat_rings[-1]
    for si in range(segs):
        sn = (si + 1) % segs
        bm.faces.new((rl[si], rl[sn], v_top))


def _bmesh_add_leaf_card(
    bm: bmesh.types.BMesh,
    origin: Vector,
    rot_euler: Euler,
    length: float = 0.085,
    width: float = 0.046,
) -> None:
    """Append a sculpted curved 6-vert diamond-ovate botanical apple leaf (3 faces / 4 tris)."""
    rmat = rot_euler.to_matrix()

    def xf(lx, ly, lz):
        return origin + (rmat @ Vector((lx, ly, lz)))

    v0 = bm.verts.new(xf(0.0, 0.0, 0.0))
    v1l = bm.verts.new(xf(0.35 * length, +0.48 * width, +0.055 * length))
    v1r = bm.verts.new(xf(0.35 * length, -0.48 * width, +0.055 * length))
    v2l = bm.verts.new(xf(0.72 * length, +0.34 * width, +0.015 * length))
    v2r = bm.verts.new(xf(0.72 * length, -0.34 * width, +0.015 * length))
    v3 = bm.verts.new(xf(length, 0.0, -0.095 * length))

    bm.faces.new((v0, v1r, v1l))
    bm.faces.new((v1l, v1r, v2r, v2l))
    bm.faces.new((v2l, v2r, v3))


def _bmesh_add_ivy_leaf(
    bm: bmesh.types.BMesh,
    origin: Vector,
    rot_euler: Euler,
    size: float = 0.058,
    mat_index: int = 1,
) -> None:
    """Append a sculpted curved 7-vert 3-lobed English ivy leaf (4 faces, natural heart-ovate silhouette)."""
    rmat = rot_euler.to_matrix()

    def xf(lx: float, ly: float, lz: float) -> Vector:
        return origin + (rmat @ Vector((lx * size, ly * size, lz * size)))

    v_base = bm.verts.new(xf(0.0, 0.0, 0.0))
    v_mid = bm.verts.new(xf(0.46, 0.0, 0.045))
    v_ll = bm.verts.new(xf(0.28, +0.46, 0.025))
    v_lr = bm.verts.new(xf(0.28, -0.46, 0.025))
    v_ul = bm.verts.new(xf(0.68, +0.34, -0.010))
    v_ur = bm.verts.new(xf(0.68, -0.34, -0.010))
    v_tip = bm.verts.new(xf(1.00, 0.0, -0.085))

    for verts in (
        (v_base, v_mid, v_ll),
        (v_base, v_lr, v_mid),
        (v_ll, v_mid, v_tip, v_ul),
        (v_mid, v_lr, v_ur, v_tip),
    ):
        face = bm.faces.new(verts)
        face.material_index = mat_index


def _bmesh_add_heirloom_apple(
    bm: bmesh.types.BMesh,
    center: Vector,
    radius: float = 0.044,
    rot_euler: Euler | None = None,
    mat_index: int = 0,
) -> None:
    """Append a sculpted 5-lobed dimpled heirloom Woolsthorpe apple into bm."""
    rmat = rot_euler.to_matrix() if rot_euler is not None else None
    n_seg = 16
    n_ring = 10
    rings = []
    for i in range(n_ring + 1):
        u = i / float(n_ring)
        th = u * math.pi
        s = math.sin(th)
        c = -math.cos(th)
        shoulder = 1.0 + 0.14 * math.sin(u * math.pi) * (0.4 + 0.6 * u) - 0.10 * (1.0 - u)
        dimple = (0.16 - u) * 1.15 if u < 0.16 else (-(u - 0.82) * 1.60 if u > 0.82 else 0.0)
        rr = max(0.002, s * shoulder) * radius
        zz = (c * 0.94 + dimple) * radius
        ring = []
        for s_i in range(n_seg):
            ang = (s_i / float(n_seg)) * math.pi * 2.0
            lobe = 1.0 + 0.032 * math.cos(5.0 * ang) * math.sin(u * math.pi)
            local_pt = Vector((math.cos(ang) * rr * lobe, math.sin(ang) * rr * lobe, zz))
            world_pt = center + (rmat @ local_pt if rmat is not None else local_pt)
            ring.append(bm.verts.new(world_pt))
        rings.append(ring)

    for i in range(n_ring):
        for s_i in range(n_seg):
            sn = (s_i + 1) % n_seg
            face = bm.faces.new((rings[i][s_i], rings[i][sn], rings[i + 1][sn], rings[i + 1][s_i]))
            face.material_index = mat_index


# ============================================================================
# 5. SECTION BUILDERS (SKY, COSMOS, MEADOW, BALUSTRADE, TREE, LOGGIA, PLINTHS)
# ============================================================================
def _build_animated_physical_sky(scene):
    """Build Cycles World blending NISHITA Golden-Hour Sunset (Beats 1-4) -> Deep-Sapphire & Starlight Milky Way Nebula Veil (Beats 5-6)."""
    world = bpy.data.worlds.new("WoolsthorpeOrchardCosmosWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    mix_sh.inputs["Fac"].default_value = 0.0  # Golden-hour sunset at frame 1!

    # Branch A: Calibrated Physical NISHITA Golden-Hour Amber-to-Sapphire Sunset (Beats 1-4)
    sky_tex = nt.nodes.new("ShaderNodeTexSky")
    sky_tex.sky_type = "NISHITA"
    sky_tex.sun_disc = False
    sky_tex.sun_elevation = math.radians(5.8)
    sky_tex.sun_rotation = math.radians(48.0)
    sky_tex.air_density = 1.9
    sky_tex.dust_density = 1.15
    sky_tex.ozone_density = 3.8

    bg_day = nt.nodes.new("ShaderNodeBackground")
    bg_day.inputs["Strength"].default_value = 0.24

    # Branch B: Deep-Sapphire Celestial Twilight + Subtle Diagonal Milky Way Starlight Nebula Veil (Beats 5-6)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    map_z = nt.nodes.new("ShaderNodeMapRange")
    map_z.clamp = True
    map_z.inputs["From Min"].default_value = 0.0
    map_z.inputs["From Max"].default_value = 0.68
    map_z.inputs["To Min"].default_value = 0.0
    map_z.inputs["To Max"].default_value = 1.0

    night_rmp = nt.nodes.new("ShaderNodeValToRGB")
    cr = night_rmp.color_ramp
    cr.elements[0].position = 0.00
    cr.elements[0].color = (0.012, 0.024, 0.068, 1.0)  # Deep twilight indigo horizon
    e_mid = cr.elements.new(0.32)
    e_mid.color = (0.006, 0.014, 0.044, 1.0)  # Midnight celestial sapphire
    cr.elements[-1].position = 0.96
    cr.elements[-1].color = (0.002, 0.006, 0.022, 1.0)  # Deep velvety cosmos zenith

    # V9 Diagonal Milky Way Nebula Band: tilted smoothly across Beat 5-6 sky behind Crescent Moon & Gravity Well!
    mp_mw = nt.nodes.new("ShaderNodeMapping")
    mp_mw.inputs["Rotation"].default_value = (math.radians(-18.0), math.radians(32.0), math.radians(-8.0))

    sep_mw = nt.nodes.new("ShaderNodeSeparateXYZ")
    mw_abs = nt.nodes.new("ShaderNodeMath")
    mw_abs.operation = "ABSOLUTE"

    band_env = nt.nodes.new("ShaderNodeMapRange")
    band_env.clamp = True
    band_env.interpolation_type = "SMOOTHSTEP"
    band_env.inputs["From Min"].default_value = 0.0
    band_env.inputs["From Max"].default_value = 0.26
    band_env.inputs["To Min"].default_value = 1.0
    band_env.inputs["To Max"].default_value = 0.0

    horizon_fade = nt.nodes.new("ShaderNodeMapRange")
    horizon_fade.clamp = True
    horizon_fade.interpolation_type = "SMOOTHSTEP"
    horizon_fade.inputs["From Min"].default_value = 0.05
    horizon_fade.inputs["From Max"].default_value = 0.22
    horizon_fade.inputs["To Min"].default_value = 0.0
    horizon_fade.inputs["To Max"].default_value = 1.0

    mp_mw_stretch = nt.nodes.new("ShaderNodeMapping")
    mp_mw_stretch.inputs["Scale"].default_value = (2.6, 2.6, 6.2)

    ns_neb_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_neb_macro.inputs["Scale"].default_value = 2.4
    ns_neb_macro.inputs["Detail"].default_value = 8.0
    ns_neb_macro.inputs["Roughness"].default_value = 0.62
    ns_neb_macro.inputs["Distortion"].default_value = 0.65

    ns_neb_fine = nt.nodes.new("ShaderNodeTexNoise")
    ns_neb_fine.inputs["Scale"].default_value = 8.5
    ns_neb_fine.inputs["Detail"].default_value = 10.0
    ns_neb_fine.inputs["Roughness"].default_value = 0.68
    ns_neb_fine.inputs["Distortion"].default_value = 0.30

    neb_cloud = nt.nodes.new("ShaderNodeMath")
    neb_cloud.operation = "MULTIPLY"

    neb_band_mask = nt.nodes.new("ShaderNodeMath")
    neb_band_mask.operation = "MULTIPLY"

    neb_density = nt.nodes.new("ShaderNodeMath")
    neb_density.operation = "MULTIPLY"

    nebula_rmp = nt.nodes.new("ShaderNodeValToRGB")
    ncr = nebula_rmp.color_ramp
    ncr.elements[0].position = 0.05
    ncr.elements[0].color = (0.000, 0.000, 0.000, 1.0)  # Clean dark cosmic rift
    ne_mid = ncr.elements.new(0.20)
    ne_mid.color = (0.018, 0.038, 0.095, 1.0)  # Subtle sapphire-violet Milky Way veil
    ncr.elements[-1].position = 0.42
    ncr.elements[-1].color = (0.045, 0.068, 0.135, 1.0)  # Faint golden-white star-cloud core

    mix_neb = nt.nodes.new("ShaderNodeMix")
    mix_neb.data_type = "RGBA"
    mix_neb.blend_type = "ADD"
    mn_fac, mn_a, mn_b, mn_out = get_rgba_mix_sockets(mix_neb)
    mn_fac.default_value = 0.78  # Gentle IMAX contrast so stars, Moon, Gravity Well & title pop!

    bg_night = nt.nodes.new("ShaderNodeBackground")
    bg_night.inputs["Strength"].default_value = 0.90

    nt.links.new(sky_tex.outputs["Color"], bg_day.inputs["Color"])
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], map_z.inputs["Value"])
    nt.links.new(map_z.outputs["Result"], night_rmp.inputs["Fac"])

    nt.links.new(tc.outputs["Generated"], mp_mw.inputs["Vector"])
    nt.links.new(mp_mw.outputs["Vector"], sep_mw.inputs["Vector"])
    nt.links.new(sep_mw.outputs["Z"], mw_abs.inputs[0])
    nt.links.new(mw_abs.outputs["Value"], band_env.inputs["Value"])
    nt.links.new(sep.outputs["Z"], horizon_fade.inputs["Value"])
    nt.links.new(mp_mw.outputs["Vector"], mp_mw_stretch.inputs["Vector"])
    nt.links.new(mp_mw_stretch.outputs["Vector"], ns_neb_macro.inputs["Vector"])
    nt.links.new(mp_mw_stretch.outputs["Vector"], ns_neb_fine.inputs["Vector"])
    nt.links.new(ns_neb_macro.outputs["Fac"], neb_cloud.inputs[0])
    nt.links.new(ns_neb_fine.outputs["Fac"], neb_cloud.inputs[1])
    nt.links.new(band_env.outputs["Result"], neb_band_mask.inputs[0])
    nt.links.new(horizon_fade.outputs["Result"], neb_band_mask.inputs[1])
    nt.links.new(neb_cloud.outputs["Value"], neb_density.inputs[0])
    nt.links.new(neb_band_mask.outputs["Value"], neb_density.inputs[1])
    nt.links.new(neb_density.outputs["Value"], nebula_rmp.inputs["Fac"])
    nt.links.new(night_rmp.outputs["Color"], mn_a)
    nt.links.new(nebula_rmp.outputs["Color"], mn_b)
    nt.links.new(mn_out, bg_night.inputs["Color"])

    nt.links.new(bg_day.outputs["Background"], mix_sh.inputs[1])
    nt.links.new(bg_night.outputs["Background"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])

    return world, mix_sh.inputs["Fac"]


def _build_starfield_and_crescent_moon(scene, star_gold_mat, star_sapphire_mat, moon_mat):
    """Build 160 true 3D spherical random stars (random.Random(1687)) + sculpted 3D Crescent Moon (ZERO black ball!)."""
    # Shared tiny unit-radius icosphere mesh datablocks for memory-light 160-star dome
    star_meshes = []
    for m_name, mat in (("SkyStarGold_Mesh", star_gold_mat), ("SkyStarSapphire_Mesh", star_sapphire_mat)):
        bm = bmesh.new()
        _bmesh_add_uv_sphere(bm, Vector((0.0, 0.0, 0.0)), 1.0, 1.0, 1.0, segs=8, rings=5)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh = bpy.data.meshes.new(m_name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.materials.append(mat)
        star_meshes.append(mesh)

    rng = random.Random(1687)  # Year Principia Mathematica was published!
    moon_dir = CRESCENT_MOON_POS.normalized()
    sin_phi_min = math.sin(math.radians(12.0))
    sin_phi_max = math.sin(math.radians(62.0))
    radius_dome = 22.0

    star_entries = []
    attempts = 0
    while len(star_entries) < 160 and attempts < 800:
        attempts += 1
        theta = math.radians(rng.uniform(-75.0, 75.0))
        phi = math.asin(rng.uniform(sin_phi_min, sin_phi_max))
        sx = radius_dome * math.cos(phi) * math.sin(theta)
        sy = radius_dome * math.cos(phi) * math.cos(theta)
        sz = radius_dome * math.sin(phi)
        s_vec = Vector((sx, sy, sz))
        # Keep a clean breathing halo around the Sculpted Crescent Moon
        if s_vec.normalized().dot(moon_dir) > math.cos(math.radians(5.2)):
            continue

        # Natural astronomical magnitude distribution (varied 0.035..0.085m)
        base_r = 0.035 + (rng.random() ** 1.75) * 0.050
        mesh_idx = 0 if rng.random() < 0.56 else 1
        idx = len(star_entries)
        obj = bpy.data.objects.new(f"CosmosStar_{idx:03d}", star_meshes[mesh_idx])
        obj.location = s_vec
        obj.scale = (0.0001, 0.0001, 0.0001)  # Hidden cleanly in daylight (Beats 1-4)
        obj.visible_shadow = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        _link_obj(scene, obj)

        freq = rng.uniform(2.2, 5.4)
        phase = rng.uniform(0.0, math.pi * 2.0)
        star_entries.append((obj, base_r, freq, phase))

    # TRUE 3D SCULPTED CRESCENT MOON via bmesh (outer semicircle + inner concave terminator arc — ZERO occluder!)
    bm_m = bmesh.new()
    moon_r = 0.72
    inner_k = 0.54  # True spherical terminator ellipse ratio -> crisp romantic lunar crescent!
    n_arc = 44
    roll_rad = math.radians(-22.0)
    c_r, s_r = math.cos(roll_rad), math.sin(roll_rad)

    def moon_pt(lx: float, ly: float, lz: float) -> Vector:
        rx = lx * c_r - lz * s_r
        rz = lx * s_r + lz * c_r
        return Vector((rx, ly, rz))

    v_bot = bm_m.verts.new(moon_pt(0.0, 0.0, -moon_r))
    v_top = bm_m.verts.new(moon_pt(0.0, 0.0, +moon_r))
    rings = []
    for i in range(1, n_arc):
        u = i / float(n_arc)
        phi = -math.pi * 0.5 + u * math.pi
        cp = math.cos(phi)
        sp = math.sin(phi)
        x_out = moon_r * cp
        x_in = moon_r * inner_k * cp
        x_mid = 0.5 * (x_out + x_in)
        z_p = moon_r * sp
        y_thick = 0.14 * moon_r * cp

        v_out = bm_m.verts.new(moon_pt(x_out, 0.0, z_p))
        v_frt = bm_m.verts.new(moon_pt(x_mid, -y_thick, z_p))
        v_inn = bm_m.verts.new(moon_pt(x_in, 0.0, z_p))
        v_bck = bm_m.verts.new(moon_pt(x_mid, +y_thick, z_p))
        rings.append((v_out, v_frt, v_inn, v_bck))

    # Bottom cusp triangles
    r_first = rings[0]
    for j in range(4):
        jn = (j + 1) % 4
        bm_m.faces.new((v_bot, r_first[jn], r_first[j]))
    # Body quads
    for i in range(len(rings) - 1):
        ra = rings[i]
        rb = rings[i + 1]
        for j in range(4):
            jn = (j + 1) % 4
            bm_m.faces.new((ra[j], ra[jn], rb[jn], rb[j]))
    # Top cusp triangles
    r_last = rings[-1]
    for j in range(4):
        jn = (j + 1) % 4
        bm_m.faces.new((r_last[j], r_last[jn], v_top))

    moon_obj = _finalize_bmesh(
        scene,
        "WoolsthorpeCrescentMoon3D",
        bm_m,
        mats=moon_mat,
        loc=CRESCENT_MOON_POS,
        rot=(0.0, 0.0, math.radians(-5.0)),
        smooth=True,
    )
    moon_obj.scale = (0.0001, 0.0001, 0.0001)
    moon_obj.visible_shadow = False
    moon_obj.visible_diffuse = False
    moon_obj.visible_glossy = False

    return star_entries, moon_obj


def _build_meadow_terrace_and_hills(
    scene,
    meadow_mat,
    flagstone_mat,
    stone_mat,
    hill_mid_mat,
    hill_far_mat,
    distant_tree_mat,
    grass_blade_mat,
    wf_gold_mat,
    wf_white_mat,
    soil_mat,
):
    """Build rolling Woolsthorpe meadow terrain, curved mossy flagstone courtyard (Z=0.0), distant hills & 140 grass/wildflower clumps."""
    # 1. Undulating Meadow Grass Terrain (22m x 18m: X in [-11, +11], Y in [-4.5, +13.5])
    bm_m = bmesh.new()
    nx, ny = 68, 54
    x0, x1 = -11.0, 11.0
    y0, y1 = -4.5, 13.5
    grid = []
    for iy in range(ny + 1):
        vy = lerp(y0, y1, iy / float(ny))
        row = []
        for ix in range(nx + 1):
            vx = lerp(x0, x1, ix / float(nx))
            # Smoothly recess right beneath the flagstone terrace so zero grass pokes through stone paving
            in_court_x = smoothstep(-4.35, -3.70, vx) * (1.0 - smoothstep(3.75, 4.40, vx))
            in_court_y = smoothstep(-2.05, -1.42, vy) * (1.0 - smoothstep(1.85, 2.35, vy))
            court_mask = in_court_x * in_court_y

            # Gentle natural meadow undulation around terrace -> lush English valley -> rolling green pastures behind Y > 2.2
            lawn_wave = 0.032 * math.sin(vx * 0.55 + 0.4) * math.cos(vy * 0.62) + 0.018 * math.cos(vx * 1.15 - vy * 0.85)
            hill_rise = 0.0
            if vy > 2.0:
                hy = vy - 2.0
                # Soft valley dip behind balustrade (Y=2..5.5), then rolling countryside hills (Y=5.5..13.5)
                valley = -0.32 * math.sin(clamp01(hy / 4.2) * math.pi)
                rolling = smoothstep(3.2, 11.5, hy) * (
                    1.05
                    + 0.42 * math.sin(vx * 0.28 + 0.6)
                    + 0.26 * math.cos(vx * 0.52 - hy * 0.25)
                    - 0.28 * math.exp(-((vx - 0.6) / 3.8) ** 2)  # Gentle open valley saddle in center sightline!
                )
                hill_rise = valley + rolling

            vz = lerp(-0.018 + lawn_wave + hill_rise, -0.055, court_mask)
            row.append(bm_m.verts.new((vx, vy, vz)))
        grid.append(row)

    for iy in range(ny):
        for ix in range(nx):
            bm_m.faces.new((grid[iy][ix], grid[iy][ix + 1], grid[iy + 1][ix + 1], grid[iy + 1][ix]))

    meadow_obj = _finalize_bmesh(scene, "WoolsthorpeMeadowTerrain", bm_m, mats=meadow_mat, smooth=True)

    # 2. Curved Mossy Cotswold Limestone Flagstone Courtyard (EXACT Z_top = 0.000, X in [-3.75, +3.85], Y in [-1.45, +1.88])
    bm_f = bmesh.new()
    seg_x = 32
    y_back = 1.90
    z_top = FLOOR_Z
    z_bot = -0.16

    top_front, top_back = [], []
    bot_front, bot_back = [], []
    for i in range(seg_x + 1):
        u = i / float(seg_x)
        fx = lerp(-3.78, 3.82, u)
        # Graceful bowed English garden terrace curve along the southern front edge!
        fy_front = -1.42 - 0.16 * math.sin(u * math.pi)
        top_front.append(bm_f.verts.new((fx, fy_front, z_top)))
        top_back.append(bm_f.verts.new((fx, y_back, z_top)))
        bot_front.append(bm_f.verts.new((fx, fy_front - 0.03, z_bot)))
        bot_back.append(bm_f.verts.new((fx, y_back + 0.03, z_bot)))

    for i in range(seg_x):
        bm_f.faces.new((top_front[i], top_front[i + 1], top_back[i + 1], top_back[i]))
        bm_f.faces.new((bot_front[i + 1], bot_front[i], top_front[i], top_front[i + 1]))
        bm_f.faces.new((top_back[i], top_back[i + 1], bot_back[i + 1], bot_back[i]))
    bm_f.faces.new((bot_back[0], bot_front[0], top_front[0], top_back[0]))
    bm_f.faces.new((top_back[-1], top_front[-1], bot_front[-1], bot_back[-1]))

    terrace_obj = _finalize_bmesh(
        scene,
        "CotswoldFlagstoneCourtyard",
        bm_f,
        mats=flagstone_mat,
        smooth=True,
        edge_split_deg=35.0,
    )

    # Carved Cotswold curved garden edging curb along the south front of the flagstones + mossy tree-well surround
    bm_curb = bmesh.new()
    # Tree-well circular carved stone border around TREE_BASE (-0.18, 1.35, 0.0) — clear of roots & 3 fallen apples!
    _bmesh_add_lathe(
        bm_curb,
        [(0.58, -0.04), (0.58, 0.038), (0.55, 0.056), (0.50, 0.056), (0.48, 0.010)],
        center=TREE_BASE,
        segs=28,
        cap_bot=False,
        cap_top=False,
    )
    _finalize_bmesh(scene, "AppleTreeStoneWell", bm_curb, mats=stone_mat, smooth=True, edge_split_deg=38.0)

    # Rich dark mossy earth mound inside the tree well so gnarled roots & fallen apples rest cleanly on real soil
    bm_soil = bmesh.new()
    _bmesh_add_lathe(
        bm_soil,
        [(0.50, -0.02), (0.46, 0.005), (0.35, 0.010), (0.24, 0.048), (0.04, 0.085)],
        center=TREE_BASE,
        segs=24,
        cap_bot=False,
        cap_top=True,
    )
    _finalize_bmesh(scene, "AppleTreeSoilMound", bm_soil, mats=soil_mat, smooth=True)

    # 3. Distant Rolling English Countryside Hills (3 Layered Atmospheric Ridges at Y = +11..+23m — 100% OPEN HORIZON!)
    ridge_specs = [
        ("DistantHillRidge_Near", 11.8, 16.2, 1.05, 0.72, 0.25, 0.5, meadow_mat),
        ("DistantHillRidge_Mid", 15.6, 19.6, 1.75, 0.92, 0.21, 1.8, hill_mid_mat),
        ("DistantHillRidge_Far", 18.8, 23.2, 2.35, 1.15, 0.17, 3.1, hill_far_mat),
    ]
    hill_objs = []
    for h_name, y_near, y_far, base_h, amp_h, freq_x, phase, h_mat in ridge_specs:
        bm_h = bmesh.new()
        h_nx, h_ny = 68, 12
        h_grid = []
        for iy in range(h_ny + 1):
            uy = iy / float(h_ny)
            hy = lerp(y_near, y_far, uy)
            ridge_env = math.sin(uy * math.pi)
            crest_focus = ridge_env * ridge_env
            row = []
            for ix in range(h_nx + 1):
                ux = ix / float(h_nx)
                hx = lerp(-18.5, 18.5, ux)
                # Keep a graceful scenic low dip in the central sky corridor (X in [-1.5, +3.0]) so moon & stars shine!
                center_saddle = 1.0 - 0.34 * math.exp(-((hx - 0.8) / 4.8) ** 2)
                # Subtle multi-frequency tree-line silhouette noise breaking smooth humps into realistic wooded hills
                treeline_rip = (
                    0.16 * math.sin(hx * 0.95 + phase * 1.3)
                    + 0.095 * math.cos(hx * 2.15 - phase * 0.9)
                    + 0.055 * math.sin(hx * 4.35 + phase * 2.1)
                ) * crest_focus
                crest = (
                    base_h
                    + amp_h * math.sin(hx * freq_x + phase)
                    + 0.38 * amp_h * math.cos(hx * freq_x * 1.9 - phase * 0.7)
                    + treeline_rip
                ) * center_saddle
                hz = -0.35 + ridge_env * max(0.35, crest)
                row.append(bm_h.verts.new((hx, hy, hz)))
            h_grid.append(row)
        for iy in range(h_ny):
            for ix in range(h_nx):
                bm_h.faces.new((h_grid[iy][ix], h_grid[iy][ix + 1], h_grid[iy + 1][ix + 1], h_grid[iy + 1][ix]))
        hill_objs.append(_finalize_bmesh(scene, h_name, bm_h, mats=h_mat, smooth=True))

    # Distant English Hedgerow & Copse Tree Clusters along the hills (merged single bmesh, zero shadow BVH overhead)
    bm_dt = bmesh.new()
    rng_dt = random.Random(1642)  # Newton's birth year at Woolsthorpe!
    copse_centers = [
        (-8.5, 12.6, 1.35), (-6.8, 13.0, 1.48), (-5.2, 12.2, 1.15), (-3.8, 13.6, 1.40),
        (4.4, 12.8, 1.22), (5.8, 13.4, 1.45), (7.6, 12.5, 1.30), (9.4, 13.8, 1.55),
        (-10.5, 16.8, 2.25), (-7.4, 17.2, 2.15), (-4.2, 17.6, 1.85),
        (5.0, 17.2, 2.05), (8.2, 16.8, 2.30), (11.0, 17.5, 2.40),
    ]
    for cx, cy, cz in copse_centers:
        for _ in range(4):
            ox = cx + rng_dt.uniform(-0.95, 0.95)
            oy = cy + rng_dt.uniform(-0.55, 0.55)
            rad = rng_dt.uniform(0.36, 0.78)
            oz = cz + rad * 0.45
            _bmesh_add_uv_sphere(
                bm_dt,
                Vector((ox, oy, oz)),
                rx=rad * 1.12,
                ry=rad * 0.85,
                rz=rad * 0.96,
                segs=10,
                rings=7,
                ruffle_amp=0.22,
                ruffle_freq=4.2,
            )
    distant_copses_obj = _finalize_bmesh(scene, "DistantLincolnshireCopses", bm_dt, mats=distant_tree_mat, smooth=True)
    distant_copses_obj.visible_shadow = False

    # 4. 140 Scattered 3D Meadow Grass Clumps & Tiny Golden/White Wildflowers (2 Merged bmesh Meshes!)
    bm_gr = bmesh.new()
    bm_wf = bmesh.new()
    rng_g = random.Random(1665)

    def _clump_clear_of_anchors(gx: float, gy: float) -> bool:
        # Keep flat stone walkways around prism tripod, orbit tripod, and character standing zone clean
        if (gx - PRISM_ANCHOR.x) ** 2 + (gy - PRISM_ANCHOR.y) ** 2 < 0.45 ** 2:
            return False
        if (gx - ORBIT_ANCHOR.x) ** 2 + (gy - ORBIT_ANCHOR.y) ** 2 < 0.45 ** 2:
            return False
        if (gx - PRINCIPIA_PLINTH.x) ** 2 + (gy - PRINCIPIA_PLINTH.y) ** 2 < 0.30 ** 2:
            return False
        return True

    clump_positions = []
    # Zone A: Southern curved edge of flagstones (42 clumps)
    for _ in range(42):
        u = rng_g.random()
        gx = lerp(-3.65, 3.70, u)
        gy = -1.42 - 0.16 * math.sin(u * math.pi) - rng_g.uniform(0.02, 0.45)
        clump_positions.append((gx, gy, -0.015))
    # Zone B: Hugging the stone balustrade base (38 clumps)
    for _ in range(38):
        gx = rng_g.uniform(-1.75, 3.58)
        gy = 1.85 + rng_g.choice([-1.0, 1.0]) * rng_g.uniform(0.11, 0.28)
        clump_positions.append((gx, gy, 0.0 if gy < 1.85 else -0.015))
    # Zone C: Around the mossy Woolsthorpe Apple Tree roots (24 clumps)
    for _ in range(24):
        ang = rng_g.uniform(0.0, math.pi * 2.0)
        rr = rng_g.uniform(0.36, 0.68)
        gx = TREE_BASE.x + math.cos(ang) * rr
        gy = TREE_BASE.y + math.sin(ang) * rr
        clump_positions.append((gx, gy, 0.005 if rr < 0.46 else 0.0))
    # Zone D: Flanking the outer meadow east/west edges & beyond balustrade (36 clumps)
    for _ in range(36):
        if rng_g.random() < 0.55:
            gx = rng_g.uniform(-4.5, 5.2)
            gy = rng_g.uniform(2.15, 4.4)
            gz = -0.04 - 0.12 * clamp01((gy - 2.15) / 2.5)
        else:
            gx = rng_g.choice([-1.0, 1.0]) * rng_g.uniform(3.85, 5.4)
            gy = rng_g.uniform(-1.6, 2.2)
            gz = -0.02
        if _clump_clear_of_anchors(gx, gy):
            clump_positions.append((gx, gy, gz))

    for c_idx, (gx, gy, gz) in enumerate(clump_positions[:140]):
        n_blades = rng_g.randint(5, 7)
        for b_i in range(n_blades):
            b_ang = (b_i / float(n_blades)) * math.pi * 2.0 + rng_g.uniform(-0.35, 0.35)
            h_b = rng_g.uniform(0.085, 0.21)
            w_b = rng_g.uniform(0.008, 0.015)
            lean = rng_g.uniform(0.035, 0.095)
            ca, sa = math.cos(b_ang), math.sin(b_ang)
            px_v, py_v = -sa * w_b, ca * w_b
            ox = gx + ca * rng_g.uniform(0.0, 0.03)
            oy = gy + sa * rng_g.uniform(0.0, 0.03)

            vb0 = bm_gr.verts.new((ox - px_v, oy - py_v, gz - 0.01))
            vb1 = bm_gr.verts.new((ox + px_v, oy + py_v, gz - 0.01))
            vm0 = bm_gr.verts.new((ox + ca * lean * 0.45 - px_v * 0.55, oy + sa * lean * 0.45 - py_v * 0.55, gz + h_b * 0.56))
            vm1 = bm_gr.verts.new((ox + ca * lean * 0.45 + px_v * 0.55, oy + sa * lean * 0.45 + py_v * 0.55, gz + h_b * 0.56))
            vtip = bm_gr.verts.new((ox + ca * lean, oy + sa * lean, gz + h_b))
            bm_gr.faces.new((vb0, vb1, vm1, vm0))
            bm_gr.faces.new((vm0, vm1, vtip))

        # Add delicate golden buttercup / ivory English daisy blooms to ~65 clumps
        if c_idx % 2 == 0:
            mat_slot = 0 if (c_idx % 4 == 0) else 1
            for _ in range(rng_g.randint(1, 2)):
                fx = gx + rng_g.uniform(-0.06, 0.06)
                fy = gy + rng_g.uniform(-0.06, 0.06)
                fh = rng_g.uniform(0.11, 0.22)
                fr = rng_g.uniform(0.014, 0.024)
                fc = bm_wf.verts.new((fx, fy, gz + fh))
                petals = 5
                for p_i in range(petals):
                    pa0 = (p_i / float(petals)) * math.pi * 2.0
                    pa1 = pa0 + 0.52
                    pa2 = pa0 + 1.04
                    vp1 = bm_wf.verts.new((fx + math.cos(pa0) * fr * 0.62, fy + math.sin(pa0) * fr * 0.62, gz + fh + 0.004))
                    vp2 = bm_wf.verts.new((fx + math.cos(pa1) * fr, fy + math.sin(pa1) * fr, gz + fh + 0.008))
                    vp3 = bm_wf.verts.new((fx + math.cos(pa2) * fr * 0.62, fy + math.sin(pa2) * fr * 0.62, gz + fh + 0.004))
                    face = bm_wf.faces.new((fc, vp1, vp2, vp3))
                    face.material_index = mat_slot

    grass_clumps_obj = _finalize_bmesh(scene, "MeadowGrassClumps", bm_gr, mats=grass_blade_mat, smooth=True)
    wildflowers_obj = _finalize_bmesh(scene, "MeadowWildflowers", bm_wf, mats=[wf_gold_mat, wf_white_mat], smooth=True)
    # Tiny grass blades, flowers & distant hills don't cast visible shadows onto foreground actors — skip shadow BVH rays!
    grass_clumps_obj.visible_shadow = False
    wildflowers_obj.visible_shadow = False
    for h_obj in hill_objs:
        h_obj.visible_shadow = False

    return meadow_obj, terrace_obj, hill_objs, grass_clumps_obj, wildflowers_obj


def _build_carved_stone_balustrade(scene, stone_mat):
    """Build Cotswold stone balustrade (Y=+1.85, X in [-1.80, +3.60]): plinth base, 18 turned urn balusters, coping rail & 3 pier finials."""
    y_bal = 1.85
    x_left = -1.80
    x_mid = 1.85
    x_right = 3.60

    bm_rail = bmesh.new()
    # Continuous stepped beveled stone plinth base (Z = -0.18..0.16)
    span_w = x_right - x_left
    cx_span = 0.5 * (x_left + x_right)
    _bmesh_add_box(bm_rail, (cx_span, y_bal, -0.04), (span_w, 0.28, 0.28), bevel_w=0.008)
    _bmesh_add_box(bm_rail, (cx_span, y_bal, 0.13), (span_w, 0.22, 0.065), bevel_w=0.008)

    # Continuous molded stone coping rail (Z = 0.78..0.86)
    _bmesh_add_box(bm_rail, (cx_span, y_bal, 0.795), (span_w, 0.22, 0.036), bevel_w=0.006)
    _bmesh_add_box(bm_rail, (cx_span, y_bal, 0.836), (span_w + 0.04, 0.275, 0.052), bevel_w=0.010)

    # 3 Carved Corner & Center Pier Posts (X = -1.80, +1.85, +3.60)
    for px in (x_left, x_mid, x_right):
        _bmesh_add_box(bm_rail, (px, y_bal, 0.36), (0.30, 0.30, 1.08), bevel_w=0.010)
        _bmesh_add_box(bm_rail, (px, y_bal, 0.925), (0.36, 0.36, 0.068), bevel_w=0.010)
        if abs(px - x_mid) < 1e-3:
            # Stepped sculpted Cotswold stone seat on middle pier (X=+1.85, Y=+1.85, Z=0.96) for the wrought-iron lantern!
            lantern_seat_prof = [
                (0.118, 0.956),
                (0.114, 0.974),
                (0.096, 0.986),
                (0.088, 0.992),
            ]
            _bmesh_add_lathe(bm_rail, lantern_seat_prof, center=Vector((px, y_bal, 0.0)), segs=18)
        else:
            # Classical turned Cotswold stone spherical garden finial on flanking piers (Z = 0.96..1.18)
            finial_prof = [
                (0.110, 0.958),
                (0.105, 0.978),
                (0.062, 0.996),
                (0.048, 1.018),
                (0.088, 1.052),
                (0.102, 1.092),
                (0.086, 1.132),
                (0.046, 1.162),
                (0.012, 1.182),
            ]
            _bmesh_add_lathe(bm_rail, finial_prof, center=Vector((px, y_bal, 0.0)), segs=18)

    rail_obj = _finalize_bmesh(
        scene,
        "CotswoldBalustradePiersAndRail",
        bm_rail,
        mats=stone_mat,
        smooth=True,
        edge_split_deg=38.0,
    )

    # 18 Turned Classical Stone Urn-Profile Balusters (12 in left span [-1.80..+1.85], 6 in right span [+1.85..+3.60])
    bm_urns = bmesh.new()
    baluster_xs = []
    # Span 1: 12 balusters cleanly clear of pier widths (0.30m)
    for i in range(12):
        u = (i + 0.5) / 12.0
        baluster_xs.append(lerp(x_left + 0.20, x_mid - 0.20, u))
    # Span 2: 6 balusters
    for i in range(6):
        u = (i + 0.5) / 6.0
        baluster_xs.append(lerp(x_mid + 0.20, x_right - 0.20, u))

    urn_profile = [
        (0.060, 0.158),
        (0.060, 0.188),
        (0.042, 0.205),
        (0.030, 0.235),
        (0.050, 0.285),
        (0.062, 0.345),  # Classical urn belly
        (0.056, 0.415),
        (0.036, 0.495),
        (0.028, 0.575),  # Graceful concave neck
        (0.038, 0.640),
        (0.048, 0.695),
        (0.036, 0.722),
        (0.058, 0.752),
        (0.062, 0.782),
    ]
    for bx in baluster_xs:
        _bmesh_add_lathe(bm_urns, urn_profile, center=Vector((bx, y_bal, 0.0)), segs=16)

    urns_obj = _finalize_bmesh(
        scene,
        "CotswoldBalustrade18Urns",
        bm_urns,
        mats=stone_mat,
        smooth=True,
        edge_split_deg=42.0,
    )
    return [rail_obj, urns_obj]


def _build_woolsthorpe_apple_tree(scene, bark_mat, leaf_mat, apple_mat):
    """Build photoreal 3.5m-tall Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0).
    ZERO Play-Doh blob spheres! Real gnarled branching woody twig lattice (50+ sub-branches & twigs)
    populated with ~2,600 individual sculpted 6-vert matte botanical leaves, shaped cleanly left/low
    of Beat 6's starry sky title zone (X > +0.15, Z > 2.25), with Hero Bough ending at HERO_APPLE_ATTACH!"""
    bm_wood = bmesh.new()
    rng_t = random.Random(1666)

    def _point_clear_of_title(pt: Vector, pad: float = 0.0) -> bool:
        """Guarantee Beat 6 title zone (X > +0.15, Z > 2.25) & Beat 6 camera title corridor stay 100% unobstructed!"""
        # 1. Hard world exclusion with safety pad for leaf blade length
        if pt.x > (0.12 - pad) and pt.z > (2.22 - pad):
            return False
        # 2. Perspective sightline corridor from Beat 6 camera (start & end) through NewtonTitle3D at (0.55, 0.60, 2.52)
        if pt.y > 0.50:
            for cy_cam, cz_cam in ((-2.35, 1.52), (-2.68, 1.62)):
                denom = pt.y - cy_cam
                if denom > 0.2:
                    t_proj = (0.60 - cy_cam) / denom
                    px = 0.55 + (pt.x - 0.55) * t_proj
                    pz = cz_cam + (pt.z - cz_cam) * t_proj
                    if px > (-0.06 - pad) and pz > (2.24 - pad):
                        return False
        return True

    def _point_clear_of_hero_apple(pt: Vector) -> bool:
        """Keep immediate falling hero apple zone & Beat 1 Newton head clearance pristine."""
        if (pt - HERO_APPLE_ATTACH).length < 0.095:
            return False
        dx_a = pt.x - HERO_APPLE_ATTACH.x
        dy_a = pt.y - HERO_APPLE_ATTACH.y
        if (dx_a * dx_a + dy_a * dy_a) < (0.095 * 0.095) and pt.z < (HERO_APPLE_ATTACH.z + 0.04):
            return False
        if pt.y < 0.72 and pt.z < 1.92:
            return False
        return True

    # 1. Thick Ancient Gnarled Trunk (r=0.305m at mossy flare -> 0.172m at crown fork Z=1.66m)
    trunk_pts = [
        (-0.18, 1.35, -0.06, 0.305),
        (-0.18, 1.35, 0.12, 0.282),
        (-0.20, 1.34, 0.56, 0.230),
        (-0.22, 1.32, 1.06, 0.198),
        (-0.20, 1.30, 1.44, 0.180),
        (-0.19, 1.29, 1.66, 0.172),
    ]
    _bmesh_add_tapered_tube(bm_wood, trunk_pts, radial_segs=16, subdivs=3, bark_ridge=0.078)

    # 5 Flaring Mossy Buttress Roots + 3 surface spurs gripping into the soil well
    root_angles = [0.22, 1.48, 2.74, 4.04, 5.26]
    for r_ang in root_angles:
        ca, sa = math.cos(r_ang), math.sin(r_ang)
        r_pts = [
            (TREE_BASE.x + ca * 0.14, TREE_BASE.y + sa * 0.14, 0.36, 0.108),
            (TREE_BASE.x + ca * 0.28, TREE_BASE.y + sa * 0.28, 0.14, 0.074),
            (TREE_BASE.x + ca * 0.41, TREE_BASE.y + sa * 0.41, 0.02, 0.036),
            (TREE_BASE.x + ca * 0.48, TREE_BASE.y + sa * 0.48, -0.07, 0.018),
        ]
        _bmesh_add_tapered_tube(bm_wood, r_pts, radial_segs=9, subdivs=2, bark_ridge=0.055)

    # 2. Five Sweeping Organic Primary Boughs sprouting cleanly from crown fork (-0.19, 1.29, 1.66)
    # BOUGH 1: HERO OVERHANGING APPLE BOUGH -> arches forward-left to exact HERO_APPLE_ATTACH!
    hero_bough = [
        (-0.19, 1.29, 1.62, 0.112),
        (-0.26, 1.02, 1.98, 0.086),
        (-0.36, 0.70, 2.18, 0.064),
        (-0.44, 0.42, 2.14, 0.044),
        (-0.49, 0.22, 2.03, 0.025),
        (HERO_APPLE_ATTACH.x, HERO_APPLE_ATTACH.y, HERO_APPLE_ATTACH.z, 0.008),
    ]
    # BOUGH 2: Sweeping Upper-Left Orchard Arch (fills Beat 1 upper-left sky!)
    bough_left = [
        (-0.19, 1.29, 1.64, 0.118),
        (-0.56, 1.24, 2.12, 0.090),
        (-1.04, 1.16, 2.54, 0.064),
        (-1.50, 1.06, 2.82, 0.036),
        (-1.82, 0.96, 2.98, 0.013),
    ]
    # BOUGH 3: Tall Wind-Sculpted Central-Left Crown Leader (3.52m tall, clear of Beat 6 sky title!)
    bough_crown = [
        (-0.19, 1.29, 1.66, 0.124),
        (-0.28, 1.34, 2.24, 0.092),
        (-0.46, 1.42, 2.82, 0.058),
        (-0.58, 1.46, 3.26, 0.028),
        (-0.68, 1.48, 3.50, 0.011),
    ]
    # BOUGH 4: Rear-Left Balustrade Canopy Arch (lush depth behind trunk & over stone balustrade)
    bough_rear = [
        (-0.18, 1.32, 1.68, 0.108),
        (-0.34, 1.68, 2.22, 0.080),
        (-0.66, 2.06, 2.66, 0.052),
        (-0.98, 2.34, 2.96, 0.026),
        (-1.24, 2.48, 3.12, 0.011),
    ]
    # BOUGH 5: Lower-Right Balancing Orchard Bough (stays cleanly BELOW Z <= 2.02 so Beat 6 sky title stays 100% open!)
    bough_low_right = [
        (-0.17, 1.30, 1.62, 0.088),
        (0.06, 1.38, 1.82, 0.062),
        (0.30, 1.48, 1.93, 0.038),
        (0.50, 1.56, 1.98, 0.015),
    ]
    primary_boughs = [hero_bough, bough_left, bough_crown, bough_rear, bough_low_right]
    for b_idx, bough in enumerate(primary_boughs):
        _bmesh_add_tapered_tube(
            bm_wood,
            bough,
            radial_segs=11 if b_idx == 0 else 10,
            subdivs=3,
            bark_ridge=0.048,
        )

    # 3. 12 Sculpted Secondary Structural Limbs + 40 Slender Tapered Gnarled Sub-Twigs (= 52 branching twigs!)
    hero_twig_a = [
        (-0.36, 0.70, 2.16, 0.038),
        (-0.58, 0.60, 2.26, 0.024),
        (-0.80, 0.44, 2.18, 0.013),
        (-0.92, 0.34, 2.10, 0.005),
    ]
    hero_twig_b = [
        (-0.44, 0.42, 2.13, 0.026),
        (-0.34, 0.30, 2.24, 0.016),
        (-0.24, 0.24, 2.26, 0.007),
    ]
    hero_twig_c = [
        (-0.48, 0.24, 2.04, 0.018),
        (-0.64, 0.16, 2.08, 0.011),
        (-0.76, 0.14, 2.12, 0.005),
    ]
    hero_twig_d = [
        (-0.28, 0.96, 2.02, 0.034),
        (-0.54, 0.88, 2.24, 0.021),
        (-0.78, 0.78, 2.38, 0.011),
        (-0.96, 0.72, 2.44, 0.005),
    ]
    secondary_limbs = [
        hero_twig_a,
        hero_twig_b,
        hero_twig_c,
        hero_twig_d,
        [(-0.56, 1.24, 2.12, 0.054), (-0.86, 1.56, 2.54, 0.032), (-1.20, 1.82, 2.92, 0.012)],
        [(-1.04, 1.16, 2.54, 0.044), (-1.18, 0.82, 2.86, 0.024), (-1.34, 0.58, 3.08, 0.009)],
        [(-1.04, 1.16, 2.54, 0.040), (-1.42, 1.44, 2.86, 0.022), (-1.72, 1.66, 3.06, 0.009)],
        [(-0.28, 1.34, 2.24, 0.056), (-0.66, 1.52, 2.80, 0.032), (-0.98, 1.66, 3.24, 0.011)],
        [(-0.46, 1.42, 2.82, 0.044), (-0.36, 1.02, 3.10, 0.024), (-0.32, 0.72, 3.28, 0.009)],
        [(-0.34, 1.68, 2.22, 0.048), (-0.76, 1.92, 2.66, 0.026), (-1.12, 2.10, 2.96, 0.010)],
        [(-0.34, 1.68, 2.22, 0.042), (-0.28, 2.04, 2.62, 0.022), (-0.34, 2.28, 2.86, 0.009)],
        [(0.06, 1.38, 1.82, 0.036), (0.20, 1.12, 1.94, 0.020), (0.34, 0.96, 1.98, 0.008)],
    ]
    for limb in secondary_limbs:
        _bmesh_add_tapered_tube(bm_wood, limb, radial_segs=7, subdivs=2, bark_ridge=0.032)

    # Procedurally sprout 40 slender gnarled tertiary sub-twigs branching organically from parent boughs & limbs
    parent_pools = [
        (hero_bough[1:-1], 4),
        (bough_left[1:], 8),
        (bough_crown[1:], 8),
        (bough_rear[1:], 6),
        (bough_low_right[1:], 3),
    ]
    for s_limb in secondary_limbs[:11]:
        parent_pools.append((s_limb, 1))

    tertiary_twigs = []
    twig_attempt_cap = 0
    for p_spline, target_n in parent_pools:
        sampled_parent = _catmull_rom_sample(p_spline, subdivs=3)
        made_here = 0
        tries = 0
        while made_here < target_n and tries < target_n * 10:
            tries += 1
            twig_attempt_cap += 1
            idx_p = rng_t.randint(1, max(1, len(sampled_parent) - 1))
            px, py, pz, pr = sampled_parent[idx_p]
            start_v = Vector((px, py, pz))
            # Natural outward-and-upward branching direction biased away from Beat 6 right-sky title zone
            az = rng_t.uniform(0.0, math.pi * 2.0)
            elev = rng_t.uniform(0.08, 0.62)
            t_dir = Vector((math.cos(az) * math.cos(elev), math.sin(az) * math.cos(elev), math.sin(elev)))
            if start_v.x > -0.25 and start_v.z > 1.95:
                t_dir.x = -abs(t_dir.x) - 0.25
            if start_v.x > 0.0:
                t_dir.z = min(t_dir.z, 0.10)  # Keep lower-right balancing branch twigs strictly below Z <= 2.08!
            t_dir = t_dir.normalized()

            t_len = rng_t.uniform(0.28, 0.56)
            mid_v = start_v + t_dir * (t_len * 0.52) + Vector((rng_t.uniform(-0.07, 0.05), rng_t.uniform(-0.07, 0.07), rng_t.uniform(0.02, 0.09)))
            tip_v = start_v + t_dir * t_len + Vector((rng_t.uniform(-0.08, 0.05), rng_t.uniform(-0.08, 0.08), rng_t.uniform(-0.03, 0.07)))

            # Strictly respect Beat 6 sky title clearance, ground/head clearance & Hero Apple clearance
            if not (_point_clear_of_title(mid_v, pad=0.14) and _point_clear_of_title(tip_v, pad=0.16)):
                continue
            if not (_point_clear_of_hero_apple(mid_v) and _point_clear_of_hero_apple(tip_v)):
                continue
            if tip_v.z < 1.90 or tip_v.z > 3.65 or tip_v.x < -2.25:
                continue

            r0 = min(max(0.009, pr * 0.55), 0.020)
            twig_pts = [
                (start_v.x, start_v.y, start_v.z, r0),
                (mid_v.x, mid_v.y, mid_v.z, r0 * 0.56),
                (tip_v.x, tip_v.y, tip_v.z, 0.0035),
            ]
            _bmesh_add_tapered_tube(bm_wood, twig_pts, radial_segs=5, subdivs=2, bark_ridge=0.02)
            tertiary_twigs.append(twig_pts)
            made_here += 1

    # 4. 2,500 - 2,650 Individual Sculpted 6-Vert Matte Botanical Leaves (ZERO BLOB SPHERES!)
    bm_fol = bmesh.new()
    rng_f = random.Random(1666)
    leaf_count = 0
    TARGET_LEAVES = 2600

    def _try_add_leaf(anchor: Vector, spray_dir: Vector, min_len: float = 0.062, max_len: float = 0.108) -> bool:
        nonlocal leaf_count
        l_len = rng_f.uniform(min_len, max_len)
        l_wid = l_len * rng_f.uniform(0.48, 0.58)
        yaw = math.atan2(spray_dir.y, spray_dir.x) + rng_f.uniform(-0.62, 0.62)
        pitch = -math.asin(max(-0.85, min(0.85, spray_dir.z))) + rng_f.uniform(-0.45, 0.48)
        roll = rng_f.uniform(-0.78, 0.78)
        rot_e = Euler((roll, pitch, yaw), "XYZ")
        tip_est = anchor + (rot_e.to_matrix() @ Vector((l_len, 0.0, -0.05 * l_len)))

        if not (_point_clear_of_title(anchor, pad=0.03) and _point_clear_of_title(tip_est, pad=0.02)):
            return False
        if not (_point_clear_of_hero_apple(anchor) and _point_clear_of_hero_apple(tip_est)):
            return False
        if anchor.z < 1.88 or tip_est.z < 1.86:
            return False

        _bmesh_add_leaf_card(bm_fol, origin=anchor, rot_euler=rot_e, length=l_len, width=l_wid)
        leaf_count += 1
        return True

    # Collect all real woody leaf-bearing twig splines (hero bough outer + 12 secondary limbs + 40 tertiary twigs)
    all_leaf_twigs = [hero_bough[1:]] + secondary_limbs + tertiary_twigs

    # Pass A: Botanical phyllotaxis leaf sprays along every woody twig + rich terminal twig-tip rosettes!
    twig_samples_cache = []
    for t_idx, tw in enumerate(all_leaf_twigs):
        samp = _catmull_rom_sample(tw, subdivs=4)
        twig_samples_cache.append(samp)
        n_s = len(samp)
        is_hero = t_idx <= 4
        # Lateral leaves along outer 75% of twig
        n_lat = 28 if is_hero else 19
        for k in range(n_lat):
            if leaf_count >= TARGET_LEAVES:
                break
            u = rng_f.uniform(0.22, 0.96)
            si = min(n_s - 1, max(0, int(u * (n_s - 1))))
            sx, sy, sz, sr = samp[si]
            # Compute local twig direction for realistic outward phyllotaxis fan
            si_prev = max(0, si - 1)
            si_next = min(n_s - 1, si + 1)
            t_tan = Vector((samp[si_next][0] - samp[si_prev][0], samp[si_next][1] - samp[si_prev][1], samp[si_next][2] - samp[si_prev][2]))
            if t_tan.length < 1e-5:
                t_tan = Vector((-1.0, 0.0, 0.3))
            t_tan = t_tan.normalized()

            ang = rng_f.uniform(0.0, math.pi * 2.0)
            rad_off = sr + rng_f.uniform(0.010, 0.085 if is_hero else 0.115)
            out_dir = Vector((math.cos(ang), math.sin(ang), rng_f.uniform(-0.35, 0.72))).normalized()
            spray_dir = (out_dir * 0.72 + t_tan * 0.28).normalized()
            l_orig = Vector((sx, sy, sz)) + out_dir * rad_off
            _try_add_leaf(l_orig, spray_dir, min_len=0.062, max_len=0.104 if is_hero else 0.110)

        # Lush terminal rosette burst clustered around every twig tip (26-34 individual overlapping leaves!)
        tx, ty, tz, _ = samp[-1]
        tip_c = Vector((tx, ty, tz))
        n_tip = 34 if is_hero else 26
        for _ in range(n_tip):
            if leaf_count >= TARGET_LEAVES:
                break
            th = rng_f.uniform(0.0, math.pi * 2.0)
            cz = rng_f.uniform(-0.55, 0.88)
            sp = math.sqrt(max(0.0, 1.0 - cz * cz))
            r_dir = Vector((sp * math.cos(th), sp * math.sin(th), cz))
            dist = rng_f.uniform(0.015, 0.155 if is_hero else 0.195)
            l_orig = tip_c + r_dir * dist
            _try_add_leaf(l_orig, r_dir, min_len=0.064, max_len=0.108)

    # Pass B: Fill out lush upper-crown dappled sprays directly around outer primary boughs up to exact TARGET_LEAVES (2,600)
    crown_spray_nodes = []
    for b_spline in (bough_left[1:], bough_crown[1:], bough_rear[1:], bough_low_right[1:]):
        crown_spray_nodes.extend(_catmull_rom_sample(b_spline, subdivs=4))
    for samp in twig_samples_cache:
        if len(samp) >= 2:
            crown_spray_nodes.append(samp[-1])
            crown_spray_nodes.append(samp[len(samp) // 2])

    safety_iters = 0
    while leaf_count < TARGET_LEAVES and safety_iters < 4000 and crown_spray_nodes:
        safety_iters += 1
        nx_c, ny_c, nz_c, nr_c = crown_spray_nodes[rng_f.randrange(len(crown_spray_nodes))]
        th = rng_f.uniform(0.0, math.pi * 2.0)
        cz = rng_f.uniform(-0.45, 0.85)
        sp = math.sqrt(max(0.0, 1.0 - cz * cz))
        r_dir = Vector((sp * math.cos(th), sp * math.sin(th), cz))
        dist = nr_c + rng_f.uniform(0.03, 0.24)
        l_orig = Vector((nx_c, ny_c, nz_c)) + r_dir * dist
        _try_add_leaf(l_orig, r_dir, min_len=0.062, max_len=0.108)

    # 5. 10 Glossy Ruby-Gold Heirloom Apples tucked realistically on REAL woody twigs with stems!
    bm_app = bmesh.new()
    candidate_apple_anchors = []
    # Prefer rich visible twigs framing Beat 1 & Beat 2 (clear of HERO_APPLE_ATTACH & Beat 6 sky title!)
    apple_source_twigs = [hero_twig_a, hero_twig_c, hero_twig_d, bough_left[1:], bough_low_right[1:]] + secondary_limbs[4:11] + tertiary_twigs
    for tw in apple_source_twigs:
        samp = _catmull_rom_sample(tw, subdivs=3)
        if len(samp) >= 2:
            for idx_pick in (len(samp) - 1, max(1, (len(samp) * 2) // 3)):
                ax, ay, az, ar = samp[idx_pick]
                attach_pt = Vector((ax, ay, az - ar * 0.5))
                apple_c = Vector((ax, ay, az - ar * 0.5 - 0.068))
                if (apple_c - HERO_APPLE_ATTACH).length < 0.24:
                    continue
                if not (_point_clear_of_title(attach_pt, pad=0.08) and _point_clear_of_title(apple_c, pad=0.08)):
                    continue
                if apple_c.z < 1.88 or apple_c.z > 3.25:
                    continue
                candidate_apple_anchors.append((attach_pt, apple_c))

    placed_apples = []
    for attach_pt, apple_c in candidate_apple_anchors:
        if len(placed_apples) >= 10:
            break
        if any((apple_c - prev_c).length < 0.26 for prev_c in placed_apples):
            continue
        a_rad = rng_t.uniform(0.042, 0.047)
        # Real slender woody pedicel stem connecting the heirloom apple to its parent twig!
        stem_pts = [
            (attach_pt.x, attach_pt.y, attach_pt.z + 0.004, 0.0032),
            (attach_pt.x + 0.003, attach_pt.y, 0.5 * (attach_pt.z + apple_c.z + a_rad * 0.8), 0.0024),
            (apple_c.x, apple_c.y, apple_c.z + a_rad * 0.78, 0.0020),
        ]
        _bmesh_add_tapered_tube(bm_wood, stem_pts, radial_segs=5, subdivs=1, bark_ridge=0.0)
        _bmesh_add_heirloom_apple(bm_app, apple_c, radius=a_rad)
        placed_apples.append(apple_c)

    tree_wood_obj = _finalize_bmesh(scene, "WoolsthorpeAppleTreeWood", bm_wood, mats=bark_mat, smooth=True)
    tree_canopy_obj = _finalize_bmesh(scene, "WoolsthorpeAppleCanopy", bm_fol, mats=leaf_mat, smooth=True)
    canopy_apples_obj = _finalize_bmesh(scene, "WoolsthorpeCanopyApples", bm_app, mats=apple_mat, smooth=True)

    return tree_wood_obj, tree_canopy_obj, canopy_apples_obj


def _build_cotswold_optical_loggia(scene, ashlar_mat, alcove_dark_mat, stone_mat, oak_mat, brass_mat):
    """Build Left Shaded Cotswold Stone Manor Garden Loggia / Optical Alcove (X in [-3.75, -1.80], Y in [-0.40, +1.85]).
    Features:
      - Smooth continuous semicircular carved stone arch (ZERO Lego cubes!)
      - Dark walnut optical shutter with physical circular pinhole at exact PRISM_PINHOLE = (-3.38, 0.06, 1.54)
      - Deep shadowed dark limestone back wall (Y=+1.65, sealed down to Z=-0.30) for Beat 3 rainbow chiaroscuro!
    """
    loggia_objs = []

    # 1. Left Manor Wall (X in [-3.75, -3.40], Y in [-0.45, +1.88], Z in [-0.30, +3.25])
    #    with smooth semicircular arched optical window centered at Y=0.06, R=0.40, springing Z=1.48!
    y_c = PRISM_PINHOLE.y  # 0.06
    r_arch = 0.40
    z_sill = 0.94
    z_spr = 1.48
    z_spandrel_top = 2.16
    x_out, x_in = -3.76, -3.40

    bm_wall = bmesh.new()
    # Lower wall below sill (Z = -0.30..0.94)
    _bmesh_add_box(bm_wall, (-3.58, 0.72, 0.32), (0.36, 2.34, 1.24))
    # South front cheek pier (Y = -0.45..-0.34, Z = 0.94..3.25)
    _bmesh_add_box(bm_wall, (-3.58, -0.395, 2.095), (0.36, 0.11, 2.31))
    # North wall pier behind window (Y = +0.46..+1.88, Z = 0.94..3.25)
    _bmesh_add_box(bm_wall, (-3.58, 1.17, 2.095), (0.36, 1.42, 2.31))
    # Upper header above spandrel (Y = -0.34..+0.46, Z = 2.16..3.25)
    _bmesh_add_box(bm_wall, (-3.58, 0.06, 2.705), (0.36, 0.80, 1.09))

    # Smooth watertight masonry spandrel directly above the semicircular window arch (N=32 curved segments!)
    n_sp = 32
    in_arc_l, in_arc_r = [], []
    top_l, top_r = [], []
    for i in range(n_sp + 1):
        ang = (i / float(n_sp)) * math.pi  # 0 (+Y side: +0.46) -> pi (-Y side: -0.34)
        ay = y_c + r_arch * math.cos(ang)
        az = z_spr + r_arch * math.sin(ang)
        ty = lerp(y_c + r_arch, y_c - r_arch, i / float(n_sp))
        tz = z_spandrel_top
        in_arc_l.append(bm_wall.verts.new((x_out, ay, az)))
        in_arc_r.append(bm_wall.verts.new((x_in, ay, az)))
        top_l.append(bm_wall.verts.new((x_out, ty, tz)))
        top_r.append(bm_wall.verts.new((x_in, ty, tz)))

    for i in range(n_sp):
        # Inner curved arch soffit (smooth continuous masonry ceiling of the window!)
        bm_wall.faces.new((in_arc_l[i], in_arc_l[i + 1], in_arc_r[i + 1], in_arc_r[i]))
        # Interior (+X) spandrel face
        bm_wall.faces.new((in_arc_r[i], top_r[i], top_r[i + 1], in_arc_r[i + 1]))
        # Exterior (-X) spandrel face
        bm_wall.faces.new((top_l[i], in_arc_l[i], in_arc_l[i + 1], top_l[i + 1]))
        # Top cap
        bm_wall.faces.new((top_l[i], top_l[i + 1], top_r[i + 1], top_r[i]))
    # Left/right spandrel end caps
    bm_wall.faces.new((in_arc_l[0], in_arc_r[0], top_r[0], top_l[0]))
    bm_wall.faces.new((in_arc_r[-1], in_arc_l[-1], top_l[-1], top_r[-1]))

    left_wall_obj = _finalize_bmesh(
        scene,
        "LoggiaLeftManorWall",
        bm_wall,
        mats=ashlar_mat,
        smooth=True,
        edge_split_deg=35.0,
    )
    loggia_objs.append(left_wall_obj)

    # 2. Smooth Semicircular Carved Stone Arch Molding & Sill (continuous bmesh extrusion — ZERO Lego cubes!)
    bm_arch = bmesh.new()
    n_m = 36
    # Molded architrave profile in (radial_offset_from_r_arch, x_coord)
    arch_prof = [
        (-0.012, -3.44),
        (-0.012, -3.34),
        (0.032, -3.33),
        (0.055, -3.35),
        (0.115, -3.345),
        (0.125, -3.41),
    ]
    a_rings = []
    for i in range(n_m + 1):
        ang = (i / float(n_m)) * math.pi
        ca, sa = math.cos(ang), math.sin(ang)
        ring = []
        for dr, ax in arch_prof:
            rr = r_arch + dr
            ring.append(bm_arch.verts.new((ax, y_c + rr * ca, z_spr + rr * sa)))
        a_rings.append(ring)

    np_a = len(arch_prof)
    for i in range(n_m):
        r0, r1 = a_rings[i], a_rings[i + 1]
        for j in range(np_a):
            jn = (j + 1) % np_a
            bm_arch.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
    bm_arch.faces.new(list(reversed(a_rings[0])))
    bm_arch.faces.new(a_rings[-1])

    # Carved window jambs, impost blocks, and projecting stone window sill
    _bmesh_add_box(bm_arch, (-3.375, y_c - r_arch - 0.05, 1.21), (0.09, 0.11, 0.54), bevel_w=0.008)
    _bmesh_add_box(bm_arch, (-3.375, y_c + r_arch + 0.05, 1.21), (0.09, 0.11, 0.54), bevel_w=0.008)
    _bmesh_add_box(bm_arch, (-3.36, y_c - r_arch - 0.055, z_spr - 0.02), (0.12, 0.14, 0.065), bevel_w=0.008)
    _bmesh_add_box(bm_arch, (-3.36, y_c + r_arch + 0.055, z_spr - 0.02), (0.12, 0.14, 0.065), bevel_w=0.008)
    _bmesh_add_box(bm_arch, (-3.54, y_c, z_sill - 0.03), (0.46, 0.98, 0.085), bevel_w=0.010)

    # Carved stone eaves cornice & cantilevered loggia hood roof (casts rich chiaroscuro shade over Beat 3!)
    _bmesh_add_box(bm_arch, (-2.78, 0.72, 2.86), (2.10, 2.38, 0.14), bevel_w=0.012)
    _bmesh_add_box(bm_arch, (-2.78, 0.72, 3.04), (2.22, 2.48, 0.22), bevel_w=0.016)
    # Carved stone base plinth along the loggia floor perimeter
    _bmesh_add_box(bm_arch, (-3.38, 0.72, 0.08), (0.10, 2.30, 0.20), bevel_w=0.008)
    # Molded right corner stone pilaster at (-1.80, +1.64) tying smoothly into the garden balustrade!
    _bmesh_add_box(bm_arch, (-1.80, 1.64, 1.40), (0.28, 0.34, 3.00), bevel_w=0.012)
    _bmesh_add_box(bm_arch, (-1.80, 1.64, 2.72), (0.36, 0.42, 0.16), bevel_w=0.010)
    # Stone corbel bracket supporting the overhanging roof canopy cleanly without blocking Beat 3's camera!
    _bmesh_add_box(bm_arch, (-1.80, 1.38, 2.66), (0.24, 0.46, 0.26), bevel_w=0.012)
    _bmesh_add_box(bm_arch, (-3.34, -0.34, 2.66), (0.24, 0.24, 0.28), bevel_w=0.012)

    arch_trim_obj = _finalize_bmesh(
        scene,
        "LoggiaCarvedStoneArchAndRoof",
        bm_arch,
        mats=stone_mat,
        smooth=True,
        edge_split_deg=38.0,
    )
    loggia_objs.append(arch_trim_obj)

    # 3. Dark Shaded Limestone Back Alcove Wall (Y = +1.65, X in [-3.75, -1.80], sealed cleanly to Z = -0.30!)
    #    Provides PITCH-RICH CHIAROSCURO CONTRAST so Beat 3's 7 rainbow dispersion beams glow like neon velvet!
    bm_alc = bmesh.new()
    _bmesh_add_box(bm_alc, (-2.78, 1.80, 1.48), (2.06, 0.30, 3.56))  # Z in [-0.30, +3.26], front face at Y=+1.65!
    # Subtle smooth semicircular recessed architectural blind-arch molding framing the Beat 3 prism backdrop
    n_ba = 32
    ba_cx, ba_cz, ba_r = -2.86, 1.66, 0.48
    ba_prof = [(0.0, 1.66), (0.0, 1.60), (0.085, 1.60), (0.095, 1.66)]
    ba_rings = []
    for i in range(n_ba + 1):
        ang = (i / float(n_ba)) * math.pi
        ca, sa = math.cos(ang), math.sin(ang)
        ring = []
        for dr, by in ba_prof:
            rr = ba_r + dr
            ring.append(bm_alc.verts.new((ba_cx + rr * ca, by, ba_cz + rr * sa)))
        ba_rings.append(ring)
    for i in range(n_ba):
        r0, r1 = ba_rings[i], ba_rings[i + 1]
        for j in range(4):
            jn = (j + 1) % 4
            bm_alc.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
    bm_alc.faces.new(list(reversed(ba_rings[0])))
    bm_alc.faces.new(ba_rings[-1])
    # Dark stone plinth skirting down into floor
    _bmesh_add_box(bm_alc, (-2.78, 1.62, 0.06), (1.96, 0.08, 0.24), bevel_w=0.008)

    alcove_back_obj = _finalize_bmesh(
        scene,
        "LoggiaDarkChiaroscuroBackWall",
        bm_alc,
        mats=alcove_dark_mat,
        smooth=True,
        edge_split_deg=38.0,
    )
    loggia_objs.append(alcove_back_obj)

    # Dark shaded walnut timber roof soffit under the stone canopy
    bm_sof = bmesh.new()
    _bmesh_add_box(bm_sof, (-2.76, 0.68, 2.77), (1.96, 2.18, 0.05))
    for beam_y in (-0.18, 0.34, 0.86, 1.38):
        _bmesh_add_box(bm_sof, (-2.76, beam_y, 2.71), (1.94, 0.09, 0.11), bevel_w=0.006)
    soffit_obj = _finalize_bmesh(scene, "LoggiaDarkOakSoffit", bm_sof, mats=oak_mat, smooth=True, edge_split_deg=35.0)
    loggia_objs.append(soffit_obj)

    # 4. Dark Walnut Timber Optical Shutter at X = -3.38 with PHYSICAL CIRCULAR PINHOLE at PRISM_PINHOLE (-3.38, 0.06, 1.54)!
    bm_sh = bmesh.new()
    py0, pz0 = PRISM_PINHOLE.y, PRISM_PINHOLE.z  # (0.06, 1.54)
    r_hole = 0.044
    n_sh = 48
    sh_x_back = PRISM_PINHOLE.x - 0.025  # -3.405
    sh_x_front = PRISM_PINHOLE.x + 0.025  # -3.355

    def _shutter_outer_dist(dy: float, dz: float) -> float:
        """Exact ray distance from (py0, pz0) along unit (dy, dz) to the arched shutter perimeter."""
        d_best = 10.0
        # Left/right vertical jambs (Y = y_c ± r_arch)
        if dy > 1e-6:
            d_best = min(d_best, (y_c + r_arch - py0) / dy)
        elif dy < -1e-6:
            d_best = min(d_best, (y_c - r_arch - py0) / dy)
        # Bottom sill (Z = z_sill)
        if dz < -1e-6:
            d_best = min(d_best, (z_sill - pz0) / dz)
        # Top semicircular arch centered at (y_c, z_spr) with radius r_arch
        oy = py0 - y_c
        oz = pz0 - z_spr
        b_c = 2.0 * (oy * dy + oz * dz)
        c_c = oy * oy + oz * oz - r_arch * r_arch
        disc = b_c * b_c - 4.0 * c_c
        if disc >= 0.0:
            t_arc = (-b_c + math.sqrt(disc)) * 0.5
            if t_arc > 0.0 and (pz0 + t_arc * dz) >= z_spr - 1e-4:
                d_best = min(d_best, t_arc)
        return max(r_hole + 0.02, d_best)

    in_f, in_b, out_f, out_b = [], [], [], []
    for i in range(n_sh):
        ang = (i / float(n_sh)) * math.pi * 2.0
        dy, dz = math.cos(ang), math.sin(ang)
        d_out = _shutter_outer_dist(dy, dz)
        in_f.append(bm_sh.verts.new((sh_x_front, py0 + dy * r_hole, pz0 + dz * r_hole)))
        in_b.append(bm_sh.verts.new((sh_x_back, py0 + dy * r_hole, pz0 + dz * r_hole)))
        out_f.append(bm_sh.verts.new((sh_x_front, py0 + dy * d_out, pz0 + dz * d_out)))
        out_b.append(bm_sh.verts.new((sh_x_back, py0 + dy * d_out, pz0 + dz * d_out)))

    for i in range(n_sh):
        inext = (i + 1) % n_sh
        # Front walnut panel face (+X)
        bm_sh.faces.new((in_f[i], out_f[i], out_f[inext], in_f[inext]))
        # Back walnut panel face (-X)
        bm_sh.faces.new((out_b[i], in_b[i], in_b[inext], out_b[inext]))
        # Inner circular pinhole bore wall
        bm_sh.faces.new((in_b[i], in_f[i], in_f[inext], in_b[inext]))
        # Outer shutter perimeter rim
        bm_sh.faces.new((out_f[i], out_b[i], out_b[inext], out_f[inext]))

    # Horizontal walnut cross-battens below the optical pinhole for authentic 17th-century timber shutter craft
    _bmesh_add_box(bm_sh, (sh_x_front + 0.012, y_c, 1.12), (0.024, 0.76, 0.075), bevel_w=0.004)
    _bmesh_add_box(bm_sh, (sh_x_front + 0.012, y_c, 1.36), (0.024, 0.76, 0.065), bevel_w=0.004)

    shutter_obj = _finalize_bmesh(
        scene,
        "LoggiaWalnutOpticalShutter",
        bm_sh,
        mats=oak_mat,
        smooth=True,
        edge_split_deg=35.0,
    )
    loggia_objs.append(shutter_obj)

    # Turned Antique Brass Optical Aperture Collar framing PRISM_PINHOLE (-3.38, 0.06, 1.54)
    bm_br = bmesh.new()
    n_br = 32
    br_prof = [
        (r_hole - 0.001, -0.026),
        (r_hole - 0.001, +0.032),
        (r_hole + 0.018, +0.032),
        (r_hole + 0.022, +0.024),
        (r_hole + 0.014, -0.026),
    ]
    br_rings = []
    for i in range(n_br):
        ang = (i / float(n_br)) * math.pi * 2.0
        ca, sa = math.cos(ang), math.sin(ang)
        ring = []
        for rr, dx in br_prof:
            ring.append(bm_br.verts.new((PRISM_PINHOLE.x + dx, py0 + rr * ca, pz0 + rr * sa)))
        br_rings.append(ring)
    for i in range(n_br):
        inext = (i + 1) % n_br
        r0, r1 = br_rings[i], br_rings[inext]
        for j in range(len(br_prof)):
            jn = (j + 1) % len(br_prof)
            bm_br.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
    pinhole_collar_obj = _finalize_bmesh(
        scene,
        "OpticalPinholeBrassCollar",
        bm_br,
        mats=brass_mat,
        smooth=True,
        edge_split_deg=38.0,
    )
    loggia_objs.append(pinhole_collar_obj)

    return loggia_objs


def _build_garden_plinths(scene, stone_mat, brass_mat, bronze_mat=None):
    """Build the 3 carved stone garden plinths with exact top heights & flush antique-bronze classical laurel collar inlays."""
    if bronze_mat is None:
        bronze_mat = brass_mat

    # 1. Beat 1 Apple Plinth: APPLE_PLINTH = (-0.52, 0.12, 0.88), slender r=0.19m fluted Cotswold pedestal!
    bm_ap = bmesh.new()
    ap_prof = [
        (0.245, -0.02),
        (0.245, 0.065),
        (0.215, 0.095),
        (0.200, 0.135),
        (0.155, 0.165),
        (0.148, 0.24),
        (0.144, 0.44),
        (0.140, 0.64),
        (0.138, 0.72),
        (0.158, 0.748),
        (0.146, 0.770),
        (0.188, 0.822),
        (0.195, 0.835),
        (0.195, 0.876),
        (0.188, 0.880),  # Exact Z = 0.880 top!
    ]
    _bmesh_add_lathe(
        bm_ap,
        ap_prof,
        center=Vector((0.0, 0.0, 0.0)),
        segs=24,
        flute_count=12,
        flute_depth=0.011,
        flute_z_range=(0.20, 0.71),
    )
    # V9 Carved Antique-Bronze Classical Laurel Accent Ring inlaid flush around upper collar + necking fillet (slot 1)
    old_ap_faces = set(bm_ap.faces)
    _bmesh_add_lathe(
        bm_ap,
        [
            (0.1945, 0.842),
            (0.1986, 0.845),
            (0.1994, 0.855),
            (0.1986, 0.865),
            (0.1945, 0.868),
        ],
        center=Vector((0.0, 0.0, 0.0)),
        segs=36,
        cap_bot=False,
        cap_top=False,
        flute_count=24,
        flute_depth=0.0024,
        flute_z_range=(0.844, 0.866),
    )
    _bmesh_add_lathe(
        bm_ap,
        [(0.1575, 0.743), (0.1605, 0.748), (0.1575, 0.753)],
        center=Vector((0.0, 0.0, 0.0)),
        segs=28,
        cap_bot=False,
        cap_top=False,
    )
    for f in bm_ap.faces:
        if f not in old_ap_faces:
            f.material_index = 1
    apple_plinth_obj = _finalize_bmesh(
        scene,
        "AppleGardenPlinth",
        bm_ap,
        mats=[stone_mat, bronze_mat],
        loc=(APPLE_PLINTH.x, APPLE_PLINTH.y, FLOOR_Z),
        smooth=True,
        edge_split_deg=38.0,
    )

    # 2. Beat 2 Principia Sundial Pedestal: PRINCIPIA_PLINTH = (0.55, 0.08, 0.90), sculpted octagonal r=0.24m!
    bm_pr = bmesh.new()
    pr_prof = [
        (0.295, -0.02),
        (0.295, 0.070),
        (0.258, 0.110),
        (0.240, 0.155),
        (0.186, 0.195),
        (0.172, 0.360),
        (0.156, 0.540),
        (0.168, 0.680),
        (0.192, 0.745),
        (0.178, 0.768),
        (0.236, 0.832),
        (0.246, 0.848),
        (0.246, 0.894),
        (0.238, 0.900),  # Exact Z = 0.900 top!
    ]
    _bmesh_add_lathe(
        bm_pr,
        pr_prof,
        center=Vector((0.0, 0.0, 0.0)),
        segs=8,
        angle_offset=math.pi / 8.0,
    )
    # Flush engraved antique brass sundial compass rim on top face (material index 1)
    old_faces = set(bm_pr.faces)
    _bmesh_add_lathe(
        bm_pr,
        [(0.205, 0.896), (0.205, 0.9005), (0.185, 0.9005), (0.185, 0.896)],
        center=Vector((0.0, 0.0, 0.0)),
        segs=24,
        cap_bot=False,
        cap_top=False,
    )
    for f in bm_pr.faces:
        if f not in old_faces:
            f.material_index = 1

    # V9 Carved Antique-Bronze Classical Laurel Frieze Ring & 8 Rosettes inlaid flush in upper octagonal collar (slot 2)
    old_pr_bronze = set(bm_pr.faces)
    _bmesh_add_lathe(
        bm_pr,
        [
            (0.2458, 0.858),
            (0.2495, 0.861),
            (0.2504, 0.871),
            (0.2495, 0.881),
            (0.2458, 0.884),
        ],
        center=Vector((0.0, 0.0, 0.0)),
        segs=8,
        angle_offset=math.pi / 8.0,
        cap_bot=False,
        cap_top=False,
    )
    _bmesh_add_lathe(
        bm_pr,
        [(0.1915, 0.740), (0.1955, 0.745), (0.1915, 0.750)],
        center=Vector((0.0, 0.0, 0.0)),
        segs=8,
        angle_offset=math.pi / 8.0,
        cap_bot=False,
        cap_top=False,
    )
    # 8 sculpted bronze laurel rosettes centered cleanly on each octagonal collar facet
    facet_r = 0.246 * math.cos(math.pi / 8.0) + 0.0025
    for k_oct in range(8):
        f_ang = (k_oct / 8.0) * math.pi * 2.0
        rx_c = facet_r * math.cos(f_ang)
        ry_c = facet_r * math.sin(f_ang)
        _bmesh_add_uv_sphere(
            bm_pr,
            Vector((rx_c, ry_c, 0.871)),
            rx=0.011,
            ry=0.011,
            rz=0.010,
            segs=10,
            rings=6,
        )
    for f in bm_pr.faces:
        if f not in old_pr_bronze:
            f.material_index = 2
    principia_plinth_obj = _finalize_bmesh(
        scene,
        "PrincipiaSundialPlinth",
        bm_pr,
        mats=[stone_mat, brass_mat, bronze_mat],
        loc=(PRINCIPIA_PLINTH.x, PRINCIPIA_PLINTH.y, FLOOR_Z),
        smooth=True,
        edge_split_deg=32.0,
    )

    # 3. Beat 4 Cradle Balustrade Plinth: CRADLE_PLINTH = (2.88, 0.08, 0.90), carved rectangular 0.48m x 0.32m!
    bm_cr = bmesh.new()
    # Stepped molded garden base (Z = -0.02..0.15)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.035), (0.54, 0.38, 0.11), bevel_w=0.010)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.120), (0.47, 0.31, 0.065), bevel_w=0.008)
    # Carved central stone pedestal shaft (Z = 0.15..0.77)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.460), (0.39, 0.24, 0.62), bevel_w=0.010)
    # Decorative raised classical panel moldings on front & back faces
    _bmesh_add_box(bm_cr, (0.0, -0.122, 0.460), (0.29, 0.018, 0.46), bevel_w=0.006)
    _bmesh_add_box(bm_cr, (0.0, +0.122, 0.460), (0.29, 0.018, 0.46), bevel_w=0.006)
    # Flared cornice collar & flat beveled top slab (exact top Z = 0.900!)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.800), (0.44, 0.285, 0.065), bevel_w=0.008)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.865), (0.48, 0.32, 0.070), bevel_w=0.007)

    # V9 Carved Antique-Bronze Classical Laurel Collar Band & Front Laurel-Wreath Medallion inlaid flush (slot 1)
    old_cr_bronze = set(bm_cr.faces)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.865), (0.486, 0.326, 0.022), bevel_w=0.003)
    _bmesh_add_box(bm_cr, (0.0, 0.0, 0.800), (0.445, 0.290, 0.012), bevel_w=0.002)
    # Sculpted circular bronze laurel-wreath ring inlaid cleanly into the front cornice collar & shaft panel
    for med_y, med_z, med_r in ((-0.163, 0.865, 0.016), (-0.132, 0.560, 0.044)):
        n_wr = 18
        wr_pts = []
        for k_w in range(n_wr + 1):
            ang = (k_w / float(n_wr)) * math.pi * 2.0
            wr_pts.append((math.cos(ang) * med_r, med_y, med_z + math.sin(ang) * med_r, 0.0042 if med_r > 0.03 else 0.0028))
        _bmesh_add_tapered_tube(bm_cr, wr_pts, radial_segs=6, subdivs=1, cap_start=False, cap_end=False)
    for f in bm_cr.faces:
        if f not in old_cr_bronze:
            f.material_index = 1

    cradle_plinth_obj = _finalize_bmesh(
        scene,
        "CradleTerracePlinth",
        bm_cr,
        mats=[stone_mat, bronze_mat],
        loc=(CRADLE_PLINTH.x, CRADLE_PLINTH.y, FLOOR_Z),
        smooth=True,
        edge_split_deg=35.0,
    )

    return apple_plinth_obj, principia_plinth_obj, cradle_plinth_obj


def _build_manor_casement_window(scene, stone_mat, oak_mat, wrought_iron_mat, casement_glass_mat):
    """Build recessed 17th-century dark-oak leaded cottage casement window on the Cotswold stone pavilion wall.
    Positioned at (-2.14, 1.635, 2.02) facing -Y so it is 100% unoccluded by LoggiaLeftManorWall,
    cleanly visible in Beat 1's left edge & Beat 3's upper-right background, and completely clear of
    PRISM_PINHOLE (-3.38, 0.06, 1.54) and Beat 3's dark chiaroscuro rainbow backdrop!"""
    wx, wy, wz = -2.14, 1.635, 2.02
    win_w, win_h = 0.40, 0.54
    hw, hh = win_w * 0.5, win_h * 0.5

    bm_fr = bmesh.new()
    # Slot 0: Carved Cotswold stone chamfered jambs, drip-molded lintel hood & projecting sloped stone sill
    _bmesh_add_box(bm_fr, (wx - hw - 0.036, wy - 0.012, wz), (0.072, 0.11, win_h + 0.06), bevel_w=0.008)
    _bmesh_add_box(bm_fr, (wx + hw + 0.036, wy - 0.012, wz), (0.072, 0.11, win_h + 0.06), bevel_w=0.008)
    _bmesh_add_box(bm_fr, (wx, wy - 0.022, wz + hh + 0.044), (win_w + 0.18, 0.14, 0.082), bevel_w=0.010)
    _bmesh_add_box(bm_fr, (wx, wy - 0.032, wz - hh - 0.036), (win_w + 0.18, 0.16, 0.068), bevel_w=0.009)

    # Slot 1: Recessed 17th-century dark-oak outer casement frame, vertical mullion & horizontal transom
    old_oak = set(bm_fr.faces)
    ft = 0.028
    y_oak = wy - 0.010
    _bmesh_add_box(bm_fr, (wx - hw + ft * 0.5, y_oak, wz), (ft, 0.056, win_h), bevel_w=0.003)
    _bmesh_add_box(bm_fr, (wx + hw - ft * 0.5, y_oak, wz), (ft, 0.056, win_h), bevel_w=0.003)
    _bmesh_add_box(bm_fr, (wx, y_oak, wz + hh - ft * 0.5), (win_w, 0.056, ft), bevel_w=0.003)
    _bmesh_add_box(bm_fr, (wx, y_oak, wz - hh + ft * 0.5), (win_w, 0.056, ft), bevel_w=0.003)
    # Central dark-oak mullion & Jacobean transom dividing into 4 cozy candlelit lights
    transom_z = wz + 0.045
    _bmesh_add_box(bm_fr, (wx, y_oak - 0.004, wz), (0.026, 0.052, win_h - ft), bevel_w=0.003)
    _bmesh_add_box(bm_fr, (wx, y_oak - 0.004, transom_z), (win_w - ft, 0.052, 0.024), bevel_w=0.003)
    for f in bm_fr.faces:
        if f not in old_oak:
            f.material_index = 1

    # Slot 2: Delicate blackened wrought-iron diamond leaded cames across all 4 lights
    old_lead = set(bm_fr.faces)
    y_lead = wy + 0.002
    pane_rects = [
        (wx - hw + ft, wx - 0.013, wz - hh + ft, transom_z - 0.012),
        (wx + 0.013, wx + hw - ft, wz - hh + ft, transom_z - 0.012),
        (wx - hw + ft, wx - 0.013, transom_z + 0.012, wz + hh - ft),
        (wx + 0.013, wx + hw - ft, transom_z + 0.012, wz + hh - ft),
    ]
    for x0, x1, z0, z1 in pane_rects:
        xm = 0.5 * (x0 + x1)
        zm = 0.5 * (z0 + z1)
        for p_a, p_b in (
            ((x0, y_lead, z0), (x1, y_lead, z1)),
            ((x0, y_lead, z1), (x1, y_lead, z0)),
            ((xm, y_lead, z0), (x1, y_lead, zm)),
            ((x0, y_lead, zm), (xm, y_lead, z1)),
            ((xm, y_lead, z0), (x0, y_lead, zm)),
            ((x1, y_lead, zm), (xm, y_lead, z1)),
        ):
            _bmesh_add_tapered_tube(
                bm_fr,
                [(p_a[0], p_a[1], p_a[2], 0.0028), (p_b[0], p_b[1], p_b[2], 0.0028)],
                radial_segs=4,
                subdivs=1,
            )
    for f in bm_fr.faces:
        if f not in old_lead:
            f.material_index = 2

    frame_obj = _finalize_bmesh(
        scene,
        "ManorCasementWindowFrame",
        bm_fr,
        mats=[stone_mat, oak_mat, wrought_iron_mat],
        smooth=True,
        edge_split_deg=35.0,
    )

    # Warm candlelit hand-blown crown glass backing (visible in camera & glossy, zero diffuse wash on Beat 3 rainbow!)
    bm_gl = bmesh.new()
    _bmesh_add_box(bm_gl, (wx, wy + 0.008, wz), (win_w - 0.036, 0.008, win_h - 0.036))
    glass_obj = _finalize_bmesh(scene, "ManorCasementWindowGlass", bm_gl, mats=casement_glass_mat, smooth=False)
    glass_obj.visible_shadow = False
    glass_obj.visible_diffuse = False
    return [frame_obj, glass_obj]


def _build_climbing_english_ivy(scene, bark_mat, ivy_leaf_mat):
    """Build ~220 matte sculpted 3-lobed dark-emerald English ivy leaves + curved woody tendrils creeping up
    the Cotswold stone loggia pier (X=-1.86) and middle balustrade pier (X=+1.85), merged into 1 fast mesh!"""
    bm = bmesh.new()
    rng_ivy = random.Random(1669)

    def _add_ivy_vine(pts_r, outward_bias: Vector, n_leaves: int, min_sz: float = 0.046, max_sz: float = 0.074):
        _bmesh_add_tapered_tube(bm, pts_r, radial_segs=5, subdivs=2, bark_ridge=0.02)
        samp = _catmull_rom_sample(pts_r, subdivs=4)
        n_s = len(samp)
        if n_s < 2:
            return 0
        added = 0
        for k in range(n_leaves):
            u = lerp(0.05, 0.98, (k + rng_ivy.uniform(0.15, 0.85)) / float(max(1, n_leaves)))
            si = min(n_s - 1, max(0, int(u * (n_s - 1))))
            sx, sy, sz, sr = samp[si]
            # Strict Beat 6 sky title clearance guard
            if sx > 0.12 and sz > 2.20:
                continue
            side_sign = 1.0 if (k % 2 == 0) else -1.0
            tan_p = max(0, si - 1)
            tan_n = min(n_s - 1, si + 1)
            t_vec = Vector((samp[tan_n][0] - samp[tan_p][0], samp[tan_n][1] - samp[tan_p][1], samp[tan_n][2] - samp[tan_p][2]))
            t_vec = t_vec.normalized() if t_vec.length > 1e-5 else Vector((0.0, 0.0, 1.0))
            lat_vec = t_vec.cross(outward_bias)
            lat_vec = lat_vec.normalized() if lat_vec.length > 1e-5 else Vector((1.0, 0.0, 0.0))

            spray = (
                outward_bias * rng_ivy.uniform(0.55, 0.95)
                + lat_vec * (side_sign * rng_ivy.uniform(0.32, 0.78))
                + Vector((0.0, 0.0, rng_ivy.uniform(-0.48, 0.32)))
            ).normalized()
            l_orig = Vector((sx, sy, sz)) + outward_bias * (sr + 0.006) + lat_vec * (side_sign * rng_ivy.uniform(0.005, 0.022))
            yaw = math.atan2(spray.y, spray.x) + rng_ivy.uniform(-0.35, 0.35)
            pitch = -math.asin(max(-0.85, min(0.85, spray.z))) + rng_ivy.uniform(-0.28, 0.32)
            roll = rng_ivy.uniform(-0.55, 0.55)
            sz_l = rng_ivy.uniform(min_sz, max_sz) * lerp(1.05, 0.78, u)
            _bmesh_add_ivy_leaf(bm, l_orig, Euler((roll, pitch, yaw), "XYZ"), size=sz_l, mat_index=1)
            added += 1
        return added

    # Zone 1: Left Cotswold Stone Loggia Pilaster (X in [-1.94, -1.66], Y_front = 1.47, centered near X=-1.86)
    loggia_vines = [
        # Primary spine climbing straight up front-left corner around X = -1.86
        (
            [
                (-1.88, 1.456, 0.02, 0.011),
                (-1.82, 1.455, 0.54, 0.009),
                (-1.90, 1.456, 1.12, 0.008),
                (-1.84, 1.455, 1.72, 0.0065),
                (-1.89, 1.456, 2.24, 0.0050),
                (-1.76, 1.455, 2.54, 0.0032),
            ],
            Vector((-0.25, -0.96, 0.05)).normalized(),
            42,
        ),
        # Wrapping around the right (+X) cheek of the loggia pier (cleanly visible in Beat 1 & Beat 6!)
        (
            [
                (-1.70, 1.456, 0.04, 0.010),
                (-1.648, 1.50, 0.58, 0.0085),
                (-1.648, 1.58, 1.18, 0.0072),
                (-1.652, 1.50, 1.76, 0.0058),
                (-1.72, 1.456, 2.26, 0.0044),
                (-1.66, 1.456, 2.50, 0.0030),
            ],
            Vector((0.68, -0.73, 0.06)).normalized(),
            38,
        ),
        # Wrapping around the left (-X) cheek facing Beat 3 optical alcove!
        (
            [
                (-1.948, 1.52, 0.06, 0.0095),
                (-1.950, 1.58, 0.64, 0.0080),
                (-1.948, 1.50, 1.24, 0.0065),
                (-1.92, 1.456, 1.82, 0.0052),
                (-1.950, 1.54, 2.30, 0.0035),
            ],
            Vector((-0.74, -0.67, 0.05)).normalized(),
            34,
        ),
        # Cascading upper tendril draping gracefully beneath the stone capital at Z = 2.52 -> 2.02
        (
            [
                (-1.74, 1.452, 2.52, 0.0055),
                (-1.67, 1.448, 2.34, 0.0045),
                (-1.65, 1.455, 2.12, 0.0035),
                (-1.69, 1.452, 1.94, 0.0025),
            ],
            Vector((0.35, -0.92, -0.15)).normalized(),
            22,
        ),
    ]

    # Zone 2: Right/Middle Balustrade Pier (X = +1.85, Y = +1.85, shaft X in [1.70, 2.00], Y_front = 1.70, Z = 0..0.90)
    balustrade_vines = [
        # Front-left ascent up the stone balustrade pier beneath the wrought-iron garden lantern!
        (
            [
                (1.74, 1.688, 0.02, 0.0095),
                (1.78, 1.686, 0.28, 0.0082),
                (1.71, 1.688, 0.56, 0.0068),
                (1.78, 1.686, 0.84, 0.0050),
                (1.66, 1.700, 0.87, 0.0038),
                (1.52, 1.712, 0.82, 0.0028),
            ],
            Vector((-0.25, -0.96, 0.06)).normalized(),
            32,
        ),
        # Front-right ascent hugging right pier face & draping over coping rail
        (
            [
                (1.94, 1.688, 0.02, 0.0095),
                (2.012, 1.74, 0.30, 0.0080),
                (1.96, 1.688, 0.58, 0.0066),
                (1.89, 1.686, 0.86, 0.0048),
                (2.04, 1.702, 0.86, 0.0036),
                (2.16, 1.714, 0.81, 0.0026),
            ],
            Vector((0.32, -0.94, 0.06)).normalized(),
            30,
        ),
        # Delicate central collar vine curling right beneath the lantern pier cap (Z = 0.45..0.89)
        (
            [
                (1.85, 1.686, 0.14, 0.0075),
                (1.88, 1.685, 0.42, 0.0064),
                (1.81, 1.685, 0.68, 0.0052),
                (1.86, 1.682, 0.88, 0.0034),
            ],
            Vector((0.0, -1.0, 0.08)).normalized(),
            22,
        ),
    ]

    for pts_r, out_v, n_l in loggia_vines:
        _add_ivy_vine(pts_r, out_v, n_l, min_sz=0.048, max_sz=0.076)
    for pts_r, out_v, n_l in balustrade_vines:
        _add_ivy_vine(pts_r, out_v, n_l, min_sz=0.044, max_sz=0.068)

    ivy_obj = _finalize_bmesh(
        scene,
        "ClimbingEnglishIvy",
        bm,
        mats=[bark_mat, ivy_leaf_mat],
        smooth=True,
    )
    ivy_obj.visible_shadow = False  # Zero shadow BVH bloat!
    return ivy_obj


def _build_balustrade_garden_lantern(scene, wrought_iron_mat, brass_mat, lantern_glass_mat, lantern_flame_mat):
    """Build 17th-century hexagonal wrought-iron & frosted-amber glass garden lantern perched on middle balustrade
    pier (X=+1.85, Y=+1.85, Z=0.96) with flickering flame core & soft warm-amber twilight POINT glow."""
    lx, ly, lz = 1.85, 1.85, 0.96
    base_c = Vector((lx, ly, lz))

    bm = bmesh.new()
    # Slot 0: Hexagonal cast/wrought-iron lantern foot, rims, 6 slanted corner ribs & tiered pagoda roof hood
    foot_prof = [
        (0.086, 0.026),
        (0.082, 0.042),
        (0.052, 0.056),
        (0.056, 0.068),
        (0.072, 0.074),
        (0.072, 0.084),
    ]
    _bmesh_add_lathe(bm, foot_prof, center=base_c, segs=6)

    # Mid-height decorative wrought-iron collar band & flared upper cornice rim
    _bmesh_add_lathe(bm, [(0.084, 0.184), (0.088, 0.190), (0.084, 0.196)], center=base_c, segs=6, cap_bot=False, cap_top=False)
    _bmesh_add_lathe(bm, [(0.096, 0.292), (0.104, 0.298), (0.104, 0.308), (0.095, 0.312)], center=base_c, segs=6)

    # 6 Slanted Hexagonal Corner Wrought-Iron Struts linking lower rim (r=0.068, z=0.078) -> upper rim (r=0.098, z=0.298)
    for k_hex in range(6):
        ang = (k_hex / 6.0) * math.pi * 2.0
        ca, sa = math.cos(ang), math.sin(ang)
        rib_pts = [
            (lx + 0.068 * ca, ly + 0.068 * sa, lz + 0.076, 0.0064),
            (lx + 0.083 * ca, ly + 0.083 * sa, lz + 0.188, 0.0056),
            (lx + 0.098 * ca, ly + 0.098 * sa, lz + 0.300, 0.0064),
        ]
        _bmesh_add_tapered_tube(bm, rib_pts, radial_segs=5, subdivs=1)

    # Flared 17th-century hexagonal pagoda eaves hood, vented chimney cap & iron finial spire (z_local = 0.302..0.455)
    roof_prof = [
        (0.122, 0.302),
        (0.120, 0.314),
        (0.084, 0.344),
        (0.056, 0.376),
        (0.036, 0.396),
        (0.044, 0.402),
        (0.032, 0.412),
        (0.012, 0.422),
        (0.004, 0.455),
    ]
    _bmesh_add_lathe(bm, roof_prof, center=base_c, segs=6)

    # Slot 1: Recessed tapered hexagonal frosted-amber crown glass panes (z_local = 0.082..0.294)
    old_glass = set(bm.faces)
    _bmesh_add_lathe(
        bm,
        [(0.062, 0.082), (0.091, 0.294)],
        center=base_c,
        segs=6,
        cap_bot=False,
        cap_top=False,
    )
    for f in bm.faces:
        if f not in old_glass:
            f.material_index = 1

    # Slot 2: Internal turned antique-brass candle bobeche holder & crown finial orb
    old_brass = set(bm.faces)
    _bmesh_add_lathe(
        bm,
        [(0.026, 0.082), (0.024, 0.092), (0.014, 0.104), (0.013, 0.144), (0.010, 0.148)],
        center=base_c,
        segs=14,
    )
    _bmesh_add_uv_sphere(bm, Vector((lx, ly, lz + 0.425)), rx=0.014, ry=0.014, rz=0.014, segs=12, rings=8)
    for f in bm.faces:
        if f not in old_brass:
            f.material_index = 2

    lantern_obj = _finalize_bmesh(
        scene,
        "BalustradeGardenLantern",
        bm,
        mats=[wrought_iron_mat, lantern_glass_mat, brass_mat],
        smooth=True,
        edge_split_deg=32.0,
    )
    # Prevent lantern glass/cage from blocking internal warm POINT light rays!
    lantern_obj.visible_shadow = False

    # Sculpted flickering golden teardrop candle flame core inside the lantern
    bm_fl = bmesh.new()
    flame_prof = [
        (0.004, -0.022),
        (0.011, -0.010),
        (0.012, 0.002),
        (0.008, 0.018),
        (0.003, 0.032),
        (0.0008, 0.040),
    ]
    _bmesh_add_lathe(bm_fl, flame_prof, center=Vector((0.0, 0.0, 0.0)), segs=12)
    flame_obj = _finalize_bmesh(
        scene,
        "BalustradeLanternFlame",
        bm_fl,
        mats=lantern_flame_mat,
        loc=(lx, ly, lz + 0.172),
        smooth=True,
    )
    flame_obj.visible_shadow = False
    flame_obj.visible_diffuse = False
    flame_obj.visible_glossy = False

    # Soft warm-amber local twilight POINT glow (energy 0.0 in daylight -> 18.0 in Beats 5-6!)
    lantern_light_obj = _add_light_obj(
        scene,
        "BalustradeLanternGlow",
        "POINT",
        loc=(lx, ly, lz + 0.19),
        rot=(0.0, 0.0, 0.0),
        energy=0.0,
        color=(1.0, 0.56, 0.18),
        size=0.16,
    )
    return lantern_obj, flame_obj, lantern_light_obj


def _build_drifting_leaves_and_fireflies(scene, drifting_leaf_mat, firefly_mat):
    """Build 18 gently tumbling golden-hour orchard leaves (Beats 1-4) + 24 pulsing twilight fireflies (Beats 5-6)."""
    # 1. Shared centered matte gold-emerald orchard leaf mesh (single datablock, visible_shadow=False)
    bm_lf = bmesh.new()
    _bmesh_add_leaf_card(
        bm_lf,
        origin=Vector((-0.044, 0.0, 0.0)),
        rot_euler=Euler((0.0, 0.0, 0.0), "XYZ"),
        length=0.088,
        width=0.048,
    )
    bmesh.ops.recalc_face_normals(bm_lf, faces=bm_lf.faces)
    leaf_mesh = bpy.data.meshes.new("DriftingOrchardLeaf_Mesh")
    bm_lf.to_mesh(leaf_mesh)
    bm_lf.free()
    leaf_mesh.update()
    for poly in leaf_mesh.polygons:
        poly.use_smooth = True
    leaf_mesh.materials.append(drifting_leaf_mat)

    rng_l = random.Random(1667)
    # Distributed across Beats 1-4 sunny outdoor zones (6 orchard, 5 courtyard, 3 loggia edge, 4 right balustrade)
    leaf_zones = [
        (-1.48, 0.20, -0.30, 1.35),
        (-1.48, 0.20, -0.30, 1.35),
        (-1.48, 0.20, -0.30, 1.35),
        (-1.48, 0.20, -0.30, 1.35),
        (-1.48, 0.20, -0.30, 1.35),
        (-1.48, 0.20, -0.30, 1.35),
        (-0.30, 1.65, -0.45, 1.45),
        (-0.30, 1.65, -0.45, 1.45),
        (-0.30, 1.65, -0.45, 1.45),
        (-0.30, 1.65, -0.45, 1.45),
        (-0.30, 1.65, -0.45, 1.45),
        (-2.28, -1.55, -0.50, 0.85),
        (-2.28, -1.55, -0.50, 0.85),
        (-2.28, -1.55, -0.50, 0.85),
        (1.75, 3.35, -0.45, 1.55),
        (1.75, 3.35, -0.45, 1.55),
        (1.75, 3.35, -0.45, 1.55),
        (1.75, 3.35, -0.45, 1.55),
    ]
    drifting_leaves = []
    for idx, (xmin, xmax, ymin, ymax) in enumerate(leaf_zones):
        lx0 = rng_l.uniform(xmin, xmax)
        ly0 = rng_l.uniform(ymin, ymax)
        lz0 = rng_l.uniform(0.55, 2.48)
        base_s = rng_l.uniform(0.84, 1.18)
        obj = bpy.data.objects.new(f"DriftingOrchardLeaf_{idx:02d}", leaf_mesh)
        obj.location = Vector((lx0, ly0, lz0))
        obj.scale = (base_s, base_s, base_s)
        obj.visible_shadow = False
        _link_obj(scene, obj)

        params = (
            rng_l.uniform(0.14, 0.26),  # fall_speed (m/s)
            rng_l.uniform(0.14, 0.28),  # spiral_radius
            rng_l.uniform(0.85, 1.65),  # wx
            rng_l.uniform(0.75, 1.45),  # wy
            rng_l.uniform(-1.35, 1.35),  # spin_z
            rng_l.uniform(0.0, math.pi * 2.0),  # phase1
            rng_l.uniform(0.0, math.pi * 2.0),  # phase2
            base_s,
        )
        drifting_leaves.append((obj, Vector((lx0, ly0, lz0)), params))

    # 2. 24 Magical Warm-Gold Twilight Fireflies (hovering strictly over Z = 0.25..1.35 ONLY in Beats 5-6!)
    bm_ff = bmesh.new()
    _bmesh_add_uv_sphere(bm_ff, Vector((0.0, 0.0, 0.0)), 1.0, 1.0, 1.0, segs=8, rings=5)
    bmesh.ops.recalc_face_normals(bm_ff, faces=bm_ff.faces)
    ff_mesh = bpy.data.meshes.new("TwilightFirefly_Mesh")
    bm_ff.to_mesh(ff_mesh)
    bm_ff.free()
    ff_mesh.update()
    for poly in ff_mesh.polygons:
        poly.use_smooth = True
    ff_mesh.materials.append(firefly_mat)

    rng_ff = random.Random(1688)
    fireflies = []
    for idx in range(24):
        # Concentrate gracefully over meadow grass, stone balustrade & Beat 5-6 orchard foreground/midground!
        if idx < 10:
            fx0 = rng_ff.uniform(-1.35, 2.30)
            fy0 = rng_ff.uniform(1.45, 2.55)  # Floating along stone balustrade & rear meadow!
            fz0 = rng_ff.uniform(0.32, 1.25)
        elif idx < 18:
            fx0 = rng_ff.uniform(-1.10, 1.95)
            fy0 = rng_ff.uniform(-0.45, 1.30)  # Hovering gently around courtyard border & apple tree roots!
            fz0 = rng_ff.uniform(0.28, 1.15)
        else:
            fx0 = rng_ff.uniform(-0.85, 2.25)
            fy0 = rng_ff.uniform(0.60, 2.05)
            fz0 = rng_ff.uniform(0.55, 1.30)
        base_r = rng_ff.uniform(0.0052, 0.0070)  # Centered on exact r = 0.006m
        obj = bpy.data.objects.new(f"TwilightFirefly_{idx:02d}", ff_mesh)
        obj.location = Vector((fx0, fy0, fz0))
        obj.scale = (0.0001, 0.0001, 0.0001)  # Hidden cleanly during Beats 1-4 daylight!
        obj.visible_shadow = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        _link_obj(scene, obj)

        ff_params = (
            rng_ff.uniform(0.65, 1.35),  # drift_fx
            rng_ff.uniform(0.55, 1.20),  # drift_fy
            rng_ff.uniform(0.75, 1.45),  # drift_fz
            rng_ff.uniform(2.4, 5.2),  # pulse_freq
            rng_ff.uniform(0.0, math.pi * 2.0),  # px
            rng_ff.uniform(0.0, math.pi * 2.0),  # py
            rng_ff.uniform(0.0, math.pi * 2.0),  # pz
            rng_ff.uniform(0.0, math.pi * 2.0),  # pulse_ph
            base_r,
        )
        fireflies.append((obj, Vector((fx0, fy0, fz0)), ff_params))

    return drifting_leaves, fireflies


def _build_fallen_heirloom_apples(scene, apple_mat, bark_mat):
    """Build 3 fallen Woolsthorpe heirloom apples nestled naturally into the mossy grass around the tree roots."""
    bm = bmesh.new()
    fallen_specs = [
        (Vector((-0.38, 1.05, 0.038)), 0.041, Euler((0.36, -0.22, 0.85), "XYZ")),
        (Vector((0.12, 1.12, 0.038)), 0.042, Euler((-0.26, 0.44, -1.10), "XYZ")),
        (Vector((-0.08, 0.96, 0.036)), 0.039, Euler((0.20, 0.50, 2.25), "XYZ")),
    ]
    for pos, rad, rot_e in fallen_specs:
        _bmesh_add_heirloom_apple(bm, pos, radius=rad, rot_euler=rot_e, mat_index=0)
        # Add authentic slender woody pedicel stem tilted cleanly with the fallen apple
        old_stem = set(bm.faces)
        rmat = rot_e.to_matrix()
        s0 = pos + (rmat @ Vector((0.0, 0.0, rad * 0.72)))
        s1 = pos + (rmat @ Vector((0.004, 0.002, rad * 1.18)))
        _bmesh_add_tapered_tube(
            bm,
            [(s0.x, s0.y, s0.z, 0.0024), (s1.x, s1.y, s1.z, 0.0018)],
            radial_segs=5,
            subdivs=1,
        )
        for f in bm.faces:
            if f not in old_stem:
                f.material_index = 1

    return _finalize_bmesh(scene, "WoolsthorpeFallenApples", bm, mats=[apple_mat, bark_mat], smooth=True)


def _build_godrays_and_pollen(scene, godray_mat, pollen_mat):
    """Build 3 volumetric golden-hour sunbeam cones + 45 randomly scattered 3D drifting golden pollen motes."""
    # 3 feathered translucent golden sunbeam cones angling cleanly from upper-rear-right sun through the Apple Tree canopy
    godray_specs = [
        ("WoolsthorpeGodRay_0", (-0.42, 0.56, 1.98), 0.56, 0.16, 3.30, (math.radians(-28.0), math.radians(+22.0), 0.0)),
        ("WoolsthorpeGodRay_1", (0.64, 0.74, 2.06), 0.64, 0.19, 3.50, (math.radians(-30.0), math.radians(+20.0), 0.0)),
        ("WoolsthorpeGodRay_2", (1.96, 0.86, 1.96), 0.58, 0.17, 3.30, (math.radians(-27.0), math.radians(+18.0), 0.0)),
    ]
    godray_objs = []
    for g_name, g_loc, r_bot, r_top, depth, g_rot in godray_specs:
        bm_g = bmesh.new()
        _bmesh_add_lathe(
            bm_g,
            [(r_bot, -depth * 0.5), (r_top, +depth * 0.5)],
            center=Vector((0.0, 0.0, 0.0)),
            segs=28,
            cap_bot=False,
            cap_top=False,
        )
        gr = _finalize_bmesh(scene, g_name, bm_g, mats=godray_mat, loc=g_loc, rot=g_rot, smooth=True)
        gr.visible_shadow = False
        gr.visible_diffuse = False
        gr.visible_glossy = False
        godray_objs.append(gr)

    # 45 True 3D Randomly Scattered Golden Pollen / Dust Motes (random.Random(1666) — ZERO straight line!)
    bm_p = bmesh.new()
    _bmesh_add_uv_sphere(bm_p, Vector((0.0, 0.0, 0.0)), 1.0, 1.0, 1.0, segs=8, rings=5)
    bmesh.ops.recalc_face_normals(bm_p, faces=bm_p.faces)
    pollen_mesh = bpy.data.meshes.new("GoldenPollenMote_Mesh")
    bm_p.to_mesh(pollen_mesh)
    bm_p.free()
    pollen_mesh.update()
    for poly in pollen_mesh.polygons:
        poly.use_smooth = True
    pollen_mesh.materials.append(pollen_mat)

    rng_p = random.Random(1666)  # Newton's Annus Mirabilis at Woolsthorpe!
    pollen_entries = []
    for i in range(45):
        bx = rng_p.uniform(-2.8, 3.2)
        by = rng_p.uniform(-0.8, 1.5)
        bz = rng_p.uniform(0.6, 2.3)
        pr = rng_p.uniform(0.0032, 0.0066)
        obj = bpy.data.objects.new(f"GoldenPollenMote_{i:02d}", pollen_mesh)
        obj.location = Vector((bx, by, bz))
        obj.scale = (pr, pr, pr)
        obj.visible_shadow = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        _link_obj(scene, obj)

        drift = (
            rng_p.uniform(0.45, 1.15),  # fx
            rng_p.uniform(0.40, 1.05),  # fy
            rng_p.uniform(0.55, 1.25),  # fz
            rng_p.uniform(0.0, math.pi * 2.0),  # px
            rng_p.uniform(0.0, math.pi * 2.0),  # py
            rng_p.uniform(0.0, math.pi * 2.0),  # pz
            pr,
        )
        pollen_entries.append((obj, Vector((bx, by, bz)), drift))

    return godray_objs, pollen_entries


def _aim_euler(loc, target) -> Euler:
    """Compute exact XYZ Euler orientation aiming local -Z from loc straight at target."""
    d = Vector(target) - Vector(loc)
    if d.length < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    return d.normalized().to_track_quat("-Z", "Y").to_euler("XYZ")


def _add_light_obj(scene, name: str, ltype: str, loc, rot, energy: float, color, size: float = 1.0, angle_deg: float = 1.6):
    ld = bpy.data.lights.new(f"{name}_Data", type=ltype)
    ld.energy = energy
    ld.color = color
    if ltype == "AREA":
        ld.size = size
    elif ltype == "SUN":
        ld.angle = math.radians(angle_deg)
    elif ltype == "POINT":
        ld.shadow_soft_size = size
    obj = bpy.data.objects.new(name, ld)
    obj.location = Vector(loc)
    obj.rotation_euler = rot if isinstance(rot, Euler) else Euler(rot, "XYZ")
    obj.visible_camera = False
    _link_obj(scene, obj)
    return obj


def _build_outdoor_lighting_rig(scene):
    """Build Hollywood golden-hour chiaroscuro SUN + directional Rembrandt key + sapphire rim + reduced fill + catchlight."""
    stage_focus = Vector((0.0, 0.30, 1.30))

    # 1. Warm Low-Angle Golden-Hour Sun casting crisp dappled 2,600-leaf shadows across Newton & flagstones!
    sun_obj = _add_light_obj(
        scene,
        "WoolsthorpeGoldenSun",
        "SUN",
        loc=(4.5, 5.0, 5.5),
        rot=(math.radians(65.0), math.radians(7.0), math.radians(146.0)),
        energy=4.6,
        color=(1.0, 0.72, 0.40),
        angle_deg=1.8,
    )

    # 2. Warm Directional Golden-Hour Rembrandt Key Area Light (sculptural faces, coat folds & stone moldings!)
    key_loc = (-2.35, -2.55, 2.55)
    key_obj = _add_light_obj(
        scene,
        "WoolsthorpeRembrandtKey",
        "AREA",
        loc=key_loc,
        rot=_aim_euler(key_loc, stage_focus),
        energy=270.0,
        color=(1.0, 0.76, 0.46),
        size=1.38,
    )

    # 3. Cool Sapphire-Blue Sky / Moon Rim Light (crisply traces hair strands, shoulders & silhouettes!)
    rim_loc = (2.25, 3.15, 2.85)
    rim_obj = _add_light_obj(
        scene,
        "WoolsthorpeSkyMoonRim",
        "AREA",
        loc=rim_loc,
        rot=_aim_euler(rim_loc, stage_focus),
        energy=245.0,
        color=(0.38, 0.70, 1.0),
        size=1.95,
    )

    # 4. Restrained Outdoor Meadow Bounce Fill (40% lower energy -> dramatic chiaroscuro depth, ZERO flat wash!)
    fill_loc = (0.85, -2.50, 1.65)
    fill_obj = _add_light_obj(
        scene,
        "WoolsthorpeMeadowBounce",
        "AREA",
        loc=fill_loc,
        rot=_aim_euler(fill_loc, stage_focus),
        energy=40.0,
        color=(0.90, 0.84, 0.74),
        size=2.40,
    )

    # 5. Wet-Cornea Eye Catchlight & Brass / Prism Specular Sparkle Accent
    catch_obj = _add_light_obj(
        scene,
        "WoolsthorpeEyeCatchlight",
        "POINT",
        loc=(0.0, -1.35, 1.55),
        rot=(0.0, 0.0, 0.0),
        energy=18.0,
        color=(1.0, 0.95, 0.88),
        size=0.08,
    )

    return {
        "sun_light": sun_obj,
        "key_light": key_obj,
        "rim_light": rim_obj,
        "fill_light": fill_obj,
        "catch_light": catch_obj,
    }


# ============================================================================
# 6. PUBLIC API: build_woolsthorpe_world(scene) & animate_world_frame(...)
# ============================================================================
def build_woolsthorpe_world(scene) -> dict:
    """Build the complete V9 Cinema Outdoor Woolsthorpe Orchard & Cosmos Environment and return world_handles."""
    # Instantiate rich shared V9 photoreal Cycles AO, dew-sheen & matte botanical PBR material palette
    stone_mat = _make_cotswold_stone_mat("CotswoldCarvedLimestone", dark=False)
    ashlar_mat = _make_cotswold_ashlar_mat("CotswoldManorAshlar", dark=False)
    alcove_dark_stone_mat = _make_cotswold_ashlar_mat("LoggiaChiaroscuroDarkStone", dark=True)
    flagstone_mat = _make_mossy_flagstone_mat("CotswoldMossyFlagstone")
    meadow_grass_mat = _make_meadow_grass_mat("WoolsthorpeMeadowGrass", distant=False, haze_mix=0.0)
    hill_mid_mat = _make_meadow_grass_mat("LincolnshireMidHill", distant=True, haze_mix=0.40)
    hill_far_mat = _make_meadow_grass_mat("LincolnshireFarHorizonHill", distant=True, haze_mix=0.66)
    distant_tree_mat = _make_meadow_grass_mat("DistantEnglishHedgerow", distant=True, haze_mix=0.48)
    grass_blade_mat = _make_simple_pbr_mat("MeadowGrassBlade", (0.042, 0.135, 0.020), rough=0.78)
    wildflower_gold_mat = _make_simple_pbr_mat("MeadowButtercupGold", (0.86, 0.64, 0.05), rough=0.48)
    wildflower_white_mat = _make_simple_pbr_mat("MeadowEnglishDaisy", (0.92, 0.90, 0.82), rough=0.52)
    soil_mat = _make_simple_pbr_mat("OrchardMossySoil", (0.042, 0.030, 0.018), rough=0.95, bump_str=0.35)

    bark_mat = _make_gnarled_bark_mat("WoolsthorpeGnarledBark")
    leaf_mat = _make_translucent_leaf_mat("WoolsthorpeSunlitCanopyLeaf")
    apple_mat = _make_ruby_apple_mat("WoolsthorpeCanopyApple")

    oak_mat = _make_oak_walnut_mat("ManorDarkWalnutOak", dark=True)
    warm_oak_mat = _make_oak_walnut_mat("CarvedEnglishOak", dark=False)
    brass_mat = _make_simple_pbr_mat("CambridgeBrass", (0.90, 0.65, 0.22), rough=0.17, metallic=0.96, bump_str=0.020)
    bronze_mat = _make_antique_bronze_mat("AgedGardenBronze")
    velvet_mat = _make_simple_pbr_mat("RoyalCrimsonVelvet", (0.28, 0.024, 0.036), rough=0.88, metallic=0.0)

    # V9 Micro-Storytelling PBR & Emissive Materials
    wrought_iron_mat = _make_wrought_iron_mat("ManorBlackenedWroughtIron")
    ivy_leaf_mat = _make_english_ivy_leaf_mat("CotswoldClimbingEnglishIvy")
    drifting_leaf_mat = _make_drifting_autumn_leaf_mat("GoldenHourDriftingOrchardLeaf")
    casement_glass_mat, casement_em_sock = _make_candlelit_glass_mat("ManorCasementCandleGlass", base_strength=1.8)
    lantern_glass_mat, lantern_glass_em_sock = _make_candlelit_glass_mat("BalustradeLanternAmberGlass", base_strength=0.55)
    lantern_flame_mat, _, lantern_flame_em_sock = _make_emissive_mat("BalustradeLanternFlameGold", (1.0, 0.74, 0.24), strength=2.4)
    firefly_mat, _, _ = _make_emissive_mat("TwilightFireflyWarmGold", (1.0, 0.80, 0.24), strength=8.5)

    star_gold_mat, _, _ = _make_emissive_mat("CosmosStarWarmGold", (1.0, 0.88, 0.60), strength=5.6)
    star_sapphire_mat, _, _ = _make_emissive_mat("CosmosStarSapphireWhite", (0.76, 0.90, 1.0), strength=5.4)
    crescent_moon_mat, moon_vis_fac = _make_crescent_moon_mat()
    godray_mat, godray_opacity_sock = _make_godray_mat()
    pollen_mat, pollen_color_sock, pollen_strength_sock = _make_emissive_mat("GoldenOrchardPollen", (1.0, 0.80, 0.34), strength=2.6)

    # 1. Animated Physical Sky (NISHITA Golden-Hour Sunset -> Deep-Sapphire Milky Way Starlight Nebula Cosmos)
    world, sky_night_fac = _build_animated_physical_sky(scene)

    # 2. 160 True Random Spherical Stars + Sculpted 3D Crescent Moon
    star_entries, crescent_moon_obj = _build_starfield_and_crescent_moon(
        scene, star_gold_mat, star_sapphire_mat, crescent_moon_mat
    )

    # 3. Rolling Meadow Terrain, Weathered Dew-Sheen Cotswold Flagstones, Atmospheric Hills & 140 Grass/Wildflower Clumps
    meadow_obj, terrace_obj, hill_objs, grass_clumps_obj, wildflowers_obj = _build_meadow_terrace_and_hills(
        scene,
        meadow_grass_mat,
        flagstone_mat,
        stone_mat,
        hill_mid_mat,
        hill_far_mat,
        distant_tree_mat,
        grass_blade_mat,
        wildflower_gold_mat,
        wildflower_white_mat,
        soil_mat,
    )

    # 4. Carved Cotswold Stone Garden Balustrade (18 turned stone urns + flanking finials + middle lantern seat)
    balustrade_objs = _build_carved_stone_balustrade(scene, stone_mat)

    # 5. Monumental 3.5m Photoreal Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0) + 3 Fallen Heirloom Apples
    tree_wood_obj, tree_canopy_obj, canopy_apples_obj = _build_woolsthorpe_apple_tree(
        scene, bark_mat, leaf_mat, apple_mat
    )
    fallen_apples_obj = _build_fallen_heirloom_apples(scene, apple_mat, bark_mat)

    # 6. Shaded Left Cotswold Stone Manor Garden Loggia / Optical Alcove + Warm Candlelit Leaded Casement Window
    loggia_objs = _build_cotswold_optical_loggia(
        scene, ashlar_mat, alcove_dark_stone_mat, stone_mat, oak_mat, brass_mat
    )
    casement_window_objs = _build_manor_casement_window(
        scene, stone_mat, oak_mat, wrought_iron_mat, casement_glass_mat
    )

    # 7. Three Sculpted Stone Garden Plinths with Carved Antique-Bronze Classical Laurel Accent Rings
    apple_plinth_obj, principia_plinth_obj, cradle_plinth_obj = _build_garden_plinths(
        scene, stone_mat, brass_mat, bronze_mat
    )

    # 8. V9 Climbing Dark-Emerald English Ivy (~220 matte leaves, 1 fast mesh) & Flickering Wrought-Iron Garden Lantern
    climbing_ivy_obj = _build_climbing_english_ivy(scene, bark_mat, ivy_leaf_mat)
    lantern_obj, lantern_flame_obj, lantern_light_obj = _build_balustrade_garden_lantern(
        scene, wrought_iron_mat, brass_mat, lantern_glass_mat, lantern_flame_mat
    )

    # 9. Volumetric Golden God-Rays + 45 Golden Pollen Motes + 18 Drifting Leaves (B1-4) + 24 Twilight Fireflies (B5-6)
    godray_objs, pollen_entries = _build_godrays_and_pollen(scene, godray_mat, pollen_mat)
    drifting_leaves, fireflies = _build_drifting_leaves_and_fireflies(scene, drifting_leaf_mat, firefly_mat)

    # 10. Hollywood Golden-Hour Chiaroscuro -> Moonlit Twilight Lighting Rig
    lights = _build_outdoor_lighting_rig(scene)

    materials = {
        "stone_mat": stone_mat,
        "ashlar_mat": ashlar_mat,
        "alcove_dark_stone_mat": alcove_dark_stone_mat,
        "flagstone_mat": flagstone_mat,
        "meadow_grass_mat": meadow_grass_mat,
        "bark_mat": bark_mat,
        "leaf_mat": leaf_mat,
        "apple_mat": apple_mat,
        "oak_mat": oak_mat,
        "warm_oak_mat": warm_oak_mat,
        "brass_mat": brass_mat,
        "bronze_mat": bronze_mat,
        "velvet_mat": velvet_mat,
        "star_gold_mat": star_gold_mat,
        "star_sapphire_mat": star_sapphire_mat,
        "crescent_moon_mat": crescent_moon_mat,
        "godray_mat": godray_mat,
        "pollen_mat": pollen_mat,
        "wrought_iron_mat": wrought_iron_mat,
        "ivy_leaf_mat": ivy_leaf_mat,
        "drifting_leaf_mat": drifting_leaf_mat,
        "casement_glass_mat": casement_glass_mat,
        "lantern_glass_mat": lantern_glass_mat,
        "lantern_flame_mat": lantern_flame_mat,
        "firefly_mat": firefly_mat,
    }

    world_handles = {
        # Exact V7 anchor coordinates
        "FLOOR_Z": FLOOR_Z,
        "HERO_APPLE_ATTACH": HERO_APPLE_ATTACH.copy(),
        "APPLE_PLINTH": APPLE_PLINTH.copy(),
        "PRINCIPIA_PLINTH": PRINCIPIA_PLINTH.copy(),
        "PRISM_ANCHOR": PRISM_ANCHOR.copy(),
        "PRISM_PINHOLE": PRISM_PINHOLE.copy(),
        "CRADLE_PLINTH": CRADLE_PLINTH.copy(),
        "ORBIT_ANCHOR": ORBIT_ANCHOR.copy(),
        # Sky & Cosmos handles
        "world": world,
        "sky_night_fac": sky_night_fac,
        "stars": star_entries,
        "crescent_moon": crescent_moon_obj,
        "crescent_moon_fac": moon_vis_fac,
        # Atmosphere & V9 Micro-Storytelling handles
        "godrays": godray_objs,
        "godray_opacity": godray_opacity_sock,
        "pollen_motes": pollen_entries,
        "pollen_color": pollen_color_sock,
        "pollen_strength": pollen_strength_sock,
        "drifting_leaves": drifting_leaves,
        "fireflies": fireflies,
        "casement_window": casement_window_objs,
        "casement_emission": casement_em_sock,
        "climbing_ivy": climbing_ivy_obj,
        "garden_lantern": lantern_obj,
        "lantern_flame": lantern_flame_obj,
        "lantern_glass_emission": lantern_glass_em_sock,
        "lantern_flame_emission": lantern_flame_em_sock,
        "lantern_light": lantern_light_obj,
        "fallen_apples": fallen_apples_obj,
        # Lights
        "sun_light": lights["sun_light"],
        "key_light": lights["key_light"],
        "rim_light": lights["rim_light"],
        "fill_light": lights["fill_light"],
        "catch_light": lights["catch_light"],
        "lights": lights,
        # Architectural & Botanical Objects
        "meadow_terrain": meadow_obj,
        "flagstone_terrace": terrace_obj,
        "distant_hills": hill_objs,
        "grass_clumps": grass_clumps_obj,
        "wildflowers": wildflowers_obj,
        "balustrade": balustrade_objs,
        "apple_tree_wood": tree_wood_obj,
        "apple_tree_canopy": tree_canopy_obj,
        "canopy_apples": canopy_apples_obj,
        "loggia": loggia_objs,
        "apple_plinth": apple_plinth_obj,
        "principia_plinth": principia_plinth_obj,
        "cradle_plinth": cradle_plinth_obj,
        "plinths": {
            "apple": apple_plinth_obj,
            "principia": principia_plinth_obj,
            "cradle": cradle_plinth_obj,
        },
        # Shared PBR Materials (top-level keys + nested dict for ergonomic consumption)
        **materials,
        "materials": materials,
        # Internal keyframe cache
        "_last_anim_frame": None,
        "_stars_init_keyed": False,
        "_leaves_hide_keyed": False,
        "_fireflies_init_keyed": False,
    }
    return world_handles


def animate_world_frame(world_handles: dict, frame: int, t: float) -> None:
    """Keyframe Golden-Hour -> Starry Milky Way Twilight transition, stars, crescent moon, sunbeams, pollen,
    candlelit casement window, flickering garden lantern, 18 tumbling leaves, 24 twilight fireflies & lighting."""
    # Smooth golden-hour -> deep starry twilight blend over t = 19.8..22.2s
    night_fac = smoothstep(19.8, 22.2, t)

    # 1. Animated Physical Sky Mix Factor
    sky_night_fac = world_handles["sky_night_fac"]
    sky_night_fac.default_value = night_fac
    sky_night_fac.keyframe_insert(data_path="default_value", frame=frame)

    # 2. 160 True 3D Spherical Stars (scaled smoothly 0.0001 -> base_r * twinkle in Beats 5-6)
    last_f = world_handles.get("_last_anim_frame")
    is_sparse = last_f is None or abs(frame - last_f) > 1
    need_star_keys = (
        night_fac > 1e-5
        or not world_handles.get("_stars_init_keyed", False)
        or abs(t - 19.75) < 0.18
        or is_sparse
    )
    if need_star_keys:
        for star_obj, base_r, freq, phase in world_handles["stars"]:
            if night_fac <= 1e-5:
                s_val = 0.0001
            else:
                twinkle = 0.80 + 0.20 * math.sin(t * freq + phase)
                s_val = lerp(0.0001, base_r * twinkle, night_fac)
            star_obj.scale = (s_val, s_val, s_val)
            star_obj.keyframe_insert(data_path="scale", frame=frame)
        world_handles["_stars_init_keyed"] = True

    # 3. Sculpted 3D Crescent Moon smooth fade-in (Beats 5-6)
    moon_obj = world_handles["crescent_moon"]
    moon_fac_sock = world_handles["crescent_moon_fac"]
    m_scale = lerp(0.0001, 1.0, night_fac)
    moon_obj.scale = (m_scale, m_scale, m_scale)
    moon_obj.keyframe_insert(data_path="scale", frame=frame)
    moon_fac_sock.default_value = night_fac
    moon_fac_sock.keyframe_insert(data_path="default_value", frame=frame)

    # 4. Volumetric Golden-Hour God-Rays (gently fade down into twilight + slight Beat 3 shade dip)
    beat3_shade = smoothstep(9.2, 9.8, t) * (1.0 - smoothstep(15.2, 15.8, t))
    godray_op = (1.0 - night_fac) * (1.0 - 0.35 * beat3_shade)
    godray_sock = world_handles["godray_opacity"]
    godray_sock.default_value = godray_op
    godray_sock.keyframe_insert(data_path="default_value", frame=frame)

    # 5. 45 Randomly Scattered 3D Golden Pollen / Dust Motes (organic 3D breeze drift!)
    for pm_obj, base_pos, (fx, fy, fz, px, py, pz, pr) in world_handles["pollen_motes"]:
        pm_obj.location = Vector((
            base_pos.x + 0.062 * math.sin(t * fx + px) + 0.022 * math.cos(t * 0.45 + py),
            base_pos.y + 0.042 * math.cos(t * fy + py),
            base_pos.z + 0.052 * math.sin(t * fz + pz),
        ))
        pm_obj.keyframe_insert(data_path="location", frame=frame)

    pollen_col = world_handles["pollen_color"]
    pollen_col.default_value = (
        lerp(1.00, 0.76, night_fac),
        lerp(0.80, 0.88, night_fac),
        lerp(0.34, 1.00, night_fac),
        1.0,
    )
    pollen_col.keyframe_insert(data_path="default_value", frame=frame)

    pollen_str = world_handles["pollen_strength"]
    pollen_str.default_value = lerp(2.6, 1.55, night_fac)
    pollen_str.keyframe_insert(data_path="default_value", frame=frame)

    # 6. V9 Warm Candlelit Manor Casement Window (1.8 daylight -> 3.6 twilight with gentle hearth flicker)
    casement_em = world_handles.get("casement_emission")
    if casement_em is not None:
        c_flicker = 1.0 + 0.05 * math.sin(t * 8.3 + 0.7) + 0.03 * math.cos(t * 14.9)
        casement_em.default_value = lerp(1.8, 3.6, night_fac) * c_flicker
        casement_em.keyframe_insert(data_path="default_value", frame=frame)

    # 7. V9 Flickering Wrought-Iron Balustrade Garden Lantern (warm amber counterpoint to Beats 5-6 moonlight!)
    l_flicker = 1.0 + 0.10 * math.sin(t * 13.7 + 1.1) + 0.06 * math.cos(t * 22.9 + 0.4) + 0.04 * math.sin(t * 6.4)
    lantern_glass_em = world_handles.get("lantern_glass_emission")
    if lantern_glass_em is not None:
        lantern_glass_em.default_value = lerp(0.55, 3.80, night_fac) * (0.94 + 0.06 * l_flicker)
        lantern_glass_em.keyframe_insert(data_path="default_value", frame=frame)
    lantern_flame_em = world_handles.get("lantern_flame_emission")
    if lantern_flame_em is not None:
        lantern_flame_em.default_value = lerp(2.40, 11.50, night_fac) * l_flicker
        lantern_flame_em.keyframe_insert(data_path="default_value", frame=frame)
    lantern_flame_obj = world_handles.get("lantern_flame")
    if lantern_flame_obj is not None:
        fl_base = lerp(0.70, 1.08, night_fac)
        lantern_flame_obj.scale = (
            fl_base * (1.0 + 0.05 * math.sin(t * 17.3)),
            fl_base * (1.0 + 0.05 * math.cos(t * 15.1)),
            fl_base * (1.0 + 0.10 * math.sin(t * 12.6)),
        )
        lantern_flame_obj.keyframe_insert(data_path="scale", frame=frame)
    lantern_light_obj = world_handles.get("lantern_light")
    if lantern_light_obj is not None:
        lantern_ld = lantern_light_obj.data
        lantern_ld.energy = lerp(0.0, 18.0, night_fac) * l_flicker
        lantern_ld.keyframe_insert(data_path="energy", frame=frame)

    # 8. V9 18 Gently Tumbling Golden-Hour Orchard Leaves (Beats 1-4 -> smoothly hidden in Beats 5-6)
    day_leaf_fac = 1.0 - smoothstep(19.6, 21.4, t)
    need_leaf_keys = (
        day_leaf_fac > 1e-5
        or not world_handles.get("_leaves_hide_keyed", False)
        or abs(t - 21.4) < 0.25
        or is_sparse
    )
    if need_leaf_keys and "drifting_leaves" in world_handles:
        z_min, z_max = 0.18, 2.55
        z_span = z_max - z_min
        for lf_obj, base_pos, (fall_spd, sp_r, wx, wy, spin_z, ph1, ph2, base_s) in world_handles["drifting_leaves"]:
            if day_leaf_fac <= 1e-5:
                lf_obj.scale = (0.0001, 0.0001, 0.0001)
                lf_obj.keyframe_insert(data_path="scale", frame=frame)
            else:
                cycle_pos = (base_pos.z - z_min - fall_spd * t) % z_span
                z_cur = z_min + cycle_pos
                cycle_u = cycle_pos / z_span
                # Seamless soft fade at wrap boundaries so no leaf ever pops visibly
                edge_fade = smoothstep(0.0, 0.08, cycle_u) * (1.0 - smoothstep(0.92, 1.0, cycle_u))
                s_cur = max(0.0001, base_s * day_leaf_fac * edge_fade)
                lf_obj.location = Vector((
                    base_pos.x + sp_r * math.sin(t * wx + ph1) + 0.06 * math.sin(t * 0.55 + ph2),
                    base_pos.y + sp_r * 0.65 * math.cos(t * wy + ph2),
                    z_cur,
                ))
                lf_obj.rotation_euler = Euler((
                    0.58 * math.sin(t * wx + ph1),
                    0.46 * math.cos(t * wy + ph2),
                    ph2 + t * spin_z,
                ), "XYZ")
                lf_obj.scale = (s_cur, s_cur, s_cur)
                lf_obj.keyframe_insert(data_path="location", frame=frame)
                lf_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                lf_obj.keyframe_insert(data_path="scale", frame=frame)
        if day_leaf_fac <= 1e-5:
            world_handles["_leaves_hide_keyed"] = True

    # 9. V9 24 Magical Warm-Gold Twilight Fireflies (strictly Z=0.25..1.35 ONLY in Beats 5-6: smoothstep(20.0, 21.8, t))
    firefly_fac = smoothstep(20.0, 21.8, t)
    need_ff_keys = (
        firefly_fac > 1e-5
        or not world_handles.get("_fireflies_init_keyed", False)
        or abs(t - 19.95) < 0.20
        or is_sparse
    )
    if need_ff_keys and "fireflies" in world_handles:
        for ff_obj, base_pos, (dfx, dfy, dfz, pfreq, px, py, pz, pph, base_r) in world_handles["fireflies"]:
            if firefly_fac <= 1e-5:
                ff_obj.scale = (0.0001, 0.0001, 0.0001)
                ff_obj.keyframe_insert(data_path="scale", frame=frame)
            else:
                ff_z = max(0.25, min(1.35, base_pos.z + 0.085 * math.sin(t * dfz + pz)))
                ff_obj.location = Vector((
                    base_pos.x + 0.115 * math.sin(t * dfx + px) + 0.045 * math.cos(t * 0.58 + py),
                    base_pos.y + 0.085 * math.cos(t * dfy + py),
                    ff_z,
                ))
                pulse = 0.30 + 0.70 * ((0.5 + 0.5 * math.sin(t * pfreq + pph)) ** 1.6)
                r_cur = lerp(0.0001, base_r * pulse, firefly_fac)
                ff_obj.scale = (r_cur, r_cur, r_cur)
                ff_obj.keyframe_insert(data_path="location", frame=frame)
                ff_obj.keyframe_insert(data_path="scale", frame=frame)
        world_handles["_fireflies_init_keyed"] = True

    # 10. Hollywood Golden-Hour Chiaroscuro -> Moonlit Twilight Lighting Rig (with Beat 3 shaded alcove dip)
    sun_ld = world_handles["sun_light"].data
    sun_ld.energy = lerp(4.6, 1.15, night_fac)
    sun_ld.color = (lerp(1.00, 0.42, night_fac), lerp(0.72, 0.68, night_fac), lerp(0.40, 1.00, night_fac))
    sun_ld.keyframe_insert(data_path="energy", frame=frame)
    sun_ld.keyframe_insert(data_path="color", frame=frame)

    key_ld = world_handles["key_light"].data
    day_key_e = 270.0 * (1.0 - 0.36 * beat3_shade)
    key_ld.energy = lerp(day_key_e, 140.0, night_fac)
    key_ld.color = (lerp(1.00, 0.58, night_fac), lerp(0.76, 0.78, night_fac), lerp(0.46, 1.00, night_fac))
    key_ld.keyframe_insert(data_path="energy", frame=frame)
    key_ld.keyframe_insert(data_path="color", frame=frame)

    rim_ld = world_handles["rim_light"].data
    day_rim_e = 245.0 * (1.0 - 0.25 * beat3_shade)
    rim_ld.energy = lerp(day_rim_e, 370.0, night_fac)
    rim_ld.color = (lerp(0.38, 0.28, night_fac), lerp(0.70, 0.64, night_fac), 1.00)
    rim_ld.keyframe_insert(data_path="energy", frame=frame)
    rim_ld.keyframe_insert(data_path="color", frame=frame)

    fill_ld = world_handles["fill_light"].data
    day_fill_e = 40.0 * (1.0 - 0.48 * beat3_shade)
    fill_ld.energy = lerp(day_fill_e, 22.0, night_fac)
    fill_ld.color = (lerp(0.90, 0.42, night_fac), lerp(0.84, 0.58, night_fac), lerp(0.74, 0.90, night_fac))
    fill_ld.keyframe_insert(data_path="energy", frame=frame)
    fill_ld.keyframe_insert(data_path="color", frame=frame)

    catch_ld = world_handles["catch_light"].data
    catch_ld.energy = lerp(18.0, 22.0, night_fac)
    catch_ld.color = (lerp(1.00, 0.82, night_fac), lerp(0.95, 0.94, night_fac), lerp(0.88, 1.00, night_fac))
    catch_ld.keyframe_insert(data_path="energy", frame=frame)
    catch_ld.keyframe_insert(data_path="color", frame=frame)

    world_handles["_last_anim_frame"] = frame


__all__ = [
    "FLOOR_Z",
    "HERO_APPLE_ATTACH",
    "APPLE_PLINTH",
    "PRINCIPIA_PLINTH",
    "PRISM_ANCHOR",
    "PRISM_FLOOR",
    "PRISM_PINHOLE",
    "CRADLE_PLINTH",
    "ORBIT_ANCHOR",
    "ORBIT_FLOOR",
    "TREE_BASE",
    "CRESCENT_MOON_POS",
    "clamp01",
    "smoothstep",
    "lerp",
    "set_bsdf",
    "get_rgba_mix_sockets",
    "build_woolsthorpe_world",
    "animate_world_frame",
]
