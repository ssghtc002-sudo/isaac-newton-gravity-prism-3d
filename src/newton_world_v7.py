"""
V7 Outdoor Woolsthorpe Orchard & Cosmos Environment Module for Scientist #9: Isaac Newton
=========================================================================================

100% brand-new outdoor-to-cosmos Cycles environment (ZERO indoor brick room, ZERO bookshelves,
ZERO shopkeeper counter, ZERO blocky rotated arch cubes, ZERO linear stars, ZERO black moon ball!).

Features:
  1. Animated Physical Cycles Sky: NISHITA Golden-Hour Afternoon (Beats 1-4) smoothly blending
     into a Deep Starry Celestial Twilight Sky (Beats 5-6, t = 19.8..22.2s).
  2. 160 True 3D Random Spherical Stars (warm-gold & sapphire-white, random.Random(1687)) +
     True 3D Sculpted Crescent Moon (pure smooth bmesh lunar crescent, ZERO black occluder!).
  3. Rolling Woolsthorpe Meadow Grass Terrain, Curved Mossy Cotswold Limestone Flagstone
     Courtyard (exact Z = 0.000), Carved Cotswold Stone Garden Balustrade (18 turned stone urn
     balusters + 3 pier finials), 3 Atmospheric Rolling Countryside Hill Ridges with distant
     English tree clusters, and 140 merged 3D Meadow Grass Clumps & Golden/White Wildflowers.
  4. Monumental 3.6m-tall Sculpted Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0) with
     gnarled bark & buttress roots, 5 sweeping organic boughs including the continuous Hero
     Overhanging Apple Bough terminating at exact HERO_APPLE_ATTACH = (-0.52, 0.12, 1.94),
     14 organic canopy clusters + 280 sculpted leaves with backlit translucent Cycles shader,
     and 12 glossy ruby-gold heirloom hanging apples.
  5. Shaded Left Cotswold Stone Manor Garden Loggia / Optical Alcove (X in [-3.75, -1.80])
     featuring smooth continuous semicircular carved stone arches, dark walnut timber optical
     shutter with a physical circular brass-rimmed pinhole at exact PRISM_PINHOLE = (-3.38, 0.06, 1.54),
     and pitch-rich shadowed dark limestone back wall for Beat 3's 7-spectral rainbow showstopper.
  6. Three Sculpted Cotswold Stone Garden Plinths (APPLE_PLINTH, PRINCIPIA_PLINTH, CRADLE_PLINTH)
     plus open flagstone anchors (PRISM_ANCHOR, ORBIT_ANCHOR).
  7. Volumetric Golden-Hour God-Rays + 45 Randomly Scattered 3D Drifting Golden Pollen Motes
     (random.Random(1666)).
  8. Outdoor Golden-Hour Sun & Rembrandt Lighting Rig animating smoothly to Moonlit Twilight.
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
CRESCENT_MOON_POS = Vector((1.80, 19.50, 7.80))


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
    """Warm honey-golden Cotswold limestone (or deep shadowed charcoal-stone for Beat 3 loggia)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_macro.inputs["Scale"].default_value = 4.2
    ns_macro.inputs["Detail"].default_value = 8.0
    ns_macro.inputs["Roughness"].default_value = 0.62

    ns_fine = nt.nodes.new("ShaderNodeTexNoise")
    ns_fine.inputs["Scale"].default_value = 22.0
    ns_fine.inputs["Detail"].default_value = 9.0
    ns_fine.inputs["Roughness"].default_value = 0.68

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    if dark:
        rmp.color_ramp.elements[0].position = 0.22
        rmp.color_ramp.elements[0].color = (0.022, 0.020, 0.024, 1.0)
        e_mid = rmp.color_ramp.elements.new(0.55)
        e_mid.color = (0.044, 0.039, 0.038, 1.0)
        rmp.color_ramp.elements[-1].position = 0.86
        rmp.color_ramp.elements[-1].color = (0.072, 0.062, 0.054, 1.0)
    else:
        rmp.color_ramp.elements[0].position = 0.20
        rmp.color_ramp.elements[0].color = (0.24, 0.19, 0.125, 1.0)
        e_mid = rmp.color_ramp.elements.new(0.55)
        e_mid.color = (0.37, 0.30, 0.205, 1.0)
        rmp.color_ramp.elements[-1].position = 0.85
        rmp.color_ramp.elements[-1].color = (0.48, 0.40, 0.285, 1.0)

    # Subtle weathered garden moss undertone near crevices
    ns_moss = nt.nodes.new("ShaderNodeTexNoise")
    ns_moss.inputs["Scale"].default_value = 3.1
    ns_moss.inputs["Detail"].default_value = 6.0
    moss_rmp = nt.nodes.new("ShaderNodeValToRGB")
    moss_rmp.color_ramp.elements[0].position = 0.62
    moss_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    moss_rmp.color_ramp.elements[1].position = 0.86
    moss_rmp.color_ramp.elements[1].color = (0.26 if not dark else 0.12, 0.26 if not dark else 0.12, 0.26 if not dark else 0.12, 1.0)

    mix_moss = nt.nodes.new("ShaderNodeMix")
    mix_moss.data_type = "RGBA"
    mix_moss.blend_type = "MIX"
    mm_fac, mm_a, mm_b, mm_out = get_rgba_mix_sockets(mix_moss)
    mm_b.default_value = (0.045, 0.095, 0.028, 1.0) if not dark else (0.018, 0.036, 0.016, 1.0)

    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.inputs["From Min"].default_value = 0.20
    r_map.inputs["From Max"].default_value = 0.80
    r_map.inputs["To Min"].default_value = 0.60 if not dark else 0.72
    r_map.inputs["To Max"].default_value = 0.84 if not dark else 0.90

    bmp_fine = nt.nodes.new("ShaderNodeBump")
    bmp_fine.inputs["Strength"].default_value = 0.14
    bmp_macro = nt.nodes.new("ShaderNodeBump")
    bmp_macro.inputs["Strength"].default_value = 0.20

    nt.links.new(tc.outputs["Object"], ns_macro.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_fine.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_moss.inputs["Vector"])
    nt.links.new(ns_macro.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(ns_moss.outputs["Fac"], moss_rmp.inputs["Fac"])
    nt.links.new(moss_rmp.outputs["Color"], mm_fac)
    nt.links.new(rmp.outputs["Color"], mm_a)
    nt.links.new(mm_out, bsdf.inputs["Base Color"])
    nt.links.new(ns_fine.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns_fine.outputs["Fac"], bmp_fine.inputs["Height"])
    nt.links.new(ns_macro.outputs["Fac"], bmp_macro.inputs["Height"])
    nt.links.new(bmp_fine.outputs["Normal"], bmp_macro.inputs["Normal"])
    nt.links.new(bmp_macro.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_cotswold_ashlar_mat(name: str = "CotswoldManorAshlar", dark: bool = False):
    """Weathered Cotswold manor ashlar stone blocks with mortar relief (for exterior loggia & dark alcove)."""
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
        br.inputs["Color1"].default_value = (0.042, 0.038, 0.036, 1.0)
        br.inputs["Color2"].default_value = (0.026, 0.024, 0.025, 1.0)
        br.inputs["Mortar"].default_value = (0.010, 0.010, 0.011, 1.0)
    else:
        br.inputs["Color1"].default_value = (0.36, 0.285, 0.190, 1.0)
        br.inputs["Color2"].default_value = (0.26, 0.205, 0.135, 1.0)
        br.inputs["Mortar"].default_value = (0.11, 0.095, 0.072, 1.0)
    br.inputs["Scale"].default_value = 2.6
    br.inputs["Mortar Size"].default_value = 0.018
    br.inputs["Mortar Smooth"].default_value = 0.18
    br.inputs["Brick Width"].default_value = 0.65
    br.inputs["Row Height"].default_value = 0.28

    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 8.5
    ns.inputs["Detail"].default_value = 9.0
    ns.inputs["Roughness"].default_value = 0.65

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.25
    rmp.color_ramp.elements[0].color = (0.62, 0.58, 0.52, 1.0)
    rmp.color_ramp.elements[1].position = 0.80
    rmp.color_ramp.elements[1].color = (1.0, 0.96, 0.90, 1.0)

    mix_w = nt.nodes.new("ShaderNodeMix")
    mix_w.data_type = "RGBA"
    mix_w.blend_type = "MULTIPLY"
    mw_fac, mw_a, mw_b, mw_out = get_rgba_mix_sockets(mix_w)
    mw_fac.default_value = 0.38

    bmp_g = nt.nodes.new("ShaderNodeBump")
    bmp_g.inputs["Strength"].default_value = 0.15
    bmp_m = nt.nodes.new("ShaderNodeBump")
    bmp_m.inputs["Strength"].default_value = 0.34 if not dark else 0.22

    set_bsdf(bsdf, "Roughness", 0.78 if not dark else 0.86)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(br.outputs["Color"], mw_a)
    nt.links.new(rmp.outputs["Color"], mw_b)
    nt.links.new(mw_out, bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp_g.inputs["Height"])
    nt.links.new(br.outputs["Fac"], bmp_m.inputs["Height"])
    nt.links.new(bmp_g.outputs["Normal"], bmp_m.inputs["Normal"])
    nt.links.new(bmp_m.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_mossy_flagstone_mat(name: str = "CotswoldMossyFlagstone"):
    """Warm honey Cotswold limestone courtyard paving slabs with Voronoi/Ashlar seams & mossy crevices."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Organic slight warp so paving seams look hand-laid antique limestone
    ns_warp = nt.nodes.new("ShaderNodeTexNoise")
    ns_warp.inputs["Scale"].default_value = 2.4
    ns_warp.inputs["Detail"].default_value = 4.0

    vec_scale = nt.nodes.new("ShaderNodeVectorMath")
    vec_scale.operation = "SCALE"
    vec_scale.inputs["Scale"].default_value = 0.045

    vec_add = nt.nodes.new("ShaderNodeVectorMath")
    vec_add.operation = "ADD"

    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.42
    br.inputs["Color1"].default_value = (0.34, 0.275, 0.185, 1.0)
    br.inputs["Color2"].default_value = (0.24, 0.195, 0.132, 1.0)
    br.inputs["Mortar"].default_value = (0.048, 0.088, 0.028, 1.0)  # Mossy mortar crevice!
    br.inputs["Scale"].default_value = 2.15
    br.inputs["Mortar Size"].default_value = 0.026
    br.inputs["Mortar Smooth"].default_value = 0.24
    br.inputs["Brick Width"].default_value = 0.72
    br.inputs["Row Height"].default_value = 0.48

    # Secondary Voronoi flagstone mineral speckle + weathering
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 3.2
    ns_stone = nt.nodes.new("ShaderNodeTexNoise")
    ns_stone.inputs["Scale"].default_value = 11.0
    ns_stone.inputs["Detail"].default_value = 9.0
    ns_stone.inputs["Roughness"].default_value = 0.64

    rmp_tint = nt.nodes.new("ShaderNodeValToRGB")
    rmp_tint.color_ramp.elements[0].position = 0.20
    rmp_tint.color_ramp.elements[0].color = (0.72, 0.68, 0.60, 1.0)
    rmp_tint.color_ramp.elements[1].position = 0.85
    rmp_tint.color_ramp.elements[1].color = (1.08, 1.02, 0.92, 1.0)

    mix_tint = nt.nodes.new("ShaderNodeMix")
    mix_tint.data_type = "RGBA"
    mix_tint.blend_type = "MULTIPLY"
    mt_fac, mt_a, mt_b, mt_out = get_rgba_mix_sockets(mix_tint)
    mt_fac.default_value = 0.42

    # Moss creep around mortar joints & shady corners
    ns_moss = nt.nodes.new("ShaderNodeTexNoise")
    ns_moss.inputs["Scale"].default_value = 5.5
    ns_moss.inputs["Detail"].default_value = 8.0
    moss_rmp = nt.nodes.new("ShaderNodeValToRGB")
    moss_rmp.color_ramp.elements[0].position = 0.56
    moss_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    moss_rmp.color_ramp.elements[1].position = 0.82
    moss_rmp.color_ramp.elements[1].color = (0.45, 0.45, 0.45, 1.0)

    mix_moss = nt.nodes.new("ShaderNodeMix")
    mix_moss.data_type = "RGBA"
    mix_moss.blend_type = "MIX"
    mm_fac, mm_a, mm_b, mm_out = get_rgba_mix_sockets(mix_moss)
    mm_b.default_value = (0.052, 0.125, 0.028, 1.0)

    # Roughness: worn limestone centers (0.54) -> matte mossy joints (0.86)
    r_map = nt.nodes.new("ShaderNodeMapRange")
    r_map.inputs["From Min"].default_value = 0.0
    r_map.inputs["From Max"].default_value = 1.0
    r_map.inputs["To Min"].default_value = 0.54
    r_map.inputs["To Max"].default_value = 0.86

    bmp_grain = nt.nodes.new("ShaderNodeBump")
    bmp_grain.inputs["Strength"].default_value = 0.14
    bmp_seam = nt.nodes.new("ShaderNodeBump")
    bmp_seam.inputs["Strength"].default_value = 0.36
    bmp_seam.invert = True

    nt.links.new(tc.outputs["Object"], ns_warp.inputs["Vector"])
    nt.links.new(ns_warp.outputs["Color"], vec_scale.inputs[0])
    nt.links.new(tc.outputs["Object"], vec_add.inputs[0])
    nt.links.new(vec_scale.outputs["Vector"], vec_add.inputs[1])
    nt.links.new(vec_add.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], ns_stone.inputs["Vector"])
    nt.links.new(ns_stone.outputs["Fac"], rmp_tint.inputs["Fac"])
    nt.links.new(br.outputs["Color"], mt_a)
    nt.links.new(rmp_tint.outputs["Color"], mt_b)
    nt.links.new(tc.outputs["Object"], ns_moss.inputs["Vector"])
    nt.links.new(ns_moss.outputs["Fac"], moss_rmp.inputs["Fac"])
    nt.links.new(moss_rmp.outputs["Color"], mm_fac)
    nt.links.new(mt_out, mm_a)
    nt.links.new(mm_out, bsdf.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], r_map.inputs["Value"])
    nt.links.new(r_map.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(ns_stone.outputs["Fac"], bmp_grain.inputs["Height"])
    nt.links.new(br.outputs["Fac"], bmp_seam.inputs["Height"])
    nt.links.new(bmp_grain.outputs["Normal"], bmp_seam.inputs["Normal"])
    nt.links.new(bmp_seam.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_meadow_grass_mat(name: str = "WoolsthorpeMeadowGrass", distant: bool = False, haze_mix: float = 0.0):
    """Multi-layered procedural emerald Woolsthorpe meadow grass (Voronoi + multi-octave Noise + fine blade bump)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")

    # Macro pasture sunlight & clover patches
    ns_macro = nt.nodes.new("ShaderNodeTexNoise")
    ns_macro.inputs["Scale"].default_value = 0.65 if distant else 1.45
    ns_macro.inputs["Detail"].default_value = 7.0
    ns_macro.inputs["Roughness"].default_value = 0.58

    # Fine grass-blade + moss Voronoi/Noise
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 18.0 if distant else 42.0
    ns_blade = nt.nodes.new("ShaderNodeTexNoise")
    ns_blade.inputs["Scale"].default_value = 24.0 if distant else 58.0
    ns_blade.inputs["Detail"].default_value = 10.0
    ns_blade.inputs["Roughness"].default_value = 0.72

    rmp_macro = nt.nodes.new("ShaderNodeValToRGB")
    cr = rmp_macro.color_ramp
    cr.elements[0].position = 0.22
    cr.elements[0].color = (0.024, 0.088, 0.018, 1.0)  # Deep lush English meadow emerald
    e1 = cr.elements.new(0.54)
    e1.color = (0.056, 0.155, 0.026, 1.0)  # Rich Woolsthorpe orchard green
    cr.elements[-1].position = 0.84
    cr.elements[-1].color = (0.125, 0.215, 0.038, 1.0)  # Warm sunlit golden-clover pasture

    rmp_blade = nt.nodes.new("ShaderNodeValToRGB")
    rmp_blade.color_ramp.elements[0].position = 0.18
    rmp_blade.color_ramp.elements[0].color = (0.68, 0.74, 0.62, 1.0)
    rmp_blade.color_ramp.elements[1].position = 0.84
    rmp_blade.color_ramp.elements[1].color = (1.15, 1.12, 0.92, 1.0)

    mix_b = nt.nodes.new("ShaderNodeMix")
    mix_b.data_type = "RGBA"
    mix_b.blend_type = "MULTIPLY"
    mb_fac, mb_a, mb_b, mb_out = get_rgba_mix_sockets(mix_b)
    mb_fac.default_value = 0.45

    final_col_out = mb_out
    if haze_mix > 0.0:
        mix_haze = nt.nodes.new("ShaderNodeMix")
        mix_haze.data_type = "RGBA"
        mix_haze.blend_type = "MIX"
        mh_fac, mh_a, mh_b, mh_out = get_rgba_mix_sockets(mix_haze)
        mh_fac.default_value = clamp01(haze_mix)
        mh_b.default_value = (0.052, 0.095, 0.120, 1.0)  # Atmospheric misty English valley blue-sage
        nt.links.new(mb_out, mh_a)
        final_col_out = mh_out

    bmp_vor = nt.nodes.new("ShaderNodeBump")
    bmp_vor.inputs["Strength"].default_value = 0.18 if distant else 0.28
    bmp_bl = nt.nodes.new("ShaderNodeBump")
    bmp_bl.inputs["Strength"].default_value = 0.20 if distant else 0.36

    set_bsdf(bsdf, "Roughness", 0.86 if not distant else 0.92)
    set_bsdf(bsdf, "Specular IOR Level", 0.22)

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
    """Gnarled dark-oak Woolsthorpe apple tree bark with deep vertical fissures & pale sage lichen patches."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (4.8, 4.8, 1.35)  # Vertically elongated bark fissures!

    ns_bark = nt.nodes.new("ShaderNodeTexNoise")
    ns_bark.inputs["Scale"].default_value = 7.2
    ns_bark.inputs["Detail"].default_value = 11.0
    ns_bark.inputs["Roughness"].default_value = 0.74
    ns_bark.inputs["Distortion"].default_value = 1.25

    rmp_bark = nt.nodes.new("ShaderNodeValToRGB")
    rmp_bark.color_ramp.elements[0].position = 0.22
    rmp_bark.color_ramp.elements[0].color = (0.028, 0.017, 0.010, 1.0)  # Deep fissure crevice
    e_mid = rmp_bark.color_ramp.elements.new(0.56)
    e_mid.color = (0.088, 0.056, 0.032, 1.0)  # Aged dark-oak orchard bark
    rmp_bark.color_ramp.elements[-1].position = 0.85
    rmp_bark.color_ramp.elements[-1].color = (0.148, 0.102, 0.062, 1.0)  # Sun-bleached bark ridge

    # Organic orchard lichen & moss patches on gnarled trunk
    ns_lich = nt.nodes.new("ShaderNodeTexNoise")
    ns_lich.inputs["Scale"].default_value = 4.6
    ns_lich.inputs["Detail"].default_value = 7.0
    lich_rmp = nt.nodes.new("ShaderNodeValToRGB")
    lich_rmp.color_ramp.elements[0].position = 0.64
    lich_rmp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    lich_rmp.color_ramp.elements[1].position = 0.82
    lich_rmp.color_ramp.elements[1].color = (0.58, 0.58, 0.58, 1.0)

    mix_l = nt.nodes.new("ShaderNodeMix")
    mix_l.data_type = "RGBA"
    mix_l.blend_type = "MIX"
    ml_fac, ml_a, ml_b, ml_out = get_rgba_mix_sockets(mix_l)
    ml_b.default_value = (0.12, 0.175, 0.085, 1.0)  # Pale English orchard lichen!

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.48
    bmp.inputs["Distance"].default_value = 0.022

    set_bsdf(bsdf, "Roughness", 0.88)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns_bark.inputs["Vector"])
    nt.links.new(ns_bark.outputs["Fac"], rmp_bark.inputs["Fac"])
    nt.links.new(tc.outputs["Object"], ns_lich.inputs["Vector"])
    nt.links.new(ns_lich.outputs["Fac"], lich_rmp.inputs["Fac"])
    nt.links.new(lich_rmp.outputs["Color"], ml_fac)
    nt.links.new(rmp_bark.outputs["Color"], ml_a)
    nt.links.new(ml_out, bsdf.inputs["Base Color"])
    nt.links.new(ns_bark.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _make_translucent_leaf_mat(name: str = "WoolsthorpeSunlitCanopyLeaf"):
    """Backlit translucent emerald foliage: MixShader(Principled BSDF, Translucent BSDF, 0.35)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    mix_sh.inputs["Fac"].default_value = 0.35  # 35% sunlit leaf translucency!

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    transl = nt.nodes.new("ShaderNodeBsdfTranslucent")

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 4.5
    ns.inputs["Detail"].default_value = 7.0
    ns.inputs["Roughness"].default_value = 0.60

    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.22
    rmp.color_ramp.elements[0].color = (0.024, 0.135, 0.020, 1.0)
    e1 = rmp.color_ramp.elements.new(0.58)
    e1.color = (0.062, 0.245, 0.032, 1.0)
    rmp.color_ramp.elements[-1].position = 0.88
    rmp.color_ramp.elements[-1].color = (0.145, 0.340, 0.048, 1.0)  # Golden sunlit leaf tips!

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.12

    set_bsdf(bsdf, "Roughness", 0.36)
    set_bsdf(bsdf, "Coat Weight", 0.20)
    set_bsdf(bsdf, "Coat Roughness", 0.18)

    transl.inputs["Color"].default_value = (0.28, 0.56, 0.065, 1.0)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bmp.outputs["Normal"], transl.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(transl.outputs["BSDF"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
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
    length: float = 0.10,
    width: float = 0.048,
) -> None:
    """Append a sculpted arched botanical diamond/almond leaf with subtle V-midrib crease."""
    rmat = rot_euler.to_matrix()

    def xf(lx, ly, lz):
        return origin + (rmat @ Vector((lx, ly, lz)))

    v0 = bm.verts.new(xf(0.0, 0.0, 0.0))
    v1l = bm.verts.new(xf(0.32 * length, 0.46 * width, 0.06 * length))
    v1c = bm.verts.new(xf(0.32 * length, 0.0, 0.02 * length))
    v1r = bm.verts.new(xf(0.32 * length, -0.46 * width, 0.06 * length))
    v2l = bm.verts.new(xf(0.68 * length, 0.44 * width, 0.04 * length))
    v2c = bm.verts.new(xf(0.68 * length, 0.0, -0.01 * length))
    v2r = bm.verts.new(xf(0.68 * length, -0.44 * width, 0.04 * length))
    v3 = bm.verts.new(xf(length, 0.0, -0.10 * length))

    bm.faces.new((v0, v1c, v1l))
    bm.faces.new((v0, v1r, v1c))
    bm.faces.new((v1l, v1c, v2c, v2l))
    bm.faces.new((v1c, v1r, v2r, v2c))
    bm.faces.new((v2l, v2c, v3))
    bm.faces.new((v2c, v2r, v3))


def _bmesh_add_heirloom_apple(bm: bmesh.types.BMesh, center: Vector, radius: float = 0.044) -> None:
    """Append a sculpted 5-lobed dimpled heirloom Woolsthorpe apple into bm."""
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
            ring.append(bm.verts.new((center.x + math.cos(ang) * rr * lobe, center.y + math.sin(ang) * rr * lobe, center.z + zz)))
        rings.append(ring)

    for i in range(n_ring):
        for s_i in range(n_seg):
            sn = (s_i + 1) % n_seg
            bm.faces.new((rings[i][s_i], rings[i][sn], rings[i + 1][sn], rings[i + 1][s_i]))


# ============================================================================
# 5. SECTION BUILDERS (SKY, COSMOS, MEADOW, BALUSTRADE, TREE, LOGGIA, PLINTHS)
# ============================================================================
def _build_animated_physical_sky(scene):
    """Build Cycles World blending NISHITA Golden-Hour Sky (Beats 1-4) -> Deep Starry Twilight (Beats 5-6)."""
    world = bpy.data.worlds.new("WoolsthorpeOrchardCosmosWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    mix_sh.inputs["Fac"].default_value = 0.0  # Daylight at frame 1!

    # Branch A: Real Physical NISHITA Golden-Hour Afternoon Sky (Beats 1-4)
    sky_tex = nt.nodes.new("ShaderNodeTexSky")
    sky_tex.sky_type = "NISHITA"
    sky_tex.sun_disc = False
    sky_tex.sun_elevation = math.radians(11.0)
    sky_tex.sun_rotation = math.radians(38.0)
    sky_tex.air_density = 1.4
    sky_tex.dust_density = 1.8
    sky_tex.ozone_density = 1.5

    bg_day = nt.nodes.new("ShaderNodeBackground")
    bg_day.inputs["Strength"].default_value = 0.65

    # Branch B: Deep Starry Celestial Midnight-Indigo Twilight Gradient (Beats 5-6)
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

    bg_night = nt.nodes.new("ShaderNodeBackground")
    bg_night.inputs["Strength"].default_value = 0.90

    nt.links.new(sky_tex.outputs["Color"], bg_day.inputs["Color"])
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], map_z.inputs["Value"])
    nt.links.new(map_z.outputs["Result"], night_rmp.inputs["Fac"])
    nt.links.new(night_rmp.outputs["Color"], bg_night.inputs["Color"])
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
    # Tree-well circular carved stone collar around TREE_BASE (-0.18, 1.35, 0.0)
    _bmesh_add_lathe(
        bm_curb,
        [(0.48, -0.04), (0.48, 0.06), (0.44, 0.085), (0.37, 0.085), (0.35, 0.02)],
        center=TREE_BASE,
        segs=24,
        cap_bot=False,
        cap_top=False,
    )
    _finalize_bmesh(scene, "AppleTreeStoneWell", bm_curb, mats=stone_mat, smooth=True, edge_split_deg=38.0)

    # Rich dark mossy earth mound inside the tree well so gnarled roots grip real soil
    bm_soil = bmesh.new()
    _bmesh_add_lathe(
        bm_soil,
        [(0.40, -0.02), (0.37, 0.055), (0.24, 0.095), (0.04, 0.11)],
        center=TREE_BASE,
        segs=20,
        cap_bot=False,
        cap_top=True,
    )
    _finalize_bmesh(scene, "AppleTreeSoilMound", bm_soil, mats=soil_mat, smooth=True)

    # 3. Distant Rolling English Countryside Hills (3 Layered Atmospheric Ridges at Y = +10..+23m — 100% OPEN HORIZON!)
    ridge_specs = [
        ("DistantHillRidge_Near", 11.8, 16.2, 1.05, 0.72, 0.25, 0.5, meadow_mat),
        ("DistantHillRidge_Mid", 15.6, 19.6, 1.75, 0.92, 0.21, 1.8, hill_mid_mat),
        ("DistantHillRidge_Far", 18.8, 23.2, 2.35, 1.15, 0.17, 3.1, hill_far_mat),
    ]
    hill_objs = []
    for h_name, y_near, y_far, base_h, amp_h, freq_x, phase, h_mat in ridge_specs:
        bm_h = bmesh.new()
        h_nx, h_ny = 44, 10
        h_grid = []
        for iy in range(h_ny + 1):
            uy = iy / float(h_ny)
            hy = lerp(y_near, y_far, uy)
            ridge_env = math.sin(uy * math.pi)
            row = []
            for ix in range(h_nx + 1):
                ux = ix / float(h_nx)
                hx = lerp(-18.5, 18.5, ux)
                # Keep a graceful scenic low dip in the central sky corridor (X in [-1.5, +3.0]) so moon & stars shine!
                center_saddle = 1.0 - 0.34 * math.exp(-((hx - 0.8) / 4.8) ** 2)
                crest = (
                    base_h
                    + amp_h * math.sin(hx * freq_x + phase)
                    + 0.38 * amp_h * math.cos(hx * freq_x * 1.9 - phase * 0.7)
                ) * center_saddle
                hz = -0.35 + ridge_env * max(0.35, crest)
                row.append(bm_h.verts.new((hx, hy, hz)))
            h_grid.append(row)
        for iy in range(h_ny):
            for ix in range(h_nx):
                bm_h.faces.new((h_grid[iy][ix], h_grid[iy][ix + 1], h_grid[iy + 1][ix + 1], h_grid[iy + 1][ix]))
        hill_objs.append(_finalize_bmesh(scene, h_name, bm_h, mats=h_mat, smooth=True))

    # Distant English Hedgerow & Copse Tree Clusters along the hills (merged single bmesh)
    bm_dt = bmesh.new()
    rng_dt = random.Random(1642)  # Newton's birth year at Woolsthorpe!
    copse_centers = [
        (-8.5, 12.6, 1.35), (-6.8, 13.0, 1.48), (-5.2, 12.2, 1.15), (-3.8, 13.6, 1.40),
        (4.4, 12.8, 1.22), (5.8, 13.4, 1.45), (7.6, 12.5, 1.30), (9.4, 13.8, 1.55),
        (-10.5, 16.8, 2.25), (-7.4, 17.2, 2.15), (-4.2, 17.6, 1.85),
        (5.0, 17.2, 2.05), (8.2, 16.8, 2.30), (11.0, 17.5, 2.40),
    ]
    for cx, cy, cz in copse_centers:
        for _ in range(3):
            ox = cx + rng_dt.uniform(-0.85, 0.85)
            oy = cy + rng_dt.uniform(-0.55, 0.55)
            rad = rng_dt.uniform(0.42, 0.86)
            oz = cz + rad * 0.55
            _bmesh_add_uv_sphere(
                bm_dt,
                Vector((ox, oy, oz)),
                rx=rad,
                ry=rad * 0.88,
                rz=rad * 1.15,
                segs=10,
                rings=7,
                ruffle_amp=0.14,
                ruffle_freq=3.1,
            )
    _finalize_bmesh(scene, "DistantLincolnshireCopses", bm_dt, mats=distant_tree_mat, smooth=True)

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
        # Classical turned Cotswold stone spherical garden finial on top of each pier (Z = 0.96..1.18)
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
    """Build monumental 3.6m-tall sculpted Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0).
    Hero overhanging apple bough terminates seamlessly at exact HERO_APPLE_ATTACH = (-0.52, 0.12, 1.94)!"""
    bm_wood = bmesh.new()

    # 1. Thick Rooted Gnarled Trunk (r=0.28m at mossy flare -> 0.17m at crown fork Z=1.65m)
    trunk_pts = [
        (-0.18, 1.35, -0.06, 0.295),
        (-0.18, 1.35, 0.12, 0.275),
        (-0.20, 1.34, 0.55, 0.225),
        (-0.21, 1.32, 1.05, 0.195),
        (-0.18, 1.30, 1.45, 0.178),
        (-0.16, 1.28, 1.66, 0.168),
    ]
    _bmesh_add_tapered_tube(bm_wood, trunk_pts, radial_segs=14, subdivs=3, bark_ridge=0.055)

    # 5 Flaring Mossy Buttress Roots gripping into the soil well
    root_angles = [0.2, 1.45, 2.75, 4.05, 5.25]
    for r_ang in root_angles:
        ca, sa = math.cos(r_ang), math.sin(r_ang)
        r_pts = [
            (TREE_BASE.x + ca * 0.14, TREE_BASE.y + sa * 0.14, 0.34, 0.105),
            (TREE_BASE.x + ca * 0.27, TREE_BASE.y + sa * 0.27, 0.14, 0.072),
            (TREE_BASE.x + ca * 0.40, TREE_BASE.y + sa * 0.40, 0.02, 0.036),
            (TREE_BASE.x + ca * 0.47, TREE_BASE.y + sa * 0.47, -0.07, 0.018),
        ]
        _bmesh_add_tapered_tube(bm_wood, r_pts, radial_segs=8, subdivs=2, bark_ridge=0.04)

    # 2. Five Sweeping Organic Primary Boughs sprouting cleanly from crown fork (-0.16, 1.28, 1.65)
    # BOUGH 1: HERO OVERHANGING APPLE BOUGH -> arches gracefully forward-left to exact HERO_APPLE_ATTACH!
    hero_bough = [
        (-0.16, 1.28, 1.62, 0.108),
        (-0.23, 1.04, 1.98, 0.084),
        (-0.33, 0.72, 2.18, 0.062),
        (-0.42, 0.42, 2.15, 0.044),
        (-0.49, 0.22, 2.04, 0.026),
        (HERO_APPLE_ATTACH.x, HERO_APPLE_ATTACH.y, HERO_APPLE_ATTACH.z, 0.010),
    ]
    _bmesh_add_tapered_tube(bm_wood, hero_bough, radial_segs=11, subdivs=3, bark_ridge=0.035)

    # Companion fruiting twigs branching cleanly off the Hero Bough (staying above Z >= 2.0m!)
    hero_twig_a = [
        (-0.33, 0.72, 2.16, 0.038),
        (-0.56, 0.62, 2.26, 0.024),
        (-0.78, 0.46, 2.18, 0.013),
        (-0.88, 0.38, 2.09, 0.006),
    ]
    hero_twig_b = [
        (-0.43, 0.40, 2.14, 0.026),
        (-0.32, 0.28, 2.24, 0.016),
        (-0.22, 0.22, 2.28, 0.008),
    ]
    hero_twig_c = [
        (-0.48, 0.24, 2.05, 0.018),
        (-0.62, 0.16, 2.08, 0.011),
        (-0.72, 0.14, 2.12, 0.005),
    ]
    for tw in (hero_twig_a, hero_twig_b, hero_twig_c):
        _bmesh_add_tapered_tube(bm_wood, tw, radial_segs=8, subdivs=2, bark_ridge=0.02)

    # BOUGH 2: Sweeping Upper-Left Canopy Arch
    bough_left = [
        (-0.17, 1.28, 1.64, 0.115),
        (-0.52, 1.26, 2.14, 0.088),
        (-0.98, 1.22, 2.58, 0.062),
        (-1.46, 1.18, 2.86, 0.036),
        (-1.76, 1.12, 3.04, 0.014),
    ]
    _bmesh_add_tapered_tube(bm_wood, bough_left, radial_segs=10, subdivs=3, bark_ridge=0.04)

    # BOUGH 3: Tall Monumental Central Crown Leader (reaches 3.6m tall!)
    bough_crown = [
        (-0.16, 1.29, 1.65, 0.125),
        (-0.14, 1.34, 2.25, 0.092),
        (-0.18, 1.40, 2.85, 0.060),
        (-0.08, 1.44, 3.34, 0.030),
        (-0.04, 1.46, 3.58, 0.012),
    ]
    _bmesh_add_tapered_tube(bm_wood, bough_crown, radial_segs=10, subdivs=3, bark_ridge=0.04)

    # BOUGH 4: Sweeping Right Orchard Bough (high Z >= 2.35 so center sky stays open for Beats 2/5/6!)
    bough_right = [
        (-0.15, 1.29, 1.66, 0.115),
        (0.26, 1.34, 2.18, 0.085),
        (0.76, 1.42, 2.58, 0.058),
        (1.24, 1.48, 2.84, 0.032),
        (1.58, 1.52, 2.98, 0.012),
    ]
    _bmesh_add_tapered_tube(bm_wood, bough_right, radial_segs=10, subdivs=3, bark_ridge=0.04)

    # BOUGH 5: Rear Courtyard Depth Bough reaching over the stone balustrade
    bough_rear = [
        (-0.15, 1.32, 1.68, 0.105),
        (0.08, 1.68, 2.22, 0.078),
        (0.38, 2.06, 2.66, 0.052),
        (0.72, 2.38, 2.98, 0.026),
        (0.95, 2.58, 3.16, 0.011),
    ]
    _bmesh_add_tapered_tube(bm_wood, bough_rear, radial_segs=9, subdivs=3, bark_ridge=0.035)

    # Secondary fork limbs enriching the orchard crown silhouette
    secondary_limbs = [
        [(-0.52, 1.26, 2.14, 0.056), (-0.82, 1.62, 2.58, 0.034), (-1.15, 1.88, 2.96, 0.014)],
        [(-0.98, 1.22, 2.58, 0.042), (-1.12, 0.88, 2.92, 0.024), (-1.28, 0.68, 3.18, 0.010)],
        [(-0.14, 1.34, 2.25, 0.058), (-0.52, 1.56, 2.84, 0.034), (-0.76, 1.72, 3.32, 0.013)],
        [(-0.18, 1.40, 2.85, 0.045), (0.34, 1.52, 3.24, 0.026), (0.68, 1.64, 3.48, 0.011)],
        [(0.26, 1.34, 2.18, 0.054), (0.52, 1.02, 2.56, 0.032), (0.76, 0.84, 2.82, 0.012)],
        [(0.76, 1.42, 2.58, 0.042), (1.06, 1.84, 2.96, 0.024), (1.32, 2.12, 3.22, 0.010)],
        [(0.08, 1.68, 2.22, 0.050), (-0.36, 2.08, 2.72, 0.028), (-0.64, 2.38, 3.08, 0.012)],
        [(-0.18, 1.40, 2.85, 0.038), (-0.12, 0.94, 3.18, 0.022), (-0.06, 0.66, 3.38, 0.010)],
    ]
    for limb in secondary_limbs:
        _bmesh_add_tapered_tube(bm_wood, limb, radial_segs=7, subdivs=2, bark_ridge=0.025)

    tree_wood_obj = _finalize_bmesh(scene, "WoolsthorpeAppleTreeWood", bm_wood, mats=bark_mat, smooth=True)

    # 3. Rich Layered Sunlit Foliage Canopy (14 Sculpted Organic Clusters + 280 Sculpted Botanical Leaf Cards!)
    bm_fol = bmesh.new()
    rng_f = random.Random(1666)

    # 14 organic irregular foliage bough clusters
    cluster_specs = [
        # (cx, cy, cz, rx, ry, rz)
        (-0.76, 0.48, 2.26, 0.42, 0.36, 0.28),
        (-0.34, 0.72, 2.34, 0.46, 0.38, 0.30),
        (-1.24, 0.84, 2.98, 0.56, 0.48, 0.38),
        (-1.58, 1.18, 2.96, 0.58, 0.50, 0.40),
        (-1.02, 1.68, 2.94, 0.62, 0.54, 0.44),
        (-0.68, 1.68, 3.30, 0.64, 0.56, 0.46),
        (-0.08, 1.44, 3.48, 0.72, 0.62, 0.48),  # Monumental 3.6m+ sunlit crown!
        (-0.08, 0.78, 3.24, 0.54, 0.46, 0.36),
        (0.58, 1.58, 3.36, 0.64, 0.54, 0.44),
        (0.72, 0.94, 2.78, 0.50, 0.44, 0.34),
        (1.34, 1.48, 2.94, 0.58, 0.50, 0.38),
        (1.18, 2.04, 3.12, 0.58, 0.52, 0.40),
        (0.76, 2.48, 3.06, 0.60, 0.54, 0.42),
        (-0.52, 2.28, 3.02, 0.62, 0.54, 0.42),
    ]
    for cx, cy, cz, rx, ry, rz in cluster_specs:
        _bmesh_add_uv_sphere(
            bm_fol,
            Vector((cx, cy, cz)),
            rx=rx,
            ry=ry,
            rz=rz,
            segs=14,
            rings=9,
            ruffle_amp=0.22,
            ruffle_freq=5.6,
        )

    # 56 Crisp Hero Close-Up Sculpted Leaves along the Hero Overhanging Apple Bough (framing Beat 1!)
    hero_leaf_anchors = _catmull_rom_sample(hero_bough[1:], subdivs=5) + _catmull_rom_sample(hero_twig_a, subdivs=3) + _catmull_rom_sample(hero_twig_b, subdivs=3) + _catmull_rom_sample(hero_twig_c, subdivs=3)
    for idx_h in range(56):
        hx, hy, hz, hr = hero_leaf_anchors[idx_h % len(hero_leaf_anchors)]
        # Keep immediate 0.07m stem area directly below HERO_APPLE_ATTACH clean for the hero falling apple!
        if Vector((hx, hy, hz)).distance(HERO_APPLE_ATTACH) < 0.06:
            hz += 0.045
        ang_z = rng_f.uniform(0.0, math.pi * 2.0)
        off = Vector((math.cos(ang_z) * (hr + 0.025), math.sin(ang_z) * (hr + 0.025), rng_f.uniform(-0.02, 0.06)))
        _bmesh_add_leaf_card(
            bm_fol,
            origin=Vector((hx, hy, hz)) + off,
            rot_euler=Euler((rng_f.uniform(-0.55, 0.55), rng_f.uniform(-0.55, 0.55), ang_z), "XYZ"),
            length=rng_f.uniform(0.085, 0.135),
            width=rng_f.uniform(0.042, 0.068),
        )

    # 224 Sculpted Canopy Silhouette Leaf Cards bursting organically around all 14 canopy clusters (total = 280 leaves!)
    for cx, cy, cz, rx, ry, rz in cluster_specs:
        for _ in range(16):
            th = rng_f.uniform(0.0, math.pi * 2.0)
            cp = rng_f.uniform(-0.75, 0.88)
            sp = math.sqrt(max(0.0, 1.0 - cp * cp))
            dir_v = Vector((sp * math.cos(th), sp * math.sin(th), cp))
            l_pos = Vector((cx + dir_v.x * rx * 0.96, cy + dir_v.y * ry * 0.96, cz + dir_v.z * rz * 0.96))
            _bmesh_add_leaf_card(
                bm_fol,
                origin=l_pos,
                rot_euler=Euler((rng_f.uniform(-0.7, 0.7), rng_f.uniform(-0.7, 0.7), th), "XYZ"),
                length=rng_f.uniform(0.11, 0.18),
                width=rng_f.uniform(0.055, 0.092),
            )

    tree_canopy_obj = _finalize_bmesh(scene, "WoolsthorpeAppleCanopy", bm_fol, mats=leaf_mat, smooth=True)

    # 4. 12 Glossy Ruby-Gold Hanging Heirloom Apples tucked naturally among the canopy (clear of HERO_APPLE_ATTACH!)
    bm_app = bmesh.new()
    canopy_apple_coords = [
        (-0.84, 0.40, 2.05, 0.044),
        (-0.34, 0.68, 2.11, 0.045),
        (-0.72, 0.16, 2.06, 0.042),
        (-1.32, 0.86, 2.66, 0.048),
        (-1.54, 1.15, 2.64, 0.046),
        (-0.92, 1.54, 2.58, 0.046),
        (-0.16, 0.82, 2.94, 0.048),
        (0.66, 0.92, 2.48, 0.046),
        (1.26, 1.44, 2.62, 0.048),
        (0.46, 1.52, 2.96, 0.046),
        (0.78, 2.26, 2.74, 0.048),
        (-0.48, 2.02, 2.68, 0.046),
    ]
    for ax, ay, az, ar in canopy_apple_coords:
        _bmesh_add_heirloom_apple(bm_app, Vector((ax, ay, az)), radius=ar)
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
    # Subtle smooth semicircular recessed architectural blind-arch molding on the dark back wall
    n_ba = 32
    ba_cx, ba_cz, ba_r = -2.76, 1.68, 0.72
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


def _build_garden_plinths(scene, stone_mat, brass_mat):
    """Build the 3 elegant carved stone garden plinths with exact top heights (ZERO shopkeeper counter!)."""
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
    apple_plinth_obj = _finalize_bmesh(
        scene,
        "AppleGardenPlinth",
        bm_ap,
        mats=stone_mat,
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
    principia_plinth_obj = _finalize_bmesh(
        scene,
        "PrincipiaSundialPlinth",
        bm_pr,
        mats=[stone_mat, brass_mat],
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

    cradle_plinth_obj = _finalize_bmesh(
        scene,
        "CradleTerracePlinth",
        bm_cr,
        mats=stone_mat,
        loc=(CRADLE_PLINTH.x, CRADLE_PLINTH.y, FLOOR_Z),
        smooth=True,
        edge_split_deg=35.0,
    )

    return apple_plinth_obj, principia_plinth_obj, cradle_plinth_obj


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
    """Build golden-hour outdoor SUN + Rembrandt AREA key + sapphire rim + bounce fill + eye catchlight."""
    stage_focus = Vector((0.0, 0.30, 1.30))

    # 1. Warm Golden-Hour Outdoor Sun casting crisp dappled apple-leaf shadows across terrace & actors
    sun_obj = _add_light_obj(
        scene,
        "WoolsthorpeGoldenSun",
        "SUN",
        loc=(4.5, 5.0, 5.5),
        rot=(math.radians(66.0), math.radians(8.0), math.radians(142.0)),
        energy=4.2,
        color=(1.0, 0.74, 0.42),
        angle_deg=1.6,
    )

    # 2. Warm Golden-Hour Rembrandt Key Area Light (sculpts faces & stone relief cleanly without harsh shadows)
    key_loc = (-2.35, -2.55, 2.55)
    key_obj = _add_light_obj(
        scene,
        "WoolsthorpeRembrandtKey",
        "AREA",
        loc=key_loc,
        rot=_aim_euler(key_loc, stage_focus),
        energy=260.0,
        color=(1.0, 0.80, 0.52),
        size=1.65,
    )

    # 3. Cool Sky / Moon Sapphire Rim Light (separates hair strands, shoulders & tree silhouette!)
    rim_loc = (2.25, 3.15, 2.85)
    rim_obj = _add_light_obj(
        scene,
        "WoolsthorpeSkyMoonRim",
        "AREA",
        loc=rim_loc,
        rot=_aim_euler(rim_loc, stage_focus),
        energy=220.0,
        color=(0.52, 0.78, 1.0),
        size=2.05,
    )

    # 4. Soft Outdoor Meadow & Limestone Bounce Fill
    fill_loc = (0.85, -2.50, 1.65)
    fill_obj = _add_light_obj(
        scene,
        "WoolsthorpeMeadowBounce",
        "AREA",
        loc=fill_loc,
        rot=_aim_euler(fill_loc, stage_focus),
        energy=68.0,
        color=(0.94, 0.88, 0.78),
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
    """Build the complete V7 Outdoor Woolsthorpe Orchard & Cosmos Environment and return world_handles."""
    # Instantiate rich shared V7 PBR material palette
    stone_mat = _make_cotswold_stone_mat("CotswoldCarvedLimestone", dark=False)
    ashlar_mat = _make_cotswold_ashlar_mat("CotswoldManorAshlar", dark=False)
    alcove_dark_stone_mat = _make_cotswold_ashlar_mat("LoggiaChiaroscuroDarkStone", dark=True)
    flagstone_mat = _make_mossy_flagstone_mat("CotswoldMossyFlagstone")
    meadow_grass_mat = _make_meadow_grass_mat("WoolsthorpeMeadowGrass", distant=False, haze_mix=0.0)
    hill_mid_mat = _make_meadow_grass_mat("LincolnshireMidHill", distant=True, haze_mix=0.34)
    hill_far_mat = _make_meadow_grass_mat("LincolnshireFarHorizonHill", distant=True, haze_mix=0.62)
    distant_tree_mat = _make_simple_pbr_mat("DistantEnglishHedgerow", (0.024, 0.072, 0.032), rough=0.92)
    grass_blade_mat = _make_simple_pbr_mat("MeadowGrassBlade", (0.052, 0.175, 0.024), rough=0.72)
    wildflower_gold_mat = _make_simple_pbr_mat("MeadowButtercupGold", (0.86, 0.64, 0.05), rough=0.48)
    wildflower_white_mat = _make_simple_pbr_mat("MeadowEnglishDaisy", (0.92, 0.90, 0.82), rough=0.52)
    soil_mat = _make_simple_pbr_mat("OrchardMossySoil", (0.042, 0.030, 0.018), rough=0.95, bump_str=0.35)

    bark_mat = _make_gnarled_bark_mat("WoolsthorpeGnarledBark")
    leaf_mat = _make_translucent_leaf_mat("WoolsthorpeSunlitCanopyLeaf")
    apple_mat = _make_ruby_apple_mat("WoolsthorpeCanopyApple")

    oak_mat = _make_oak_walnut_mat("ManorDarkWalnutOak", dark=True)
    warm_oak_mat = _make_oak_walnut_mat("CarvedEnglishOak", dark=False)
    brass_mat = _make_simple_pbr_mat("CambridgeBrass", (0.90, 0.65, 0.22), rough=0.17, metallic=0.96, bump_str=0.020)
    bronze_mat = _make_simple_pbr_mat("AgedGardenBronze", (0.28, 0.19, 0.095), rough=0.32, metallic=0.88, bump_str=0.024)
    velvet_mat = _make_simple_pbr_mat("RoyalCrimsonVelvet", (0.28, 0.024, 0.036), rough=0.88, metallic=0.0)

    star_gold_mat, _, _ = _make_emissive_mat("CosmosStarWarmGold", (1.0, 0.88, 0.60), strength=5.6)
    star_sapphire_mat, _, _ = _make_emissive_mat("CosmosStarSapphireWhite", (0.76, 0.90, 1.0), strength=5.4)
    crescent_moon_mat, moon_vis_fac = _make_crescent_moon_mat()
    godray_mat, godray_opacity_sock = _make_godray_mat()
    pollen_mat, pollen_color_sock, pollen_strength_sock = _make_emissive_mat("GoldenOrchardPollen", (1.0, 0.80, 0.34), strength=2.6)

    # 1. Animated Physical Sky (NISHITA Golden-Hour -> Starry Twilight)
    world, sky_night_fac = _build_animated_physical_sky(scene)

    # 2. 160 True Random Spherical Stars + Sculpted 3D Crescent Moon
    star_entries, crescent_moon_obj = _build_starfield_and_crescent_moon(
        scene, star_gold_mat, star_sapphire_mat, crescent_moon_mat
    )

    # 3. Rolling Meadow Terrain, Curved Mossy Flagstones, Distant Hills & 140 Grass/Wildflower Clumps
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

    # 4. Carved Cotswold Stone Garden Balustrade (18 turned stone urns + 3 pier finials)
    balustrade_objs = _build_carved_stone_balustrade(scene, stone_mat)

    # 5. Monumental 3.6m Sculpted Woolsthorpe Apple Tree rooted at (-0.18, +1.35, 0.0)
    tree_wood_obj, tree_canopy_obj, canopy_apples_obj = _build_woolsthorpe_apple_tree(
        scene, bark_mat, leaf_mat, apple_mat
    )

    # 6. Shaded Left Cotswold Stone Manor Garden Loggia / Optical Alcove (Beat 3 Prism!)
    loggia_objs = _build_cotswold_optical_loggia(
        scene, ashlar_mat, alcove_dark_stone_mat, stone_mat, oak_mat, brass_mat
    )

    # 7. Three Sculpted Stone Garden Plinths (APPLE, PRINCIPIA, CRADLE)
    apple_plinth_obj, principia_plinth_obj, cradle_plinth_obj = _build_garden_plinths(
        scene, stone_mat, brass_mat
    )

    # 8. Volumetric Golden God-Rays + 45 Randomly Scattered Golden Pollen Motes
    godray_objs, pollen_entries = _build_godrays_and_pollen(scene, godray_mat, pollen_mat)

    # 9. Outdoor Golden-Hour -> Moonlit Twilight Lighting Rig
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
        # Atmosphere handles
        "godrays": godray_objs,
        "godray_opacity": godray_opacity_sock,
        "pollen_motes": pollen_entries,
        "pollen_color": pollen_color_sock,
        "pollen_strength": pollen_strength_sock,
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
    }
    return world_handles


def animate_world_frame(world_handles: dict, frame: int, t: float) -> None:
    """Keyframe Golden-Hour -> Starry Cosmic Twilight transition, stars, crescent moon, sunbeams, pollen & outdoor lighting."""
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

    # 6. Outdoor Golden-Hour -> Moonlit Twilight Lighting Rig (with Beat 3 shaded chiaroscuro boost)
    sun_ld = world_handles["sun_light"].data
    sun_ld.energy = lerp(4.2, 1.15, night_fac)
    sun_ld.color = (lerp(1.00, 0.42, night_fac), lerp(0.74, 0.68, night_fac), lerp(0.42, 1.00, night_fac))
    sun_ld.keyframe_insert(data_path="energy", frame=frame)
    sun_ld.keyframe_insert(data_path="color", frame=frame)

    key_ld = world_handles["key_light"].data
    day_key_e = 260.0 * (1.0 - 0.34 * beat3_shade)
    key_ld.energy = lerp(day_key_e, 145.0, night_fac)
    key_ld.color = (lerp(1.00, 0.58, night_fac), lerp(0.80, 0.78, night_fac), lerp(0.52, 1.00, night_fac))
    key_ld.keyframe_insert(data_path="energy", frame=frame)
    key_ld.keyframe_insert(data_path="color", frame=frame)

    rim_ld = world_handles["rim_light"].data
    day_rim_e = 220.0 * (1.0 - 0.25 * beat3_shade)
    rim_ld.energy = lerp(day_rim_e, 360.0, night_fac)
    rim_ld.color = (lerp(0.52, 0.34, night_fac), lerp(0.78, 0.72, night_fac), 1.00)
    rim_ld.keyframe_insert(data_path="energy", frame=frame)
    rim_ld.keyframe_insert(data_path="color", frame=frame)

    fill_ld = world_handles["fill_light"].data
    day_fill_e = 68.0 * (1.0 - 0.45 * beat3_shade)
    fill_ld.energy = lerp(day_fill_e, 38.0, night_fac)
    fill_ld.color = (lerp(0.94, 0.45, night_fac), lerp(0.88, 0.62, night_fac), lerp(0.78, 0.92, night_fac))
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
