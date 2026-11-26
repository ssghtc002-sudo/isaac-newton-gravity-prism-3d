"""
Master Scene Assembler: How a Grenade Actually Works — Hollywood Action + Matrix Bullet-Time 4K X-Ray Short
Unifies all 5 specialist production modules:
  1. battlefield_env.py  — Golden-hour desert tactical outpost + 4K MakeHuman Hero Soldier & 2 Rival Sentries
  2. grenade_asset.py    — Hyper-realistic M67 cast-iron grenade + local-space X-ray cutaway + 3D pointer labels
  3. cinema_camera.py    — Dynamic 6-beat 9:16 portrait camera rig (50mm Hero -> 35mm Throw -> 60mm X-Ray -> 32mm Blast)
  4. director_shotlist   — Synchronized 720-frame (30.0s @ 24fps) continuity milestones
  5. generate_audio.py   — Studio Edge-TTS narration + 19 battlefield/mechanical Foley SFX + karaoke subtitles

Follows AGENTS.md V10 Cinema Standard (AgX color management, photoreal eyes, shadow-bypass X-ray glass,
open-ended additive optics, exact DOF focus, 100% verified 9:16 portrait framing).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EPISODE_DIR.parent.parent

for p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import battlefield_env  # noqa: E402
import cinema_camera  # noqa: E402
import grenade_asset  # noqa: E402

FPS = 24
TOTAL_FRAMES = 720
FONT_PATH = EPISODE_DIR / "subtitles" / "fonts" / "DejaVuSans-Bold.ttf"


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _set_bezier(obj) -> None:
    ad = getattr(obj, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"


def _make_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def configure_render(samples: int, scale: int, device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = scale
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95

    # AgX Cinema Color Management
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    cycles = scene.cycles
    cycles.samples = samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.025
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 3
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 8
    cycles.transparent_max_bounces = 14
    cycles.volume_bounces = 0
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.sample_clamp_direct = 12.0
    cycles.sample_clamp_indirect = 5.0

    # Subtle action motion blur (clean during throw & flying spoon, sharp in bullet-time)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.28

    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device.upper() == "CPU":
        cycles.device = "CPU"
        print("[Render] Using CPU Cycles + OIDN")
        return

    chosen = None
    for backend in ([device.upper(), "OPTIX", "CUDA"] if device.upper() in {"OPTIX", "CUDA"} else ["OPTIX", "CUDA"]):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                cycles.device = "GPU"
                chosen = backend
                break
        except Exception:
            continue

    if chosen:
        print(f"[Render] Using GPU Cycles ({chosen}) + OIDN")
    else:
        cycles.device = "CPU"
        print("[Render] No GPU found -> CPU Cycles + OIDN")


def animate_grenade_world_path(
    grenade_root: bpy.types.Object,
    hero_socket_positions: dict[int, Vector],
    release_frame: int = 134,
    total_frames: int = TOTAL_FRAMES,
) -> None:
    """
    Seamlessly drives Grenade_Root from Hero Soldier's gripping right hand (f=1..134),
    launches out of his palm into Matrix Bullet-Time mid-air X-Ray (f=134..440),
    dives into the Enemy Sandbag Bunker (f=440..495) with a crisp dirt micro-hop (f=495..508),
    and scales up the 48-shard blast burst cleanly at detonation (f=612..720) so fragments
    read heroically from the wide 32mm Hollywood action camera!
    """
    rel_pos = Vector(hero_socket_positions.get(release_frame, (0.24, 0.42, 1.62)))
    bt_entry_pos = Vector(cinema_camera.grenade_pos_at(152))

    # 1. Held in Hero's right hand (f = 1..134) — facing Beat 1 camera (+Y side) with pull ring toward Hero's mouth!
    for f in range(1, release_frame + 1):
        p = Vector(hero_socket_positions.get(f, rel_pos))
        grenade_root.location = p
        # Rotate +162° Z so olive body faces Beat 1 camera & chrome pull-ring angles cleanly toward Hero's lips/teeth!
        # Subtle wrist tilt during pin tug (f=24..56) and wind-up throw (f=108..134)
        tilt_x = math.radians(8.0) if f <= 108 else math.radians(8.0 + 18.0 * _smoothstep((f - 108) / 26.0))
        rot_z = math.radians(158.0)
        grenade_root.rotation_euler = Euler((tilt_x, math.radians(-10.0), rot_z))
        grenade_root.scale = (1.0, 1.0, 1.0)
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)
        if f in (1, release_frame):
            grenade_root.keyframe_insert("scale", frame=f)

    # 2. Explosive release transition into Matrix Bullet-Time (f = 135..151)
    for f in range(release_frame + 1, 152):
        u = (f - release_frame) / (152.0 - release_frame)
        # Parabolic arc from exact hand release to bullet-time entry point
        p = rel_pos.lerp(bt_entry_pos, u)
        p.z += 0.22 * math.sin(math.pi * u)
        grenade_root.location = p
        # Smoothly spin into front cutaway alignment (rot_z = 0°) facing the -Y bullet-time macro camera!
        su = _smoothstep(u)
        grenade_root.rotation_euler = Euler((
            math.radians(26.0 * (1.0 - su) + 4.0 * su),
            math.radians(-10.0 * (1.0 - su)),
            math.radians(158.0 * (1.0 - su)),
        ))
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 3. Matrix Bullet-Time X-Ray Glide + Ballistic Dive to Enemy Sandbags (f = 152..495)
    for f in range(152, 496):
        p = Vector(cinema_camera.grenade_pos_at(f))
        grenade_root.location = p
        if f <= 440:
            # Ultra-subtle suspended bullet-time float (keeps X-ray cutaway & 3D labels 100% locked to lens!)
            t_bt = (f - 152) / 288.0
            rx = math.radians(4.0 * math.cos(t_bt * math.pi * 1.5))
            ry = math.radians(5.0 * math.sin(t_bt * math.pi * 2.0))
            rz = math.radians(3.0 * math.sin(t_bt * math.pi))
        else:
            # Accelerating dive into enemy sandbag ground zero, stabilizing upright at landing
            u = (f - 440) / 55.0
            su = _smoothstep(u)
            rx = math.radians(-12.0 * math.sin(math.pi * u))
            ry = math.radians(14.0 * math.sin(math.pi * u))
            rz = math.radians(0.0 * su)
        grenade_root.rotation_euler = Euler((rx, ry, rz))
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 4. Crisp Ground-Zero Dirt Micro-Hop & Settle at Enemy Bunker (f = 496..611)
    land_pos = Vector(cinema_camera.BUNKER_DEFAULT_POS)
    for f in range(496, 612):
        hop = 0.0
        tilt = 0.0
        if f <= 508:
            u = (f - 495) / 13.0
            hop = 0.048 * math.sin(math.pi * u) * (1.0 - 0.35 * u)
            tilt = math.radians(14.0 * math.sin(math.pi * u))
        grenade_root.location = Vector((land_pos.x, land_pos.y, land_pos.z + hop))
        grenade_root.rotation_euler = Euler((tilt, 0.0, 0.0))
        grenade_root.keyframe_insert("location", frame=f)
        grenade_root.keyframe_insert("rotation_euler", frame=f)

    # 5. Beat 6 Wide-Shot Detonation Scale Boost (f = 611..720)
    # Clean 2.8x hero scale at Z=0.48 makes all 48 dark iron shards & shockwave ring pop alongside bunker & sentries!
    grenade_root.scale = (1.0, 1.0, 1.0)
    grenade_root.keyframe_insert("scale", frame=611)
    grenade_root.location = Vector((land_pos.x, land_pos.y, land_pos.z))
    grenade_root.keyframe_insert("location", frame=611)

    grenade_root.location = Vector((land_pos.x, land_pos.y, 0.44))
    grenade_root.scale = (2.6, 2.6, 2.6)
    grenade_root.keyframe_insert("location", frame=624)
    grenade_root.keyframe_insert("scale", frame=624)

    grenade_root.location = Vector((land_pos.x, land_pos.y, 0.56))
    grenade_root.scale = (3.0, 3.0, 3.0)
    grenade_root.keyframe_insert("location", frame=total_frames)
    grenade_root.keyframe_insert("scale", frame=total_frames)
    _set_bezier(grenade_root)


def _tune_beat6_explosion_cinematography(env_info: dict, grenade: dict) -> None:
    """
    Prevent Beat 6 white-out while keeping dramatic Christopher-Nolan-style high-contrast
    fiery core + dark sunlit smoke plume + crisp flying iron shards + diving sentries!
    """
    # 1. Hide handheld Beat 1 safety_pin label so no sideways 3D text floats over Hero's fingers
    pin_lbl = grenade.get("labels", {}).get("safety_pin")
    if pin_lbl is not None:
        objs_to_hide = []
        if isinstance(pin_lbl, dict):
            for v in pin_lbl.values():
                if isinstance(v, bpy.types.Object):
                    objs_to_hide.append(v)
                    objs_to_hide.extend(list(v.children_recursive))
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, bpy.types.Object):
                            objs_to_hide.append(item)
                            objs_to_hide.extend(list(item.children_recursive))
        elif isinstance(pin_lbl, bpy.types.Object):
            objs_to_hide = [pin_lbl] + list(pin_lbl.children_recursive)
        for o in set(objs_to_hide):
            o.animation_data_clear()
            o.hide_viewport = True
            o.hide_render = True

    # 2. Clamp ALL explosion / flash / burn point lights in Beat 6 so sunlit desert & bunker stay richly exposed!
    for light_obj in bpy.data.objects:
        if light_obj.type == "LIGHT" and light_obj.data.type == "POINT":
            ld = light_obj.data
            if ld.animation_data and ld.animation_data.action:
                for fc in ld.animation_data.action.fcurves:
                    if fc.data_path == "energy":
                        for kp in fc.keyframe_points:
                            if kp.co.x >= 610:
                                kp.co.y = min(kp.co.y, 140.0)
            else:
                ld.energy = min(ld.energy, 140.0)

    # 3. Hide milky translucent beige dust-cloud spheres that obscured the diving Rival Sentries behind the bunker,
    #    keeping ONLY the sleek ground-level shockwave dust ring (darkened & crisp) + our fiery turbulent fireball + 48 iron shards!
    for m_name in ("Mat_ExplosionDust_Outer", "Mat_ExplosionDust_Lit"):
        mat = bpy.data.materials.get(m_name)
        if mat and mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    n.inputs["Base Color"].default_value = (0.14, 0.09, 0.05, 1.0)
                    if "Emission Strength" in n.inputs:
                        n.inputs["Emission Strength"].default_value = 0.0
                elif n.type == "MAP_RANGE" and n.inputs["To Max"].default_value > 0.3:
                    n.inputs["To Max"].default_value = 0.35

    fx_dict = env_info.get("explosion_fx", {})
    for fxo in fx_dict.get("all_fx", []):
        if fxo.name != "Explosion_ShockwaveDustRing":
            fxo.animation_data_clear()
            fxo.hide_viewport = True
            fxo.hide_render = True
            continue
        if fxo.animation_data and fxo.animation_data.action:
            for fc in fxo.animation_data.action.fcurves:
                if fc.data_path == "scale":
                    for kp in fc.keyframe_points:
                        kp.co.y *= 0.55

    # Boost turbulent fiery orange contrast on grenade plasma core during Beat 6 wide shot
    for pname in ("Turbulent_Plasma_Core", "Turbulent_Plasma_Fire", "Additive_Shockwave_Ring"):
        pmat = bpy.data.materials.get(pname)
        if pmat and pmat.use_nodes:
            for n in pmat.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Color"].default_value = (1.0, 0.42, 0.04, 1.0)


def build_sky_finale_title(col: bpy.types.Collection, total_frames: int = TOTAL_FRAMES) -> None:
    """
    AGENTS.md Section 2 Compliant World-Space 2-Zone Brushed-Gold Sky Monument Title for Beat 6 Finale
    Cleanly placed in the open blue-gold sky above the bunker canopy (Z = 2.48m),
    with generous horizontal margins and smooth camera-tracking yaw!
    """
    font = bpy.data.fonts.load(str(FONT_PATH)) if FONT_PATH.is_file() else None

    mat_gold = bpy.data.materials.new("FinaleTitle_SatinGold")
    mat_gold.use_nodes = True
    nt = mat_gold.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    vt = nt.nodes.new("ShaderNodeVectorTransform")
    vt.vector_type = "NORMAL"
    vt.convert_from = "WORLD"
    vt.convert_to = "OBJECT"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    az = nt.nodes.new("ShaderNodeMath")
    az.operation = "ABSOLUTE"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.82
    mr.inputs["From Max"].default_value = 0.92
    nt.links.new(geom.outputs["Normal"], vt.inputs["Vector"])
    nt.links.new(vt.outputs["Vector"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], az.inputs[0])
    nt.links.new(az.outputs["Value"], mr.inputs["Value"])

    face = nt.nodes.new("ShaderNodeBsdfPrincipled")
    face.inputs["Base Color"].default_value = (1.0, 0.80, 0.24, 1.0)
    face.inputs["Metallic"].default_value = 0.72
    face.inputs["Roughness"].default_value = 0.24
    face.inputs["Emission Color"].default_value = (1.0, 0.78, 0.22, 1.0)
    face.inputs["Emission Strength"].default_value = 1.75

    side = nt.nodes.new("ShaderNodeBsdfPrincipled")
    side.inputs["Base Color"].default_value = (0.04, 0.035, 0.03, 1.0)
    side.inputs["Metallic"].default_value = 0.92
    side.inputs["Roughness"].default_value = 0.30

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(side.outputs["BSDF"], mix.inputs[1])
    nt.links.new(face.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    mat_key = bpy.data.materials.new("FinaleTitle_ObsidianKeyline")
    mat_key.use_nodes = True
    bsdf_k = mat_key.node_tree.nodes["Principled BSDF"]
    bsdf_k.inputs["Base Color"].default_value = (0.015, 0.018, 0.025, 1.0)
    bsdf_k.inputs["Roughness"].default_value = 0.65

    rig = bpy.data.objects.new("Finale_SkyTitle_Rig", None)
    rig.location = Vector((0.0, 7.45, 2.38))
    col.objects.link(rig)

    for f, deg_z in [(614, -22.0), (634, -20.0), (676, 6.0), (total_frames, 26.0)]:
        rig.rotation_euler = Euler((math.pi / 2.0, 0.0, math.radians(deg_z)))
        rig.keyframe_insert("rotation_euler", frame=f)
    _set_bezier(rig)

    cdata = bpy.data.curves.new("Finale_SkyTitle", type="FONT")
    cdata.body = "PRE-SCORED SHOCKWAVE"
    if font:
        cdata.font = font
    cdata.size = 0.175
    cdata.align_x = "CENTER"
    cdata.align_y = "CENTER"
    cdata.extrude = 0.010
    cdata.bevel_depth = 0.0012
    cdata.bevel_resolution = 2
    tobj = bpy.data.objects.new("Finale_SkyTitle", cdata)
    tobj.data.materials.append(mat_gold)
    tobj.parent = rig
    col.objects.link(tobj)

    kdata = bpy.data.curves.new("Finale_SkyTitle_Keyline", type="FONT")
    kdata.body = "PRE-SCORED SHOCKWAVE"
    if font:
        kdata.font = font
    kdata.size = 0.175
    kdata.align_x = "CENTER"
    kdata.align_y = "CENTER"
    kdata.extrude = 0.007
    kdata.offset = 0.009
    kobj = bpy.data.objects.new("Finale_SkyTitle_Keyline", kdata)
    kobj.location = Vector((0.0, 0.0, -0.014))
    kobj.data.materials.append(mat_key)
    kobj.parent = rig
    col.objects.link(kobj)

    for o in (tobj, kobj):
        for f, hidden in [(1, True), (620, True), (622, False), (total_frames, False)]:
            o.hide_viewport = hidden
            o.hide_render = hidden
            o.keyframe_insert("hide_viewport", frame=f)
            o.keyframe_insert("hide_render", frame=f)
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def build_master_scene(samples: int, scale: int, device: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(samples=samples, scale=scale, device=device)

    col_env = _make_collection("Battlefield_Env")
    col_chars = _make_collection("Soldiers")
    col_grenade = _make_collection("Hero_Grenade")
    col_cam = _make_collection("Cinema_Camera")

    # 1. Golden-Hour Desert Tactical Outpost + Enemy Sandbag Bunker + Beat 6 Explosion Plume
    env_info = battlefield_env.build_battlefield(col=col_env, total_frames=TOTAL_FRAMES)

    # 2. Real 4K MakeHuman MPFB2 Hero Soldier + 2 Rival Armored Sentries
    soldiers = battlefield_env.build_soldiers(col=col_chars, total_frames=TOTAL_FRAMES)

    # 3. Hyper-Realistic M67 Grenade + Local-Space X-Ray Cutaway + 3D Component Pointer Labels
    grenade = grenade_asset.build_grenade_assembly(
        collection=col_grenade,
        include_labels=True,
        include_fx=True,
        font_path=str(FONT_PATH),
    )

    grenade_asset.animate_grenade(
        grenade,
        pin_pull=(24, 92),
        xray_scan=(154, 186),
        spoon_fly=(244, 322),
        striker_slam=(244, 294),
        fuse_burn=(355, 498),
        pressure_swell=(499, 611),
        fracture_frame=612,
        total_frames=TOTAL_FRAMES,
        show_labels=True,
    )

    animate_grenade_world_path(
        grenade_root=grenade["root"],
        hero_socket_positions=soldiers["hero_socket_world_positions"],
        release_frame=soldiers["release_frame"],
        total_frames=TOTAL_FRAMES,
    )

    # Tune Beat 1 handheld label cleanliness & Beat 6 Hollywood explosion contrast (zero white-out!)
    _tune_beat6_explosion_cinematography(env_info, grenade)

    mac_l = bpy.data.lights.new("Grenade_Macro_Key", type="AREA")
    mac_l.shape = "DISK"
    mac_l.size = 0.22
    mac_l.color = (0.92, 0.96, 1.0)
    mac_obj = bpy.data.objects.new("Grenade_Macro_Key", mac_l)
    mac_obj.parent = grenade["root"]
    mac_obj.location = Vector((-0.14, -0.24, 0.16))
    mac_obj.rotation_euler = (Vector((0.0, 0.0, 0.02)) - mac_obj.location).to_track_quat("-Z", "Y").to_euler()
    col_grenade.objects.link(mac_obj)
    for f, e in [(1, 0.0), (150, 0.0), (160, 1.6), (608, 1.6), (614, 0.0), (TOTAL_FRAMES, 0.0)]:
        mac_l.energy = e
        mac_l.keyframe_insert("energy", frame=f)

    # 4. World-Space Beat 6 Open-Sky Monument Title
    build_sky_finale_title(col=col_grenade, total_frames=TOTAL_FRAMES)

    # 5. Hollywood 6-Beat Cinema Camera Rig
    cinema_camera.build_cinema_camera(
        col=col_cam,
        grenade_root=grenade["root"],
        hero_pos=Vector(battlefield_env.HERO_POS),
        bunker_pos=Vector(cinema_camera.BUNKER_DEFAULT_POS),
        total_frames=TOTAL_FRAMES,
    )

    bpy.context.scene.cycles.transparent_max_bounces = 14


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Build & Render How a Grenade Works Hollywood 4K Master Scene")
    p.add_argument("--frames-dir", type=str, default="/tmp/grenade_master_frames")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=TOTAL_FRAMES)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--device", type=str, default="OPTIX")
    p.add_argument("--preview", action="store_true", help="Render 6 representative beat keyframes for visual QA")
    p.add_argument("--frames", type=str, default="", help="Optional comma-separated explicit frame list")
    p.add_argument("--save-blend", type=str, default="")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_master_scene(samples=args.samples, scale=args.scale, device=args.device)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"[Master] Saved blend -> {args.save_blend}")

    os.makedirs(args.frames_dir, exist_ok=True)
    scene = bpy.context.scene

    if args.frames:
        frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    elif args.preview:
        frames = [35, 195, 294, 420, 540, 665]
    else:
        frames = list(range(args.start_frame, args.end_frame + 1))

    for f in frames:
        scene.frame_set(f)
        out_path = os.path.join(args.frames_dir, f"frame_{f:04d}.jpg")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] frame_{f:04d}.jpg", flush=True)


if __name__ == "__main__":
    main()
