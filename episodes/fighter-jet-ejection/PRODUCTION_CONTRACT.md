# Episode #1 — PRODUCTION CONTRACT (Fighter Jet Ejection, 4K 60fps)

**Title:** Why Ejecting From a Fighter Jet Shrinks You by 1 Inch · **30.0 s @ 60 fps = 1800 frames** · 1080×1920 Cycles → True 4K.
**Playbook:** `/home/antigravity/ssghtc002-sudo/isaac-newton-gravity-prism-3d/AGENTS.md` (V11) — mandatory for every agent.
**Single source of truth:** [`src/ejection_contract.py`](src/ejection_contract.py) — import ALL timings, coordinates, trajectories,
X-ray windows, label texts from it. Never redefine them. If you think the contract is wrong, report it to the Lead Director instead of diverging.

## Crew (1 Agent = 1 Single Focused Task, non-overlapping files)

| Agent | Sole file(s) | Public API (called by the master integrator `src/build_ejection_scene.py`) |
|---|---|---|
| Scriptwriter | `config/timeline.json` | — |
| Jet Cockpit & Sky Env | `src/jet_cockpit_env.py` | `build_environment(col, total_frames=1800) -> dict` |
| Canopy & MDC Shatter FX | `src/canopy_fx.py` | `build_canopy(col, jet_root, total_frames=1800) -> dict` |
| Ejection Seat & Rocket | `src/ejection_seat_asset.py` | `build_ejection_seat(col, total_frames=1800) -> dict` |
| Parachutes | `src/parachute_asset.py` | `build_parachutes(col, seat_root, pilot_carrier, total_frames=1800) -> dict` |
| Pilot Character Animator | `src/pilot_character.py` | `build_pilot(col, total_frames=1800) -> dict` |
| Spine X-Ray Rig | `src/spine_xray.py` | `build_spine_xray(col, carrier, sacrum_local, c1_local, total_frames=1800) -> dict` |
| Cinema Camera & Framing QA | `src/cinema_camera.py`, `src/verify_framing.py` | `camera_state_at(f) -> dict` (pure python), `build_cinema_camera(col, total_frames=1800) -> bpy Object` |
| 3D Labels & Gauges (wave 2) | `src/labels_3d.py` | `build_labels(col, font_path, seat_root, pilot_carrier, jet_root, total_frames=1800) -> dict` |
| TTS Voice (wave 2) | `audio/gen_voice.py` | writes `audio/output/voice_stem.wav` + `audio/output/voice_manifest.json` |
| Foley SFX (wave 2) | `audio/gen_sfx.py` | writes `audio/output/sfx_stem.wav` |
| BGM Composer | `audio/gen_bgm.py` | writes `audio/output/bgm_stem.wav` |
| Subtitles (wave 3) | `audio/gen_subtitles.py` | writes `subtitles/subtitles.ass` + `subtitles/subtitles.srt` |
| Master Mix (wave 3) | `audio/generate_audio.py` | writes `audio/output/master_audio.wav` (−14.0 LUFS, TP ≤ −1.5 dBFS) |
| Lead Director / Integrator | `src/ejection_contract.py`, `src/build_ejection_scene.py`, `render_on_modal.py`, `README.md` | — |

## Universal rules for every 3D module
- `import bpy`, `from mathutils import Vector, Euler, Matrix`; add `SCRIPT_DIR` and `REPO_ROOT/src` to `sys.path`, `import ejection_contract as C`.
- Every returned dict value is a real `bpy.types.Object` / `Material` / list thereof (document exact types in the module docstring).
- Name every object with a unique module prefix (`Env_`, `Canopy_`, `Seat_`, `Chute_`, `Pilot_`, `Spine_`, `Cam_`, `Label_`) — never rely on Blender auto-suffixes of other modules.
- Do NOT call `bpy.ops.wm.read_factory_settings` (only the integrator does). Do NOT set render settings / world unless you are the Env agent.
- Keyframed trajectories: keyframe every frame (or every 2 frames + `LINEAR`) straight from the contract functions so integration error is 0.000 mm.
- Detached/finished props: `CONSTANT` `hide_render`/`hide_viewport` keys (never reappear).
- Local-space shaders (`TexCoord.Object`), X-ray glass shadow bypass (`LightPath.Is Shadow Ray → Transparent`), open-ended additive FX (`Emission + Transparent` via `AddShader`), AgX-safe emission (≤ 2.1 for text, fire cores may go higher but must not white-out frames).
- **Render budget:** the final render is 1800 frames on 10 cloud GPUs (L4/A10G) at 16 samples → target **≤ 10 s/frame**. No large volumetrics, no heavy particle hair, subdivision ≤ level 2 on hero props, procedural textures preferred.
- Provide a standalone self-test: `if __name__ == "__main__":` that builds just your module into an empty scene (factory reset allowed ONLY inside this self-test).

## VM resource rule (2 CPUs / 7 GB RAM shared by the whole crew!)
- Run Blender ONLY through the shared lock, low priority, headless:
  `flock /tmp/blender_qa.lock nice -n 10 /tmp/blender/blender -b --python-exit-code 1 --python <script> -- <args>`
- Test renders: Cycles CPU, `resolution_percentage ≤ 25`, `samples ≤ 6`, at most 3 frames per run. Inspect every rendered PNG/JPG with `view_file`.
- Save scratch renders under `/tmp/fje_<your_module>/` only. Never write outside your sole files (plus `/tmp`).
- Do NOT git commit or push — the Lead Director integrates and commits.
