# PRODUCTION CONTRACT — Episode #16: Falling Elevator Myth-Buster (`4K 60fps`)

**Single Source of Truth:** [`src/elevator_contract.py`](src/elevator_contract.py)
**Timeline & Mix Spec:** `60 fps`, `1800 frames`, `30.000 s`, `1080×1920` Cycles (`OPTIX + OIDN`) -> True 4K (`2160×3840`, `High@L5.2`).

## Strict `1 Agent = 1 Single Focused Task` Crew Allocation
Every specialist owns ONLY their assigned file and must import `elevator_contract.py` for coordinates/timings:

1. **Elevator Shaft & High-Rise Environment (`src/shaft_env.py`)**:
   - Owns `build_environment(col, total_frames=1800) -> dict` returning `shaft_root`, `world`, `lights` (including mandatory `0.42 m` `AREA` eye catchlight aimed cleanly at passenger eye height!), `t_rails` (`X = ±0.96`), `buffers` (`X = ±0.42`), scrolling multi-story elevator shaft wall rings (`shaft_scroll_z(f)`), shaft floor doors (`FL 18 → FL 01`), and `LightPath.Is Shadow Ray -> Transparent BSDF` cutaway concrete shaft front/side walls.
2. **Elevator Cab, Doors & Safety Brakes (`src/elevator_cab_asset.py`)**:
   - Owns `build_elevator_cab(col, total_frames=1800) -> dict` returning `root` (`Cab_Root` animated via `C.cab_world_location(f)`), `cables` + `snap_sparks` (`f=325` whip/snap + orange additive spark burst), `safety_clamps` (`f=1275` rail bite + friction sparks), `floor_led` (`18 → 01`), `alarm_light`, and `force_grid` (Beat 6 emerald 12× force-distribution grid under `flat_pose`). Cab walls & doors include `xray_cab_weight(f)` transparency + shadow bypass.
3. **Passenger Character & 6-Beat Acting (`src/passenger_character.py`)**:
   - Owns `build_passenger(col, total_frames=1800) -> dict` returning `carrier`, `rig`, `body`, `equipped`, `pelvis_local`, `c1_local`, `ankle_L`, `ankle_R`.
   - 1.78m MPFB2 adult male in fitted dark navy jacket / slate chinos / sneakers (`AGENTS.md` Sec 1 photoreal eyes + `visible_shadow=False` + natural lids/jaw `<= 12 deg`).
   - Exact acting: Beat 1–2 standing -> athletic crouch (`z=0`); Beat 3 explosive jump (`+0.52m`) slamming ceiling + shocked reaction; Beat 4 `28 G` impact knee buckle & `-3.5 cm` axial compression inside `xray_body_weight(f)`; Beat 5a braced at rear handrail (`1237..1385`); Beat 5b–6 lying diagonally **FLAT ON BACK** (`PASSENGER["flat_head"]` -> `flat_feet`, arms guarding head, zero wall clipping!).
4. **Full-Body X-Ray Skeleton Rig (`src/skeleton_xray.py`)**:
   - Owns `build_skeleton_xray(col, passenger["carrier"], passenger["rig"], total_frames=1800) -> dict` driving articulated 24-vertebrae spine + sacrum/pelvis + femurs/tibias/ankles + skull glow (`C.xray_body_weight(f)`) + crimson-gold `28 G` axial shockwave (`f=940..1210`) + Beat 6 cool cyan/emerald safe-load glow (`f=1545..1800`).
5. **Cinema Camera & Framing QA (`src/cinema_camera.py` + `src/verify_framing.py`)**:
   - Owns `camera_state_at(f, include_motion_fx=True)`, `project_ndc(...)`, `build_cinema_camera(col, total_frames=1800)`. Anti-gimbal `max |fwd.z| < 0.85`, smooth Beat transitions, 100% outside solid walls, hero heads/feet/clamps/labels cleanly inside safe 9:16 zone above `NDC Y >= -0.55`.
6. **3D Obsidian Engineering Labels (`src/labels_3d.py`)**:
   - Owns `build_labels(col, font_path, cab_root, carrier, total_frames=1800) -> dict` covering all 10 `C.LABELS` badges on dark obsidian glass plaques (`pad_l = pad_r + 0.038`), `DejaVuSans-Bold.ttf`, AgX-safe emission (`1.35–2.05`), zero overlap with character or bottom subtitles.
7. **Master Scene Integrator (`src/build_elevator_scene.py`)**:
   - Wires all modules, configures Cycles (`OPTIX` + GPU `OIDN`, `transparent_max_bounces >= 14`, `AgX - Punchy`), renders `--start-frame/--end-frame` or `--frames`.
8. **Audio & Subtitles (`config/timeline.json`, `audio/gen_*.py`, `audio/generate_audio.py`, `subtitles/subtitles.ass`)**:
   - 1 Agent = 1 Single Task (`Scriptwriter`, `TTS Voice Engineer` with `edge-tts`, `BGM Composer`, `Foley SFX Designer`, `Subtitles Engineer`, `Master Mix Engineer` targeting `-14.0 LUFS` & peak `<= -1.5 dBFS`, zero word truncation).
