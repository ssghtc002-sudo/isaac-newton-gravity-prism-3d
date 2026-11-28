# Episode #2 Production Contract: *Why Bullets Stop Dead Just 3 Feet Underwater*

- **Episode Folder**: `episodes/underwater-bullets/`
- **Standard**: `AGENTS.md` **V12** (`@zackdfilms` 10-Short Blueprint + Blender `5.3.0` + `60 fps` / `1800 frames` / `30.0s` + `48 kHz / 320 kbps` broadcast audio + **720p+ HD `67% / 16 samples` Modal QA**).
- **Single Source of Truth**: `src/bullet_contract.py` (`FPS=60`, `TOTAL_FRAMES=1800`, `BEATS`, `EV`, `TANK`, `BULLET`, `SWIMMER`, `LABELS`, and deterministic trajectory evaluators + `iter_action_fcurves()`).

## Coordinate System & Metric Scale
- **1:1 Physical Metric Scale (`FT_TO_M = 0.3048`)**:
  - `X = 0.00 m`: Water entry datum / barrel firing line (`0 FT`).
  - `X = +0.9144 m`: **`3 FT`** supersonic fragmentation/stop line (`THREE_FT_X`).
  - `X = +1.8288 m`: **`6 FT`** slower heavy subsonic glide zone (`SIX_FT_X`).
  - `Z = 0.00 m`: Waterline meniscus inside **Crystal Ballistic Tank & Underwater Studio** (`Z = -2.20 m` illuminated measurement floor grid, glowing cyan distance bars `0 FT..6 FT`).
  - `Z = -0.55 m`: Primary Beat 1–4 bullet flight path (`FLIGHT_Z`).
  - Beat 5 Split Race Lanes: Top Lane `Z = -0.28 m` (`SUPERSONIC`, `1,900 mph` → stops dead at `3 FT`), Bottom Lane `Z = -0.84 m` (`SLOWER ROUND`, `750 mph` → glides smoothly past `3 FT` toward `6 FT`).

## 6-Beat Storyboard (`1..1800` @ `60 fps`)

| Beat | Frames | Time | Action & Camera Flow | Ultra-Short 3D Callout |
|------|--------|------|----------------------|------------------------|
| **1. Hook** | `1–210` | `0.0–3.5s` | Subsurface tracking shot: glowing copper bullet streaks toward submerged diver (`X=1.48m`), screeches dead at **`3 FT`** (`X=0.914m`) before reaching chest. | `3 FT STOP` |
| **2. Density Wall** | `211–540` | `3.5–9.0s` | Hard cut to muzzle flash (`X=-0.35m`) -> bullet punches air-to-water barrier (`X=0.0m`) into `800×` water density streamlines. | `800× DENSER` |
| **3. Supercavitation** | `541–900` | `9.0–15.0s` | Tight action-locked macro ride (`f/3.6` -> `f/7.5` during label): spinning nose boils water into a shimmering silver gas cavitation envelope & wake spirals. | `VAPOR BUBBLE` |
| **4. Violent Shred** | `901–1260` | `15.0–21.0s` | X-ray jacket cutaway exposes stressed lead core (`amber -> crimson`) as asymmetric bubble collapse pitches bullet `24°` sideways and shatters it into 8 spinning fragments at **`3 FT`**. | `SHATTERED` |
| **5. Paradox Race** | `1261–1560` | `21.0–26.0s` | Clean `@zackdfilms` blue studio blueprint split race: Top (`1,900 mph`) explodes & stops at `3 FT`; Bottom (`750 mph`) heavy round holds laminar stream past `3 FT -> 6 FT`. | `FASTER: 3 FT` / `SLOWER: 6 FT` |
| **6. Survival Payoff** | `1561–1800` | `26.0–30.0s` | Diver calmly drifts safely below `Z=-1.05m` past the glowing cyan `3 FT` zone as crushed supersonic copper scrap flutters slowly downward harmlessly. | `SAFE BELOW 3 FT` |
