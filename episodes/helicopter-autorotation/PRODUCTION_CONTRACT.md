# PRODUCTION CONTRACT — Episode #5: "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock"

Single source of truth for all 3D modules, photoreal 4K MPFB2 cockpit crew, white-and-red rescue twin-blade helicopter, local-space freewheeling sprag clutch X-ray, animated 3D maple seed comparison, golden-hour skyline & rooftop helipad environment, camera framing, Foley SFX, narration, subtitles, and Modal 4K / `--draft-preview` cloud rendering. Follows **AGENTS.md V12** (`60 fps`, `1800 frames`, `30.0 s`, `1080×1920` native → `2160×3840` True 4K, Blender `5.3.0` Cycles OPTIX + GPU OIDN, `AgX - Punchy`).

---

## 1. 6-Beat Action-Science Architecture (`1..1800` @ `60 fps` = `30.0 s`)

| Beat | Frames | Time | Visual Story, Crew & Camera | 3D Callout (`<= 3 words`) |
|---|---|---|---|---|
| `B1_HOOK` | `1..240` | `0.00–4.00 s` | 1:1 Rescue Cockpit over golden-hour city skyline: `ENG OUT` crimson panel flashes, smoke trails outside, MPFB2 `Heli_Pilot` slams the left-hand collective lever straight down while `Heli_CoPilot` spots the rooftop helipad (`f/7.5`) | `ENGINE DEAD` (`64..228`) |
| `B2_CLUTCH` | `241..570` | `4.00–9.50 s` | 3/4 dorsal macro on white-and-red rescue helicopter: local-space hollow Fresnel X-ray sweeps cowling revealing seized crimson turbine shaft locking dead while golden steel sprag clutch cam rollers snap open (`f/7.8`) | `SPRAG CLUTCH` (`295..552`) |
| `B3_AIRFLOW` | `571..960` | `9.50–16.00 s` | Dynamic aerial descent above sunset towers: open-ended glowing emerald-cyan air streamlines surge UP through the negatively pitched rotor disk, accelerating stored flywheel spin (`f/7.8`) | `UPWARD AIRFLOW` (`620..942`) |
| `B4_MAPLE` | `961..1290` | `16.00–21.50 s` | Side-by-side golden-hour physics comparison: sculpted spinning 3D maple seed (samara) on the left glides alongside the descending rescue rotor on the right inside matched helical emerald vortex ribbons (`f/7.5`) | `MAPLE SEED LIFT` (`1005..1272`) |
| `B5A_COCKPIT` | `1291..1450` | `21.50–24.17 s` | Label-free shallow-DOF (`f/3.6`) 1:1 cockpit macro: rooftop helipad rises outside the panoramic glass as `Heli_Pilot` pulls the collective lever UP and cyclic back while `Heli_CoPilot` calls altitude on comms | *(pure visual macro)* |
| `B5B_FLARE` | `1451..1580` | `24.17–26.33 s` | Crisp `f/8.0` exterior low-angle beside the rooftop helipad: rescue helicopter pitches nose-up (`+21°`) at 30 ft, blasting a glowing golden-cyan downward cushion wash ring | `30 FT FLARE` (`1468..1568`) |
| `B6_LANDING` | `1581..1800` | `26.33–30.00 s` | Golden-hour rooftop payoff: rescue chopper levels smoothly onto the yellow `H` helipad (`Z = 0.90 m` flush skid contact at `f = 1685`) as the coasting blades slow safely on zero power (`f/7.5`) | `ZERO ENGINE` (`1615..1782`) |

Hard cut boundaries (`CONSTANT` interpolation): `241`, `571`, `961`, `1291`, `1451`, `1581`.

---

## 2. Exact Spatial Coordinate System

- **World Frame:** `+X` right, `+Y` away from camera (skyline depth), `+Z` up.
- **1:1 Rescue Cockpit & Crew (`Beats 1 & 5A`):**
  - Cockpit floor at `Z = 0.00 m`; `Heli_Pilot` (PIC, right seat) at `(+0.36, +0.34, 0.00)` and `Heli_CoPilot` (left seat) at `(-0.36, +0.34, 0.00)`, both facing `-Y` (`yaw = 180°`, seated eye `Z = 1.38 m`).
  - Glare-shield & flashing `ENG OUT` annunciator at `(0.00, -0.36, 1.14)`; animated collective pitch lever pivoted at `(+0.10, +0.20, 0.66)`.
- **Exterior White-and-Red Rescue Helicopter (`Heli_Root`, 1:6 scale diorama, Beats `2, 3, 4, 5B, 6`):**
  - Main rotor radius `1.15 m` (`2.30 m` diameter), hub at local `Z = +0.44 m`, sprag clutch at local `(0.00, +0.02, +0.30)`, twin steel skid bottoms strictly at local `Z = -0.34 m`.
  - Rooftop helipad surface at `Z = 0.90 m` (`(0.08, 0.45, 0.90)`) -> `Heli_Root` settles to `(0.08, 0.45, 1.24)` (`1.24 - 0.34 = 0.900 m`, exact `0.000 mm` flush contact!).
- **Sculpted 3D Maple Seed (`Heli_MapleSeed_Root`, Beat 4):**
  - Descends & autorotates on the left `(-0.56, -0.10, 2.26)` → `(-0.52, -0.06, 1.76)` alongside `Heli_Root` on the right `(+0.58, +0.40, 2.24)` → `(+0.54, +0.34, 1.78)`.

---

## 3. Exact Frame Milestones (`EV` synced 1:1 with `config/timeline.json`)

| Event Key | Frame | Time (s) | Action / Foley Trigger |
|---|---:|---:|---|
| `b1_engine_flameout` | `14` | `0.23` | Turbine flame-out spool-down & smoke puff outside canopy |
| `b1_caution_alarm` | `42` | `0.70` | Crimson `ENG OUT` cockpit Master Caution chime |
| `b1_collective_slam` | `120` | `2.00` | Pilot slams collective pitch lever straight down |
| `b2_xray_sweep` | `270` | `4.50` | Local-space Fresnel X-ray peels back dorsal cowling |
| `b2_shaft_seize` | `340` | `5.67` | Seized crimson turbine drive shaft clunks to dead stop |
| `b2_sprag_unlock` | `415` | `6.92` | Golden sprag clutch rollers snap open with bicycle ratchet click |
| `b3_airflow_surge` | `605` | `10.08` | Emerald upward airflow streamlines rush through rotor disk |
| `b3_rotor_windmill` | `740` | `12.33` | Blades accelerate in wind-turbine autorotation slap |
| `b3_flywheel_charge` | `865` | `14.42` | Rotor tips glow with stored kinetic flywheel ring |
| `b4_maple_spin` | `995` | `16.58` | 3D winged maple seed glides & whirls beside rescue rotor |
| `b4_vortex_sync` | `1125` | `18.75` | Matched emerald helical vortices lock both rotors in sync |
| `b5_cockpit_pull` | `1350` | `22.50` | Pilot yanks collective lever UP in 1:1 cockpit close-up |
| `b5_flare_bite` | `1465` | `24.42` | 30-ft exterior nose-up flare bites air at `+13°` blade pitch |
| `b5_cushion_wash` | `1525` | `25.42` | Glowing golden-cyan downwash cushion ring kills sink rate |
| `b6_skid_touchdown` | `1685` | `28.08` | Twin skids kiss yellow `H` rooftop helipad smoothly |
| `b6_rotor_coast` | `1735` | `28.92` | Unpowered main rotor coasts gently to rest in golden sunset |

---

## 4. Blender 5.3.0 & V12 Compliance Checklist

1. **Slotted Actions:** All FCurves accessed exclusively via `heli_contract.iter_action_fcurves(action)`.
2. **ShaderNodeMix:** All `(Factor, A, B)` sockets resolved via `heli_contract.get_mix_sockets(node)` (`inputs[2:]`).
3. **Materials Before `to_mesh`:** Every material appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
4. **Open-Ended Volumetric Streamlines & Rings:** `cap_ends=False` / `end_fill_type="NOTHING"` + `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
5. **Local-Space Cowling X-Ray & Glass Bypass:** Driven off `ShaderNodeTexCoord.Object` (`Heli_Root`), hollow Fresnel `pow(Facing, 3.8)`, plus `LightPath.Is Shadow Ray -> Transparent BSDF` on all bubble canopy glass.
6. **MPFB2 V12 Eye Realism:** Independent `EnhancedEye` node group copy per aviator, `visible_shadow = False` on eyes/lashes, front-facing specular corneas, `-5°` relaxed upper eyelids, jaw `<= 14°`, `UnderClothes_Shrink` (`-0.011`) before `ARMATURE`, `0.42 m` softbox eye catchlight (`AREA`).
7. **Razor-Sharp 3D Labels:** `6` ultra-short (`1–3 word`) callouts on dark obsidian backing plaques (`DejaVuSans-Bold.ttf`, Object-Space `|N.z| > 0.85`, focus-plane pinned at `f/7.5..f/8.0`, `NDC Y >= -0.45`, `visible_glossy = visible_transmission = visible_diffuse = False`).
8. **Audio, Subtitles & Modal Warm Cache:** `94 words` (`en-US-BrianNeural`, `atempo <= 1.12x`), `48000 Hz` Stereo `-14.0 LUFS` / `<= -1.5 dBFS`, synced `\kf` ASS karaoke (`NDC Y = -0.583 < -0.55`), byte-for-byte cached `studio_image`, `--draft-preview`, and automatic narration-vs-manifest rebuild check.
