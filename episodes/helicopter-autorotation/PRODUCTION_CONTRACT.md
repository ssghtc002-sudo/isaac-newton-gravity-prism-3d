# PRODUCTION CONTRACT — Episode #5 (V2 Ultimate): "Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock"

Single source of truth for all 3D modules, photoreal 4K MPFB2 cockpit crew in dark-navy Nomex flight suits, modern twin-engine rescue helicopter (Airbus H145 / Coast Guard Dauphin silhouette), local-space 12-sprag freewheeling clutch X-ray, hero 2.1× botanical 3D maple seed comparison, curtain-wall glass skyline with rising sunset cloud strata & weathered rooftop helipad, 8-segment cinema camera, Hollywood suspense score + blockbuster Foley SFX, narration, subtitles, and Modal 4K / `--draft-preview` cloud rendering. Follows **AGENTS.md V12** (`60 fps`, `1800 frames`, `30.0 s`, `1080×1920` native → `2160×3840` True 4K, Blender `5.3.0` Cycles OPTIX + GPU OIDN, `AgX - Punchy`).

---

## 1. 8-Segment Action-Science Architecture (`1..1800` @ `60 fps` = `30.0 s`)

| Segment | Frames | Time | Visual Story, Crew & Camera | 3D Callout (`<= 3 words`) |
|---|---|---|---|---|
| `B1A_FIRE` | `1..115` | `0.00–1.92 s` | Exterior modern rescue helicopter banking over sunlit curtain-wall glass skyline: dorsal turbine erupts in 3D turbulent compressor-stall fire burst, curved metallic sparks & billowing dark exhaust smoke plume (`f/7.5`) | `ENGINE DEAD` (`26..108`) |
| `B1B_COCKPIT` | `116..240` | `1.92–4.00 s` | Hard cut INSIDE dark enclosed 1:1 cockpit (bulkhead, side doors, overhead panel, dual MFDs): crimson `ENG OUT` bar flashes, live 3D `ENG` needle crashes to `0%`, wide-eyed `Heli_Pilot` slams the collective lever straight down while `Heli_CoPilot` spots the helipad (`f/7.5`) | *(pure cockpit drama)* |
| `B2_CLUTCH` | `241..570` | `4.00–9.50 s` | 3/4 dorsal macro on rescue helicopter: local-space hollow Fresnel X-ray reveals crimson-hot seized turbine bevel gear + stall halo locking dead while **12 chamfered golden cam sprags** tilt open and spinning cyan-gold freewheel motion-arc arrows detach (`f/7.8`) | `SPRAG CLUTCH` (`290..552`) |
| `B3_AIRFLOW` | `571..960` | `9.50–16.00 s` | Dynamic aerial autorotation glide with rising sunset cloud strata parallax: **6 wide sculpted translucent emerald 3D ribbon arrows** surge UP through the spinning blades to charge stored kinetic flywheel spin (`f/7.8`) | `UPWARD AIRFLOW` (`615..942`) |
| `B4_MAPLE` | `961..1290` | `16.00–21.50 s` | Side-by-side golden-hour comparison: enlarged `2.1×` sculpted botanical 3D maple samara (veined papery wing + weighted nutlet) glides beside the modern rescue rotor inside ONE clean lift swirl (`f/7.5`) | `MAPLE SEED LIFT` (`1000..1272`) |
| `B5A_COCKPIT` | `1291..1450` | `21.50–24.17 s` | Label-free shallow-DOF (`f/3.6`) dark enclosed cockpit close-up: radar altimeter bar drops `125 ft → 32 ft` as focused `Heli_Pilot` hauls collective UP & cyclic back while `Heli_CoPilot` calls altitude | *(pure visual macro)* |
| `B5B_FLARE` | `1451..1580` | `24.17–26.33 s` | Crisp `f/8.0` exterior low-angle beside weathered rooftop helipad: chopper pitches nose-up (`+22°`, `+4.2°` rotor blade coning) at 30 ft, blasting an air-cushion shockwave | `30 FT FLARE` (`1466..1568`) |
| `B6_LANDING` | `1581..1800` | `26.33–30.00 s` | Weathered rooftop helipad payoff (safety netting, pulsing green/white strobes, whipping windsock): chopper levels smoothly to exact `0.000 mm` flush skid contact (`Z = 0.90 m`) kicking up an expanding golden ground-effect dust wave (`f/7.5`) | `ZERO ENGINE` (`1615..1782`) |

Hard cut boundaries (`CONSTANT` interpolation): `116`, `241`, `571`, `961`, `1291`, `1451`, `1581`.

---

## 2. Exact Spatial Coordinate System

- **World Frame:** `+X` right, `+Y` away from camera (skyline depth), `+Z` up.
- **Dark Enclosed 1:1 Rescue Cockpit & Crew (`Beats 1B & 5A`):**
  - Cockpit floor at `Z = 0.00 m`; widened `Heli_Pilot` (PIC, right seat) at `(+0.40, +0.34, 0.00)` and `Heli_CoPilot` (left seat) at `(-0.40, +0.34, 0.00)` in matte dark-navy Nomex flight suits + sculpted 4-point harnesses (`yaw = 180°`, seated eye `Z = 1.38 m`, zero arm crossover).
  - Flashing `ENG OUT` caution bar at `(0.00, -0.34, 1.11)`, live 3D `ENG`/`ROTOR` RPM gauges at `(0.00, -0.26, 1.06)`, collective lever pivoted at `(+0.12, +0.16, 0.64)`.
- **Exterior Modern Twin-Engine Rescue Helicopter (`Heli_Root`, 1:6 scale diorama, Beats `1A, 2, 3, 4, 5B, 6`):**
  - Multi-section lofted H145/Dauphin white-and-crimson fuselage (`~2.35 m` length with shrouded Fenestron 8-blade tail, nose radome, chin windows, wire cutters, hoist winch, searchlight, dual heat-blued exhausts).
  - 4-blade main rotor radius `1.22 m` with radial motion-blur disc & `+4.2°` flare coning; hub at local `Z = +0.45 m`, sprag clutch at local `(0.00, +0.02, +0.30)`, turbine exhaust at `(0.00, +0.42, +0.26)`, stepped skid bottoms strictly at local `Z = -0.340 m`.
  - Rooftop helipad surface at `Z = 0.900 m` (`(0.08, 0.45, 0.90)`) -> `Heli_Root` settles to `(0.08, 0.45, 1.24)` (`1.24 - 0.34 = 0.900 m`, exact `0.000 mm` flush contact!).
- **Hero 2.1× Sculpted Botanical 3D Maple Seed (`Heli_MapleSeed_Root`, Beat 4):**
  - Descends & autorotates on the left `(-0.54, -0.18, 2.24)` → `(-0.50, -0.14, 1.76)` alongside `Heli_Root` on the right `(+0.58, +0.42, 2.24)` → `(+0.54, +0.36, 1.78)`.

---

## 3. Exact Frame Milestones (`EV` synced 1:1 with `config/timeline.json`)

| Event Key | Frame | Time (s) | Action / Foley Trigger |
|---|---:|---:|---|
| `b1_engine_flameout` | `18` | `0.30` | Exterior compressor-stall explosion BANG, fire burst, sparks & turbine spool-down |
| `b1_caution_alarm` | `122` | `2.03` | Hard cut into dark cockpit: dual-tone Master Caution horn, `ENG 0%` crash & heartbeat |
| `b1_collective_slam` | `152` | `2.53` | Pilot slams collective pitch lever straight down into bottom detent |
| `b2_xray_sweep` | `265` | `4.42` | Local-space Fresnel X-ray sweeps dorsal turbine cowling |
| `b2_shaft_seize` | `345` | `5.75` | Seized crimson turbine gear clunks dead with glowing stall halo |
| `b2_sprag_unlock` | `405` | `6.75` | 12 golden cam sprags tilt open with metallic bicycle-freewheel ratchet |
| `b3_airflow_surge` | `600` | `10.00` | 6 wide sculpted translucent emerald 3D ribbon arrows surge up through blades |
| `b3_rotor_windmill` | `735` | `12.25` | 4-blade BVI rotor slap (`whop-whop-whop`) accelerates in autorotation |
| `b3_flywheel_charge` | `860` | `14.33` | Rotor tips glow with stored kinetic flywheel energy ring |
| `b4_maple_spin` | `995` | `16.58` | Hero 2.1× botanical maple samara whirls beside rescue rotor |
| `b4_vortex_sync` | `1130` | `18.83` | Clean matched helical lift swirls synchronize seed and rotor |
| `b5_cockpit_pull` | `1345` | `22.42` | Pilot hauls collective UP inside enclosed cockpit as altimeter drops to 32 ft |
| `b5_flare_bite` | `1462` | `24.37` | Post-vacuum (`0.35s` drop) massive 30-ft flare blade-bite air chop + Braam! |
| `b5_cushion_wash` | `1522` | `25.37` | Downwash air-cushion shockwave kills sink rate above weathered helipad |
| `b6_skid_touchdown` | `1685` | `28.08` | Stepped skids kiss weathered rooftop helipad smoothly + golden dust wave |
| `b6_rotor_coast` | `1735` | `28.92` | Unpowered main rotor coasts gently to rest under resolving Dmaj9 chord |

---

## 4. Blender 5.3.0 & V12 Compliance Checklist

1. **Slotted Actions:** All FCurves accessed exclusively via `heli_contract.iter_action_fcurves(action)`.
2. **ShaderNodeMix:** All `(Factor, A, B)` sockets resolved via `heli_contract.get_mix_sockets(node)` (`inputs[2:]`).
3. **Materials Before `to_mesh`:** Every material appended to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
4. **Open-Ended Volumetric Ribbons, Swirls & Dust Waves:** `cap_ends=False` / `end_fill_type="NOTHING"` + `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
5. **Local-Space Cowling X-Ray & Glass Bypass:** Driven off `ShaderNodeTexCoord.Object` (`Heli_Root`), hollow Fresnel `pow(Facing, 3.8)`, plus `LightPath.Is Shadow Ray -> Transparent BSDF` on all canopy glass.
6. **MPFB2 V12 Eye Realism & Flight Suits:** Independent `EnhancedEye` node group copy per aviator, `visible_shadow = False` on eyes/lashes, front-facing specular corneas, matte dark-navy Nomex material override + sculpted 4-point rescue harnesses, `-5°` relaxed upper eyelids + panic/focus/relief acting, jaw `<= 14°`, `UnderClothes_Shrink` (`-0.011`) before `ARMATURE`, `0.42 m` softbox eye catchlight (`AREA`).
7. **Razor-Sharp 3D Labels:** `6` ultra-short (`1–3 word`) callouts on dark obsidian backing plaques (`DejaVuSans-Bold.ttf`, Object-Space `|N.z| > 0.85`, focus-plane pinned at `f/7.5..f/8.0`, `NDC Y >= -0.45`, `visible_glossy = visible_transmission = visible_diffuse = False`).
8. **Audio, Subtitles & Modal Warm Cache:** `94 words` (`en-US-BrianNeural`, `atempo <= 1.12x`), `48000 Hz` Stereo `-14.0 LUFS` / `<= -1.5 dBFS`, Hollywood orchestral-hybrid score + `0.35s` pre-flare vacuum + layered aviation Foley, synced `\kf` ASS karaoke (`NDC Y = -0.583 < -0.55`), byte-for-byte cached `studio_image`, `--draft-preview`, and automatic narration-vs-manifest rebuild check.
