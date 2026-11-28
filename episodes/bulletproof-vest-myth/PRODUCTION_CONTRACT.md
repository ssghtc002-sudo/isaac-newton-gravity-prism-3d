# Episode #3 Production Contract: *Why Bulletproof Vests DON'T Send You Flying Backward*

- **Episode Folder**: `episodes/bulletproof-vest-myth/`
- **Standard**: `AGENTS.md` **V12** (`@zackdfilms` 10-Short Blueprint + Blender `5.3.0` + `60 fps` / `1800 frames` / `30.0 s` + `48 kHz / 320 kbps` broadcast audio + **720p+ HD `67% / 16 samples` Modal QA**).
- **Single Source of Truth**: `src/vest_contract.py` (`FPS=60`, `TOTAL_FRAMES=1800`, `BEATS`, `EV`, `RANGE`, `VEST`, `HERO`, `SHOOTER`, `LABELS`, `CAMERA_TARGETS`, deterministic trajectory/recoil/cutaway evaluators, `iter_action_fcurves()`, and `get_mix_sockets()`).

---

## 1. Coordinate System & 1:1 Metric Scale
- **World Frame (`1.0 BU = 1.0 m`, `INCH_TO_M = 0.0254`, `HALF_INCH_M = 0.0127`)**:
  - `Z = 0.00 m`: Ballistic testing range floor datum (matte dark basalt-concrete with recessed glowing cyan metric range markers every `0.50 m`).
  - `Z = 1.32 m`: Primary sternum / chest-plate line of fire (`CHEST_Z`).
  - `Z = 1.62 m`: Hero & shooter eye height (`EYE_Z`, targeted by `0.42 m` softbox `AREA` catchlights).
  - `X = -1.60 m`: Beat 2 shooter standing position (`muzzle_x = -1.10 m`).
  - `X = +1.00 m`: Hero chest-armor strike surface (`VEST_SURFACE_X`) & local origin of `Vest_Macro_Root` `(1.00, 0.00, 1.32)`.
  - `X = +1.15 m`: Hero target standing position (Beats 1, 2, 6), facing `-X` (`yaw = -82°` in B1, `-76°` in B6) for 3/4 face + chest-plate readability from camera `-Y`.
  - `X = +1.85 m`: Pristine frameless tempered-glass backdrop wall behind hero (Beat 1 & 6 visual proof that hero does NOT crash through glass—shifts only `0.0127 m = 0.5 in`; zero opaque center mullions).
- **Macro Vest Cutaway Stack (`Vest_Macro_Root` Local Space, 100% `ShaderNodeTexCoord.Object`)**:
  - `local_x = -0.006 .. 0.000 m`: Ballistic nylon outer carrier (fades via local-space X-ray scan in Beats 3–4).
  - `local_x = 0.000 .. +0.010 m`: **Silicon-Carbide (`SiC`) Ceramic Strike Plate** (`10 mm` dark tactical graphite-ceramic tile, shatters rifle tip in Beat 4).
  - `local_x = +0.010 .. +0.028 m`: **Woven Golden Kevlar Aramid Mesh** (18 mm multi-ply macro weave that ripples radially like a taut golden trampoline in Beat 3).
  - `local_x = +0.028 .. +0.044 m`: **UHMWPE Dyneema Trauma Backer** (16 mm luminous pearlescent cyan-white laminate catching ceramic/rifle dust in Beat 4).
- **Beat 5 Studio Blueprint Arena (`1261..1560`)**:
  - Clean `@zackdfilms` blue/cyan studio blueprint backdrop + metric grid floor (`blueprint_bg_weight = 1.0`).
  - Shooter at `X = -1.20 m` (facing `+X`, muzzle `-0.82 m`) vs. Armored Target at `X = +1.20 m` (facing `-X`, vest strike face `+1.05 m`).
  - Symmetrical opposing 3D force vectors (`F₁` crimson-amber pushing shooter `-X`, `−F₂` electric cyan pushing target `+X`, equal `0.55 m` shaft length & matching `0.0127 m` recoil).

---

## 2. 6-Beat Storyboard (`1..1800` @ `60 fps` = `30.0 s`)

| Beat | Frames | Time | Action, Physics & Camera Plan | Ultra-Short 3D Callout (`NDC Y >= -0.45`) |
|------|--------|------|-------------------------------|-------------------------------------------|
| **1. Hook (`B1_HOOK`)** | `1–210` | `0.0–3.5s` | Slow-mo 3/4 chest-level tracking shot (`lens=32mm`, `f/7.5`): glowing 9mm round slams dead-center into hero's vest (`f=75`); sparks erupt off the plate while hero barely shifts `0.5 in` (`12.7 mm`) in front of the unbroken glass wall. | `0.5 INCH SHIFT` (`92..204`) |
| **2. Momentum Myth (`B2_MOMENTUM`)** | `211–510` | `3.5–8.5s` | Hard cut (`f=211`) to wide ballistic range profile (`lens=26mm`, `f/7.5`): shooter fires (`f=245`), bullet streaks across range and hits chest plate (`f=335`), exposing tiny linear momentum (`~3 kg·m/s` vs `6 kg·m/s` fastball). | `TINY MOMENTUM` (`330..495`) |
| **3. Kevlar Trampoline (`B3_KEVLAR`)** | `511–900` | `8.5–15.0s` | Hard cut (`f=511`) to action-locked macro vest cutaway (`lens=38mm`, `f/4.0 -> f/7.5` before label): outer carrier X-rays open (`f=535`); 9mm round strikes golden Kevlar mesh (`f=615`), sending concentric trampoline ripples radiating across woven aramid strands (`f=735`). | `KEVLAR NET` (`595..880`) |
| **4. Ceramic + Dyneema (`B4_CERAMIC`)** | `901–1260` | `15.0–21.0s` | Smooth macro pull-back & orbit across `900->901` (`lens=38->30mm`, `f/7.5`): pointed rifle round hits front silicon-carbide ceramic plate (`f=965`), fracturing ceramic & pulverizing the steel-core tip into glowing dust (`f=1030`) caught dead by rear Dyneema backing (`f=1125`). | `CERAMIC + DYNEEMA` (`990..1245`) |
| **5. Newton's Third Law (`B5_NEWTON`)** | `1261–1560` | `21.0–26.0s` | Hard cut (`f=1261`) to clean `@zackdfilms` blue studio blueprint (`lens=26mm`, `f/8.0`): shooter fires (`f=1295`), vest absorbs impact (`f=1365`), symmetrical `F₁` & `−F₂` opposing force arrows lock in (`f=1405`) showing equal `0.5 in` recoil on both shooter and target. | `F₁ = −F₂` (`1320..1545`) / `EQUAL RECOIL` (`1385..1545`) |
| **6. Bruise Payoff (`B6_FINAL`)** | `1561–1800` | `26.0–30.0s` | Hard cut (`f=1561`) to moody 3/4 hero finale (`lens=28mm`, `f/7.5`): hero stands rock-steady (`f=1620`) before the intact glass wall as spent copper/ceramic dust drifts harmlessly to the floor—zero cinematic launch, just blunt impact energy absorbed. | `JUST A BRUISE` (`1615..1785`) |

---

## 3. Narration Script (`95 Words`, `3.17 wps`, `en-US-BrianNeural`, Comma-Phrased Zero Truncation)

- **L1 (`B1_HOOK`, `0.08–3.45s`, 13w)**: *"If a bullet hits your vest, you don't fly backward, you barely budge."*
- **L2 (`B2_MOMENTUM`, `3.62–8.40s`, 15w)**: *"Movies send heroes crashing through glass, yet a bullet carries less momentum than a fastball."*
- **L3 (`B3_KEVLAR`, `8.65–14.85s`, 18w)**: *"Instead, woven golden Kevlar catches the spinning bullet like a trampoline, spreading that force across your whole chest."*
- **L4 (`B4_CERAMIC`, `15.15–20.85s`, 18w)**: *"For faster rifle rounds, a ceramic plate shatters the metal tip to dust, while Dyneema traps every shard."*
- **L5 (`B5_NEWTON`, `21.15–25.85s`, 18w)**: *"And Newton's Third Law seals it, any shot strong enough to launch you would launch the shooter too."*
- **L6 (`B6_FINAL`, `26.15–29.85s`, 13w)**: *"So real armor never sends you flying, it just leaves one brutal bruise."*

---

## 4. Technical & QA Gates (Non-Negotiable V12 Contracts)
- **Blender 5.3.0 Safety**: All FCurve access MUST use `vest_contract.iter_action_fcurves(action)`; all `ShaderNodeMix` socket access MUST use `vest_contract.get_mix_sockets(node)` (slicing `node.inputs[2:]`); all mesh materials MUST be appended BEFORE `bm.to_mesh(mesh)`; all open X-ray/spark glow cones use `end_fill_type="NOTHING"` (`cap_ends=False`) + `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`; all refractive glass/X-ray shells bypass `LightPath.Is Shadow Ray -> Transparent BSDF`.
- **MPFB2 Eyes & Wardrobe**: Verified assets from `swimmer_character.py`, `eye_obj.visible_shadow = False`, `lash_obj.visible_shadow = False`, per-actor copied `EnhancedEye`, front-facing Glossy cornea, `-5°` relaxed upper eyelids, jaw clamped `<= 14°`, `0.42 m` softbox `AREA` eye catchlights, and `upperarm`/`lowerarm` included in `UnderClothes_Shrink` (`-0.011`) before `ARMATURE`.
- **3D Callouts (`labels_3d.py`)**: Strictly `focus_dist - 0.05 m`, `f/7.5+` while active, `char_h_ndc >= 0.034`, `bevel_depth = size_ref * 0.0022`, `extrude = size_ref * 0.022`, `space_character = 1.10`, zero `Keyline` mesh, dark obsidian plaque `(0.008, 0.012, 0.020)`, Object-Space `|N.z| > 0.85` face shader (`Emission <= 1.85`), and `visible_glossy = visible_transmission = visible_diffuse = False`.
- **Audio & SFX**: Real CC0/PD WAV samples (`pd_gunshot_a/b/c.wav`, `cc0_sub_braam.wav`) + `40–140 Hz` acoustic sub-punch, `SFX_GAIN >= 1.05`, transient guard `>= 350 ms`, `-14.0 LUFS`, True Peak `<= -1.5 dBFS`, `48000 Hz` Stereo, `AAC 320 kbps`.
- **Pre-Flight QA**: `verify_framing.py` (`100% PASSED`), `labels_3d.py` (`0 errors`), `render_on_modal.py` with identical `studio_image` layers and `EPISODE_SUBDIR = "episodes/bulletproof-vest-myth"`.
