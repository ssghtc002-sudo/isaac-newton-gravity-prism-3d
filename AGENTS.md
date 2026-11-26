# AGENTS.md — 3D Science Short Production Playbook (V10 Cinema Standard)

Mandatory engineering standards learned and verified during **Scientist #9 (Isaac Newton)** and required for **Scientist #10+** repositories. Every agent working on 3D scenes, characters, shaders, audio, framing, or cloud rendering MUST follow these rules from Frame 1.

---

## 1. Photoreal Eyes, Cornea & Facial Realism (`src/mpfb_characters.py`, `src/newton_world_v7.py`)
Never ship dark/black/squinted MPFB2 eyes or gaping jaws:
- **Disable Eye & Eyelash Shadow Blocking**:
  - Set `eye_obj.visible_shadow = False` and `lash_obj.visible_shadow = False` on every character.
  - With `cycles.caustics_refractive = False`, a refractive outer cornea blocks 100% of direct shadow rays and causes TIR blackout on the inner iris/sclera.
- **Per-Actor Independent `EnhancedEye` Node Group**:
  - Always `node.node_tree = node.node_tree.copy()` per actor so secondary characters never overwrite the protagonist's iris color.
  - Sclera: warm ivory `(0.95, 0.92, 0.89)` (never dead flat grey/white).
  - `PupilSize = 0.24`, `InnerLayerRoughness = 0.22`, subtle `0.025` shadow-floor emission gated by `InnerOuterLayerColorSwitch` so the pupil stays pure `0.0` black while the iris/sclera stay legible in shadow.
- **Front-Facing Specular Cornea (No Refractive Blackout)**:
  - Replace refractive outer cornea BSDF with Camera-ray front-facing `LayerWeight(Fresnel, Blend=0.34)` mixing `Transparent BSDF` + ultra-crisp `Glossy BSDF(Roughness=0.014)` (`Backfacing` / non-camera rays -> pure `Transparent BSDF`).
- **Softbox Eye Catchlight (`AREA`, NEVER tiny `POINT`)**:
  - Use a `0.42m` softbox `AREA` light (`52 W` day / `62 W` night, `diffuse_factor = 0.28`, `specular_factor = 1.75`) aimed at eye height (`Z ≈ 1.62m`) via quaternion tracking. Tiny `0.08m` point lights are sub-pixel (`0.11 px`) and get erased by OIDN.
- **Natural Eyelids & Jaw Limits**:
  - Maintain a relaxed `-5.0°` upper-eyelid open baseline (`orbicularis03`); NEVER let smiles squint upper lids shut.
  - Clamp `jaw` rotation to `<= 14.0°` (typical speech `8.5°–9.5°`, shock peak `10.5°`) with smile attenuation (`jaw_calm`) so mouths never gape unnaturally.

---

## 2. Crisp 3D Formulas, Equations & Sky Titles (`src/build_*_scene.py`)
Never allow white-blown, blurry, or hand-overlapped 3D typography:
- **2-Zone Brushed-Metallic Face + Dark Extrusion Shader**:
  - Use Object-Space `|N.z|` separation (`> 0.85` = crisp satin-gold / cyan-gold face, side bevel/extrusion = dark obsidian-bronze metallic) — do NOT rely on `LayerWeight.Facing` alone (renders as `< 0.3 px` invisible rim).
  - Keep emission strength AgX-safe (`1.35–2.20`, NEVER `5.0+` which clips to white under `AgX - Punchy` and triggers FFmpeg bloom blow-out).
- **Thin Bevel + Dark Keyline Backing + Capsule Halo**:
  - Keep extrusion shallow (`0.004–0.0055m`) and bevel subtle (`0.0005–0.0007m`) so glyphs don't bloat.
  - Add a slightly recessed (`+0.006m` behind text) dark obsidian keyline backing + capsule (`x⁴ + y²`) sky veil halo for instant daylight & night legibility.
- **Native Unicode Math Glyphs**:
  - Use `DejaVuSans-Bold.ttf` (bundled in `subtitles/fonts/`) with real math typography (`·`, `₁`, `₂`, `²`, `−`, `×`, `→`, `ε`, `Φ`, `ρ`). Verify every glyph exists in the font before rendering.
- **Exact DOF Focus Plane & Zero Hand Overlap**:
  - Place floating 3D formulas on the **exact camera DOF focus plane** (`Y ≈ 0.20m`) and cleanly **above actor heads (`Z ≈ 2.00–2.06m`)** — NEVER at chest/hand height (`Z ≈ 1.24m`) where gesturing hands clip through text.
  - Place Beat 6 finale monument title cleanly in open sky (`Z ≈ 2.48–2.60m`), clear of Moon/props.

---

## 3. Volumetric Optics, Beams, Lightning & Glass Props (`src/build_*_scene.py`)
- **Open-Ended Additive Optical Cones/Ribbons**:
  - Always pass `end_fill_type="NOTHING"` on volumetric beam/cone meshes so flat end-caps never poke through screens, easels, or lenses.
  - Use `ShaderNodeAddShader` (`Emission` + `Transparent`) driven by `Camera Ray * (1 - Backfacing)` — NEVER high-opacity double-walled `MixShader` which turns beams into opaque pastel walls under AgX.
- **`orient_beam` Axis Discipline**:
  - `to_track_quat("Z", "Y")` maps local `+Y` to world `+Z` (vertical). Always keep cross-sectional scaling explicit (`(radius_scale * width_mult, radius_scale, length)`) and clamp beam endpoints to the **front plane** of any receiving screen/easel (zero back-face bleed-through).
- **Refractive Glass Shadow Bypass**:
  - Every `Transmission Weight > 0.5` glass prop (prism, lens, jar, vacuum tube) MUST route `LightPath.Is Shadow Ray -> Transparent BSDF` so Cycles (`caustics_refractive = False`) never casts solid black shadows or blocks internal light rays.
- **Camera-Facing Science Displays**:
  - Verify `dot(display_normal, to_camera) > +0.55` for every easel, dial, book page, or diagram — never angle a key physics display away from the lens.
- **Bounce Budget**:
  - Set `cycles.transparent_max_bounces >= 10` whenever layered additive beams + transparent hair/corneas overlap.

---

## 4. Zero-Clipping Hand Choreography & Framing QA (`src/verify_framing.py`)
- **Hand-to-Prop Clearance**:
  - Audit every gesturing hand pose against nearby 3D props (gravity wells, coils, lenses, tables) at peak swing frames; keep `>= 0.14m` minimum clearance so fingers never slice through hero props.
- **Automated 6-Beat Sightline & Station Isolation Check**:
  - Maintain `src/verify_framing.py` projecting every hero head, formula, prop, and adjacent station edge into camera NDC (`[-1, 1]`) across all 6 beats.
  - Every commit touching layout or camera MUST pass `python3 src/verify_framing.py` with **100% CLEAR** head/prop separation and **OUTSIDE** station-isolation margins.

---

## 5. Studio Multi-Voice Audio & Subtitle Sync (`audio/generate_audio.py`, `config/timeline.json`)
- **Microsoft Neural Voices (`edge-tts`) — Never Rushed Local Int8 TTS**:
  - Use `edge-tts` studio neural voices (e.g., `en-US-BrianNeural` narrator, `en-GB-ThomasNeural` / `en-GB-RyanNeural` / period-appropriate accents) instead of sped-up `1.44x` Kokoro int8.
- **Zero Word Truncation**:
  - Never hard-slice (`[:max_samples]`) speech arrays to fit a beat. Use natural comma phrasing (avoid mid-line `!`/`.` silence pauses) + adaptive gentle `ffmpeg atempo` (`<= 1.12x`) + generously sized `timeline.json` beat windows.
- **Broadcast Mix Targets**:
  - Vocal-pocket EQ cut on BGM (`250–3500 Hz`), `-9.5 dB` sidechain BGM ducking under narration, `-14.0 LUFS` integrated loudness, true peak `<= -1.5 dBFS`.
  - Always verify generated `master_audio.wav` durations against `voice_manifest.json` and `subtitles/subtitles.ass` karaoke timings.

---

## 6. Native Full HD → True 4K Cloud Render & 1-Tap Mobile Download (`render_on_modal.py`, `.github/workflows/render-modal-4k.yml`)
- **Default Pipeline = Modal 10-GPU Parallel Render (`~3–4 min` warm)**:
  - Render native **Full HD (`1080×1920`, `scale=100%`, `24 fps`, Cycles OPTIX + GPU OIDN)** across `10` parallel Modal GPUs.
  - Export **True 4K (`2160×3840`)** Lanczos ARRI-Alexa finished MP4 (`libx264`, `High@L5.1`, `CRF 15`, `yuv420p`, `AAC 192k`, `-movflags +faststart` for instant mobile/Reels/Shorts playback).
- **Tightened ARRI-Alexa Bloom (Preserves Sharp Text & Jewels)**:
  - Use `scale=2160:3840:flags=lanczos` (NEVER invalid `:accurate_rnd=1` option!).
  - Highlight bloom: `eq=brightness=-0.52:contrast=2.8,gblur=sigma=22` blended with `screen:all_opacity=0.20` (1080p: `sigma=12`, `opacity=0.22`).
- **Keep Modal Container Cache Warm**:
  - Do NOT alter `modal.Image` setup layers unnecessarily once cached; pull latest repo code at runtime via `git fetch --depth 1 origin main && git reset --hard origin/main`.
- **1-Tap Mobile Download via GitHub Actions + Releases**:
  - Include `.github/workflows/render-modal-4k.yml` (`workflow_dispatch`, `permissions: contents: write`) using repo secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, publishing the finished 4K `.mp4` to GitHub Release tag `latest-4k` for instant 1-tap mobile download (`https://github.com/<owner>/<repo>/releases/download/latest-4k/<file>.mp4`).
  - Always test the exact `ffmpeg -filter_complex` string locally on a `0.1s` synthetic clip before pushing.

---

## 7. Multi-Agent QA Workflow Before Every Final 4K Render
Before kicking off any 4K cloud render, run parallel verification:
1. `python3 -m py_compile` + AST undefined-name check on all Python files.
2. `python3 src/verify_framing.py` (100% PASSED across all 6 beats).
3. Live `ffmpeg` 0.1s filter-graph validation with bundled `subtitles/subtitles.ass` & `subtitles/fonts/`.
4. Audio LUFS / peak / duration audit (zero truncated trailing syllables).
