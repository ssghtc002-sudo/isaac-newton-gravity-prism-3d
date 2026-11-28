# AGENTS.md — 3D Science & Action Short Production Playbook (V12 Viral Shorts Standard)

Mandatory engineering & creative standards verified across **Episode #1–#16 (`fighter-jet-ejection`, `falling-elevator`)** plus a **10-Short empirical breakdown of `@zackdfilms`** (`8R9-se0PJmQ`, `XlN1Ms4T6NU`, `RwYfQpbhpi0`, `1Tdmv3pWopU`, `2QIBEpASFJQ`, `IBstPhjKTJs`, `UjhTobLcDio`, `ZQob0-WKtR0`, `COyHk-qhq8k`, `wavxRZZNgaQ`). Every agent working on scripts, voiceover, camera, shaders, 3D graphics, subtitles, or Modal 4K rendering MUST follow V12 from Frame 1.

---

## 0. `@zackdfilms` Viral Craft Blueprint (Mandatory Script, Graphics, Camera & Audio Formula)
Apply this proven 10-Short formula to every new episode:
- **Script & Voiceover Cadence (`85–125 words` for `30s`, `~2.8–3.3 words/sec`)**:
  - **Sentence 1 In-Medias-Res Hook (`0–3s`)**: Jump straight into action or a concrete premise (`"If you go through..."`, `"He shot the pistol..."`, `"You strap..."`). Zero greetings or intros.
  - **Cause-and-Effect Escalation**: Use tight 8–12 word beats connected by crisp pivots (`"But..."`, `"Instead..."`, `"So..."`, `"Now..."`) toward a twist, engineering loophole, malfunction, or dark origin payoff. Match voiceover POV to **2nd-person `"you/your"`** for explainer/survival guides and **3rd-person past** for real events.
  - **Calm Deadpan Authority**: Deliver narration smoothly, neutrally, and clearly (`edge-tts`) with **zero dead air** — let calm voice contrast against high-stakes action, layered with crisp action-synced **Foley SFX** (clicks, fractures, snaps, sub-drops).
- **Camera & Motion Grammar (`cinema_camera.py`)**:
  - **Never Static**: Keep camera motion continuous — action-locked projectile/mechanism tracking, shallow-DOF macro close-ups (`f/2.8–4.0` on pure visual macro beats; widen to `f/7.1+` whenever floating 3D labels are onscreen!), fast whip-pans / match cuts, and clear juxtaposition between tight macro detail and wide context shots.
- **Clean Stylized Backdrops, Rim Lighting & Minimal Visual Cues (`*_env.py`, `labels_3d.py`)**:
  - **Two Backdrop Modes**: (a) Clean **Blue/Cyan Studio Blueprint + Grid Floor** for inventions/mechanics, or (b) **Dark Moody Architectural Interior** lit by strong practicals + cool cyan/blue **Rim Light** so hero geometry pops at high contrast.
  - **Clean Minimal 3D Diagrams**: Prefer glowing Fresnel highlights, cutaways, directional 3D arrows, and **ultra-short `1–3 word` callouts** over wordy plaques.
- **Lower-Third Punch Captions (`subtitles/subtitles.ass`)**:
  - Burned-in **1–4 word chunks** synced to narration cadence, bold clean sans-serif white with crisp dark outline (no bulky boxes), centered **~70–75% down** (`NDC Y ≈ -0.45 .. -0.58`) safely above Shorts bottom UI chrome, with selective amber/crimson highlights on high-stakes numbers/words.

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
  - Use a `0.42m` softbox `AREA` light (`52 W` day / `62 W` night, `diffuse_factor = 0.11–0.28` tuned to avoid cheek blowout, `specular_factor = 1.75–1.80`) aimed at eye height (`Z ≈ 1.62m`) via quaternion tracking. Tiny `0.08m` point lights are sub-pixel (`0.11 px`) and get erased by OIDN.
- **Natural Eyelids & Jaw Limits**:
  - Maintain a relaxed `-5.0°` upper-eyelid open baseline (`orbicularis03`); NEVER let smiles squint upper lids shut.
  - Clamp `jaw` rotation to `<= 14.0°` (typical speech `8.5°–9.5°`, shock/spit peak `10.5°`) with smile attenuation (`jaw_calm`) so mouths never gape unnaturally.

---

## 2. Crisp 3D Engineering Labels, Formulas & Typography (`src/build_*_scene.py`)
Never allow white-blown, blurry, washed-out, or hand-overlapped 3D typography:
- **High-Contrast Dark Obsidian Glass Backing Plaques (Mandatory for Sunlit/Daylight Scenes)**:
  - Behind every 3D component callout or countdown timer, add a chamfered dark obsidian rounded-rect plaque (`Base Color = (0.008, 0.012, 0.020)`, `Roughness = 0.45`, low specular `0.10`, `visible_shadow = False`) with sleek glowing cyan/gold trim accent bars and dark-rimmed leader pointer lines (`>15:1` contrast even on bright sunlit sky/desert!).
  - Do NOT add redundant Beat 6 sky title text over clean action finales unless the user explicitly asks for a monument title.
- **2-Zone Brushed-Metallic Face + Dark Extrusion Shader**:
  - Use Object-Space `|N.z|` separation (`> 0.85` = crisp pure-white/satin-gold/cyan face, side bevel/extrusion = dark obsidian-bronze metallic) — do NOT rely on `LayerWeight.Facing` alone (renders as `< 0.3 px` invisible rim).
  - Keep emission strength AgX-safe (`1.35–2.10`, NEVER `5.0+` which clips to white under `AgX - Punchy` and triggers FFmpeg bloom blow-out).
- **Razor-Sharp Bevel, Letter Spacing & Glyph Clarity (`zero fat/blurry text`)**:
  - Use crisp readable glyph scale (`char_h_ndc >= 0.032–0.042`), ultra-subtle bevel (`bevel_depth <= 0.00025m` / `size_ref * 0.0025`, shallow extrusion `0.0018–0.0030m`), and airy tracking (`space_character = 1.08–1.12`) so letters never puff, merge, or turn blurry under denoising/upscaling.
  - Omit bloated thick `Keyline` meshes (`offset <= 0.015` max if needed) when text already sits on a high-contrast dark obsidian backing plaque (`>15:1` contrast).
  - Use `DejaVuSans-Bold.ttf` (bundled in `subtitles/fonts/`) with real math typography (`·`, `₁`, `₂`, `²`, `−`, `×`, `→`, `ε`, `Φ`, `ρ`). Verify every glyph exists in the font before rendering.
- **Exact DOF Focus Plane & Zero Hand / Subtitle Overlap**:
  - Place floating 3D labels strictly on the **exact camera DOF focus plane (`focus_dist ± 0.15m`)** — NEVER push plaques far in front of the focus plane where lens bokeh blurs small text! When placing foreground UI callouts, keep `fstop >= 7.1` so labels and hero remain simultaneously razor-sharp, and clear of hands & subtitles (`NDC Y >= -0.45`).

---

## 3. Volumetric Optics, X-Ray Cutaways, Fireballs & Local-Space Shaders (`src/build_*_scene.py`)
- **100% Local-Space Prop Shaders**:
  - Always drive spatial cutaway scans (`ScanZ`), burn fronts (`BurnZ`), procedural bump, and turbulent plasma off `ShaderNodeTexCoord.Object` bound to the prop's local root Empty (NEVER World `Geometry.Position`) so X-ray effects work flawlessly while props fly/spin across the scene.
- **Procedural 3D Turbulent Fireballs (Zero Flat Concentric Circles)**:
  - Displace icosphere fireballs with 3D turbulence and shade via `(1 - Facing)^p * 3D Noise` radial soft-edge falloff so explosions look organic and cinematic, never like hard flat bullseye rings.
- **Open-Ended Additive Optical Cones/Ribbons**:
  - Always pass `end_fill_type="NOTHING"` (`cap_ends=False`) on volumetric beam/spark/cone meshes so flat end-caps never poke through screens, easels, or lenses.
  - Use `ShaderNodeAddShader` (`Emission` + `Transparent`) driven by `Camera Ray * (1 - Backfacing)` — NEVER high-opacity double-walled `MixShader` which turns beams into opaque pastel walls under AgX.
- **Refractive & X-Ray Glass Shadow Bypass**:
  - Every `Transmission Weight > 0.5` glass prop or X-ray cutaway shell MUST route `LightPath.Is Shadow Ray -> Transparent BSDF` so Cycles (`caustics_refractive = False`) never casts solid black shadows or blocks internal light rays.
- **`bmesh` Material Slot Preservation & Bounce Budget**:
  - Always append materials to `mesh.materials` BEFORE `bm.to_mesh(mesh)` and NEVER call `mesh.materials.clear()` afterwards.
  - Set `cycles.transparent_max_bounces >= 12` whenever layered X-ray shells + additive sparks + hair/corneas overlap.

---

## 4. Zero-Clipping Hand/Prop Choreography, World-Space Handoffs & Framing QA (`src/verify_framing.py`)
- **World-Space Multi-Phase Prop Physics**:
  - Whenever a prop transfers between an object, a character's hand/teeth, and ballistic flight (e.g., grenade pull-pin, thrown weapon, surgical suture), unparent it into World Space and drive exact phase handoffs from evaluated bone/vertex trajectories (`0.000mm` transition error), then cleanly hide detached debris with `CONSTANT` visibility keyframes once off-screen so it never reappears later.
- **Hand-to-Prop Clearance**:
  - Audit every gesturing hand pose and cuff sleeve against nearby 3D props at peak frames so fingers/sleeves never slice through hero props.
- **Automated 6-Beat Sightline & Subtitle-Zone Check**:
  - Maintain `src/verify_framing.py` projecting every hero head, prop, 3D label, and action zone into 9:16 camera NDC (`[-1, 1]`) across all 6 beats, clear of the bottom subtitle zone (`NDC Y < -0.55`).
  - Every commit touching layout or camera MUST pass `python3 src/verify_framing.py` with **100% PASSED**.

---

## 5. Studio Multi-Voice Audio & Subtitle Sync (`audio/generate_audio.py`, `config/timeline.json`)
- **Microsoft Neural Voices (`edge-tts`) — Never Rushed Local Int8 TTS**:
  - Use `edge-tts` studio neural voices (e.g., `en-US-BrianNeural` narrator, `en-GB-ThomasNeural` / `en-GB-RyanNeural` / period-appropriate accents) instead of sped-up `1.44x` Kokoro int8.
- **Zero Word Truncation**:
  - Never hard-slice (`[:max_samples]`) speech arrays to fit a beat. Use natural comma phrasing (avoid mid-line `!`/`.` silence pauses) + adaptive gentle `ffmpeg atempo` (`<= 1.12x`) + generously sized `timeline.json` beat windows.
- **Broadcast Mix & 4K Audio Targets**:
  - Master at **`48000 Hz` stereo**, vocal-pocket EQ cut on BGM (`250–3500 Hz`), `-9.5 dB` sidechain BGM ducking under narration, `-14.0 LUFS` integrated loudness, true peak `<= -1.5 dBFS`, encoded in final MP4 as **`AAC 320 kbps` @ `48 kHz`** (`-b:a 320k -ar 48000`).
  - Always verify generated `master_audio.wav` durations against `voice_manifest.json` and `subtitles/subtitles.ass` karaoke timings.

---

## 6. Default `60 fps` YouTube Shorts → True 4K Unified Modal Cloud Pipeline (`episodes/<name>/`, `.github/workflows/render-episode-4k.yml`)
- **Default Frame Rate = `60 fps` (`1800` frames for `30.0s` Shorts)**:
  - All **NEW YouTube Shorts** MUST default to **`60 fps`** (`FPS = 60`, `TOTAL_FRAMES = 1800` for `30.0s`) — ultra-smooth buttery motion on `60Hz/120Hz` mobile displays AND unlocks YouTube's highest **4K 60fps high-bitrate codec tier** (~50% higher bitrate for razor-sharp explosions, X-ray rotations, and fast action!).
- **Unified Studio Episodes Architecture (Zero Repeated Modal Secret Setup!)**:
  - Place all new shorts inside `ssghtc002-sudo/isaac-newton-gravity-prism-3d/episodes/<episode-name>/` and render via the shared workflow `.github/workflows/render-episode-4k.yml` (`-f episode=<episode-name> -f workers=10 -f samples=16 -f upscale_4k=true`), which automatically reuses the repo's already-configured `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` and publishes directly to GitHub Release tag `<episode-name>-4k` (`https://github.com/ssghtc002-sudo/isaac-newton-gravity-prism-3d/releases/download/<episode-name>-4k/<safe_name>_4k.mp4`).
- **Default Pipeline = Modal 10-GPU Parallel Render (`Blender 5.3.0`, `4K 60fps`)**:
  - Render native **Full HD (`1080×1920`, `scale=100%`, `60 fps`, Blender `5.3.0` Cycles OPTIX + GPU OIDN)** across `10` parallel Modal GPUs (note: use `iter_action_fcurves()` for Blender 5.x Slotted Actions and skip `inputs[0..1]` when indexing `ShaderNodeMix` `VALUE` sockets).
  - Export **True 4K (`2160×3840`, `60 fps`)** Lanczos ARRI-Alexa finished MP4 (`libx264`, `High@L5.2` for full 4K@60fps compliance, `CRF 15`, `yuv420p`, `AAC 320k @ 48 kHz`, `-movflags +faststart` for instant mobile/Reels/Shorts playback).
- **Tightened ARRI-Alexa Bloom & Warm Container Cache**:
  - Use `scale=2160:3840:flags=lanczos` (NEVER invalid `:accurate_rnd=1` option!).
  - Highlight bloom: `eq=brightness=-0.52:contrast=2.8,gblur=sigma=22` blended with `screen:all_opacity=0.20` (1080p: `sigma=12`, `opacity=0.22`).
  - Keep `render_on_modal.py`'s `modal.Image` setup layers byte-for-byte identical to the cached `studio_image` (`/opt/newton_repo`) so Modal boots warm in seconds and pulls latest code via `git fetch --depth 1 origin main && git reset --hard origin/main`.
  - Always test the exact `ffmpeg -filter_complex` string locally on a `0.2s` synthetic clip before pushing.

---

## 7. Strict `1 Agent = 1 Single Focused Task` Crew Rule
- **NEVER assign 2–3 combined roles to a single subagent.**
- Split complex productions into dedicated single-responsibility agents working on non-overlapping files (e.g., `Lead Director`, `Camera & Framing`, `Hero Prop / X-Ray Asset`, `Hero Character Animator`, `Environment / Battlefield`, `3D Typography & Labels`, `Scriptwriter`, `Voice / TTS`, `Foley SFX`, `BGM Composer`, `Finale Cleanup`).
- Define explicit mathematical handoff contracts (exact world coordinates, socket dicts, frame milestones) upfront so all parallel agents integrate cleanly with zero conflicts.

---

## 8. Mandatory Local Frame-by-Frame Cycles Visual QA Before Every 4K Cloud Render
Never push blind to Modal 4K — always run full QA first:
1. `python3 -m py_compile` + AST undefined-name check on all Python files.
2. `python3 src/verify_framing.py` (**100% PASSED** across all 6 beats + gimbal audit).
3. Live `ffmpeg` 0.2s 4K filter-graph validation with bundled `subtitles/subtitles.ass` & `subtitles/fonts/`.
4. Audio LUFS (`-14.0 ± 0.15 LUFS`) / True Peak (`<= -1.5 dBFS`) / duration / zero-truncation audit.
5. **720p HD Full-Quality Cycles Visual Inspection on Modal (`Blender 5.3.0`, `scale=67%` → `>= 720p HD`, `samples=16`, OPTIX + GPU OIDN)**:
   - NEVER rely on low-res `35%–45%` previews for final sign-off — always render QA verification frames at **`scale >= 67%` (`724×1286`, true `720p+` HD)** with **`samples >= 16` full quality** so every 3D text glyph edge, face/eye catchlight, shader detail, and contact seam is clearly caught.
   - Render all 6 representative beat frames **PLUS a dense consecutive frame sequence of Beat 1's viral hook** (to verify acting, prop contact, swag timing, crisp readable text, and zero clipping).
   - Visually inspect **EVERY rendered HD frame with `read_file`**, fix any text blur or visual flaw immediately, re-verify at `720p` full quality, and ONLY THEN push & trigger the 10-GPU Modal 4K render!

---

## 9. Viral Shorts Roadmap Memory (`VIRAL_SHORTS_ROADMAP.md`)
- The next 30 approved episode topics (hooks, X-ray money shots, viral triggers, folder names) live in [`VIRAL_SHORTS_ROADMAP.md`](VIRAL_SHORTS_ROADMAP.md).
- When the user says "next video" / "next topic", pick the first `TODO` row, mark it `🟡 IN PROGRESS` when production starts and `✅ 4K` (with Release tag) once delivered — in the same commit.
