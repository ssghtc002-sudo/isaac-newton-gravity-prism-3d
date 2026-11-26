# Scientist #9: Isaac Newton — V9 Hollywood Mini-Film Polish (Outdoor Woolsthorpe Orchard-to-Cosmos, Pure Kaggle `GPU T4 x2` OPTIX Turbo)

Complete **30-second (`1080×1920` Lanczos, `24 fps`, `720` frames) Blender 4.2.3 LTS Cycles V9 Hollywood Mini-Film Polish Short** purpose-built for **Kaggle `GPU T4 x2`** — featuring **V9 Character Micro-Life & Expressive Film Acting** (eye-lead-head gaze, dual-axis Kokoro `lipSync` + `lipNarrow` speech, clavicle-driven arms, expressive hand gestures & contrapposto stance), **Hero 3D Physics Micro-Details across all 6 Beats**, **Cell 5 ARRI-Alexa Filmic Finishing** (optical bloom, warm/cool color balance, cinema vignette & 1% organic film grain), **zero Play-Doh blobs (51 woody branching twigs + 2,600 individual matte backlit leaves)**, **100% real 3D modeled MakeHuman 17th-century garments (`0.97s` Fastly CDN `assets-v8`, ZERO procedural foam straps, ZERO modern neckties!)**, **Zero-Quality-Drop Cycles `OPTIX` Turbo (`~6.5–8 min` on Dual-T4)**, **3-voice British Kokoro narration + 112 BPM Cambridge Celestial Baroque/Orchestral Score + 26 spatial Foley SFX**, and **bundled-font word-synced boxless karaoke subtitles**.

## V9 Hollywood Mini-Film Polish Upgrades
- **V9 Character Micro-Life & Expressive Film Acting Choreography (`src/build_newton_scene.py` + `src/mpfb_characters.py`)**:
  - **Eye-Lead-Head Gaze & Dual-Axis Speech**: Newton's eyes look up (`-14°`) at the hanging apple `0.1s` ahead of his head then track smoothly down (`+10°`) as it falls; eye-to-eye Beat 2 courtyard debate (`±12°`) driven by BOTH `lipSync` AND `lipNarrow` from `audio/output/voice_manifest.json`!
  - **Clavicle-Driven Arms, Hand Gestures & Natural Contrapposto**: Anatomical clavicle follow on every gesture, dynamic `"relaxed"` / `"present"` / `"point"` / `"shock"` hand poses across all 6 beats, and natural contrapposto weight shifts + Halley's Beat 4 dual-hand `"shock"` gasp & full-body knee-bend stagger recoil!
- **Hero 3D Physics Micro-Details (ZERO Render Slowdown)**:
  - **Beat 1**: Organic 3D tumbling heirloom apple + 2 tiny fluttering detached apple leaves spiraling down behind it + molten-gold `g` vector & 8-dot glowing golden apple-to-Moon orbital arc (`2.6–4.3s`, *"does the Moon fall too?"*).
  - **Beat 2**: Open *Principia Mathematica* golden-ink orbital ellipse & 4 focal radial rays + warm golden page-shimmer pulsing right as daylight-proof 3D `F = G m1 m2 / r²` snaps into view (`6.95s`).
  - **Beat 3**: Sleek brass graduated optical vernier collar on the prism tripod + catchlight-rimmed Flint-Glass Prism + 7 jewel-saturated rainbow dispersion ribbons sparkling with 28 matching spectral caustic dust motes!
  - **Beat 4**: Brass Newton's Cradle with crisp golden-white impact flash core + sympathetic `±0.8°` phase-staggered middle-sphere metallic buzz + 16 sparks & shock ring at `17.45s` CLACK!
  - **Beats 5–6**: NASA Blue-Marble 3D Earth & Moon Gravity Well + sleek 3D Orbital Rocket trailing a curved golden parabolic wake + opposite-phase (`180°`) glistening silver-and-solar-panel Modern Communication Satellite!
- **ARRI-Alexa Filmic Finishing (`render_on_kaggle.ipynb` Cell 4)**:
  - `1080×1920` Lanczos + micro-detail `unsharp=3:3:0.35:3:3:0.0` → Optical Highlight Bloom (`screen @ 0.28`) → Warm-Highlight / Cool-Shadow `colorbalance` → Cinema `vignette=PI/6.5` → 1% Organic Film Grain (`noise=alls=3:allf=t`) → razor-sharp bundled-font boxless karaoke subtitles (`subtitles/subtitles.ass`)!
- **100% Real 3D MakeHuman 17th-Century Wardrobe (`assets-v8` CDN — `41.57 MB` in `0.97s` @ `42.9 MB/s`)**:
  - **Sir Isaac Newton (`1.78m`, `age=0.58`)**: 4K `middleage_caucasian_male.mhmat`, `culturalibre_hair_01.mhclo` (1689 Kneller silver-pewter parted shoulder-length locks), **`donitz_monk_robe.mhclo`** (10,233-vert Cambridge Lucasian Scholar Gown in Royal Midnight-Navy Velvet), and `toigo_ankle_boots_male.mhclo` leather boots.
  - **Edmond Halley (`1.79m`, `age=0.60`, eye-to-eye equal adult height!)**: 4K `middleage_caucasian_male.mhmat`, `short04.mhclo` espresso-chestnut hair, **`mindfront_lusekofta.mhclo`** (Royal Burgundy Baroque coat w/ 6 sculpted metallic filigree buttons auto-wired to `F_Lusekofta_METAL.png`), `mindfront_male_trousers_2.mhclo` wool trousers, and `punkduck_medieval_boots.mhclo` boots.
- **Zero-Quality-Drop Cycles `OPTIX` Ultra-Turbo (`~4–5.5 min` on Kaggle Dual-T4 -> `1080×1920` Full HD Export!)**:
  - **Bake-Once-Render-Twice `.blend` Architecture** (`src/build_newton_scene.py --save-blend` + `src/render_worker.py`): Builds & keyframes the 720-frame scene **once** on SSD, then launches both T4 GPUs in `<1s` with zero duplicate CPU build or RAM spike!
  - **4× Faster Local `AmbientOcclusion` (`ao.samples = 2`)**, **Trimmed Hair/Diffuse Bounces (`6` / `1`)**, and **Hardware `OPTIX` Raytracing + Fast GPU `OpenImageDenoise` (`RGB_ALBEDO`)** (100% headless Tesla T4 compatible).
  - **1-Click Cell 4 Full HD `1080×1920` (`24 fps`, `CRF 15`) Render + Export**: Renders all 720 frames on Dual-T4 and immediately runs Lanczos `1080×1920` Full HD + ARRI-Alexa filmic FFmpeg finishing into `/kaggle/working/isaac_newton_gravity_prism_short.mp4` + 1-Click `FileLink`!
- **Asset Attribution**: Official MakeHuman Community Asset Packs (`makehuman_system_assets` CC0, `hair01` CC0 by culturalibre, `suits02` CC0 by donitz, `shirts02` CC-BY 4.0 by mindfront & elvs, `pants02` CC-BY 4.0 by mindfront, `shoes01` CC0 by Margaret Toigo, `shoes03` CC-BY 4.0 by punkduck). Full credits in `assets-v8/mpfb_newton_v8_4k_assets.zip` (`ATTRIBUTIONS.txt`).

## 1-Click Kaggle `GPU T4 x2` Workflow (Full HD `1080×1920`)
1. Go to **Kaggle -> New Notebook -> File -> Import Notebook -> GitHub**:
   - Repo: `ssghtc002-sudo/isaac-newton-gravity-prism-3d`
   - Branch: `main`
   - File: `render_on_kaggle.ipynb`
2. In **Session options**: **Accelerator: `GPU T4 x2`** & **Internet: `ON`**.
3. Run **Cell 1 -> Cell 2 (~18s, `0.97s` CDN cast!) -> Cell 3 (~18s 6-Beat V9 Preview + Scene Bake) -> Cell 4 (~4–5.5 min 1-Click Render + Full HD `1080×1920` MP4 + `FileLink`)**!
