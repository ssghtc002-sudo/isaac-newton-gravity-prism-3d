# Scientist #9: Isaac Newton — V8 Photoreal Outdoor Woolsthorpe Orchard-to-Cosmos (Pure Kaggle `GPU T4 x2` OPTIX Turbo)

Complete **30-second (`1080×1920` Lanczos, `24 fps`, `720` frames) Blender 4.2.3 LTS Cycles V8 Photoreal Cinema Short** purpose-built for **Kaggle `GPU T4 x2`** — featuring **zero Play-Doh blobs (51 real woody branching twigs + 2,600 individual matte backlit leaves)**, **Amber-to-Cobalt Golden-Hour Nishita Sunset transforming into Starry Cosmos**, **real Cycles `ShaderNodeAmbientOcclusion` stone & bark**, **100% real 3D modeled MakeHuman 17th-century garments (`0.97s` Fastly CDN `assets-v8`, ZERO procedural foam straps, ZERO modern neckties!)**, **Zero-Quality-Drop Cycles `OPTIX` Turbo (`~6.5–8 min` on Dual-T4 + 3s FFmpeg optical highlight bloom)**, **3-voice British Kokoro narration + 112 BPM Cambridge Celestial Baroque/Orchestral Score + 26 spatial Foley SFX**, and **bundled-font word-synced boxless karaoke subtitles**.

## V8 Photorealism & Authentic Wardrobe Upgrades
- **100% Real 3D MakeHuman 17th-Century Wardrobe (`assets-v8` CDN — `41.57 MB` in `0.97s` @ `42.9 MB/s`)**:
  - **Sir Isaac Newton (`1.78m`, `age=0.58`)**: 4K `middleage_caucasian_male.mhmat`, `culturalibre_hair_01.mhclo` (1689 Kneller silver-pewter parted shoulder-length locks), **`donitz_monk_robe.mhclo`** (10,233-vert male-sculpted full-length Cambridge Lucasian Scholar Gown with real draped shoulder mantle, deep pleats, 2K OpenGL normal map & braided waist cord — graded in Royal Midnight-Navy Velvet w/ silver-pewter cord!), and `toigo_ankle_boots_male.mhclo` leather boots.
  - **Edmond Halley (`1.79m`, `age=0.60`, eye-to-eye equal adult height!)**: 4K `middleage_caucasian_male.mhmat`, `short04.mhclo` espresso-chestnut gentleman hair, **`mindfront_lusekofta.mhclo`** (tailored Baroque brocade gentleman's coat with 6 sculpted metallic filigree buttons auto-wired to `F_Lusekofta_METAL.png` Metallic mask + 2K normal/spec maps — graded in Warm Royal Burgundy / Espresso-Mahogany!), `mindfront_male_trousers_2.mhclo` wool trousers, and `punkduck_medieval_boots.mhclo` cuffed medieval leather boots.
- **Zero Cartoon Blobs — True Cinema Outdoor Foliage & Lighting (`src/newton_world_v7.py`)**:
  - **51 branching woody twigs + 2,600 individual matte sculpted leaves** (`Roughness=0.56, Coat=0.04, BURLEY SSS`, Beat 6 starry sky title 100% unobstructed!).
  - **Amber-to-Cobalt Golden-Hour Nishita Sunset (`sun_elevation=5.8°, ozone=3.8, Strength=0.24`) → Deep Starry Cosmic Twilight** with 160 true random 3D stars & sculpted 3D Crescent Moon.
  - **Cycles `ShaderNodeAmbientOcclusion` (`samples=8`)** on weathered Cotswold flagstones (dark earthy recessed mortar, subtle corner moss), carved plinths & balustrade, and deep fissured apple bark (`Bump 0.62`), lit by golden low-angle chiaroscuro sun + cool sapphire rim!
- **Natural Relaxed Gestures & Daylight-Proof 3D Optics (`src/build_newton_scene.py`)**:
  - Natural asymmetric hip-level resting arms across all 6 beats, daylight-proof Beat 2 glowing formulas (`F = m a` → `F = G m1 m2 / r²`) with soft shadow halos, jewel-saturated 7-spectral flint-glass prism rainbow, high-contrast Blue-Marble Earth & Moon Gravity Well, and sleek 3D Orbital Rocket finale!
- **Zero-Quality-Drop Cycles `OPTIX` Turbo (`~6.5–8 min` on Kaggle Dual-T4)**:
  - Hardware **`OPTIX` Tensor-Core Denoiser (`RGB_ALBEDO_NORMAL`)**, full `75% → 1080×1920` Lanczos, full `2048px` textures, `12 spp` adaptive, zero CPU compositor stall (true cinema optical bloom runs in Cell 5 FFmpeg in `~3s`!), `/kaggle/working/.cache` SSD (`0 MB /tmp` RAM!), `10s` staggered Dual-T4 launch, and direct `/kaggle/working/isaac_newton_gravity_prism_short.mp4` + `FileLink`.
- **Asset Attribution**: Official MakeHuman Community Asset Packs (`makehuman_system_assets` CC0, `hair01` CC0 by culturalibre, `suits02` CC0 by donitz, `shirts02` CC-BY 4.0 by mindfront & elvs, `pants02` CC-BY 4.0 by mindfront, `shoes01` CC0 by Margaret Toigo, `shoes03` CC-BY 4.0 by punkduck). Full credits in `assets-v8/mpfb_newton_v8_4k_assets.zip` (`ATTRIBUTIONS.txt`).

## 1-Click Kaggle `GPU T4 x2` Workflow
1. Go to **Kaggle -> New Notebook -> File -> Import Notebook -> GitHub**:
   - Repo: `ssghtc002-sudo/isaac-newton-gravity-prism-3d`
   - Branch: `main`
   - File: `render_on_kaggle.ipynb`
2. In **Session options**: **Accelerator: `GPU T4 x2`** & **Internet: `ON`**.
3. Run **Cell 1 -> Cell 2 (~18s, `0.97s` CDN cast!) -> Cell 3 (~12s 6-Beat V8 Preview Grid)**, then **Cell 4 (~6.5–8 min!) -> Cell 5** -> tap **`FileLink('isaac_newton_gravity_prism_short.mp4')`**!
