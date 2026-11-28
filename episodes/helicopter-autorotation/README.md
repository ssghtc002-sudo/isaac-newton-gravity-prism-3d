# Episode #5 — Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Characters:** 2 Photoreal 4K MakeHuman MPFB2 Rigged Rescue Aviators (`Heli_Pilot` + `Heli_CoPilot` in 1:1 rescue cockpit)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `<= -1.5 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4)  
**Narration:** `94` words (`en-US-BrianNeural`), synced to lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`, `NDC Y = -0.583`)

---

## 6-Beat Production Breakdown

| Beat | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1 — Engine Failure Hook** | `1..240` | `0.00–4.00s` | Inside the 1:1 rescue cockpit over a golden-hour city skyline, `ENG OUT` crimson warnings flash as the Pilot slams the collective pitch lever straight down while the Co-Pilot tracks the rooftop helipad | `ENGINE DEAD` |
| **2 — Freewheel Sprag Clutch** | `241..570` | `4.00–9.50s` | Local-space hollow Fresnel X-ray peels back the white-and-red rescue helicopter's dorsal cowling to show the dead crimson turbine shaft seizing while golden steel sprag clutch rollers snap open | `SPRAG CLUTCH` |
| **3 — Upward Airflow Windmill** | `571..960` | `9.50–16.00s` | Descending over sunlit glass towers, glowing emerald-cyan air streamlines surge UP through the negatively pitched rotor disk, spinning the blades like a wind turbine to store flywheel energy | `UPWARD AIRFLOW` |
| **4 — 3D Maple Seed Comparison** | `961..1290` | `16.00–21.50s` | A sculpted spinning 3D winged maple seed glides side-by-side next to the descending rescue rotor in matching golden-hour helical emerald vortices | `MAPLE SEED LIFT` |
| **5A — 1:1 Cockpit Flare Pull** | `1291..1450` | `21.50–24.17s` | Shallow-DOF (`f/3.6`) cockpit macro: rooftop helipad rushes upward outside the bubble canopy as the Pilot pulls the collective lever UP and cyclic back while the Co-Pilot calls altitude | *(pure visual macro)* |
| **5B — 30-Foot Exterior Flare** | `1451..1580` | `24.17–26.33s` | Crisp `f/8.0` low-angle exterior beside the skyscraper helipad: rescue chopper pitches nose-up (`+21°`), blasting a golden-cyan downward cushion wash ring | `30 FT FLARE` |
| **6 — Rooftop Touchdown** | `1581..1800` | `26.33–30.00s` | Rescue helicopter levels smoothly onto the illuminated yellow `H` rooftop helipad with flush skid contact as the unpowered blades coast down in sunset light | `ZERO ENGINE` |

---

## Build Audio & Trigger Cloud Render

```bash
# 1. Generate Edge-TTS voice + procedural rotor/clutch Foley SFX + tension-pulse BGM + synced ASS/SRT
python3 episodes/helicopter-autorotation/audio/generate_audio.py --rebuild-all
rm -f episodes/helicopter-autorotation/audio/output/voice_stem.wav \
      episodes/helicopter-autorotation/audio/output/bgm_stem.wav \
      episodes/helicopter-autorotation/audio/output/sfx_stem.wav

# 2. Optional budget 720p 30fps draft preview (~$0.25-$0.35 on 10 parallel Modal GPUs;
#    render_on_modal.py automatically builds audio & subtitles if missing or updated!)
modal run episodes/helicopter-autorotation/render_on_modal.py --draft-preview

# 3. Render True 4K 60fps on 10 parallel Modal GPUs & publish GitHub Release
gh workflow run render-episode-4k.yml -f episode=helicopter-autorotation -f workers=10 -f samples=16 -f upscale_4k=true
```
