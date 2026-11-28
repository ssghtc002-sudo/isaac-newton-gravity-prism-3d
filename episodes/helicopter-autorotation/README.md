# Episode #5 — Why a Helicopter With a Dead Engine DOESN'T Fall Like a Rock (V2 Ultimate)

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Hero Asset:** Modern Twin-Engine Rescue Helicopter (Airbus H145 / Coast Guard Dauphin silhouette, 4-blade coning rotor + motion-blur disc + Fenestron tail)  
**Characters:** 2 Photoreal 4K MakeHuman MPFB2 Rigged Rescue Aviators in dark-navy Nomex flight suits (`Heli_Pilot` + `Heli_CoPilot` in dark enclosed 1:1 cockpit with live crashing 3D `ENG` / `ROTOR` RPM gauges)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `<= -1.5 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4) — Hollywood 112 BPM suspense score (`0.35s` pre-flare vacuum drop + Dmaj9 landing) & layered aviation Foley  
**Narration:** `94` words (`en-US-BrianNeural`), synced to lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`, `NDC Y = -0.583`)

---

## 8-Segment Production Breakdown

| Segment | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1A — Exterior Engine Explosion** | `1..115` | `0.00–1.92s` | Modern twin-engine rescue chopper banking over sunlit curtain-wall glass skyline erupts with a dorsal turbine compressor-stall fire burst, curved metallic sparks & dark smoke plume | `ENGINE DEAD` |
| **1B — Enclosed Panic Cockpit** | `116..240` | `1.92–4.00s` | Hard cut inside dark enclosed cockpit: flashing crimson `ENG OUT` bar, live 3D `ENG` needle crashing to `0%`, wide-eyed Pilot slams the collective lever straight down while Co-Pilot spots the helipad | *(pure cockpit drama)* |
| **2 — 12-Sprag Clutch X-Ray** | `241..570` | `4.00–9.50s` | Local-space hollow Fresnel X-ray peels back dorsal cowling revealing crimson-hot seized turbine gear + stall halo while 12 chamfered golden cam sprags tilt open & freewheel arcs spin free | `SPRAG CLUTCH` |
| **3 — Upward Airflow Windmill** | `571..960` | `9.50–16.00s` | Descending past rising golden-hour cloud strata, 6 wide sculpted translucent emerald 3D ribbon arrows surge UP through the spinning blades to charge kinetic flywheel energy | `UPWARD AIRFLOW` |
| **4 — Botanical Maple Seed Lift** | `961..1290` | `16.00–21.50s` | Enlarged `2.1×` sculpted botanical 3D winged maple samara glides side-by-side next to the modern rescue rotor inside clean matched lift swirls | `MAPLE SEED LIFT` |
| **5A — Enclosed Cockpit Flare** | `1291..1450` | `21.50–24.17s` | Shallow-DOF (`f/3.6`) dark cockpit close-up: radar altimeter bar plunges `125 ft → 32 ft` as focused Pilot hauls collective UP & cyclic back | *(pure visual macro)* |
| **5B — 30-Foot Exterior Flare** | `1451..1580` | `24.17–26.33s` | Post-vacuum (`0.35s` drop) crisp `f/8.0` exterior beside weathered skyscraper helipad: chopper pitches nose-up (`+22°`, `+4.2°` blade coning) blasting an air-cushion shockwave | `30 FT FLARE` |
| **6 — Rooftop Helipad Landing** | `1581..1800` | `26.33–30.00s` | Chopper levels smoothly to exact `0.000 mm` flush skid contact on the weathered rooftop helipad (safety netting, green/white strobes, whipping windsock) kicking up an expanding golden dust wave | `ZERO ENGINE` |

---

## Build Audio & Trigger Cloud Render

```bash
# 1. Generate Edge-TTS voice + V2 blockbuster aviation Foley + Hollywood score + synced ASS/SRT
python3 episodes/helicopter-autorotation/audio/generate_audio.py --rebuild-all
rm -f episodes/helicopter-autorotation/audio/output/voice_stem.wav \
      episodes/helicopter-autorotation/audio/output/bgm_stem.wav \
      episodes/helicopter-autorotation/audio/output/sfx_stem.wav

# 2. Optional budget 720p 30fps draft preview (~$0.25-$0.35 on 10 parallel Modal GPUs;
#    render_on_modal.py automatically rebuilds audio & subtitles when timeline.json changes!)
modal run episodes/helicopter-autorotation/render_on_modal.py --draft-preview

# 3. Render True 4K 60fps on 10 parallel Modal GPUs & publish GitHub Release
gh workflow run render-episode-4k.yml -f episode=helicopter-autorotation -f workers=10 -f samples=16 -f upscale_4k=true
```
