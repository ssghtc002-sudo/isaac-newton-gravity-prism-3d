# Episode #4 — How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Characters:** 3 Photoreal 4K MakeHuman MPFB2 Rigged Crew (`Radar_Operator`, `Defense_Commander`, `Stealth_Pilot` in 1:1 B-2 cockpit)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `<= -1.5 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4)  
**Narration:** `97` words (`en-US-BrianNeural`), synced to lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`, `NDC Y = -0.583`)

---

## 6-Beat Production Breakdown

| Beat | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1 — Bunker Ambush** | `1..220` | `0.00–3.67s` | Inside a sunlit enemy radar bunker, the Air-Defense Commander & Radar Operator watch a massive USAF B-2 Spirit glide past the panoramic glass window — yet their glowing green radar scope registers zero contacts | `0 TARGETS` |
| **2 — Missile Lock vs. 33° Facets** | `221..540` | `3.67–9.00s` | Bright daylight split intercept: a conventional jet triggers a blazing crimson radar return & 3D missile-lock reticle box while the banking B-2's angled facets & serrated edges bounce pulses sideways | `33° DEFLECTION` |
| **3 — Banking Macro X-Ray** | `541..960` | `9.00–16.00s` | Local-space hollow Fresnel X-ray sweeps across the B-2's twin dorsal nacelles, exposing deep serpentine S-shaped air tunnels burying the spinning titanium turbines so incoming RF pulses bounce and die inside | `HIDDEN TURBINES` |
| **4 — RAM Heat Trap & Bumblebee** | `961..1320` | `16.00–22.00s` | Macro hexagonal carbon-ferrite RAM skin swallows leftover RF waves as warm thermal ripples; a `100 m²` holographic RCS sphere collapses down onto a sculpted fluttering `0.001 m²` bumblebee | `HEAT ABSORPTION` / `0.001 m² BLIP` |
| **5A — 1:1 B-2 Pilot Cockpit** | `1321..1430` | `22.00–23.83s` | Shallow-DOF (`f/3.8`) close-up inside the B-2 cockpit: helmeted Stealth Pilot with gleaming gold visor & holographic green HUD calmly banks the flight stick as a SAM radar sweep glints across the canopy | — (pure visual macro) |
| **5B — SAM Radar Showdown** | `1431..1560` | `23.83–26.00s` | Crisp `f/8.0` live-action daylight showdown over the enemy SAM dish: conventional jet blasts a crimson return straight back while the banking B-2 deflects every pulse skyward | `MISSILE BLIND` |
| **6 — Overhead Payoff Twist** | `1561..1800` | `26.00–30.00s` | Back inside the bunker as the panicked Commander & Operator spot one solitary `0.001 m²` bumblebee speck on screen — right as the massive B-2 looms overhead outside and cracks open its twin belly weapons-bay doors | `BUMBLEBEE SIZE` |

---

## Build Audio & Trigger Cloud Render

```bash
# 1. Generate Edge-TTS voice + procedural radar Foley SFX + C#-minor BGM + synced ASS/SRT
python3 episodes/stealth-bomber-radar/audio/generate_audio.py --rebuild-all
rm -f episodes/stealth-bomber-radar/audio/output/voice_stem.wav \
      episodes/stealth-bomber-radar/audio/output/bgm_stem.wav \
      episodes/stealth-bomber-radar/audio/output/sfx_stem.wav

# 2. Optional budget 720p 30fps draft preview (~$0.25-$0.35 on 10 parallel Modal GPUs;
#    render_on_modal.py automatically rebuilds audio & subtitles if timeline.json changed!)
modal run episodes/stealth-bomber-radar/render_on_modal.py --draft-preview

# 3. Render True 4K 60fps on 10 parallel Modal GPUs & publish GitHub Release
gh workflow run render-episode-4k.yml -f episode=stealth-bomber-radar -f workers=10 -f samples=16 -f upscale_4k=true
```
