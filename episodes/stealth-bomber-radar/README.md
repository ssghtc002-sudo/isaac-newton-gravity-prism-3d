# Episode #4 — How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `<= -1.5 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4)  
**Narration:** `96` words (`en-US-BrianNeural`), synced to lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`, `NDC Y = -0.583`)

---

## 6-Beat Production Breakdown

| Beat | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1 — Hook** | `1..220` | `0.00–3.67s` | Massive 172-ft black B-2 flying wing glides through moonlit clouds above an enemy radar station; glowing cyan RF wavefronts deflect off-axis into empty sky | `172 FT WING` |
| **2 — Facets** | `221..540` | `3.67–9.00s` | Split-elevation comparison: conventional jet blazes a huge crimson radar return straight back to the dish while stealth origami facets & sawtooth trailing edges scatter pulses sideways | `FACETED DEFLECTION` |
| **3 — S-Ducts X-Ray** | `541..960` | `9.00–16.00s` | Local-space Fresnel X-ray sweep peels back carbon skin to reveal serpentine S-shaped air tunnels burying the spinning metallic turbofan blades as an RF wave bounces and dies inside | `S-SHAPED DUCTS` |
| **4 — RAM & Bumblebee** | `961..1320` | `16.00–22.00s` | Macro hexagonal Radar-Absorbent Material (RAM) skin swallows RF energy into warm heat ripples; a `100 m²` holographic RCS sphere collapses down onto a sculpted fluttering `0.001 m²` bumblebee | `RAM COATING` / `0.001 m² BLIP` |
| **5 — CAD Blueprint** | `1321..1560` | `22.00–26.00s` | Clean `@zackdfilms` cyan-blue CAD Blueprint studio contrasting a 90° mirror echoing beams directly back against a 33° tilted stealth facet scattering pulses skyward | `TILTED FACETS` |
| **6 — Moonlight Payoff** | `1561..1800` | `26.00–30.00s` | Bomber banks silently overhead in silver moonlight rim-light while a sweeping tactical phosphor radar scope registers only one solitary tiny `0.001 m²` bumblebee speck | `BUMBLEBEE SIZE` |

---

## Build Audio & Trigger Cloud Render

```bash
# 1. Generate Edge-TTS voice + procedural radar Foley SFX + C#-minor BGM + synced ASS/SRT
python3 episodes/stealth-bomber-radar/audio/generate_audio.py --rebuild-all
rm -f episodes/stealth-bomber-radar/audio/output/voice_stem.wav \
      episodes/stealth-bomber-radar/audio/output/bgm_stem.wav \
      episodes/stealth-bomber-radar/audio/output/sfx_stem.wav

# 2. Optional budget 720p 30fps draft preview (~$0.25-$0.35 on 10 parallel Modal GPUs)
modal run episodes/stealth-bomber-radar/render_on_modal.py --draft-preview

# 3. Render True 4K 60fps on 10 parallel Modal GPUs & publish GitHub Release
gh workflow run render-episode-4k.yml -f episode=stealth-bomber-radar -f workers=10 -f samples=16 -f upscale_4k=true
```
