# Episode #2 — Why Bullets Stop Dead Just 3 Feet Underwater

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `-1.6 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4)  
**Narration:** `101` words (`en-US-BrianNeural`), synced to `32` lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`)

---

## 6-Beat Production Breakdown

| Beat | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1 — Hook** | `1..210` | `0.00–3.50s` | Submerged shooter/swimmer behind glowing `3 FT` datum bar; supersonic spitzer bullet screeches dead at `0.914 m` | `3 FT STOP` |
| **2 — Entry Slam** | `211..540` | `3.50–9.00s` | Gunmetal rifle barrel muzzle flash bursts into water wall (`800×` denser), launching aerodynamic bow streamlines | `800× DENSER` |
| **3 — Supercavitation** | `541..900` | `9.00–15.00s` | Action-locked `35mm` macro alongside copper ogive inside glowing freshwater vapor cavitation cone + spiraling wake | `VAPOR BUBBLE` |
| **4 — Hydro Shatter** | `901..1260` | `15.00–21.00s` | Copper jacket fades into X-ray cutaway revealing spinning fluted lead core shifting amber → crimson under radial stress, exploding into 8 fragments at `3 FT` | `SHATTERED` |
| **5 — Physics Race** | `1261..1560` | `21.00–26.00s` | Clean `@zackdfilms` Studio Blueprint comparison: top-lane supersonic round shatters violently at `3 FT` while slower heavy subsonic slug glides smoothly to `6 FT` | `FASTER: 3 FT` / `SLOWER: 6 FT` |
| **6 — Safe Finale** | `1561..1800` | `26.00–30.00s` | Peaceful submerged swimmer hovering safely below & past `3 FT` as crushed copper flakes drift downward | `SAFE BELOW 3 FT` |

---

## Cloud 4K Render Command

```bash
gh workflow run render-episode-4k.yml -f episode=underwater-bullets -f workers=10 -f samples=16 -f upscale_4k=true
```
