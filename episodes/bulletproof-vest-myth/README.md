# Episode #3 — Why Bulletproof Vests DON'T Send You Flying Backward

**Format:** `9:16` Vertical (`1080×1920` native OPTIX → **`2160×3840` True 4K**) @ **`60 fps`** (`1800 frames`, `30.0 s`)  
**Pipeline:** Blender **`5.3.0`** Cycles OPTIX + GPU OIDN (`AGENTS.md` V12 + `@zackdfilms` craft blueprint)  
**Audio Master:** `48000 Hz` Stereo, `-14.0 LUFS` integrated, `<= -1.5 dBFS` true peak (`AAC 320 kbps @ 48 kHz` in final MP4)  
**Narration:** `95` words (`en-US-BrianNeural`), synced to lower-third `\kf` karaoke cards (`subtitles/subtitles.ass`, `NDC Y = -0.583`)

---

## 6-Beat Production Breakdown

| Beat | Frames (`60 fps`) | Time (`s`) | Visual & Physics Action | 3D Callout (`NDC Y >= -0.45`) |
| :--- | :--- | :--- | :--- | :--- |
| **1 — Hook** | `1..210` | `0.00–3.50s` | Slow-mo dead-center chest hit inside moody ballistic range; Hero shifts barely `0.5 in` (`12.7 mm`) as sparks burst off plate while glass wall behind stays intact | `0.5 INCH SHIFT` |
| **2 — Momentum** | `211..510` | `3.50–8.50s` | Wide range profile showing Shooter firing at Hero; bullet carries less linear momentum than a fastball | `TINY MOMENTUM` |
| **3 — Kevlar X-Ray** | `511..900` | `8.50–15.00s` | Macro vest cutaway: X-ray carrier peels back to reveal woven golden Kevlar rippling outward like a trampoline catching the spinning 9mm round | `KEVLAR NET` |
| **4 — Ceramic + Dyneema** | `901..1260` | `15.00–21.00s` | High-velocity rifle penetrator strikes Silicon-Carbide (`SiC`) ceramic plate, shattering into 16 fragments while UHMWPE Dyneema backer traps every shard | `CERAMIC + DYNEEMA` |
| **5 — Newton's 3rd Law** | `1261..1560` | `21.00–26.00s` | Clean `@zackdfilms` Studio Blueprint side-by-side: equal-and-opposite `F₁ = −F₂` recoil vectors prove any shot launching the target must launch the shooter too | `F₁ = −F₂` / `EQUAL RECOIL` |
| **6 — Reality Payoff** | `1561..1800` | `26.00–30.00s` | Hero stands rock-steady beside intact tempered glass as spent ceramic dust drifts down | `JUST A BRUISE` |

---

## Build Audio & Trigger Cloud 4K Render

```bash
# 1. Generate Edge-TTS voice + real CC0/PD gunshots SFX + D-minor BGM + synced ASS/SRT
python3 episodes/bulletproof-vest-myth/audio/generate_audio.py --rebuild-all
rm -f episodes/bulletproof-vest-myth/audio/output/voice_stem.wav \
      episodes/bulletproof-vest-myth/audio/output/bgm_stem.wav \
      episodes/bulletproof-vest-myth/audio/output/sfx_stem.wav

# 2. Render True 4K 60fps on 10 parallel Modal GPUs & publish GitHub Release
gh workflow run render-episode-4k.yml -f episode=bulletproof-vest-myth -f workers=10 -f samples=16 -f upscale_4k=true
```
