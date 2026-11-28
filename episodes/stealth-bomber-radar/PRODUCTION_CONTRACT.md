# PRODUCTION CONTRACT — Episode #4: "How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar"

Single source of truth for all 3D modules, camera framing, shaders, Foley SFX, narration, subtitles, and Modal 4K / `--draft-preview` cloud rendering. Follows **AGENTS.md V12** (`60 fps`, `1800 frames`, `30.0 s`, `1080×1920` native → `2160×3840` True 4K, Blender `5.3.0` Cycles OPTIX + GPU OIDN, `AgX - Punchy`).

---

## 1. 6-Beat Production Architecture (`1..1800` @ `60 fps` = `30.0 s`)

| Beat | Frames | Time | Visual Story & Camera | 3D Callout (`<= 3 words`) |
|---|---|---|---|---|
| `B1_HOOK` | `1..220` | `0.00–3.67 s` | Night sky over ground radar station; massive 172-ft black flying-wing bomber glides overhead as cyan radar pulses ricochet off-axis into empty sky | `172 FT WING` (`70..212`) |
| `B2_FACETS` | `221..540` | `3.67–9.00 s` | Vertical split contrast: conventional jet (top) blasts huge crimson radar echo back vs. stealth wing's sawtooth origami facets deflecting waves sideways | `FACETED DEFLECTION` (`260..528`) |
| `B3_SDUCTS` | `541..960` | `9.00–16.00 s` | Macro X-ray sweep (local `TexCoord.Object` + hollow Fresnel `AddShader`) exposing deep S-shaped serpentine air ducts hiding spinning titanium turbines | `S-SHAPED DUCTS` (`615..945`) |
| `B4_RAM_BEE` | `961..1320` | `16.00–22.00 s` | Macro hexagonal carbon-ferrite RAM skin absorbing RF waves into harmless warm heat + holographic RCS orb shrinking from `100 m²` down to a `0.001 m²` 3D bumblebee | `RAM COATING` (`990..1145`) → `0.001 m² BLIP` (`1155..1308`) |
| `B5_BLUEPRINT` | `1321..1560` | `22.00–26.00 s` | Clean `@zackdfilms` Cyan-Blue CAD Blueprint Grid (`Z = -0.15..8.80 m`): 90° corner reflector (`Echo Back`) vs. tilted stealth facets (`Scattered Skyward`) | `TILTED FACETS` (`1345..1545`) |
| `B6_FINAL` | `1561..1800` | `26.00–30.00 s` | Bomber banks overhead in cool moonlight rim-light while a sweeping 3D phosphor tactical radar scope reveals one tiny solitary bumblebee speck | `BUMBLEBEE SIZE` (`1595..1785`) |

Hard cut boundaries (camera `CONSTANT` interpolation): `221`, `541`, `961`, `1321`, `1561`.

---

## 2. Exact Spatial Coordinate System (1:15 Tactical Diorama Scale)

- **Origin & World Frame:** `+X` right (flight direction / wing lateral), `+Y` away from camera, `+Z` up.
- **Stealth Flying-Wing Bomber (`Stealth_Bomber_Root`):**
  - Scaled `1:15` (`172 ft = 52.4 m` → `3.60 m` wingspan, `1.48 m` nose-to-sawtooth-tail chord).
  - Beat 1 glide (`1..220`): sweeps from `(-1.25, +0.25, 2.25)` to `(+0.55, -0.05, 2.05)` with `24°` sleek banking roll showing the jagged flying-wing belly & leading edges.
  - Beat 2 contrast (`221..540`): **Conventional Jet** at `(0.00, 0.20, 2.75)` (upper frame, huge crimson echo back to radar) vs. **Stealth Bomber** at `(0.00, 0.00, 1.35)` (lower-mid frame, clean cyan deflection).
  - Beat 3 macro X-ray (`541..960`): Bomber stabilized at `(0.00, 0.00, 1.85)` (`pitch = -8°`, `roll = +14°`, `yaw = -24°`) so dorsal intake scoop & internal S-shaped ducts (`X ∈ [-0.52, -0.10]`, `Z ∈ [1.72, 1.98]`) fill center-screen cleanly.
  - Beat 4 RAM + Bumblebee (`961..1320`): Macro hexagonal RAM panel at `(0.00, 0.00, 1.85)` (`0.88 × 0.56 m`) transitioning to holographic RCS shrink orb & sculpted 3D Bumblebee at `(+0.22, -0.06, 1.88)`.
  - Beat 5 CAD Blueprint (`1321..1560`): Dual stacked optical comparison rigs at `Y = 0.00` — Top 90° dihedral reflector at `Z = 2.42 m` (crimson return), Bottom tilted stealth facet at `Z = 1.18 m` (cyan skyward scatter), against Blueprint Wall (`Y = +2.65 m`, `X = -7.0..+7.0 m`, `Z = -0.15..8.80 m`) & Blueprint Grid Floor (`X = -7.0..+7.0 m`, `Y = -5.8..+5.8 m`, `Z = 0.00 m`).
  - Beat 6 Finale (`1561..1800`): Bomber banks smoothly across upper sky `(-0.70, +0.55, 2.75)` → `(+0.65, +0.35, 2.55)` while the 3D Tactical Radar Scope sits cleanly in lower-mid foreground `(0.00, -0.45, 1.12)` tilted toward camera showing a solitary `0.001 m²` bumblebee speck.

---

## 3. Exact Frame Milestones (`EV` synced 1:1 with `config/timeline.json`)

| Event Key | Frame | Time (s) | Action / Foley Trigger |
|---|---:|---:|---|
| `b1_radar_ping` | `18` | `0.30` | Ground radar dish emits first cyan pulse cone |
| `b1_wing_deflect` | `68` | `1.13` | Pulse hits bomber leading edge & ricochets sideways |
| `b1_blank_scope` | `135` | `2.25` | Second radar sweep returns dead blank |
| `b2_conv_ping` | `265` | `4.42` | Radar hits conventional jet fuselage & vertical tail |
| `b2_conv_echo` | `318` | `5.30` | Blazing crimson radar return locks onto conventional jet |
| `b2_stealth_deflect` | `410` | `6.83` | Stealth sawtooth facets deflect cyan waves off-axis |
| `b3_xray_sweep` | `575` | `9.58` | Local-space X-ray peel reveals serpentine S-ducts |
| `b3_duct_enter` | `655` | `10.92` | Radar pulse enters dorsal intake & bounces inside S-curve |
| `b3_duct_trap` | `785` | `13.08` | Pulse dies in RAM duct lining before reaching spinning blades |
| `b4_ram_wave` | `995` | `16.58` | RF wavefront strikes hexagonal carbon-ferrite RAM lattice |
| `b4_heat_absorb` | `1075` | `17.92` | Energy dissipates into faint warm amber heat ripples |
| `b4_rcs_shrink` | `1160` | `19.33` | Holographic `100 m²` RCS sphere collapses to `0.001 m²` bumblebee |
| `b5_mirror_echo` | `1365` | `22.75` | Blueprint top 90° mirror fires crimson echo back |
| `b5_facet_scatter` | `1445` | `24.08` | Blueprint bottom tilted stealth facet scatters cyan wave upward |
| `b6_moon_pass` | `1600` | `26.67` | Stealth bomber banks past camera in moonlight rim-light |
| `b6_final_blip` | `1695` | `28.25` | Tactical radar sweep reveals lone `0.001 m²` bumblebee speck |

---

## 4. Blender 5.3.0 & V12 Compliance Checklist

1. **Slotted Actions:** Always iterate FCurves via `stealth_contract.iter_action_fcurves(action)`.
2. **ShaderNodeMix:** Always resolve `(Factor, A, B)` via `stealth_contract.get_mix_sockets(node)` (slicing `inputs[2:]`).
3. **Materials Before `to_mesh`:** Append every material slot to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
4. **Open-Ended Volumetric Beams & Waves:** `cap_ends=False` / `end_fill_type="NOTHING"` + `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
5. **Local-Space X-Ray Skin:** Driven exclusively by `ShaderNodeTexCoord.Object` bound to `Stealth_Bomber_Root`, hollow Fresnel `pow(Facing, 3.8)` with zero opaque fill.
6. **Razor-Sharp 3D Labels:** `1–3 words`, `DejaVuSans-Bold.ttf`, `focus_dist - 0.05 m`, `f/7.5+` when visible, `NDC Y >= -0.45`, dark obsidian backing plaque `(0.008, 0.012, 0.020)`, Object-Space `|N.z| > 0.85` face shader (`Emission <= 1.85`), `visible_glossy = visible_transmission = visible_diffuse = False`.
7. **Audio & Subtitles:** `en-US-BrianNeural` (`96 words`), zero truncation (`atempo <= 1.12x`), `48000 Hz` Stereo `-14.0 LUFS` / `<= -1.5 dBFS`, synced `\kf` ASS karaoke (`NDC Y = -0.583 < -0.55`).
8. **Modal & `--draft-preview`:** Byte-for-byte cached `studio_image`, `build_stealth_scene.py --frame-step`, and `render_on_modal.py --draft-preview` (`724×1286`, `30 fps`, `8 spp`, `900` frames).
