# PRODUCTION CONTRACT — Episode #4: "How a 172-Foot Stealth Bomber Looks Like a Bumblebee on Radar"

Single source of truth for all 3D modules, MPFB2 rigged characters, B-2 Spirit hero geometry, camera framing, shaders, Foley SFX, narration, subtitles, and Modal 4K / `--draft-preview` cloud rendering. Follows **AGENTS.md V12** (`60 fps`, `1800 frames`, `30.0 s`, `1080×1920` native → `2160×3840` True 4K, Blender `5.3.0` Cycles OPTIX + GPU OIDN, `AgX - Punchy`).

---

## 1. 6-Beat Action-Thriller Architecture (`1..1800` @ `60 fps` = `30.0 s`)

| Beat | Frames | Time | Visual Story, Characters & Camera | 3D Callout (`<= 3 words`) |
|---|---|---|---|---|
| `B1_HOOK` | `1..220` | `0.00–3.67 s` | Bright daylight Enemy Air-Defense Bunker: MPFB2 `Defense_Commander` & `Radar_Operator` stare at the massive USAF B-2 Spirit banking outside the frameless panoramic glass while their phosphor radar scope stays blank | `0 TARGETS` (`68..212`) |
| `B2_FACETS` | `221..540` | `3.67–9.00 s` | Daylight aerial intercept: conventional jet (upper) blazes a huge crimson return & 3D glowing missile-lock reticle box (`FX_B2_MissileLockBox`, `312..540`) while the photo-accurate B-2's 33° angled facets deflect cyan pulses sideways | `33° DEFLECTION` (`260..528`) |
| `B3_SDUCTS` | `541..960` | `9.00–16.00 s` | Macro dorsal banking X-ray sweep (local `TexCoord.Object` + hollow Fresnel `AddShader`) peeling back B-2 nacelles to reveal deep serpentine S-ducts hiding spinning titanium turbofans | `HIDDEN TURBINES` (`615..945`) |
| `B4_RAM_BEE` | `961..1320` | `16.00–22.00 s` | Macro hexagonal carbon-ferrite RAM lattice absorbing RF waves into harmless warm heat + holographic `100 m²` RCS orb collapsing onto a sculpted fluttering `0.001 m²` 3D bumblebee | `HEAT ABSORPTION` (`990..1145`) → `0.001 m² BLIP` (`1155..1308`) |
| `B5A_COCKPIT` | `1321..1430` | `22.00–23.83 s` | 1:1 scale B-2 Spirit cockpit close-up (`f/3.8` cinematic bokeh): MPFB2 `Stealth_Pilot` (HGU-55 helmet, gold visor, oxygen mask) works the flight stick behind a glowing green HUD as a SAM radar sweep glints across the canopy | — (pure visual macro) |
| `B5B_SHOWDOWN` | `1431..1560` | `23.83–26.00 s` | Live-action daylight SAM radar showdown (`f/8.0`): enemy ground radar dish hits conventional jet (crimson return) vs. banking B-2 Spirit deflecting every pulse harmlessly skyward | `MISSILE BLIND` (`1445..1550`) |
| `B6_FINAL` | `1561..1800` | `26.00–30.00 s` | Bunker climax twist: alarmed Commander & Operator spot a single tiny `0.001 m²` bumblebee speck on the console scope right as the B-2 looms overhead outside and opens its twin belly weapons-bay doors (`1665..1800`) | `BUMBLEBEE SIZE` (`1595..1785`) |

Hard cut boundaries (camera `CONSTANT` interpolation): `221`, `541`, `961`, `1321`, `1431`, `1561`.

---

## 2. Hero Assets & Characters (`src/characters.py`, `src/stealth_aircraft_asset.py`)

- **3 Photoreal 4K MakeHuman MPFB2 Crew (`02_MPFB2_Crew_Cockpit`):**
  - `Radar_Operator` (Beats 1 & 6): seated at radar console, comms headset, subtle leaning & head-turn reactions between scope and window.
  - `Defense_Commander` (Beats 1 & 6): standing beside console, officer visor cap & gold epaulets, gesturing urgently toward the looming B-2 outside.
  - `Stealth_Pilot` (Beat 5A `1321..1430`): seated in 1:1 scale `Char_B2_CockpitFrame` with glowing `Char_B2_CockpitHUD`, animated `Char_B2_FlightStick`, HGU-55 helmet, reflective gold visor, oxygen mask, and flight harness; lit by `Light_Pilot_VisorCatchlight` + `Light_Pilot_HUDBounce`.
  - All MPFB2 actors follow `AGENTS.md` V12 eye realism (`visible_shadow = False` on eyes/lashes, independent `EnhancedEye` node group copy per actor, front-facing specular cornea, `0.42m` softbox eye catchlight `Light_Crew_EyeCatchlight`, natural `-5°` eyelid baseline, clamped jaw `<= 14°`).
- **Photo-Accurate USAF B-2 Spirit (`Stealth_Bomber_Root`):**
  - Sunlit gunmetal-slate RAM finish, higher-relief hawk-beak cockpit spine with 4-pane gleaming silver-gold wrap-around windshield, twin dorsal engine nacelles with serrated W-lip intakes, boundary-layer splitter teeth & crimson red military intake warning triangles, crisp light-grey USAF walkway hatch outlines, recessed dark exhaust troughs with splitter vanes, outer split trailing-edge elevons/drag-rudders, and Beat 6 animated twin belly weapons-bay doors with warm amber bay glow (`1665..1800`).

---

## 3. Exact Frame Milestones (`EV` synced 1:1 with `config/timeline.json`)

| Event Key | Frame | Time (s) | Action / Foley Trigger |
|---|---:|---:|---|
| `b1_radar_ping` | `18` | `0.30` | Ground radar dish emits first cyan pulse cone |
| `b1_wing_deflect` | `68` | `1.13` | Pulse hits bomber leading edge & ricochets sideways outside bunker |
| `b1_blank_scope` | `135` | `2.25` | Bunker radar scope sweeps dead blank (`0 TARGETS`) |
| `b2_conv_ping` | `265` | `4.42` | Radar hits conventional jet fuselage & vertical tail |
| `b2_conv_echo` | `318` | `5.30` | Blazing crimson radar return & 3D missile-lock box snap onto conventional jet |
| `b2_stealth_deflect` | `410` | `6.83` | B-2 33° sawtooth facets deflect cyan waves sideways |
| `b3_xray_sweep` | `575` | `9.58` | Local-space X-ray peel reveals serpentine S-ducts |
| `b3_duct_enter` | `655` | `10.92` | Radar pulse enters dorsal intake & bounces inside S-curve |
| `b3_duct_trap` | `785` | `13.08` | Pulse dies in RAM duct lining before reaching spinning turbines |
| `b4_ram_wave` | `995` | `16.58` | RF wavefront strikes hexagonal carbon-ferrite RAM lattice |
| `b4_heat_absorb` | `1075` | `17.92` | Energy dissipates into faint warm amber heat ripples |
| `b4_rcs_shrink` | `1160` | `19.33` | Holographic `100 m²` RCS sphere collapses to `0.001 m²` bumblebee |
| `b5_mirror_echo` | `1365` | `22.75` | SAM radar warning sweep glints across B-2 pilot's cockpit canopy |
| `b5_facet_scatter` | `1445` | `24.08` | SAM Showdown: B-2 banking facets scatter radar beam skyward |
| `b6_moon_pass` | `1600` | `26.67` | B-2 looms overhead outside bunker & cracks open twin belly weapons-bay doors |
| `b6_final_blip` | `1695` | `28.25` | Bunker radar scope resolves lone `0.001 m²` bumblebee speck |

---

## 4. Blender 5.3.0 & V12 Compliance Checklist

1. **Slotted Actions:** Always iterate FCurves via `stealth_contract.iter_action_fcurves(action)`.
2. **ShaderNodeMix:** Always resolve `(Factor, A, B)` via `stealth_contract.get_mix_sockets(node)` (slicing `inputs[2:]`).
3. **Materials Before `to_mesh`:** Append every material slot to `mesh.materials` BEFORE `bm.to_mesh(mesh)`.
4. **Open-Ended Volumetric Beams & Waves:** `cap_ends=False` / `end_fill_type="NOTHING"` + `AddShader(Transparent, Emission * CameraRay * (1 - Backfacing))`.
5. **Local-Space X-Ray Skin:** Driven exclusively by `ShaderNodeTexCoord.Object` bound to `Stealth_Bomber_Root`, hollow Fresnel `pow(Facing, 3.8)` with zero opaque fill.
6. **Razor-Sharp 3D Labels:** `1–3 words`, `DejaVuSans-Bold.ttf`, `focus_dist - 0.05 m`, `f/7.5+` when visible (`f/3.8` reserved exclusively for label-free Beat 5A cockpit macro), `NDC Y >= -0.45`, dark obsidian backing plaque `(0.008, 0.012, 0.020)`, Object-Space `|N.z| > 0.85` face shader (`Emission <= 1.85`), `visible_glossy = visible_transmission = visible_diffuse = False`.
7. **Audio & Subtitles:** `en-US-BrianNeural` (`97 words`), zero truncation (`atempo <= 1.12x`), `48000 Hz` Stereo `-14.0 LUFS` / `<= -1.5 dBFS`, synced `\kf` ASS karaoke (`NDC Y = -0.583 < -0.55`).
8. **Modal & `--draft-preview`:** Byte-for-byte cached `studio_image`, `build_stealth_scene.py --frame-step`, automatic audio/subtitle rebuild on narration mismatch, and `render_on_modal.py --draft-preview` (`724×1286`, `30 fps`, `8 spp`, `900` frames).
