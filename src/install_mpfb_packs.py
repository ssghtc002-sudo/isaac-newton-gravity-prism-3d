"""
Headless MPFB2 V8 Official 4K Asset Pack Installer for Kaggle GPU T4 x2 (Blender 4.2.3 LTS).
Extracts curated MakeHuman Community 4K cast & real 3D historical wardrobe ZIPs cleanly into MPFB2's
exact LocationService.get_user_data() directory on Kaggle SSD disk (never /tmp tmpfs!), verifies
ZIP CRC32 integrity before extraction, and refreshes MPFB2's asset library cache.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import zipfile

import bpy


def _enable_mpfb2():
    if not any(m.endswith("mpfb.services.assetservice") for m in sys.modules):
        for mod_name in ("bl_ext.user_default.mpfb", "mpfb"):
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                if any(m.endswith("mpfb.services.assetservice") for m in sys.modules):
                    break
            except Exception:
                continue

    def dyn(pkg: str, key: str):
        for amod in list(sys.modules.keys()):
            if amod.endswith(pkg):
                m = importlib.import_module(amod)
                if hasattr(m, key):
                    return getattr(m, key)
        raise ImportError(f"MPFB2 module '{pkg}.{key}' not found!")

    return (
        dyn("mpfb.services.locationservice", "LocationService"),
        dyn("mpfb.services.assetservice", "AssetService"),
    )


def main():
    LocationService, AssetService = _enable_mpfb2()
    data_dir = LocationService.get_user_data()
    os.makedirs(data_dir, exist_ok=True)
    print(f"[MPFB2 Packs] Target MPFB2 user data dir: {data_dir}")

    # Pure Kaggle SSD cache default (never use /tmp tmpfs on Kaggle!)
    default_dir = (
        "/kaggle/working/.cache/mpfb_packs"
        if os.path.isdir("/kaggle/working")
        else os.path.join(os.path.expanduser("~"), ".cache", "mpfb_packs")
    )
    pack_dir = os.environ.get("MPFB_PACKS_DIR", default_dir)
    zips = sorted(glob.glob(os.path.join(pack_dir, "*.zip")))
    if not zips:
        print(f"[MPFB2 Packs] Warning: no zip files found in {pack_dir}")
        return

    marker_dir = os.path.join(data_dir, ".installed_packs")
    os.makedirs(marker_dir, exist_ok=True)

    known_roots = {"skins", "eyes", "eyebrows", "eyelashes", "teeth", "tongue", "hair", "clothes", "bodyparts"}

    for zp in zips:
        base = os.path.basename(zp)
        marker = os.path.join(marker_dir, base + ".ok")
        if os.path.exists(marker):
            print(f"[MPFB2 Packs] Already installed: {base}")
            continue

        # Validate CRC32 integrity before extracting so truncated downloads fail loudly
        with zipfile.ZipFile(zp, "r") as zf_chk:
            bad = zf_chk.testzip()
            if bad is not None:
                raise RuntimeError(f"[MPFB2 Packs] Corrupt ZIP entry '{bad}' inside {zp} — re-download pack!")

        print(f"[MPFB2 Packs] Extracting {base} -> {data_dir} ...")
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                top_dirs = {n.split("/", 1)[0].lower() for n in zf.namelist() if "/" in n}
                if top_dirs & known_roots:
                    zf.extractall(data_dir)
                else:
                    check = AssetService.check_asset_pack_zip(zp)
                    if check is not None:
                        res = AssetService.fix_and_extract_asset_pack_zip(zp, data_dir)
                        if res is not None:
                            print(f"[MPFB2 Packs] fix_and_extract notice ({res}), direct extract fallback")
                            zf.extractall(data_dir)
                    else:
                        zf.extractall(data_dir)
        except Exception as err:
            print(f"[MPFB2 Packs] Direct extract fallback for {base} ({err})")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(data_dir)
        with open(marker, "w", encoding="utf-8") as mf:
            mf.write("ok\n")

    AssetService.update_all_asset_lists()
    skins = AssetService.list_mhmat_assets("skins")
    clothes = AssetService.list_mhclo_assets("clothes")
    hair = AssetService.list_mhclo_assets("hair")
    eyes = AssetService.list_mhclo_assets("eyes")
    brows = AssetService.list_mhclo_assets("eyebrows")
    lashes = AssetService.list_mhclo_assets("eyelashes")
    teeth = AssetService.list_mhclo_assets("teeth")
    tongue = AssetService.list_mhclo_assets("tongue")
    print(
        f"[MPFB2 Packs] READY! Installed -> "
        f"skins={len(skins)}, eyes={len(eyes)}, eyebrows={len(brows)}, "
        f"eyelashes={len(lashes)}, teeth={len(teeth)}, tongue={len(tongue)}, "
        f"hair={len(hair)}, clothes={len(clothes)}"
    )
    print(f"[MPFB2 Packs] Hair installed:    {sorted(os.path.basename(str(p)) for p in hair)}")
    print(f"[MPFB2 Packs] Clothes installed: {sorted(os.path.basename(str(p)) for p in clothes)}")
    print(f"[MPFB2 Packs] Skins installed:   {sorted(os.path.basename(str(p)) for p in skins)}")


if __name__ == "__main__":
    main()
