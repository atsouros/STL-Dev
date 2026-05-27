from pathlib import Path
import csv
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import Galactic, SkyCoord, SkyOffsetFrame


CONTENTS_PATH = Path(__file__).with_name("contents.txt")
NPIX = 384
PIXEL_SCALE_ARCMIN = 3.4
LAT_THRESHOLD_DEG = 45.0


def parse_patches(contents_path):
    pattern = re.compile(
        r"patch_(?P<patch_id>\d+)_.*?_cen_pix_lon_"
        r"(?P<lon>[-+]?\d+(?:\.\d+)?)_lat_"
        r"(?P<lat>[-+]?\d+(?:\.\d+)?)_"
    )
    records_by_id = {}
    for line in contents_path.read_text().splitlines():
        line = line.strip()
        match = pattern.search(line)
        if match is None:
            continue
        patch_id = int(match.group("patch_id"))
        records_by_id.setdefault(
            patch_id,
            {
                "patch_id": patch_id,
                "center_l_deg": float(match.group("lon")),
                "center_b_deg": float(match.group("lat")),
                "example_file": line,
            },
        )
    return [records_by_id[k] for k in sorted(records_by_id)]


def boundary_offsets(half_side_deg, n_per_edge=NPIX + 1):
    t = np.linspace(-half_side_deg, half_side_deg, n_per_edge)
    x = np.concatenate(
        [t, np.full_like(t, half_side_deg), t, np.full_like(t, -half_side_deg)]
    )
    y = np.concatenate(
        [np.full_like(t, half_side_deg), t, np.full_like(t, -half_side_deg), t]
    )
    return x, y


def min_abs_b_for_patch(center_l_deg, center_b_deg, x_offsets_deg, y_offsets_deg):
    center = SkyCoord(l=center_l_deg * u.deg, b=center_b_deg * u.deg, frame="galactic")
    offset_frame = SkyOffsetFrame(origin=center)
    boundary = SkyCoord(
        lon=x_offsets_deg * u.deg,
        lat=y_offsets_deg * u.deg,
        frame=offset_frame,
    ).transform_to(Galactic)
    return float(np.min(np.abs(boundary.b.deg)))


def main():
    side_deg = NPIX * PIXEL_SCALE_ARCMIN / 60.0
    half_side_deg = side_deg / 2.0
    x_offsets_deg, y_offsets_deg = boundary_offsets(half_side_deg)

    patches = parse_patches(CONTENTS_PATH)
    selected = []
    for rec in patches:
        rec = dict(rec)
        rec["min_abs_b_boundary_deg"] = min_abs_b_for_patch(
            rec["center_l_deg"], rec["center_b_deg"], x_offsets_deg, y_offsets_deg
        )
        if rec["min_abs_b_boundary_deg"] > LAT_THRESHOLD_DEG:
            selected.append(rec)

    out_csv = CONTENTS_PATH.with_name("patches_entirely_abs_b_gt_45.csv")
    out_txt = CONTENTS_PATH.with_name("patch_ids_entirely_abs_b_gt_45.txt")

    fieldnames = [
        "patch_id",
        "center_l_deg",
        "center_b_deg",
        "min_abs_b_boundary_deg",
        "example_file",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selected:
            writer.writerow({key: rec[key] for key in fieldnames})

    out_txt.write_text("\n".join(str(rec["patch_id"]) for rec in selected) + "\n")

    print(f"unique patches: {len(patches)}")
    print(f"selected patches: {len(selected)}")
    print(f"side_deg: {side_deg}")
    print(f"half_side_deg: {half_side_deg}")
    print("ids:")
    print([rec["patch_id"] for rec in selected])
    print(f"csv: {out_csv}")
    print(f"txt: {out_txt}")


if __name__ == "__main__":
    main()
