from pathlib import Path
import re

from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame


CONTENTS_PATH = Path(__file__).with_name("contents.txt")
NPIX = 384
PIXEL_SCALE_ARCMIN = 3.4
HALF_SIDE_DEG = NPIX * PIXEL_SCALE_ARCMIN / 60.0 / 2.0


def parse_patches():
    pattern = re.compile(
        r"patch_(?P<patch_id>\d+)_.*?_cen_pix_lon_"
        r"(?P<lon>[-+]?\d+(?:\.\d+)?)_lat_"
        r"(?P<lat>[-+]?\d+(?:\.\d+)?)_"
    )
    records = {}
    for line in CONTENTS_PATH.read_text().splitlines():
        line = line.strip()
        match = pattern.search(line)
        if match is None:
            continue
        patch_id = int(match.group("patch_id"))
        records.setdefault(
            patch_id,
            {
                "patch_id": patch_id,
                "l": float(match.group("lon")),
                "b": float(match.group("lat")),
                "example": line,
            },
        )
    return [records[k] for k in sorted(records)]


def main():
    # BK18 states that the BICEP2/Keck sky region is centered at RA=0h,
    # Dec=-57.5 deg. BICEP3 has a larger slightly shifted scan region, but the
    # numerical shifted center is not given in the local extracted text.
    bicep_icrs = SkyCoord(ra=0 * u.deg, dec=-57.5 * u.deg, frame="icrs")
    bicep_gal = bicep_icrs.galactic

    print("BICEP2/Keck center from BK18: RA=0 deg, Dec=-57.5 deg")
    print(
        "BICEP2/Keck center in Galactic coordinates: "
        f"l={bicep_gal.l.deg:.9f} deg, b={bicep_gal.b.deg:.9f} deg"
    )
    print(f"Patch half-side: {HALF_SIDE_DEG:.6f} deg")

    contains = []
    nearest = []
    for rec in parse_patches():
        center = SkyCoord(l=rec["l"] * u.deg, b=rec["b"] * u.deg, frame="galactic")
        offsets = bicep_gal.transform_to(SkyOffsetFrame(origin=center))
        dx = offsets.lon.to_value(u.deg)
        dy = offsets.lat.to_value(u.deg)
        sep = center.separation(bicep_gal).deg
        inside = abs(dx) <= HALF_SIDE_DEG and abs(dy) <= HALF_SIDE_DEG
        out = {
            **rec,
            "dx_deg": dx,
            "dy_deg": dy,
            "sep_deg": sep,
            "inside": inside,
        }
        nearest.append(out)
        if inside:
            contains.append(out)

    print(f"Containing patches: {len(contains)}")
    for rec in contains:
        print(
            f"patch_{rec['patch_id']}: "
            f"center l={rec['l']:.9f}, b={rec['b']:.9f}; "
            f"offset dx={rec['dx_deg']:.3f} deg, dy={rec['dy_deg']:.3f} deg; "
            f"separation={rec['sep_deg']:.3f} deg"
        )
        print(f"example file: {rec['example']}")

    print("Nearest patches:")
    for rec in sorted(nearest, key=lambda row: row["sep_deg"])[:5]:
        print(
            f"patch_{rec['patch_id']}: "
            f"center l={rec['l']:.9f}, b={rec['b']:.9f}; "
            f"offset dx={rec['dx_deg']:.3f} deg, dy={rec['dy_deg']:.3f} deg; "
            f"separation={rec['sep_deg']:.3f} deg; inside={rec['inside']}"
        )


if __name__ == "__main__":
    main()
