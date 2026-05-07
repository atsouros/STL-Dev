# Balanced Galaxy Zoo Image Dataset

This folder contains a script for creating a balanced image dataset from the
Galaxy Zoo 1 metadata table.

The metadata file contains galaxy labels and sky coordinates, but not image
pixels. The script uses the coordinates to download JPEG cutouts from SDSS
SkyServer.

## Files

- `build_balanced_galaxy_zoo_images.py`: dataset builder script
- `GalaxyZoo1_DR_table2.csv.zip`: Galaxy Zoo 1 metadata table used as input
- `GalaxyZoo1_DR_table2.fits`: same catalog in FITS format, not required by the script
- `check_fits_shape.ipynb`: notebook for inspecting the FITS file

## Requirements

The script uses only the Python standard library. No extra Python packages are
required.

You need:

- Python 3
- Internet access
- `GalaxyZoo1_DR_table2.csv.zip` in this folder, or passed with `--catalog-zip`

## Create the Dataset

Run:

```bash
python3 build_balanced_galaxy_zoo_images.py
```

By default this creates a dataset with 15,000 images:

- 5,000 spiral galaxies
- 5,000 elliptical galaxies
- 5,000 uncertain galaxies

The output folder is:

```text
galaxy_zoo_balanced_15000/
```

Its structure is:

```text
galaxy_zoo_balanced_15000/
  manifest.csv
  images/
    spiral/
    elliptical/
    uncertain/
```

Each image is saved as:

```text
images/<class>/<OBJID>.jpg
```

## Manifest

The script writes `manifest.csv`, which records the selected galaxies and their
labels.

Important columns include:

- `objid`: SDSS object ID
- `class`: one of `spiral`, `elliptical`, or `uncertain`
- `ra`, `dec`: original catalog coordinates
- `ra_deg`, `dec_deg`: coordinates converted to decimal degrees
- `image_path`: local image path relative to the output folder
- `source_url`: SDSS SkyServer image URL

To create only the manifest without downloading images:

```bash
python3 build_balanced_galaxy_zoo_images.py --manifest-only
```

## Reproducibility

The default random seed is `42`, so the same input catalog should produce the
same balanced sample.

To use a different seed:

```bash
python3 build_balanced_galaxy_zoo_images.py --seed 123
```

## Resume Downloads

The script is resumable. If it is interrupted, run the same command again:

```bash
python3 build_balanced_galaxy_zoo_images.py
```

Existing non-empty image files are skipped.

To force redownloading existing files:

```bash
python3 build_balanced_galaxy_zoo_images.py --overwrite
```

## Useful Options

Create a smaller test dataset:

```bash
python3 build_balanced_galaxy_zoo_images.py --per-class 10 --output-dir galaxy_zoo_test
```

Use a different input catalog path:

```bash
python3 build_balanced_galaxy_zoo_images.py --catalog-zip /path/to/GalaxyZoo1_DR_table2.csv.zip
```

Change image size:

```bash
python3 build_balanced_galaxy_zoo_images.py --size 128
```

Change the SDSS cutout scale:

```bash
python3 build_balanced_galaxy_zoo_images.py --scale 0.396
```

Use fewer parallel downloads:

```bash
python3 build_balanced_galaxy_zoo_images.py --workers 4
```

## Notes

The classes come from the Galaxy Zoo catalog columns:

- `SPIRAL`
- `ELLIPTICAL`
- `UNCERTAIN`

Only rows with exactly one of these flags set to `1` are used.

The downloaded images are SDSS JPEG cutouts centered on each galaxy's catalog
coordinates. They are useful for visual machine learning experiments, but they
are not calibrated scientific FITS image frames.
