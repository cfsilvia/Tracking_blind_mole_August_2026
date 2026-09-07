"""Export rat centroid tracks from a SAM2 mask .npz file."""

import argparse
from pathlib import Path

import numpy as np

from sam2_video_utils import Sam2VideoUtils


def main():
    """
    Input: command-line paths to a SAM2 mask NPZ and optional output CSV.
    Output: centroid CSV file written next to the mask file or at the requested path.
    """
    parser = argparse.ArgumentParser(description="Export SAM2 mask centroids to CSV.")
    parser.add_argument("--masks", required=True, help="Input *_sam2_masks.npz file.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    masks_path = Path(args.masks).resolve()
    if not masks_path.exists():
        raise FileNotFoundError(masks_path)

    output_csv = (
        Path(args.output).resolve()
        if args.output
        else masks_path.with_name(masks_path.stem.replace("_masks", "_centroids") + ".csv")
    )

    data = np.load(masks_path)
    num_animals = int(data["num_animals"]) if "num_animals" in data.files else 2
    Sam2VideoUtils.write_centroids_csv(
        output_csv,
        data["rat1"].astype(bool),
        data["rat2"].astype(bool),
        data["seen"].astype(bool),
        num_animals,
    )
    print(f"Saved centroids: {output_csv}")


if __name__ == "__main__":
    main()
