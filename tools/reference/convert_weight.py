import argparse
from pathlib import Path

import torch


def save_only_ema_weights(checkpoint_file: Path):
    """Extract and save only the EMA weights."""
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    weights = {}
    if "ema" in checkpoint:
        weights["model"] = checkpoint["ema"]["module"]
    else:
        raise ValueError("The checkpoint does not contain 'ema'.")

    output_file: Path = checkpoint_file.with_stem(f"{checkpoint_file.stem}_converted")

    torch.save(weights, output_file)
    print(f"EMA weights saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save only EMA weights.")
    parser.add_argument(
        "-c", "--checkpoint_dir",
        type=Path,
        help=argparse.SUPPRESS,  # Hide this flag if positional is preferred
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        help=argparse.SUPPRESS,  # Hide this flag if positional is preferred
        default="",
    )

    args = parser.parse_args()
    for file in args.checkpoint_dir.glob(f"*{args.filter}*.pth"):
        if "_converted" not in file.name:
            save_only_ema_weights(file)
