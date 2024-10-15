"""Command-line interfaces shared across FSC per conformation commands."""

import argparse
import os


def add_calc_args() -> argparse.ArgumentParser:
    """Command-line interface used in commands calculating FSCs per conformation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="dir contains weights, config, z")
    parser.add_argument("-o", help="Output directory")
    parser.add_argument(
        "--epoch", default=19, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--num-vols",
        default=100,
        type=int,
        help="Number of total reconstructed volumes",
    )
    parser.add_argument("--Apix", default=3.0, type=float)
    parser.add_argument(
        "--num-imgs",
        default=1000,
        type=int,
        help="Number of images per model (structure)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="override auto label for type of method, used for output subfolder",
    )
    parser.add_argument(
        "--mask",
        default=None,
        type=os.path.abspath,
        help="Path to mask .mrc to compute the masked metric",
    )
    parser.add_argument("--gt-dir", help="Directory of gt volumes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", type=int, default=1)
    parser.add_argument("--cuda-device", default=0, type=int)
    parser.add_argument("--no-fscs", action="store_false", dest="calc_fsc_vals")

    return parser
