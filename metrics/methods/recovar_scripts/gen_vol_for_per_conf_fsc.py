import os
import argparse


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "result_dir",
        type=os.path.abspath,
        help="result dir (output dir of pipeline)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=False,
        help="Output directory to save model",
    )
    parser.add_argument("--Bfactor", type=float, default=0, help="0")

    parser.add_argument(
        "--n-bins",
        type=float,
        default=50,
        dest="n_bins",
        help="number of bins for reweighting",
    )

    parser.add_argument(
        "--zdim", type=int, default=20, help="z dim of the latent space"
    )
    parser.add_argument(
        "--num-imgs", type=int, default=1000, help="z dim of the latent space"
    )
    parser.add_argument(
        "--num-vols", default=100, type=int, help="number of G.T Volumes"
    )

    return parser
