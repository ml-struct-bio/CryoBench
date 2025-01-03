"""Calculate FSCs across RECOVAR model conformations.

Example usage
-------------
$ python metrics/fsc/old/per_conf/re_covar.py results/recovar/001/ \
            -o cryobench-outputs/recovar/ --gt-dir IgG-1D/vols/128_org/ \
            --mask IgG-1D/init_mask/mask.mrc --num-imgs 1000 --num-vols 100

"""
import os
import sys
import argparse
import logging
import numpy as np
from cryodrgn import analysis

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOTDIR, "methods", "recovar"))
from recovar import dataset, embedding, output

sys.path.append(os.path.join(ROOTDIR, "fsc"))
from utils import volumes, conformations, interface

logging.basicConfig(
    level=logging.INFO,
    format="(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--zdim", default=10, type=int)
    parser.add_argument(
        "--n-bins",
        type=float,
        default=50,
        dest="n_bins",
        help="number of bins for reweighting",
    )
    parser.add_argument("--Bfactor", type=float, default=0, help="0")

    return parser


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by RECOVAR."""

    pipeline_output = output.PipelineOutput(args.input_dir)
    cryos = pipeline_output.get("lazy_dataset")
    zs = pipeline_output.get("zs")[args.zdim]
    zs_reordered = dataset.reorder_to_original_indexing(zs, cryos)

    latent_path = os.path.join(args.input_dir, "reordered_z.npy")
    umap_path = os.path.join(args.input_dir, "reordered_z_umap.npy")
    if os.path.exists(latent_path) and not args.overwrite:
        logger.info("latent coordinates file already exists, skipping...")
    else:
        np.save(latent_path, zs_reordered)

    if os.path.exists(umap_path) and not args.overwrite:
        logger.info("latent UMAP clustering already exists, skipping...")
    else:
        umap_pkl = analysis.run_umap(zs_reordered)
        np.save(umap_path, umap_pkl)

    cryos = pipeline_output.get("dataset")
    embedding.set_contrasts_in_cryos(cryos, pipeline_output.get("contrasts")[args.zdim])
    zs = pipeline_output.get("zs")[args.zdim]
    cov_zs = pipeline_output.get("cov_zs")[args.zdim]
    noise_variance = pipeline_output.get("noise_var_used")
    zs_reordered = dataset.reorder_to_original_indexing(zs, cryos)
    num_imgs = int(args.num_imgs) if zs.shape[0] == 100000 else "ribo"
    nearest_z_array = conformations.get_nearest_z_array(
        zs_reordered, args.num_vols, num_imgs
    )

    output.mkdir_safe(args.outdir)
    log_file = os.path.join(args.outdir, "run.log")
    if os.path.exists(log_file) and not args.overwrite:
        logger.info("run.log file exists, skipping...")
    else:
        logger.addHandler(logging.FileHandler(log_file))
        logger.info(args)

        output.compute_and_save_reweighted(
            cryos,
            nearest_z_array,
            zs,
            cov_zs,
            noise_variance,
            args.outdir,
            args.Bfactor,
            args.n_bins,
        )

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(args.outdir, args.gt_dir, flip=args.flip_align)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            args.outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=lambda i: os.path.join(
                f"vol{i:03d}", "ml_optimized_locres_filtered"
            ),
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                args.outdir,
                args.gt_dir,
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
                vol_fl_function=lambda i: os.path.join(
                    f"vol{i:03d}", "ml_optimized_locres_filtered"
                ),
            )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
