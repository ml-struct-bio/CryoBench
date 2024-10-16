import argparse
import os
import sys
import logging
import interface
import utils
import numpy as np
from cryodrgn import analysis

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "methods", "recovar")
)
from recovar import dataset, embedding, output

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
    if args.method is None:
        logger.info('No method label specified, using "recovar" as default...')
        method_lbl = "recovar"
    else:
        method_lbl = str(args.method)

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

    z_lst = []
    z_mean_lst = []
    for i in range(args.num_vols):
        z_nth = zs_reordered[i * args.num_imgs : (i + 1) * args.num_imgs]
        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)
    nearest_z_lst = []
    centers_ind_lst = []
    for i in range(args.num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind)
    target_zs = np.array(nearest_z_lst)

    output.mkdir_safe(args.outdir)
    logger.addHandler(logging.FileHandler(os.path.join(args.outdir, "run.log")))
    logger.info(args)

    output.compute_and_save_reweighted(
        cryos,
        target_zs,
        zs,
        cov_zs,
        noise_variance,
        args.outdir,
        args.Bfactor,
        args.n_bins,
    )
    outdir = str(os.path.join(args.outdir, method_lbl, "per_conf_fsc"))
    logger.info(f"Putting output under: {outdir} ...")

    if args.calc_fsc_vals:

        def vol_fl_function(i: int):
            return os.path.join(format(i, "03d"), "ml_optimized_locres_filtered.mrc")

        utils.get_fsc_curves(
            outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=vol_fl_function,
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
