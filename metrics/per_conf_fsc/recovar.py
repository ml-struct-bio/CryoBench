import argparse
import os
import sys
import logging
import interface
import utils
import numpy as np
from cryodrgn import analysis

sys.path.append(os.path.join("methods", "recovar"))
import recovar

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    if args.method is None:
        logger.info('No method label specified, using "recovar" as default...')
        method_lbl = "recovar"
    else:
        method_lbl = str(args.method)

    pipeline_output = recovar.output.PipelineOutput(args.input_dir)
    cryos = pipeline_output.get("lazy_dataset")
    zs = pipeline_output.get("zs")[args.zdim]
    zs_reordered = recovar.dataset.reorder_to_original_indexing(zs, cryos)

    latent_path = os.path.join(args.recovar_result_dir, "reordered_z.npy")
    umap_path = os.path.join(args.recovar_result_dir, "reordered_z_umap.npy")
    if os.path.exists(latent_path) and not args.overwrite:
        print("latent exists, skipping...")
    else:
        np.save(latent_path, zs_reordered)

    if os.path.exists(umap_path) and not args.overwrite:
        print("latent exists, skipping...")
    else:
        umap_pkl = analysis.run_umap(zs_reordered)
        np.save(umap_path, umap_pkl)

    output_folder = args.outdir

    cryos = pipeline_output.get("dataset")
    recovar.embedding.set_contrasts_in_cryos(
        cryos, pipeline_output.get("contrasts")[args.zdim]
    )
    zs = pipeline_output.get("zs")[args.zdim]
    cov_zs = pipeline_output.get("cov_zs")[args.zdim]
    noise_variance = pipeline_output.get("noise_var_used")
    n_bins = args.n_bins
    zs_reordered = recovar.dataset.reorder_to_original_indexing(zs, cryos)

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

    recovar.output.mkdir_safe(output_folder)
    logger.addHandler(logging.FileHandler(f"{output_folder}/run.log"))
    logger.info(args)
    recovar.output.compute_and_save_reweighted(
        cryos,
        target_zs,
        zs,
        cov_zs,
        noise_variance,
        output_folder,
        args.Bfactor,
        n_bins,
    )

    outdir = str(os.path.join(args.o, method_lbl, "per_conf_fsc"))
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
    main(interface.add_calc_args().parse_args())
