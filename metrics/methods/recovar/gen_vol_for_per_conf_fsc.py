import logging
import numpy as np
from recovar import output as o
from recovar import dataset, embedding
import os, argparse

logger = logging.getLogger(__name__)
from cryodrgn import analysis


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


def compute_state(args):

    po = o.PipelineOutput(args.result_dir + "/")
    output_folder = args.outdir

    cryos = po.get("dataset")
    embedding.set_contrasts_in_cryos(cryos, po.get("contrasts")[args.zdim])
    zs = po.get("zs")[args.zdim]
    cov_zs = po.get("cov_zs")[args.zdim]
    noise_variance = po.get("noise_var_used")
    n_bins = args.n_bins
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

    o.mkdir_safe(output_folder)
    logger.addHandler(logging.FileHandler(f"{output_folder}/run.log"))
    logger.info(args)
    o.compute_and_save_reweighted(
        cryos,
        target_zs,
        zs,
        cov_zs,
        noise_variance,
        output_folder,
        args.Bfactor,
        n_bins,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)
