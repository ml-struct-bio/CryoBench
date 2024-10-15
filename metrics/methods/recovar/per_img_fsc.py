import os
import argparse
import jax.numpy as jnp
import recovar
import logging
import plotly.offline as py
import numpy as np
from scipy.spatial import distance_matrix

ftu = recovar.fourier_transform_utils.fourier_transform_utils(jnp)
logger = logging.getLogger(__name__)
py.init_notebook_mode()


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "result_dir",
        # dest="result_dir",
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
    parser.add_argument(
        "--zdim",
        type=int,
        help="Dimension of latent variable (a single int, not a list)",
    )

    parser.add_argument(
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=40,
        help="number of k-means clusters (default 40)",
    )

    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=6,
        dest="n_trajectories",
        help="number of trajectories to compute between k-means clusters (default 6)",
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)",
    )

    parser.add_argument(
        "--skip-centers",
        dest="skip_centers",
        action="store_true",
        help="whether to generate the volume of the k-means centers",
    )

    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="whether to use the adapative discretization scheme in reweighing to compute trajectory volumes",
    )

    parser.add_argument(
        "--n-vols-along-path",
        type=int,
        default=6,
        dest="n_vols_along_path",
        help="number of volumes to compute along each trajectory (default 6)",
    )

    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="quantile used for reweighting (default = 0.95)",
    )

    parser.add_argument("--Bfactor", type=float, default=0, help="0")

    parser.add_argument(
        "--n-bins",
        type=float,
        default=30,
        dest="n_bins",
        help="number of bins for reweighting",
    )

    parser.add_argument(
        "--n-std",
        metavar=float,
        type=float,
        default=None,
        help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)",
    )

    return parser


def pick_pairs(centers, n_pairs):
    # We try to pick some pairs that cover the latent space in some way.
    # This probably could be improved
    #
    # Pick some pairs that are far away from each other.
    pairs = []
    X = distance_matrix(centers[:, :], centers[:, :])

    for _ in range(n_pairs // 2):

        i_idx, j_idx = np.unravel_index(np.argmax(X), X.shape)
        X[i_idx, :] = 0
        X[:, i_idx] = 0
        X[j_idx, :] = 0
        X[:, j_idx] = 0
        pairs.append([i_idx, j_idx])

    # Pick some pairs that are far in the first few principal components.
    zdim = centers.shape[-1]
    max_k = np.min([n_pairs // 2, zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:, k])
        j_idx = np.argmin(centers[:, k])
        pairs.append([i_idx, j_idx])

    return pairs


def mkdir_safe(folder):
    os.makedirs(folder, exist_ok=True)


def main(
    result_dir,
    output_folder,
    zdim,
    n_clusters,
    n_paths,
    skip_umap,
    q,
    n_std,
    adaptive,
    B_factor,
    n_bins,
    n_vols_along_path,
    skip_centers,
):
    po = recovar.output.PipelineOutput(args.recovar_result_dir + "/")
    cryos = po.get("dataset")
    zdim = 10
    recovar.embedding.set_contrasts_in_cryos(cryos, po.get("contrasts")[zdim])
    zs = po.get("zs")[zdim]
    cov_zs = po.get("cov_zs")[zdim]
    noise_variance = po.get("noise_var_used")
    B_factor = 0
    n_bins = 30
    num_imgs = 1000

    new_zs = zs[::num_imgs]
    new_cov_zs = cov_zs[::num_imgs]

    print(noise_variance, B_factor, n_bins, new_zs, new_cov_zs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(
        args.result_dir,
        output_folder=args.outdir,
        zdim=args.zdim,
        n_clusters=args.n_clusters,
        n_paths=args.n_trajectories,
        skip_umap=args.skip_umap,
        q=args.q,
        n_std=args.n_std,
        adaptive=args.adaptive,
        B_factor=args.Bfactor,
        n_bins=args.n_bins,
        n_vols_along_path=args.n_vols_along_path,
        skip_centers=args.skip_centers,
    )
