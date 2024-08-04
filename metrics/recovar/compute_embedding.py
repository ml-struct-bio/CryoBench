## something
import recovar.config 
import logging
import numpy as np
from recovar import output as o
from recovar import dataset, utils, latent_density, embedding
from scipy.spatial import distance_matrix
import pickle
import os, argparse
logger = logging.getLogger(__name__)

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
        "--zdim", type=int, help="Dimension of latent variable (a single int, not a list)"
    )

    parser.add_argument(
        "--n-clusters", metavar=int, default=40, help="mask options: from_halfmaps (default), input, sphere, none"
    )

    parser.add_argument(
        "--n-trajectories", type=int, default=6, dest="n_trajectories", help="how many trajectories to compute between k-means clusters"
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)"
    )

    parser.add_argument(
        "--q", metavar=float, default=None, help="quantile used for reweighting (default = 0.95)"
    )

    parser.add_argument(
        "--n-std", metavar=float, default=None, help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)"
    )

    return parser


def compute_embedding(recovar_result_dir):
    import time
    results = o.load_results_new(recovar_result_dir + '/')
    cryos = dataset.load_dataset_from_args(results['input_args'])
    options = utils.make_algorithm_options(results['input_args'])

    gpu_memory = utils.get_gpu_memory_total()
    # Compute embeddings
    zs = {}; cov_zs = {}; est_contrasts = {} 
    for zdim in zdims:
        z_time = time.time()
        zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(
            results['means']['combined'], results['u']['rescaled'], results['s']['rescaled'], zdim,
                                                                results['cov_noise'], cryos, results['volume_mask'], gpu_memory, 'linear_interp',
                                                                contrast_grid = None, contrast_option = options['contrast'] )
        logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")

    return zs, cov_zs, est_contrasts


def compute_embedding_and_save(recovar_result_dir):
    zs, cov_zs, est_contrasts = compute_embedding(recovar_result_dir)

    

    # if zdim is None and len(results['zs']) > 1:
    #     logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    #     raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    
    # elif zdim is None:
    #     zdim = list(results['zs'].keys())[0]
    #     logger.info(f"using zdim={zdim}")

    # if q is None and n_std is None:
    #     likelihood_threshold = latent_density.get_log_likelihood_threshold( k = zdim)
    # elif q is not None and n_std is not None:
    #     logger.error("either q or n_std should be set, not both")
    # elif n_std is not None:
    #     likelihood_threshold = n_std
    # else: 
    #     likelihood_threshold = latent_density.get_log_likelihood_threshold(q=q, k = zdim)

    # if output_folder is None:
    #     output_folder = recovar_result_dir + '/output/'

    # # if zdim is None and len(results['zs']) > 1:
    # #     logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

    # if zdim not in results['zs']:
        
    #     logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in results['zs'].keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_embedding(args.result_dir,
                    output_folder = args.outdir, zdim=args.zdim,
                    n_clusters = args.n_clusters, n_paths= args.n_trajectories, 
                    skip_umap = args.skip_umap, q=args.q, n_std=args.n_std )
