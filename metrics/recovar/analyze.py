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
        "--n-clusters", dest= "n_clusters", type=int, default=40, help="number of k-means clusters (default 40)"
    )

    parser.add_argument(
        "--n-trajectories", type=int, default=6, dest="n_trajectories", help="number of trajectories to compute between k-means clusters (default 6)"
    )

    parser.add_argument(
        "--skip-umap",
        dest="skip_umap",
        action="store_true",
        help="whether to skip u-map embedding (can be slow for large dataset)"
    )

    parser.add_argument(
        "--skip-centers",
        dest="skip_centers",
        action="store_true",
        help="whether to generate the volume of the k-means centers"
    )


    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="whether to use the adapative discretization scheme in reweighing to compute trajectory volumes"
    )

    parser.add_argument(
        "--n-vols-along-path", type=int, default=6, dest="n_vols_along_path", help="number of volumes to compute along each trajectory (default 6)"
    )


    parser.add_argument(
        "--q",  type =float, default=None, help="quantile used for reweighting (default = 0.95)"
    )


    parser.add_argument(
        "--Bfactor",  type =float, default=0, help="0"
    )

    parser.add_argument(
        "--n-bins",  type =float, default=30, dest="n_bins",help="number of bins for reweighting"
    )


    parser.add_argument(
        "--n-std", metavar=float, type=float, default=None, help="number of standard deviations to use for reweighting (don't set q and this parameter, only one of them)"
    )

    return parser


def analyze(recovar_result_dir, output_folder = None, zdim = 4, n_clusters = 40, n_paths = 6, skip_umap = False, q = None, n_std = None, compute_reproj = False, adaptive= False, B_factor=0, n_bins=30, n_vols_along_path = 6, skip_centers = False):


    po = o.PipelineOutput(recovar_result_dir + '/')

    if zdim is None and len(po.get('zs')) > 1:
        logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
        raise Exception("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")
    
    elif zdim is None:
        zdim = list(po.get('zs').keys())[0]
        logger.info(f"using zdim={zdim}")

    if q is None and n_std is None:
        likelihood_threshold = latent_density.get_log_likelihood_threshold( k = zdim)
    elif q is not None and n_std is not None:
        logger.error("either q or n_std should be set, not both")
    elif n_std is not None:
        likelihood_threshold = n_std
    else: 
        likelihood_threshold = latent_density.get_log_likelihood_threshold(q=q, k = zdim)
        logger.info(f"using input q={q} from input ")

    if output_folder is None:
        output_folder = recovar_result_dir + '/output/analysis_' + str(zdim)  + '/'


    # if zdim is None and len(po.get('zs']) > 1:
    #     logger.error("z-dim is not set, and multiple zs are found. You need to specify zdim with e.g. --z-dim=4")

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    cryos = po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim])

    zs = po.get('zs')[zdim]
    cov_zs = po.get('cov_zs')[zdim]
    noise_variance = po.get('noise_var_used')
    B_factor = args.Bfactor
    n_bins = args.n_bins


    output_folder_kmeans = output_folder + '/' #+ '/kmeans'+'_'+ str(n_clusters) + '/'    
    o.mkdir_safe(output_folder_kmeans)    
    logger.addHandler(logging.FileHandler(f"{output_folder_kmeans}/run.log"))
    logger.info(args)

    centers, labels = o.kmeans_analysis(output_folder_kmeans, zs, n_clusters = n_clusters)
    output_folder_kmeans_centers = output_folder_kmeans + '/centers/'
    o.mkdir_safe(output_folder_kmeans_centers)    
    if not skip_centers:
        o.compute_and_save_reweighted(cryos, centers, zs, cov_zs, noise_variance, output_folder_kmeans_centers, B_factor, n_bins)

    if (not skip_umap) and (zdim > 1):
        mapper = o.umap_latent_space(zs)
        o.mkdir_safe(output_folder + '/umap/')    
        utils.pickle_dump(mapper.embedding_, output_folder + '/umap/embedding.pkl')
        from cryodrgn import analysis
        _, kmeans_ind = analysis.get_nearest_point(zs, centers)
        o.plot_umap(output_folder + '/umap/', mapper.embedding_, mapper.embedding_[kmeans_ind])


    kmeans_result = { 'centers' : centers, 'labels': labels  }
    pickle.dump(kmeans_result, open(output_folder_kmeans + 'centers.pkl', 'wb'))

    if zdim > 1:
        # Recompute density
        logger.info("recomputing density. Take out?")
        density, latent_space_bounds  = latent_density.compute_latent_space_density(zs, cov_zs, pca_dim_max = np.min([4,zs.shape[-1]]), num_points = 50, density_option = 'kde')
        po.params['density'] = density

        pairs = pick_pairs(centers, n_paths)
        for pair_idx in range(len(pairs)):
            pair = pairs[pair_idx]
            z_st = centers[pair[0],:]
            z_end = centers[pair[1],:]

            path_folder = output_folder_kmeans + 'path' + str(pair_idx) + '/'        
            o.mkdir_safe(path_folder)
            print("HERE")
            full_path, subsampled_path = o.make_trajectory_plots_from_results(po, zdim, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, compute_reproj = compute_reproj, likelihood_threshold = likelihood_threshold, adaptive = adaptive)        
            logger.info(f"path {pair_idx} done")
            o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins)

    else:
        path_folder = output_folder_kmeans + 'path' + str(0) + '/'        
        o.mkdir_safe(path_folder)
        q = 0.03
        pairs = np.percentile(po.get('zs')[zdim], [q, 100-q])
        z_st = pairs[0]
        z_end = pairs[1]
        # n_vols_along_path = 80
        # z_points = np.linspace(z_st, z_end, n_vols_along_path)
        # pairs = [ [z_points[0], z_points[40-1]], [z_points[40], z_points[80-1]] ]
        subsampled_path = np.linspace(z_st, z_end, n_vols_along_path)[:,None]
        o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins, save_all_estimates = False)

    # for pair_idx in range(len(pairs)):
    #     pair = pairs[pair_idx]
    #     z_st = centers[pair[0],:]
    #     z_end = centers[pair[1],:]

    #     path_folder = output_folder_kmeans + 'path' + str(pair_idx) + '/'        
    #     o.mkdir_safe(path_folder)
    #     print("HERE")
    #     full_path, subsampled_path = o.make_trajectory_plots_from_results(po, zdim, path_folder, cryos = cryos, z_st = z_st, z_end = z_end, gt_volumes= None, n_vols_along_path = n_vols_along_path, plot_llh = False, compute_reproj = compute_reproj, likelihood_threshold = likelihood_threshold, adaptive = adaptive)        
    #     logger.info(f"path {pair_idx} done")
    #     o.compute_and_save_reweighted(cryos, subsampled_path, zs, cov_zs, noise_variance, path_folder, B_factor, n_bins)

    kmeans_res = { 'centers': centers.tolist(), 'pairs' : pairs }
    pickle.dump(kmeans_res, open(output_folder_kmeans + 'trajectory_endpoints.pkl', 'wb'))

def pick_pairs(centers, n_pairs):
    # We try to pick some pairs that cover the latent space in some way.
    # This probably could be improved
    #     
    # Pick some pairs that are far away from each other.
    pairs = []
    X = distance_matrix(centers[:,:], centers[:,:])

    for _ in range(n_pairs//2):

        i_idx,j_idx = np.unravel_index(np.argmax(X), X.shape)
        X[i_idx, :] = 0 
        X[:, i_idx] = 0 
        X[j_idx, :] = 0 
        X[:, j_idx] = 0 
        pairs.append([i_idx, j_idx])

    # Pick some pairs that are far in the first few principal components.
    zdim = centers.shape[-1]
    max_k = np.min([n_pairs//2, zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:,k])
        j_idx = np.argmin(centers[:,k])
        pairs.append([i_idx, j_idx])


    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    analyze(args.result_dir, output_folder = args.outdir, zdim=  args.zdim, n_clusters = args.n_clusters, n_paths= args.n_trajectories, skip_umap = args.skip_umap, q=args.q, n_std=args.n_std , adaptive = args.adaptive, B_factor = args.Bfactor, n_bins = args.n_bins, n_vols_along_path = args.n_vols_along_path, skip_centers = args.skip_centers)
