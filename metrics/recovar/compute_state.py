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
        "--latent-points", type=os.path.abspath,
        required=True,
        help="path to latent points (.txt file)",
    )

    parser.add_argument(
        "--Bfactor",  type =float, default=0, help="0"
    )

    parser.add_argument(
        "--n-bins",  type =float, default=50, dest="n_bins",help="number of bins for reweighting"
    )


    parser.add_argument(
        "--zdim1",  action="store_true", help="Whether dimension 1 is used. This is an annoying corner case for np.loadtxt..."
    )

    return parser


def compute_state(args):

    po = o.PipelineOutput(args.result_dir + '/')
    target_zs = np.loadtxt(args.latent_points)
    output_folder = args.outdir

    if args.zdim1:
        zdim =1
        target_zs = target_zs[:,None]
    else:
        zdim = target_zs.shape[-1]
        if target_zs.ndim ==1:
            logger.warning("Did you mean to use --zdim1?")
            target_zs = target_zs[None]

    if zdim not in po.get('zs'):
        logger.error("z-dim not found in results. Options are:" + ','.join(str(e) for e in po.get('zs').keys()))

    cryos = po.get('dataset')
    embedding.set_contrasts_in_cryos(cryos, po.get('contrasts')[zdim])
    zs = po.get('zs')[zdim]
    cov_zs = po.get('cov_zs')[zdim]
    noise_variance = po.get('noise_var_used')
    n_bins = args.n_bins
    o.mkdir_safe(output_folder)    
    logger.addHandler(logging.FileHandler(f"{output_folder}/run.log"))
    logger.info(args)
    o.compute_and_save_reweighted(cryos, target_zs, zs, cov_zs, noise_variance, output_folder, args.Bfactor, n_bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    compute_state(args)