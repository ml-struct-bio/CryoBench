import numpy as np
from recovar import output as o
from recovar import dataset
import os, argparse

from cryodrgn import analysis

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-result-dir", type=str, help="recovar result dir")
    parser.add_argument("--zdim", type=int, default=10)
    parser.add_argument('--overwrite',action='store_true')
    return parser

def main(args):
    pipeline_output = o.PipelineOutput(args.recovar_result_dir + '/')
    cryos = pipeline_output.get('lazy_dataset')
    zs = pipeline_output.get('zs')[args.zdim]
    zs_reordered = dataset.reorder_to_original_indexing(zs, cryos)
    
    latent_path = os.path.join(args.recovar_result_dir, 'reordered_z.npy')
    umap_path = os.path.join(args.recovar_result_dir, 'reordered_z_umap.npy')
    if os.path.exists(latent_path) and not args.overwrite:
        print('latent exists, skipping...')
    else:
        np.save(latent_path, zs_reordered)

    if os.path.exists(umap_path) and not args.overwrite:
        print('latent exists, skipping...')
    else:
        umap_pkl = analysis.run_umap(zs_reordered)
        np.save(umap_path, umap_pkl)

if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
    print('saved!')