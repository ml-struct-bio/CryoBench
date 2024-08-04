import numpy as np
from cryodrgn import analysis

import pickle
import os, sys
import re
import argparse
import matplotlib.pyplot as plt
log = print
import glob
from matplotlib import colors

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--is_cryosparc", action="store_true", help="cryosparc or not")
    parser.add_argument("--num_classes", type=int, default=20, help="number of classes")
    parser.add_argument("--num_vols", type=int, default=100, help="number of classes")
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    parser.add_argument('--result-path', type=os.path.abspath, required=True, help='umap & latent folder before method name (e.g. /scratch/gpfs/ZHONGE/mj7341/cryosim/results/conf_het_v1/snr01)')
    parser.add_argument('--cryosparc_path', type=os.path.abspath, default='/scratch/gpfs/ZHONGE/mj7341/cryosparc/CS-new-mask-for-confhet-v1', help='cryosparc folder path')
    parser.add_argument("--cryosparc_job_num", type=str, help="cryosparc job number")

    return parser

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def gt_colors(num_imgs):
    c_lst = []
    c_num = 0
    for num_img in num_imgs:
        for i in range(num_img):
            c_lst.append(c_num)
        c_num += 1
    c_all = np.array(c_lst)
    return c_all

def plt_umap_labels():
    plt.xticks([])
    plt.yticks([])

def plot_methods(args, v, is_umap=True, use_axis=False):
    fig, ax = plt.subplots(figsize=(4,4))
    # Whole
    colorList = ['#3182bd',
        '#6baed6',
        '#9ecae1',
        '#e6550d',
        '#fd8d3c',
        '#fdae6b',
        '#fdd0a2',
        '#e377c2',
        '#f7b6d2',
        '#31a354',
        '#74c476',
        '#a1d99b',
        '#756bb1',
        '#9e9ac8',
        '#bcbddc',
        '#dadaeb'
    ]
    cmap = colors.ListedColormap(colorList[:args.num_vols])
    print('cmap.N:',cmap.N)

    num_imgs = [9076, 14378, 23547, 44366, 30647, 38500, 3915, 3980, 12740, 11975, 17988, 5001, 35367, 37448, 40540, 5772]
    c_lst = []
    c_num = 0
    for num_img in num_imgs:
        for i in range(num_img):
            c_lst.append(c_num)
        c_num += 1
    c_all = np.array(c_lst)
    plt.scatter(v[:,0], v[:,1], alpha=0.1, s=1, cmap=cmap, c=c_all, label=c_all, rasterized=True)

    if is_umap:
        if use_axis:
            plt.savefig(f"{args.o}/{args.method}/{args.method}_umap.pdf", bbox_inches='tight')
        else:
            plt_umap_labels()
            plt.savefig(f"{args.o}/{args.method}/{args.method}_umap_no_axis.pdf", bbox_inches='tight')
    else:
        if use_axis:
            plt.savefig(f"{args.o}/{args.method}/{args.method}_latent.pdf", bbox_inches='tight')
        else:
            plt_umap_labels()
            plt.savefig(f"{args.o}/{args.method}/{args.method}_latent_no_axis.pdf", bbox_inches='tight')
    plt.close()

def main(args):
    if args.is_cryosparc:    
        if args.method == '3dva':
            path = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_particles.cs"

            x = np.load(path)
            v = np.empty((len(x),3)) # component_0,1,2
            for i in range(3):
                v[:,i] = x[f'components_mode_{i}/value']
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)
            
            umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            if not os.path.exists(umap_path):
                umap_latent = analysis.run_umap(v) # v: latent space
                np.save(umap_path, umap_latent)
            else:
                umap_latent = np.load(umap_path)
            plot_methods(args, umap_latent, is_umap=True)

    else:
        if args.method == 'cryodrgn':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.49/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, is_umap=True)
        
        elif args.method == 'cryodrgn2':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.29/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, is_umap=True)
        
        elif args.method == 'drgnai_fixed':
            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, is_umap=True)
        
        elif args.method == 'drgnai_abinit':
            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == 'opus-dsd':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, is_umap=True)
        
        elif args.method == 'recovar':
            latent_path = os.path.join(args.result_path, args.method, "reordered_z.npy")
            latent_z = np.load(latent_path)
            umap_path = f"{args.result_path}/{args.method}/reordered_z_umap.npy"
            umap_pkl = analysis.run_umap(latent_z) 
            np.save(umap_path, umap_pkl)
            plot_methods(args, umap_pkl, is_umap=True)

if __name__ == "__main__":
    args = parse_args().parse_args()
    if not os.path.exists(args.o+'/'+args.method):
        os.makedirs(args.o+'/'+args.method)
    
    main(args)
    print('done!')

