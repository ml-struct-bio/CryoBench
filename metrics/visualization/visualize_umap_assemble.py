import numpy as np
from cryodrgn import analysis

import pickle
import os
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
    parser.add_argument("--num_imgs", type=int, help="number of images")
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

def parse_class_assignments(path_for_label, path_for_model, K):
    # for labels
    cs = np.load(path_for_label)
    keys = ["alignments_class3D_{}/class_posterior".format(i) for i in range(K)]
    classes = [[x[k] for k in keys] for x in cs]
    classes = np.asarray(classes)
    class_id = classes.argmax(axis=1)
    
    # for models
    model_lst = []
    path_for_model.split('/')[-2]
    cs_model_path = [path_for_model+path_for_model.split('/')[-2]+"_passthrough_particles_class_{}.cs".format(i) for i in range(K)]
    for i in range(K):
        cs_model = np.load(cs_model_path[i])
        for j in range(len(cs_model)):
            num_vol = cs_model[j][1].split(b'_')[1].decode('utf-8')
            model_lst.append(num_vol)
    model_id = np.asarray(model_lst)
            
    return class_id, model_id

def parse_class_abinit_assignments(path_for_label, path_for_model, K):
    # for labels
    cs = np.load(path_for_label, allow_pickle=True)
    keys = ["alignments_class_{}/class_posterior".format(i) for i in range(K)]
    classes = [[x[k] for k in keys] for x in cs]
    classes = np.asarray(classes)
    class_id = classes.argmax(axis=1)
    
    # for models
    model_lst = []
    path_for_model.split('/')[-2]
    cs_model_path = [path_for_model+path_for_model.split('/')[-2]+"_class_{:02d}_final_particles.cs".format(i) for i in range(K)]
    for i in range(K):
        cs_model = np.load(cs_model_path[i])
        for j in range(len(cs_model)):
            num_vol = cs_model[j][1].split(b'_')[1].decode('utf-8')
            model_lst.append(num_vol)
    model_id = np.asarray(model_lst)
            
    return class_id, model_id

def plot_3dcls(args, dihedral_angles, labels_3dcls, jitter = 0.04):
    x = np.repeat(dihedral_angles, 1000)
    xx = np.cos(x/180*np.pi)
    yy = np.sin(x/180*np.pi)
    xx_jittered = xx + np.random.randn(len(xx))*jitter
    yy_jittered = yy + np.random.randn(len(yy))*jitter
    
    # colorList = plt.cm.tab20.colors
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
    cmap = colors.ListedColormap(colorList[:args.num_classes])

    plt.scatter(xx_jittered,yy_jittered,cmap=cmap,s=1,alpha=.1, c=labels_3dcls, vmin=0,vmax=args.num_classes, rasterized=True)
    plt_umap_labels()
    plt.savefig(f"{args.o}/{args.method}/{args.method}_{args.num_classes}.pdf", dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close()

def plt_umap_labels():
    plt.xticks([])
    plt.yticks([])
    
def plot_methods_only_whole(args, v, is_umap=True, use_axis=False):
    fig, ax = plt.subplots(figsize=(4,4))
    # Whole
    plot_dim = (0,1)
    # Dihedral angles
    # colorList = plt.cm.tab20c.colors
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
    norm = colors.BoundaryNorm([-1]+[i for i in range(cmap.N)], cmap.N)
    c_all = np.repeat(np.arange(args.num_vols),1000)
    c = c_all
    plt.scatter(v[:,0], v[:,1], alpha=0.5, s=1, cmap=cmap, norm=norm, c=c, label=c, rasterized=True)

    if is_umap:
        if use_axis:
            plt.savefig(f"{args.o}/{args.method}/{args.method}_umap.pdf", dpi=1200, bbox_inches='tight')
        else:
            plt_umap_labels()
            plt.savefig(f"{args.o}/{args.method}/{args.method}_umap_no_axis.pdf", dpi=1200, bbox_inches='tight')
    else:
        if use_axis:
            plt.savefig(f"{args.o}/{args.method}/{args.method}_latent.pdf", dpi=1200, bbox_inches='tight')
        else:
            plt_umap_labels()
            plt.savefig(f"{args.o}/{args.method}/{args.method}_latent_no_axis.pdf", dpi=1200, bbox_inches='tight')
    plt.close()

def main(args):
    assemble_info = np.arange(0,16)
    if args.is_cryosparc:
        if args.method == '3dcls':
            file_pattern = "*.mrc"
            files = glob.glob(os.path.join(args.result_path, args.method, 'cls_'+ str(args.num_classes) ,file_pattern))
            pred_dir = sorted(files, key=natural_sort_key)
            cryosparc_num = pred_dir[0].split('/')[-1].split('.')[0].split('_')[3]
            cryosparc_job = pred_dir[0].split('/')[-1].split('.')[0].split('_')[0]
            print('cryosparc_num:',cryosparc_num)
            print('cryosparc_job:',cryosparc_job)
            path_for_label = f"{args.cryosparc_path}/{cryosparc_job}/{cryosparc_job}_{cryosparc_num}_particles.cs"
            path_for_model = f"{args.cryosparc_path}/{cryosparc_job}/"
            labels_3dcls, models_3dcls = parse_class_assignments(path_for_label, path_for_model, args.num_classes)
            plot_3dcls(args, assemble_info, labels_3dcls, jitter=0.01)

        elif args.method == '3dcls_abinit':
            path_for_label = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_final_particles.cs"
            path_for_model = f"{args.cryosparc_path}/{args.cryosparc_job_num}/"
            labels_3dcls, models_3dcls = parse_class_abinit_assignments(path_for_label, path_for_model, args.num_classes)
            print('labels_3dcls:',labels_3dcls, labels_3dcls.shape)
            plot_3dcls(args, assemble_info, labels_3dcls, jitter = 0.01)
            
            
        elif args.method == '3dva':
            path = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_particles.cs"

            x = np.load(path)
            v = np.empty((len(x),3)) # component_0,1,2
            for i in range(3):
                v[:,i] = x[f'components_mode_{i}/value']
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)

            # UMap
            umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            if not os.path.exists(umap_path):
                umap_latent = analysis.run_umap(v) # v: latent space
                np.save(umap_path, umap_latent)
            else:
                umap_latent = np.load(umap_path)
            plot_methods_only_whole(args, umap_latent, is_umap=True)

        elif args.method == '3dflex':
            path = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_latents_011200.cs"
            x = np.load(path)

            v = np.empty((len(x),2))
            for i in range(2):
                v[:,i] = x[f'components_mode_{i}/value']
            
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)

            plot_methods_only_whole(args, v, is_umap=True)

            # # UMap
            # umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            # if not os.path.exists(umap_path):
            #     umap_latent = analysis.run_umap(v) # v: latent space
            #     np.save(umap_path, umap_latent)
            # else:
            #     umap_latent = np.load(umap_path)
            # umap_latent = analysis.run_umap(v) # v: latent space
            # plot_methods_only_whole(args, v, axial_angles, dihedral_angles, is_umap=True)
    
    else:
        if args.method == 'cryodrgn':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods_only_whole(args, umap_pkl, is_umap=True)
        
        elif args.method == 'cryodrgn2':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.29/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods_only_whole(args, umap_pkl, is_umap=True)
        
        elif args.method == 'drgnai_fixed':

            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            
            plot_methods_only_whole(args, umap_pkl, is_umap=True)
        
        elif args.method == 'drgnai_abinit':
            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods_only_whole(args, umap_pkl, is_umap=True)

        elif args.method == 'opus-dsd':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    

            plot_methods_only_whole(args, umap_pkl, is_umap=True)
        
        elif args.method == 'recovar':
            latent_path = os.path.join(args.result_path, args.method, "reordered_z.npy")
            latent_z = np.load(latent_path)

            umap_path = f"{args.result_path}/{args.method}/reordered_z_umap.npy"
            umap_pkl = analysis.run_umap(latent_z) # v: latent space
            np.save(umap_path, umap_pkl)

            plot_methods_only_whole(args, umap_pkl, is_umap=True)

if __name__ == "__main__":
    args = parse_args().parse_args()
    if not os.path.exists(args.o+'/'+args.method):
        os.makedirs(args.o+'/'+args.method)
    
    main(args)
    print('done!')
