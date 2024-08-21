
import pandas as pd
import numpy as np
from cryodrgn import analysis

import pickle
import os
import re
import argparse
import matplotlib.pyplot as plt
log = print

import glob, re

print('loaded')
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--cv_num", type=int, help="cv number to use for color",required=True)
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

def plt_umap_labels():
    plt.xticks([])
    plt.yticks([])

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

def plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04):
    
    x = np.repeat(dihedral_angles, 1000)
    xx = np.cos(x/180*np.pi)
    yy = np.sin(x/180*np.pi)

    xx_jittered = xx + np.random.randn(len(xx))*jitter
    yy_jittered = yy + np.random.randn(len(yy))*jitter
    
    plt.scatter(xx_jittered,yy_jittered,cmap='viridis',s=1,alpha=.1, c=labels_3dcls, vmin=0,vmax=args.num_classes)
    plt_umap_labels()
    plt.savefig(f"{args.o}/{args.method}/{args.method}_{args.num_classes}_{args.cv_num}.png", dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_methods(args, v, dihedral_angles, is_umap=True, use_axis=False):
    fig, ax = plt.subplots(figsize=(4,4))
    # Whole
    plot_dim = (0,1)
    c_all = np.repeat(dihedral_angles,1000)
    c = c_all
    plot_args = dict(alpha=.1, s=1, cmap='gist_rainbow', vmin=np.amin(dihedral_angles),vmax=np.amax(dihedral_angles))
    plt.scatter(v[:,plot_dim[0]], v[:,plot_dim[1]], c=c, **plot_args)
    use_axis = True
    plt.xlabel("UMap1")
    plt.ylabel("UMap2")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if use_axis:
        plt.savefig(f"{args.o}/{args.method}/{args.method}_umap_{args.cv_num}.png", dpi=1200, bbox_inches='tight')
    else:
        plt_umap_labels()
        plt.savefig(f"{args.o}/{args.method}/{args.method}_umap_no_axis_{args.cv_num}.png", dpi=1200, bbox_inches='tight')
    plt.close()
    

def main(args):
    if args.cv_num == 0:
        dihedral_angles = np.load('conf-het-2_CV_dihedral_distance.npy')[:,0] 
    if args.cv_num == 1:
        dihedral_angles = np.load('conf-het-2_CV_dihedral_distance.npy')[:,1] 
    print(dihedral_angles)


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
            plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04)

        elif args.method == '3dcls_abinit':
            path_for_label = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_final_particles.cs"
            path_for_model = f"{args.cryosparc_path}/{args.cryosparc_job_num}/"
            labels_3dcls, models_3dcls = parse_class_abinit_assignments(path_for_label, path_for_model, args.num_classes)
            plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04)
            
        elif args.method == '3dva':
            path = f"/mnt/ceph/users/mastore/cryobench_2024_ellen_zhong_colab/conf-het-2/pdbs/snr001/3dva/3dva_latents.npy"

            x = np.load(path)
            v = x 

            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)
            umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            if not os.path.exists(umap_path):
                umap_latent = analysis.run_umap(v) # v: latent space
                np.save(umap_path, umap_latent)
            else:
                umap_latent = np.load(umap_path)
            plot_methods(args, umap_latent, dihedral_angles, is_umap=True)

        elif args.method == '3dflex':
            path = f"/mnt/ceph/users/mastore/cryobench_2024_ellen_zhong_colab/conf-het-2/pdbs/snr001/3dflex/3dflex_latents.npy"
            x = np.load(path)

            v = np.empty((len(x),2))
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            latent_path = 'temp.txt'
            np.savetxt(latent_path, v)

            v [np.isinf(v)] = 0
            v [v > 10**30] = 0
            v = np.array(v,dtype=np.float32)
            umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            plot_methods(args, v, dihedral_angles, is_umap = False)
    
    else:
        print('chooseing method')
        if args.method == 'cryodrgn':            
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)
        
        elif args.method == 'cryodrgn2':
            latent_z = f"{args.result_path}/{args.method}/z.29.pkl"
            latent_z = open(latent_z, 'rb')
            latent_z = pickle.load(latent_z)
            umap_pkl = f"{args.result_path}/{args.method}/analyze.29/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)
        
        elif args.method == 'drgnai_fixed':
            umap_pkl = f"{args.result_path}/{args.method}/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)
        
        elif args.method == 'drgnai_abinit':
            umap_pkl = f"{args.result_path}/{args.method}/analysis_100/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)

        elif args.method == 'opus-dsd':
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"
    
            umap_pkl = open(umap_pkl, 'rb')
            umap_pkl = pickle.load(umap_pkl)    
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)
        
        elif args.method == 'recovar':
            print('loading recovar')
            latent_path = os.path.join(args.result_path, args.method, "reordered_z.npy")
            latent_z = np.load(latent_path)

            umap_path = f"{args.result_path}/{args.method}/reordered_z_umap.npy"
            umap_pkl = np.load(umap_path)
            plot_methods(args, umap_pkl, dihedral_angles, is_umap=True)
        

if __name__ == "__main__":
    args = parse_args().parse_args()
    if not os.path.exists(args.o+'/'+args.method):
        os.makedirs(args.o+'/'+args.method)
    
    main(args)

