import argparse
import numpy as np
import sys, os
import pickle
import glob, re
import subprocess
import utils
from cryodrgn import analysis
from cryodrgn.commands_utils.fsc import calculate_fsc
from cryodrgn import mrcfile
import torch
log = utils.log 

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='dir contains weights, config, z')
    parser.add_argument('-o', help='Output directory')
    parser.add_argument('--epoch', default=19, type=int, help="Number of training epochs")
    parser.add_argument('--num-vols', default=100, type=int, help="Number of total reconstructed volumes")
    parser.add_argument('--Apix', default=3.0, type=float)
    parser.add_argument('--num-imgs', default=1000, type=int, help="Number of images per model (structure)")
    parser.add_argument("--method", type=str, help="type of methods (Each method folder name)")
    parser.add_argument("--mask", default=None, help="Use mask to compute the masked metric")
    parser.add_argument('--gt-dir', help='Directory of gt volumes')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--fast',type=int, default=1)
    parser.add_argument('--cuda-device', default=0, type=int)
    return parser

def get_cutoff(fsc, t):
    w = np.where(fsc[:,1] < t)
    log(w)
    if len(w[0]) >= 1:
        x = fsc[:,0][w]
        return 1/x[0]
    else:
        return 2

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args):
    print('method:',args.method)
    if not os.path.exists(os.path.join(args.o, args.method, "per_conf_fsc")):
        os.makedirs(os.path.join(args.o, args.method, "per_conf_fsc"))

    weights = os.path.join(args.input_dir, f"weights.{args.epoch}.pkl")
    config = os.path.join(args.input_dir, "config.yaml")
    z_path = os.path.join(args.input_dir, f"z.{args.epoch}.pkl")

    gt = np.repeat(np.arange(0,args.num_vols),args.num_imgs)
    z = pickle.load(open(z_path,'rb'))
    assert len(gt) == len(z)

    z_lst = []
    z_mean_lst = [] 
    for i in range(args.num_vols):
        z_nth = z[i*args.num_imgs:(i+1)*args.num_imgs]
        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1,-1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)
    nearest_z_lst = []
    centers_ind_lst = []
    for i in range(args.num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind)
    nearest_z_array = np.array(nearest_z_lst)
    
    file_pattern = "*.mrc"
    files = glob.glob(os.path.join(args.gt_dir, file_pattern))
    gt_dir = sorted(files, key=natural_sort_key)
    # Generate cdrgn volumes
    if not os.path.exists('{}/{}/per_conf_fsc/vols'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/vols'.format(args.o, args.method))
    out_zfile = '{}/{}/per_conf_fsc/zfile.txt'.format(args.o, args.method)
    log(out_zfile)
    cmd = 'CUDA_VISIBLE_DEVICES={} cryodrgn eval_vol {} -c {} --zfile {} -o {}/{}/per_conf_fsc/vols --Apix {}'.format(
        args.cuda_device, weights, config, out_zfile, args.o, args.method, args.Apix)
    
    log(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        log('Z file exists, skipping...')
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    # Compute FSC cdrgn
    if not os.path.exists('{}/{}/per_conf_fsc/fsc'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc'.format(args.o, args.method))
    if not os.path.exists('{}/{}/per_conf_fsc/fsc_no_mask'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc_no_mask'.format(args.o, args.method))
    for ii in range(len(gt_dir)):
        if ii % args.fast != 0:
            continue
        if args.mask is not None:
            out_fsc = '{}/{}/per_conf_fsc/fsc/{}.txt'.format(args.o, args.method, ii)
        else:
            out_fsc = '{}/{}/per_conf_fsc/fsc_no_mask/{}.txt'.format(args.o, args.method, ii)

        vol_file = '{}/{}/per_conf_fsc/vols/vol_{:03d}.mrc'.format(args.o, args.method, ii)

        vol1 = mrcfile.parse_mrc(gt_dir[ii])[0]
        vol2 = mrcfile.parse_mrc(vol_file)[0]
        if os.path.exists(out_fsc) and not args.overwrite:
            log('FSC exists, skipping...')
        else:
            fsc_vals = calculate_fsc(torch.tensor(vol1), torch.tensor(vol2), args.mask)
            np.savetxt(out_fsc, fsc_vals)

    # Summary statistics
    fsc = [np.loadtxt(x) for x in glob.glob('{}/{}/per_conf_fsc/fsc/*txt'.format(args.o, args.method))]
    fsc143 = [get_cutoff(x,0.143) for x in fsc]
    fsc5 = [get_cutoff(x,.5) for x in fsc]
    log('cryoDRGN FSC=0.143')
    log('Mean: {}'.format(np.mean(fsc143)))
    log('Median: {}'.format(np.median(fsc143)))
    log('cryoDRGN FSC=0.5')
    log('Mean: {}'.format(np.mean(fsc5)))
    log('Median: {}'.format(np.median(fsc5)))


if __name__ == '__main__':
    main(parse_args().parse_args())
