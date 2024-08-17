import argparse
import numpy as np
import os
import pickle
import glob, re
import subprocess
import utils
from cryodrgn import analysis
from cryodrgn.commands_utils.fsc import calculate_fsc
from cryodrgn.source import ImageSource

log = utils.log 

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', help='dir contains weights, config, z')
    parser.add_argument('-o', help='Output directory')
    parser.add_argument('--apix', default=3.0)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num-vols', default=100, type=int)
    parser.add_argument('--num-imgs', default=1000, type=int)
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--mask", default=None)
    parser.add_argument('--gt-dir', help='Directory of gt volumes')
    parser.add_argument('--cuda-device', default=0, type=int)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--fast',type=int, default=1)
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

    gt = np.repeat(np.arange(0,args.num_vols),args.num_imgs)
    z_path = os.path.join(args.input_dir, "out", f"conf.{args.epoch}.pkl")
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
    gt_dir = sorted(files, key=natural_sort_key)    # Generate volumes
    
    if not os.path.exists('{}/{}/per_conf_fsc/vols'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/vols'.format(args.o, args.method))
    out_zfile = '{}/{}/per_conf_fsc/zfile.txt'.format(args.o, args.method)
    log(out_zfile)

    np.savetxt(out_zfile, nearest_z_array)
    cmd = 'CUDA_VISIBLE_DEVICES={} drgnai analyze {} --volume-metrics --z-values-txt {} --epoch {} --invert -o {}/{}/per_conf_fsc/vols --Apix {}'.format(
                args.cuda_device, args.input_dir, out_zfile, args.epoch, args.o, args.method, args.apix)
    
    log(cmd)
    if not args.dry_run:
        subprocess.check_call(cmd, shell=True)

    # Compute FSC
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

        vol_name = "vol_{:03d}.mrc".format(ii)
        vol_file = '{}/{}/per_conf_fsc/vols/vol_{:03d}.mrc'.format(args.o, args.method, ii)

        vol1 = ImageSource.from_file(gt_dir[ii])
        vol2 = ImageSource.from_file(vol_file)
        if os.path.exists(out_fsc) and not args.overwrite:
            log('FSC exists, skipping...')
        else:
            fsc_vals = calculate_fsc(vol1.images(), vol2.images(), args.mask)
            np.savetxt(out_fsc, fsc_vals)

    # Summary statistics
    if args.mask is not None:
        fsc = [np.loadtxt(x) for x in glob.glob('{}/{}/per_conf_fsc/fsc/*txt'.format(args.o, args.method))]
    else:
        fsc = [np.loadtxt(x) for x in glob.glob('{}/{}/per_conf_fsc/fsc_no_mask/*txt'.format(args.o, args.method))]

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
