'''Skeleton script'''

import argparse
import numpy as np
import sys, os
import pickle
import glob, re
import subprocess
import utils
log = utils.log 

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', help='Output directory')
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--mask", default=None)
    parser.add_argument('--gt-dir', help='Directory with gt models')
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
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    file_pattern = "*.mrc"
    files = glob.glob(os.path.join(args.gt_dir, file_pattern))
    gt_dir = sorted(files, key=natural_sort_key)

    
    # Compute FSC cdrgn
    if not os.path.exists('{}/{}/per_conf_fsc/fsc'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc'.format(args.o, args.method))
    if not os.path.exists('{}/{}/per_conf_fsc/fsc_no_mask'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc_no_mask'.format(args.o, args.method))

    if not os.path.exists('{}/{}/per_conf_fsc/fsc_flipped'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc_flipped'.format(args.o, args.method))
    if not os.path.exists('{}/{}/per_conf_fsc/fsc_flipped_no_mask'.format(args.o, args.method)):
        os.makedirs('{}/{}/per_conf_fsc/fsc_flipped_no_mask'.format(args.o, args.method))

    for ii in range(len(gt_dir)):
        if ii % args.fast != 0:
            continue
        if args.mask is not None:
            out_fsc = '{}/{}/per_conf_fsc/fsc/{}.txt'.format(args.o, args.method, ii)
        else:
            out_fsc = '{}/{}/per_conf_fsc/fsc_no_mask/{}.txt'.format(args.o, args.method, ii)
        
        cmd = 'python /scratch/gpfs/ZHONGE/mj7341/cryodrgn/cryodrgn/analysis_scripts/fsc.py {} {}/{}/per_conf_fsc/vols/aligned/vol_{:03d}.mrc -o {} --mask {}'.format(
                gt_dir[ii], args.o, args.method, ii, out_fsc, args.mask)
        print('cmd:',cmd)
        log(cmd)
        if os.path.exists(out_fsc) and not args.overwrite:
            log('FSC exists, skipping...')
        else:
            if not args.dry_run:
                subprocess.check_call(cmd, shell=True)
                
        if args.mask is not None:
            out_fsc = '{}/{}/per_conf_fsc/fsc_flipped/{}.txt'.format(args.o, args.method, ii)
        else:
            out_fsc = '{}/{}/per_conf_fsc/fsc_flipped_no_mask/{}.txt'.format(args.o, args.method, ii)
        
        cmd = 'python /scratch/gpfs/ZHONGE/mj7341/cryodrgn/cryodrgn/analysis_scripts/fsc.py {} {}/{}/per_conf_fsc/vols/flipped_aligned/vol_{:03d}.mrc -o {} --mask {}'.format(
                gt_dir[ii], args.o, args.method, ii, out_fsc, args.mask)
        print('cmd:',cmd)
        log(cmd)
        if os.path.exists(out_fsc) and not args.overwrite:
            log('FSC exists, skipping...')
        else:
            if not args.dry_run:
                subprocess.check_call(cmd, shell=True)

    # Summary statistics
    # No Flip
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

    # Flipped
    # Flipped
    if args.mask is not None:
        fsc = [np.loadtxt(x) for x in glob.glob('{}/{}/per_conf_fsc/fsc_flipped/*txt'.format(args.o, args.method))]
    else:
        fsc = [np.loadtxt(x) for x in glob.glob('{}/{}/per_conf_fsc/fsc_flipped_no_mask/*txt'.format(args.o, args.method))]

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