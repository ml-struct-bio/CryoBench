'''
Subsample CTF parameters from existing ctf.pkl
'''

import pickle
import os
import re
import numpy as np
import argparse
from cryodrgn import utils
from cryodrgn import ctf

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ctf_file', type=os.path.abspath, required=True, help='Input ctf.pkl to subsample')
    parser.add_argument('-o', '--out-ctf', type=os.path.abspath, required=True, help='Output ctf.pkl file')
    parser.add_argument('-N', type=int, required=True, help='Number of CTFs')
    parser.add_argument('--Apix', type=float, default=1.5, help='Overwrite pixel size (A/pix) (default: %(default)s)')
    parser.add_argument('-D', '--img-size', type=int, default=256, help='Overwrite image size (pixels) (default: %(default)s)')
    parser.add_arugment('--seed', type=int, default=np.random.randint(), help='Random seed')
    return parser

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def main(args):
    np.random.seed(args.seed)
    mkbasedir(args.out_ctf)

    data = utils.load_pkl(args.ctf_file)
    print(f'Loaded {data.shape} ctf.pkl')
    
    sampled_indices = np.random.choice(data.shape[0], size=args.N, replace=False)
    new_ctf = data[sampled_indices]
    if args.D:
        new_ctf[:,0] = args.img_size
    if args.Apix:
        new_ctf[:,1] = args.Apix

    ctf.print_ctf_params(new_ctf)
    utils.save_pkl(new_ctf, args.out_ctf)

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)
