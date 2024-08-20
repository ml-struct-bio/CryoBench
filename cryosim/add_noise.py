'''Add noise to a particle stack at a desired SNR'''

import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt

from cryodrgn import utils
from cryodrgn import mrcfile
from cryodrgn.lattice import EvenLattice

log = utils.log

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mrcs', nargs='+', help='Input particles (.mrcs, .star, or .txt)')
    parser.add_argument('--snr', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--invert', action="store_true", help="invert data (mult by -1)")
    parser.add_argument('--mask', choices=('none','strict','circular'), help='Type of mask for computing signal variance')
    parser.add_argument('--mask-r', type=int, help='Radius for circular mask')
    parser.add_argument('--datadir', help='Optionally overwrite path to starfile .mrcs if loading from a starfile')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output particle stack')
    parser.add_argument('--out-png')
    return parser

def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        log('Warning: {} already exists. Overwriting.'.format(out))

def main(args):
    assert (args.snr is None) != (args.sigma is None) # xor

    mkbasedir(args.o)
    warnexists(args.o)

    # load particles
    particles = mrcfile.parse_mrc(args.mrcs)[0]
    log(particles.shape)
    Nimg, D, D = particles.shape
    
    # compute noise variance
    if args.sigma:
        sigma = args.sigma
    else:
        Nstd = min(10000,Nimg)
        if args.mask == 'strict':
            mask = np.where(particles[:Nstd]>0)
            std = np.std(particles[mask])
        elif args.mask == 'circular':
            lattice = EvenLattice(D)
            mask = lattice.get_circular_mask(args.mask_r)
            mask = np.where(mask)[0] # convert from torch uint mask to array index
            std = np.std(particles[:Nstd].reshape(Nstd,-1)[:,mask])
        else:
            std = np.std(particles[:Nstd])
        sigma = std/np.sqrt(args.snr)

    # add noise
    log('Adding noise with std {}'.format(sigma))
    particles += np.random.normal(0,sigma,particles.shape)
    if args.invert:
        print('invert data!')
        particles *= -1
    # save particles
    mrcfile.write_mrc(args.o, particles.astype(np.float32))

    if args.out_png:
        plot_projections(args.out_png, particles[:9])

    log('Done')

if __name__ == '__main__':
    main(parse_args().parse_args())