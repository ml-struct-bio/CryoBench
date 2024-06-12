'''Add noise to a particle stack at a desired SNR'''

import argparse
import numpy as np
import sys, os
import matplotlib
import matplotlib.pyplot as plt

import glob, re

from cryodrgnai.cryodrgn import utils
from cryodrgnai.cryodrgn import mrc
from cryodrgnai.cryodrgn import dataset
from cryodrgnai.cryodrgn.lattice import EvenLattice

log = utils.log

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mrcs', help='Input particles folder')
    parser.add_argument('--snr', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--num-imgs', type=int, default=1000)
    parser.add_argument('--apix', type=float, default=1.5, help='apix of input mrc')
    parser.add_argument('--invert', action="store_true", help="invert data (mult by -1)")
    parser.add_argument('--mask', choices=('none','strict','circular'), help='Type of mask for computing signal variance')
    parser.add_argument('--mask-r', type=int, help='Radius for circular mask')
    parser.add_argument('--datadir', help='Optionally overwrite path to starfile .mrcs if loading from a starfile')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output particle stack')
    parser.add_argument('--out-png', type=os.path.abspath, help='Montage of first 9 projections')
    return parser

def plot_projections(out_png, imgs):
    matplotlib.use('Agg')
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i], cmap='gray')
    plt.savefig(out_png)
    plt.close()

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        log('Warning: {} already exists. Overwriting.'.format(out))

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args):
    assert (args.snr is None) != (args.sigma is None) # xor

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    if not os.path.exists(args.out_png):
        os.makedirs(args.out_png)
    
    file_pattern = "*.mrcs"
    mrcs_files = glob.glob(os.path.join(args.mrcs, file_pattern))
    sorted_mrcs_files = sorted(mrcs_files, key=natural_sort_key)

    # load particles
    total_particles = []
    for mrcs_file in sorted_mrcs_files:
        particles = dataset.load_particles(mrcs_file)
        total_particles.append(particles)
    total_particle = np.stack(total_particles, axis=0)
    total_particle = np.reshape(total_particle, (len(total_particle)*args.num_imgs, total_particle.shape[-1], total_particle.shape[-1]))
    print('total_particle:',total_particle.shape) # [100000, 128, 128]

    # log(particles.shape)
    Nimg, D, D = total_particle.shape
    std = np.std(total_particle)
    sigma = std/np.sqrt(args.snr)
    # # add noise
    for mrcs_file in sorted_mrcs_files:
        particles = dataset.load_particles(mrcs_file)

        log('Adding noise with std {}'.format(sigma))
        particles += np.random.normal(0,sigma,particles.shape)
        if args.invert:
            print('invert data!')
            particles *= -1
    
        # save particles
        outfile = os.path.join(args.o, mrcs_file.split('/')[-1])
        print('outfile:',outfile)
        mrc.write(outfile, particles.astype(np.float32), Apix=args.apix)

        if args.out_png:
            png_file = os.path.join(args.out_png, mrcs_file.split('/')[-1].split('.')[0]+'.png')
            plot_projections(png_file, particles[:9])

    # log('Done')

if __name__ == '__main__':
    main(parse_args().parse_args())
