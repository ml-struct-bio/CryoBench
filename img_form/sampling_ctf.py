'''
Sample CTF
'''

import pickle
import os
import re
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctf-dir', type=os.path.abspath, required=True, help='directory to save sampled ctfs')
    parser.add_argument('--ctf-file', type=os.path.abspath, required=True, help='experimental ctf that we will sample from')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='path to save the integrated ctf file')
    parser.add_argument('--N', type=int, default=100, help='Number of models')
    parser.add_argument('--apix', type=float, default=1.5, help='Number of models')
    parser.add_argument('--img-size', type=int, default=256, help='Size of image')
    parser.add_argument('--num-ctfs', type=int, default=1000, help='Number of CTFs per model (= the number of image)')
    return parser

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args):
    if not os.path.exists(args.ctf_dir):
        os.makedirs(args.ctf_dir)
    with open(args.ctf_file, 'rb') as file:
        data = pickle.load(file)
    
    for i in range(args.N):
        sampled_indices = np.random.choice(data.shape[0], size=args.num_ctfs, replace=False)
        new_ctf = data[sampled_indices]
        new_ctf[:,0] = args.img_size #128.0
        new_ctf[:,1] = args.apix
        new_ctf[:,5] = 300.0
        save_file_path = os.path.join(args.ctf_dir, 'ctfs_'+str(i)+'.pkl')
        with open(save_file_path, 'wb') as file:
            pickle.dump(new_ctf, file)

    # generate integrated CTF file
    ctfs = os.listdir(args.ctf_dir)
    lst = []
    for i in range(len(ctfs)):
        lst.append(os.path.join(args.ctf_dir,ctfs[i]))
    ctfs_sorted_pkl = sorted(lst, key=natural_sort_key)

    x = [pickle.load(open(f,'rb')) for f in ctfs_sorted_pkl]
    if type(x[0]) == tuple: # pose tuples
        r = [xx[0] for xx in x]
        t = [xx[1] for xx in x]
        r2 = np.concatenate(r)
        t2 = np.concatenate(t)
        x2 = (r2,t2)
    else:
        x2 = np.concatenate(x)
    out_filenames = args.o 
    pickle.dump(x2, open(out_filenames,'wb'))

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)
