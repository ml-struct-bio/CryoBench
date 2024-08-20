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
    parser.add_argument('--poses-dir', type=os.path.abspath, required=True, help='directory of poses from 3d_projected')
    parser.add_argument('--integrated-pose', type=os.path.abspath, required=True, help='path to save integrated pose')
    parser.add_argument('--mrcs', type=os.path.abspath, required=True, help='mrcs files from add_noise.py')
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
    # Integrate pose
    poses = os.listdir(args.poses_dir)
    lst = []
    for i in range(len(poses)):
        lst.append(os.path.join(args.poses_dir, poses[i]))
    poses_sorted_pkl = sorted(lst, key=natural_sort_key)
    x = [pickle.load(open(f,'rb')) for f in poses_sorted_pkl]
    if type(x[0]) == tuple: # pose tuples
        r = [xx[0] for xx in x]
        t = [xx[1] for xx in x]
        r2 = np.concatenate(r)
        t2 = np.concatenate(t)
        x2 = (r2,t2)
    else:
        x2 = np.concatenate(x)
    
    pickle.dump(x2, open(args.integrated_pose,'wb'))
    mrcs_lst = os.listdir(args.mrcs)
    extension = 'mrcs'
    mrcs_lst = [file for file in mrcs_lst if file.endswith('mrcs')]
    sorted_lst = sorted(mrcs_lst, key=natural_sort_key)
    print(len(sorted_lst))

    output_file = os.path.join(args.mrcs, 'sorted_particles.128.txt')
    with open(output_file, 'w') as output_file:
        for item in sorted_lst:
    #         print(item)
            output_file.write(item + '\n')

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)
