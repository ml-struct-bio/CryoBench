import argparse
import numpy as np
import os
import glob, re
import subprocess
import utils
from cryodrgn import analysis
from cryodrgn import mrcfile
from cryodrgn.commands_utils.fsc import calculate_fsc
from cryodrgn.source import ImageSource

log = utils.log 

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', help='Output directory')
    parser.add_argument('--num-vols', default=16, type=int)
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--mask", default=None)
    parser.add_argument('--gt-dir', help='Directory of gt volumes')
    parser.add_argument('--cryosparc-dir', help='Directory of cryosparc')
    parser.add_argument('--cryosparc-job', help='job number of of cryosparc')
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
    if not os.path.exists(os.path.join(args.o, args.method, "per_conf_fsc", "vols")):
        os.makedirs(os.path.join(args.o, args.method, "per_conf_fsc", "vols"))
    # 3DVA
    # weights z_ik
    work_dir = os.path.join(args.cryosparc_dir, args.cryosparc_job)
    cs_path = os.path.join(work_dir, f"{args.cryosparc_job}_particles.cs")
    map_mrc_path = os.path.join(work_dir, f"{args.cryosparc_job}_map.mrc")
    # ref
    v_0 = mrcfile.parse_mrc(map_mrc_path)[0]
    x = np.load(cs_path)
    component_mrc_path = os.path.join(work_dir, f"{args.cryosparc_job}_component_0.mrc")
    v_k1 = mrcfile.parse_mrc(component_mrc_path)[0] # [128 128 128]

    component_mrc_path = os.path.join(work_dir, f"{args.cryosparc_job}_component_1.mrc")
    v_k2 = mrcfile.parse_mrc(component_mrc_path)[0] # [128 128 128]

    component_mrc_path = os.path.join(work_dir, f"{args.cryosparc_job}_component_2.mrc")
    v_k3 = mrcfile.parse_mrc(component_mrc_path)[0] # [128 128 128]
    
    num_img_for_centers = 0    
    centers_inds_1 = 0
    centers_inds_2 = 0
    centers_inds_3 = 0

    # Ribosembly number of images per structure (total 16 structures)
    num_imgs = [9076, 14378, 23547, 44366, 30647, 38500, 3915, 3980, 12740, 11975, 17988, 5001, 35367, 37448, 40540, 5772]
    for i in range(args.num_vols):
        components_1 = x[f'components_mode_0/value']
        components_2 = x[f'components_mode_1/value']
        components_3 = x[f'components_mode_2/value']
        
        z_1 = components_1[sum(num_imgs[:i]):sum(num_imgs[:i+1])].reshape(-1,1)
        z_2 = components_2[sum(num_imgs[:i]):sum(num_imgs[:i+1])].reshape(-1,1)
        z_3 = components_3[sum(num_imgs[:i]):sum(num_imgs[:i+1])].reshape(-1,1)

        z1_nth_avg = z_1.mean(axis=0)
        z1_nth_avg = z1_nth_avg.reshape(1,-1)
        
        z2_nth_avg = z_2.mean(axis=0)
        z2_nth_avg = z2_nth_avg.reshape(1,-1)
        
        z3_nth_avg = z_3.mean(axis=0)
        z3_nth_avg = z3_nth_avg.reshape(1,-1)
        nearest_z1, centers_ind1 = analysis.get_nearest_point(z_1, z1_nth_avg)
        nearest_z2, centers_ind2 = analysis.get_nearest_point(z_2, z2_nth_avg)
        nearest_z3, centers_ind3 = analysis.get_nearest_point(z_3, z3_nth_avg)
        
        centers_inds_1 = centers_ind1 + num_img_for_centers
        centers_inds_2 = centers_ind2 + num_img_for_centers
        centers_inds_3 = centers_ind3 + num_img_for_centers
        
        num_img_for_centers += num_imgs[i]
        
        nearest_z_array_1 = components_1[centers_inds_1]
        nearest_z_array_2 = components_2[centers_inds_2]
        nearest_z_array_3 = components_3[centers_inds_3]
        
        vol = v_0 + (nearest_z_array_1*(v_k1) + nearest_z_array_2*(v_k2) + nearest_z_array_3*(v_k3))
        vol_name = "vol_{:03d}.mrc".format(i)
        mrcfile.write(f'{args.o}/{args.method}/per_conf_fsc/vols/{vol_name}', vol.astype(np.float32))

    file_pattern = "*.mrc"
    files = glob.glob(os.path.join(args.gt_dir, file_pattern))
    gt_dir = sorted(files, key=natural_sort_key)

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

        vol_name = "vol_{:03d}.mrc".format(ii)
        vol_file = '{}/{}/per_conf_fsc/vols/{}'.format(args.o, args.method, vol_name)

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
