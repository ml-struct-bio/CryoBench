import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import pyro.distributions as dist
import kornia.geometry.conversions as conversions
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def mean_rotation(rot_matrices):
    qs = conversions.rotation_matrix_to_quaternion(torch.from_numpy(rot_matrices)).numpy()
    average_q = averageQuaternions(qs)
    mean_rot_matrix = conversions.quaternion_to_rotation_matrix(torch.from_numpy(average_q)).numpy()
    return mean_rot_matrix


def estimate_global_rotation(rotations_1, rotations_2):
    '''
    Estimate the global rotation between two sets of rotations.
    
    Estimates R_global under the model:
        R_error_i ~ p(R)
        R_2_i = R_global R_error_i R_1_i
    '''
    relative_poses = rotations_2 @ np.transpose(rotations_1, (0,2,1))
    global_rotation = mean_rotation(relative_poses)
    return global_rotation


def sample_small_rotation(n_rotations, concentration=10):
    quaternions = dist.ProjectedNormal(concentration*torch.tensor([0,0,0,1.0])).sample((n_rotations,))
    rotations = R.from_quat(quaternions.numpy()).as_matrix()
    return rotations

def align_rot_median(rotA, rotB, N, do_right_mult_R=True):
    best_medse = np.inf
    best_rotation = None
    for i in range(N):
        if do_right_mult_R:
            mean_rot = np.dot(rotB[i].T, rotA[i])
            rotB_hat = np.matmul(rotB, mean_rot)
        else:
            mean_rot = np.dot(rotA[i],rotB[i].T)
            rotB_hat = np.matmul(mean_rot,rotB)
        medse = np.median(np.sum((rotB_hat - rotA) ** 2, axis=(1, 2)))
        if medse < best_medse:
            best_medse = medse
            best_rotation = mean_rot

    # align B into A's reference frame
    if do_right_mult_R:
        rotA_hat = np.matmul(rotA, mean_rot.T).astype(rotA.dtype)
        rotB_hat = np.matmul(rotB, mean_rot).astype(rotB.dtype)
    else:
        rotA_hat = np.matmul(mean_rot.T,rotA).astype(rotA.dtype)
        rotB_hat = np.matmul(mean_rot,rotB).astype(rotB.dtype)
    dist2 = np.sum((rotB_hat - rotA) ** 2, axis=(1, 2))

    return rotA_hat, rotB_hat, best_rotation, dist2, best_medse


def presets():
    figsize = (2,2)
    markersize=8
    elinewidth=1
    markerscale=1.5

    color_map = {
        'random': 'gray',
        'cryodrgn': '#6190e6',
        'drgnai-fixed': '#88B4E6',
        'opusdsd_mu': '#b0e0e6',
        'cryosparc-3dflex': '#98fb98',
        'cryosparc-3dva': '#f4a460',
        'recovar': '#f08080',
        'cryodrgn2': '#7b68ee',
        'drgnai-abinit': '#a569bd',
    'cryosparc-3d': '#d8bfd8',
    'cryosparc-3dabinit': '#da70d6',
    #    'G.T': '#bfbfbf'
    }


    label_map = {
        'random': 'Random Shuffle',
        'cryodrgn': 'CryoDRGN',
        'drgnai-fixed': 'DRGN-AI-fixed',
        'opusdsd_mu': 'Opus-DSD',
        'cryosparc-3dflex': '3DFlex',
        'cryosparc-3dva': '3DVA',
        'recovar': 'RECOVAR',
        'cryosparc-3d': '3D Class',
        'cryodrgn2': 'CryoDRGN2',
        'drgnai-abinit': 'DRGN-AI',
        'cryosparc-3dabinit': '3D Class abinit',
        }


    marker_map = {
        'cryodrgn': '+',
        'drgnai-fixed': '.',
        'opusdsd_mu': 'p',
        'cryosparc-3dflex': '8',
        'cryosparc-3dva': 's',
        'recovar': 'D',
        'cryosparc-3d': 'x',
        'cryodrgn2': 'P',
        'drgnai-abinit': 'o',
        'cryosparc-3dabinit': 'X',
        }
    return color_map, label_map, marker_map


def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0])


def weightedAverageQuaternions(Q, w):
    '''
    from https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions/12439567#12439567
    https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py#L66C1-L87C44
    https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf
    '''
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0])

def expmap(e):
    """Convert axis-angle vector into 3D rotation matrix
    https://github.com/asarnow/pyem/blob/master/pyem/geom/convert.py#L157
    """
    theta = np.linalg.norm(e)
    if theta < 1e-16:
        return np.identity(3, dtype=e.dtype)
    w = e / theta
    k = np.array([[0, w[2], -w[1]],
                  [-w[2], 0, w[0]],
                  [w[1], -w[0], 0]], dtype=e.dtype)
    r = np.identity(3, dtype=e.dtype) + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
    return r

def main():
    parser = argparse.ArgumentParser(description='Process pose data.')
    parser.add_argument('--fname', type=str, help='File name of the pose data')
    parser.add_argument('--cryosparc_3cdclass_pose_fname', type=str, help='Optional: File name of the cryosparc 3cdclass pose data', default=None)
    parser.add_argument('--save_3cdclass_pose', action='store_true', help='Save mean cryosparc 3cdabinit pose data')
    parser.add_argument('--logyfalse', help='Plot y-axis on log scale', action='store_false')
    parser.add_argument('--logxtrue', help='Plot x-axis on log scale', action='store_true')
    parser.add_argument('--random_baseline_method', action='store_true', help="For each method's plot panel, put gray background of shuffled residuals as a random baseline")
    parser.add_argument('--no_dashed_median', action='store_false', help='Put verical bar of dashed median')
    parser.add_argument('--update_test', action='store_true', help='Put verical bar of dashed median')
    parser.add_argument('--no_dashed_mean', action='store_false', help='Put verical bar of dashed mean')
    parser.add_argument('--n_conformation_batch', type=int, help='n_conformation_batch to average over (in order in input)', default=10**3)
    parser.add_argument('--ticks_fontsize', type=int, help='Ploting label_fontsize', default=15)
    parser.add_argument('--label_fontsize', type=int, help='Plotting label_fontsize', default=20)
    parser.add_argument('--minx_auto', action='store_true', help="auto-adjust minimum value on x-axis to minimum on each method's panel")
    parser.add_argument('--alignment_method', type=str, help='Method to align poses', default='median')
    parser.add_argument('--plot_output', type=str, help='Method to align poses', default='pose_residual.pdf')
    parser.add_argument('--autobins', help='Default bins', action='store_false')
    parser.add_argument('--do_right_mult_R', help='right (R_gt = R_method R_g) vs left (R_gt = R_g R_method) multiply: ', action='store_true')
    

    
    args = parser.parse_args()
    print(args)

    poses = np.load(args.fname)
    if not args.minx_auto and args.logxtrue: Warning('minx_auto and logx are incompatible')


    qs = conversions.axis_angle_to_quaternion(torch.from_numpy(poses['cryosparc_3dabinit_poses']))


    if args.cryosparc_3cdclass_pose_fname is not None:
        cryosparc_3dabinit_mean_poses_q = torch.from_numpy(np.load(args.cryosparc_3cdclass_pose_fname))
    else:
        n = len(poses['cryosparc_3dabinit_embeddings'])
        cryosparc_3dabinit_mean_poses_q = torch.from_numpy(np.array([weightedAverageQuaternions(qs[idx], poses['cryosparc_3dabinit_embeddings'][idx]) for idx in range(n)]))
        dir = os.path.dirname(args.fname)
        np.save(os.path.join(dir,'cryosparc_3dabinit_mean_poses_q.npy'), cryosparc_3dabinit_mean_poses_q)
    
    cryosparc_3d_mean_poses_3x3 = conversions.quaternion_to_rotation_matrix(cryosparc_3dabinit_mean_poses_q).transpose(1,2).numpy().reshape(-1,9)
    if 'cryosparc_3d_poses' in poses.keys():
        cryosparc_3d_poses_3x3  = np.apply_along_axis(expmap, 1, poses['cryosparc_3d_poses']).reshape(-1,9)

    color_map, label_map, _ = presets()
    
    def plot():
        ticks_fontsize = args.ticks_fontsize
        label_fontsize = args.label_fontsize
        np.random.seed(0)

        abinit_method = {
            'random': poses['gt_poses'][np.random.choice(poses['gt_poses'].shape[0], poses['gt_poses'].shape[0], replace=False)],
            # 'drgnai-fixed': poses['drgnai_fixed_poses'],
            # 'cryosparc-3d': cryosparc_3d_poses_3x3,
            'cryodrgn2': poses['cryodrgn2_poses'],
            'drgnai-abinit': poses['drgnai_abinit_poses'],
            'cryosparc-3dabinit': cryosparc_3d_mean_poses_3x3,
                                    }
        if args.update_test:
            test_methods = {'Rjs_right_global': poses['Rjs_right_global'], 'Rjs_left_global': poses['Rjs_left_global']}
            abinit_method.update(test_methods)
            color_map.update({'Rjs_right_global': 'red', 'Rjs_left_global': 'blue'})
            label_map.update({'Rjs_right_global': 'Rjs_right_global', 'Rjs_left_global': 'Rjs_left_global'})


        fig, axes = plt.subplots(len(abinit_method), 1, figsize=(6, 14))
        # axes = [axes]
        logy = args.logyfalse
        print('logy', logy)
        logx = args.logxtrue
        n_rotations = len(poses['gt_poses'])
        random_idx = np.random.choice(n_rotations,n_rotations, replace=False)
        n_conformation_batch = args.n_conformation_batch
        for idx, (k,v) in enumerate(abinit_method.items()):
            print(k)
            if args.update_test and k not in test_methods.keys():
                continue
            rotations_A, rotations_B = poses['gt_poses'].reshape(-1, 3,3), v.reshape(-1, 3,3)
            
            v_poses_correct_frame =  np.zeros_like(v)
            for conf_idx in range(0,len(poses['gt_poses']), n_conformation_batch):

                if args.alignment_method == 'median':
                    rotA_hat, rotB_hat, mean_rot, dist2, best_medse = align_rot_median(rotations_A[conf_idx:conf_idx+n_conformation_batch], 
                                                                                       rotations_B[conf_idx:conf_idx+n_conformation_batch], 
                                                                                       N=n_conformation_batch,
                                                                                       do_right_mult_R=args.do_right_mult_R)
                    if args.do_right_mult_R:
                        rotations_B_in_frame_A = np.matmul(rotations_B[conf_idx:conf_idx+n_conformation_batch], mean_rot)
                    else:
                        rotations_B_in_frame_A = np.matmul(mean_rot,rotations_B[conf_idx:conf_idx+n_conformation_batch])

                elif args.alignment_method == 'mean_quaternion':
                    global_rotation = estimate_global_rotation(rotations_A[conf_idx:conf_idx+n_conformation_batch], rotations_B[conf_idx:conf_idx+n_conformation_batch])
                    rotations_B_in_frame_A = global_rotation.T @ rotations_B[conf_idx:conf_idx+n_conformation_batch]
                else:
                    raise ValueError(f'Alignment method {args.alignment_method} not implemented')


                v_poses_correct_frame[conf_idx:conf_idx+n_conformation_batch] = rotations_B_in_frame_A.reshape(-1, 9)
            
            resid = (poses['gt_poses'] - v_poses_correct_frame)
            norms = np.linalg.norm(resid, axis=1)
            min_x = np.min(norms) if args.minx_auto else 0
            bins = np.linspace(min_x,np.sqrt(8),30)
            pd.Series(norms).plot.hist(ax=axes[idx], bins=bins, alpha=0.5, label=label_map[k], legend=True, logy=logy, logx=logx, color=color_map[k])
            if k != 'random' and args.random_baseline_method: 
                random_idx = np.random.choice(n_rotations, n_rotations, replace=False)
                resid_random = (v - v[random_idx])
                norms_random = np.linalg.norm(resid_random, axis=1)
                pd.Series(norms_random).plot.hist(ax=axes[idx], bins=bins, alpha=0.25, label='random baseline', legend=True, logy=logy, logx=logx, color='gray')
            if k=='random' and args.no_dashed_median:
                solid = '-'
                axes[idx].axvline(x=np.mean(norms), color='black', linestyle=solid, linewidth=2)
            if k=='random' and args.no_dashed_median:
                dashed = '--'
                axes[idx].axvline(x=np.median(norms), color='black', linestyle=dashed, linewidth=2)
            axes[idx].set_ylim(bottom=1)
            axes[idx].set_xlim(min_x,np.sqrt(8))
            axes[idx].legend(loc='upper right')
            if k == 'cryosparc-3dabinit':
                axes[idx].set_xlabel(r'$||R_{GT} - R_{*}||_F$', fontsize=label_fontsize)
                axes[idx].set_ylabel('Frequency', fontsize=label_fontsize)
            else:
                axes[idx].set_ylabel('')

            
        fig.suptitle(f'Distribution of pose residuals (w.r.t. g.t.) \n fname={args.fname} \n per conf. global pose: n_conformation_batch={n_conformation_batch}', y=0.95)
        fig.savefig(args.plot_output, dpi=1200, bbox_inches='tight')
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

    plot()


if __name__ == '__main__':
    main()