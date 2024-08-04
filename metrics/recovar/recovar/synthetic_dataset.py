# This code is meant only to evaluate performance of the algorithm on synthetic data.

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from recovar import core, utils, simulator, linalg, mask, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)

# Maybe should take out these dependencies?
from cryodrgn import mrc

#

def load_heterogeneous_reconstruction(simulation_info_file, volumes_path_root = None, load_volumes = True):
    if isinstance(simulation_info_file, dict):
        simulation_info = simulation_info_file 
    else:
        simulation_info = utils.pickle_load(simulation_info_file)

    volumes_path_root = simulation_info['volumes_path_root'] if volumes_path_root is None else volumes_path_root

    if load_volumes:
        volumes = simulator.load_volumes_from_folder(volumes_path_root, simulation_info['grid_size'] , simulation_info['trailing_zero_format_in_vol_name'] )
    else:
        volumes

    return HeterogeneousVolumeDistribution(volumes, simulation_info['image_assignment'], simulation_info['per_image_contrast'] )




class HeterogeneousVolumeDistribution():
    def __init__(self, volumes, image_assignments,  contrasts, valid_indices = None, vol_batch_size = None ):

        self.volume_shape = utils.guess_vol_shape_from_vol_size(volumes.shape[-1])
        self.vol_batch_size = utils.get_vol_batch_size(self.volume_shape[0], utils.get_gpu_memory_total()) if vol_batch_size is None else vol_batch_size
        self.volumes = volumes
        # self.volumes = linalg.batch_dft3(volumes, self.volume_shape, self.vol_batch_size)
        valid_indices = mask.get_radial_mask(self.volume_shape, radius = None) if valid_indices is None else valid_indices
        self.valid_indices = np.array(valid_indices.reshape(-1))
        if self.volumes is not None:
            self.volumes *= self.valid_indices[None,:]
        self.image_assignments = image_assignments
        self.contrasts = contrasts
        
        # Get Image assignment
        self.probs_of_state = None
        self.percent_outliers = None
        self.compute_probs_of_state()
        self.u = None
        self.s = None

        # self.get_u 

        self.mean = None
        self.covariance_cols = None

    def get_u(self):
        if self.u is None:
            self.compute_u_s()
        return self.u
    
    def get_s(self):    
        if self.s is None:
            self.compute_u_s()
        return self.s

    def compute_u_s(self, contrasted = False):
        self.u, self.s = self.get_covariance_eigendecomposition(contrasted = contrasted)

    def get_probs_of_state(self):
        # if self.probs_of_state is None:
        self.compute_probs_of_state()
        return self.probs_of_state

    def get_mean(self):
        if self.mean is None:
            self.compute_mean_conformation()
        return self.mean
    
    def compute_probs_of_state(self):        
        n_gt_molecules = self.volumes.shape[0]

        probs_of_molecule = np.zeros(n_gt_molecules, dtype = np.float32).real
        for k in range(n_gt_molecules):
            probs_of_molecule[k] = np.sum(self.image_assignments == k) 
        
        probs_of_molecule /= np.sum(probs_of_molecule)
        self.probs_of_state = probs_of_molecule
        self.percent_outliers = np.sum(self.image_assignments == -1) / self.image_assignments.size
    
    def compute_mean_conformation(self):
        self.mean = np.array(np.sum( self.volumes * self.get_probs_of_state()[:,None], axis = 0 ))
    
    def get_covariance_columns(self, picked_frequencies, contrasted = False ):
        vols = self.get_covariance_square_root(contrasted)
        return vols @ np.conj(vols[picked_frequencies,:]).T 


    def get_covariance_square_root(self, contrasted):
        contrast_variance = np.var(self.contrasts[self.image_assignments !=-1]) if contrasted else 0

        vols = np.sqrt(1 + contrast_variance)  * ( self.volumes - self.get_mean()[None,:] ) * np.sqrt(self.get_probs_of_state()[:,None])

        if contrast_variance > 0:
            vols = np.concatenate( [ np.sqrt(contrast_variance) * self.get_mean()[None] , vols ])
        return vols.T

    def get_vol_svd(self, contrasted = False, real_space = False, random_svd_pcs = None):

        vols = self.get_covariance_square_root( contrasted)

        if real_space:
            vols = linalg.batch_idft3(vols, self.volume_shape, self.vol_batch_size).real
        
        if random_svd_pcs is None:
            u,s,v = np.linalg.svd(vols, full_matrices = False)
        else:
            u,s,v = linalg.randomized_svd(vols, random_svd_pcs)

        return u,s,v

    def get_covariance_eigendecomposition(self, contrasted = False):
        u,s,_ = self.get_vol_svd(contrasted)
        zero_freq = core.frequencies_to_vec_indices( jnp.array([[0,0,0]] ), self.volume_shape)
        # u[zero_freq] == np.sum(Fu)== < Fu, 1 > 
        # ip = u[zero_freq,:] / np.abs(u[zero_freq,:])
        ip = np.where(np.abs(u[zero_freq,:])  > constants.ROOT_EPSILON, u[zero_freq,:] / np.abs(u[zero_freq,:]), 1 )
        u /= ip
        return u, s**2
    

    def get_fourier_variances(self, contrasted = False):
        vols = self.get_covariance_square_root(contrasted)
        return np.linalg.norm(vols, axis=-1)**2

    def get_spatial_variances(self, contrasted = False):
        vols = self.get_covariance_square_root(contrasted)
        vols = linalg.get_batch_idft3(vols, self.volume_shape, self.vol_batch_size)
        return np.linalg.norm(vols, axis=-1)**2





# def generate_ground_truth_volumes(image_option, volume_params, grid_size, voxel_size, padding):
#     if "from_mrc" in image_option:
#         # gt_volumes =  generate_volumes_from_mrcs(volume_params)
#         gt_volumes, voxel_size = generate_volumes_from_mrcs(volume_params, grid_size, padding)

#     elif "from_pdb" in image_option:
#         import simulate_scattering_potential as gsm
#         gt_volumes = gsm.generate_volumes_from_atom_groups(volume_params, voxel_size, grid_size)
#     return gt_volumes, voxel_size

# def get_gt_reconstruction(grid_size, voxel_size, padding, exp_name, valid_indices ):
#     datadir, vol_datadir, fake_vol_exp_name, fake_vol_datadir, indf, label_file, cov_noise_inp, uninvert_data, ctf_pose_datadir = preprocessed_datasets.get_dataset_params(exp_name, on_della=True)
#     gt_reconstruction = ExperimentReconstruction(grid_size, voxel_size, padding, exp_name, datadir, fake_vol_datadir,
#                                                      label_file = label_file, valid_indices = valid_indices)
#     return gt_reconstruction

def get_col_covariance_for_one_X_one_index(X, Xmean, vec_index):
    return (X - Xmean) * jnp.conj(X[vec_index] - Xmean[vec_index])

get_col_covariance_for_one_X_many_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes = (None, None, 0 ) )
get_col_covariance_for_many_X_one_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes = (0, None, None ) )

def get_col_covariance(Xs, X_mean, vec_indices, prob_of_X):
    # Batching over both dimensions

    cov = np.zeros_like(Xs, shape = [X_mean.size, vec_indices.size])
    # for k in range(Xs.shape[0]):
    #     for v_idx,v in enumerate(vec_indices):
    #         cov[:,v_idx] += prob_of_X[k] * get_col_covariance_for_one_X_one_index(Xs[k], X_mean, v, np = np ).T
    
    Xs_j = jnp.array(Xs)
    for v_idx,v in enumerate(vec_indices):
        cov[:,v_idx] = jnp.sum(prob_of_X[...,None] * get_col_covariance_for_many_X_one_index(Xs_j, X_mean, v ), axis =0)

    return cov



