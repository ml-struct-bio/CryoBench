import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants, noise, homogeneous, linalg, embedding, adaptive_kernel_discretization
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)


def M_step_batch(images, lhs_summed, rhs_summed, mean_batch, covariance_batch, CTF_params, rotation_matrices, translations, image_shape, volume_shape, grid_size, voxel_size, noise_variance,  CTF_fun):

    # Precomp piece
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    ctf_over_noise_variance = CTF**2 / noise_variance

    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape)
    volume_size = np.prod(volume_shape)

    second_moments = covariance_batch + linalg.broadcast_outer(mean_batch, mean_batch) * ctf_over_noise_variance
    second_moments = second_moments.reshape(second_moments.shape[0], -1)
    #Summed seconds moments
    lhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, second_moments, grid_point_indices.reshape(-1),  lhs_summed)


    images = core.translate_images(images, translations, image_shape)
    images = images * CTF / noise_variance
    images_means_h = linalg.broadcast_outer(images, mean_batch) 

    rhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, images_means_h.reshape(images_means_h.shape[0], -1), grid_point_indices.reshape(-1), rhs_summed)

    return lhs_summed, rhs_summed



# @functools.partial(jax.jit, static_argnums = [5])    
def M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size ):
    
            
    basis_size = latent_means.shape[-1]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    rhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype)
    lhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size *  basis_size), dtype = experiment_dataset.dtype)
        
    for batch, batch_image_ind in data_generator:
        
        lhs_summed, rhs_summed = M_step_batch(batch, lhs_summed, rhs_summed,
                                            latent_means[batch_image_ind], latent_covariances[batch_image_ind], 
                                            experiment_dataset.CTF_params[batch_image_ind],
                                            experiment_dataset.rotation_matrices[batch_image_ind],
                                            experiment_dataset.translations[batch_image_ind],
                                            experiment_dataset.image_shape, 
                                            experiment_dataset.volume_shape, 
                                            experiment_dataset.grid_size, 
                                            experiment_dataset.voxel_size, 
                                            noise_variance,
                                            experiment_dataset.CTF_fun)
        
    # Solve least squares
    lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)
    W = linalg.batch_solve(lhs_summed, rhs_summed)
    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    W = U @ jnp.diag(S)
    
    return W


def batch_vec(x):
    return x.swapaxes(-1,-2).reshape(-1, x.shape[-1]**2)

def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1,n,n).swapaxes(-1,-2)



def EM(experiment_dataset, mean_estimate, noise_variance, EM_iter = 20, basis_size = 10):

    # Initialize
    import jax.random as jr

    matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype_real)
    W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    eigenvalue = np.ones(basis_size)
    volume_mask = np.ones(experiment_dataset.volume_shape)
    contrast_grid = np.ones([1])
    batch_size = 1000
    disc_type = 'nearest'
    for iter_i in range(EM_iter):
        # E-step
        latent_means, latent_covariances, _ = embedding.get_coords_in_basis_and_contrast_3(experiment_dataset, mean_estimate, W, eigenvalue, volume_mask, noise_variance, contrast_grid, batch_size, disc_type, parallel_analysis = False, compute_covariances = True )

        # M-step
        W = M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size)


    return W