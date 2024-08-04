import logging
import jax.numpy as jnp
import numpy as np
import jax, functools, time

from recovar import core, regularization, constants, noise, linalg, relion_functions, dataset
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils

logger = logging.getLogger(__name__)

## Low level functions
def make_X_mat(rotation_matrices, volume_shape, image_shape, pol_degree = 0, dtype = np.float32):

    grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
    if pol_degree ==0:
        return jnp.ones(grid_point_vec_indices.shape, dtype = dtype )[...,None], grid_point_vec_indices

    grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape ).astype(dtype)
    # Discretized grid points
    # This could be done more efficiently
    grid_points_coords_nearest = core.round_to_int(grid_points_coords)
    differences = grid_points_coords - grid_points_coords_nearest

    if pol_degree==1:
        X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences**1], axis = -1)
        return X_mat, grid_point_vec_indices

    differences_squared = linalg.broadcast_outer(differences, differences)
    differences_squared = keep_upper_triangular(differences_squared)
    X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_squared], axis = -1)

    return X_mat, grid_point_vec_indices

## Handling triangular to full matrix representation

def keep_upper_triangular(XWX):
    iu1 = np.triu_indices(XWX.shape[-1])
    return XWX[...,iu1[0], iu1[1]]

def undo_keep_upper_triangular_one(XWX):
    # m = n(n+1)/2
    # n = (sqrt(8m+1) -1)/2
    m = XWX.shape[-1]
    n = np.round((np.sqrt(8*m+1) -1)/2).astype(int)

    triu_indices = jnp.triu_indices(n)
    matrix = jnp.empty((n,n), dtype = XWX.dtype)
    matrix = matrix.at[triu_indices[0], triu_indices[1]].set(XWX)

    i_lower = jnp.tril_indices(n, -1)
    matrix = matrix.at[i_lower[0], i_lower[1]].set(matrix.T[i_lower[0], i_lower[1]])
    return matrix

def find_smaller_pol_indices(max_pol_degree, target_pol_degree):
    max_num_pol_params = get_feature_size(max_pol_degree)
    max_triu_indices = np.triu_indices(max_num_pol_params)
    target_num_pol_params = get_feature_size(target_pol_degree)
    target_triu_indices = np.triu_indices(target_num_pol_params)

    target_triu_indices = np.array(target_triu_indices).T
    max_triu_indices = np.array(max_triu_indices).T

    indices = np.zeros(target_triu_indices.shape[0], dtype = int)
    for k , searchval in enumerate(target_triu_indices):
        indices[k] = np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0]
    return indices

def find_diagonal_pol_indices(max_pol_degree):
    max_num_pol_params = get_feature_size(max_pol_degree)
    max_triu_indices = np.triu_indices(max_num_pol_params)
    max_triu_indices = np.array(max_triu_indices).T
    diagonal_indices = np.concatenate([np.arange(max_num_pol_params)[...,None],np.arange(max_num_pol_params)[...,None]], axis=-1)
    indices = np.zeros(diagonal_indices.shape[0], dtype = int)
    for k, searchval in enumerate(diagonal_indices):
        indices[k] = np.where(np.linalg.norm(max_triu_indices - searchval, axis=-1) == 0)[0]
    return indices

undo_keep_upper_triangular = jax.vmap(undo_keep_upper_triangular_one, in_axes = (0), out_axes = 0)


def volume_shape_to_half_volume_shape(volume_shape):
    return (volume_shape[0]//2 + 1, volume_shape[1], volume_shape[2] )

def half_volume_shape_to_volume_shape(volume_shape): 
    volume_shape[0] = volume_shape[0] * 2
    return ((volume_shape[0]-1)*2, *volume_shape[1:])

## NOTE There is something weird and jitting this. Compilation takes a very long time. Check it out.
@functools.partial(jax.jit, static_argnums = [1,2])
def vec_index_to_half_vec_index(indices, volume_shape, flip_positive = False):
    vol_indices_full = core.vec_indices_to_vol_indices(indices, volume_shape)

    grid_size = volume_shape[0]
    negative_frequencies = vol_indices_full[...,0] < (grid_size // 2 + 1) 
    if flip_positive:
        frequencies = core.vol_indices_to_frequencies(vol_indices_full, volume_shape)
        # flipped_first = jnp.where(frequencies, -frequencies , frequencies)
        frequencies_flipped = jnp.where(frequencies[...,0:1] > 0, -frequencies , frequencies)
        vol_indices_full_flipped = core.frequencies_to_vol_indices(frequencies_flipped, volume_shape)
        vol_indices_full = vol_indices_full_flipped

    in_bound = core.check_vol_indices_in_bound(vol_indices_full, grid_size)

    vec_indices = core.vol_indices_to_vec_indices(vol_indices_full, volume_shape)
    vec_indices = jnp.where(in_bound, vec_indices, -1*jnp.ones_like(vec_indices))
    return vec_indices, negative_frequencies

@functools.partial(jax.jit, static_argnums = [1])
def half_volume_to_full_volume(half_volume, volume_shape):
    volume_size = np.prod(volume_shape)
    indices = jnp.arange(volume_size) # will this JIT this whole thing? ugh
    half_indices, negative_frequencies = vec_index_to_half_vec_index(indices, volume_shape, flip_positive = True)
    volume = jnp.zeros(volume_size, dtype = half_volume.dtype)
    volume = half_volume[half_indices]
    volume = jnp.where(negative_frequencies, volume, jnp.conj(volume))
    return volume

batch_half_volume_to_full_volume = jax.vmap(half_volume_to_full_volume, in_axes = (0,None), out_axes = 0)

def full_volume_to_half_volume(volume, volume_shape):
    half_volume_shape = volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = np.prod(half_volume_shape)
    return volume[:half_volume_size]

batch_full_volume_to_half_volume = jax.vmap(full_volume_to_half_volume, in_axes = (0,None), out_axes = 0)


def half_vec_index_to_vec_index(indices_half, volume_shape):
    # For indices with negative frequencies, return -1 (out of bound, not used)
    vol_indices_half = core.vec_indices_to_vol_indices(indices_half, volume_shape_to_half_volume_shape(volume_shape))
    indices = core.vol_indices_to_vec_indices(vol_indices_half, volume_shape)
    bad_indices = indices_half == -1
    indices = indices.at[bad_indices].set(-1)
    return indices

## Precompute functions


@functools.partial(jax.jit, static_argnums = [5,6,7,9])    
def precompute_kernel_one_batch(images, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, CTF_fun, noise_variance, pol_degree =0, XWX = None, F = None, heterogeneity_distances = None, heterogeneity_bins = None ):

    # Precomp piece
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    ctf_over_noise_variance = CTF**2 / noise_variance

    X, grid_point_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, pol_degree = pol_degree)
    grid_point_indices, good_idx = vec_index_to_half_vec_index(grid_point_indices, volume_shape, flip_positive = True)
    X = X * good_idx[...,None]

    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))

    # XWX
    XWX_b = linalg.broadcast_outer(X * ctf_over_noise_variance[...,None] , X)
    XWX_b = keep_upper_triangular(XWX_b)

    if heterogeneity_bins is not None:
        # This could be made more efficient, but will skip for now.
        heterogeneity_bins_this = (heterogeneity_distances[...,None] <= heterogeneity_bins) 
        n_bins = heterogeneity_bins.size
    else:
        heterogeneity_bins_this = jnp.ones((images.shape[0], 1), dtype = np.bool)
        n_bins = 1

    XWX_b_one_shape = XWX_b.shape
    XWX_b = XWX_b[...,None] * heterogeneity_bins_this[...,None,None,:]
    XWX_b = XWX_b.reshape(XWX_b.shape[:-2] + (-1,))

    XWX = core.batch_over_vol_summed_adjoint_slice_by_nearest(
        half_volume_size, XWX_b,
        grid_point_indices.reshape(-1), XWX)
    
    # F
    images = core.translate_images(images, translations, image_shape)
    # In my formalism, images === images / CTF soo I guess this is right
    F_b = X * (images * CTF / noise_variance)[...,None] 

    # XWX_b_one_shape = XWX_b.shape
    F_b = F_b[...,None] * heterogeneity_bins_this[...,None,None,:]
    F_b = F_b.reshape(F_b.shape[:-2] + (-1,))

    F = core.batch_over_vol_summed_adjoint_slice_by_nearest(half_volume_size, F_b, grid_point_indices.reshape(-1),F)
    return XWX, F #.reshape(-1, XWX_b_one_shape[-1], n_bins  ), F.reshape(-1, XWX_b_one_shape[-1], n_bins  )


# Should this be set by cross validation?
def precompute_triangular_kernel(experiment_dataset, noise_variance, pol_degree=0, heterogeneity_distances = None, heterogeneity_bins = None):    
    XWX, F = 0,0
    # print(utils.report_memory_device())
    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size

    half_volume_size = np.prod(volume_shape_to_half_volume_shape(experiment_dataset.upsampled_volume_shape))
    XWX = jnp.zeros((half_volume_size, small_gram_matrix_size(pol_degree) * n_bins ), dtype = np.float32)
    F = jnp.zeros((half_volume_size, get_feature_size(pol_degree) *  n_bins), dtype = np.complex64)

    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - 3 * ( utils.get_size_in_gb(XWX)  + utils.get_size_in_gb(F))  ) )

    # batch_size = 1
    # Need to take out RR
    logger.info(f"batch size in precompute kernel: {batch_size}")
    # batch_size = 1
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    noise_variance_image = noise.make_radial_noise(noise_variance, experiment_dataset.image_shape).reshape(-1)

    idx = 0 
    for batch, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        # heterogeneity_distances = None if heterogeneity_distances is None else heterogeneity_distances[indices]
        XWX, F = precompute_triangular_kernel_one_batch(batch,
                                experiment_dataset.rotation_matrices[indices], 
                                experiment_dataset.translations[indices], 
                                experiment_dataset.CTF_params[indices], 
                                experiment_dataset.voxel_size, 
                                experiment_dataset.upsampled_volume_shape, 
                                experiment_dataset.image_shape, 
                                experiment_dataset.CTF_fun,
                                noise_variance_image, pol_degree = pol_degree, XWX = XWX, F = F, 
                                heterogeneity_distances = None if heterogeneity_distances is None else heterogeneity_distances[indices],
                                heterogeneity_bins = heterogeneity_bins)
        
        idx+=1

    XWX = XWX.reshape(-1, small_gram_matrix_size(pol_degree),  n_bins )
    F = F.reshape(-1, get_feature_size(pol_degree),  n_bins )
    logger.info(f"Done with precompute of kernel")
    return np.asarray(XWX), np.asarray(F)


@functools.partial(jax.jit, static_argnums = [5,6,7,9])    
def precompute_triangular_kernel_one_batch(images, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, CTF_fun, noise_variance, pol_degree =0, XWX = None, F = None, heterogeneity_distances = None, heterogeneity_bins = None ):

    if pol_degree > 0:
        raise NotImplementedError
    
    if heterogeneity_bins is not None:
        # This could be made more efficient, but will skip for now.
        heterogeneity_bins_this = (heterogeneity_distances[...,None] <= heterogeneity_bins) 
        n_bins = heterogeneity_bins.size
    else:
        heterogeneity_bins_this = jnp.ones((images.shape[0], 1), dtype = np.bool)
        n_bins = 1

    # print(images.shape)
    images = core.translate_images(images, translations, image_shape) / noise_variance
    images = images[...,None] * heterogeneity_bins_this[...,None,:]
    # print(images.shape)

    Ft_y = batch_im_adjoint_forward_model_from_map(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, 'linear_interp') 

    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / noise_variance
    CTF = CTF[...,None] * heterogeneity_bins_this[...,None,:]

    Ft_ctf = batch_im_adjoint_forward_model_from_map(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, 'linear_interp') 
    # A pretty naive way to do this
    return XWX + batch_full_volume_to_half_volume(Ft_ctf, volume_shape).T, F + batch_full_volume_to_half_volume(Ft_y, volume_shape).T

batch_im_adjoint_forward_model_from_map = jax.vmap(core.adjoint_forward_model_from_map, in_axes = (-1,None, None, None, None,None, None, None))




def get_differences_zero(pol_degree, differences):
    if pol_degree ==0:
        differences_zero = jnp.zeros_like(differences[...,0:1])
    elif pol_degree==1:
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences], axis = -1)
    elif pol_degree==2:
        differences_squared = linalg.broadcast_outer(differences, differences)
        differences_squared = keep_upper_triangular(differences_squared)
        differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences, differences_squared], axis = -1)
    else:
        assert(NotImplementedError)
    return differences_zero

def compute_summed_XWX_F_only():
    return

@functools.partial(jax.jit, static_argnums = [2,5,6])
def compute_summed_XWX_F(XWX, F, max_grid_dist, grid_distances, frequencies, volume_shape, pol_degree, extra_dimensions = None):

    # Might have to cast this back to frequencies vs indices frequencies
    near_frequencies = core.find_frequencies_within_grid_dist(frequencies, max_grid_dist)
    differences =  near_frequencies - frequencies[...,None,:]
    # This is just storing the same array many times over...
    differences_zero = get_differences_zero(pol_degree, differences)
    # L_inf norm distances
    distances = jnp.max(jnp.abs(differences), axis = -1)
    near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

    ## TODO SHOULD I PUT THIS BACK?
    valid_points = (distances <= (grid_distances[...,None] + 0.5) )* core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])
    # valid_points = core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

    near_frequencies_vec_indices, negative_frequencies = vec_index_to_half_vec_index(near_frequencies_vec_indices, volume_shape, flip_positive = True)

    XWX_summed_neighbor = batch_summed_over_indices(XWX, near_frequencies_vec_indices, valid_points)
    XWX_summed_neighbor = undo_keep_upper_triangular(XWX_summed_neighbor)

    Z = XWX[...,:differences_zero.shape[-1]]
    Z_summed_neighbor = batch_Z_grab(Z, near_frequencies_vec_indices, valid_points, differences_zero)

    # Rank 3 update. Z is real so no need for np.conj
    XWX_summed_neighbor += Z_summed_neighbor + jnp.swapaxes(Z_summed_neighbor, -1, -2 )

    alpha = XWX[...,0]
    XWX_summed_neighbor += batch_summed_scaled_outer(alpha, near_frequencies_vec_indices, differences_zero, valid_points)

    # F terms involve conjugates so need to be careful
    F_summed_neighbor = batch_summed_over_indices(F, near_frequencies_vec_indices, valid_points * negative_frequencies)
    F_summed_neighbor_conj = batch_summed_over_indices(jnp.conj(F), near_frequencies_vec_indices, valid_points * ~negative_frequencies)
    F_summed_neighbor += F_summed_neighbor_conj

    F0_summed_neighbor = batch_Z_grab(F[...,0:1], near_frequencies_vec_indices, valid_points * negative_frequencies, differences_zero)[...,0,:]
    F0_summed_neighbor_conj = batch_Z_grab(jnp.conj(F[...,0:1]), near_frequencies_vec_indices, valid_points * ~negative_frequencies, differences_zero)[...,0,:]
    F_summed_neighbor += F0_summed_neighbor + F0_summed_neighbor_conj
    return XWX_summed_neighbor, F_summed_neighbor




# Should this all just be vmapped instead of vmapping each piece? Not really sure.
# It allocate XWX a bunch of time if I do?
@functools.partial(jax.jit, static_argnums = [2,5,6,8, 9])
def compute_estimate_from_precompute_one(XWX, F, max_grid_dist, grid_distances, frequencies_vol_indices, volume_shape, pol_degree, prior_inverse_covariance, prior_option, return_XWX_F):
    XWX_summed_neighbor,F_summed_neighbor = compute_summed_XWX_F(XWX, F, max_grid_dist, grid_distances, frequencies_vol_indices, volume_shape, pol_degree)
    y_and_deriv, good_v, problems = compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor, F_summed_neighbor, frequencies_vol_indices, prior_inverse_covariance, prior_option, volume_shape)
    if return_XWX_F:
        return y_and_deriv, good_v, problems, keep_upper_triangular(XWX_summed_neighbor), F_summed_neighbor
    else:
        return y_and_deriv, good_v, problems, None, None

@functools.partial(jax.jit, static_argnums = [4,5,6])
def compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor, F_summed_neighbor, frequencies_vol_indices, prior_inverse_covariance, prior_option, volume_shape, XWX_in_flat = False):
    if XWX_in_flat:
        XWX_summed_neighbor = undo_keep_upper_triangular(XWX_summed_neighbor)
    vec_indices = core.vol_indices_to_vec_indices(frequencies_vol_indices.astype(int), volume_shape)
    prior_inverse_covariance = get_prior_from_options(prior_inverse_covariance, prior_option, vec_indices, volume_shape )
    y_and_deriv, good_v, problems = batch_solve_for_m(XWX_summed_neighbor,F_summed_neighbor, prior_inverse_covariance )
    return y_and_deriv, good_v, problems


def compute_estimate_from_XWX_F_summed(XWX_summed_neighbor, F_summed_neighbor, prior_inverse_covariance, prior_option, volume_shape):

    memory_per_pixel =  800 * utils.get_size_in_gb(XWX_summed_neighbor[0]) * 10
    volume_size = np.prod(volume_shape)
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    # import pdb; pdb.set_trace()
    print("mem before anything", utils.get_gpu_memory_used())
    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX_summed_neighbor) - utils.get_size_in_gb(F_summed_neighbor)  )/ (memory_per_pixel )  ) 
    # batch_size = 1000000
    # memory_per_pixel = (2*1 +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    n_batches = np.ceil(half_volume_size / batch_size).astype(int)

    frequencies_vol_indices = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    logger.info(f"compute_estimate_from_XWX_F_summed with prior option={prior_option} batch size: {batch_size}")

    prior_inverse_covariance = jnp.asarray(prior_inverse_covariance).real.astype(np.float32)

    reconstruction = np.zeros((half_volume_size, F_summed_neighbor.shape[-1]), dtype = np.complex64)
    good_pixels = np.zeros((half_volume_size), dtype = np.bool)

    logger.info(f"dtype = {prior_inverse_covariance.dtype}")
    # if prior_option == "complete":
    #     # prior_inverse_covariance = jnp.zeros_like(np.random.randn(prior_inverse_covariance.shape), dtype = np.complex64)
    #     prior_option = "complete"

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(half_volume_size, batch_size, k)

        reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems = compute_estimate_from_XWX_F_summed_one(XWX_summed_neighbor[ind_st:ind_end], F_summed_neighbor[ind_st:ind_end], frequencies_vol_indices[ind_st:ind_end], prior_inverse_covariance, prior_option, volume_shape, XWX_in_flat = True)
        print(str(k) + "...", end =" ")

    utils.report_memory_device(logger=logger)
    logger.info(f"Done with compute_estimate_from_XWX_F_summed with prior option={prior_option}")

    reconstruction = batch_half_volume_to_full_volume(reconstruction.T, volume_shape).T
    good_pixels = half_volume_to_full_volume(good_pixels, volume_shape)

    return np.asarray(reconstruction), np.asarray(good_pixels)

batch_half_volume_to_full_volume = jax.vmap(half_volume_to_full_volume, in_axes = (0,None), out_axes = 0)

def summed_scaled_outer(alpha, indices, differences_zero, valid_points):
    alpha_slices = alpha[indices] * valid_points
    return jnp.sum(linalg.multiply_along_axis(linalg.broadcast_outer(differences_zero, differences_zero), alpha_slices, 0 ), axis=0)

batch_summed_scaled_outer = jax.vmap(summed_scaled_outer, in_axes = (None,0,0, 0))

# Idk why I can't find a nice syntax to do this.
slice_first_axis = jax.vmap(lambda vec, indices: vec[indices], in_axes = (-1,None), out_axes=(-1))

def summed_over_indices(vec, indices, valid):
    sliced = slice_first_axis(vec,indices)
    return jnp.sum(sliced * valid[...,None], axis = -2)

batch_summed_over_indices = jax.vmap(summed_over_indices, in_axes = (None,0,0))

# It feels very silly to have to do this. But I guess JAX will clean up?
def Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
    sliced = slice_first_axis(Z,near_frequencies_vec_indices)
    return  (sliced * valid_points[...,None]).T @ differences

batch_Z_grab = jax.vmap(Z_grab, in_axes = (None,0,0,0))

def one_Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
    sliced = Z[near_frequencies_vec_indices]#slice_first_axis(Z,near_frequencies_vec_indices)
    return  (sliced * valid_points) * differences

batch_one_Z_grab = jax.vmap(one_Z_grab, in_axes = (None,0,0,0))

@jax.jit
def cond_from_flat_one(XWX):
    XWX = undo_keep_upper_triangular_one(XWX)
    return jnp.linalg.cond(XWX)

cond_from_flat = jax.vmap(cond_from_flat_one, in_axes = (0))

# @jax.jit
# def solve_for_m(XWX, f, regularization):

#     # regularization_expanded = make_regularization_from_reduced(regularization)
#     # XWX += jnp.diag(regularization_expanded)
#     XWX += regularization
#     # XWX = XWX.at[0,0].add(regularization)
    
#     dtype_to_solve = np.float64
#     vreal = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.real.astype(dtype_to_solve), lower=False, assume_a='pos')
#     vimag = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.imag.astype(dtype_to_solve), lower=False, assume_a='pos')
#     # vreal = jnp.linalg.solve(XWX, f.real)#, lower=False, assume_a='pos')
#     # vimag = jnp.linalg.solve(XWX, f.imag)#, lower=False, assume_a='pos')

#     v = vreal + 1j * vimag
#     good_v = jnp.linalg.cond(XWX) < 1e4

#     e1 = jnp.zeros_like(v)
#     e1 = e1.at[0].set(f[0]/XWX[0,0])

#     v = jnp.where(good_v, v, e1)
#     v = jnp.where(XWX[0,0] > 0, v, jnp.zeros_like(v))
#     v = v.astype(np.complex64) ## 128!!?!? CHANGE MAYBE??
#     # v = v.astype(np.complex128) ## 128!!?!? CHANGE MAYBE??

#     # If condition number is bad, do degree 0 approx?
#     # v = jnp.where(good_v, v, jnp.zeros_like(v))
#     # res = jnp.linalg.norm(XWX @v - f)**1 / jnp.linalg.norm(f)**1#, axis = -1)

#     bad_indices = jnp.isnan(v).any(axis=-1) + jnp.isinf(v).any(axis=-1)
#     problem = bad_indices * good_v

#     # problem = (res > 1e-6) * good_v
#     v = jnp.where(bad_indices, jnp.zeros_like(v), v)

#     return v, good_v, problem

# @jax.jit
# def solve_for_m_complex(XWX, f, regularization):

#     # regularization_expanded = make_regularization_from_reduced(regularization)
#     # XWX += jnp.diag(regularization_expanded)
#     XWX += regularization
#     # XWX = XWX.at[0,0].add(regularization)
    
#     dtype_to_solve = np.complex128
#     v = jax.scipy.linalg.solve(XWX, f.astype(dtype_to_solve), lower=False, assume_a='pos')
#     # v = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.astype(dtype_to_solve), lower=False, assume_a='pos')
#     # vimag = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.imag.astype(dtype_to_solve), lower=False, assume_a='pos')
#     # v = vreal + 1j * vimag

#     good_v = jnp.linalg.cond(XWX) < 1e4

#     e1 = jnp.zeros_like(v)
#     e1 = e1.at[0].set(f[0]/XWX[0,0])

#     v = jnp.where(good_v, v, e1)
#     v = jnp.where(XWX[0,0] > 0, v, jnp.zeros_like(v))
#     v = v.astype(np.complex64) ## 128!!?!? CHANGE MAYBE??
#     # v = v.astype(np.complex128) ## 128!!?!? CHANGE MAYBE??

#     # If condition number is bad, do degree 0 approx?
#     # v = jnp.where(good_v, v, jnp.zeros_like(v))
#     # res = jnp.linalg.norm(XWX @v - f)**1 / jnp.linalg.norm(f)**1#, axis = -1)

#     bad_indices = jnp.isnan(v).any(axis=-1) + jnp.isinf(v).any(axis=-1)
#     problem = bad_indices * good_v

#     # problem = (res > 1e-6) * good_v
#     v = jnp.where(bad_indices, jnp.zeros_like(v), v)

#     return v, good_v, problem


@jax.jit
def solve_for_m_simple(XWX, f, regularization):

    # regularization_expanded = make_regularization_from_reduced(regularization)
    # XWX += jnp.diag(regularization_expanded)
    XWX += regularization
    # XWX = XWX.at[0,0].add(regularization)
    
    # dtype_to_solve = np.float64
    # linalg.batch_hermitian_linear_solver(A,b)
    # vreal = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.real.astype(dtype_to_solve), lower=False, assume_a='pos')
    # vimag = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.imag.astype(dtype_to_solve), lower=False, assume_a='pos')
    # vreal = jnp.linalg.solve(XWX, f.real)#, lower=False, assume_a='pos')
    # vimag = jnp.linalg.solve(XWX, f.imag)#, lower=False, assume_a='pos')

    # v = vreal + 1j * vimag
    v = linalg.batch_hermitian_linear_solver(XWX, f)
    good_v = jnp.linalg.cond(XWX) < 1e4
    problem = ~good_v
    # e1 = jnp.zeros_like(v)
    # e1 = e1.at[0].set(f[0]/XWX[0,0])

    # v = jnp.where(good_v, v, e1)
    # v = jnp.where(XWX[0,0] > 0, v, jnp.zeros_like(v))
    # v = v.astype(np.complex64) ## 128!!?!? CHANGE MAYBE??
    # # v = v.astype(np.complex128) ## 128!!?!? CHANGE MAYBE??

    # # If condition number is bad, do degree 0 approx?
    # # v = jnp.where(good_v, v, jnp.zeros_like(v))
    # # res = jnp.linalg.norm(XWX @v - f)**1 / jnp.linalg.norm(f)**1#, axis = -1)

    # bad_indices = jnp.isnan(v).any(axis=-1) + jnp.isinf(v).any(axis=-1)
    # problem = bad_indices * good_v

    # problem = (res > 1e-6) * good_v
    # v = jnp.where(bad_indices, jnp.zeros_like(v), v)

    return v, good_v, problem



batch_solve_for_m = jax.vmap(solve_for_m_simple, in_axes = (0,0,0))

# Should this be set by cross validation?
def precompute_kernel(experiment_dataset, noise_variance, pol_degree=0, heterogeneity_distances = None, heterogeneity_bins = None):    
    XWX, F = 0,0
    # print(utils.report_memory_device())
    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size

    half_volume_size = np.prod(volume_shape_to_half_volume_shape(experiment_dataset.upsampled_volume_shape))
    XWX = jnp.zeros((half_volume_size, small_gram_matrix_size(pol_degree) * n_bins ), dtype = np.float32)
    F = jnp.zeros((half_volume_size, get_feature_size(pol_degree) *  n_bins), dtype = np.complex64)

    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - 2* utils.get_size_in_gb(XWX) - 2*utils.get_size_in_gb(F)  ) )

    # Need to take out RR
    logger.info(f"batch size in precompute kernel: {batch_size}")
    # batch_size = 1
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    noise_variance_image = noise.make_radial_noise(noise_variance, experiment_dataset.image_shape).reshape(-1)

    idx = 0 
    for batch, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        # heterogeneity_distances = None if heterogeneity_distances is None else heterogeneity_distances[indices]
        XWX, F = precompute_kernel_one_batch(batch,
                                experiment_dataset.rotation_matrices[indices], 
                                experiment_dataset.translations[indices], 
                                experiment_dataset.CTF_params[indices], 
                                experiment_dataset.voxel_size, 
                                experiment_dataset.upsampled_volume_shape, 
                                experiment_dataset.image_shape, 
                                experiment_dataset.CTF_fun,
                                noise_variance_image, pol_degree = pol_degree, XWX = XWX, F = F, 
                                heterogeneity_distances = None if heterogeneity_distances is None else heterogeneity_distances[indices],
                                heterogeneity_bins = heterogeneity_bins)
        idx+=1

    XWX = XWX.reshape(-1, small_gram_matrix_size(pol_degree),  n_bins )
    F = F.reshape(-1, get_feature_size(pol_degree),  n_bins )
    logger.info(f"Done with precompute of kernel")
    return np.asarray(XWX), np.asarray(F)

# Should pass a list of triplets (pol_degree : int, h : float, regularization : bool)

def get_default_discretization_params(grid_size):
    params = []
    pol_degrees = [0,1] if grid_size <= 512 else [0,]

    for pol_degree in pol_degrees:
        for h in [0,1]:
            params.append((pol_degree, h, True))
    params.append((0,1,False))
    params.append((0,2,True))

    return params
            
def small_gram_matrix_size(pol_degree):
    feature_size = get_feature_size(pol_degree)
    return (feature_size * (feature_size+1))//2

def big_gram_matrix_size(pol_degree):
    return get_feature_size(pol_degree)**2

def get_feature_size(pol_degree):
    if pol_degree==0:
        return 1
    if pol_degree==1:
        return 4
    if pol_degree==2:
        return 10
    return pol_degree



def estimate_volume_from_covariance_and_precompute(init_variance, discretization_params, XWXs, Fs, volume_shape):

    logger.info(f"Starting adaptive disc with params = {discretization_params}")
    # prior_option = discretization_params[0][2]
    h = discretization_params[1]
    max_pol_degree = discretization_params[0]
    volume_size = np.prod(volume_shape)

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]

    # All of this gets overwritten
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    pol_init_prior_inverse_covariance = 1 / init_variance[...,None]

    first_estimates = [None,None]

    for k in range(2):
        # Probably should rewrite this part
        first_estimates[k], _, XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_weights_from_precompute(volume_shape, XWXs[k], Fs[k], pol_init_prior_inverse_covariance, max_pol_degree, max_pol_degree, h = h, return_XWX_F = True, prior_option = "by_degree")

    combined, _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[0] + XWX_summed_neighbors[1], F_summed_neighbors[0] + F_summed_neighbors[1], pol_init_prior_inverse_covariance, "by_degree", volume_shape)
        
    return combined, first_estimates




def estimate_from_relion_style(cryos, discretization_params, XWXs, Fs, volume_shape, tau = None, use_spherical_mask = True, grid_correct = True, gridding_correct = "square" ):


    logger.info(f"Starting adaptive disc with params = {discretization_params}")
    # prior_option = discretization_params[0][2]
    h = discretization_params[1]
    pol_degree = discretization_params[0]
    assert pol_degree == 0, "Only p = 0 supported for now"

    volume_size = np.prod(cryos[0].volume_shape)
    from recovar import relion_functions

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]
    estimates = [None,None]
    for k in range(2):
        # Probably should rewrite this part
        XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_summed_XWX_F_only(volume_shape, XWXs[k], Fs[k], pol_degree, h)
        XWX_this = half_volume_to_full_volume(XWX_summed_neighbors[k][...,0], volume_shape)
        F_this = half_volume_to_full_volume(F_summed_neighbors[k][...,0], volume_shape)

        estimates[k] = relion_functions.post_process_from_filter(cryos[0], XWX_this, F_this, disc_type = 'nearest', use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = gridding_correct, kernel_width=h+1)

    return estimates




def estimate_multiple_disc_relion_style(experiment_datasets, noise_variance, discretization_params, heterogeneity_distances = None, heterogeneity_bins= None, residual_threshold = None, use_spherical_mask = True, grid_correct = True, gridding_correct = "square" ):
    
    cryos = experiment_datasets
    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    # prior_option = discretization_params[0][2]
    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])
    assert max_pol_degree == 0, "Only p = 0 supported for now"

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        heterogeneity_distances_this = None if heterogeneity_distances is None else heterogeneity_distances[k]
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances_this, heterogeneity_bins = heterogeneity_bins)    
        # For now just throw away this piece so it doesn't break code.
        if heterogeneity_bins is None:
            XWXs[k] = XWXs[k][...,0]
            Fs[k] = Fs[k][...,0]


    # final_estimates = 0
    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].upsampled_volume_size
    # # Compute weights for each discretization
    # final_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    # first_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)



    # final_estimates = np.zeros((n_disc_test, n_bins, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    n_bins = 1 if heterogeneity_bins is None else heterogeneity_bins.size
    first_estimates = np.zeros((n_disc_test, n_bins,  2, experiment_datasets[0].volume_size), dtype = np.complex64)

    # estimate_volume_from_covariance_and_precompute

    # estimate_volume_from_covariance_and_precompute(init_variance, discretization_params, XWXs, Fs, volume_shape)
    for idx, disc_params_this in enumerate(discretization_params):
        for b in range(n_bins):
            # final_estimates[idx,b], first_estimates[idx,b] = estimate_volume_from_covariance_and_precompute(signal_variance, disc_params_this, [XWXs[0][...,b], XWXs[1][...,b] ], [Fs[0][...,b], Fs[1][...,b] ], experiment_datasets[0].upsampled_volume_shape)
            first, second = estimate_from_relion_style(cryos, disc_params_this, [XWXs[0][...,b], XWXs[1][...,b] ], [Fs[0][...,b], Fs[1][...,b] ], cryos[0].upsampled_volume_shape, tau = None, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = gridding_correct )
            first_estimates[idx,b,0] = first.reshape(-1)
            first_estimates[idx,b,1] = second.reshape(-1)



    # n_dataset (2) x volume_size (N^3) x n_disc_test x n_bins x feature_size  
    # import pdb; pdb.set_trace()
    first_estimates = first_estimates[...,None]
    first_estimates = first_estimates.transpose(2, 3, 0, 1, 4)
    first_estimates = first_estimates.reshape([*first_estimates.shape[:2], -1, first_estimates.shape[-1]])


    # # residuals to pick best one
    # from recovar import dataset
    # # residual_threshold = None
    # # residual_num_images = 100
    # # Choose indices to test
    # residual_threshold = heterogeneity_bins[0]

    # if residual_threshold is not None:
    #     good_indices = heterogeneity_distances[1] < residual_threshold
    #     test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)
    # else:
    #     good_indices = np.argsort(heterogeneity_distances[1])[:residual_num_images]
    #     test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)

    # logger.info("Number of images used for residual computation: " + str(test_dataset.n_images))
    # upsample_fac = test_dataset.volume_upsampling_factor
    # test_dataset.update_volume_upsampling_factor(1)
    # residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, first_estimates[0], max_pol_degree )
    # # Meshgrid but idk to do it cleanly
    # all_params = []
    # for param in discretization_params:
    #     for bin in heterogeneity_bins:
    #         all_params.append((*param, bin))
    # # all_params = np.meshgrid([discretization_params, heterogeneity_bins])
            
    # index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, all_params, experiment_datasets[0].volume_shape)
    
    index_array_vol, disc_choices, residuals_averaged, summed_residuals = pick_best_heterogeneity_from_residual(first_estimates[0,...], experiment_datasets[1], heterogeneity_distances[1], heterogeneity_bins, discretization_params, residual_threshold = residual_threshold , min_number_of_images_in_bin = 50)

    opt_halfmaps = [None, None]
    opt_halfmaps[0] = jnp.take_along_axis(first_estimates[0,...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)
    opt_halfmaps[1] = jnp.take_along_axis(first_estimates[1,...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)


    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    
    # if return_all:
    #     return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(final_estimates), np.asarray(first_estimates), all_params # XWX, Z, F, alpha
    # else:
    return first_estimates, np.asarray(opt_halfmaps), np.asarray(disc_choices), np.asarray(residuals_averaged)



    first_estimates = first_estimates.swapaxes(0,2)[...,None]
    first_estimates = first_estimates.swapaxes(1,2)

    # first_estimates = first_estimates.transpose(1, 2, 0, 3)
    # residuals to pick best one
    zz = experiment_datasets[1].volume_upsampling_factor
    experiment_datasets[1].update_volume_upsampling_factor(1)
    residuals, _ = compute_residuals_many_weights_in_weight_batch(experiment_datasets[1], first_estimates[0], max_pol_degree )
    experiment_datasets[1].update_volume_upsampling_factor(zz)

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, discretization_params, experiment_datasets[0].volume_shape)

    # xx = 0.5 * (final_estimates[0][...,0] + final_estimates[1][...,0])
    weights_opt = jnp.take_along_axis(0.5 * (first_estimates[0][...,0] + first_estimates[1][...,0]), np.expand_dims(index_array_vol, axis=-1), axis=-1)

    utils.report_memory_device(logger=logger)
    return first_estimates, weights_opt, disc_choices, residuals_averaged, summed_residuals

def pick_best_heterogeneity_from_residual(estimates, full_test_dataset, heterogeneity_distances, heterogeneity_bins, discretization_params = None, residual_threshold = None , min_number_of_images_in_bin = 50):

    # Probably should separate this stuff?
    discretization_params = [(0,0,"")] if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    if residual_threshold is None:
        logger.warning("didn't specify either residual_threshold or residual_num_images, using first bin with at least " + str(min_number_of_images_in_bin) + " images")
        residual_threshold = heterogeneity_bins
        n_in_bins = np.zeros(heterogeneity_bins.size)
        for idx, b in enumerate(heterogeneity_bins):
            n_in_bins[idx] = np.sum(heterogeneity_distances < b)
        good_bins = n_in_bins >= min_number_of_images_in_bin
        bin_chosen = np.argmax(good_bins)
        residual_threshold = heterogeneity_bins[bin_chosen]

        # import pdb; pdb.set_trace()

    # residuals to pick best one
    from recovar import dataset

    if residual_threshold is not None:
        good_indices = heterogeneity_distances <= residual_threshold
        test_dataset = dataset.subsample_cryoem_dataset(full_test_dataset, good_indices)

        assert test_dataset.n_images > 0, "No images in bin"

    logger.info("Number of images used for residual computation: " + str(test_dataset.n_images))

    upsample_fac = test_dataset.volume_upsampling_factor
    test_dataset.update_volume_upsampling_factor(1)
    residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, estimates, max_pol_degree, use_linear_interp = True)

    # Meshgrid but idk to do it cleanly
    all_params = []
    for param in discretization_params:
        for bin in range(heterogeneity_bins.size):
            all_params.append((*param, bin))
            

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, all_params, test_dataset.volume_shape)

    summed_residuals = jnp.sum(residuals, axis = 0)
    # import pdb; pdb.set_trace()
    # print(summed_residuals.shape)
    return index_array_vol, disc_choices, residuals_averaged, summed_residuals
    



def estimate_optimal_covariance_and_volume(init_variance, init_prior_covariance_option, discretization_params, XWXs, Fs, volume_shape, reg_iters = 1):

    logger.info(f"Starting adaptive disc with params = {discretization_params}")
    # prior_option = discretization_params[0][2]
    h = discretization_params[1]
    max_pol_degree = discretization_params[0]
    volume_size = np.prod(volume_shape)

    # Polynomial estimates
    XWX_summed_neighbors = [None,None]
    F_summed_neighbors = [None,None]

    # All of this gets overwritten
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    if init_prior_covariance_option == "one_fixed":
        init_prior_inverse_covariance = 1 / (jnp.ones((half_volume_size), dtype = np.float32) * init_variance ) #+ (0 if use_reg else np.inf))
    else:
        init_prior_inverse_covariance = 1 / init_variance

    pol_init_prior_inverse_covariance = np.repeat(init_prior_inverse_covariance[...,None] , max_pol_degree+1, axis=-1)

    for k in range(2):
        # Probably should rewrite this part
        _, _, XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_weights_from_precompute(volume_shape, XWXs[k], Fs[k], pol_init_prior_inverse_covariance, max_pol_degree, max_pol_degree, h = h, return_XWX_F = True, prior_option = "by_degree")



    # First do a p = 0, h = input discretization with constant variance
    # half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    # init_prior_inverse_covariance = 1 / (jnp.ones((half_volume_size), dtype = np.float32) * init_variance ) #+ (0 if use_reg else np.inf))

    # First estimate signal variance with h = 0, p =0 
    first_estimates = [None,None]
    
    for k in range(2):
        first_estimates[k] = np.array( np.where(np.abs(XWX_summed_neighbors[k][...,0]) < constants.ROOT_EPSILON, 0, F_summed_neighbors[k][...,0] / (XWX_summed_neighbors[k][...,0] + init_prior_inverse_covariance ) ))
        first_estimates[k] = half_volume_to_full_volume(first_estimates[k], volume_shape)



    # From this, estimate a first prior with correct decay
    lhs = (XWX_summed_neighbors[0][...,0] + XWX_summed_neighbors[1][...,0]) /2
    lhs = half_volume_to_full_volume(lhs, volume_shape)
    pol_init_prior_inverse_covariance = 1/estimate_signal_variance_from_correlation(first_estimates[0], first_estimates[1], lhs, half_volume_to_full_volume(init_prior_inverse_covariance, volume_shape), volume_shape)
    if (pol_init_prior_inverse_covariance < 0).any():
        a =1
        # import pdb; pdb.set_trace()

    # Now do a p = input, h = input discretization with diagonal variance
    pol_init_prior_inverse_covariance = np.repeat(pol_init_prior_inverse_covariance[...,None] , discretization_params[0]+1, axis=-1)
    init_prior_inverse_covariance_avg = regularization.batch_average_over_shells(pol_init_prior_inverse_covariance.T, volume_shape,0).T
    init_prior_inverse_covariance_avg = batch_diag(make_regularization_from_reduced(init_prior_inverse_covariance_avg))


    current_prior_covariance_option = "by_degree"
    pol_current_prior_inverse_covariance = pol_init_prior_inverse_covariance.copy()
    local_covar_tmp = []
    pol_current_tmp = []
    estimate_tmp = []
    for reg_iter in range(reg_iters):

        first_estimates = np.zeros((2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
        for k in range(2): 
            first_estimates[k], _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[k], F_summed_neighbors[k], pol_current_prior_inverse_covariance , current_prior_covariance_option, volume_shape)

        # print("TAKE OUT THE 0")
        estimate_tmp.append(np.array(first_estimates))

        local_covariances, bad_covars = estimate_local_covariances(XWX_summed_neighbors[0], XWX_summed_neighbors[1], first_estimates[0], first_estimates[1], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape, max_pol_degree)

        ##
        local_covar_tmp.append(np.array(local_covariances))
        ##
         
        ## This is getting very messy...
        ## TODO think about what you are doing with your life

        # Project onto Hermitian positive definite matrices
        local_covariances = 0.5*(local_covariances + jnp.conj(local_covariances.swapaxes(1,2))).real # Assume things are real
        eigs , U = jnp.linalg.eigh(local_covariances)

        # Make eigenvalues not too small so that 
        EPS = 1e-4
        # Threshold away negatives
        eigs = jnp.where(eigs > 0, eigs , 0)

        # If eigenvalues are too small, maybe should bump them all the way to signal variance
        get_inv_s = jax.vmap( lambda eigs :  jnp.where( eigs >  EPS * jnp.max(jnp.abs(eigs)) , 1/eigs, 1/ (EPS * jnp.max(jnp.abs(eigs))) ) )
        s =get_inv_s(eigs)
        
        invert_from_svd = jax.vmap( lambda U, s :  (U * s[None]) @ jnp.conj(U).T  , in_axes = (0,0))
        prior_inverse_covariance = invert_from_svd(U,s)

        # If any are bad, revert to previous one?
        bad_prior_inverse_covariance = jnp.isnan(prior_inverse_covariance).any(axis=(-1, -2))  +  np.isinf(prior_inverse_covariance).any(axis=(-1, -2)) + bad_covars
        print("bad ones:", bad_prior_inverse_covariance.sum())
        # Now do a p = input, h = input discretization with covariance
        pol_current_prior_inverse_covariance = jnp.where(bad_prior_inverse_covariance[...,None,None], init_prior_inverse_covariance_avg, prior_inverse_covariance)
        if jnp.isnan(pol_current_prior_inverse_covariance).any():
            bad_prior_inverse_covariance2 = jnp.isnan(pol_current_prior_inverse_covariance).any(axis=(-1, -2)) #* np.isinf(pol_current_prior_inverse_covariance).any(axis=(-1, -2))
            jnp.isnan(init_prior_inverse_covariance_avg).any(axis=(-1, -2))
            import pdb; pdb.set_trace()

        pol_current_tmp.append(np.array(pol_current_prior_inverse_covariance))
        current_prior_covariance_option = "complete"

    # Now solve again with new covariances
    final_estimates = np.zeros((2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    for k in range(2): 
        final_estimates[k], _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[k], F_summed_neighbors[k], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape)

    combined, _ = compute_estimate_from_XWX_F_summed(XWX_summed_neighbors[0] + XWX_summed_neighbors[1], F_summed_neighbors[0] + F_summed_neighbors[1], pol_current_prior_inverse_covariance, current_prior_covariance_option, volume_shape)

    return final_estimates, first_estimates, [local_covar_tmp, pol_current_tmp, estimate_tmp, init_prior_inverse_covariance_avg, combined,
    batch_half_volume_to_full_volume(XWX_summed_neighbors[0].T, volume_shape).T, batch_half_volume_to_full_volume(F_summed_neighbors[0].T,volume_shape).T]


def homogeneous_multiple_relion_style(experiment_datasets, noise_variance, signal_variance, discretization_params, return_all, heterogeneity_distances, heterogeneity_bins, residual_threshold = None, residual_num_images = None ):


    return


def heterogeneous_reconstruction_fixed_variance(experiment_datasets, noise_variance, signal_variance, discretization_params, return_all, heterogeneity_distances, heterogeneity_bins, residual_threshold = None, residual_num_images = None ):

    if residual_threshold is None and residual_num_images is None:
        logger.warning("didn't specify either residual_threshold or residual_num_images, using first bin")
        residual_threshold = heterogeneity_bins[0]

    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    # prior_option = discretization_params[0][2]
    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])
    if max_pol_degree > 0:
        logger.warning("probably not implemented for pol_degree > 0")

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances[k], heterogeneity_bins = heterogeneity_bins)    
        
    # A crude signal variance estimation
    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
    if noise_variance is None:
        noise_variance, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)

    final_estimates = 0
    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].volume_size
    # Compute weights for each discretization
    n_bins = heterogeneity_bins.size
    final_estimates = np.zeros((n_disc_test, n_bins, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    first_estimates = np.zeros((n_disc_test, n_bins,  2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)

    # estimate_volume_from_covariance_and_precompute

    # estimate_volume_from_covariance_and_precompute(init_variance, discretization_params, XWXs, Fs, volume_shape)
    for idx, disc_params_this in enumerate(discretization_params):
        for b in range(n_bins):
            final_estimates[idx,b], first_estimates[idx,b] = estimate_volume_from_covariance_and_precompute(signal_variance, disc_params_this, [XWXs[0][...,b], XWXs[1][...,b] ], [Fs[0][...,b], Fs[1][...,b] ], experiment_datasets[0].upsampled_volume_shape)

    # volume_size (N^3) x n_disc_test x n_bins x feature_size  
    final_estimates = final_estimates.transpose(2, 0, 1, 3)
    final_estimates = final_estimates.reshape(final_estimates.shape[0], final_estimates.shape[1] * final_estimates.shape[2], final_estimates.shape[-1])


    # n_dataset (2) x volume_size (N^3) x n_disc_test x n_bins x feature_size  
    # import pdb; pdb.set_trace()
    first_estimates = first_estimates.transpose(2, 3, 0, 1, 4)
    first_estimates = first_estimates.reshape([*first_estimates.shape[:2], -1, first_estimates.shape[-1]])


    # residuals to pick best one
    # I guess one way to do this without changed the function is to make CTF 0 for all bad images?
    from recovar import dataset
    # Choose indices to test
    if residual_threshold is not None:
        good_indices = heterogeneity_distances[1] < residual_threshold
        test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)
    else:
        good_indices = np.argsort(heterogeneity_distances[1])[:residual_num_images]
        test_dataset = dataset.subsample_cryoem_dataset(experiment_datasets[1], good_indices)

    logger.info("Number of images used for residual computation: " + str(test_dataset.n_images))
    residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, first_estimates[0], max_pol_degree )
    # Meshgrid but idk to do it cleanly
    all_params = []
    for param in discretization_params:
        for bin in heterogeneity_bins:
            all_params.append((*param, bin))
    # all_params = np.meshgrid([discretization_params, heterogeneity_bins])

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, all_params, experiment_datasets[0].upsampled_volume_shape)

    weights_opt = jnp.take_along_axis(final_estimates[...,0] , np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(final_estimates), np.asarray(first_estimates), all_params # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)



def naive_heterogeneity_scheme_relion_style(experiment_dataset, noise_variance, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = 100, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp'):
    import gc

    # residuals to pick best one
    # I guess one way to do this without changed the function is to make CTF 0 for all bad images?
    estimates = []#heterogeneity_bins.size * [None]
    lhs, rhs = [], []
    og_contrast = experiment_dataset.CTF_params[:,8]
    idx =0 
    for residual_threshold in heterogeneity_bins:
        from recovar import dataset
        good_indices = heterogeneity_distances <= residual_threshold
        # experiment_dataset.CTF_params[:,8] = og_contrast * good_indices
        # logger.info("Number of images used for residual computation: " + str(np.sum(good_indices)))
        # utils.report_memory_device(logger=logger)
        # from recovar import relion_functions
        # estimate = relion_functions.relion_reconstruct(experiment_dataset, noise_variance, batch_size, 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = True, gridding_correct = "nosq" )

        test_dataset = dataset.subsample_cryoem_dataset(experiment_dataset, good_indices)
        # logger.info("Number of images used for residual computation: " + str(np.sum(good_indices)))
        utils.report_memory_device(logger=logger)
        from recovar import relion_functions
        # estimate = relion_functions.relion_reconstruct(test_dataset, noise_variance, batch_size, 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = True, gridding_correct = "nosq" )

        # estimate = relion_functions.relion_reconstruct(test_dataset, noise_variance, batch_size, 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = True, gridding_correct = "square", tau = tau )

        if compute_lhs_rhs:
            # This uses an upsampling factor of 1, because we want the lhs/rhs on the smaller grid for
            # doing 
            # estimate = relion_functions.relion_reconstruct(test_dataset, noise_variance, batch_size, disc_type, use_spherical_mask = False, upsampling_factor = 1, grid_correct = grid_correct, gridding_correct = "square", tau = tau )

            test_dataset.update_volume_upsampling_factor(1)
            Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(test_dataset , noise_variance.astype(np.float32),  batch_size,  disc_type = disc_type, return_lhs_rhs = False )

            estimate = relion_functions.post_process_from_filter(test_dataset, Ft_ctf, F_ty, tau = tau, disc_type = disc_type, use_spherical_mask = False, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1 )

            lhs.append(np.array(Ft_ctf))
            rhs.append(np.array(F_ty))
        else:
            estimate = relion_functions.relion_reconstruct(test_dataset, noise_variance, batch_size, 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = grid_correct, gridding_correct = "square", tau = tau )

        test_dataset.delete()
        del test_dataset
        gc.collect()

        logger.info(f"Number of images used in estimator {idx}: " + str(np.sum(good_indices)))
        idx+=1
        estimates.append(np.array(estimate.reshape(-1)))

    estimates = np.array(estimates)
    if compute_lhs_rhs:
        return estimates, lhs, rhs

    return estimates


def less_naive_heterogeneity_scheme_relion_style(experiment_dataset, noise_variance, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = 100, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp'):

    # residuals to pick best one
    # I guess one way to do this without changed the function is to make CTF 0 for all bad images?
    estimates = []#heterogeneity_bins.size * [None]
    experiment_dataset.update_volume_upsampling_factor(2)
    XWX, F = precompute_triangular_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=0, heterogeneity_distances= heterogeneity_distances, heterogeneity_bins = heterogeneity_bins)

    XWX = XWX[:,0,:].T
    F = F[:,0,:].T


    for idx in range(heterogeneity_bins.size):
        from recovar import dataset, relion_functions

        estimate = relion_functions.post_process_from_filter(experiment_dataset, half_volume_to_full_volume(XWX[idx], experiment_dataset.upsampled_volume_shape), half_volume_to_full_volume(F[idx],experiment_dataset.upsampled_volume_shape), tau = tau, disc_type = disc_type, use_spherical_mask = True, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1 )
        estimates.append(np.array(estimate.reshape(-1)))
    estimates = np.array(estimates)

    return estimates

def even_less_naive_heterogeneity_scheme_relion_style(experiment_dataset, noise_variance, signal_variance, heterogeneity_distances, heterogeneity_bins, batch_size = 100, tau = None, compute_lhs_rhs = False, grid_correct = True, disc_type = 'linear_interp', use_spherical_mask = True, return_lhs_rhs = False):

    # residuals to pick best one
    # I guess one way to do this without changed the function is to make CTF 0 for all bad images?
    estimates = []#heterogeneity_bins.size * [None]

    bins = heterogeneity_bins.copy()
    inds = np.digitize(heterogeneity_distances, bins, right = True)
    n_bins = bins.size
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(experiment_dataset.upsampled_volume_shape))

    lhs_all = np.zeros((n_bins, half_volume_size), dtype =experiment_dataset.dtype_real )
    rhs_all = np.zeros((n_bins, half_volume_size), dtype =experiment_dataset.dtype )

    # logger.info(f"batch size in het comp: {batch_size}")
    # data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size)
    # weights = jnp.asarray(weights)

    # print("APPLYING IMAGE MASK!?!")
    # print("APPLYING IMAGE MASK!?!")


    for bin_idx in range(n_bins):
        image_inds = np.where(inds == bin_idx)[0]
        # print(image_inds.size)

        data_generator = experiment_dataset.get_dataset_subset_generator( batch_size=batch_size, subset_indices = image_inds)
        lhs = jnp.zeros(half_volume_size, dtype = experiment_dataset.dtype_real )
        rhs = jnp.zeros(half_volume_size, dtype = experiment_dataset.dtype )
        for batch, indices in data_generator:

            # import pdb; pdb.set_trace()
            # Only place where image mask is used ?
            batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
            # print("APPLYING IMAGE MASK!?!")
            Ft_y_b, Ft_ctf_b = relion_functions.relion_style_triangular_kernel_batch(batch,
                                                                    experiment_dataset.CTF_params[indices], 
                                                                    experiment_dataset.rotation_matrices[indices], 
                                                                    experiment_dataset.translations[indices], 
                                                                    experiment_dataset.image_shape, 
                                                                    experiment_dataset.upsampled_volume_shape, 
                                                                    experiment_dataset.voxel_size, 
                                                                    experiment_dataset.CTF_fun, 
                                                                    disc_type, 
                                                                    noise_variance)
        

            # Ft_y_b2, Ft_ctf_b2 = relion_functions.relion_style_triangular_kernel_batch_trilinear(batch,
            #                                                         experiment_dataset.CTF_params[indices], 
            #                                                         experiment_dataset.rotation_matrices[indices], 
            #                                                         experiment_dataset.translations[indices], 
            #                                                         experiment_dataset.image_shape, 
            #                                                         experiment_dataset.upsampled_volume_shape, 
            #                                                         experiment_dataset.voxel_size, 
            #                                                         experiment_dataset.CTF_fun, 
            #                                                         disc_type, 
            #                                                         noise_variance)
            # print(jnp.linalg.norm(Ft_y_b - Ft_y_b2) / jnp.linalg.norm(Ft_y_b))


            rhs += full_volume_to_half_volume(Ft_y_b, experiment_dataset.upsampled_volume_shape)
            lhs += full_volume_to_half_volume(Ft_ctf_b, experiment_dataset.upsampled_volume_shape)

        rhs_all[bin_idx] = rhs
        lhs_all[bin_idx] = lhs

    # A slight improvement would be an almost triangular kernel/ pyramid kernel
    #    _
    #  _| |_
    #_|     |_ 
    # or almost epachenikov 
    #
    use_Epanechnikov = True
    if use_Epanechnikov:
        # distances = np.zeros(bins.size)
        # distances[1:] = bins[:-1]
        distances = bins
        h_grid = 2 * bins 
        # h_grid = bins
        logger.info("SHOULD THIS BE SQUARE ROOTED?")

        np_to_use = np
        Epanechnikov = lambda dist : np_to_use.where( np_to_use.abs(dist) < 1, 3/4 * (1- dist**2), 0)
        # h_grid = bins # Skip the first bin
        weight_matrix = np_to_use.zeros((n_bins, n_bins)).astype(np.float32)
        weight_matrix[0,0] = 1
        for idx in range(1, n_bins):
            weights = Epanechnikov(np_to_use.sqrt(distances/h_grid[idx]))
            weight_matrix[:,idx] = weights
        rhs_all = linalg.blockwise_A_X(rhs_all.T, weight_matrix).T
        lhs_all = linalg.blockwise_A_X(lhs_all.T, weight_matrix).T

        # rhs_all_presum = np_to_use.asarray(rhs_all.copy())
        # for idx in range(1, n_bins):
        #     weights = Epanechnikov(np_to_use.sqrt(distances/h_grid[idx]))#/ (heterogeneity_bins[idx] - heterogeneity_bins[idx-1]))
        #     rhs_all[idx] = np_to_use.sum(weights[:,None] * rhs_all_presum, axis = 0)
        # del rhs_all_presum

        # lhs_all_presum = np_to_use.asarray(lhs_all.copy())
        # for idx in range(1, n_bins):
        #     weights = Epanechnikov(np_to_use.sqrt(distances/h_grid[idx]))#/ (heterogeneity_bins[idx] - heterogeneity_bins[idx-1]))
        #     lhs_all[idx] = np_to_use.sum(weights[:,None] * lhs_all_presum, axis = 0)
        # del lhs_all_presum 
        # logger.info("done with Epanechnikov weights")
        # print(np.linalg.norm(lhs_all - lhs_all_presum2) / np.linalg.norm(lhs_all) )
        # print(np.linalg.norm(rhs_all - rhs_all_presum2) / np.linalg.norm(rhs_all) )

        # import pdb; pdb.set_trace()
        # weight_matrix = np_to_use.zeros((n_bins, n_bins))
        # for idx in range(0, n_bins):
        #     weights = Epanechnikov(np_to_use.sqrt(distances/h_grid[idx]))
        #     weight_matrix[idx] = weights
        # rhs_all_presum2 = linalg.blockwise_A_X(rhs_all, weight_matrix)
        # lhs_all_presum2 = linalg.blockwise_A_X(lhs_all, weight_matrix)

    else:
        rhs_all = np.cumsum(rhs_all, axis=0)
        lhs_all = np.cumsum(lhs_all, axis=0)
    # logger.info(f"done with precomp")

    for idx in range(heterogeneity_bins.size):
        estimate = relion_functions.post_process_from_filter(experiment_dataset, 
            half_volume_to_full_volume(lhs_all[idx], experiment_dataset.upsampled_volume_shape), 
            half_volume_to_full_volume(rhs_all[idx],experiment_dataset.upsampled_volume_shape),
            tau = tau, disc_type = disc_type, 
            use_spherical_mask = use_spherical_mask, grid_correct = grid_correct,
            gridding_correct = "square", kernel_width = 1 )

        # estimate = relion_functions.post_process_from_filter_v2( 
        #     half_volume_to_full_volume(lhs_all[idx], experiment_dataset.upsampled_volume_shape), 
        #     half_volume_to_full_volume(rhs_all[idx],experiment_dataset.upsampled_volume_shape),
        #     experiment_dataset.volume_shape, experiment_dataset.volume_upsampling_factor,
        #     tau = tau, disc_type = disc_type, 
        #     use_spherical_mask = use_spherical_mask, grid_correct = grid_correct,
        #     gridding_correct = "square", kernel_width = 1 )

        estimates.append(np.array(estimate.reshape(-1)))

    estimates = np.array(estimates)
    if return_lhs_rhs:
        return estimates, lhs_all, rhs_all
    
    return estimates



def compute_lhs_rhs(cryo,noise_variance, heterogeneity_distances, residual_threshold, batch_size  = 100, disc_type = 'linear_interp' ):
    
    good_indices = heterogeneity_distances <= residual_threshold
    test_dataset = dataset.subsample_cryoem_dataset(cryo, good_indices)
    # og_upsampling = cryo.volume_upsampling_factor
    test_dataset.update_volume_upsampling_factor(1)
    Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(test_dataset , noise_variance.astype(np.float32),  batch_size,  disc_type = disc_type, return_lhs_rhs = False )
    # cryo.update_volume_upsampling_factor(og_upsampling)

    return Ft_ctf, F_ty


    # residuals, _ = compute_residuals_many_weights_in_weight_batch(test_dataset, first_estimates[0], max_pol_degree )
    # # Meshgrid but idk to do it cleanly
    # all_params = []
    # for param in discretization_params:
    #     for bin in heterogeneity_bins:
    #         all_params.append((*param, bin))


    # all_params = np.meshgrid([discretization_params, heterogeneity_bins])





def estimate_variance_and_discretization_params(experiment_datasets, noise_variance, discretization_params, return_all, heterogeneity_distances = None, heterogeneity_bins= None ):
    discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

    # prior_option = discretization_params[0][2]
    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWXs = [None,None]; Fs = [None,None]
    for k in range(2):
        XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
                                         noise_variance.astype(np.float32), pol_degree=max_pol_degree, heterogeneity_distances= heterogeneity_distances, heterogeneity_bins = heterogeneity_bins)    
        # For now just throw away this piece so it doesn't break code.
        XWXs[k] = XWXs[k][...,0]
        Fs[k] = Fs[k][...,0]

    # A crude signal variance estimation
    gpu_memory = utils.get_gpu_memory_total()
    batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
    if noise_variance is None:
        noise_variance, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
        signal_var = np.max(signal_var)

    final_estimates = 0
    n_disc_test = len(discretization_params)
    volume_size = experiment_datasets[0].upsampled_volume_size
    # Compute weights for each discretization
    final_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    first_estimates = np.zeros((n_disc_test, 2, volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)


    for idx, disc_params_this in enumerate(discretization_params):
        final_estimates[idx], first_estimates[idx], prior_inverse_covariance = estimate_optimal_covariance_and_volume(signal_var, "one_fixed", disc_params_this, XWXs, Fs, experiment_datasets[0].upsampled_volume_shape)



    final_estimates = final_estimates.transpose(1, 2, 0, 3)
    first_estimates = first_estimates.transpose(1, 2, 0, 3)

    # residuals to pick best one
    residuals, _ = compute_residuals_many_weights_in_weight_batch(experiment_datasets[1], final_estimates[0], max_pol_degree )

    index_array_vol, disc_choices, residuals_averaged = pick_best_params(residuals, discretization_params, experiment_datasets[0].upsampled_volume_shape)

    # xx = 0.5 * (final_estimates[0][...,0] + final_estimates[1][...,0])
    weights_opt = jnp.take_along_axis(0.5 * (final_estimates[0][...,0] + final_estimates[1][...,0]), np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)
    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(final_estimates), np.asarray(first_estimates), discretization_params, prior_inverse_covariance # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)

def pick_best_params(residuals, discretization_params, volume_shape):

    # residuals to pick best one
    residuals_averaged = regularization.batch_average_over_shells(residuals.T, volume_shape,0)
    # Make choice. Impose that h must be increasing
    index_array = jnp.argmin(residuals_averaged, axis = 0)
    # This assumes that the discretization params are ordered by h

    # Check that all polynomial terms are the same:
    pol_degrees = np.array( [ param[0] for param in discretization_params ])

    if np.isclose(pol_degrees, pol_degrees[0]).all():
        # index_array = np.maximum.accumulate(index_array)
        index_array = index_array
    else:
        print("PROBLEM HERE")
        print(pol_degrees)

    disc_choices = np.array(discretization_params)[index_array]
    hs_choices = disc_choices[:,1].astype(int)
    hs_choices = np.maximum.accumulate(hs_choices)
    disc_choices[:,1] = hs_choices

    index_array_vol = utils.make_radial_image(index_array, volume_shape)
    # import pdb; pdb.set_trace()
    return index_array_vol, disc_choices, residuals_averaged




# def estimate_signal_variance(experiment_datasets, noise_variance, discretization_params, use_reg = True):
#     discretization_params = get_default_discretization_params(experiment_datasets[0].grid_size) if discretization_params is None else discretization_params

#     # prior_option = discretization_params[0][2]
#     h = discretization_params[0][1]
#     max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

#     # Precomputation
#     XWXs = [None,None]; Fs = [None,None]
#     for k in range(2):
#         XWXs[k], Fs[k] = precompute_kernel(experiment_datasets[k], 
#                                          noise_variance.astype(np.float32), pol_degree=max_pol_degree)
    
#     gpu_memory = utils.get_gpu_memory_total()
#     batch_size = utils.get_image_batch_size(experiment_datasets[0].grid_size, gpu_memory)
#     if noise_variance is None:
#         noise_variance, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
#         signal_var = np.max(signal_var)
#     else:
#         _, signal_var = noise.estimate_noise_variance(experiment_datasets[0], batch_size)
#         signal_var = np.max(signal_var)
    
#     prior_inverse_covariance = 1 / (jnp.ones((experiment_datasets[0].volume_size), dtype = np.float32) * signal_var  + (0 if use_reg else np.inf))#* np.inf #+ ~discretization_params[0][2] * np.inf
#     # First estimate signal variance with h = 0, p =0 
#     first_estimates = [None,None]
#     for k in range(2):
#         first_estimates[k] = np.array( np.where(np.abs(XWXs[k][...,0]) < constants.ROOT_EPSILON, 0, Fs[k][...,0] / (XWXs[k][...,0] + 1/ signal_var ) ))
#         first_estimates[k] = half_volume_to_full_volume(first_estimates[k], experiment_datasets[k].volume_shape)


#     lhs = (XWXs[0][...,0] + XWXs[1][...,0]) /2
#     lhs = half_volume_to_full_volume(lhs, experiment_datasets[0].volume_shape)
#     prior_inverse_covariance = 1/estimate_signal_variance_from_correlation(first_estimates[0], first_estimates[1], lhs, prior_inverse_covariance, experiment_datasets[0].volume_shape)

#     # Set all regularization params to prior_inverse_covariance
#     prior_inverse_covariance = np.repeat(prior_inverse_covariance[...,None] , max_pol_degree+1, axis=-1)  + (0 if use_reg else np.inf)

#     # Polynomial estimates
#     pol_estimates = [None,None]
#     XWX_summed_neighbors = [None,None]
#     F_summed_neighbors = [None,None]
#     for k in range(2):
#         pol_estimates[k], _, XWX_summed_neighbors[k], F_summed_neighbors[k] = compute_weights_from_precompute(experiment_datasets[0].volume_shape, XWXs[k], Fs[k], prior_inverse_covariance, max_pol_degree, max_pol_degree, h = h, return_XWX_F = True, prior_option = "by_degree")

#     local_covariances = estimate_local_covariances(XWX_summed_neighbors[0], XWX_summed_neighbors[1], pol_estimates[0], pol_estimates[1], prior_inverse_covariance, "by_degree", experiment_datasets[0].volume_shape, max_pol_degree)


#     diag_indices = find_diagonal_pol_indices(max_pol_degree)
#     degrees = get_degree_of_each_term(max_pol_degree)
#     diagonal_M_terms = (XWX_summed_neighbors[0][...,diag_indices] + XWX_summed_neighbors[1][...,diag_indices]) /2
#     diagonal_M_terms = batch_half_volume_to_full_volume(diagonal_M_terms.T, experiment_datasets[0].volume_shape)

#     # Compute the variance of that term
#     num_params = Fs[0].shape[-1]
#     signal_variance_final = np.zeros((experiment_datasets[0].volume_size, num_params), dtype = np.float32)
#     for i in range(Fs[0].shape[-1]):
#         # signal_variance_final[:,i], _ , _  = regularization.compute_fsc_prior_gpu_v2(experiment_datasets[0].volume_shape, pol_estimates[0][...,i], pol_estimates[1][...,i], diagonal_M_terms[i] , prior_inverse_covariance[:,degrees[i]], frequency_shift = jnp.array([0,0,0]))
#         signal_variance_final[:,i] = estimate_signal_variance_from_correlation(pol_estimates[0][...,i], pol_estimates[1][...,i], diagonal_M_terms[i], prior_inverse_covariance[:,degrees[i]], experiment_datasets[0].volume_shape)

#     return signal_variance_final, prior_inverse_covariance, local_covariances




def get_prior_from_options(prior_inverse_covariance, prior_option, vec_indices, volume_shape ):
    if prior_option == "by_degree":
        # vec_indices = core.vol_indices_to_vec_indices(frequencies_vol_indices.astype(int), volume_shape)
        regularization_expanded = make_regularization_from_reduced(prior_inverse_covariance[vec_indices])
        regularization = batch_diag(regularization_expanded)
    elif prior_option == "complete":
        frequencies = core.vec_indices_to_frequencies(vec_indices, volume_shape)
        radiuses = jnp.round(jnp.linalg.norm(frequencies, axis=-1)).astype(int)
        # I should probably store the inverse of these of pseudo inverse, which is not great but they are small so should be okay...
        regularization = prior_inverse_covariance[radiuses]
    elif prior_option == "none":
        # dim = pol_degree + 1
        # dim = get_feature_size(pol_degree)
        # regularization = jnp.zeros((frequencies.shape[0], dim, dim), dtype = np.float32)
        regularization = jnp.zeros((vec_indices.shape[0],1,1 ), dtype = np.float32)
    return regularization

batch_kron = jax.vmap(jnp.kron, in_axes=(0,0))
batch_diag = jax.vmap(jnp.diag, in_axes=(0))

def batch_vec(x):
    return x.swapaxes(-1,-2).reshape(-1, x.shape[-1]**2)

def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1,n,n).swapaxes(-1,-2)


def estimate_local_pol_covariances_inner(XWX, estimate_0, estimate_1, prior_inverse_covariance, prior_option, frequency_vec_indices, volume_shape):
    frequencies = core.vec_indices_to_frequencies(frequency_vec_indices, volume_shape)
    # if freq is not on the 0-line, can double it (since there is one XWX for each side)
    doubling = jnp.where(frequencies[...,0] == 0, 1, jnp.sqrt(2))
    radiuses = jnp.round(jnp.linalg.norm(frequencies, axis=-1)).astype(int)

    prior_inverse_covariance = get_prior_from_options(prior_inverse_covariance, prior_option, frequency_vec_indices, volume_shape )
    # signal_variance_expanded = make_regularization_from_reduced(prior_inverse_covariance)
    XWX = undo_keep_upper_triangular(XWX)
    U = (XWX + prior_inverse_covariance)
    # This takes frequency_vec_indices.size * gram_size * gram_size memory
    # K1= jax.scipy.linalg.solve(U, XWX, lower=False,assume_a='pos')
    K = linalg.batch_hermitian_linear_solver(U, XWX) * doubling[...,None,None]
    # K = jax.scipy.linalg.solve(U, XWX, lower=False,assume_a='pos') * doubling[...,None,None]
    krons = batch_kron(K,K)
    summed_krons = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_shape[0]//2-1, krons.reshape(krons.shape[0], -1), radiuses, None).reshape([-1, krons.shape[-1], krons.shape[-1] ])

    # vec
    # Should there be an extra one half here? This is effectively saying we are treating real and imaginary part as independent
    # 
    estimate_covariance = linalg.broadcast_outer(estimate_0 , estimate_1 )#.swapaxes(-1,-2).reshape(-1, estimate_0.shape[-1]**2)
    estimate_covariance = 0.5 * (estimate_covariance + jnp.conj(estimate_covariance.swapaxes(-1,-2)))
    estimate_covariance = batch_vec(estimate_covariance)

    # Get the 
    estimate_covariance_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_shape[0]//2-1, estimate_covariance.reshape(krons.shape[0], -1), radiuses, None)#.reshape([-1, estimate_0.shape[-1], estimate_0.shape[-1] ])

    good_v = jnp.where(frequencies[...,0] == 0, 0, 1)

    # Add the hermitian conjugate
    estimate_covariance = linalg.broadcast_outer(jnp.conj(estimate_0) * good_v[...,None], jnp.conj(estimate_1) * good_v[...,None])#.swapaxes(-1,-2).reshape(-1, estimate_0.shape[-1]**2)
    estimate_covariance = 0.5 * (estimate_covariance + jnp.conj(estimate_covariance.swapaxes(-1,-2)))
    estimate_covariance = batch_vec(estimate_covariance)

    estimate_covariance_summed += core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_shape[0]//2-1, estimate_covariance.reshape(krons.shape[0], -1), radiuses, None)#.reshape([-1, estimate_0.shape[-1], estimate_0.shape[-1] ])


    ## TODO There is a wild 0.5 here b/c we are treating real and imaginary part as independent
    return summed_krons, 0.5 * batch_unvec(estimate_covariance_summed)


def estimate_local_covariances(XWX_0, XWX_1, estimate_0, estimate_1, prior_inverse_covariance, prior_option, volume_shape, pol_degree):

    logger.info("starting local covariances estimation")
    volume_size = np.prod(volume_shape)
    half_volume_size = np.prod(volume_shape_to_half_volume_shape(volume_shape))
    num_params = get_feature_size(pol_degree)

    krons = jnp.zeros((volume_shape[0]//2-1, num_params**2, num_params**2), dtype = XWX_0.dtype)
    vec_indices = np.arange(half_volume_size)
    # Covariances of estimates
    estimate_0 = estimate_0.astype(np.complex64)
    estimate_1 = estimate_1.astype(np.complex64)
    utils.report_memory_device(logger=logger)
    ## TODO change this?
    batch_size = int((utils.get_gpu_memory_total()  )/ ( 1 * 4 * num_params**4 * 4 / 1e9 )  ) 
    n_batches = np.ceil(half_volume_size / batch_size).astype(int)
    krons, estimate_covariance_averaged = 0,0
    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(half_volume_size, batch_size, k)
        krons_this, covs_this = estimate_local_pol_covariances_inner((XWX_0[ind_st:ind_end] + XWX_1[ind_st:ind_end])/2, estimate_0[ind_st:ind_end], estimate_1[ind_st:ind_end],  prior_inverse_covariance, prior_option, vec_indices[ind_st:ind_end], volume_shape)

        krons += krons_this
        estimate_covariance_averaged += covs_this
    logger.info("end of local covariances batch")
    utils.report_memory_device(logger=logger)

    # batch_vec()

    # estimated_covariances = jnp.linalg.solve(krons, estimate_covariance_averaged.swapaxes(-1,-2).reshape(krons.shape[0], -1))
    estimated_covariances = linalg.batch_hermitian_linear_solver(krons, batch_vec(estimate_covariance_averaged).real)#, assume_a='pos' )

    # estimated_covariances = jax.scipy.linalg.solve(krons, batch_vec(estimate_covariance_averaged).real, assume_a='pos' )
    estimated_covariances = batch_unvec(estimated_covariances)#.reshape([-1, num_params,num_params]).swapaxes(-1,-2)
    logger.info("end of local covariances estimation")

    bad_covars = jnp.linalg.cond(krons) > 1e4

    return estimated_covariances, bad_covars



def estimate_signal_variance_from_correlation(vol1, vol2, lhs, prior, volume_shape):
    correlation = jnp.conj(vol1) * vol2
    correlation_avg = regularization.average_over_shells(correlation.real, volume_shape, frequency_shift = np.array([0,0,0]))
    correlation_avg = jnp.where( correlation_avg > constants.ROOT_EPSILON , correlation_avg , constants.ROOT_EPSILON )

    top = lhs**2 / (lhs + 1/prior)**2
    sum_top = regularization.average_over_shells(top,  volume_shape, frequency_shift = np.array([0,0,0]))
    prior_avg = jnp.where( sum_top > 0 , correlation_avg / sum_top , constants.ROOT_EPSILON )
    # prior_avg = jnp.where( sum_top > 0 , correlation_avg  , constants.ROOT_EPSILON )

    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = np.array([0,0,0])).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]

    return prior


def get_degree_of_each_term(max_pol_degree):
    # signal variance, deriv_variance, hessian_variance
    if max_pol_degree==0:
        return np.array([0])
    if max_pol_degree==1:
        return np.array([0,1,1,1])
    if max_pol_degree==2:
        return np.array([0,1,1,1,2,2,2,2,2,2])
    assert(NotImplementedError)

# May want to do delte this
def test_multiple_disc2(experiment_dataset, noise_variance, discretization_params, prior_inverse_covariance):

    discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWX, F = precompute_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=max_pol_degree)
    n_disc_test = len(discretization_params)

    # Compute weights for each discretization
    weights = np.zeros((n_disc_test, experiment_dataset.upsampled_volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    valid_weights = np.zeros((n_disc_test, experiment_dataset.upsampled_volume_size), dtype = bool)
    utils.report_memory_device(logger=logger)
    XWX = np.asarray(XWX)
    F = np.asarray(F)

    XWX_s = dict()
    F_s = dict()
    for idx, (pol_degree, h, reg) in enumerate(discretization_params):
        logger.info(f"computing discretization with params: degree={pol_degree}, h={h}, reg={reg}")
        # reg_used = prior_inverse_covariance if reg else None
        weights_this, valid_weights_this, XWX_s[idx],F_s[idx] = compute_weights_from_precompute(experiment_dataset.upsampled_volume_shape, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, prior_option = reg)
        weights[idx,:,:weights_this.shape[-1]] = weights_this
        valid_weights[idx] = valid_weights_this

    return XWX_s, F_s




def test_multiple_disc(experiment_dataset, cross_validation_dataset, noise_variance,  batch_size, discretization_params, prior_inverse_covariance, return_all = False):

    discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

    max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

    # Precomputation
    XWX, F = precompute_kernel(experiment_dataset, noise_variance.astype(np.float32), pol_degree=max_pol_degree)
    n_disc_test = len(discretization_params)

    # Compute weights for each discretization
    weights = np.zeros((n_disc_test, experiment_dataset.upsampled_volume_size, get_feature_size(max_pol_degree)), dtype = np.complex64)
    valid_weights = np.zeros((n_disc_test, experiment_dataset.upsampled_volume_size), dtype = bool)
    utils.report_memory_device(logger=logger)
    XWX = np.asarray(XWX)
    F = np.asarray(F)
    for idx, (pol_degree, h, reg) in enumerate(discretization_params):
        logger.info(f"computing discretization with params: degree={pol_degree}, h={h}, reg={reg}")
        # reg_used = prior_inverse_covariance if reg else None
        weights_this, valid_weights_this, _,_ = compute_weights_from_precompute(experiment_dataset.upsampled_volume_shape, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, prior_option = reg)
        weights[idx,:,:weights_this.shape[-1]] = weights_this
        valid_weights[idx] = valid_weights_this


    logger.info(f"Done computing params")
    weights = weights.swapaxes(0,1)
    utils.report_memory_device(logger=logger)
    del XWX, F

    # residuals to pick best one
    residuals, _ = compute_residuals_many_weights_in_weight_batch(cross_validation_dataset, weights, max_pol_degree )
    residuals_averaged = regularization.batch_average_over_shells(residuals.T, experiment_dataset.upsampled_volume_shape,0)

    # Make choice. Impose that h must be increasing
    index_array = jnp.argmin(residuals_averaged, axis = 0)
    index_array_vol = utils.make_radial_image(index_array, experiment_dataset.upsampled_volume_shape)

    disc_choices = np.array(discretization_params)[index_array]
    hs_choices = disc_choices[:,1].astype(int)
    hs_choices = np.maximum.accumulate(hs_choices)
    disc_choices[:,1] = hs_choices

    weights_opt = jnp.take_along_axis(weights[...,0], np.expand_dims(index_array_vol, axis=-1), axis=-1)

    logger.info("Done with adaptive disc")
    utils.report_memory_device(logger=logger)

    if return_all:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals.T), np.asarray(weights), discretization_params # XWX, Z, F, alpha
    else:
        return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)


def make_regularization_from_reduced(regularization_reduced):
    # signal variance, deriv_variance, hessian_variance
    if regularization_reduced.shape[-1] == 1:
        return regularization_reduced
    if regularization_reduced.shape[-1] == 2:
        return jnp.concatenate([regularization_reduced[...,0:1], 
                                jnp.repeat(regularization_reduced[...,1:2], repeats =3, axis=-1, total_repeat_length=3 ) ], axis = -1)
    if regularization_reduced.shape[-1] == 3:
        return jnp.concatenate([regularization_reduced[...,0:1], 
                                jnp.repeat(regularization_reduced[...,1:2], repeats =3, axis=-1,total_repeat_length=3),
                                jnp.repeat(regularization_reduced[...,2:3], repeats =6, axis=-1,total_repeat_length=6) ],
                                axis = -1)

def compute_weights_from_precompute(volume_shape, XWX, F, prior_inverse_covariance, pol_degree, max_pol_degree, h, return_XWX_F = False, prior_option = "by_degree"):
    # use_regularization = prior_inverse_covariance is not None
    volume_size = np.prod(volume_shape)

    # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    if type(h) == float or type(h) == int:
        h_max = int(h)
        h_ar = h*np.ones(volume_size)
    else:
        h_max = int(np.max(h))
        h_ar = h.astype(int) * 1.0

    feature_size = get_feature_size(pol_degree)

    if XWX.shape[-1] != small_gram_matrix_size(pol_degree):
        triu_indices = find_smaller_pol_indices(max_pol_degree, pol_degree)
        XWX = XWX[...,triu_indices]
        F = F[...,:feature_size]

    # take out stuff from prior
    if prior_option == "by_degree":
        prior_inverse_covariance = prior_inverse_covariance[:,:pol_degree+1]   
    elif prior_option == "complete":
        num_params = get_feature_size(pol_degree)
        prior_inverse_covariance = prior_inverse_covariance[...,:num_params, :num_params]
    else:
        prior_inverse_covariance = None

    # n_pol_param = small_gram_matrix_size(pol_degree) + 2 * get_feature_size(pol_degree) + 1
    memory_per_pixel = (2*h_max +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    ## TODO this should be fixed now. 
    ## There seems to be a strange bug with JAX. If it just barely runs out of memory, it won't throw an error but the memory will get corrupted and the answer is nonsense. This is an incredibly difficult thing to debug. 
    if h_max == 0:
        memory_per_pixel = (2*1 +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4
    ## TODO TAKE THIS OUT?
    memory_per_pixel *= 10

    XWX = jnp.asarray(XWX)
    F = jnp.asarray(F)
    prior_inverse_covariance = jnp.array(prior_inverse_covariance)

    reconstruction = np.zeros((volume_size, feature_size), dtype = np.complex64)
    good_pixels = np.zeros((volume_size), dtype = bool)
    msgs = 0
    
    if return_XWX_F:
        XWX_s = np.zeros_like(XWX)
        F_s = np.zeros_like(F)
    else:
        XWX_s = None
        F_s = None

    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX) - utils.get_size_in_gb(F)  )/ (memory_per_pixel   /1e9  )  ) 
    n_batches = np.ceil(volume_size / batch_size).astype(int)
    logger.info(f"KE batch size: {batch_size}")

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(volume_size, batch_size, k)
        # signal_variance_this = prior_inverse_covariance[ind_st:ind_end, :pol_degree+1] if use_regularization else None

        reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems, XWX_b, F_b = compute_estimate_from_precompute_one(XWX, F, h_max, h_ar[ind_st:ind_end], threed_frequencies[ind_st:ind_end], volume_shape, pol_degree =pol_degree, prior_inverse_covariance=prior_inverse_covariance, prior_option = prior_option, return_XWX_F = return_XWX_F)

        if return_XWX_F and (ind_st < XWX_s.shape[0]):
            ind_end_t = np.min([ind_end, XWX_s.shape[0]])
            # if ind_st <XWX_s.shape[0]
            XWX_s[ind_st:ind_end_t] = XWX_b[:ind_end_t-ind_st]
            F_s[ind_st:ind_end_t] = F_b[:ind_end_t-ind_st]

        if k < 3:
            utils.report_memory_device(logger=logger)
            print(k)

        if jnp.isnan(reconstruction[ind_st:ind_end]).any():
            logger.warning(f"IsNAN {jnp.isnan(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={prior_option}")

        if problems.any():
            if msgs < 10: 
                logger.warning(f"Issues in linalg solve? Problems for {problems.sum() / problems.size} pixels, pol_degree={pol_degree}, h={h}, reg={prior_option}")
                msgs +=1
                logger.warning(f"isinf {jnp.isinf(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={prior_option}")

    logger.info(f"Done with kernel estimate")
    return reconstruction, good_pixels, XWX_s, F_s



def compute_summed_XWX_F_only(volume_shape, XWX, F, pol_degree, h):
    # use_regularization = prior_inverse_covariance is not None
    volume_size = np.prod(volume_shape)

    # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0
    
    XWX_s = np.zeros_like(XWX)[...,None]
    F_s = np.zeros_like(F)

    XWX = jnp.asarray(XWX)
    F = jnp.asarray(F)
    memory_per_pixel = (2*h +1)**3 * big_gram_matrix_size(pol_degree) * 2 * 8 * 4

    batch_size = int((utils.get_gpu_memory_total() -  utils.get_size_in_gb(XWX) - utils.get_size_in_gb(F)  )/ (memory_per_pixel   /1e9  )  ) 
    n_batches = np.ceil(volume_size / batch_size).astype(int)
    logger.info(f"KE batch size: {batch_size}")
    threed_frequencies = core.vec_indices_to_vol_indices(np.arange(volume_size), volume_shape ) * 1.0

    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(volume_size, batch_size, k)
        # signal_variance_this = prior_inverse_covariance[ind_st:ind_end, :pol_degree+1] if use_regularization else None

        # probably should rewrite this to not have extra indices
        if (ind_st < XWX_s.shape[0]):
            ind_end_t = np.min([ind_end, XWX_s.shape[0]])

            XWX_s[ind_st:ind_end_t], F_s[ind_st:ind_end_t] = compute_summed_XWX_F(XWX, F, h, h, threed_frequencies[ind_st:ind_end_t], volume_shape, pol_degree)

    logger.info(f"Done with kernel estimate")
    return XWX_s[...,0], F_s[...]




@functools.partial(jax.jit, static_argnums = [5,6,7,8,9, 10])    
def compute_residuals_batch_many_weights(images, weights, rotation_matrices, translations, CTF_params, volume_shape, image_shape, CTF_fun, voxel_size, pol_degree, use_linear_interp = False ):

    if use_linear_interp:
        X_mat, gridpoint_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, pol_degree = pol_degree)
        weights_on_grid = weights[gridpoint_indices]
        X_mat = jnp.repeat(X_mat[...,None,:], axis = -2, repeats = weights_on_grid.shape[-2])
        predicted_phi = linalg.broadcast_dot(X_mat, weights_on_grid)
    else:
        predicted_phi = core.slice_volume_by_map(weights_on_grid[...,0], rotation_matrices, image_shape, volume_shape, 'linear_interp')



    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    translated_images = core.translate_images(images, translations, image_shape)
    residuals = jnp.abs(translated_images[...,None] - predicted_phi * CTF[...,None])**2
    
    volume_size = np.prod(volume_shape)
    summed_residuals = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, residuals, gridpoint_indices.reshape(-1), None)

    summed_n = core.summed_adjoint_slice_by_nearest(volume_size, jnp.ones_like(residuals[...,0]), gridpoint_indices.reshape(-1))

    return summed_residuals, summed_n




def compute_residuals_many_weights(experiment_dataset, weights , pol_degree, use_linear_interp ):
    
    batch_size =int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - utils.get_size_in_gb(weights) ) * 5 / np.prod(weights.shape[1:]))

    residuals, summed_n =0, 0
    logger.info(f"batch size in residual computation: {batch_size}")
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
    weights = jnp.asarray(weights)
    for batch, indices in data_generator:
        # Only place where image mask is used ?
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)

        residuals_t, summed_n_t = compute_residuals_batch_many_weights(batch, weights,
                                            experiment_dataset.rotation_matrices[indices],
                                            experiment_dataset.translations[indices], 
                                            experiment_dataset.CTF_params[indices], 
                                            experiment_dataset.upsampled_volume_shape, 
                                            experiment_dataset.image_shape,
                                            experiment_dataset.CTF_fun, experiment_dataset.voxel_size, pol_degree, use_linear_interp )
        residuals += residuals_t
        summed_n += summed_n_t

    return residuals , summed_n


def compute_residuals_many_weights_in_weight_batch(experiment_dataset, weights, pol_degree, use_linear_interp = False ):
    
    # Keep half memory free for other stuff

    n_batches = utils.get_size_in_gb(weights) / (0.5 * utils.get_gpu_memory_total() )
    weight_batch_size = np.floor(weights.shape[-2] / n_batches).astype(int)
    n_batches = np.ceil(weights.shape[-2] / weight_batch_size).astype(int)

    logger.info(f"number of batches in residual computation: {n_batches}, batch size: {weight_batch_size}")
    residuals = []
    for k in range(n_batches):
        ind_st, ind_end = utils.get_batch_of_indices(weights.shape[-2], weight_batch_size, k)
        res, _ = compute_residuals_many_weights(experiment_dataset,
                                                weights[:,ind_st:ind_end],
                                                pol_degree, use_linear_interp )
        residuals.append(res)

    return np.concatenate(residuals, axis = -1), 0



# ### PICK H BASED ON ASYMPTOTICS

# ## High level functions
# # Integral of u^2 K(u) du over R^3
# mu_2_kernel_statistics = { 'cube' : 1/27, 'Epanechnikov' : 3/5 } 

# # Integral of K(u)^2 du over R^3 . K(u) is normalized such that integral K(u) du = 1 
# R_kernel_statistics = { 'cube': 1/8,  'ball' : 4/3 * np.pi , 'Epanechnikov' : 1/5 } 

# # mu_2_kernel_statistics = { 'uniform' : 1/2, 'Epanechnikov' : 3/5 } 
# def Epanechnikov_kernel(dist_squared, h=1):
#     return 3/4 * jnp.where( dist_squared < h,  1- dist_squared/h, jnp.zeros_like(dist_squared) )

# def uniform_kernel(dist_squared, h=1):
#     return jnp.where( dist_squared < h,  jnp.ones_like(dist_squared), jnp.zeros_like(dist_squared) )

# def predict_optimal_h_value(noise_variance_over_density, hessian_norm, kernel = 'cube' ):
#     R = R_kernel_statistics[kernel]
#     mu_2 = mu_2_kernel_statistics[kernel]
#     # h = (R * noise_variance_over_density / mu_2 / hessian_norm_squared)**(1/5)
#     # Theorem 6.1 https://bookdown.org/egarpor/PM-UC3M/npreg-kre.html
#     B_p_squared =  ( mu_2 *  hessian_norm /2)**2
#     d = 3
#     h = (d * R * noise_variance_over_density / ( 4 * B_p_squared))**(1/(4 + d))
#     # B_p^2 * h^4  + noise_variance_over_density * R / h^d 
#     # Deriv is :
#     # B_p^2 * 4 h^3  + noise_variance_over_density * R * (-d) * h^(-d-1) = 0 
#     # Solving for h:
#     # h = (d * R * density_over_variance / (B_p^2 * 4) )**(1/(4 + d)) 
#     return h

def compute_gradient(x):
    gradients = jnp.gradient(x)
    gradients = jnp.stack(gradients, axis=0)
    return gradients

def compute_gradient_norm_squared(x):
    gradients = compute_gradient(x)
    grad_norm = jnp.linalg.norm(gradients, axis = (0), ord=2)**2
    return grad_norm

def compute_hessian(x):
    gradients = jnp.gradient(x)
    hessians = [jnp.gradient(dx) for dx in gradients ]
    hessians = np.stack(hessians, axis=0)
    return hessians

def compute_hessian_norm_squared(x):
    hessians = compute_hessian(x)
    return jnp.linalg.norm(hessians, axis = (0,1), ord =2)**2

