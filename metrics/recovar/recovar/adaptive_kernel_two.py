# import logging
# import jax.numpy as jnp
# import numpy as np
# import jax, functools, time

# from recovar import core, regularization, constants, noise, linalg
# from recovar.fourier_transform_utils import fourier_transform_utils
# ftu = fourier_transform_utils(jnp)
# from recovar import utils

# logger = logging.getLogger(__name__)

# # This is a highly experimental feature.

# ## Low level functions
# def make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = 0, dtype = np.float32):

#     grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
#     if pol_degree ==0:
#         return jnp.ones(grid_point_vec_indices.shape, dtype = dtype )[...,None], grid_point_vec_indices

#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size ).astype(dtype)
#     # Discretized grid points
#     # This could be done more efficiently
#     grid_points_coords_nearest = core.round_to_int(grid_points_coords)
#     differences = grid_points_coords - grid_points_coords_nearest

#     if pol_degree==1:
#         X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences**2], axis = -1)
#         return X_mat, grid_point_vec_indices

#     differences_squared = linalg.broadcast_outer(differences, differences)
#     differences_squared = keep_upper_triangular(differences_squared)
#     X_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences, differences_squared], axis = -1)

#     return X_mat, grid_point_vec_indices

    
# def keep_upper_triangular(XWX):
#     iu1 = np.triu_indices(XWX.shape[-1])
#     return XWX[...,iu1[0], iu1[1]]

# def undo_keep_upper_triangular_one(XWX, n):
#     # m = n(n+1)/2
#     # n = (sqrt(8m+1) -1)/2
#     m = XWX.shape[-1]
#     n = np.round((np.sqrt(8*m+1) -1)/2).astype(int)


#     triu_indices = jnp.triu_indices(n)
#     matrix = jnp.empty((n,n), dtype = XWX.dtype)
#     matrix = matrix.at[triu_indices[0], triu_indices[1]].set(XWX)

#     i_lower = jnp.tril_indices(n, -1)
#     matrix = matrix.at[i_lower[0], i_lower[1]].set(matrix.T[i_lower[0], i_lower[1]])
#     return matrix

# undo_keep_upper_triangular = jax.vmap(undo_keep_upper_triangular_one, in_axes = (0,None), out_axes = 0)

# @functools.partial(jax.jit, static_argnums = [5,6,7,8, 10])    
# def precompute_kernel_one_batch(images, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, noise_variance, pol_degree =0):

#     # Precomp piece
#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     volume_size = np.prod(volume_shape)
#     ctf_over_noise_variance = CTF**2 / noise_variance

#     X, grid_point_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = pol_degree)

#     # XWX
#     XWX = linalg.broadcast_outer(X * ctf_over_noise_variance[...,None] , X)
#     XWX = keep_upper_triangular(XWX)
#     XWX_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(
#         volume_size, XWX,
#         grid_point_indices.reshape(-1),None)
    
#     # Z
#     # Z  = X * ctf_over_noise_variance[...,None]
#     # Z_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, Z, grid_point_indices.reshape(-1))

#     # F
#     images = core.translate_images(images, translations, image_shape) 
#     # In my formalism, images === images / CTF soo I guess this is right
#     F = X * (images * CTF / noise_variance)[...,None] 
#     F_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, F, grid_point_indices.reshape(-1),None)

#     # alpha
#     # alpha_summed = core.summed_adjoint_slice_by_nearest(volume_size, ctf_over_noise_variance, grid_point_indices.reshape(-1))
#     return XWX_summed, F_summed

# def get_differences_zero(pol_degree, differences):
#     if pol_degree ==0:
#         differences_zero = jnp.zeros_like(differences[...,0:1])
#     elif pol_degree==1:
#         differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences**2], axis = -1)
#     elif pol_degree==2:
#         differences_squared = linalg.broadcast_outer(differences, differences)
#         differences_squared = keep_upper_triangular(differences_squared)
#         differences_zero = jnp.concatenate([jnp.zeros_like(differences[...,0:1]), differences, differences_squared], axis = -1)
#     else:
#         assert(NotImplementedError)
#     return differences_zero

# # Should this all just be vmapped instead of vmapping each piece? Not really sure.
# # It allocate XWX a bunch of time if I do?
# @functools.partial(jax.jit, static_argnums = [2,5,6,8])
# def compute_estimate_from_precompute_one(XWX, F, max_grid_dist, grid_distances, frequencies, volume_shape, pol_degree, signal_variance, use_regularization):

#     # Might have to cast this back to frequencies vs indices frequencies
#     near_frequencies = core.find_frequencies_within_grid_dist(frequencies, max_grid_dist)
#     differences =  near_frequencies - frequencies[...,None,:]
#     # This is just storing the same array many times over...
#     differences_zero = get_differences_zero(pol_degree, differences)
#     # L_inf norm distances
#     distances = jnp.max(jnp.abs(differences), axis = -1)
#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
#     valid_points = (distances <= grid_distances[...,None] )* core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

    
#     XWX_summed_neighbor = batch_summed_over_indices(XWX, near_frequencies_vec_indices, valid_points)
#     XWX_summed_neighbor = undo_keep_upper_triangular(XWX_summed_neighbor, differences_zero.shape[-1])

#     # NOTE I think Z is just the first row of XWX
#     Z = XWX[...,:differences_zero.shape[-1]]
#     Z_summed_neighbor = batch_Z_grab(Z, near_frequencies_vec_indices, valid_points, differences_zero)

#     # Rank 3 update. Z is real so no need for np.conj
#     XWX_summed_neighbor += Z_summed_neighbor + jnp.swapaxes(Z_summed_neighbor, -1, -2 )

#     ### NOTE THAT ALPHA === XWX[0,0]. TAKE OUT ALPHA
#     alpha = XWX[...,0]
#     XWX_summed_neighbor += batch_summed_scaled_outer(alpha, near_frequencies_vec_indices, differences_zero, valid_points)


#     # Guess this is kinda dumb?
#     F_summed_neighbor = batch_summed_over_indices(F, near_frequencies_vec_indices, valid_points)

#     F0_summed_neighbor = batch_Z_grab(F[...,0:1], near_frequencies_vec_indices, valid_points, differences_zero)[...,0,:]
#     F_summed_neighbor += F0_summed_neighbor

#     if use_regularization:
#         regularization = 1 / signal_variance
#     else:
#         regularization = jnp.zeros(frequencies.shape[0], dtype = XWX_summed_neighbor.dtype)

#     y_and_deriv, good_v, problems = batch_solve_for_m(XWX_summed_neighbor,F_summed_neighbor, regularization )

#     return y_and_deriv, good_v, problems


# def summed_scaled_outer(alpha, indices, differences_zero, valid_points):
#     alpha_slices = alpha[indices] * valid_points
#     return jnp.sum(linalg.multiply_along_axis(linalg.broadcast_outer(differences_zero, differences_zero), alpha_slices, 0 ), axis=0)

# batch_summed_scaled_outer = jax.vmap(summed_scaled_outer, in_axes = (None,0,0, 0))

# # Idk why I can't find a nice syntax to do this.
# slice_first_axis = jax.vmap(lambda vec, indices: vec[indices], in_axes = (-1,None), out_axes=(-1))

# def summed_over_indices(vec, indices, valid):
#     sliced = slice_first_axis(vec,indices)
#     return jnp.sum(sliced * valid[...,None], axis = -2)

# batch_summed_over_indices = jax.vmap(summed_over_indices, in_axes = (None,0,0))

# # It feels very silly to have to do this. But I guess JAX will clean up?
# def Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
#     sliced = slice_first_axis(Z,near_frequencies_vec_indices)
#     return  (sliced * valid_points[...,None]).T @ differences

# batch_Z_grab = jax.vmap(Z_grab, in_axes = (None,0,0,0))

# def one_Z_grab(Z,near_frequencies_vec_indices, valid_points, differences  ):
#     sliced = Z[near_frequencies_vec_indices]#slice_first_axis(Z,near_frequencies_vec_indices)
#     return  (sliced * valid_points) * differences

# batch_one_Z_grab = jax.vmap(one_Z_grab, in_axes = (None,0,0,0))

# @jax.jit
# def solve_for_m(XWX, f, regularization):
    
#     XWX = XWX.at[0,0].add(regularization)
#     # XWX += jnp.diag(regularization)
    
#     dtype_to_solve = np.float64
#     vreal = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.real.astype(dtype_to_solve), lower=False, assume_a='pos')
#     vimag = jax.scipy.linalg.solve(XWX.astype(dtype_to_solve), f.imag.astype(dtype_to_solve), lower=False, assume_a='pos')

#     v = vreal + 1j * vimag
#     good_v = jnp.linalg.cond(XWX) < 1e8

#     e1 = jnp.zeros_like(v)
#     e1 = e1.at[0].set(f[0]/XWX[0,0])

#     v = jnp.where(good_v, v, e1)
#     v = jnp.where(XWX[0,0] > 0, v, jnp.zeros_like(v))
#     v = v.astype(np.complex64)

#     # If condition number is bad, do degree 0 approx?
#     # v = jnp.where(good_v, v, jnp.zeros_like(v))
#     # res = jnp.linalg.norm(XWX @v - f)**1 / jnp.linalg.norm(f)**1#, axis = -1)

#     # v = v.astype(np.complex64)
#     bad_indices = jnp.isnan(v).any(axis=-1) + jnp.isinf(v).any(axis=-1)
#     problem = bad_indices * good_v

#     # problem = (res > 1e-6) * good_v
#     v = jnp.where(bad_indices, jnp.zeros_like(v), v)


#     return v, good_v, problem

# batch_solve_for_m = jax.vmap(solve_for_m, in_axes = (0,0,0))

# # Should this be set by cross validation?

# def precompute_kernel(experiment_dataset, cov_noise,  batch_size, pol_degree=0):    
#     XWX, F = 0,0
#     # Need to take out RR
#     logger.info(f"batch size in precompute kernel: {batch_size}")
#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
#     cov_noise_image = noise.make_radial_noise(cov_noise, experiment_dataset.image_shape)


#     for batch, indices in data_generator:
        
#         batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
    
#         XWX_this, F_this = precompute_kernel_one_batch(batch,
#                                 experiment_dataset.rotation_matrices[indices], 
#                                 experiment_dataset.translations[indices], 
#                                 experiment_dataset.CTF_params[indices], 
#                                 experiment_dataset.voxel_size, 
#                                 experiment_dataset.volume_shape, 
#                                 experiment_dataset.image_shape, 
#                                 experiment_dataset.grid_size, 
#                                 experiment_dataset.CTF_fun,
#                                 cov_noise_image, pol_degree = pol_degree)
#         XWX += XWX_this
#         F += F_this

#     logger.info(f"Done with precompute of kernel")
#     return XWX, F

# # Should pass a list of triplets (pol_degree : int, h : float, regularization : bool)

# def get_default_discretization_params(grid_size):
#     params = []
#     pol_degrees = [0,1] if grid_size <= 512 else [0,]

#     for pol_degree in pol_degrees:
#         for h in [0,1]:
#             params.append((pol_degree, h, True))
#     params.append((0,1,False))
#     params.append((0,2,True))

#     return params
            
# def get_num_params(pol_degree):

#     if pol_degree ==0:
#         return 4
#     elif pol_degree ==1:
#         return 20
#     elif pol_degree ==2:
#         return 10
#     else:
#         raise NotImplementedError

# def small_gram_matrix_size(pol_degree):
#     feature_size = get_feature_size(pol_degree)
#     return (feature_size * (feature_size+1))//2

# def big_grad_matrix_size(pol_degree):
#     return get_feature_size(pol_degree)**2

# def get_feature_size(pol_degree):
#     if pol_degree==0:
#         return 1
#     if pol_degree==1:
#         return 4
#     if pol_degree==2:
#         return 10
#     return pol_degree




# def test_multiple_disc(experiment_dataset, cross_validation_dataset, cov_noise,  batch_size, discretization_params, signal_variance, return_all = False):

#     discretization_params = get_default_discretization_params(experiment_dataset.grid_size) if discretization_params is None else discretization_params

#     max_pol_degree = np.max([ pol_degree for pol_degree, _, _ in discretization_params ])

#     num_params = small_gram_matrix_size(max_pol_degree) + 2 * get_feature_size(max_pol_degree) + 1
#     # num_params = 4 if max_pol_degree == 0 else 20

#     batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() -  num_params * experiment_dataset.volume_size * 8 / 1e9 ) / 2)#/ np.prod(weights.shape[1:])).astype(int)

#     XWX, F = precompute_kernel(experiment_dataset, cov_noise.astype(np.float32),  batch_size, pol_degree=max_pol_degree)
#     n_disc_test = len(discretization_params)


#     weight_size = get_feature_size(max_pol_degree)# 1 if max_pol_degree == 0 else 4
#     weights = np.zeros((n_disc_test, experiment_dataset.volume_size, weight_size), dtype = np.complex64)
#     valid_weights = np.zeros((n_disc_test, experiment_dataset.volume_size), dtype = bool)

#     XWX = np.asarray(XWX)

#     for idx, (pol_degree, h, reg) in enumerate(discretization_params):
#         logger.info(f"computing discretization with params: degree={pol_degree}, h={h}, reg={reg}")
#         reg_used = signal_variance if reg else None
#         weights_this, valid_weights_this = compute_weights_from_precompute(experiment_dataset, XWX, F, reg_used, pol_degree, h)
#         weights[idx,:,:weights_this.shape[-1]] = weights_this
#         valid_weights[idx] = valid_weights_this

#     logger.info(f"Done computing params")
#     weights = weights.swapaxes(0,1)

#     del XWX, F

#     batch_size =int(utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total() - utils.get_size_in_gb(weights) ) * 5 / np.prod(weights.shape[1:]))

#     residuals, _ = compute_residuals_many_weights(cross_validation_dataset, weights,  batch_size, max_pol_degree )
#     residuals_averaged = regularization.batch_average_over_shells(residuals.T, experiment_dataset.volume_shape,0)

#     index_array = jnp.argmin(residuals_averaged, axis = 0)
#     index_array_vol = utils.make_radial_image(index_array, experiment_dataset.volume_shape)

#     disc_choices = np.array(discretization_params)[index_array]
#     weights_opt = jnp.take_along_axis(weights[...,0], np.expand_dims(index_array_vol, axis=-1), axis=-1)

#     logger.info("Done with adaptive disc")
    
#     if return_all:
#         return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged), np.asarray(weights), discretization_params # XWX, Z, F, alpha
#     else:
#         return np.asarray(weights_opt), np.asarray(disc_choices), np.asarray(residuals_averaged)



# def compute_weights_from_precompute(experiment_dataset, XWX, F, signal_variance, pol_degree, h):
#     use_regularization = signal_variance is not None

#     # NOTE the 1.0x is a weird hack to make sure that JAX doesn't compile store some arrays when compiling. I don't know why it does that.
#     threed_frequencies = core.vec_indices_to_vol_indices(np.arange(experiment_dataset.volume_size), experiment_dataset.volume_shape ) * 1.0

#     if type(h) == float or type(h) == int:
#         h_max = h
#         h_ar = h*np.ones(experiment_dataset.volume_size)
#     else:
#         h_max = int(np.max(h))
#         h_ar = h.astype(int) * 1.0

#     # if pol_degree ==0:
#     feature_size = get_feature_size(pol_degree)
#     # og_feature_size = 4

#     XWX = undo_keep_upper_triangular(XWX, 0)
#     XWX = XWX[...,:feature_size, :feature_size]
#     XWX = keep_upper_triangular(XWX)

#     # Z = Z[...,:feature_size]
#     F = F[...,:feature_size]

#     n_pol_param = small_gram_matrix_size(pol_degree) + 2 * get_feature_size(pol_degree) + 1
#     # n_pol_param = 1 if pol_degree == 0 else 10
#     memory_per_pixel = (2*h_max +1)**3 * n_pol_param
#     batch_size = int(utils.get_gpu_memory_total() / (memory_per_pixel * 16 * 20/ 1e9))
#     n_batches = np.ceil(experiment_dataset.volume_size / batch_size).astype(int)

#     weight_size = get_feature_size(pol_degree)
#     reconstruction = np.zeros((experiment_dataset.volume_size, weight_size), dtype = np.complex64)
#     good_pixels = np.zeros((experiment_dataset.volume_size), dtype = bool)
#     logger.info(f"KE batch size: {batch_size}")
#     msgs = 0
#     for k in range(n_batches):
#         ind_st, ind_end = utils.get_batch_of_indices(experiment_dataset.volume_size, batch_size, k)
#         n_pix = ind_end - ind_st
#         signal_variance_this = signal_variance[ind_st:ind_end] if use_regularization else None
#         reconstruction[ind_st:ind_end], good_pixels[ind_st:ind_end], problems = compute_estimate_from_precompute_one(XWX, F, h_max, h_ar[ind_st:ind_end], threed_frequencies[ind_st:ind_end], experiment_dataset.volume_shape, pol_degree =pol_degree, signal_variance=signal_variance_this, use_regularization = use_regularization)
        
#         # if problems.any():
#         #     logger.warning(f"Issues in linalg solve? Problems for {problems.sum() / problems.size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")

#         if jnp.isnan(reconstruction[ind_st:ind_end]).any():
#             logger.warning(f"IsNAN {jnp.isnan(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")
#         # if jnp.isinf(reconstruction[ind_st:ind_end]).any():
#         if problems.any():
#             # logger.warning(f"Issues in linalg solve? Problems for {problems.sum() / problems.size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")
#         # if problems.any():
#             if msgs < 10: 
#                 msgs +=1
#                 logger.warning(f"isinf {jnp.isinf(reconstruction[ind_st:ind_end]).sum() / reconstruction[ind_st:ind_end].size} pixels, pol_degree={pol_degree}, h={h}, reg={use_regularization}")

#             # XWX_undo = undo_keep_upper_triangular(XWX, reconstruction.shape[-1])

#             # recon2 , good_v_v, problems_v = batch_solve_for_m(XWX_undo[ind_st:ind_end], F[ind_st:ind_end], 1/signal_variance_this )

#             # bad_idx = jnp.isinf(reconstruction[ind_st:ind_end]).any(axis=-1)
#             # bad_indices = np.arange(ind_st, ind_end)[bad_idx]
#             # # bad_indices = np.nonzero(bad_idx)[0]
#             # # import pdb; pdb.set_trace()
#             # vmapcond = jax.vmap(jnp.linalg.cond, in_axes = 0)

#             # # print(np.max(vmapcond(XWX_undo[bad_indices])))
#             # x = core.vec_indices_to_frequencies(bad_indices, experiment_dataset.volume_shape)
#             # print(x)
#             # print(np.min(np.linalg.norm(x, axis=-1)))
#             # import pdb; pdb.set_trace()
#     logger.info(f"Done with kernel estimate")
#     return reconstruction, good_pixels#, np.asarray(density_over_noise), XWX, F


# def adaptive_disc(experiment_dataset, cov_noise,  batch_size, pol_degree=0, h =1, signal_variance=None):
#     use_regularization = signal_variance is not None

#     XWX, F = precompute_kernel(experiment_dataset, cov_noise,  batch_size, pol_degree=pol_degree)
#     return compute_weights_from_precompute(experiment_dataset, XWX, F, signal_variance, pol_degree, h)

# @functools.partial(jax.jit, static_argnums = [5,6,7,8,9, 10])    
# def compute_residuals_batch(images, weights, rotation_matrices, translations, CTF_params, volume_shape, image_shape, grid_size, CTF_fun, voxel_size, pol_degree ):

#     X_mat, gridpoint_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = pol_degree)

#     weights_on_grid = weights[gridpoint_indices]
#     predicted_phi = linalg.broadcast_dot(X_mat, weights_on_grid)
#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     translated_images = core.translate_images(images, translations, image_shape)
#     residuals = jnp.abs(translated_images - predicted_phi * CTF)**2
#     volume_size = np.prod(volume_shape)
#     summed_residuals = core.summed_adjoint_slice_by_nearest(volume_size, residuals, gridpoint_indices.reshape(-1))

#     summed_n = core.summed_adjoint_slice_by_nearest(volume_size, jnp.ones_like(residuals), gridpoint_indices.reshape(-1))

#     return summed_residuals, summed_n



# def compute_residuals(experiment_dataset, weights,  batch_size, pol_degree ):
    
#     residuals, summed_n =0, 0
#     logger.info(f"batch size in second order: {batch_size}")
#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)

#     for batch, indices in data_generator:
#         # Only place where image mask is used ?
#         batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)

#         residuals_t, summed_n_t = compute_residuals_batch(batch, weights,
#                                             experiment_dataset.rotation_matrices[indices],
#                                             experiment_dataset.translations[indices], 
#                                             experiment_dataset.CTF_params[indices], 
#                                             experiment_dataset.volume_shape, 
#                                             experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, experiment_dataset.voxel_size, pol_degree )
#         residuals += residuals_t
#         summed_n += summed_n_t

#     return residuals,  summed_n #/ (summed_n + constants.ROOT_EPSILON)


# batch_over_weight_dot = jax.vmap(linalg.broadcast_dot, in_axes = (None,-2))#, in_axes = )

# # @functools.partial(jax.jit, static_argnums = [5,6,7,8,9, 10])    
# def compute_residuals_batch_many_weights(images, weights, rotation_matrices, translations, CTF_params, volume_shape, image_shape, grid_size, CTF_fun, voxel_size, pol_degree ):

#     X_mat, gridpoint_indices = make_X_mat(rotation_matrices, volume_shape, image_shape, grid_size, pol_degree = pol_degree)
#     weights_on_grid = weights[gridpoint_indices]

#     X_mat = jnp.repeat(X_mat[...,None,:], axis = -2, repeats = weights_on_grid.shape[-2])

#     predicted_phi = linalg.broadcast_dot(X_mat, weights_on_grid)

#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     translated_images = core.translate_images(images, translations, image_shape)
#     residuals = jnp.abs(translated_images[...,None] - predicted_phi * CTF[...,None])**2
    
#     volume_size = np.prod(volume_shape)
#     summed_residuals = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, residuals, gridpoint_indices.reshape(-1),None)

#     summed_n = core.summed_adjoint_slice_by_nearest(volume_size, jnp.ones_like(residuals[...,0]), gridpoint_indices.reshape(-1))

#     return summed_residuals, summed_n


# def compute_residuals_many_weights(experiment_dataset, weights,  batch_size, pol_degree ):
    
#     residuals, summed_n =0, 0
#     logger.info(f"batch size in residual computation: {batch_size}")
#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
#     weights = jnp.asarray(weights)
#     for batch, indices in data_generator:
#         # Only place where image mask is used ?
#         batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)

#         residuals_t, summed_n_t = compute_residuals_batch_many_weights(batch, weights,
#                                             experiment_dataset.rotation_matrices[indices],
#                                             experiment_dataset.translations[indices], 
#                                             experiment_dataset.CTF_params[indices], 
#                                             experiment_dataset.volume_shape, 
#                                             experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, experiment_dataset.voxel_size, pol_degree )
#         residuals += residuals_t
#         summed_n += summed_n_t

#     return residuals , summed_n# / (summed_n[...,None] + constants.ROOT_EPSILON)



# # ### PICK H BASED ON ASYMPTOTICS

# # ## High level functions
# # # Integral of u^2 K(u) du over R^3
# # mu_2_kernel_statistics = { 'cube' : 1/27, 'Epanechnikov' : 3/5 } 

# # # Integral of K(u)^2 du over R^3 . K(u) is normalized such that integral K(u) du = 1 
# # R_kernel_statistics = { 'cube': 1/8,  'ball' : 4/3 * np.pi , 'Epanechnikov' : 1/5 } 

# # # mu_2_kernel_statistics = { 'uniform' : 1/2, 'Epanechnikov' : 3/5 } 
# # def Epanechnikov_kernel(dist_squared, h=1):
# #     return 3/4 * jnp.where( dist_squared < h,  1- dist_squared/h, jnp.zeros_like(dist_squared) )

# # def uniform_kernel(dist_squared, h=1):
# #     return jnp.where( dist_squared < h,  jnp.ones_like(dist_squared), jnp.zeros_like(dist_squared) )

# # def predict_optimal_h_value(noise_variance_over_density, hessian_norm, kernel = 'cube' ):
# #     R = R_kernel_statistics[kernel]
# #     mu_2 = mu_2_kernel_statistics[kernel]
# #     # h = (R * noise_variance_over_density / mu_2 / hessian_norm_squared)**(1/5)
# #     # Theorem 6.1 https://bookdown.org/egarpor/PM-UC3M/npreg-kre.html
# #     B_p_squared =  ( mu_2 *  hessian_norm /2)**2
# #     d = 3
# #     h = (d * R * noise_variance_over_density / ( 4 * B_p_squared))**(1/(4 + d))
# #     # B_p^2 * h^4  + noise_variance_over_density * R / h^d 
# #     # Deriv is :
# #     # B_p^2 * 4 h^3  + noise_variance_over_density * R * (-d) * h^(-d-1) = 0 
# #     # Solving for h:
# #     # h = (d * R * density_over_variance / (B_p^2 * 4) )**(1/(4 + d)) 

# #     return h

# # def compute_gradient_norm(x):
# #     gradients = jnp.gradient(x)
# #     gradients = jnp.stack(gradients, axis=0)
# #     grad_norm = jnp.linalg.norm(gradients, axis = (0), ord=2)
# #     # grad_norm2 = scipy.ndimage.maximum_filter(grad_norm, size =2)
# #     return grad_norm

# # def compute_hessian_norm(x):
# #     # x = x.reshape(volume_shape)
# #     gradients = jnp.gradient(x)
# #     # gradients = jnp.stack(gradients, axis=0)

# #     hessians = [jnp.gradient(dx) for dx in gradients ]
# #     hessians = np.stack(hessians, axis=0)
# #     return jnp.linalg.norm(hessians, axis = (0,1), ord =2)


# # batch_compute_gradient_norm = jax.vmap(compute_gradient_norm, in_axes = (0,))
# # batch_make_radial_image = jax.vmap(utils.make_radial_image, in_axes = (0, None, None))