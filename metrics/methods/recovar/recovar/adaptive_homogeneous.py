# import logging
# import jax.numpy as jnp
# import numpy as np
# import jax, functools, time

# from recovar import core, regularization, constants, noise
# from recovar.fourier_transform_utils import fourier_transform_utils
# ftu = fourier_transform_utils(jnp)
# from recovar import utils

# logger = logging.getLogger(__name__)

# # This is a highly experimental feature.

# ## High level functions


# def predict_optimal_h_value(summed_grad_terms ,summed_squared_weights, noise_variance, grad_norm, max_dist=1,order =1 ):
    
#     # A random divided by 10 term here...
#     normalized_c1 = (grad_norm * summed_grad_terms )**2 / 10
#     normalized_c2 =  noise_variance* (summed_squared_weights * max_dist**3)
#     h = 1/((2*normalized_c1/ (3* normalized_c2)))**(1/5)
#     lower_bound = 0.5
#     h = np.where(h > lower_bound, h,  np.ones_like(h)* lower_bound)
#     return h

# def compute_gradient_norm(x):
#     gradients = jnp.gradient(x)
#     gradients = jnp.stack(gradients, axis=0)
#     grad_norm = jnp.linalg.norm(gradients, axis = (0), ord=2)
#     # grad_norm2 = scipy.ndimage.maximum_filter(grad_norm, size =2)
#     return grad_norm

# batch_compute_gradient_norm = jax.vmap(compute_gradient_norm, in_axes = (0,))
# batch_make_radial_image = jax.vmap(utils.make_radial_image, in_axes = (0, None, None))

# def compute_with_adaptive_discretization(cryo, mean_lhs, mean_prior, mean_estimate, noise_variance, batch_size, image_weights):

#     num_reconstructions = image_weights.shape[0]
#     num_recon_by_shape = [num_reconstructions, *cryo.volume_shape]

#     # lhs, rhs = solve_least_squares_mean_iteration(cryo , np.inf, 1,  1000, None, image_weights = None, disc_type = 'nearest', return_lhs_rhs = True )
#     # x = (rhs[0]/(lhs[0] + 1 /tau_prior))

#     bias_fac = (mean_lhs/ (mean_lhs + 1/mean_prior)).real
#     grad_norm = batch_compute_gradient_norm(mean_estimate.reshape(num_recon_by_shape)).reshape(num_reconstructions, -1)
#     grad_norm_radial_mod = regularization.batch_average_over_shells(grad_norm**1, cryo.volume_shape, 0)/ np.sqrt(regularization.batch_average_over_shells(bias_fac, cryo.volume_shape, 0))
#     grad_norm_est = batch_make_radial_image(grad_norm_radial_mod, cryo.volume_shape, True)
#     vol, h_adapt = adaptive_discretization(cryo, grad_norm_est, mean_prior/ noise_variance, noise_variance,batch_size, image_weights)
#     # vol, h_adapt, prior = adaptive_discretization(cryo, grad_norm_est, mean_prior, noise_variance)
#     return vol, h_adapt


# def adaptive_discretization(cryo, grad_norm, prior, noise_variance, batch_size_inp, image_weights):
    
#     num_reconstructions = image_weights.shape[0]
#     summed_squared_weights = {}
#     xs = {}
#     summed_grad_terms = {}
#     order =0
#     dist = 1
#     print("here BEFORE,", batch_size_inp)
#     print("here BEFORE,", utils.report_memory_device())

#     memory_to_use = utils.get_gpu_memory_total() -  utils.get_size_in_gb(prior) * 20 
#     print("mem to use,", memory_to_use)
#     assert memory_to_use > 0, "reduce number of volumes computed at once"
#     batch_size_inp = 2 * utils.get_image_batch_size(cryo.grid_size, memory_to_use) * 20

#     batch_size = np.ceil(batch_size_inp/ (1*2 + 1)**3 /3 / num_reconstructions).astype(int)
#     logger.info(f"batch size in reweighting: {batch_size}")
#     print("here 1,batch_size_inp", batch_size_inp)
#     print("here 1,batch_size", batch_size)
#     utils.report_memory_device()

#     # A first pass to compute optimal h
#     xs[(order, dist)], _, summed_squared_weights[(order, dist)], _, _, summed_grad_terms[(order, dist)], bias_fac = solve_least_squares_mean_iteration_second_order(
#     cryo ,prior * 0, 1e-8,  batch_size, None, image_weights = image_weights, disc_type = None, return_lhs_rhs = False, 
#     grid_dist = np.ceil(dist), max_dist=dist*np.ones([num_reconstructions, cryo.volume_size], dtype = cryo.dtype_real), order = order  )
        
#     h_adapt = predict_optimal_h_value(summed_grad_terms[(order, dist)] ,summed_squared_weights[(order, dist)], noise_variance, grad_norm, max_dist=1,order =order )
#     # regularization.average_over_shells()

#     print("weight,", batch_size)

#     # The actual computation
#     max_grid_dist = 2#3 if cryo.grid_size < 256 else 2

#     batch_size = np.ceil(batch_size_inp/ (max_grid_dist*2 + 1)**3 /3/ num_reconstructions).astype(int)
#     logger.info(f"batch size in reweighting: {batch_size}")
#     print("here 3,", batch_size)

#     dist = 'a'
#     xs[(order, dist)], _, summed_squared_weights[(order, dist)], _, _ , summed_grad_terms[(order, dist)], _ = solve_least_squares_mean_iteration_second_order(
#     cryo ,1/prior, 1e-8,  batch_size, None, image_weights = image_weights, disc_type = None, return_lhs_rhs = False, 
#     grid_dist = max_grid_dist, max_dist=h_adapt, order = order  )

#     print("here 4 ,", batch_size)
#     print("est 4 ")
#     utils.report_memory_device()
#     return xs[(order, dist)], h_adapt

# ## Low level functions

# def make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun):

#     grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     # Discretized grid points
#     # This could be done more efficiently
#     grid_points_coords_nearest = core.round_to_int(grid_points_coords)
#     differences = grid_points_coords - grid_points_coords_nearest
#     C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
#     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     C_mat *= CTF[...,None]

#     return C_mat, grid_point_vec_indices


# def make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist, order=2):
#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
#     differences =  grid_points_coords[...,None,:] - near_frequencies
#     distances = jnp.linalg.norm(differences, axis = -1)
#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

#     # if a
#     max_dist_this = max_dist[...,near_frequencies_vec_indices.reshape(-1)].reshape([max_dist.shape[0], *near_frequencies_vec_indices.shape])
#     # print(jnp.std(max_dist_this),jnp.mean(max_dist_this) )
#     # import pdb; pdb.set_trace()
#     valid_points = (distances < max_dist_this) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)
    
#     # This could be done more efficiently
#     if order==2:
#         C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
#         CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#         C_mat *= CTF[...,None,None]
#         C_mat = C_mat #* valid_points[...,None]
#     else:
#         CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#         C_mat = CTF[...,None]
#         C_mat = C_mat #* valid_points[...]

#     # C_mat_veced = C_mat#.reshape([C_mat.shape[0]* C_mat.shape[1]*C_mat.shape[2], C_mat.shape[3]]) 
    
#     near_frequencies_vec_indices = near_frequencies_vec_indices#.reshape(-1)

#     return C_mat, near_frequencies_vec_indices, valid_points


# def make_kernel_estimator(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist, order=2, kernel_fun = None):


#     grid_points_coords = core.batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     near_frequencies = core.find_frequencies_within_grid_dist(grid_points_coords, grid_dist)
#     differences =  grid_points_coords[...,None,:] - near_frequencies

#     evaluate_kernel = kernel_fun(differences)
#     distances = jnp.linalg.norm(differences, axis = -1)
#     # near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)

#     # if a
#     max_dist_this = max_dist[...,near_frequencies_vec_indices.reshape(-1)].reshape([max_dist.shape[0], *near_frequencies_vec_indices.shape])
#     # print(jnp.std(max_dist_this),jnp.mean(max_dist_this) )
#     # import pdb; pdb.set_trace()
#     valid_points = (distances < max_dist_this) * core.check_vol_indices_in_bound(near_frequencies,volume_shape[0])

#     near_frequencies_vec_indices = core.vol_indices_to_vec_indices(near_frequencies, volume_shape)


#     # # This could be done more efficiently
#     # if order==2:
#     #     C_mat = jnp.concatenate([jnp.ones_like(differences[...,0:1]), differences], axis = -1)
#     #     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     #     C_mat *= CTF[...,None,None]
#     #     C_mat = C_mat #* valid_points[...,None]
#     # else:
#     #     CTF = CTF_fun( CTF_params, image_shape, voxel_size)
#     #     C_mat = CTF[...,None]
#     #     C_mat = C_mat #* valid_points[...]

#     # # C_mat_veced = C_mat#.reshape([C_mat.shape[0]* C_mat.shape[1]*C_mat.shape[2], C_mat.shape[3]]) 
    
#     # near_frequencies_vec_indices = near_frequencies_vec_indices#.reshape(-1)

#     return C_mat, near_frequencies_vec_indices, valid_points







# @functools.partial(jax.jit, static_argnums = [3,4,5,6,7])    
# def compute_weight_matrix_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist, image_weights ):
#     volume_size = np.prod(np.array(volume_shape))
#     if grid_dist is None:
#         C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
#     else:
#         C_mat, grid_point_indices, valid_points = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)

#     C_mat_outer = broadcast_outer(C_mat, C_mat)
#     # RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer.reshape(-1,C_mat_outer.shape[-2]*C_mat_outer.shape[-1] ), grid_point_indices.reshape(-1))
#     # return RR
#     return batch_compute_weight_matrix_inner_last_step(C_mat_outer, grid_point_indices, image_weights, valid_points, volume_size)

# @functools.partial(jax.jit, static_argnums = [4])    
# def compute_weight_matrix_inner_last_step( C_mat_outer, grid_point_indices, image_weights, valid_points, volume_size ):
#     C_mat_outer = multiply_along_axis(C_mat_outer,image_weights, 0 ) * valid_points[...,None,None]

#     # C_mat_outer *= image_weights
#     RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer.reshape(-1,C_mat_outer.shape[-2]*C_mat_outer.shape[-1] ), grid_point_indices.reshape(-1))
#     return RR

# batch_compute_weight_matrix_inner_last_step = jax.vmap(compute_weight_matrix_inner_last_step, in_axes = (None,None,0, 0, None))

# batch_outer = jax.vmap(lambda x: jnp.outer(x,x), in_axes = (0))
# batch_batch_outer = jax.vmap(batch_outer, in_axes = (0))

# def compute_discretization_weights(experiment_dataset, prior, batch_size, order=1, grid_dist = None, max_dist = None, image_weights = None ):

#     order_t = 1   # Only implemented for order 1 but could generalize
#     rr_size = 3*order_t + 1
#     RR = jnp.zeros((experiment_dataset.volume_size, (rr_size)**2 ), dtype = experiment_dataset.dtype_real)
#     # batch_size = utils.get_image_batch_size(experiment_dataset.grid_size, utils.get_gpu_memory_total()) * 3

#     for i in range(utils.get_number_of_index_batch(experiment_dataset.n_images, batch_size)):
#         batch_st, batch_end = utils.get_batch_of_indices(experiment_dataset.n_images, batch_size, i)
#         # Make sure mean_estimate is size # volume_size ?
#         RR_this = compute_weight_matrix_inner(experiment_dataset.rotation_matrices[batch_st:batch_end], experiment_dataset.CTF_params[batch_st:batch_end], experiment_dataset.voxel_size, experiment_dataset.volume_shape, experiment_dataset.image_shape, experiment_dataset.grid_size, experiment_dataset.CTF_fun, grid_dist, max_dist, image_weights[...,batch_st:batch_end])        
#         RR += RR_this

#     RR = RR.reshape([RR_this.shape[0], RR_this.shape[1], rr_size, rr_size])
#     # weights, good_weights = batch_solve_for_weights(RR, prior)
#     weights = jnp.zeros_like(RR, shape = RR.shape[0:-1])
#     # weights, good_weights = batch_solve_for_weights(RR, prior)
#     # If bad weights, just do Weiner filtering with 0th order disc
#     good_weights = None#(good_weights*0).astype(bool)
#     bias_multiple = RR[...,0,0] /(RR[...,0,0] + prior)
#     other_weights = jnp.zeros_like(weights)
#     # weights = jnp.zeros_like(prior)

#     weiner_weights = 1 / (RR[...,0,0] + prior)
#     other_weights = other_weights.at[...,0].set(weiner_weights)
#     if order ==0:
#         weights = other_weights
#         good_weights = None#(good_weights*0).astype(bool)
#     else:
#         weights, good_weights = batch_solve_for_weights(RR, prior)
#         weights = weights.at[~good_weights].set(other_weights[~good_weights])

#     ### THROW AWAY RR
#     RR = None
#     return weights, good_weights, RR, bias_multiple

# @jax.jit
# def solve_for_weights(RR, prior):
#     e1 = jnp.zeros(RR.shape[0])
#     e1 = e1.at[0].set(RR[0,0] / (RR[0,0] + prior))
#     # Maybe could just check for conditioning
#     v = jax.scipy.linalg.solve(RR, e1, lower=False, assume_a='pos')
#     # Probably should replace with numpy.linalg.eigvalsh
#     good_v = jnp.linalg.cond(RR) < 1e4
#     # good_v = jnp.min(jnp.diag(RR)) > constants.ROOT_EPSILON
#     return v, good_v

# batch_solve_for_weights = jax.vmap(solve_for_weights, in_axes = (0,0))


# batch_batch_slice_volume_by_nearest = jax.vmap(core.batch_slice_volume_by_nearest, in_axes = (0, None))

# bcast0_broadcast_dot = jax.vmap(broadcast_dot, in_axes = (None,0))
# @functools.partial(jax.jit, static_argnums = [8,9,10,11,12,13,14])    
# def compute_mean_least_squares_rhs_lhs_with_weights(images, precomp_weights, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun, grid_dist, max_dist ):

#     corrected_images = core.translate_images(images, translations, image_shape)    

#     # Now use weights: w_i = C.^T v
#     if grid_dist is None:
#         C_mat, grid_point_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
#     else:
#         C_mat, grid_point_indices, valid_points = make_C_mat_many_gridpoints(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun, grid_dist, max_dist)
#         corrected_images = corrected_images[...,None]

#     precomp_weights_on_pixel = batch_batch_slice_volume_by_nearest(precomp_weights,grid_point_indices)

#     # precomp_weights_on_pixel = core.batch_slice_volume_by_nearest(precomp_weights,grid_point_indices)
#     # weights = broadcast_dot(C_mat[None], precomp_weights_on_pixel)
#     weights = bcast0_broadcast_dot(C_mat, precomp_weights_on_pixel)

#     residuals = weights[...,None] * C_mat[None]
#     C_squared = jnp.abs(C_mat[...,0])**2

#     volume_size = np.prod((volume_shape))

#     return batch_compute_mean_least_squares_rhs_lhs_with_weights_last_step(image_weights, weights, valid_points, residuals, C_squared, grid_point_indices, corrected_images, volume_size )

# def multiply_along_axis(A, B, axis):
#     return jnp.swapaxes(jnp.swapaxes(A, axis, -1) * B, -1, axis)


# @functools.partial(jax.jit, static_argnums = [7])    
# def compute_mean_least_squares_rhs_lhs_with_weights_last_step(image_weights, weights, valid_points, residuals, C_squared, grid_point_indices, corrected_images, volume_size ):
    
#     weights = multiply_along_axis(weights,image_weights, 0 ) * valid_points
#     residuals = multiply_along_axis(residuals,image_weights, 0 ) * valid_points[...,None]
#     C_squared = multiply_along_axis(C_squared,image_weights, 0 ) * valid_points
#     corrected_images = multiply_along_axis(corrected_images,image_weights, 0 ) * valid_points

#     # weights *= image_weights
#     # residuals *= image_weights
#     # C_squared *= image_weights
#     # corrected_images *= image_weights


#     res_summed = jnp.linalg.norm(residuals[...,1:], axis=-1).reshape(-1)
#     residuals_summed2 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))

#     res_summed = jnp.abs(residuals[...,0]) 

#     residuals_summed1 = core.summed_adjoint_slice_by_nearest(volume_size, res_summed.reshape(-1)**1, grid_point_indices.reshape(-1))

#     C_squared_summed = core.summed_adjoint_slice_by_nearest(volume_size, C_squared.reshape(-1)**1, grid_point_indices.reshape(-1))

#     estimate = core.summed_adjoint_slice_by_nearest(volume_size, corrected_images * weights, grid_point_indices)
#     summed_weights_squared = core.summed_adjoint_slice_by_nearest(volume_size, weights**2, grid_point_indices)
#     return estimate, summed_weights_squared, residuals_summed1, residuals_summed2, C_squared_summed

# batch_compute_mean_least_squares_rhs_lhs_with_weights_last_step = jax.vmap(compute_mean_least_squares_rhs_lhs_with_weights_last_step, in_axes = (0, 0, 0, 0, None, None, None, None))

# # Solves the linear system Dx = b.
# def solve_least_squares_mean_iteration_second_order(experiment_dataset , prior, cov_noise,  batch_size, mean_estimate, image_weights = None, disc_type = None, return_lhs_rhs = False, grid_dist = None, max_dist = None, order=1 ):
#     # all_one_volume = jnp.ones(volume_size)
#     estimate =0;  summed_weights_squared = 0 ; residuals_summed1 = 0; residuals_summed2 = 0; residuals_summed3 = 0
#     # Need to take out RR
#     logger.info(f"batch size in second order: {batch_size}")

#     weights, good_weights, RR, bias_multiple = compute_discretization_weights(experiment_dataset, prior, batch_size, order=order, grid_dist = grid_dist, max_dist = max_dist, image_weights=image_weights )
#     logger.info(f"time done with weights")
#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size)
#     for batch, indices in data_generator:
        
#         # Only place where image mask is used ?
#         batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
                    
#         estimate_this, summed_weights_squared_this, residuals_summed_this1, residuals_summed_this2, residuals_summed_this3  = compute_mean_least_squares_rhs_lhs_with_weights(batch,
#                                          weights,
#                                          experiment_dataset.rotation_matrices[indices], 
#                                          experiment_dataset.translations[indices], 
#                                          experiment_dataset.CTF_params[indices], 
#                                          mean_estimate,
#                                          cov_noise,
#                                          image_weights[...,indices],
#                                          experiment_dataset.voxel_size, 
#                                          experiment_dataset.volume_shape, 
#                                          experiment_dataset.image_shape, 
#                                          experiment_dataset.grid_size, 
#                                          disc_type,
#                                          experiment_dataset.CTF_fun,
#                                          grid_dist,max_dist)
        
#         estimate += estimate_this
#         summed_weights_squared += summed_weights_squared_this
#         residuals_summed1 += residuals_summed_this1
#         residuals_summed2 += residuals_summed_this2
#         residuals_summed3 += residuals_summed_this3
#     logger.info(f"time done with estimate")

#     return estimate, good_weights, summed_weights_squared, weights, residuals_summed1, residuals_summed2, bias_multiple



