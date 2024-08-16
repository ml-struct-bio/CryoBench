import logging
import jax.numpy as jnp
import numpy as np
import jax, time
import functools
from recovar import core, covariance_core, regularization, utils, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)
## There is currently two ways to estimate noise:
# White and radial
# From my observations, white seems fine for most datasets but some need some other noise distribution
# Neither solution implemented here are very satisfying. Guessing noise in presence of heterogeneity is not trivial, since the residual doesn't seem like the correct way to do it.
# It makes me think we should have "noise pickers".



## should probably clean up all this?

# Perhaps it should be mean at low freq and median at high freq?
mean_fn = np.mean

def estimate_noise_variance(experiment_dataset, batch_size):
    sum_sq = 0

    
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []

    for batch, _ in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch)
        sum_sq += jnp.sum(np.abs(batch)**2, axis =0)

    mean_PS =  sum_sq / experiment_dataset.n_images
    cov_noise_mask = jnp.median(mean_PS)

    average_image_PS = regularization.average_over_shells(mean_PS, experiment_dataset.image_shape)

    return np.asarray(cov_noise_mask.astype(experiment_dataset.dtype_real)), np.asarray(np.array(average_image_PS).astype(experiment_dataset.dtype_real))
    

def estimate_white_noise_variance_from_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):
    _, predicted_pixel_variances, _ = estimate_noise_variance_from_outside_mask_v2(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp')
    return np.median(predicted_pixel_variances)


def estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []
    image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    masked_image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    for batch, batch_ind in data_generator:
        masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask_inner(batch, 
                    volume_mask, experiment_dataset.rotation_matrices[batch_ind], 
                    experiment_dataset.translations[batch_ind], 
                    image_mask, 
                    experiment_dataset.volume_mask_threshold, 
                    experiment_dataset.image_shape, 
                    experiment_dataset.volume_shape, 
                    experiment_dataset.grid_size, 
                    experiment_dataset.padding, 
                    disc_type, 
                    experiment_dataset.image_stack.process_images)
        image_PSs[batch_ind] = np.array(image_PS)
        masked_image_PSs[batch_ind] = np.array(masked_image_PS)

    return masked_image_PSs, image_PSs



def estimate_noise_variance_from_outside_mask_v2(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp'):

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    # all_shell_avgs = []
    images_estimates = np.empty([experiment_dataset.n_images, *experiment_dataset.image_shape])

    image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
    top_fraction = 0
    kernel_sq_sum =0 
    for batch, batch_ind in data_generator:
        top_fraction_this, kernel_sq_sum_this, per_image_est = estimate_noise_variance_from_outside_mask_inner_v2(batch, 
                    volume_mask, experiment_dataset.rotation_matrices[batch_ind], 
                    experiment_dataset.translations[batch_ind], 
                    image_mask, 
                    experiment_dataset.volume_mask_threshold, 
                    experiment_dataset.image_shape, 
                    experiment_dataset.volume_shape, 
                    experiment_dataset.grid_size, 
                    experiment_dataset.padding, 
                    disc_type, 
                    experiment_dataset.image_stack.process_images)
        top_fraction += top_fraction_this
        kernel_sq_sum+= kernel_sq_sum_this
        images_estimates[batch_ind] = np.array(per_image_est)
        # image_PSs[batch_ind] = np.array(image_PS)
        # masked_image_PSs[batch_ind] = np.array(masked_image_PS)

    predicted_pixel_variances= top_fraction / kernel_sq_sum
    predicted_pixel_variances = jnp.fft.ifft2( predicted_pixel_variances).real * experiment_dataset.image_size

    # per_image_est = jnp.fft.ifft2( per_image_est).real * experiment_dataset.image_size

    pred_noise = regularization.average_over_shells(predicted_pixel_variances, experiment_dataset.image_shape, 0) 
    return pred_noise, predicted_pixel_variances, per_image_est




@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10,11])    
def estimate_noise_variance_from_outside_mask_inner_v2(batch, volume_mask, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, process_fn):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften =5)
    # image_mask = image_mask > 0
    
    # Invert mask
    image_mask = 1 - image_mask

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    return get_masked_image_noise_fractions(batch, image_mask, image_shape)



def get_masked_image_noise_fractions(images, image_masks, image_shape):
    images = covariance_core.apply_image_masks(images, image_masks, image_shape)

    masked_variance = jnp.abs(images.reshape([-1, *image_shape]))**2
    masked_variance_ft = jnp.fft.fft2(masked_variance)

    # mask = image_mask
    f_mask = jnp.fft.fft2(image_masks)
    kernels = jnp.fft.ifft2(jnp.abs(f_mask)**2)
    kernel_sq_sum = jnp.sum(jnp.abs(kernels)**2, axis=0)
    top_fraction= jnp.sum(masked_variance_ft * jnp.conj(kernels), axis=0) 

    # get a per image one
    kernels_bad = jnp.abs(kernels)  < constants.EPSILON
    kernels = jnp.where(kernels_bad, jnp.ones_like(kernels_bad) , kernels )
    per_image_estimate = jnp.where( kernels_bad, jnp.zeros_like(masked_variance_ft),  masked_variance_ft / kernels )

    return top_fraction, kernel_sq_sum, jnp.fft.ifft2(per_image_estimate).real * np.prod(image_shape)

def get_masked_noise_variance_from_noise_variance(image_masks, image_cov_noise, image_shape):

    f_mask = jnp.fft.ifft2(image_masks)
    f_mask = jnp.fft.fft2(jnp.abs(f_mask)**2)

    image_cov_noise_ft = jnp.fft.fft2(image_cov_noise.reshape(image_shape))
    masked_noise_variance = jnp.fft.ifft2( f_mask * image_cov_noise_ft )

    return masked_noise_variance


# def upper_bound_noise_by_reprojected_mean(experiment_dataset, mean_estimate, volume_mask, batch_size, disc_type = 'linear_interp'):

#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
#     # all_shell_avgs = []
#     image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

#     masked_image_PSs = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)

#     soften_mask =5 # no softening
#     image_mask = jnp.ones_like(experiment_dataset.image_stack.mask)
#     for batch, batch_ind in data_generator:

#         image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
#                                               experiment_dataset.rotation_matrices[batch_ind], 
#                                               experiment_dataset.image_stack.mask, 
#                                               experiment_dataset.volume_mask_threshold, 
#                                               experiment_dataset.image_shape, 
#                                               experiment_dataset.volume_shape, experiment_dataset.grid_size, 
#                                             experiment_dataset.padding, disc_type, soften = soften_mask )#*0 + 1

#         images = experiment_dataset.image_stack.process_images(batch)
#         images = covariance_core.get_centered_images(images, mean_estimate,
#                                      experiment_dataset.CTF_params[batch_ind],
#                                      experiment_dataset.rotation_matrices[batch_ind],
#                                      experiment_dataset.translations[batch_ind],
#                                      experiment_dataset.image_shape, 
#                                      experiment_dataset.volume_shape,
#                                      experiment_dataset.grid_size, 
#                                      experiment_dataset.voxel_size,
#                                      experiment_dataset.CTF_fun,
#                                      disc_type )
#         image_PSs[batch_ind] = regularization.batch_average_over_shells(jnp.abs(images)**2, experiment_dataset.image_shape, 0)
#         masked_images = covariance_core.apply_image_masks(images, image_mask, experiment_dataset.image_shape)  
#         # image_size = batch.shape[-1]
#         # # Integral of mask:
#         image_mask_sums = jnp.sum(image_mask, axis =(-2, -1)) / experiment_dataset.image_size
#         masked_image_PSs[batch_ind] = regularization.batch_average_over_shells(jnp.abs(masked_images)**2, experiment_dataset.image_shape, 0) / image_mask_sums[:,None]


#     return masked_image_PSs, image_PSs


@functools.partial(jax.jit, static_argnums = [5,6,7,8,9,10,11])    
def estimate_noise_variance_from_outside_mask_inner(batch, volume_mask, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, padding, disc_type, process_fn):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften =10)
    # image_mask = image_mask > 0
    
    # Invert mask
    image_mask = 1 - image_mask

    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)

    image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    image_size = batch.shape[-1]
    # Integral of mask:
    image_mask_2 = ftu.get_dft2(image_mask)
    image_mask_sums = jnp.sum(jnp.abs(image_mask_2)**2, axis =(-2, -1)) / image_size**2 
    masked_image_PS = regularization.batch_average_over_shells(jnp.abs(batch)**2, image_shape, 0) / image_mask_sums[:,None]

    # masked_image_PS = masked_image_PS.at[:,0].set(masked_image_PS[:,0] *  image_mask_sums[:] )
    # import pdb; pdb.set_trace()

    return masked_image_PS, image_PS
    

# # Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
# def estimate_radial_noise_upper_bound_from_inside_mask(experiment_dataset, mean_estimate, volume_mask, batch_size):
#     masked_image_PS, image_PS = upper_bound_noise_by_reprojected_mean(experiment_dataset, mean_estimate , volume_mask, batch_size, disc_type = 'linear_interp')
#     return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0), mean_fn(image_PS, axis =0), np.std(image_PS, axis =0)


def estimate_radial_noise_upper_bound_from_inside_mask_v2(experiment_dataset, mean_estimate, volume_mask, batch_size):
    noise_dist, per_pixel, aa = get_average_residual_square_just_mean(experiment_dataset, volume_mask, mean_estimate, batch_size, disc_type = 'linear_interp')
    return noise_dist, per_pixel, aa


# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_radial_noise_statistic_from_outside_mask(experiment_dataset, volume_mask, batch_size):
    masked_image_PS, image_PS = estimate_noise_variance_from_outside_mask(experiment_dataset, volume_mask, batch_size, disc_type = 'linear_interp')
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0), mean_fn(image_PS, axis =0), np.std(image_PS, axis =0)



def make_radial_noise(average_image_PS, image_shape):
    # If you pass a scalar, return a constant
    if average_image_PS.size == 1:
        return np.ones(image_shape, dtype =average_image_PS.dtype ) * average_image_PS
    
    return utils.make_radial_image(average_image_PS, image_shape, extend_last_frequency = True)




# Assume noise constant across images and within frequency bands. Estimate the noise by the outside of the mask, and report some statistics
def estimate_noise_from_heterogeneity_residuals_inside_mask(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    masked_image_PS =  get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type )
    return mean_fn(masked_image_PS, axis =0), np.std(masked_image_PS, axis =0)

# @functools.partial(jax.jit, static_argnums = [5])    
def get_average_residual_square(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    
    # basis_size = basis.shape[-1]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    residual_squared = jnp.zeros(experiment_dataset.image_stack.image_size, dtype = basis.dtype)
    # residuals_squared_per_image = jnp.zeros_like(residual_squared, shape = experiment_dataset.n_images, dtype = basis.dtype)
    all_averaged_residual_squared = np.empty((experiment_dataset.n_images,experiment_dataset.grid_size//2-1), dtype = experiment_dataset.dtype_real)
    # all_averaged_residual_squared = 
    # soften_mask = -1
    basis = jnp.asarray(basis.T)
    for batch, batch_image_ind in data_generator:
        # batch = experiment_dataset.image_stack.process_images(batch)
        averaged_residual_squared = get_average_residual_square_inner(batch, mean_estimate, volume_mask, 
                                                                        basis,
                                                                        experiment_dataset.CTF_params[batch_image_ind],
                                                                        experiment_dataset.rotation_matrices[batch_image_ind],
                                                                        experiment_dataset.translations[batch_image_ind],
                                                                        experiment_dataset.image_stack.mask,
                                                                        experiment_dataset.volume_mask_threshold,
                                                                        experiment_dataset.image_shape, 
                                                                        experiment_dataset.volume_shape, 
                                                                        experiment_dataset.grid_size, 
                                                                        experiment_dataset.voxel_size, 
                                                                        experiment_dataset.padding, 
                                                                        disc_type, 
                                                                        experiment_dataset.image_stack.process_images,
                                                                        experiment_dataset.CTF_fun, 
                                                                        contrasts[batch_image_ind], basis_coordinates[batch_image_ind])
        all_averaged_residual_squared[batch_image_ind] = np.array(averaged_residual_squared)

    return all_averaged_residual_squared


def get_average_residual_square_inner(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, process_fn, CTF_fun, contrasts,basis_coordinates):
    
    # Memory to do this is ~ size(volume_mask) * batch_size
    image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                          rotation_matrices,
                                          image_mask, 
                                          volume_mask_threshold,
                                          image_shape, 
                                          volume_shape, grid_size, 
                                          padding, 
                                          disc_type, soften = 5 )
    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)

    projected_mean = core.forward_model_from_map(mean_estimate,
                                         CTF_params,
                                         rotation_matrices, 
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                           
                                          )

    projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)

    ## DO MASK BUSINESS HERE.
    batch = covariance_core.apply_image_masks(batch, image_mask, image_shape)
    # projected_mean = covariance_core.apply_image_masks(projected_mean, image_mask, image_shape)
    AUs = covariance_core.batch_over_vol_forward_model_from_map(basis,
                                         CTF_params, 
                                         rotation_matrices,
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type ) 
    # Apply mask on operator
    AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, image_shape )
    AUs = AUs.transpose(1,2,0)
    image_mask_sums = jnp.sum(image_mask, axis =(-2, -1)) / batch.shape[-1]

    predicted_images = contrasts[...,None] * (jax.lax.batch_matmul(AUs, basis_coordinates[...,None])[...,0] + projected_mean)
    residual_squared = jnp.abs(batch - predicted_images)**2    / image_mask_sums[...,None]
    averaged_residual_squared = regularization.batch_average_over_shells(residual_squared, image_shape,0) 

    return averaged_residual_squared#, averaged_residual_squared
    


# # @functools.partial(jax.jit, static_argnums = [5])    
# def get_average_residual_square_v3(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp', noise_var = None, index_subset = None):
    
#     # basis_size = basis.shape[-1]
#     data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
#     residual = 0 
#     basis = jnp.asarray(basis.T)
#     for batch, batch_image_ind in data_generator:
#         # batch = experiment_dataset.image_stack.process_images(batch)
#         residual += get_residual_square_inner_v2(batch, mean_estimate, volume_mask, 
#                                                                         basis,
#                                                                         experiment_dataset.CTF_params[batch_image_ind],
#                                                                         experiment_dataset.rotation_matrices[batch_image_ind],
#                                                                         experiment_dataset.translations[batch_image_ind],
#                                                                         experiment_dataset.image_stack.mask,
#                                                                         experiment_dataset.volume_mask_threshold,
#                                                                         experiment_dataset.image_shape, 
#                                                                         experiment_dataset.volume_shape, 
#                                                                         experiment_dataset.grid_size, 
#                                                                         experiment_dataset.voxel_size, 
#                                                                         experiment_dataset.padding, 
#                                                                         disc_type, 
#                                                                         experiment_dataset.image_stack.process_images,
#                                                                         experiment_dataset.CTF_fun, 
#                                                                         contrasts[batch_image_ind], basis_coordinates[batch_image_ind], noise_var, averaged_over_shells = False)
#     return residual

# @functools.partial(jax.jit, static_argnums = [9,10,11,13,14,15,16,20])
# def get_residual_square_inner_v2(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, process_fn, CTF_fun, contrasts,basis_coordinates, noise_var, averaged_over_shells = False):
    
#     if volume_mask is not None:
#         image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
#                                               rotation_matrices,
#                                               image_mask, 
#                                               volume_mask_threshold,
#                                               image_shape, 
#                                               volume_shape, grid_size, 
#                                               padding, 
#                                               disc_type, soften = 5 )
        
#     predicted_vols = contrasts[None] * (( basis @ basis_coordinates.T) + mean_estimate[...,None])

#     projected_vols = core.batch_forward_model_from_map(predicted_vols.T,
#                                          CTF_params[:,None],
#                                          rotation_matrices[:,None],
#                                          image_shape, 
#                                          volume_shape, 
#                                         voxel_size, 
#                                         CTF_fun, 
#                                         disc_type                                      
#                                           )[:,0]
#     batch = process_fn(batch)
#     batch = core.translate_images(batch, translations , image_shape)
#     diff = batch - projected_vols
#     # import matplotlib.pyplot as plt
#     # plt.imshow(ftu.get_idft2(batch[0].reshape(image_shape)).real)
#     # plt.show()
#     # plt.figure()
#     # plt.imshow(ftu.get_idft2(projected_vols[0].reshape(image_shape)).real)
#     # plt.show()
#     # import pdb; pdb.set_trace()

#     if volume_mask is not None:
#         diff = covariance_core.apply_image_masks(diff, image_mask, image_shape)

#     if noise_var is not None:
#         diff = noise_var * diff

#     residual_squared = jnp.abs(diff)**2 

#     if averaged_over_shells:
#         averaged_residual_squared = regularization.batch_average_over_shells(residual_squared, image_shape,0) 
#         return averaged_residual_squared
    
#     return jnp.sum(residual_squared)
    

def get_average_residual_square_just_mean(experiment_dataset, volume_mask, mean_estimate, batch_size, disc_type = 'linear_interp'):
    contrasts = np.ones(experiment_dataset.n_images, dtype = experiment_dataset.dtype_real)
    basis = np.zeros((experiment_dataset.volume_size, 0))
    zs = np.zeros((experiment_dataset.n_images, 0))

    return get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,zs, batch_size, disc_type = disc_type)



def estimate_noise_from_heterogeneity_residuals_inside_mask_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):
    # masked_image_PS =  get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type )
    return get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type )


# @functools.partial(jax.jit, static_argnums = [5])    
def get_average_residual_square_v2(experiment_dataset, volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp'):

    assert basis.shape[0] == experiment_dataset.volume_size, "input u should be volume_size x basis_size"
    st_time = time.time()    
    basis = np.asarray(basis[:, :basis_coordinates.shape[-1]]).T

    if disc_type == 'cubic':
        st_time = time.time()
        from recovar import cryojax_map_coordinates, covariance_estimation
        mean_estimate = cryojax_map_coordinates.compute_spline_coefficients(mean_estimate.reshape(experiment_dataset.volume_shape))
        basis = covariance_estimation.compute_spline_coeffs_in_batch(basis, experiment_dataset.volume_shape, gpu_memory= None)
        logger.info("Time to compute spline coefficients: %f", time.time() - st_time)
        # basis = basis.T


    images_estimates = np.empty([experiment_dataset.n_images, *experiment_dataset.image_shape], dtype = experiment_dataset.dtype)
    # basis_size = basis.shape[-1]
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    basis = jnp.asarray(basis.T)
    top_fraction = 0
    kernel_sq_sum =0 

    for batch, batch_image_ind in data_generator:
        top_fraction_this, kernel_sq_sum_this, per_image_est = get_average_residual_square_inner_v2(batch, mean_estimate, volume_mask, 
                                basis,
                                experiment_dataset.CTF_params[batch_image_ind],
                                experiment_dataset.rotation_matrices[batch_image_ind],
                                experiment_dataset.translations[batch_image_ind],
                                experiment_dataset.image_stack.mask,
                                experiment_dataset.volume_mask_threshold,
                                experiment_dataset.image_shape, 
                                experiment_dataset.volume_shape, 
                                experiment_dataset.grid_size, 
                                experiment_dataset.voxel_size, 
                                experiment_dataset.padding, 
                                disc_type, 
                                experiment_dataset.image_stack.process_images,
                                experiment_dataset.CTF_fun, 
                                contrasts[batch_image_ind], basis_coordinates[batch_image_ind])

        top_fraction += top_fraction_this
        kernel_sq_sum+= kernel_sq_sum_this
        images_estimates[batch_image_ind] = np.array(per_image_est)
        # image_PSs[batch_ind] = np.array(image_PS)
        # masked_image_PSs[batch_ind] = np.array(masked_image_PS)

    predicted_pixel_variances= top_fraction / kernel_sq_sum
    predicted_pixel_variances = jnp.fft.ifft2( predicted_pixel_variances).real * experiment_dataset.image_size

    pred_noise = regularization.average_over_shells(predicted_pixel_variances, experiment_dataset.image_shape, 0) 
    return pred_noise, predicted_pixel_variances, images_estimates



def basis_times_coords(basis, coords):
    assert basis.shape[-1] == coords.shape[-1]
    # basis_shape_inp = basis.shape
    # summed = basis.reshape((basis.shape[0], -1,)).T @ coords.T
    return jnp.sum(basis * coords, axis=-1)
batch_basis_times_coords = jax.vmap(basis_times_coords, in_axes = (None,0))


# An atrocious function to do this without allocating too much memory.
# Basically writes the product of array which would be of size (256,256,256, 10) x (1000,10) where 1000 is n_images, 10 is size of basis and 256 volume size, as a matvec, then reshapes things back.
def batch_basis_times_coords2(basis, coords):

    assert basis.shape[-1] == coords.shape[-1]
    basis_shape_inp = basis.shape

    basis = basis.transpose(-1, *np.arange(basis.ndim-1) )
    basis = basis.reshape((coords.shape[-1], np.prod(basis_shape_inp[:-1])))

    # # Put into a matrix of size n_coeffs x dim of basis
    # basis = basis.reshape((basis.shape[0], -1,))
    summed = basis.T @ coords.T

    summed = summed.T
    summed = summed.reshape(coords.shape[0], *basis_shape_inp[:-1])
    # summed.transpose(-1, *np.arange(summed.ndim-1) )
    return summed#.transpose(-1, *np.arange(summed.ndim-1) )



@functools.partial(jax.jit, static_argnums = [9,10,11,13,14,15,16])
def get_average_residual_square_inner_v2(batch, mean_estimate, volume_mask, basis, CTF_params, rotation_matrices, translations, image_mask, volume_mask_threshold, image_shape, volume_shape, grid_size, voxel_size, padding, disc_type, process_fn, CTF_fun, contrasts,basis_coordinates):
    

    if volume_mask is not None:
        image_mask = covariance_core.get_per_image_tight_mask(volume_mask, 
                                              rotation_matrices,
                                              image_mask, 
                                              volume_mask_threshold,
                                              image_shape, 
                                              volume_shape, grid_size, 
                                              padding, 
                                              disc_type, soften = 5 )
    else:
        image_mask = jnp.ones_like(batch).real
    # zz1 = batch_basis_times_coords(basis,basis_coordinates)
    # zz2 = batch_basis_times_coords2(basis,basis_coordinates)
    # zz3 = batch_basis_times_coords3(basis,basis_coordinates)
    # print(jnp.linalg.norm(zz1 -zz2)/ jnp.linalg.norm(zz1))
    # print(jnp.linalg.norm(zz1 -zz2)/ jnp.linalg.norm(zz2))
    # print(jnp.linalg.norm(zz2 -zz3)/ jnp.linalg.norm(zz2))
    # import pdb; pdb.set_trace()
    
    if basis.shape[-1] == 0:
        predicted_vols = contrasts.reshape((contrasts.shape[0], *np.ones(mean_estimate.ndim, dtype = int) ) ) * mean_estimate[None]
    else:
        predicted_vols = contrasts.reshape((contrasts.shape[0], *np.ones(mean_estimate.ndim, dtype = int) ) ) * ( batch_basis_times_coords2(basis,basis_coordinates) + mean_estimate[None])

    # Are spline coefficients linear map? Yes!
    projected_vols = core.batch_forward_model_from_map(predicted_vols,
                                         CTF_params[:,None],
                                         rotation_matrices[:,None],
                                         image_shape, 
                                         volume_shape, 
                                        voxel_size, 
                                        CTF_fun, 
                                        disc_type                                      
                                          )[:,0]
    
    batch = process_fn(batch)
    batch = core.translate_images(batch, translations , image_shape)
    substracted_images = batch - projected_vols

    return get_masked_image_noise_fractions(substracted_images, image_mask, image_shape)
    