import jax.numpy as jnp
import numpy as np
import jax, functools

from recovar import core, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

## Mean prior computation

#@functools.partial(jax.jit, static_argnums = [7,8,9,10, 11, 12,13])    
def compute_batch_prior_quantities(rotation_matrices, translations, CTF_params, noise_variance, voxel_size, dtype, volume_shape, image_shape, grid_size, CTF_fun , for_whitening = False):
    volume_size = np.prod(np.array(volume_shape))
    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    all_one_volume = jnp.ones(volume_size, dtype = dtype)    
    
    # if for_whitening:
    #     ones_CTF_mapped = jnp.sqrt(core.forward_model(all_one_volume, CTF, grid_point_indices) * CTF )
    # else:
    ones_CTF_mapped = core.forward_model(all_one_volume, CTF, grid_point_indices) * CTF / noise_variance[None] 
    diag_mean = core.sum_adj_forward_model(volume_size, ones_CTF_mapped, jnp.ones_like(CTF), grid_point_indices)
    
    return diag_mean

def compute_prior_quantites(experiment_datasets, cov_noise, batch_size, for_whitening = False ):
    bottom_of_fraction = jnp.zeros(experiment_datasets[0].volume_size, dtype = experiment_datasets[0].image_stack.dtype)
    for experiment_dataset in experiment_datasets:
        n_images = experiment_dataset.n_images
        # Compute the bottom of fraction.
        for k in range(0, int(np.ceil(n_images/batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min( [(k+1) * batch_size, n_images]))
            indices = jnp.arange(batch_st, batch_end)
            bottom_of_fraction_this = compute_batch_prior_quantities(
                                             experiment_dataset.rotation_matrices[indices], 
                                             experiment_dataset.translations[indices], 
                                             experiment_dataset.CTF_params[indices], 
                                             cov_noise,
                                             experiment_dataset.voxel_size, 
                                             experiment_dataset.dtype,
                                             experiment_dataset.volume_shape, 
                                             experiment_dataset.image_shape, 
                                             experiment_dataset.grid_size, 
                                             experiment_dataset.CTF_fun, 
                                             for_whitening)
            
            bottom_of_fraction += bottom_of_fraction_this
        
    bottom_of_fraction = bottom_of_fraction.real / len(experiment_datasets)
    return bottom_of_fraction 
    

def compute_relion_prior(experiment_datasets, cov_noise, image0, image1, batch_size, estimate_merged_SNR = False, noise_level = None):
    
    if noise_level is not None:
        bottom_of_fraction = noise_level
        from_noise_level = True
    else:
        bottom_of_fraction = compute_prior_quantites(experiment_datasets, cov_noise, batch_size, for_whitening = False )
        from_noise_level = False
    
    return compute_fsc_prior_gpu(experiment_datasets[0].volume_shape, image0, image1, bottom_of_fraction, estimate_merged_SNR = estimate_merged_SNR, from_noise_level = from_noise_level )


def get_fsc(vol1, vol2, volume_shape, substract_shell_mean = False, frequency_shift = 0):
    return get_fsc_gpu(vol1, vol2, volume_shape, substract_shell_mean, frequency_shift)

# @functools.partial(jax.jit, static_argnums = [7,8,9,10, 11, 12,13])    
def get_fsc_gpu(vol1, vol2, volume_shape, substract_shell_mean = False, frequency_shift = 0):
    
    if substract_shell_mean:
        # Center two volumes.
        vol1_avg = average_over_shells(vol1, volume_shape, frequency_shift = frequency_shift)
        vol2_avg = average_over_shells(vol2, volume_shape, frequency_shift = frequency_shift)
        radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
        vol1 -= vol1_avg[radial_distances].reshape(vol1.shape)
        vol2 -= vol2_avg[radial_distances].reshape(vol2.shape)

    top = jnp.conj(vol1) * vol2    
    top_avg = average_over_shells(top.real, volume_shape, frequency_shift = frequency_shift)
    bot1 = average_over_shells(jnp.abs(vol1)**2, volume_shape, frequency_shift = frequency_shift)
    bot2 = average_over_shells(jnp.abs(vol2)**2, volume_shape, frequency_shift = frequency_shift)    
    bot = jnp.sqrt(bot1 * bot2)
    fsc = jnp.where(bot  > constants.EPSILON , top_avg / bot, constants.EPSILON)
    fsc = fsc.at[0].set(fsc[1]) # Always set this 1st shell?
    return fsc


# @functools.partial(jax.jit, static_argnums = [1,2])
def average_over_shells(input_vec, volume_shape, frequency_shift = 0 ):
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1) 
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0]//2 - 1)        ## I  wish there wasn't a -1 here, hmmmm
    return jax_scipy_nd_image_mean(input_vec.reshape(-1), labels = labels, index = indices)    


def jax_scipy_nd_image_mean(input, labels=None, index=None):
    if input.dtype == 'complex64':
        input = input.astype('complex128') #jax.numpy.bincount complex64 version seems to be bugged.
        return jax_scipy_nd_image_mean(input.reshape(-1), labels = labels, index = index).astype('complex64')
    return jax_scipy_nd_image_mean_inner(input, labels = labels, index = index)

def jax_scipy_nd_image_mean_inner(input, labels=None, index=None):
    # A jittable simplified scipy.ndimage.mean method
    # numpy = np
    # unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    # new_labels = labels
    
    # # counts = numpy.bincount(new_labels,length = index.size )
    # counts = numpy.bincount(new_labels)#,length = index.size )

    # # sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    # sums = numpy.bincount(new_labels, weights=input.ravel())#,length = index.size )


    numpy = jnp
    unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    new_labels = labels
    
    # counts = numpy.bincount(new_labels,length = index.size )
    counts = numpy.bincount(new_labels,length = index.size )

    # sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )


    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = numpy.where( idxs >= int(unique_labels.size), 0, idxs)

    found = (unique_labels[idxs] == index)
    counts = counts[idxs]
    counts = numpy.where(found, counts, 0)
    sums = sums[idxs]

    sums = numpy.where(sums, sums, 0)
    return sums / counts


def sum_over_shells(input_vec, volume_shape, frequency_shift = 0 ):
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1) 
    labels = radial_distances.reshape(-1)
    indices = jnp.arange(0, volume_shape[0]//2 - 1)
    return jax_scipy_nd_image_sum(input_vec.reshape(-1), labels = labels, index = indices)    

def jax_scipy_nd_image_sum(input, labels=None, index=None):
    # A jittable simplified scipy.ndimage.mean method
    numpy = jnp
    unique_labels = index #, new_labels = numpy.unique(labels, return_inverse=True)
    new_labels = labels
    
    counts = numpy.bincount(new_labels,length = index.size )
    sums = numpy.bincount(new_labels, weights=input.ravel(),length = index.size )
    
    idxs = numpy.searchsorted(unique_labels, index)
    # make all of idxs valid
    idxs = jnp.where( idxs >= int(unique_labels.size), 0, idxs)

    found = (unique_labels[idxs] == index)
    counts = counts[idxs]
    counts = jnp.where(found, counts, 0)
    sums = sums[idxs]

    sums = jnp.where(sums, sums, 0)
    return sums


    
def compute_fsc_prior_gpu(volume_shape, image0, image1, bottom_of_fraction = None, estimate_merged_SNR = False, substract_shell_mean = False, frequency_shift = 0, from_noise_level = False):
    epsilon = constants.FSC_ZERO_THRESHOLD
    # FSC top:
    fsc = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)

    if substract_shell_mean:
        # Set the first 2 to zeros b/c could run in trouble, since killing all signal
        fsc = fsc.at[0:2].set(1)

    fsc = jnp.where(fsc > epsilon , fsc, epsilon )
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon )
    if estimate_merged_SNR:
        fsc = 2 * fsc / ( 1 + fsc )
        
    # SNR = jnp.where(fsc < 1 - epsilon, fsc / ( 1 - fsc), jnp.inf)
    SNR = fsc / (1 - fsc)
    
    # Bottom of fraction
    if from_noise_level:        
        # bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)        
        prior_avg = SNR * bottom_of_fraction #jnp.where( bottom_avg > 0 , SNR * bottom_avg, epsilon )
        print("used outdated prior!!!! Change this!")
    else:
        bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape, frequency_shift)        
        prior_avg = jnp.where( bottom_avg > 0 , SNR / bottom_avg, constants.EPSILON )
    
    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]
    
    return prior, fsc, prior_avg


def downsample_lhs(lhs, volume_shape, upsampling_factor = 1):
    # Downsample lhs by a factor of 2
    # radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = -1)
    # lhs_inp_shape = lhs.shape
    kernel = jnp.ones( 3 * [2 * upsampling_factor - 1], dtype = jnp.float32)
    kernel = kernel / jnp.sum(kernel)
    lhs = jax.scipy.signal.fftconvolve(lhs, kernel, mode = 'same')
    lhs = lhs[::upsampling_factor,::upsampling_factor,::upsampling_factor]
    lhs = jnp.where(lhs > 0, lhs, 0)
    return (lhs * (2 **len(volume_shape)))


@functools.partial(jax.jit, static_argnums = [0,6, 7])    
def compute_fsc_prior_gpu_v2(volume_shape, image0, image1, lhs , prior, frequency_shift , substract_shell_mean = False, upsampling_factor = 1 ):
    epsilon = constants.FSC_ZERO_THRESHOLD
    # FSC top:
    fsc = get_fsc_gpu(image0, image1, volume_shape, substract_shell_mean, frequency_shift)
    fsc = jnp.where(fsc > epsilon , fsc, epsilon )
    fsc = jnp.where(fsc < 1 - epsilon, fsc, 1 - epsilon )
        
    # SNR = jnp.where(fsc < 1 - epsilon, fsc / ( 1 - fsc), jnp.inf)
    SNR = fsc / (1 - fsc)
    
    # Gotta somehow downsample lhs by a factor of 2
    upsampled_volume_shape = tuple([ upsampling_factor * i for i in volume_shape])
    lhs = downsample_lhs(lhs.reshape(upsampled_volume_shape), upsampled_volume_shape, upsampling_factor = upsampling_factor).reshape(-1)

    if prior is None:
        top = jnp.ones_like(lhs)
        bot = 1 / lhs
    else:
        top = lhs**2 / (lhs + 1/prior)**2
        bot = lhs / (lhs + 1/prior)**2

    sum_top = average_over_shells(top,  volume_shape, frequency_shift)    
    sum_bot = average_over_shells(bot,  volume_shape, frequency_shift)
    
    prior_avg = jnp.where( sum_top > 0 , SNR * sum_bot / sum_top , constants.ROOT_EPSILON ).real
    
    # Put back in array
    radial_distances = ftu.get_grid_of_radial_distances(volume_shape, scaled = False, frequency_shift = frequency_shift).astype(int).reshape(-1)
    prior = prior_avg[radial_distances]

    return prior, fsc, prior_avg



def covariance_update_col(H, B, prior, epsilon = constants.EPSILON):
    # H is not divided by sigma.
    cov = jnp.where( jnp.abs(H) < epsilon , 0,  B / ( H + (1 / prior) ) )
    return cov

def covariance_update_col_with_mask(H, B, prior, volume_mask, valid_idx, volume_shape, epsilon = constants.EPSILON):
    # H is not divided by sigma.
    cov = (jnp.where( jnp.abs(H) < epsilon , 0,  B / ( H + (1 / prior) ) ) * valid_idx).reshape(volume_shape)
    cov = ftu.get_dft3( ftu.get_idft3(cov ) * volume_mask ).reshape(-1)
    return cov

@functools.partial(jax.jit, static_argnums = [6,7,8])    
def prior_iteration(H0, H1, B0, B1, frequency_shift, init_regularization, substract_shell_mean, volume_shape, prior_iterations ):
    
    H_comb = (H0 +  H1)/2
    prior = init_regularization

    # Harcoded iterations because I couldn't figure out to make jit the loop properly...

    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)

    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)
    cov_col0 =  covariance_update_col(H0,B0, prior)
    cov_col1 =  covariance_update_col(H1,B1, prior)
    prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = 0)
    
    return prior, fsc

from recovar import relion_functions
@functools.partial(jax.jit, static_argnums = [6,7,8,9,10, 12])    
def prior_iteration_relion_style(H0, H1, B0, B1, frequency_shift, init_regularization, substract_shell_mean, volume_shape, kernel = 'triangular', use_spherical_mask = True, grid_correct = True, volume_mask = None, prior_iterations = 3):
    # assert substract_shell_mean == False
    # assert jnp.linalg.norm(frequency_shift) < 1e-8

    H_comb = (H0 +  H1)/2
    prior = init_regularization.real

    def body_fun(prior, fsc):
        cov_col0 = relion_functions.post_process_from_filter_v2(H0, B0, volume_shape, volume_upsampling_factor = 1, tau = prior, kernel = kernel, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1, volume_mask = volume_mask )
        cov_col1 = relion_functions.post_process_from_filter_v2(H1, B1, volume_shape, volume_upsampling_factor = 1, tau = prior, kernel = kernel, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1 , volume_mask = volume_mask )
        prior, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, cov_col0, cov_col1, H_comb, prior, frequency_shift = frequency_shift, substract_shell_mean = substract_shell_mean)
        return prior, fsc
    
    ## TODO: Surely there is a better way to do this...
    def body_fun_no_fsc(i, prior):
        prior, _ = body_fun(prior, None)
        return prior

    if prior_iterations > 0:
        prior = jax.lax.fori_loop(0, prior_iterations, body_fun_no_fsc, prior)
        _, fsc = body_fun(prior, None)
    elif prior_iterations == -1:
        prior = None
        _, fsc = body_fun(prior, None)
    elif prior_iterations == 0:
        _, fsc = body_fun(prior, None)
    else:
        raise ValueError("Prior iterations must be a non-negative integer or -1 (no reg)")
    
    ## NOTE THIS ONE IS NEVER MASKED. IT GETS MASKED LATER. PERHAPS SHOULD BE MASKED HERE
    cov_col0 = relion_functions.post_process_from_filter_v2(H0 + H1, B0 + B1, volume_shape, volume_upsampling_factor = 1, tau = prior, kernel = kernel, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1, volume_mask = volume_mask )

    return cov_col0.reshape(-1), prior, fsc


prior_iteration_batch = jax.vmap(prior_iteration, in_axes = (0,0,0,0, 0, 0, None, None, None) )
prior_iteration_relion_style_batch = jax.vmap(prior_iteration_relion_style, in_axes = (0,0,0,0, 0, 0, None, None, None, None, None, None, None))


# def compute_masked_fscs(H0, B0, H1, B1, prior, volume_shape, volume_mask):
#     volumes1 =  covariance_update_col(H0,B0, prior)
#     volumes2 =  covariance_update_col(H1,B1, prior)
    
#     def apply_masks(volumes):
#         vols_real = ftu.get_idft3(volumes.reshape([-1, *volume_shape])) * volume_mask
#         return ftu.get_dft3(vols_real).reshape([vols_real.shape[0], -1])
    
#     volumes1_masked = apply_masks(volumes1)
#     volumes2_masked = apply_masks(volumes2)
    
#     _, fsc, _ = compute_fsc_prior_gpu_v2(volume_shape, volumes1_masked, volumes2_masked, H0 , prior, frequency_shift = 0)
#     return fsc

batch_average_over_shells = jax.vmap(average_over_shells, in_axes = (0,None,None))


