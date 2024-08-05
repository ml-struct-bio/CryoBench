import logging
import jax.numpy as jnp
import numpy as np
import jax, functools, time

from recovar import core, regularization, constants, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import utils

logger = logging.getLogger(__name__)

def image_weight_parse(image_weights, dtype = np.float32):
    if image_weights is None:
        return [None, None]
    else:
        return [w.astype(dtype) for w in image_weights]


def get_mean_conformation_relion(cryos, batch_size, noise_variance = None,  use_regularization = False):
    st_time = time.time()

    cryo = cryos[0]
    from recovar import relion_functions

    means = dict()
    lhs_l = 2 * [None]
    for idx, cryo in enumerate(cryos):
        means["corrected" + str(idx)], lhs_l[idx] = relion_functions.relion_reconstruct(cryo, noise_variance,batch_size )
    lhs = (lhs_l[0] + lhs_l[1])/2

    mean_prior, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, means["corrected0"], means["corrected1"], lhs, jnp.ones(cryos[0].volume_size, dtype = cryos[0].dtype_real) * np.inf, frequency_shift = jnp.array([0,0,0]), upsampling_factor = 2)

    if use_regularization:
        for idx, cryo in enumerate(cryos):
            means["corrected" + str(idx)], lhs_l[idx] = relion_functions.relion_reconstruct(cryo, noise_variance,batch_size, tau = mean_prior )
    lhs = (lhs_l[0] + lhs_l[1])/2

    logger.info(f"Using new prior")
    mean_prior = np.array(mean_prior)
    means["combined"] = (means["corrected0"] + means["corrected1"])/2
    means["prior"] = mean_prior
    means["lhs"] = lhs

    for key in means:
        means[key] = np.array(means[key])

    end_time = time.time()
    logger.info(f"time to compute means: {end_time- st_time}")

    return means, mean_prior, fsc, lhs


def get_mean_conformation(cryos, batch_size, noise_variance = None, valid_idx = None, disc_type ='linear_interp', use_noise_level_prior = False, grad_n_iter = 5, image_weights = None):
    
    cryo = cryos[0]
    valid_idx = cryo.get_valid_frequency_indices() if valid_idx is None else valid_idx
    if noise_variance is None:
        noise_variance, signal_var = noise.estimate_noise_variance(cryo, batch_size)
        signal_var = np.max(signal_var)
    else:
        _, signal_var = noise.estimate_noise_variance(cryo, batch_size)
        signal_var = np.max(signal_var)

    regularization_init = jnp.ones(cryo.volume_size, dtype = cryo.dtype_real) * signal_var
    logger.info("regularization init done")
    image_weights = image_weight_parse(image_weights, dtype = cryos[0].dtype_real)
    means = {}

    # This is kind of a stupid way to code this. Should probably rewrite next 3 forloops
    st_time = time.time()
    for cryo_idx,cryo in enumerate(cryos):
        means["init" +str(cryo_idx) ] = np.array(compute_mean_volume(cryo, noise_variance,  regularization_init, batch_size, disc_type = disc_type, n_iter = 1, image_weights = image_weights[cryo_idx]))
        
    mean_prior, fsc, _ = regularization.compute_relion_prior(cryos, noise_variance, means["init0"], means["init1"], batch_size)
    for cryo_idx, cryo in enumerate(cryos):
        means["corrected" +str(cryo_idx) ] = np.array(compute_mean_volume(cryo, noise_variance,  mean_prior, batch_size, disc_type = disc_type, n_iter = grad_n_iter, image_weights = image_weights[cryo_idx]))
        
    mean_prior, fsc, _ = regularization.compute_relion_prior(cryos, noise_variance, means["corrected0"], means["corrected1"], batch_size)        
    
    lhs = 0; rhs = 0 
    for cryo_idx, cryo in enumerate(cryos):
        lhs_t, rhs_t = compute_mean_volume(cryo, noise_variance,  mean_prior, batch_size, disc_type = disc_type, n_iter = 1, return_lhs_rhs = True, mean_estimate = (means["corrected0"] + means["corrected1"])/2 , image_weights = image_weights[cryo_idx] )
        lhs +=  lhs_t
        rhs +=  rhs_t
            
    
    # Probably should be rewritten around here.
    if use_noise_level_prior:
        key = "corrected"
        mean_prior, fsc, prior_avg = regularization.compute_relion_prior(cryos, noise_variance, means["corrected0"], means["corrected1"], batch_size, estimate_merged_SNR = False)
        logger.info(f"Using RELION prior")

    else:    
        mean_prior, fsc, prior_avg = regularization.compute_fsc_prior_gpu_v2(cryo.volume_shape, means["corrected0"], means["corrected1"], lhs/2 , mean_prior, frequency_shift = jnp.array([0,0,0]))
        logger.info(f"Using new prior")

    mean_prior = np.array(mean_prior)
    means["combined"] = np.array( jnp.where(jnp.abs(lhs) < 1e-8, 0, rhs / (lhs + 1 / mean_prior ) ))
    means["prior"] = mean_prior
    means["lhs"] = lhs/2

    end_time = time.time()
    logger.info(f"time to compute means: {end_time- st_time}")

    return means, mean_prior, fsc, lhs



def get_multiple_conformations(cryos, noise_variance, disc_type, batch_size, mean_prior, mean ,image_weights, recompute_prior = True, volume_mask = None, adaptive = True ):
    image_weights = image_weight_parse(image_weights, dtype = cryos[0].dtype_real)

    st_time = time.time()

    num_reconstructions = image_weights[0].shape[0]
    lhs = 0; rhs = 0 
    lhs_l = []
    rhs_l = []
    for cryo_idx, cryo in enumerate(cryos):
        lhs_t, rhs_t = compute_mean_volume(cryo, noise_variance,  mean_prior, batch_size, disc_type = disc_type, n_iter = 1, return_lhs_rhs = True, mean_estimate = mean , image_weights = image_weights[cryo_idx].astype(cryo.dtype_real) )
        lhs_l.append(jnp.real(lhs_t).astype(cryo.dtype_real))
        rhs_l.append(rhs_t)
        # lhs +=  lhs_t
        # rhs +=  rhs_t
        
    if recompute_prior:
        priors, fscs_this = regularization.prior_iteration_batch(lhs_l[0], lhs_l[1], rhs_l[0], rhs_l[1], np.zeros((num_reconstructions,3)), mean_prior[None].repeat(num_reconstructions,0) , False,  cryos[0].volume_shape, 3 ) 
    else:
        priors = mean_prior[None]
        logger.info("Not recomputing prior?? Make sure this is what you want")

    # Dump on CPU

    # I have no idea why all of the sudden this broke. Something about JAX -> np?
    priors = np.abs(np.array(priors)) + constants.EPSILON

    lhs_l[0] = np.array(lhs_l[0], dtype = lhs_l[0].dtype )
    lhs_l[1] = np.array(lhs_l[1], dtype = lhs_l[1].dtype )
    rhs_l[0] = np.array(rhs_l[0], dtype = rhs_l[0].dtype )
    rhs_l[1] = np.array(rhs_l[1], dtype = rhs_l[1].dtype )

    half_maps = []

    for s in range(2):
        half_maps.append(np.array( np.where(np.abs(lhs_l[s]) < constants.ROOT_EPSILON, 0, rhs_l[s] / (lhs_l[s] + 1 / priors ) )))

    # Add second to first
    # lhs_l[0] += lhs_l[1]
    # rhs_l[0] += rhs_l[1]
    reconstructions = np.array( np.where( np.abs(lhs_l[0] + lhs_l[1]) < constants.ROOT_EPSILON, 0, (rhs_l[0] + rhs_l[1])/ ((lhs_l[0] + lhs_l[1]) + 1 / priors ) ))
    end_time = time.time()
    logger.info(f"time to compute reweighted conformations: {end_time- st_time}")
    
    if adaptive:
        adaptive_reconstructions_halfmaps = np.zeros([2 , *reconstructions.shape], dtype = cryo.dtype)
        # Do one by one for now...
        # Initial contrasts
        # initial_contrasts = [ cryo.CTF_params[:,8]  for cryo in cryos]

        # for k in range(num_reconstructions):
        #     for cryo_idx, cryo in enumerate(cryos):
        #         # Instead of passing weights... this should do the same.
        #         cryo.CTF_params[:,8] = initial_contrasts[cryo_idx] * image_weights[cryo_idx][k]
        #         adaptive_reconstructions_halfmaps[cryo_idx,k],_ = compute_with_adaptive_discretization(cryo, lhs_l[cryo_idx][k], priors[k], reconstructions[k], noise_variance, batch_size)
        from recovar.adaptive_discretization import compute_with_adaptive_discretization
        for cryo_idx, cryo in enumerate(cryos):
            adaptive_reconstructions_halfmaps[cryo_idx],_ = compute_with_adaptive_discretization(cryo, lhs_l[cryo_idx], priors, reconstructions, noise_variance, batch_size, image_weights[cryo_idx])

        reconstructions = np.mean(adaptive_reconstructions_halfmaps, axis=0)

        return reconstructions, [adaptive_reconstructions_halfmaps[0], adaptive_reconstructions_halfmaps[1]]

    return reconstructions, half_maps



## Mean functions
def compute_mean_volume(experiment_dataset, noise_variance, cov_diag_prior, batch_size, disc_type, n_iter = 2, image_weights = None, return_lhs_rhs = False, mean_estimate = None ):

    if disc_type =="nearest":
        n_iter = 1        
        
    if return_lhs_rhs:
        n_iter =1
    
    for _ in range(n_iter):
        # Make sure mean_estimate is size # volume_size ?
        if mean_estimate is not None:
            mean_estimate = mean_estimate.reshape(experiment_dataset.volume_size)
        mean_estimate = solve_least_squares_mean_iteration(experiment_dataset , 
                                            cov_diag_prior,
                                            noise_variance, 
                                            batch_size, 
                                            mean_estimate = mean_estimate, 
                                            image_weights = image_weights,
                                            disc_type = disc_type,
                                            return_lhs_rhs = return_lhs_rhs)
    return mean_estimate


# Solves the linear system Dx = b.
def solve_least_squares_mean_iteration(experiment_dataset , cov_diag_prior, noise_variance,  batch_size, mean_estimate, image_weights = None, disc_type = None, return_lhs_rhs = False ):
    # all_one_volume = jnp.ones(volume_size)
    
    mean_rhs = jnp.zeros(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
    diag_mean = jnp.zeros(experiment_dataset.volume_size, dtype = experiment_dataset.dtype)
    logger.warning('TOOK OUT IMAGE MASK IN MEAN!!! PUT IT BACK??')

    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    for batch, indices in data_generator:
        
        # Only place where image mask is used ?
        # print('TOOK OUT IMAGE MASK IN MEAN!!! PUT IT BACK??')
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        
        if image_weights is None:
            image_weights_batch = jnp.ones((1,batch.shape[0]), dtype = experiment_dataset.dtype_real )
        else:
            image_weights_batch = image_weights[...,indices]
            
        mean_rhs_this, diag_mean_this = compute_mean_least_squares_rhs_lhs(batch,
                                         experiment_dataset.rotation_matrices[indices], 
                                         experiment_dataset.translations[indices], 
                                         experiment_dataset.CTF_params[indices], 
                                         mean_estimate,
                                         noise_variance,
                                         image_weights_batch,
                                         experiment_dataset.voxel_size, 
                                         experiment_dataset.volume_shape, 
                                         experiment_dataset.image_shape, 
                                         experiment_dataset.grid_size, 
                                         disc_type,
                                         experiment_dataset.CTF_fun)
        mean_rhs += mean_rhs_this
        diag_mean += diag_mean_this

    if return_lhs_rhs:
        return diag_mean, mean_rhs

    X_mean = jnp.where(jnp.abs(diag_mean) < 1e-8, 0, mean_rhs / (diag_mean + 1 / cov_diag_prior ) )
    return X_mean
    

@functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
def compute_mean_least_squares_rhs_lhs(images, rotation_matrices, translations, CTF_params, mean_estimate, noise_variance, image_weights, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    volume_size = np.prod(np.array(volume_shape))
    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    translated_images = core.translate_images(images, translations, image_shape)
    if disc_type != "nearest" and mean_estimate is not None:
        grad_correction = (core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, disc_type) \
                        - core.slice_volume_by_map(mean_estimate, rotation_matrices, image_shape, volume_shape, "nearest")) * CTF
        corrected_images = translated_images - grad_correction
    else:
        corrected_images = translated_images
    
    all_one_volume = jnp.ones(volume_size, dtype = images.dtype)    
    ones_mapped = core.forward_model(all_one_volume, CTF, grid_point_indices) / noise_variance[None] * image_weights[...,None]
    # diag_mean = core.sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)
    diag_mean = batch_over_weights_sum_adj_forward_model(volume_size, ones_mapped, CTF, grid_point_indices)

    corrected_images = corrected_images * image_weights[...,None] / noise_variance[None]
    # mean_rhs = core.sum_adj_forward_model(volume_size, corrected_images  / noise_variance[None] , CTF, grid_point_indices)
    mean_rhs = batch_over_weights_sum_adj_forward_model(volume_size, corrected_images , CTF, grid_point_indices)
    return mean_rhs, diag_mean





batch_over_weights_sum_adj_forward_model = jax.vmap(core.sum_adj_forward_model, in_axes = (None,0, None,None))

# def compute_weight_matrix_inner(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun ):
#     volume_size = np.prod(np.array(volume_shape))
#     C_mat, grid_point_vec_indices = make_C_mat(rotation_matrices, CTF_params, voxel_size, volume_shape, image_shape, grid_size, CTF_fun)
#     # This is going to be stroed twice as much stuff as it needs to be
#     C_mat_outer = batch_batch_outer(C_mat).reshape([C_mat.shape[0], C_mat.shape[1], C_mat.shape[2]*C_mat.shape[2]])#.transpose([2,0,1])
#     RR = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, C_mat_outer, grid_point_vec_indices)
#     return RR
