import logging
import jax.numpy as jnp
import numpy as np
import jax, time

from recovar import core, covariance_estimation, embedding, plot_utils, linalg, constants, utils, noise
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)



def estimate_principal_components(cryos, options,  means, mean_prior, cov_noise, volume_mask,
                                dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use,
                                noise_model,  
                                covariance_options = None, variance_estimate = None):
    
    covariance_options = covariance_estimation.get_default_covariance_computation_options() if covariance_options is None else covariance_options

    volume_shape = cryos[0].volume_shape
    vol_batch_size = utils.get_vol_batch_size(cryos[0].grid_size, gpu_memory_to_use)

    # Different way of sampling columns: 
    # - from low to high frequencies
    # This is the way it was done in the original code. 
    # - Highest SNR columns, computed by lhs of mean estimation. May want to not take frequencies that are too similar
    # - Highest variance columns. Also want want to diversify.
    # For the last one, could also batch by doing randomized-Cholesky like choice

    if covariance_options['column_sampling_scheme'] == 'low_freqs':
        from recovar import covariance_core
        volume_shape = cryos[0].volume_shape
        if cryos[0].grid_size == 16:
            picked_frequencies = np.arange(cryos[0].volume_size) 
        else:
            picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = covariance_options['column_radius'], use_half = True))
    elif covariance_options['column_sampling_scheme'] == 'high_snr' or covariance_options['column_sampling_scheme'] == 'high_lhs' or covariance_options['column_sampling_scheme'] == 'high_snr_p' or covariance_options['column_sampling_scheme'] =='high_snr_from_var_est':
        from recovar import regularization
        upsampling_factor = np.round((means['lhs'].size / cryos[0].volume_size)**(1/3)).astype(int)
        upsampled_volume_shape = tuple(upsampling_factor * np.array(volume_shape))
        lhs = regularization.downsample_lhs(means['lhs'].reshape(upsampled_volume_shape), volume_shape, upsampling_factor = upsampling_factor).reshape(-1)
        # At low freqs, signal variance decays as ~1/rad^2

        dist = (ftu.get_grid_of_radial_distances(volume_shape)+1)**2
        if covariance_options['column_sampling_scheme'] == 'high_snr':
            lhs = lhs / dist.reshape(-1)
        if covariance_options['column_sampling_scheme'] == 'high_snr_p':
            lhs = lhs * means['prior']
        if covariance_options['column_sampling_scheme'] == 'high_snr_from_var_est':
            if variance_estimate is None:
                raise ValueError("variance_estimate must be provided")
            lhs = lhs * variance_estimate

        if covariance_options['randomize_column_sampling']:
            picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.randomized_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])
        else:
            picked_frequencies, picked_frequencies_in_frequencies_format = covariance_estimation.greedy_column_choice(lhs, covariance_options['sampling_n_cols'], volume_shape, avoid_in_radius = covariance_options['sampling_avoid_in_radius'])

        logger.info(f"Largest frequency computed: {np.max(np.abs(picked_frequencies_in_frequencies_format))}")
        if np.max(np.abs(picked_frequencies_in_frequencies_format)) > cryos[0].grid_size//2-1:
            logger.warning("Largest frequency computed is larger than grid size//2-1. This may cause big issues in SVD. This probably means variance estimates were wrong")
        # print("chosen cols", picked_frequencies_in_frequencies_format.T)
        # import pdb; pdb.set_trace()
    else:
        raise NotImplementedError('unrecognized column sampling scheme')
    
    covariance_cols, picked_frequencies, column_fscs = covariance_estimation.compute_regularized_covariance_columns_in_batch(cryos, means, mean_prior, cov_noise, volume_mask, dilated_volume_mask, valid_idx, gpu_memory_to_use, noise_model, covariance_options, picked_frequencies)
    logger.info("memory after covariance estimation")
    utils.report_memory_device(logger=logger)
    


    if options['ignore_zero_frequency']:
        zero_freq_index = np.asarray(core.frequencies_to_vec_indices( np.array([0,0,0]), cryos[0].volume_shape)).astype(int)
        zero_freq_in_picked_freq = np.where(picked_frequencies == zero_freq_index)[0].astype(int)

        # Set covariances with frequency 0 to 0.
        # I am not this if this is a good idea...
        # covariance_cols['est_mask'][:,zero_freq_in_picked_freq] *= 0 
        # covariance_cols['est_mask'][zero_freq_index,:] *= 0 

    # First approximation of eigenvalue decomposition
    u,s = get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape, vol_batch_size, gpu_memory_to_use, options['ignore_zero_frequency'], covariance_options['randomized_sketch_size'])
    
    if not options['keep_intermediate']:
        for key in covariance_cols.keys():
            covariance_cols[key] = None

    # # Let's see?
    # if noise_model == "white":
    #     cov_noise = cov_noise
    # else:
    #     # This probably should be moved into embedding
    if options['ignore_zero_frequency']:
        # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
        logger.info('ignoring zero frequency')
        cov_noise[0] *=1e16
        
    image_cov_noise = np.asarray(noise.make_radial_noise(cov_noise, cryos[0].image_shape))

    u['rescaled'], s['rescaled'] = pca_by_projected_covariance(cryos, u['real'], means['combined'], image_cov_noise, dilated_volume_mask, disc_type = covariance_options['disc_type'], disc_type_u = covariance_options['disc_type_u'], gpu_memory_to_use= gpu_memory_to_use, use_mask = covariance_options['mask_images_in_proj'], parallel_analysis = False ,ignore_zero_frequency = False, n_pcs_to_compute = covariance_options['n_pcs_to_compute'])

    if not options['keep_intermediate']:
        u['real'] = None

    # u['pa'], s['pa'] = pca_by_projected_covariance(cryos, u['real'], means['combined'], image_cov_noise, dilated_volume_mask, disc_type ='linear_interp', gpu_memory_to_use= gpu_memory_to_use, use_mask = True, parallel_analysis = True ,ignore_zero_frequency = False)


    if options['ignore_zero_frequency']:
        logger.warning("FIX THIS OPTION!! NOT SURE IT WILL STILL WORK")
    # u['rescaled_old'],s['rescaled_old'], zss['init'] = rescale_eigs(cryos, u['real'],s['real'], means['combined'], volume_mask, image_cov_noise, basis_size = constants.N_PCS_TO_COMPUTE, gpu_memory_to_use = gpu_memory_to_use, use_mask = True, ignore_zero_frequency = options['ignore_zero_frequency'])

    # logger.info(f"u after rescale dtype: {u['rescaled'].dtype}")
    logger.info("memory after rescaling")
    utils.report_memory_device(logger=logger)
    if (options['contrast'] == "contrast_qr"):
        c_time = time.time()
        # Going to keep a copy around for debugging purposes. Probably should delete at some point to reduce memory 
        u['rescaled_no_contrast'] = u['rescaled'].copy()
        s['rescaled_no_contrast'] = s['rescaled'].copy()

        mean_used = means['combined']
        # if options['ignore_zero_frequency']:
        #     mean_used[...,zero_freq_index] = 0
        u['rescaled'],s['rescaled'] = knock_out_mean_component_2(u['rescaled'], s['rescaled'],mean_used, volume_mask, volume_shape, vol_batch_size, options['ignore_zero_frequency'])

        if not options['keep_intermediate']:
            u['rescaled_no_contrast'] = None

        logger.info(f"knock out time: {time.time() - c_time}")
        if u['rescaled'].dtype != cryos[0].dtype:
            logger.warning(f"u['rescaled'].dtype: {u['rescaled'].dtype}")

            
    return u, s, covariance_cols, picked_frequencies, column_fscs


def get_cov_svds(covariance_cols, picked_frequencies, volume_mask, volume_shape,  vol_batch_size, gpu_memory_to_use, ignore_zero_frequency, randomized_sketch_size ):
    u = {}; s = {}    
    
    u['real'], s['real'],_ = randomized_real_svd_of_columns(covariance_cols["est_mask"], picked_frequencies, volume_mask, volume_shape, vol_batch_size, test_size=randomized_sketch_size, gpu_memory_to_use=gpu_memory_to_use, ignore_zero_frequency=ignore_zero_frequency)

    # u['real'], s['real'] = randomized_real_svd_of_columns_with_s_guess(covariance_cols["est_mask"],  picked_frequencies, volume_mask, volume_shape, vol_batch_size, test_size = constants.RANDOMIZED_SKETCH_SIZE, gpu_memory_to_use = gpu_memory_to_use, ignore_zero_frequency = ignore_zero_frequency )

    # n_components = covariance_cols["est_mask"].shape[-1]
    # u['real'], s['real'] = real_svd3(covariance_cols["est_mask"],  picked_frequencies, volume_shape, vol_batch_size, n_components = n_components)
    # Will cause out of memory problems.
    # if volume_shape[0] <= 128:
    #     u['real4'], s['real4'] = real_svd4(covariance_cols["est_mask"],  picked_frequencies, volume_shape, vol_batch_size, n_components = n_components)
    # plot_utils.plot_cov_results(u,s)
    return u, s


def pca_by_projected_covariance(cryos, basis, mean, noise_variance, volume_mask, disc_type , disc_type_u, gpu_memory_to_use= 40, use_mask = True, parallel_analysis = False ,ignore_zero_frequency = False, n_pcs_to_compute = -1):

    # basis_size = basis.shape[-1]
    basis_size = n_pcs_to_compute
    basis = basis[:,:basis_size]

    ####
    memory_left_over_after_kron_allocate = utils.get_gpu_memory_total() -  2*basis_size**4*8/1e9
    batch_size = utils.get_embedding_batch_size(basis, cryos[0].image_size, np.ones(1), basis_size, memory_left_over_after_kron_allocate )

    logger.info('batch size for covariance computation: ' + str(batch_size))

    covariance = covariance_estimation.compute_projected_covariance(cryos, mean, basis, volume_mask, noise_variance, batch_size,  disc_type, disc_type_u, parallel_analysis = parallel_analysis, do_mask_images = use_mask )

    ss, u = np.linalg.eigh(covariance)
    u =  np.fliplr(u)
    s = np.flip(ss)
    u = basis @ u 

    s = np.where(s >0 , s, np.ones_like(s)*constants.EPSILON)
 
    return u , s


# EVERYTHING BELOW HERE IS NOT USED IN CURRENT VERSION OF THE CODE. DELETE?

def knock_out_mean_component_2(u,s, mean, volume_mask, volume_shape, vol_batch_size,ignore_zero_frequency):
    # This assumes s has been kept around
    # cov == u s u^*
    # Want to compute eigendecomposition of the projection onto complement of mean:
    # (I - qq^*) cov ( I - q q^*)
    
    volume_size = np.prod(volume_shape)
    u_real = linalg.batch_idft3(u, volume_shape, vol_batch_size).real
    u_real /= np.linalg.norm(u_real, axis =0)
    # u2_norm = np.linalg.norm(u_real, axis =0)

    # Mask mean
    masked_mean = ( ftu.get_idft3(mean.reshape(volume_shape)) * volume_mask.reshape(volume_shape) ).reshape(-1).real
    masked_mean /= np.linalg.norm(masked_mean)
    
    # Make it orthogonal to mask
    if ignore_zero_frequency:
        # knockout volume_mask direction stuff?
        norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
        # substract component in direction of mask?
        # Apply matrix (I - mask mask.T / \|mask^2\| ) 
        masked_mean -= norm_volume_mask * (norm_volume_mask.T @ masked_mean)


    # Project out the mean
    u_m_proj = u_real - masked_mean[:,None] @  (np.conj(masked_mean).T @ u_real )[None]
    cov_chol = u_m_proj * np.sqrt(s)

    # Reorthogonalize
    # Replaced by a slower but stable.
    # new_u, new_s, _ = linalg.thin_svd_in_blocks(cov_chol)
    cov_chol = jax.device_put(cov_chol, device=jax.devices("cpu")[0])
    new_u, new_s, _ = jnp.linalg.svd(cov_chol, full_matrices = False)

    new_u = np.array(new_u) 
    new_s = np.array(new_s)


    ones_vol = np.ones_like(masked_mean)
    ones_vol /= np.linalg.norm(ones_vol)

    # Align to positive. Not really necessary, but 
    ip = ones_vol.T @ new_u
    ip = np.where(np.abs(ip) > constants.ROOT_EPSILON, ip / np.abs(ip) , np.ones_like(ip))
    new_u *= ip

    # back to Fourier domain
    new_u = linalg.batch_dft3(new_u, volume_shape, vol_batch_size)
    new_u /= np.linalg.norm(new_u, axis =0)
    
    return np.array(new_u.astype(u.dtype)), np.array(new_s.astype(s.dtype)**2)



# A lot of implementation of the same things, having to do with taking the real SVD of
# the columns of Sigma_col
# - Third does a randomized SVD in the spatial domain.

def flip_vec(column, volume_shape):
    column = column.reshape(volume_shape)
    column_flipped = jnp.zeros_like(column)
    column_flipped = column_flipped.at[1:,1:,1:].set(jnp.conj(jnp.flip(column[1:,1:,1:])))
    return column_flipped.reshape(-1)

def get_zero_boundary_mask(volume_shape, dtype):
    ones = np.zeros(volume_shape, dtype = dtype)
    ones[1:,1:,1:] = 1
    return ones.reshape(-1)


def get_minus_vec_index(picked_v_idx,volume_shape):
        # Get - vec
        freq = core.vec_indices_to_frequencies(picked_v_idx, volume_shape)
        minus_idx = core.frequencies_to_vec_indices(-freq, volume_shape)
        return minus_idx
    
flip_vec_cpu = jax.jit(flip_vec, backend = 'cpu', static_argnums = (1,))
batch_flip_vec = jax.vmap(flip_vec_cpu, in_axes = (1, None))

def make_symmetric_columns(columns, picked_frequencies, volume_shape):
    freqs = core.vec_indices_to_frequencies(picked_frequencies, volume_shape)
    
    good_idx = freqs[:,0] > 0
    # freqs = freqs.at[good_idx].get()
    minus_freqs = -freqs
    minus_indices = core.frequencies_to_vec_indices(minus_freqs, volume_shape)
    columns_flipped = batch_flip_vec(columns, volume_shape)
    
    return columns_flipped.T, minus_indices, good_idx

make_symmetric_columns_cpu = jax.jit(make_symmetric_columns, backend = 'cpu', static_argnums = (2))

# to check if there is. abug... delete?
def make_symmetric_columns_np(columns, picked_frequencies, volume_shape):
    freqs = np.array(core.vec_indices_to_frequencies(picked_frequencies, volume_shape))
    
    good_idx = freqs[:,0] > 0
    # freqs = freqs.at[good_idx].get()
    minus_freqs = -freqs
    minus_indices = np.array(core.frequencies_to_vec_indices(minus_freqs, volume_shape))
    columns_flipped = batch_flip_vec2(columns, volume_shape)
    return columns_flipped.T, minus_indices, good_idx
 

def batch_flip_vec2(columns,volume_shape):
    mapped_idx = np.array(get_minus_vec_index(np.arange(np.prod(volume_shape)),volume_shape))
    one_mask = get_zero_boundary_mask(volume_shape, columns.dtype)
    return np.conj(columns[mapped_idx,:] * one_mask[...,None]).T


def IDFT_from_both_sides(cube_smaller_matrix, left_volume_shape, right_volume_shape, vol_batch_size_left, vol_batch_size_right):
    # Apply fft along rows
    # Compute C = F Sigma
    # Then C^*
    cube_smaller_matrix = np.conj(linalg.batch_idft3(cube_smaller_matrix, left_volume_shape, vol_batch_size_left).T)
    # Apply fft along cols
    
    # Computes C F^* by (F C^*)^*
    cube_smaller_matrix = np.conj(linalg.batch_idft3(cube_smaller_matrix, right_volume_shape, vol_batch_size_right).T)
    return cube_smaller_matrix

def get_all_copied_columns(columns, picked_frequencies, volume_shape):
    
    # Make symmetric columns
    columns_flipped, minus_indices, good_idx = make_symmetric_columns_np(columns, picked_frequencies, volume_shape)
    all_frequencies = np.concatenate([picked_frequencies, minus_indices[good_idx]])
    all_columns = np.concatenate([columns, columns_flipped[:,good_idx]], axis =-1)
    return all_columns, all_frequencies


# IMPLEMENTS THE TWO MATVECS WE NEED TO RUN THE RANDOMIZED SVD.

def right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time = time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    logger.info(f"make big mat 1 {time.time() - st_time}")
    # columns_flipped = np.array(columns_flipped)
    utils.report_memory_device(logger=logger)

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # F_2r^* test_mat
    F_t = linalg.batch_dft3(test_mat, smaller_vol_shape, vol_batch_size) / smaller_vol_size
    logger.info(f"DFT time 1 {time.time() - st_time}")

    
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
    utils.report_memory_device(logger=logger)

    C_F_t = linalg.blockwise_A_X(columns, F_t[original_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use)
    logger.info(f"AX 1: {time.time() - st_time}")

    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))

    columns_flipped = columns_flipped[:,good_idx]
    F_t2 = F_t[flipped_frequencies_indices_in_smaller,:].copy()
    C_F_t_2 = linalg.blockwise_A_X(columns_flipped , F_t2, memory_to_use = memory_to_use) 
    C_F_t += C_F_t_2 
    logger.info(f"AX 2: {time.time() - st_time}")


    F_C_F_t = linalg.batch_idft3(C_F_t, volume_shape, vol_batch_size)
    logger.info(f"IDFT: {time.time() - st_time}")

    return F_C_F_t

def right_matvec_with_spatial_Sigma_v2(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time = time.time()
    # Some precompute

    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]

    logger.info(f"make big mat 1 {time.time() - st_time}")
    # columns_flipped = np.array(columns_flipped)
    utils.report_memory_device(logger=logger)

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # F_2r^* test_mat
    F_t = linalg.batch_dft3(test_mat, smaller_vol_shape, vol_batch_size) / smaller_vol_size
    logger.info(f"DFT time 1 {time.time() - st_time}")

    
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
    utils.report_memory_device(logger=logger)

    C_F_t = linalg.blockwise_A_X(columns, F_t[original_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use)
    logger.info(f"AX 1: {time.time() - st_time}")

    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = np.array(core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape))

    C_F_t += linalg.blockwise_A_X(columns_flipped , F_t[flipped_frequencies_indices_in_smaller,:], memory_to_use = memory_to_use) 
    # C_F_t += C_F_t_2 
    logger.info(f"AX 2: {time.time() - st_time}")

    # Store FT in same array
    C_F_t = linalg.batch_idft3(C_F_t, volume_shape, vol_batch_size)
    logger.info(f"IDFT 1: {time.time() - st_time}")

    return C_F_t



def left_matvec_with_spatial_Sigma(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    st_time =time.time()
    # Some precompute
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]

    # Compute frequencies and all that stuff...
    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    
    ## TODO: CHECK THAT THIS MAKES SENSE
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # Now do compute:
    # F = IDFT here
    # so F^* = DFT
    
    # Q should be real I think?
    F_star_Q_star = linalg.batch_dft3( np.conj(Q), volume_shape, vol_batch_size) / np.prod(volume_shape)
    # Q_F = np.conj(F_star_Q_star)
    Q_F = F_star_Q_star
    logger.info(f"DFT: {time.time() - st_time}")

    # Frequencies in new grid
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
        
    Q_F_C = np.zeros((Q.shape[-1], smaller_vol_size), dtype = columns.dtype)
    Q_F_C[:,original_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")

    # Flipped Frequencies in new grid
    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape)
    flipped_frequencies_indices_in_smaller = np.array(flipped_frequencies_indices_in_smaller)
    # Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_A_X(columns_flipped[:,good_idx].T, Q_F.T, memory_to_use = 20).T

    Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q_F, columns_flipped, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")
    
    # DFT back
    # X F^* = (F X^*)^*
    Q_F_C_F = np.conj(linalg.batch_idft3(np.conj(Q_F_C).T, smaller_vol_shape, vol_batch_size)).T
    logger.info(f"DFT2: {time.time() - st_time}")

    return Q_F_C_F

use_v2_fn = True
report_memory = True

def left_matvec_with_spatial_Sigma_v2(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = 40):
    if report_memory:
        utils.report_memory_device(logger=logger)

    st_time =time.time()
    Q_shape_inp = Q.shape
    # Some precompute
    # Compute frequencies and all that stuff...
    columns_flipped, minus_frequency_indices, good_idx = make_symmetric_columns_np(columns, picked_frequency_indices, volume_shape)
    columns_flipped = columns_flipped[:,good_idx]

    all_frequency_indices = np.concatenate([picked_frequency_indices, minus_frequency_indices[good_idx]])
    all_frequencies = core.vec_indices_to_frequencies(all_frequency_indices, volume_shape)
    if report_memory:
        logger.info(f"after make symmetric columns")
        utils.report_memory_device(logger=logger)

    ## TODO: CHECK THAT THIS MAKES SENSE
    # Size of smaller grid.
    smaller_size = int(2 * (np.max(np.abs(all_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])
    smaller_vol_size = np.prod(smaller_vol_shape)
    
    # Now do compute:
    # F = IDFT here
    # so F^* = DFT
    
    # Q should be real I think?
    # 2x memory of Q
    Q = linalg.batch_dft3( np.conj(Q), volume_shape, vol_batch_size) / np.prod(volume_shape)
    # Q_F = np.conj(F_star_Q_star)
    # Q_F = F_star_Q_star
    logger.info(f"DFT: {time.time() - st_time}")
    utils.report_memory_device(logger=logger)


    # Frequencies in new grid
    original_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    original_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(original_frequencies, smaller_vol_shape)
        
    Q_F_C = np.zeros((Q_shape_inp[-1], smaller_vol_size), dtype = columns.dtype)
    Q_F_C[:,original_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q, columns, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")
    utils.report_memory_device(logger=logger)

    # Flipped Frequencies in new grid
    flipped_frequencies = core.vec_indices_to_frequencies(minus_frequency_indices[good_idx], volume_shape)
    flipped_frequencies_indices_in_smaller = core.frequencies_to_vec_indices(flipped_frequencies, smaller_vol_shape)
    flipped_frequencies_indices_in_smaller = np.array(flipped_frequencies_indices_in_smaller)
    # Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_A_X(columns_flipped[:,good_idx].T, Q_F.T, memory_to_use = 20).T


    Q_F_C[:,flipped_frequencies_indices_in_smaller] = linalg.blockwise_Y_T_X(Q, columns_flipped, memory_to_use = memory_to_use)
    logger.info(f"Y^T @ X: {time.time() - st_time}")
    utils.report_memory_device(logger=logger)

    del columns_flipped

    # DFT back
    # X F^* = (F X^*)^*
    Q_F_C_F = np.conj(linalg.batch_idft3(np.conj(Q_F_C).T, smaller_vol_shape, vol_batch_size)).T
    logger.info(f"DFT2: {time.time() - st_time}")
    utils.report_memory_device(logger=logger)

    return Q_F_C_F


def randomized_real_svd_of_columns(columns, picked_frequency_indices, volume_mask, volume_shape, vol_batch_size, test_size = 300, gpu_memory_to_use= 40, ignore_zero_frequency = False):
    st_time = time.time()


    # memory_to_use = utils.get_gpu_memory_total() - 5
    utils.report_memory_device(logger=logger)
    picked_frequencies = core.vec_indices_to_frequencies(picked_frequency_indices, volume_shape)
    smaller_size = int(2 * (np.max(np.abs(picked_frequencies)) + 1))
    smaller_vol_shape = tuple(3*[smaller_size])

    smaller_vol_size = np.prod(smaller_vol_shape)
    test_mat = np.random.randn(smaller_vol_size, test_size).real.astype(np.float32)

    st_time = time.time()
    if use_v2_fn:
        Q = right_matvec_with_spatial_Sigma_v2(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    else:
        Q = right_matvec_with_spatial_Sigma(test_mat, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use ).real.astype(np.float32)
    del test_mat

    ## Do masking here ?
    # Q *= volume_mask.reshape(-1,1)

    if ignore_zero_frequency:
        # knockout volume_mask direction stuff?
        norm_volume_mask = volume_mask.reshape(-1) / np.linalg.norm(volume_mask)
        # substract component in direction of mask?
        # Apply matrix (I - mask mask.T / \|mask^2\| ) 
        Q -= np.outer(norm_volume_mask, (norm_volume_mask.T @ Q))
        logger.info('ignoring zero frequency')

    logger.info(f"right matvec {time.time() - st_time}")
    utils.report_memory_device(logger=logger)
    Q = jax.device_put(Q, device=jax.devices("cpu")[0])
    Q,_ = jnp.linalg.qr(Q)
    Q = np.array(Q) # I don't know why but not doing this causes massive slowdowns sometimes?
    logger.info(f"QR time: {time.time() - st_time}")

    # In principle, should apply (I - mask mask.T / \|mask\|^2 )  again, but should already be orthogonal
    # 
    # Q_mask = Q*volume_mask.reshape(-1,1)
    # Q_mask = Q
    utils.report_memory_device(logger=logger)
    if use_v2_fn:
        C_F_t_2 = left_matvec_with_spatial_Sigma_v2(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    else:
        C_F_t_2 = left_matvec_with_spatial_Sigma(Q, columns, picked_frequency_indices, volume_shape, vol_batch_size, memory_to_use = gpu_memory_to_use).real.astype(np.float32)
    utils.report_memory_device(logger=logger)
    logger.info(f"left matvec {time.time() - st_time}")


    U,S,V = np.linalg.svd(C_F_t_2, full_matrices = False)
    logger.info(f"big SVD {time.time() - st_time}")
    utils.report_memory_device(logger=logger)

    vol_size = np.prod(volume_shape)
    # To save some memory... Q = FQ
    Q = linalg.batch_dft3(Q, volume_shape, vol_batch_size)
    Q = linalg.blockwise_A_X(Q, U, memory_to_use = gpu_memory_to_use) / np.sqrt(vol_size)
    logger.info(f"FQU matvec {time.time() - st_time}")
    utils.report_memory_device(logger=logger)

    volume_size = np.prod(volume_shape)
    # Factors due to IDFT on both sides
    S_fd = S * np.sqrt(smaller_vol_size) * np.sqrt(volume_size)
    return np.array(Q), np.array(S_fd), np.array(V)



def test_different_embeddings(cryos, volume_mask, mean_estimate, basis, eigenvalues, noise_variance, batch_size, disc_type = 'linear_interp', use_contrast = False, zdims= None):
    ## TODO
    
    # noise.get_average_residual_square_v2(cryos[0], volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

    # likelihoods = np.array([cryo.likelihood for cryo in cryos])
    from recovar import noise, embedding
    residuals = np.zeros(zdims.size)
    cryo = cryos[0]
    contrast_grid = np.ones([1])
    residuals_flipped = np.zeros(zdims.size)
    for zdim_idx, zdim in enumerate(zdims):
        if zdim > 0:
            basis_coordinates, image_latent_covariances, estimated_contrasts = embedding.get_coords_in_basis_and_contrast_3(cryo, mean_estimate, basis[:,:zdim], eigenvalues[:zdim], volume_mask, noise_variance, contrast_grid, batch_size, disc_type, parallel_analysis = False, compute_covariances = False )
        else:
            basis_coordinates = np.zeros((cryo.n_images, 0))
        logger.info("embd ")#, end="")
        contrasts = np.ones(cryo.n_images)
        residuals[zdim_idx] = noise.get_average_residual_square_v3(cryos[0], volume_mask, mean_estimate, basis[:,:zdim].T, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

        # import pdb; pdb.set_trace()
        basis_flipped = basis[:,:zdim].copy()
        basis_flipped[:,zdim-1] = basis[:,zdim-1] * (np.random.randint(0,2, size = basis[:,zdim-1].shape)*2 - 1)

        if zdim > 0:
            basis_coordinates, image_latent_covariances, estimated_contrasts = embedding.get_coords_in_basis_and_contrast_3(cryo, mean_estimate, basis_flipped, eigenvalues[:zdim], volume_mask, noise_variance, contrast_grid, batch_size, disc_type, parallel_analysis = False, compute_covariances = False )
        else:
            basis_coordinates = np.zeros((cryo.n_images, 0))
        logger.info("embd ")#, end="")
        contrasts = np.ones(cryo.n_images)
        residuals_flipped[zdim_idx] = noise.get_average_residual_square_v3(cryos[0], volume_mask, mean_estimate, basis_flipped.T, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')


        logger.info(f"zdim {zdim}")#, end="")
    return residuals, residuals_flipped




def test_different_embeddings_from_volumes(cryos, zs, cov_zs, noise_variance, zdims= None, n_images = 1000, volume_mask= None):
    ## TODO
    
    # noise.get_average_residual_square_v2(cryos[0], volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

    # likelihoods = np.array([cryo.likelihood for cryo in cryos])
    from recovar import noise, embedding
    from recovar import latent_density
    metrics = {'locerr_median':np.zeros(zdims.size), 'locerr_90':np.zeros(zdims.size), 'locerr_mean':np.zeros(zdims.size)}

    global_likely_choice1 = np.zeros(zdims.size)
    global_likely_choice2 = np.zeros(zdims.size)
    all_estimators = {}
    all_lhs = {}

    # First pick some points?
    zdim_target = 4
    zs_this = zs[zdim_target]#[:cryo.n_images]
    covs_this = cov_zs[zdim_target]#[:cryo.n_images]

    image_likelihoods = latent_density.compute_latent_space_density_at_pts(zs_this,zs_this,  covs_this)
    most_likelihood_idx = np.argmax(image_likelihoods)

    percentages = np.array([ 0, 0.1 ])#, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    index = np.round(percentages * zs_this.shape[0] ).astype(int)
    likelihoods_sorted = np.flip(np.argsort(image_likelihoods))
    images_chosen_index = likelihoods_sorted[index]


    # print(most_likelihood_idx)



    for zdim_idx, zdim in enumerate(zdims):
        zs_this = zs[zdim]#[:cryo.n_images]
        covs_this = cov_zs[zdim]#[:cryo.n_images]
        zs_split = [zs_this[:cryos[0].n_images], zs_this[cryos[0].n_images:]]
        cov_zs_split = [covs_this[:cryos[0].n_images], covs_this[cryos[0].n_images:]]

        for most_likelihood_idx in images_chosen_index:
            best_zs = zs_this[most_likelihood_idx]
            lhs, rhs, estimators = {}, {}, {}
            all_estimators[zdim] = {}
            all_lhs[zdim] = {}
            real_estimators = {}


            for cryo_idx, cryo in enumerate(cryos):
                # Compute likelihood to best_zs
                # Take n_images most likely image for this point
                best_likelihood = latent_density.compute_latent_quadratic_forms_in_batch(best_zs[None], zs_split[cryo_idx], cov_zs_split[cryo_idx]).reshape(-1)
                single_bin = np.array(np.sort(best_likelihood)[n_images-1])[None]
                inds = np.digitize(best_likelihood, single_bin, right = True)
                images_used = inds == 0 
                # if zdim >1:
                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.scatter(zs_split[cryo_idx][:,0], zs_split[cryo_idx][:,1])
                #     plt.scatter(zs_split[cryo_idx][images_used,0], zs_split[cryo_idx][images_used,1])
                #     plt.scatter(best_zs[0], best_zs[1])
                #     plt.show()
                # # print('n in batch:,', np.sum(inds == 0))

                # import pdb; pdb.set_trace()
                # Compute two estimators
                from recovar import adaptive_kernel_discretization
                estimators[cryo_idx], lhs[cryo_idx], rhs[cryo_idx] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[cryo_idx], noise_variance.astype(np.float32), None, best_likelihood, single_bin, tau= None, grid_correct=False, use_spherical_mask=False, return_lhs_rhs=True)
                lhs[cryo_idx] = adaptive_kernel_discretization.half_volume_to_full_volume(lhs[cryo_idx][0], cryos[cryo_idx].volume_shape)
                lhs[cryo_idx] = (lhs[cryo_idx] * cryos[0].get_valid_frequency_indices())#.reshape(cryos[0].volume_shape)
                real_estimators[cryo_idx] = ftu.get_idft3(estimators[cryo_idx].reshape(cryos[0].volume_shape))
                all_estimators[zdim][cryo_idx] = estimators[cryo_idx]
                all_lhs[zdim][cryo_idx] = lhs[cryo_idx]


            weighting1 = 1/(1/lhs[0] + 1/lhs[1])
            # weighting1 = lhs[0]
            #weighted0 = estimators[0] * np.sqrt(lhs[1])

            # global_likely_choice1[zdim_idx] += jnp.linalg.norm( (estimators[0] - estimators[1]) )**2
            global_likely_choice1[zdim_idx] += jnp.linalg.norm(np.sqrt(weighting1) * (estimators[0] - estimators[1]) )**2
            from recovar import locres
            weighting1 = weighting1.reshape(cryos[0].volume_shape)
            loc_error = locres.expensive_local_error_with_cov(real_estimators[0], real_estimators[1], cryos[0].voxel_size, weighting1.reshape(real_estimators[0].shape), locres_sampling = 25, locres_maskrad= 25, locres_edgwidth= 0)
            volume_mask = volume_mask > 1e-3
            good_errors = loc_error.reshape(-1)[volume_mask.reshape(-1)]
            metrics['locerr_median'][zdim_idx] += np.median(good_errors)
            metrics['locerr_90'][zdim_idx] += np.percentile(good_errors, 90)
            metrics['locerr_mean'][zdim_idx] += np.mean(good_errors)



            # global_likely_choice1[zdim_idx] = locres.expensive_local_error_with_cov(estimators[0], estimators[1], cryos[0].voxel_size, weighting1.reshape(target0.shape), locres_sampling = 25, locres_maskrad= 25, locres_edgwidth= 0)


        logger.info(f"zdim {zdim}")#, end="")
    return global_likely_choice1, metrics, all_estimators, all_lhs




def test_different_embeddings_from_variance(cryos, zs, cov_zs, noise_variance, zdims= None, n_images = 1000, tau = None):
    ## TODO
    
    # noise.get_average_residual_square_v2(cryos[0], volume_mask, mean_estimate, basis, contrasts,basis_coordinates, batch_size, disc_type = 'linear_interp')

    # likelihoods = np.array([cryo.likelihood for cryo in cryos])
    from recovar import noise, embedding
    from recovar import latent_density
    zdims = np.asarray(zdims)
    metrics = {'variance':np.zeros(zdims.size), 'filt_var': np.zeros(zdims.size) , 'var_avg': {}, 'variance_lp': np.zeros(zdims.size) }
    for k in range(zdims.size):
        metrics['var_avg'][k] = 0

    all_estimators = {}
    all_lhs = {}

    # # First pick some points?
    # zdim_target = 4
    # zs_this = zs[zdim_target]#[:cryo.n_images]
    # covs_this = cov_zs[zdim_target]#[:cryo.n_images]

    # image_likelihoods = latent_density.compute_latent_space_density_at_pts(zs_this,zs_this,  covs_this)
    # most_likelihood_idx = np.argmax(image_likelihoods)

    # percentages = np.array([ 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ])
    

    # index = np.round(percentages * zs_this.shape[0] ).astype(int)
    # likelihoods_sorted = np.flip(np.argsort(image_likelihoods))
    # images_chosen_index = likelihoods_sorted[index]


    images_chosen_index = np.random.randint(0, zs[zdims[0]].shape[0], 30)
    first = True
    for zdim_idx, zdim in enumerate(zdims):
        zs_this = zs[zdim]#[:cryo.n_images]
        covs_this = cov_zs[zdim]#[:cryo.n_images]
        zs_split = [zs_this[:cryos[0].n_images], zs_this[cryos[0].n_images:]]
        cov_zs_split = [covs_this[:cryos[0].n_images], covs_this[cryos[0].n_images:]]

        for most_likelihood_idx in images_chosen_index:
            best_zs = zs_this[most_likelihood_idx]
            lhs, rhs, estimators = {}, {}, {}
            all_estimators[zdim] = {}
            all_lhs[zdim] = {}
            real_estimators = {}


            for cryo_idx, cryo in enumerate(cryos):
                # Compute likelihood to best_zs
                # Take n_images most likely image for this point
                best_likelihood = latent_density.compute_latent_quadratic_forms_in_batch(best_zs[None], zs_split[cryo_idx], cov_zs_split[cryo_idx]).reshape(-1)
                single_bin = np.array(np.sort(best_likelihood)[n_images-1])[None]
                inds = np.digitize(best_likelihood, single_bin, right = True)
                images_used = inds == 0 
                import matplotlib.pyplot as plt
                # if zdim >1:
                #     plt.figure()
                #     plt.scatter(zs_split[cryo_idx][:,0], zs_split[cryo_idx][:,1])
                #     plt.scatter(zs_split[cryo_idx][images_used,0], zs_split[cryo_idx][images_used,1])
                #     plt.scatter(best_zs[0], best_zs[1])
                #     plt.show()
                # # print('n in batch:,', np.sum(inds == 0))

                # import pdb; pdb.set_trace()
                # Compute two estimators
                from recovar import adaptive_kernel_discretization, relion_functions, regularization, utils
                estimators[cryo_idx], lhs[cryo_idx], rhs[cryo_idx] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[cryo_idx], noise_variance.astype(np.float32), None, best_likelihood, single_bin, tau= tau, grid_correct=True, use_spherical_mask=True, return_lhs_rhs=True)

                # if zdim == 4:
                #     from recovar import output
                #     plt.figure()
                #     plt.scatter(zs_split[cryo_idx][:,0], zs_split[cryo_idx][:,1])
                #     plt.scatter(zs_split[cryo_idx][images_used,0], zs_split[cryo_idx][images_used,1])
                #     plt.scatter(best_zs[0], best_zs[1])
                #     plt.show()
                #     print(cryos[cryo_idx].dataset_indices[images_used])
                #     output.save_volume(estimators[cryo_idx], f"estimator_{cryo_idx}", from_ft = True)
                #     # return cryos[cryo_idx].dataset_indices[images_used]
                #     import pdb; pdb.set_trace()
                image_subset = np.argwhere(images_used).reshape(-1)
                Ft_ctf, Ft_y = relion_functions.residual_relion_style_triangular_kernel(cryos[cryo_idx], estimators[cryo_idx][0], noise_variance.astype(np.float32) * 0 ,  100,  index_subset = image_subset)
                Ft_y = (Ft_y * cryos[0].get_valid_frequency_indices())#.reshape(cryos[0].volume_shape)

                if first == True:
                    Ft_ctf_avg = regularization.average_over_shells(Ft_ctf, cryos[0].volume_shape) / 10
                    Ft_avg_radial = utils.make_radial_image(Ft_ctf_avg, cryos[0].volume_shape)
                    first = False

                variance_estimate = Ft_y / (Ft_ctf + Ft_avg_radial)
                avg_variance = regularization.average_over_shells(variance_estimate, cryos[0].volume_shape)
                plt.plot(avg_variance)
                # plt.show()

                # avg_variance = regularization.average_over_shells(Ft_y, cryos[0].volume_shape)
                # plt.plot(avg_variance)
                # plt.show()
                # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()
            metrics['var_avg'][zdim_idx] = avg_variance
            metrics['filt_var'][zdim_idx] += np.sum(Ft_y)
            metrics['variance'][zdim_idx] += np.sum(variance_estimate)
            metrics['variance_lp'][zdim_idx] += np.sum(variance_estimate * cryos[0].get_valid_frequency_indices(rad = 20))

            # global_likely_choice1[zdim_idx] = locres.expensive_local_error_with_cov(estimators[0], estimators[1], cryos[0].voxel_size, weighting1.reshape(target0.shape), locres_sampling = 25, locres_maskrad= 25, locres_edgwidth= 0)


        logger.info(f"zdim {zdim}")#, end="")
    return metrics, all_estimators, all_lhs

