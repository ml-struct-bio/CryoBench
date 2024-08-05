import numpy as np
from recovar import core
from recovar import regularization, constants, utils, padding, mask
from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
ftu = fourier_transform_utils(jnp)
import logging, functools, jax

logger = logging.getLogger(__name__)


def griddingCorrect(vol_in, ori_size, padding_factor, order = 0,):

    # Correct real-space map by dividing it by the Fourier transform of the interpolator(s)
    pixels = ftu.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) + 0.
    og_shape = vol_in.shape
    r = np.linalg.norm(pixels, axis = -1)
    vol_in = vol_in.reshape(-1)

    mask = r > 0.
    
    rval = r / (ori_size * padding_factor)
    rval[~mask] = 1.
    sinc = np.sin(np.pi * rval) / (np.pi * rval)
    sinc[~mask] = 1.

    if order ==0:
        vol_out = vol_in/ sinc
    elif order ==1:
        vol_out = vol_in/ (sinc**2)
        sinc = sinc**2
    else:
        raise ValueError("Order not implemented")
    
    return vol_out.reshape(og_shape), sinc.reshape(og_shape)

# I think this is the correct Fourier transform of the trilinear interpolator: sinc(x) * sinc(y) * sinc(z)
def griddingCorrect_square(vol_in, ori_size, padding_factor, order = 0,):
    og_shape = vol_in.shape

    pixels = ftu.get_k_coordinate_of_each_pixel(vol_in.shape, 1, scaled = False) 
    pixels_rescaled = pixels / (ori_size * padding_factor)

    def sinc(ar):
        # ar_scaled = ar / (ori_size * padding_factor)
        return jnp.where(jnp.abs(ar) < 1e-8, 1., jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

    if order ==0:
        kernel = sinc
    elif order ==1:
        kernel = lambda x : sinc(x)**2
    else:
        raise ValueError("Order not implemented")

    kernel_ar = kernel(pixels_rescaled[:,0]) * kernel(pixels_rescaled[:,1]) * kernel(pixels_rescaled[:,2])
    vol_out = vol_in / kernel_ar.reshape(og_shape)

    return vol_out.reshape(og_shape), kernel_ar.reshape(og_shape)


# My understanding of what relion does.
def relion_style_triangular_kernel(experiment_dataset , cov_noise,  batch_size,  disc_type = 'linear_interp', return_lhs_rhs = False ):
    
    data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    Ft_y, Ft_ctf = 0, 0 

    for batch, indices in data_generator:

        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        Ft_y_b, Ft_ctf_b = relion_style_triangular_kernel_batch(batch,
                                                                experiment_dataset.CTF_params[indices], 
                                                                experiment_dataset.rotation_matrices[indices], 
                                                                experiment_dataset.translations[indices], 
                                                                experiment_dataset.image_shape, 
                                                                experiment_dataset.upsampled_volume_shape, 
                                                                experiment_dataset.voxel_size, 
                                                                experiment_dataset.CTF_fun, 
                                                                disc_type, 
                                                                cov_noise)
        Ft_y += Ft_y_b
        Ft_ctf += Ft_ctf_b
    # To agree with order of other fcns.
    return Ft_ctf, Ft_y

@functools.partial(jax.jit, static_argnums=[4,5,6,7,8])
def relion_style_triangular_kernel_batch(images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, cov_noise):
    # images = process_images(images, apply_image_mask = True)
    
    images = core.translate_images(images, translations, image_shape) / cov_noise
    # Ft_y = core.adjoint_forward_model_from_trilinear(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 
    Ft_y = core.adjoint_forward_model_from_map(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 


    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / cov_noise
    # Ft_ctf = core.adjoint_forward_model_from_trilinear(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 
    Ft_ctf = core.adjoint_forward_model_from_map(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 

    return Ft_y, Ft_ctf


@functools.partial(jax.jit, static_argnums=[4,5,6,7,8])
def relion_style_triangular_kernel_batch_trilinear(images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, cov_noise):
    # images = process_images(images, apply_image_mask = True)
    
    images = core.translate_images(images, translations, image_shape) / cov_noise
    Ft_y = core.adjoint_forward_model_from_trilinear(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 
    # Ft_y = core.adjoint_forward_model_from_map(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 


    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / cov_noise
    Ft_ctf = core.adjoint_forward_model_from_trilinear(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 
    # Ft_ctf = core.adjoint_forward_model_from_map(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 

    return Ft_y, Ft_ctf


@functools.partial(jax.jit, static_argnums=[4,5,6,7,8])
def relion_style_triangular_kernel_batch_trilinear_linear_in_het(images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, cov_noise, het_coords):
    # Assumes het_coords := het_coords - target
    
    images = core.translate_images(images, translations, image_shape) / cov_noise
    Ft_y = core.adjoint_forward_model_from_trilinear(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 
    # Ft_y = core.adjoint_forward_model_from_map(images, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 


    CTF = CTF_fun( CTF_params, image_shape, voxel_size) / cov_noise
    Ft_ctf = core.adjoint_forward_model_from_trilinear(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 

    het_CTF = jnp.ones_likes(het_coords)

    # Ft_ctf = core.adjoint_forward_model_from_trilinear(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 

    # Ft_ctf = core.adjoint_forward_model_from_map(CTF, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type) 

    return Ft_y, Ft_ctf


# import functools, jax
@functools.partial(jax.jit, static_argnums=[5,6,7,8])
def residual_relion_style_triangular_kernel_batch_trilinear(mean_estimate, images, CTF_params, rotation_matrices, translations, image_shape, volume_shape, voxel_size, CTF_fun, cov_noise):
    # images = process_images(images, apply_image_mask = True)
    CTF_squared = CTF_fun( CTF_params, image_shape, voxel_size)

    proj_mean = core.slice_volume_by_trilinear(mean_estimate, rotation_matrices, image_shape, volume_shape)
    images = core.translate_images(images, translations, image_shape) 
    # import pdb; pdb.set_trace()

    images = images - core.slice_volume_by_trilinear(mean_estimate, rotation_matrices, image_shape, volume_shape) * CTF_squared

    # Maybe apply mask

    images_squared = jnp.abs(images)**2  - cov_noise #* np.sum(mask) # May need to do something with mask
    CTF_squared = CTF_squared**2
    #images_squared *= CTF_squared

    Ft_y = core.adjoint_slice_volume_by_trilinear(images_squared, rotation_matrices, image_shape, volume_shape)

    Ft_ctf = core.adjoint_slice_volume_by_trilinear(CTF_squared**2, rotation_matrices, image_shape, volume_shape)

    return Ft_y, Ft_ctf




# My understanding of what relion does.
def residual_relion_style_triangular_kernel(experiment_dataset, mean_estimate, cov_noise,  batch_size, index_subset = None):
    if index_subset is None:
        data_generator = experiment_dataset.get_dataset_generator(batch_size=batch_size) 
    else:
        data_generator = experiment_dataset.get_dataset_subset_generator(batch_size=batch_size, subset_indices = index_subset)

    Ft_y, Ft_ctf = 0, 0 
    for batch, indices in data_generator:
        batch = experiment_dataset.image_stack.process_images(batch, apply_image_mask = False)
        Ft_y_b, Ft_ctf_b = residual_relion_style_triangular_kernel_batch_trilinear(mean_estimate, 
                                                                batch,
                                                                experiment_dataset.CTF_params[indices], 
                                                                experiment_dataset.rotation_matrices[indices], 
                                                                experiment_dataset.translations[indices], 
                                                                experiment_dataset.image_shape, 
                                                                experiment_dataset.upsampled_volume_shape, 
                                                                experiment_dataset.voxel_size, 
                                                                experiment_dataset.CTF_fun, 
                                                                cov_noise)
        Ft_y += Ft_y_b
        Ft_ctf += Ft_ctf_b
    return Ft_ctf, Ft_y


def upscale_tau(tau, padding_factor, volume_shape, tau_is_1d = False):

    if not tau_is_1d:
        tau = regularization.average_over_shells(tau, volume_shape)

    # int ires = ROUND(sqrt((RFLOAT)r2) / padding_factor);
    # RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

    # RFLOAT invtau2;
    # if (DIRECT_A1D_ELEM(tau2, ires) > 0.)

    pixels = ftu.get_k_coordinate_of_each_pixel(np.array(volume_shape)*padding_factor, 1, scaled = False)
    radius = jnp.round(jnp.linalg.norm(pixels, axis = -1) / padding_factor).astype(jnp.int32)
    upscaled_tau = tau[radius]

    return upscaled_tau

def adjust_regularization_relion_style(filter, volume_shape, tau = None, padding_factor = 1, max_res_shell = None):

    # Original code here https://github.com/3dem/relion/blob/e5c4835894ea7db4ad4f5b0f4861b33269dbcc77/src/backprojector.cpp#L1082

    # There is an "oversampling" factor of 8 in the FSC, I guess due to the fact that they swap back and forth between a padded and unpadded grid

    if tau is not None:
        oversampling_factor = padding_factor ** (3)
        og_volume_shape = (volume_shape[0]//padding_factor, volume_shape[1]//padding_factor, volume_shape[2]//padding_factor)
        tau = upscale_tau(tau, padding_factor, og_volume_shape, tau_is_1d = False)
        inv_tau = 1 / (oversampling_factor * tau)
        # filter_this =  jnp.where(lhs > 1e-20 , 1/ ( 0.001 * jnp.where(filter > 1e-20, filter, 0 )
        inv_tau = jnp.where( (tau < 1e-20) * (filter > 1e-20 ),  1./ ( 0.001 * filter), inv_tau)
        inv_tau = jnp.where( (tau < 1e-20) * (filter <= 1e-20 ),  0, inv_tau)


        # This is funky business
        # print("WARNING CHECK THIS IS CORRECT OVERSAMPLING SOMEHOW?")
        # tau2 = regularization.average_over_shells(tau, og_volume_shape)
        # tau3 = regularization.average_over_shells(tau_new, volume_shape)
        # import pdb; pdb.set_trace()
        # filter_avg = regularization.average_over_shells(filter, volume_shape)
        # tau_avg = regularization.average_over_shells(inv_tau, volume_shape)
        # print(filter_avg/tau_avg)
        # assert False, ""

        regularized_filter = filter + inv_tau
    else:
        regularized_filter = filter

    # This may be a little different b/c I keep things scaled slightly differently. Perhaps should be fixed in fourier_transform_utils
        
    # Take max of weight of 1/1000 of spherically averaged weight 
    # const RFLOAT weight =  XMIPP_MAX(DIRECT_A3D_ELEM(Fweight, k, i, j), DIRECT_A1D_ELEM(radavg_weight, (ires < r_max) ? ires : (r_max - 1)));
    # Compute spherically averaged 
    avged_reg = regularization.average_over_shells(regularized_filter, volume_shape, frequency_shift = 0) / 1000
    # For the things below that frequency, set them to averaged.
    if max_res_shell is not None:
        avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])
    else:
        max_res_shell = volume_shape[0]//2 - 1
        # avged_reg = avged_reg.at[max_res_shell:].set(avged_reg[max_res_shell - 1])

    avged_reg_volume_shape = utils.make_radial_image(avged_reg, volume_shape)

    regularized_filter = jnp.maximum(regularized_filter, avged_reg_volume_shape)
    regularized_filter = jnp.maximum(regularized_filter, constants.EPSILON)

    return regularized_filter


# @functools.partial(jax.jit, static_argnums=[4,5,6,7,8])
def post_process_from_filter(cryo, Ft_ctf, F_ty, tau = None, disc_type = 'nearest', use_spherical_mask = True, grid_correct = True, gridding_correct = "square", kernel_width = 1 ):
    
    Ft_ctf= Ft_ctf.real
    F_ty =  F_ty * cryo.get_valid_upsampled_frequency_indices() # Zero-out FT outside sphere

    # valid_indices = mask.get_radial_mask(upsampled_volume_shape, radius = radius).reshape(-1)

    # Adjust reg for small values
    Ft_ctf2 = adjust_regularization_relion_style(Ft_ctf, cryo.upsampled_volume_shape, tau = tau, padding_factor = cryo.volume_upsampling_factor, max_res_shell = None)
    
    myreliontest = F_ty / Ft_ctf2
    
    # Window real space
    myreliontest = ftu.get_idft3(myreliontest.reshape(cryo.upsampled_volume_shape))
    from recovar import padding, mask

    myreliontest = padding.unpad_volume_spatial_domain(myreliontest, (cryo.upsampled_grid_size - cryo.grid_size) )
    
    # Soft Spherical mask
    if use_spherical_mask:
        myreliontest, mask2 = mask.soft_mask_outside_map(myreliontest, cosine_width = 3)
    
    # Correct gridding effect
    if grid_correct:
        order = 1 if disc_type == 'linear_interp' else 0

        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        myreliontest, sinc = grid_fn(myreliontest.reshape(cryo.volume_shape), cryo.grid_size, cryo.volume_upsampling_factor/kernel_width, order = order)
        # print(cryo.volume_upsampling_factor/kernel_width)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sinc[sinc.shape[0]//2])
        # plt.colorbar()
        # plt.show()
    myreliontest = ftu.get_dft3(myreliontest.reshape(cryo.volume_shape))

    return myreliontest



@functools.partial(jax.jit, static_argnums=[2,3,5,6,7,8,9])
def post_process_from_filter_v2(Ft_ctf, F_ty, og_volume_shape, volume_upsampling_factor, tau = None, kernel = 'triangular', use_spherical_mask = True, grid_correct = True, gridding_correct = "square", kernel_width = 1, volume_mask = None ):
    
    Ft_ctf= Ft_ctf.real
    upsampled_volume_shape = tuple(3*[(og_volume_shape[0]*volume_upsampling_factor)])
    valid_indices = mask.get_radial_mask(upsampled_volume_shape, radius = upsampled_volume_shape[0]//2-1).reshape(-1).astype(Ft_ctf.dtype)
    F_ty =  F_ty * valid_indices # Zero-out FT outside sphere

    # Adjust reg for small values
    Ft_ctf2 = adjust_regularization_relion_style(Ft_ctf, upsampled_volume_shape, tau = tau, padding_factor = volume_upsampling_factor, max_res_shell = None)
    
    myreliontest = F_ty / Ft_ctf2
    
    # Window real space
    myreliontest = ftu.get_idft3(myreliontest.reshape(upsampled_volume_shape))

    myreliontest = padding.unpad_volume_spatial_domain(myreliontest, (upsampled_volume_shape[0] - og_volume_shape[0]) )
    
    # Soft Spherical mask
    if use_spherical_mask:
        myreliontest, mask2 = mask.soft_mask_outside_map(myreliontest, cosine_width = 3)
    
    if volume_mask is not None:
        logger.warning("should probably check whether this is correct") 
        myreliontest = myreliontest * volume_mask

    # Correct gridding effect
    if grid_correct:

        if kernel == 'triangular':
            order = 1
        elif kernel == 'square':
            order = 0
        else:
            raise ValueError("Kernel not implemented")
        # order = 1 if disc_type == 'linear_interp' else 0

        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        myreliontest, sinc = grid_fn(myreliontest.reshape(og_volume_shape), og_volume_shape[0], volume_upsampling_factor/kernel_width, order = order)
        # print(cryo.volume_upsampling_factor/kernel_width)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sinc[sinc.shape[0]//2])
        # plt.colorbar()
        # plt.show()
    myreliontest = ftu.get_dft3(myreliontest.reshape(og_volume_shape))

    return myreliontest.astype(F_ty.dtype)


batch_post_process_from_filter =  jax.vmap(post_process_from_filter_v2, in_axes = (0, 0, None, None, None, None, None, None, None) )


def relion_reconstruct(cryo, noise_variance, batch_size = 100, disc_type = 'linear_interp', use_spherical_mask = True, upsampling_factor = 2, grid_correct = True, gridding_correct = "square", tau = None ):

    og_upsampling = cryo.volume_upsampling_factor
    cryo.update_volume_upsampling_factor(upsampling_factor)

    Ft_ctf, F_ty = relion_style_triangular_kernel(cryo , noise_variance.astype(np.float32),  batch_size,  disc_type = disc_type, return_lhs_rhs = False )

    estimate = post_process_from_filter(cryo, Ft_ctf, F_ty, tau = tau, disc_type = disc_type, use_spherical_mask = use_spherical_mask, grid_correct = grid_correct, gridding_correct = "square", kernel_width = 1 )
    cryo.update_volume_upsampling_factor(og_upsampling)

    return estimate, Ft_ctf