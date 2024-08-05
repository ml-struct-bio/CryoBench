import numpy as np
import jax.numpy as jnp
from recovar import mask as mask_fn
from recovar.fourier_transform_utils import fourier_transform_utils
import recovar
from recovar import utils, simulator
ftu = fourier_transform_utils(jnp)
import logging
# from tqdm.notebook import tqdm
import jax
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


## A copy of the relion local resolution function. See postprocessing.cpp in relion

def integral_fsc(fsc, fourier_pixel_size = 1):
    last_idx = find_first_zero_in_bool(fsc>=0)
    good_idx = jnp.where(jnp.arange(fsc.size) <= last_idx, 1, 0)
    return np.sum(fsc * good_idx) * fourier_pixel_size

integral_fscs = jax.vmap(integral_fsc, in_axes = [0, None])


def local_resolution(map1, map2, B_factor, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = True, filter_edgewidth=2):

    # if use_filter:
    #     use_v2 = False

    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size



    step_size = np.round(locres_sampling / angpix).astype(int)
    maskrad_pix = np.round(locres_maskrad / angpix).astype(int)

    if maskrad_pix < 5:
        logger.warning(f"radius of local resolution mask is only {maskrad_pix} pixels. Result will probably be nonsense. Should either increase locres_maskrad or do global resolution estimate")



    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)
    logger.info(f"Step size: {step_size}, maskrad_pix: {maskrad_pix}, edgewidth_pix: {edgewidth_pix}")
    myrad = map1.shape[0]//2 - 1*maskrad_pix
    # print('CHANGE THIS BACK!!')
    # myrad = 40
    myradf = myrad / step_size
    sampling_points = []

    logger.info(f"Starting...")

    grid = np.array(ftu.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)

    fourier_pixel_size = 1/(map1.shape[0] * voxel_size)
    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    logger.info(f"Calculating local resolution in {nr_samplings} sampling points ...")

    ft_sum = 0.5*(ftu.get_dft3(map1) + ftu.get_dft3(map2))
    # Need to apply B-factor here I guess
    ft_sum *= simulator.get_B_factor_scaling(map1.shape, voxel_size, -B_factor).reshape(map1.shape).astype(map1.dtype)

    i_ft_sum_orig = ftu.get_idft3(ft_sum).real


    # for now will do batch of 1.
    i_fil = jnp.zeros_like(map1)
    i_loc_res = 0
    i_loc_auc = 0
    i_sum_w = 0 
    local_resols, fscs = [], []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)
    single_batch = False
    # if single_batch:
    #     for k in range(nr_samplings):
    #         batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, 1, k)
    #         batch = sampling_points[batch_st:batch_end]

    #         if use_v2:
    #             if use_filter:
    #                 ift_sum, loc_mask, fsc, local_resol = compute_local_fsc_v2(batch[0], ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter, filter_edgewidth = filter_edgewidth)

    #                 i_fil += ift_sum * loc_mask
    #                 i_loc_res += loc_mask / local_resol
    #                 i_sum_w += loc_mask
    #             else:
    #                 fsc, local_resol = compute_local_fsc_v2(batch[0], ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter, filter_edgewidth = filter_edgewidth)

    #         else:
    #             if use_filter:
    #                 ift_sum, loc_mask, fsc, local_resol = compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter, filter_edgewidth = filter_edgewidth)

    #                 i_fil += ift_sum * loc_mask
    #                 i_loc_res += loc_mask / local_resol
    #                 i_sum_w += loc_mask
    #             else:
    #                 fsc, local_resol = compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = fsc_threshold, use_filter = use_filter, filter_edgewidth = filter_edgewidth)



    #         fscs.append(fsc)
    #         local_resols.append(local_resol[None])
    #         if k % 100 == 0:
    #             logger.info(f"Sampling point {k} out of {nr_samplings} done")

    #         if jnp.isnan(i_fil).any() or jnp.isnan(i_loc_res).any():
    #             import pdb; pdb.set_trace()
    #         # print(k)
    if True:
        # a =1
        # Doesn't seem to be faster.
        vol_batch_size = recovar.utils.get_vol_batch_size(map1.shape[0], recovar.utils.get_gpu_memory_total())/2
        if use_v2:
            vol_batch_size *=4
        # print(vol_batch_size)
        n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)
        # print(n_batch)

        for k in range(n_batch):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
            batch = sampling_points[batch_st:batch_end]

            if use_v2:
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol, offset, radius = batch_compute_local_fsc_v2(batch, i_ft_sum_orig, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold, use_filter, filter_edgewidth )
                    # i_fil2 = np.zeros_like(i_fil)

                    # for k in range(batch.shape[0]):
                    #     i_fil = add_subarray_to_array(i_fil, ift_sum[k] * loc_mask[k], offset[k], int(radius[k]))
                    #     i_fil2 = add_subarray_to_array_2(i_fil2, ift_sum[k] * loc_mask[k], offset[k], int(radius[k]))
                    # print(np.linalg.norm(i_fil- i_fil2)/ np.linalg.norm(i_fil))
                    # i_fil2 = np.zeros_like(i_fil)
                    i_fil = add_subarrays_to_array(i_fil, ift_sum * loc_mask, offset, int(radius[0]))
                    # print(np.linalg.norm(i_fil- i_fil2)/ np.linalg.norm(i_fil))

                    # import pdb; pdb.set_trace()        
                    #     print(np.linalg.norm(i_fil))
                    # import pdb; pdb.set_trace()
                        # i_loc_res = add_subarray_to_array(loc_mask / local_resol, ift_sum[0] * loc_mask[0], offset[0], radius[0])
                    # i_fil += ift_sum * loc_mask
                    # i_loc_res += loc_mask / local_resol
                    # i_sum_w += loc_mask
                else:
                    fsc, local_resol = batch_compute_local_fsc_v2(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold , use_filter, filter_edgewidth)

            else:
                    
                if use_filter:
                    ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter, filter_edgewidth)

                    i_fil += jnp.sum(ift_sum * loc_mask, axis=0)

                    # i_loc_auc += jnp.sum(loc_mask * integral_fscs(fsc, fourier_pixel_size)[:,None,None,None], axis=0)
                    # i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)
                    # i_sum_w += jnp.sum(loc_mask, axis=0)
                else:
                    fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_threshold,  use_filter, filter_edgewidth)

                # i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)
                # i_sum_w += jnp.sum(loc_mask, axis=0)

            # ift_sum, loc_mask, fsc, local_resol = batch_compute_local_fsc(batch, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, 1/8,  True)

            # i_fil += jnp.sum(ift_sum * loc_mask, axis=0)
            # i_loc_res += jnp.sum(loc_mask / local_resol[:,None,None,None], axis=0)
            # i_sum_w += jnp.sum(loc_mask, axis=0)
            fscs.append(fsc)
            local_resols.append(local_resol)
            if k % 100 == 0:
                logger.info(f"Sampling point batch {k} out of {n_batch} done")

            if jnp.isnan(i_fil).any() or jnp.isnan(i_loc_res).any():
                import pdb; pdb.set_trace()

    fscs = np.concatenate(fscs)
    local_resols = np.concatenate(local_resols)
    full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
    int_fscs = integral_fscs(fscs,fourier_pixel_size)
    i_loc_res = make_local_resol_map(sampling_points, 1/local_resols, full_mask)
    i_loc_auc = make_auc_map(sampling_points, int_fscs, full_mask)

    if not use_filter:
        # full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
        # i_loc_res = make_local_resol_map(sampling_points, 1/local_resols, full_mask)
        # int_fscs = integral_fscs(fscs,fourier_pixel_size)
        # i_loc_auc = make_auc_map(sampling_points, int_fscs, full_mask)
        # import pdb; pdb.set_trace()
        return fscs, local_resols, i_loc_res, i_loc_auc
    

    # i_fil3 = jnp.where( i_sum_w > 0,  i_fil / i_sum_w, 0)
    i_fil = make_i_fil_map(sampling_points, i_fil, full_mask)
    logger.info(f"Done")

    # mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones(sampling_points.shape[0]), full_mask)

    # print(np.linalg.norm(i_fil2 - i_fil3) / np.linalg.norm(i_fil3))
    # plt.imshow(mask_conv[64]); plt.colorbar(); plt.show()
    # plt.imshow(i_sum_w[64] - mask_conv[64]); plt.colorbar(); plt.show()
    # plt.imshow(np.abs(i_sum_w[64] - mask_conv[64]) / np.abs(mask_conv[64])); plt.colorbar(); plt.show()

    # plt.imshow(np.abs(i_fil2[64] - i_fil3[64]) / np.abs(i_fil3[64])); plt.colorbar(); plt.show()
    # i_fil = jnp.where( i_sum_w > 0,  i_fil / i_sum_w, 0)

    # print(np.linalg.norm(mask_conv - i_sum_w) / np.linalg.norm(mask_conv))
    # plt.imshow(i_fil[64]); plt.colorbar(); plt.show()
    # plt.imshow(i_fil2[64]); plt.colorbar(); plt.show()

    # i_loc_res_og = i_loc_res.copy()
    # Does this make sense?
    # i_loc_res = jnp.where( i_sum_w > 0,  1/ (i_loc_res/ i_sum_w) , 0)
    # i_loc_res = make_local_resol_map(sampling_points, 1/local_resols, full_mask)
    # # i_loc_auc = jnp.where( i_sum_w > 0,  i_loc_auc / i_sum_w, 0)
    # i_loc_au = make_auc_map(sampling_points, int_fscs, full_mask)

    # plt.imshow(i_loc_auc[64]); plt.colorbar(); plt.show()
    # plt.imshow(i_loc_auc2[64]); plt.colorbar(); plt.show()
    # plt.imshow((i_loc_auc-i_loc_auc2)[64]); plt.colorbar(); plt.show()

    # print(np.linalg.norm(i_loc_res - i_loc_res_2) / np.linalg.norm(i_loc_res))
    # print(np.linalg.norm(i_loc_auc - i_loc_auc2) / np.linalg.norm(i_loc_auc))
    # import pdb; pdb.set_trace()

    return i_fil, i_loc_res, i_loc_auc, fscs, local_resols#, sampling_points

import jax.scipy
def convolve_mask_at_sampling_points(sampling_pts, local_resols, full_mask):
    # full_array = jnp.zeros(full_mask.shape)
    # full_array = full_array.at[sampling_points].set(local_resol)

    full_array = jnp.zeros(full_mask.shape, dtype = full_mask.dtype)
    sampling_points = sampling_pts + full_mask.shape[0]//2
    
    sampling_idx = jax.numpy.ravel_multi_index(sampling_points.T, full_mask.shape)

    full_array = full_array.reshape(-1)
    full_array = full_array.at[sampling_idx].set(local_resols)
    full_array = full_array.reshape(full_mask.shape)

    return jax.scipy.signal.fftconvolve(full_mask, full_array, mode = 'same')

def make_local_resol_map(sampling_points, inv_local_resol, full_mask):

    local_resol_conv = convolve_mask_at_sampling_points(sampling_points, inv_local_resol, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(inv_local_resol), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-4,  mask_conv/ local_resol_conv, 0)
    # import pdb; pdb.set_trace()
    return i_loc_res


def make_auc_map(sampling_points, auc, full_mask):

    auc_conv = convolve_mask_at_sampling_points(sampling_points, auc, full_mask)
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones_like(auc), full_mask)

    i_loc_res = jnp.where( mask_conv > 1e-4,  auc_conv/ mask_conv , 0)
    return i_loc_res

def make_i_fil_map(sampling_points, i_fil, full_mask):
    mask_conv = convolve_mask_at_sampling_points(sampling_points, jnp.ones(sampling_points.shape[0]), full_mask.astype(np.float64))
    i_fil_divided = jnp.where( mask_conv > 1e-4,  i_fil / mask_conv, 0)
    return i_fil_divided




def get_subsample_indices(array_shape, offset, radius):
    size = 2*radius
    grid = jnp.mgrid[:size,:size,:size]
    grid = grid.reshape(3, -1).T
    grid += offset - radius
    vec_indices = jnp.ravel_multi_index(grid.T, array_shape, mode = 'clip')#.reshape(og_shape[:-1])

    good_idx =  (grid >= 0).all(axis = 1) * (grid < array_shape[0]).all(axis = 1) 

    return vec_indices, good_idx

batch_get_subsample_indices = jax.vmap(get_subsample_indices, in_axes = (None, 0, None))

def subsample_array(array, offset, radius):
    # l_bounds = offset - radius
    # u_bounds = offset + radius
    size = 2*radius
    # grid = jnp.mgrid[:size,:size,:size]
    # grid = grid.reshape(3, -1).T
    # grid += offset - radius
    # vec_indices = jnp.ravel_multi_index(grid.T, array.shape, mode = 'clip')#.reshape(og_shape[:-1])
    vec_indices, good_idx = get_subsample_indices(array.shape, offset, radius)
    return (array.ravel()[vec_indices] * good_idx).reshape(size, size, size)

# def subsample_array(array, offset, radius):
#     l_bounds = offset - radius
#     # u_bounds = offset + radius
#     size = (2 * radius, 2 * radius, 2 * radius)
#     subarray  = jax.lax.dynamic_slice(array, l_bounds, size)
#     return subarray


import functools
@functools.partial(jax.jit, static_argnums = [3])
def add_subarray_to_array_2(array, subarray, offset, radius):
    array_shape = array.shape
    vec_indices, good_idx = get_subsample_indices(array.shape, offset, radius)
    array = array.ravel()
    array = array.at[vec_indices].add(subarray.ravel() * good_idx)
    return array.reshape(array_shape)


import functools
@functools.partial(jax.jit, static_argnums = [3])
def add_subarrays_to_array(array, subarrays, offset, radius):
    array_shape = array.shape
    vec_indices, good_idx = batch_get_subsample_indices(array.shape, offset, radius)
    array = array.ravel()
    array = array.at[vec_indices.reshape(-1)].add(subarrays.reshape(-1) * good_idx.reshape(-1))
    return array.reshape(array_shape)


@functools.partial(jax.jit, static_argnums = [3])
def add_subarray_to_array(array, subarray, offset, radius):
    l_bounds = offset - radius
    u_bounds = offset + radius
    # subarray jax.lax.dynamic_slice(array, l_bounds, [2 * radius, 2 * radius, 2 * radius])
    # array = array.at[l_bounds[0]:u_bounds[0], l_bounds[1]:u_bounds[1], l_bounds[2]:u_bounds[2]].set(subarray)
    # jax.lax.dynamic_slice(array, subarray, l_bounds)

    return jax.lax.dynamic_update_slice(array, subarray + subsample_array(array, offset, radius), l_bounds)



# def get_

import functools
@functools.partial(jax.jit, static_argnums = [4,5,9,10])    
def compute_local_fsc_v2(offset, ift_sum_orig, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter, filter_edgewidth):

    offset = offset + map1.shape[0]//2
    # Compute masked fsc

    radius = maskrad_pix + edgewidth_pix
    # l_bounds = offset - radius
    # u_bounds = offset + radius
    multiplier = 3
    # smaller_size = multiplier*radius#
    map1_sub = subsample_array(map1, offset, multiplier*radius)
    # print(map1_sub.shape)
    map2_sub = subsample_array(map2, offset, multiplier*radius)
    ift_sum_sub = subsample_array(ift_sum_orig, offset, multiplier*radius)
    mask = mask_fn.raised_cosine_mask(map1_sub.shape, maskrad_pix, radius, 0)
    map1_sub = map1_sub * mask
    map2_sub = map2_sub * mask

    fsc = recovar.regularization.get_fsc(ftu.get_dft3(map1_sub), ftu.get_dft3(map2_sub), volume_shape = map1_sub.shape)


    # local_resol = jnp.argmin(fsc >= fsc_treshold)
    # # If all above threhsold
    # local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)
    local_resol = find_fsc_resol(fsc, fsc_treshold)
    local_resol = jnp.where(local_resol > 0, map1_sub.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)


    if use_filter:
        ft_sum_sub = ftu.get_dft3(ift_sum_sub)
        ift_sum = filter_with_local_fsc(ft_sum_sub, fsc, local_resol, voxel_size, filter_edgewidth)
        # if jnp.isnan(ift_sum).any():
        #     import pdb; pdb.set_trace()
        return ift_sum, mask, fsc, local_resol, offset, multiplier*radius

    return fsc, local_resol

# def filter_by_fsc():




import functools
# def compute_local_fsc(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold = 1/7, use_filter = True):
@functools.partial(jax.jit, static_argnums = [9,10])    
def compute_local_fsc(offset, ft_sum, map1, map2, maskrad_pix, edgewidth_pix, locres_minres, voxel_size, fsc_treshold, use_filter, filter_edgewidth):

    # Compute masked fsc

    mask = mask_fn.raised_cosine_mask(ft_sum.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    map1 = map1 * mask
    map2 = map2 * mask
    fsc = recovar.regularization.get_fsc(ftu.get_dft3(map1), ftu.get_dft3(map2), volume_shape = map1.shape)

    # first fsc above threshold

    local_resol = jnp.argmin(fsc >= fsc_treshold)
    # If all above threhsold
    local_resol = jnp.where(fsc[local_resol] >= fsc_treshold, fsc.size-1 , local_resol)
    local_resol = jnp.where(local_resol > 0, map1.shape[0] * voxel_size / local_resol, 999)
    local_resol = jnp.where(local_resol < locres_minres, local_resol, locres_minres)

    # import matplotlib.pyplot as plt
    # radius = maskrad_pix + edgewidth_pix

    # map1_sub = subsample_array(mask, offset[0] + map1.shape[0]//2, radius)

    # offset = offset[0]
    # l_bounds = offset - radius + map1.shape[0]//2
    # u_bounds = offset + radius + map1.shape[0]//2
    # mask2 = mask[l_bounds[0]:u_bounds[0], l_bounds[1]:u_bounds[1], l_bounds[2]:u_bounds[2]] 
    # plt.figure()
    # plt.imshow(map1_sub.sum(axis=0))
    # plt.show()
    # plt.figure()
    # plt.imshow(mask.sum(axis=0))
    # plt.show()
    # plt.figure()
    # plt.plot(fsc)
    # plt.show()
    # import pdb; pdb.set_trace()

    if use_filter:
        ift_sum = filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, filter_edgewidth)
        # if jnp.isnan(ift_sum).any():
        #     import pdb; pdb.set_trace()
        return ift_sum, mask, fsc, local_resol

    return fsc, local_resol

batch_compute_local_fsc = jax.vmap(compute_local_fsc, in_axes = (0, None, None, None, None, None, None, None, None, None, None ) )

batch_compute_local_fsc_v2 = jax.vmap(compute_local_fsc_v2, in_axes = (0, None, None, None, None, None, None, None, None, None, None ) )


def find_first_zero_in_bool(array):
    ires_max = jnp.argmin(array)
    # There is a corner case. If iresmax is 0 because everything is 1.
    # If ires_max is 0, there are two options: either the array is all 0 or all 1.
    all_ones_flag = (ires_max ==0 ) * (array[ires_max] == True)
    ires_max = jnp.where(all_ones_flag, array.size-1, ires_max)

    return ires_max#, all_ones_flag


def find_fsc_resol(fsc_curve, threshold = 1/7):
    ires_zero = find_first_zero_in_bool(fsc_curve >= threshold)
    ires_before = jnp.where(ires_zero > 0, ires_zero - 1, 0)
    fsc_subcurve = jnp.array([fsc_curve[ires_zero], fsc_curve[ires_before]])
    ires_interp = ires_before + jnp.interp(threshold, fsc_subcurve, jnp.array([1,0]) * 1.0 )
    # zz = jnp.interp(threshold, fsc_subcurve, jnp.arange(2) * 1.0 )

    ires_interp = jnp.where(ires_zero > 0, ires_interp, 0)
    ires_interp = jnp.where((ires_zero == fsc_curve.size-1) * fsc_curve[-1] >= threshold, fsc_curve.size-1, ires_interp)

    # # ires_interp = jnp.interp(threshold, fsc_subcurve, jnp.array([1,0]) * 1.0 )

    # plt.plot(fsc_subcurve, [1,0]); plt.plot(1/7, ires_interp, 'o'); plt.show()
    # plt.show()

    # plt.plot(fsc_curve); plt.plot(ires_interp, 1/7, 'o'); plt.show()

    # # import matplotlib.pyplot as plt

    # ires_interp = 
    # jnp.where(ires_zero > 0 )
    # if ires_zero == fsc_curve.size - 1:
    #     return 999
    # from scipy import interpolate
    # fsc_curve = np.where(np.isnan(fsc_curve), 0, fsc_curve)
    # f = interpolate.interp1d( np.array([fsc_curve[idx], fsc_curve[idx+1]]), np.array([freq[idx], freq[idx+1]]) )
    return ires_interp


def apply_fsc_weighting(FT, fsc):
	#  Find resolution where fsc_true drops below zero for the first time
	#  Set all weights to zero beyond that resolution
    distances = ftu.get_grid_of_radial_distances(FT.shape,)

    # ires_max = jnp.argmin(fsc >= 0.0001)
    ires_max = find_first_zero_in_bool(fsc >= 0.0001)
    # ires_max = jnp.where(all_ones_flag, fsc.size, ires_max) # If == 0, fsc >= 0.0001 for all ires

    fsc = jnp.where( jnp.arange(fsc.size) < ires_max, fsc, 0)
    # fsc = fsc.at[ires_max:].set(0)
    fsc = jnp.sqrt((2 * fsc) / (1 + fsc))
    fsc_mask = fsc[distances]
    FT = FT * fsc_mask
    return FT

def filter_with_local_fsc(ft_sum, fsc, local_resol, voxel_size, filter_edgewidth):
    # ft_sum_inp = ft_sum.copy()
    ft_sum  = apply_fsc_weighting(ft_sum, fsc) # 
    ft_sum = low_pass_filter_map(ft_sum, ft_sum.shape[0], local_resol, voxel_size, filter_edgewidth)#.astype(ft_sum.dtype)
    ift_sum = ftu.get_idft3(ft_sum).real
    # import pdb; pdb.set_trace()
    return  ift_sum

# def low_pass_filter_map(FT, ori_size, low_pass, angpix, filter_edgewidth=2, do_highpass_instead = False):
#     ires_filter = jnp.round((ori_size * angpix) / low_pass)
#     filter_edge_halfwidth = filter_edgewidth // 2

#     edge_low = jnp.maximum(0., (ires_filter - filter_edge_halfwidth) / ori_size)
#     edge_high = jnp.minimum(FT.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
#     edge_width = edge_high - edge_low
#     res = ftu.get_grid_of_radial_distances(FT.shape) / ori_size

#     if do_highpass_instead:
#         filter = jnp.where(res <  edge_low , 0, 1)
#         filter = jnp.where(res >= edge_low * (res < edge_high), 0.5 - 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
#     else:
#         filter = jnp.where(res <  edge_low , 1, 0)
#         filter = jnp.where(res >= edge_low * (res < edge_high), 0.5 + 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
#     filter = filter.astype(FT.dtype)
#     return FT * filter

def low_pass_filter_map(FT, ori_size, low_pass, angpix, filter_edgewidth, do_highpass_instead = False):
    ires_filter = jnp.round((ori_size * angpix) / low_pass)
    filter_edge_halfwidth = filter_edgewidth // 2

    edge_low = jnp.maximum(0., (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = jnp.minimum(FT.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low
    res = ftu.get_grid_of_radial_distances(FT.shape) / ori_size
    if do_highpass_instead:
        filter = jnp.where(res <  edge_low , 0, 1)
        filter = jnp.where((res >= edge_low) * (res < edge_high), 0.5 - 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    else:
        filter = jnp.where(res <=  edge_low , 1, 0)
        filter = jnp.where( (res >= edge_low) * (res < edge_high), 0.5 + 0.5 * jnp.cos(jnp.pi * (res - edge_low) / edge_width), filter)
    filter = filter.astype(utils.dtype_to_real(FT.dtype))
    
    return FT * filter


def local_error(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size

    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)

    # mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)

    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)
    mask /= np.linalg.norm(mask)
    # Compute error with convolution
    
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map1 = ftu.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map2 = ftu.get_idft3(map2_ft).real

    mask_ft = ftu.get_dft3(mask)
    map1_square_ft = ftu.get_dft3(map1*map1)
    map2_square_ft = ftu.get_dft3(map2*map2)
    map1map2_ft = ftu.get_dft3(map1*map2)
    # map2_ft = ftu.get_dft3(map2)

    local_errors = (ftu.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * ftu.get_idft3(map1map2_ft * mask_ft) \
    + (ftu.get_idft3(map2_square_ft * mask_ft) )
    
    # local_errors = (ftu.get_idft3(map1_ft * mask_ft).real )**2 \
    # - 2 * ftu.get_idft3(map1_ft * map2_ft * mask_ft) \
    # + (ftu.get_idft3(map2_ft * mask_ft) )**2
    
    local_errors = local_errors.real

    return local_errors


def local_error_with_cov(map1, map2, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, low_pass_filter_res = None, noise_variance = None):
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size

    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)


    # TODO FIX this +/- 1 business somewhere once and for all...
    # mask = mask_fn.create_hard_edged_kernel_pxl(locres_maskrad, map1.shape)
    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)


    # Compute error with convolution
    if low_pass_filter_res is not None:
        map1_ft = low_pass_filter_map(map1_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map1 = ftu.get_idft3(map1_ft).real

        map2_ft = low_pass_filter_map(map2_ft, map1_ft.shape[0], low_pass_filter_res, angpix, edgewidth_pix, do_highpass_instead = False)
        map2 = ftu.get_idft3(map2_ft).real

    ## Whiten maps
    if noise_variance is not None:
        noise_variance = noise_variance.reshape(map1.shape)
        map1 = ftu.get_idft3(ftu.get_dft3(map1) * jnp.sqrt(noise_variance).reshape(map1.shape)).real
        map2 = ftu.get_idft3(ftu.get_dft3(map2) * jnp.sqrt(noise_variance).reshape(map1.shape)).real

    mask_ft = ftu.get_dft3(mask)
    map1_square_ft = ftu.get_dft3(map1*map1)
    map2_square_ft = ftu.get_dft3(map2*map2)
    map1map2_ft = ftu.get_dft3(map1*map2)
    # map2_ft = ftu.get_dft3(map2)

    local_errors = (ftu.get_idft3(map1_square_ft * mask_ft).real ) \
    - 2 * ftu.get_idft3(map1map2_ft * mask_ft) \
    + (ftu.get_idft3(map2_square_ft * mask_ft) )
        
    local_errors = local_errors.real

    return local_errors




def expensive_local_error_with_cov(map1, map2, voxel_size, noise_variance, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, use_v2 = False, debug = False, split_shell = False):

    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth


    angpix = voxel_size


    step_size = np.round(locres_sampling / angpix).astype(int)
    maskrad_pix = np.round(locres_maskrad / angpix).astype(int)
    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)
    logger.info(f"Step size: {step_size}, maskrad_pix: {maskrad_pix}, edgewidth_pix: {edgewidth_pix}")
    myrad = map1.shape[0]//2 - maskrad_pix
    # myrad = 40
    myradf = myrad / step_size
    sampling_points = []

    # logger.info(f"Starting...")

    grid = np.array(ftu.get_1d_frequency_grid(map1.shape[0], 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)

    # sampling_points = jnp.array(sampling_points).astype(int)[:1]


    nr_samplings = sampling_points.shape[0]

    logger.info(f"Calculating local error in {nr_samplings} sampling points ...")
    

    # for now will do batch of 1.
    diffs = []
    # Put stuff on GPU
    map1 = jnp.asarray(map1)
    map2 = jnp.asarray(map2)
    noise_variance = jnp.asarray(noise_variance)

    # if np.log2(map1.shape[0]) % 1 != 0:
    #     raise ValueError("Map size must be a power of 2")
    
    # sqrt_noise_variance_real = ftu.get_idft3(jnp.sqrt(noise_variance))
    # radius = maskrad_pix + edgewidth_pix
    # multiplier = 2
    # want_size = 2*(multiplier*radius)
    # actual_size = int(2**(np.ceil(np.log2(want_size))))
    # factor = map1.shape[0]//actual_size 
    # from skimage.transform import downscale_local_mean
    
    if use_v2:
        multiplier = 3
        rad = maskrad_pix * multiplier
        # downsampled_noise_variance_ift = ftu.get_idft3(jnp.sqrt(noise_variance) )
        downsampled_noise_variance_ift = ftu.get_idft3((noise_variance) )#.real

        downsampled_noise_variance_ift_subs = subsample_array(downsampled_noise_variance_ift, map1.shape[0]//2+1, rad)
        noise_variance_small = ftu.get_dft3(downsampled_noise_variance_ift_subs ) * noise_variance.size / downsampled_noise_variance_ift_subs.size


        # downsampled_noise_variance_ift_subs = subsample_array(downsampled_noise_variance_ift, map1.shape[0]//2+0, rad)
        # noise_variance_small2 = ftu.get_dft3(downsampled_noise_variance_ift_subs) * noise_variance.size / downsampled_noise_variance_ift_subs.size#, norm = "ortho") 
        # print(np.linalg.norm(noise_variance_small - noise_variance_small2) / np.linalg.norm(noise_variance_small))
        # import pdb; pdb.set_trace()


    # map1_sub = subsample_array(diff, offset, multiplier*radius)
    # use_v2 = True
    diff_map = jnp.array(map1- map2)
    noise_variance = jnp.array(noise_variance)


    single_batch = not use_v2
    if single_batch:
        # AssertionError("Not implemented")
        for k in range(nr_samplings):
            batch = sampling_points[k]
            if use_v2:
                # By noting that diagonal blocks of MCM are toeplitz, we can compute this fast
                diff = masked_noisy_error_3(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
            else:
                diff = masked_noisy_error(diff_map, noise_variance, batch, maskrad_pix, edgewidth_pix )
            diffs.append(diff[None])
            # i_loc_res += loc_mask * diff
            # i_sum_w += loc_mask
            if k % 1000 ==0:
                print(k, end = '  ')

            # print(np.linalg.norm(diff - diff2))
            # import pdb; pdb.set_trace()


    else:
        # Doesn't seem to be faster.
        if use_v2:
            vol_batch_size = recovar.utils.get_vol_batch_size(noise_variance_small.shape[0], recovar.utils.get_gpu_memory_total()) * 1
        else:
            vol_batch_size = recovar.utils.get_vol_batch_size(noise_variance.shape[0], recovar.utils.get_gpu_memory_total()) / 4

        n_batch = utils.get_number_of_index_batch(sampling_points.shape[0], vol_batch_size)

        for k in range(n_batch):
            batch_st, batch_end = utils.get_batch_of_indices(nr_samplings, vol_batch_size, k)
            batch = sampling_points[batch_st:batch_end]

            if split_shell:
                diff = batch_masked_noisy_error_split_over_shells(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
            else:
                if use_v2:
                    # By noting that diagonal blocks of MCM are toeplitz, we can compute this fast
                    diff = batch_masked_noisy_error_3(diff_map, noise_variance_small, batch, maskrad_pix, edgewidth_pix )
                else:
                    diff = batch_masked_noisy_error(diff_map, noise_variance, batch, maskrad_pix, edgewidth_pix )



            diffs.append(diff)
            # if k % 10 ==0:
                # print(k, end = '  ')


    diffs = np.concatenate(diffs)
    if split_shell:
        return diffs

    full_mask = mask_fn.raised_cosine_mask(map1.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)
    i_local_error = make_auc_map(sampling_points, diffs, full_mask)

    # i_fil = jnp.where( i_sum_w > 0,  i_loc_res / i_sum_w, 0)
    # return diffs
    if debug:
        return i_local_error, diffs
    
    return i_local_error





@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error(diff, noise_variance, offset, maskrad_pix, edgewidth_pix ):
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    diff = ftu.get_dft3((diff) * mask) * jnp.sqrt(noise_variance)
    return jnp.linalg.norm(diff)**2


@functools.partial(jax.jit, static_argnums = [3,4])    
def masked_noisy_error_3(diff, noise_variance_small, offset, maskrad_pix, edgewidth_pix ):

    ### THIS ALMOST WORKS BUT NOT QUITE. SOME SCALING FACTOR IN HERE
    # radius = maskrad_pix + edgewidth_pix
    # # l_bounds = offset - radius
    # # u_bounds = offset + radius
    # multiplier = 3
    # smaller_size = multiplier*radius#

    multiplier = 3
    offset += diff.shape[0]//2
    # import pdb; pdb.set_trace()

    diff = subsample_array(diff, offset, multiplier*maskrad_pix)
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, edgewidth_pix, 0)
    diff_masked = diff * mask
    # import pdb; pdb.set_trace()

    diff_masked = ftu.get_dft3(diff_masked) * jnp.sqrt(noise_variance_small)

    # import pdb; pdb.set_trace()
    # mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, maskrad_pix + edgewidth_pix, offset)
    # diff = ftu.get_dft3((diff) * mask) * jnp.sqrt(noise_variance)
    return jnp.linalg.norm(diff_masked)**2
from recovar import regularization


@functools.partial(jax.jit, static_argnums = [3,4,5])    
def masked_noisy_error_split_over_shells(diff, noise_variance_small, offset, maskrad_pix, edgewidth_pix,multiplier =3 ):
    # NOT IMPLEMENTED
    offset += diff.shape[0]//2

    diff = subsample_array(diff, offset, multiplier*maskrad_pix)
    mask = mask_fn.raised_cosine_mask(diff.shape, maskrad_pix, edgewidth_pix, 0)
    diff_masked = diff * mask
    diff_masked = ftu.get_dft3(diff_masked) * jnp.sqrt(noise_variance_small)

    diff_masked_norm = regularization.sum_over_shells(jnp.abs(diff_masked)**2, diff_masked.shape)

    return diff_masked_norm

batch_subsample_array_at_same_offset = jax.vmap(subsample_array, in_axes = (0, None, None) )

@functools.partial(jax.jit, static_argnums = [3,4,5])    
def recombine_with_choice(estimators, choice, offset, maskrad_pix, edgewidth_pix, multiplier =3):
    # NOT IMPLEMENTED
    # multiplier = 3
    offset += estimators.shape[1]//2
    estimators_subsampled = batch_subsample_array_at_same_offset(estimators, offset, multiplier*maskrad_pix)

    mask = mask_fn.raised_cosine_mask(estimators_subsampled.shape[1:], maskrad_pix, edgewidth_pix, 0)
    estimators_subsampled = estimators_subsampled * mask[None]
    estimators_subsampled_ft = ftu.get_dft3(estimators_subsampled)

    choice_radial = utils.make_radial_image(choice, estimators_subsampled.shape[1:]).reshape(estimators_subsampled.shape[1:])
    # combined_est = jnp.take(estimators_subsampled, choice_radial, axis = 0)
    combined_est = jnp.take_along_axis(estimators_subsampled_ft , choice_radial[None], axis=0)[0]
    combined_est = ftu.get_idft3(combined_est).real

    return combined_est


# @functools.partial(jax.jit, static_argnums = [3,4])    
# def masked_noisy_error2(diff, sqrt_noise_variance_real, offset, maskrad_pix, edgewidth_pix ):

#     offset = offset + diff.shape[0]//2
#     radius = maskrad_pix + edgewidth_pix
#     multiplier = 2
#     map1_sub = subsample_array(diff, offset, multiplier*radius)


#     noise_variance_real = subsample_array(sqrt_noise_variance_real, offset, multiplier*radius)
#     mask = mask_fn.raised_cosine_mask(map1_sub.shape, maskrad_pix, radius, 0)
#     result = jnp.linalg.norm(map1_sub * mask * noise_variance_real )**2
#     import pdb; pdb.set_trace()
#     return result

# @functools.partial(jax.jit, static_argnums = [3,4])    
# def masked_noisy_error2(diff, downsampled_noise_variance, offset, maskrad_pix, edgewidth_pix ):

#     offset = offset + diff.shape[0]//2
#     radius = maskrad_pix + edgewidth_pix
#     # multiplier = 2
#     diff_sub = subsample_array(diff, offset, downsampled_noise_variance.shape[0]//2)
#     # noise_variance_real = subsample_array(sqrt_noise_variance_real, offset, multiplier*radius)
#     mask = mask_fn.raised_cosine_mask(diff_sub.shape, maskrad_pix, radius, 0)
#     result = jnp.linalg.norm( jnp.sqrt(downsampled_noise_variance) * ftu.get_dft3(diff_sub * mask) )**2
#     # import pdb; pdb.set_trace()
#     return result


batch_masked_noisy_error = jax.vmap(masked_noisy_error, in_axes = (None,None,0, None, None) )
batch_masked_noisy_error_3 = jax.vmap(masked_noisy_error_3, in_axes = (None,None,0, None, None) )
batch_masked_noisy_error_split_over_shells = jax.vmap(masked_noisy_error_split_over_shells, in_axes = (None,None,0, None, None) )


def recombine_estimates(estimators, choice, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None):
    assert locres_edgwidth ==0
    locres_maskrad= 0.5 *locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    angpix = voxel_size

    step_size = np.round(locres_sampling / angpix).astype(int)
    maskrad_pix = np.round(locres_maskrad / angpix).astype(int)

    if maskrad_pix < 5:
        logger.warning(f"radius of local resolution mask is only {maskrad_pix} pixels. Result will probably be nonsense. Should either increase locres_maskrad or do global resolution estimate")

    edgewidth_pix = np.round(locres_edgwidth / angpix).astype(int)
    logger.info(f"Step size: {step_size}, maskrad_pix: {maskrad_pix}, edgewidth_pix: {edgewidth_pix}")
    myrad = estimators.shape[1]//2 - 1*maskrad_pix
    # print('CHANGE THIS BACK!!')
    # myrad = 40
    myradf = myrad / step_size
    sampling_points = []

    logger.info(f"Starting...")

    # TODO: REally need to put this in a function
    grid = np.array(ftu.get_1d_frequency_grid(estimators.shape[1], 1, scaled = False)[::step_size])
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    sampling_points = jnp.array(sampling_points).astype(int)

    nr_samplings = sampling_points.shape[0]
    logger.info(f"Recombining estimates at {nr_samplings} sampling points ...")

    # for now will do batch of 1.
    # Put stuff on GPU
    estimators = jnp.asarray(estimators)
    optimized_estimator = jnp.zeros_like(estimators[0])
    n_batch = sampling_points.shape[0]
    multiplier = 3
    radius =  multiplier*maskrad_pix

    # with jax.disable_jit():
    for k in range(n_batch):
        offset = sampling_points[k]
        est_patch = recombine_with_choice(estimators, choice[k], offset, maskrad_pix, edgewidth_pix, multiplier )
        offset = offset + estimators.shape[1]//2
        optimized_estimator = add_subarrays_to_array(optimized_estimator, est_patch[None], offset[None], int(radius))

    full_mask = mask_fn.raised_cosine_mask(optimized_estimator.shape, maskrad_pix, maskrad_pix + edgewidth_pix, -1)    
    optimized_estimator = make_i_fil_map(sampling_points, optimized_estimator, full_mask)

    return optimized_estimator
