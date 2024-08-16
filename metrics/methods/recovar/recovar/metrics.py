import jax
import jax.numpy as jnp
import numpy as np
import pickle
from recovar import core, utils, simulator, linalg, mask, constants, locres
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)
import matplotlib.pyplot as plt
# Maybe should take out these dependencies?
from cryodrgn import mrc

def qr_on_cpu(Q):
    Q = jax.device_put(Q, device=jax.devices("cpu")[0])
    Q,R = jnp.linalg.qr(Q)
    Q = np.array(Q) # I don't know why but not doing this causes massive slowdowns sometimes?
    R = np.array(R)
    return Q, R

def captured_variance(test_v, U, s):
    # test_v, _ = qr_on_cpu(test_v)
    
    x = (jnp.conj(test_v.T) @ U) * np.sqrt(s)
    norms = np.linalg.norm(x, axis=-1)**2
    return np.cumsum(norms)

def relative_variance_from_captured_variance(variance, s):
    all_variance = np.sum(s)
    return (variance) / all_variance

def relative_variance(test_v, U, s):
    variance = captured_variance(test_v, U, s)
    return relative_variance_from_captured_variance(variance, s)

def normalized_variance(test_v, U, s):
    variance = captured_variance(test_v, U, s)
    return normalized_variance_from_captured_variance(variance, s)

def normalized_variance_from_captured_variance(variance, s):
    all_variance_up_to = np.cumsum(np.asarray(s))
    if variance.size > all_variance_up_to.size:
        all_variance_up_to_padded = np.ones(variance.size) * all_variance_up_to[-1]
        all_variance_up_to_padded[:all_variance_up_to.size] = all_variance_up_to
    else:
        all_variance_up_to_padded = all_variance_up_to
    return (variance) / all_variance_up_to_padded[:variance.size]

def get_all_variance_scores(test_v, U, s):
    
    variance = captured_variance(test_v, U, s)
    rel_variance = relative_variance_from_captured_variance(variance, s)
    normalized_variance = normalized_variance_from_captured_variance(variance, s)
    return variance, rel_variance, normalized_variance

def get_variance_error():

    return


def get_covariance_fsc_score():
    # Maybe summed auc across columns
    return


# def local_fsc_metric(map1, map2, voxel_size, mask, fsc_threshold=1/7, locres_sampling = 25 ):
    
    
#     fscs, local_resols, i_loc_res, i_loc_auc = locres.local_resolution(map1, map2, 0, voxel_size, locres_sampling = locres_sampling, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = False, use_v2 = True, fsc_threshold = fsc_threshold)
#     mask = mask > 1e-3
    
#     good_resols = i_loc_res[mask]
#     good_aucs = i_loc_auc[mask]
    
    
#     median_locres = np.median(good_resols)
#     ninety_pc_locres = np.percentile(good_aucs, 90)

#     median_auc = np.median(good_resols)
#     ninety_pc_auc = np.percentile(good_aucs, 90)

#     return median_locres, ninety_pc_locres, ninety_pc_auc



def local_fsc_metric(map1, map2, voxel_size, mask, fsc_threshold=1/7, locres_sampling = 25 ):
    
    fscs, local_resols, i_loc_res, i_loc_auc = locres.local_resolution(map1, map2, 0, voxel_size, locres_sampling = locres_sampling, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = False, use_v2 = True, fsc_threshold = fsc_threshold)
    
    good_resols = i_loc_res[mask]
    good_aucs = i_loc_auc[mask]
    
    i_loc_res = np.array(i_loc_res)
    i_loc_res[~mask] = None
    plt.imshow(i_loc_res[128]); plt.show()

    median_locres = np.median(good_resols)
    ninety_pc_locres = np.percentile(good_resols, 90)

    median_auc = np.median(good_aucs)
    ninety_pc_auc = np.percentile(good_aucs, 10)
    # import pdb; pdb.set_trace()
    return median_locres, ninety_pc_locres, median_auc, ninety_pc_auc


def local_error_metric(map1, map2, voxel_size, mask, normalize_by_map1 = True, locres_sampling = 15 ):
    
    errors_gt = locres.local_error(map1, map2, voxel_size, locres_sampling = locres_sampling)

    good_errors = errors_gt[mask]

    median_error = np.median(good_errors) 
    ninety_pc_error = np.percentile(good_errors, 90)

    if normalize_by_map1:
        errors_scale = locres.local_error(map1, map2 * 0, voxel_size, locres_sampling = locres_sampling)
        med_local_norm = np.median(errors_scale[mask])
        median_error = median_error/ med_local_norm
        ninety_pc_error = ninety_pc_error/ med_local_norm

    return median_error, ninety_pc_error


def masked_l2_difference(gt_map, target_map, voxel_size, mask= None ):
    
    if mask is None:
        mask = gt_mask_fn(gt_map)

    l2_error = np.mean(np.abs((gt_map-target_map)[mask])**2)
    bias = np.mean((gt_map-target_map)[mask])
    # variance =  np.var((map1-target)[mask])
    return l2_error, bias

def gt_mask_fn(gt_map):
    from recovar import mask as mask_fn
    mask = mask_fn.make_mask_from_gt(gt_map, smax = 3, iter = 1 , from_ft = False)
    mask = (mask > 0.5).astype(bool)

    return mask


def compute_volume_error_metrics_from_gt(gt_map, estimate_map, voxel_size, mask , partial_mask = None , normalize_by_map1 = True ):
    
    if mask is None:
        mask = gt_mask_fn(gt_map)
    
    errors_metrics = {}    
    errors_metrics['median_locres'], errors_metrics['ninety_pc_locres'], errors_metrics['median_auc'], errors_metrics['ten_pc_auc'] =  local_fsc_metric(estimate_map, gt_map, voxel_size, mask, fsc_threshold=1/2 )
    errors_metrics['median_error'], errors_metrics['ninety_pc_error'] =  local_error_metric(gt_map , estimate_map, voxel_size, mask, normalize_by_map1 = normalize_by_map1)
    errors_metrics['mask'] = mask

    if partial_mask is not None:
        errors_metrics['partial_median_locres'], errors_metrics['partial_ninety_pc_locres'], errors_metrics['partial_median_auc'], errors_metrics['partial_ten_pc_auc'] =  local_fsc_metric(estimate_map, gt_map, voxel_size, partial_mask, fsc_threshold=1/2 )
        errors_metrics['partial_median_error'], errors_metrics['partial_ninety_pc_error'] =  local_error_metric(gt_map, estimate_map, voxel_size, partial_mask,normalize_by_map1 = normalize_by_map1)
        errors_metrics['partial_mask'] = partial_mask

    return errors_metrics


def compute_volume_error_metrics_from_halfmaps(estimate1, estimate2, voxel_size, mask , partial_mask = None , normalize_by_map1 = False ):
    
    if mask is None:
        from recovar import mask as mask_fn
        mask = mask_fn.make_mask_from_half_maps(estimate1, estimate2)
        mask = (mask > 1e-3).astype(bool)
    
    errors_metrics = {}    
    errors_metrics['median_locres'], errors_metrics['ninety_pc_locres'], errors_metrics['median_auc'], errors_metrics['ten_pc_auc'] =  local_fsc_metric(estimate1, estimate2, voxel_size, mask, fsc_threshold=1/7 )
    errors_metrics['median_error'], errors_metrics['ninety_pc_error'] =  local_error_metric(estimate1, estimate2, voxel_size, mask, normalize_by_map1 = normalize_by_map1 )
    errors_metrics['mask'] = mask

    if partial_mask is not None:
        errors_metrics['partial_median_locres'], errors_metrics['partial_ninety_pc_locres'], errors_metrics['partial_median_auc'], errors_metrics['partial_ten_pc_auc'] =  local_fsc_metric(estimate1, estimate2, voxel_size, partial_mask, fsc_threshold=1/7 )
        errors_metrics['partial_median_error'], errors_metrics['partial_ninety_pc_error'] =  local_error_metric(estimate1, estimate2, voxel_size, partial_mask, normalize_by_map1 =normalize_by_map1)
        errors_metrics['partial_mask'] = partial_mask

    return errors_metrics

from recovar import metrics, locres
import recovar
import os.path

def evaluate_this_choice(target_real, output_folder, voxel_size, mask = None, partial_mask = None, shell_split = False ):
    
    mask = gt_mask_fn(target_real) if mask is None else mask

    file = lambda k : output_folder + "estimates_half2_unfil"+format(k, '03d')  +".mrc"
    k = 0 
    errors_gt= {}
    while os.path.isfile(file(k)):
        map2 = recovar.utils.load_mrc(file(k))
        errors_gt[k] = locres.local_error(target_real, map2, voxel_size, locres_sampling = 15)
        k = k + 1

    gt_choice = np.argmin(np.array(list(errors_gt.values())), axis=0)
    if shell_split:    
        error_metrics = {"gt_choice": gt_choice}
        error_metrics["choice_l2_error"] = None
        error_metrics["choice_l2_bias"] = None
        if partial_mask is not None:
            error_metrics["choice_partial_l2_error"] = None
            error_metrics["choice_partial_l2_bias"] = None
    else:
        choice1 = recovar.utils.load_mrc(output_folder + "ml_optimized_choice.mrc")
        error_metrics = {"gt_choice": gt_choice}
        error_metrics["choice_l2_error"], error_metrics["choice_l2_bias"]  = metrics.masked_l2_difference(gt_choice, choice1, voxel_size, mask= mask)
        if partial_mask is not None:
            error_metrics["choice_partial_l2_error"], error_metrics["choice_partial_l2_bias"]  = metrics.masked_l2_difference(gt_choice, choice1, voxel_size, mask= partial_mask)

    unfiltered_map = recovar.utils.load_mrc(output_folder + "ml_optimized_unfiltered.mrc")
    gt_unfilt_metrics = metrics.compute_volume_error_metrics_from_gt(target_real, unfiltered_map, voxel_size, mask= mask, partial_mask = partial_mask )
    add_dict_with_prefix(error_metrics, gt_unfilt_metrics, "gt_unfilt_")

    filtered_map = recovar.utils.load_mrc(output_folder + "ml_optimized_locres_filtered.mrc")
    gt_filt_metrics = metrics.compute_volume_error_metrics_from_gt(target_real, filtered_map, voxel_size, mask= mask, partial_mask = partial_mask )
    add_dict_with_prefix(error_metrics, gt_filt_metrics, "gt_filt_")

    filtered_map = recovar.utils.load_mrc(output_folder + "ml_optimized_locres_filtered_before.mrc")
    gt_filt_metrics = metrics.compute_volume_error_metrics_from_gt(target_real, filtered_map, voxel_size, mask= mask, partial_mask = partial_mask )
    add_dict_with_prefix(error_metrics, gt_filt_metrics, "gt_filt_before")

    halfmap1 = recovar.utils.load_mrc(output_folder + "ml_optimized_half1_unfil.mrc")
    halfmap2 = recovar.utils.load_mrc(output_folder + "ml_optimized_half2_unfil.mrc")
    halfmap_metrics = metrics.compute_volume_error_metrics_from_halfmaps(halfmap1, halfmap2, voxel_size, mask= mask, partial_mask = partial_mask )
    add_dict_with_prefix(error_metrics, halfmap_metrics, "halfmap_")
    
    return error_metrics
    
def add_dict_with_prefix(dict1, dict_to_add, prefix):
    for key,value in dict_to_add.items():
        dict1[prefix + key] = value


def embed_from_median_label(z, gt_image_assignment):
    max_im = np.max(gt_image_assignment)+1
    median_labels = np.zeros((max_im, z.shape[1]))
    for k in range(max_im):
        median_labels[k] = np.median(z[gt_image_assignment == k], axis=0)
    return median_labels

def variance_of_zs(z, gt_image_assignment):

    max_im = np.max(gt_image_assignment)+1
    variances = np.zeros(max_im)
    total_variance = 0
    for k in range(max_im):
        sub_zs = z[gt_image_assignment == k]
        if sub_zs.size ==0:
            continue
        
        variances[k] = np.var(z[gt_image_assignment == k])
        mean_label = np.mean(z[gt_image_assignment == k], axis=0)
        total_variance += np.sum( (z[gt_image_assignment == k] - mean_label)**2)
    
    var_z = np.sum((z - np.mean(z, axis=0))**2) / z.shape[0]
    return variances, total_variance / z.shape[0], var_z


def get_embedding_from_median(zs, image_assignment, n_classes = None):
    n_classes = np.max(image_assignment) if n_classes is None else n_classes 
    # labels = np.unique(image_assignments)
    embeddings = np.zeros((n_classes, zs.shape[-1]))
    for lab in range(n_classes):
        embeddings[lab] = np.median(zs[image_assignment == lab], axis=0)
    return embeddings
