from importlib import reload
import recovar.config
import numpy as np
from recovar import dataset
import jax.numpy as jnp
from recovar import locres
from recovar import adaptive_kernel_discretization
import logging
from recovar.fourier_transform_utils import fourier_transform_utils

ftu = fourier_transform_utils(jnp)
logger = logging.getLogger(__name__)


def pick_minimum_discretization_size(ndim, log_likelihoods, q=0.5, min_images=50):
    if ndim > 0:
        disc_latent_dist = recovar.latent_density.get_log_likelihood_threshold(
            k=ndim, q=0.5
        )
    else:
        disc_latent_dist = -1
    value = np.max(
        [np.sort(log_likelihoods)[min_images], disc_latent_dist]
    )  # Bump a lil bit
    return value * (1 + 1e-6)


# def pick_heterogeneity_bins(ndim, log_likelihoods, q = 0.5, min_images = 50, n_bins = 11):
#     disc_latent_dist = pick_minimum_discretization_size(ndim, log_likelihoods, q , min_images )
#     max_latent_dist = np.percentile(log_likelihoods, 0.9)
# return (np.linspace(1, 6 , n_bins ) **2) * disc_latent_dist


def pick_heterogeneity_bins2(ndim, log_likelihoods, q=0.5, min_images=50, n_bins=11):
    disc_latent_dist = pick_minimum_discretization_size(
        ndim, log_likelihoods, q, min_images
    )

    max_latent_dist = np.percentile(log_likelihoods, 95)
    # dist = np.linspace(np.sqrt(disc_latent_dist), np.sqrt(max_latent_dist), 11 ) **2
    return np.linspace(np.sqrt(disc_latent_dist), np.sqrt(max_latent_dist), n_bins) ** 2


def make_volumes_kernel_estimate_from_results(
    latent_point,
    results,
    ndim,
    cryos=None,
    n_bins=11,
    output_folder=None,
    B_factor=0,
    metric_used="locmost_likely",
    n_min_images=50,
):

    cryos = (
        dataset.load_dataset_from_args(results["input_args"], lazy=False)
        if cryos is None
        else cryos
    )
    output_folder = (
        results["input_args"].outdir + "/output/"
        if output_folder is None
        else output_folder
    )
    print("Dumping to ", output_folder)
    recovar.output.mkdir_safe(output_folder)
    noise_variance = results["cov_noise"]
    latent_points = latent_point[None]

    log_likelihoods = recovar.latent_density.compute_latent_quadratic_forms_in_batch(
        latent_points[:, :ndim], results["zs"][ndim], results["cov_zs"][ndim]
    )[..., 0]
    heterogeneity_distances = [
        log_likelihoods[: cryos[0].n_images],
        log_likelihoods[cryos[0].n_images :],
    ]
    if metric_used == "global":
        a = 1
        # make_volumes_kernel_estimate( heterogeneity_distances, cryos, noise_variance, output_folder, ndim, n_bins)
    else:
        print("CHOOSING THREHSOLD ONLY BASED ON NUMBER OF IMAGES! FIX?")
        make_volumes_kernel_estimate_local(
            heterogeneity_distances,
            cryos,
            noise_variance,
            output_folder,
            -1,
            n_bins,
            B_factor,
            tau=None,
            n_min_images=n_min_images,
            metric_used=metric_used,
        )


# from recovar import relion_functions


# def make_volumes_kernel_estimate(heterogeneity_distances, cryos, noise_variance, output_folder, ndim, bins, tau = None, n_min_images = 80):

#     if type(bins) == int:
#         heterogeneity_bins = pick_heterogeneity_bins2(ndim, heterogeneity_distances[1], 0.5, n_min_images, n_bins = bins)
#     else:
#         heterogeneity_bins = bins

#     residual_threshold = heterogeneity_bins[0]

#     n_images_per_bin = [ np.sum(heterogeneity_distances[1] <= b) for b in heterogeneity_bins ]
#     print(n_images_per_bin)
#     # import pdb; pdb.set_trace()
#     estimates = [None, None]
#     lhs = [None, None]
#     rhs = [None, None]
#     for k in range(2):
#         estimates[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, compute_lhs_rhs=False)

#     cross_validation_estimators = [None, None]
#     for k in range(2):
#         cross_validation_estimators[k], lhs[k], rhs[k] = adaptive_kernel_discretization.naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins[0:1], tau= tau, compute_lhs_rhs=True)


#     import recovar
#     recovar.utils.pickle_dump( { "lhs" : lhs, "rhs" : rhs } ,  output_folder  + "lhs_rhs.pkl")

#     logger.info(f"Computing estimates done")

#     import jax.numpy as jnp
#     index_array_vol, fdisc, residuals_avged,  summed_residuals  = adaptive_kernel_discretization.pick_best_heterogeneity_from_residual((estimates[0].T)[...,None],
#                                         cryos[1], heterogeneity_distances[1], heterogeneity_bins,
#                                         residual_threshold = residual_threshold , min_number_of_images_in_bin = 50)
#     logger.info(f"Computing estimates done")


#     opt_halfmaps = [None, None]
#     for k in range(2):
#         opt_halfmaps[k] = jnp.take_along_axis(estimates[k].T , np.expand_dims(index_array_vol, axis=-1), axis=-1)

#     recovar.output.save_volume(opt_halfmaps[0], output_folder + "optimized_half1_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)
#     recovar.output.save_volume(opt_halfmaps[1], output_folder + "optimized_half2_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)

#     recovar.output.save_volumes(estimates[0], output_folder + "estimates_half1_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)
#     recovar.output.save_volumes(estimates[1], output_folder + "estimates_half2_unfil", cryos[0].volume_shape, voxel_size = cryos[0].voxel_size)

#     recovar.utils.pickle_dump( { "index_array_vol" : index_array_vol, "fdisc" : fdisc, "residuals_avged" : residuals_avged, "n_images_per_bin" :n_images_per_bin, "summed_residuals" : summed_residuals } ,  output_folder  + "stuff.pkl")
#     return estimates, opt_halfmaps, index_array_vol, fdisc, residuals_avged


def make_volumes_kernel_estimate_local(
    heterogeneity_distances,
    cryos,
    noise_variance,
    output_folder,
    ndim,
    bins,
    B_factor,
    tau=None,
    n_min_images=50,
    metric_used="locmost_likely",
    upsampling_for_ests=1,
    use_mask_ests=False,
    grid_correct_ests=False,
    locres_sampling=25,
    locres_maskrad=None,
    locres_edgwidth=None,
    kernel_rad=4,
    save_all_estimates=False,
):

    if type(bins) == int:
        # heterogeneity_bins = pick_heterogeneity_bins2(ndim, heterogeneity_distances[1], 0.5, n_min_images, n_bins = bins)

        logger.warning(
            f"Picking bins based on number of images only. n_min_images = {n_min_images}"
        )
        heterogeneity_bins = pick_heterogeneity_bins2(
            -1, heterogeneity_distances[1], 0.5, n_min_images, n_bins=bins
        )

    else:
        heterogeneity_bins = bins

    logger.info(f"bins {heterogeneity_bins}")
    n_images_per_bin = [
        (
            np.sum(heterogeneity_distances[0] < b)
            + np.sum(heterogeneity_distances[1] < b)
        )
        for b in heterogeneity_bins
    ]
    # logger.info(f"images per bin {*n_images_per_bin}")
    print("images per bin", n_images_per_bin)
    # print(n_images_per_bin)
    # import pdb; pdb.set_trace()
    estimates = [None, None]
    lhs, rhs = [None, None], [None, None]
    cross_validation_estimators = [None, None]
    for k in range(2):
        logger.info(f"Computing estimates start")
        ##
        # print("OHHHHHH HERE")
        cryos[k].update_volume_upsampling_factor(upsampling_for_ests)

        estimates[
            k
        ] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(
            cryos[k],
            noise_variance.astype(np.float32),
            None,
            heterogeneity_distances[k],
            heterogeneity_bins,
            tau=tau,
            grid_correct=grid_correct_ests,
            use_spherical_mask=use_mask_ests,
        )
        estimates[k] = ftu.get_idft3(
            estimates[k].reshape(-1, *cryos[0].volume_shape)
        ).real.astype(np.float32)

        # print(heterogeneity_distances[k][:10])
        # estimates2 = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, grid_correct=False, use_spherical_mask=False)

        # print(heterogeneity_distances[k][:10])

        # estimates3 = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins, tau= tau, grid_correct=False, use_spherical_mask=False)

        # print(np.linalg.norm(estimates3 - estimates[k]) / np.linalg.norm(estimates2))

        # import pdb; pdb.set_trace()
        # print(np.linalg.norm(estimates3 - estimates[k]) / np.linalg.norm(estimates2))

        # print(np.linalg.norm(estimates3 - estimates2) / np.linalg.norm(estimates2))
        # print(np.linalg.norm(lhs - lhs2) / np.linalg.norm(lhs2))
        # print(np.linalg.norm(lhs[0] - lhs2[0]) / np.linalg.norm(lhs2[0]))
        # # print(np.linalg.norm(lhs[0] - lhs2[0]) / np.linalg.norm(lhs2[0]))
        # print(np.linalg.norm(rhs[0] - rhs2[0]) / np.linalg.norm(rhs[0]))

        # print(np.linalg.norm(rhs - rhs2) / np.linalg.norm(rhs))
        # import pdb; pdb.set_trace()
        # np.linalg.norm(estimates[k][-1] - estimates2[-1]) / np.linalg.norm(estimates2[-1])

        logger.info(f"Computing estimates done")

        # cross_validation_estimators[k], lhs[k], rhs[k] =
        #
        cryos[k].update_volume_upsampling_factor(1)

        (
            cross_validation_estimators[k],
            lhs[k],
            rhs[k],
        ) = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(
            cryos[k],
            noise_variance.astype(np.float32),
            None,
            heterogeneity_distances[k],
            heterogeneity_bins[0:1],
            tau=tau,
            grid_correct=False,
            use_spherical_mask=False,
            return_lhs_rhs=True,
        )
        # return_lhs_rhs=True)
        # import pdb; pdb.set_trace()

        lhs[k] = adaptive_kernel_discretization.half_volume_to_full_volume(
            lhs[k][0], cryos[k].volume_shape
        )
        # Zero out things after Nyquist - these won't be used in CV
        lhs[k] = (lhs[k] * cryos[0].get_valid_frequency_indices()).reshape(
            cryos[0].volume_shape
        )

        cross_validation_estimators[k] = ftu.get_idft3(
            cross_validation_estimators[k].reshape(cryos[0].volume_shape)
        ).real.astype(np.float32)

        cryos[k].update_volume_upsampling_factor(upsampling_for_ests)

    # for k in range(2):
    #     cryos[k].update_volume_upsampling_factor(1)

    #     cross_validation_estimators[k], lhs[k], rhs[k] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(cryos[k], noise_variance.astype(np.float32), None, heterogeneity_distances[k], heterogeneity_bins[0:1], tau= tau, grid_correct=False, use_spherical_mask=False, return_lhs_rhs=True)

    #     lhs[k] = adaptive_kernel_discretization.half_volume_to_full_volume(lhs[k][0], cryos[k].volume_shape)
    #     # Zero out things after Nyquist - these won't be used in CV
    #     lhs[k] = (lhs[k] * cryos[0].get_valid_frequency_indices()).reshape(cryos[0].volume_shape)

    #     cross_validation_estimators[k] = ftu.get_idft3(cross_validation_estimators[k].reshape(cryos[0].volume_shape)).real.astype(np.float32)

    logger.info(f"Computing estimates done")

    from_ft = False

    # metric_used = "locshellmost_likely"
    # # Choice from these estimators, but then recompute nicer ones?
    # ml_choice, ml_errors = choice_most_likely(estimates[0], estimates[1], cross_validation_estimators[0], cross_validation_estimators[1], lhs[0], lhs[1], cryos[0].voxel_size, locres_sampling=locres_sampling, locres_maskrad=locres_maskrad, locres_edgwidth=locres_edgwidth)
    # split_choice, _ = choice_most_likely_split(estimates[0], estimates[1], cross_validation_estimators[0], cross_validation_estimators[1], lhs[0], lhs[1], cryos[0].voxel_size, locres_sampling=locres_sampling, locres_maskrad=locres_maskrad, locres_edgwidth=locres_edgwidth)
    do_smooth_error = "smooth" in metric_used

    if metric_used == "locmost_likely":
        ml_choice, ml_errors = choice_most_likely(
            estimates[0],
            estimates[1],
            cross_validation_estimators[0],
            cross_validation_estimators[1],
            lhs[0],
            lhs[1],
            cryos[0].voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
        )
    elif "locshellmost_likely" in metric_used:
        ml_choice, ml_errors = choice_most_likely_split(
            estimates[0],
            estimates[1],
            cross_validation_estimators[0],
            cross_validation_estimators[1],
            lhs[0],
            lhs[1],
            cryos[0].voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            smooth_error=do_smooth_error,
        )
    else:
        raise ValueError("Metric used not recognized")

    # locres_choice, locres_score, auc_choice, auc_score = choice_best_locres(estimates[0], estimates[1][0], cryos[0].voxel_size)

    # estimates = np.asarray(estimates)
    estimates = [None, None]

    for k in range(2):
        logger.info(f"Computing estimates start")
        cryos[k].update_volume_upsampling_factor(2)
        estimates[
            k
        ] = adaptive_kernel_discretization.even_less_naive_heterogeneity_scheme_relion_style(
            cryos[k],
            noise_variance.astype(np.float32),
            None,
            heterogeneity_distances[k],
            heterogeneity_bins,
            tau=None,
            grid_correct=True,
            use_spherical_mask=True,
        )
        estimates[k] = ftu.get_idft3(
            estimates[k].reshape(-1, *cryos[0].volume_shape)
        ).real.astype(np.float32)

    # for k in range(2):
    #     for i in range(estimates[k].shape[0]):
    #         from recovar import mask as mask_fn
    #         if use_mask_ests is False:
    #             estimates[k][i], _ = mask_fn.soft_mask_outside_map(estimates[k][i].reshape(cryos[0].volume_shape), cosine_width = 3)
    #         else:
    #             estimates[k][i] = estimates[k][i].reshape(cryos[0].volume_shape)

    #         if grid_correct_ests is False:
    #             gridding_correct = "square"
    #             grid_fn = relion_functions.griddingCorrect_square if gridding_correct == "square" else relion_functions.griddingCorrect
    #             kernel_width = 1
    #             order =1
    #             estimates[k][i], _ = grid_fn(estimates[k][i].reshape(cryos[0].volume_shape), cryos[0].grid_size, cryos[0].volume_upsampling_factor/kernel_width, order = order)

    ########## FROM HERE ##############
    def use_choice_and_filter(choice, name):

        # Take best then filter

        opt_halfmaps = [None, None]
        # opt_halfmaps2 = [None, None]

        for k in range(2):
            if metric_used == "locmost_likely":
                opt_halfmaps[k] = jnp.take_along_axis(
                    estimates[k], choice[None], axis=0
                )[0]
                _, smoothed_choice = smoothed_best_choice(
                    estimates[0], choice, kernel_rad=kernel_rad
                )
            elif "locshellmost_likely" in metric_used:
                opt_halfmaps[k] = locres.recombine_estimates(
                    estimates[k],
                    choice,
                    cryos[0].voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=locres_maskrad,
                    locres_edgwidth=locres_edgwidth,
                )

        best_filtered, best_filtered_res, best_auc, fscs, _ = locres.local_resolution(
            opt_halfmaps[0],
            opt_halfmaps[1],
            B_factor,
            cryos[0].voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=None,
            locres_edgwidth=None,
            locres_minres=50,
            use_filter=True,
            fsc_threshold=1 / 7,
            use_v2=False,
        )

        # best_filtered, best_filtered_res, best_auc, fscs, _ = locres.local_resolution(opt_halfmaps2[0], opt_halfmaps2[1], B_factor, cryos[0].voxel_size, locres_sampling = locres_sampling, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True, fsc_threshold = 1/7, use_v2 = False)
        # recovar.utils.write_mrc(output_folder + name + "optimized_locres_filtered_split.mrc", best_filtered, voxel_size = cryos[0].voxel_size)

        best_filtered_nob, _, _, _, _ = locres.local_resolution(
            opt_halfmaps[0],
            opt_halfmaps[1],
            0,
            cryos[0].voxel_size,
            locres_sampling=locres_sampling,
            locres_maskrad=None,
            locres_edgwidth=None,
            locres_minres=50,
            use_filter=True,
            fsc_threshold=1 / 7,
            use_v2=True,
        )

        recovar.utils.write_mrc(
            output_folder + name + "optimized_locres_filtered_nob.mrc",
            best_filtered_nob,
            voxel_size=cryos[0].voxel_size,
        )

        # Best filtered
        recovar.utils.write_mrc(
            output_folder + name + "optimized_locres_filtered.mrc",
            best_filtered,
            voxel_size=cryos[0].voxel_size,
        )
        recovar.utils.write_mrc(
            output_folder + name + "optimized_locres.mrc",
            best_filtered_res,
            voxel_size=cryos[0].voxel_size,
        )
        recovar.utils.write_mrc(
            output_folder + name + "optimized_auc.mrc",
            best_auc,
            voxel_size=cryos[0].voxel_size,
        )

        # Also store halfmaps. This naming is important to also import into relion
        recovar.utils.write_mrc(
            output_folder + name + "optimized_half1_unfil.mrc",
            opt_halfmaps[0],
            voxel_size=cryos[0].voxel_size,
        )
        recovar.utils.write_mrc(
            output_folder + name + "optimized_half2_unfil.mrc",
            opt_halfmaps[1],
            voxel_size=cryos[0].voxel_size,
        )
        recovar.utils.write_mrc(
            output_folder + name + "optimized_unfiltered.mrc",
            (opt_halfmaps[0] + opt_halfmaps[1]) / 2,
            voxel_size=cryos[0].voxel_size,
        )

        if save_all_estimates:

            if metric_used == "locmost_likely":
                # Take best smoothed then filter
                opt_halfmaps = [None, None]
                for k in range(2):
                    opt_halfmaps[k], _ = smoothed_best_choice(
                        estimates[k], choice, kernel_rad=kernel_rad
                    )

                (
                    best_filtered,
                    best_filtered_res,
                    best_auc,
                    fscs,
                    _,
                ) = locres.local_resolution(
                    opt_halfmaps[0],
                    opt_halfmaps[1],
                    B_factor,
                    cryos[0].voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=None,
                    locres_edgwidth=None,
                    locres_minres=50,
                    use_filter=True,
                    fsc_threshold=1 / 7,
                    use_v2=False,
                )

                recovar.utils.write_mrc(
                    output_folder + name + "optimized_locres_filtered_smooth.mrc",
                    best_filtered,
                    voxel_size=cryos[0].voxel_size,
                )
                recovar.utils.write_mrc(
                    output_folder + name + "optimized_locres_smooth.mrc",
                    best_filtered_res,
                    voxel_size=cryos[0].voxel_size,
                )
                recovar.utils.write_mrc(
                    output_folder + name + "optimized_auc_smooth.mrc",
                    best_auc,
                    voxel_size=cryos[0].voxel_size,
                )

            # Filter then take best
            loc_filtered_estimates = np.zeros_like(estimates[0])
            for i in range(estimates[0].shape[0]):
                loc_filtered_estimates[i], _, _, _, _ = locres.local_resolution(
                    estimates[0][i],
                    estimates[1][i],
                    B_factor,
                    cryos[0].voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=None,
                    locres_edgwidth=None,
                    locres_minres=50,
                    use_filter=True,
                    fsc_threshold=1 / 7,
                    use_v2=True,
                )

            if metric_used == "locmost_likely":
                opt_filtered_before = jnp.take_along_axis(
                    loc_filtered_estimates, choice[None], axis=0
                )[0]
            elif "locshellmost_likely" in metric_used:
                opt_filtered_before = locres.recombine_estimates(
                    loc_filtered_estimates,
                    choice,
                    cryos[0].voxel_size,
                    locres_sampling=locres_sampling,
                    locres_maskrad=locres_maskrad,
                    locres_edgwidth=locres_edgwidth,
                )

            # opt_filtered_before = jnp.take_along_axis(loc_filtered_estimates , choice[None], axis=0)[0]
            recovar.utils.write_mrc(
                output_folder + name + "optimized_locres_filtered_before.mrc",
                opt_filtered_before,
                voxel_size=cryos[0].voxel_size,
            )

            if metric_used == "locmost_likely":
                opt_filtered_before, smoothed_choice = smoothed_best_choice(
                    loc_filtered_estimates, choice, kernel_rad=kernel_rad
                )
                recovar.utils.write_mrc(
                    output_folder
                    + name
                    + "optimized_locres_filtered_before_smooth.mrc",
                    opt_filtered_before,
                    voxel_size=cryos[0].voxel_size,
                )

            recovar.output.save_volumes(
                loc_filtered_estimates,
                output_folder + "estimates_filt",
                cryos[0].volume_shape,
                voxel_size=cryos[0].voxel_size,
                from_ft=from_ft,
            )

            # recovar.utils.write_mrc(output_folder + name + "est_filtered.mrc", loc_filtered_estimates, voxel_size = cryos[0].voxel_size)

            ## TODO: I am not sure whether local filtering should be done before or after combining

        if "locshellmost_likely" in metric_used:
            recovar.utils.pickle_dump(
                {"split_choice": ml_choice, "ml_errors": ml_errors},
                output_folder + "split_choice.pkl",
            )
        else:
            recovar.utils.write_mrc(
                output_folder + name + "optimized_choice.mrc",
                choice,
                voxel_size=cryos[0].voxel_size,
            )
            recovar.utils.write_mrc(
                output_folder + name + "optimized_choice_smooth.mrc",
                smoothed_choice,
                voxel_size=cryos[0].voxel_size,
            )

        output_dict = {
            "heterogeneity_bins": heterogeneity_bins,
            "n_images_per_bin": n_images_per_bin,
            "fscs": fscs,
            "heterogeneity_distances": heterogeneity_distances,
        }
        recovar.utils.pickle_dump(output_dict, output_folder + name + "params.pkl")

    use_choice_and_filter(ml_choice, "ml_")
    # use_choice_and_filter(locres_choice, "locres_")
    # use_choice_and_filter(auc_choice, "auc_")

    if save_all_estimates:
        recovar.output.save_volumes(
            estimates[0],
            output_folder + "estimates_half1_unfil",
            cryos[0].volume_shape,
            voxel_size=cryos[0].voxel_size,
            from_ft=from_ft,
        )
        recovar.output.save_volumes(
            estimates[1],
            output_folder + "estimates_half2_unfil",
            cryos[0].volume_shape,
            voxel_size=cryos[0].voxel_size,
            from_ft=from_ft,
        )

        recovar.utils.write_mrc(
            output_folder + "CV_estimates_half1_unfil.mrc",
            cross_validation_estimators[0],
            voxel_size=cryos[0].voxel_size,
        )
        recovar.utils.write_mrc(
            output_folder + "CV_noise_half1.mrc", lhs[0], voxel_size=cryos[0].voxel_size
        )
        recovar.utils.write_mrc(
            output_folder + "CV_noise_half2.mrc", lhs[1], voxel_size=cryos[0].voxel_size
        )
        recovar.utils.write_mrc(
            output_folder + "CV_estimates_half2_unfil.mrc",
            cross_validation_estimators[1],
            voxel_size=cryos[0].voxel_size,
        )

    return


def choice_most_likely(
    estimates0,
    estimates1,
    target0,
    target1,
    noise_variances_target0,
    noise_variances_target1,
    voxel_size,
    locres_sampling,
    locres_maskrad,
    locres_edgwidth,
):

    n_estimators = estimates0.shape[0]
    errors = np.zeros_like(estimates0)
    use_v2 = True
    for k in range(n_estimators):
        errors[k] = locres.expensive_local_error_with_cov(
            target0,
            estimates1[k],
            voxel_size,
            noise_variances_target0.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
        )
        errors[k] += locres.expensive_local_error_with_cov(
            estimates0[k],
            target1,
            voxel_size,
            noise_variances_target1.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
        )

    choice = np.argmin(errors, axis=0)
    return choice, errors


import jax.scipy


def smooth_shell_error(
    shell_error, voxel_size, subarray_size, sum_up_up_to_res=50, smooth_mean_filter=3
):
    # Smooth out the shell error
    kernel = jnp.ones(smooth_mean_filter, dtype=jnp.float32)
    # kernel = kernel / jnp.sum(kernel)
    vmapped_convolve = jax.vmap(jax.scipy.signal.convolve, in_axes=(0, None, None))
    shell_choice_new = vmapped_convolve(shell_error, kernel, "same")

    # For very low frequencies, just sum up
    full_grids = ftu.get_1d_frequency_grid(subarray_size, voxel_size, scaled=True)
    grids = full_grids[-shell_error.shape[-1] - 1 : -1]
    # print(grids)
    # import pdb; pdb.set_trace()
    low_res_indices = grids <= 1 / sum_up_up_to_res
    logger.info(
        f"Averaging first {jnp.sum(low_res_indices)} shells out of {shell_error.shape[-1]} until resolution {sum_up_up_to_res}. Smoothing shells with kernel size {smooth_mean_filter}"
    )
    shell_choice_new = jnp.where(
        grids <= 1 / sum_up_up_to_res,
        jnp.sum(shell_error * low_res_indices),
        shell_choice_new,
    )
    # shell_choice_new = shell_choice_new.at[low_res_indices].set(jnp.sum(shell_error * low_res_indices ))
    return shell_choice_new


batch_smooth_shell_error = jax.vmap(
    smooth_shell_error, in_axes=(0, None, None, None, None)
)


def choice_most_likely_split(
    estimates0,
    estimates1,
    target0,
    target1,
    noise_variances_target0,
    noise_variances_target1,
    voxel_size,
    locres_sampling,
    locres_maskrad,
    locres_edgwidth,
    smooth_error=False,
):

    n_estimators = estimates0.shape[0]
    errors = n_estimators * [None]
    use_v2 = True
    for k in range(n_estimators):
        errors[k] = locres.expensive_local_error_with_cov(
            target0,
            estimates1[k],
            voxel_size,
            noise_variances_target0.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
            split_shell=True,
        )
        errors[k] += locres.expensive_local_error_with_cov(
            estimates0[k],
            target1,
            voxel_size,
            noise_variances_target1.reshape(target0.shape),
            locres_sampling=locres_sampling,
            locres_maskrad=locres_maskrad,
            locres_edgwidth=locres_edgwidth,
            use_v2=use_v2,
            split_shell=True,
        )

    errors = np.asarray(errors)
    if smooth_error:
        subarray_size = int((errors.shape[-1] + 1) * 2)
        print("Subarray size", subarray_size)
        sum_up_up_to_res = 40
        smooth_mean_filter = 3
        errors = batch_smooth_shell_error(
            errors, voxel_size, subarray_size, sum_up_up_to_res, smooth_mean_filter
        )

    choice = np.argmin(errors, axis=0)
    return choice, errors


# def choice_best_locres(estimates0, estimates1, target_idx, voxel_size):
def choice_best_locres(estimates1, target0, voxel_size):

    from recovar import locres

    reload(locres)

    n_estimators = estimates1.shape[0]
    locressol = np.zeros_like(estimates1)
    auc_score = np.zeros_like(estimates1)

    for k in range(n_estimators):
        # _, locressol[k], auc_score[k] , _, _ = locres.local_resolution(estimates0[target_idx], estimates1[k], 0, voxel_size, locres_sampling = 25, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = True)
        _, _, locressol[k], auc_score[k] = locres.local_resolution(
            target0,
            estimates1[k],
            0,
            voxel_size,
            locres_sampling=15,
            locres_maskrad=None,
            locres_edgwidth=None,
            locres_minres=50,
            use_filter=False,
            use_v2=True,
        )

    choice = np.argmin(locressol, axis=0)
    choice2 = np.argmax(auc_score, axis=0)

    return choice, locressol, choice2, auc_score


from recovar import mask


def smoothed_best_choice(estimates, choice, kernel_rad=4):
    smoothed_choice = mask.soften_volume_mask(choice, kernel_rad)
    bot_boundary = jnp.floor(smoothed_choice).astype(int)

    max_choice = np.max(choice)
    min_choice = np.min(choice)
    bot_boundary = np.where(bot_boundary < min_choice, min_choice, bot_boundary)
    bot_boundary = np.where(bot_boundary > max_choice, max_choice, bot_boundary)

    weight = smoothed_choice - bot_boundary
    bot_estimate = jnp.take_along_axis(estimates, bot_boundary[None], axis=0)[0]

    top_boundary = bot_boundary + 1
    top_boundary = np.where(top_boundary < min_choice, min_choice, top_boundary)
    top_boundary = np.where(top_boundary > max_choice, max_choice, top_boundary)

    top_estimate = jnp.take_along_axis(estimates, top_boundary[None], axis=0)[0]

    smoothed_estimate = (1 - weight) * bot_estimate + (weight) * top_estimate
    return smoothed_estimate, smoothed_choice
