# If you want to extend and use recovar, you should import this first
import recovar.config 
import jax.numpy as jnp
import numpy as np

import os, argparse, time, pickle, logging
from recovar import output as o
from recovar import dataset, homogeneous, embedding, principal_components, latent_density, mask, utils, constants, noise, output
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser.add_argument('--zdim', type=list_of_ints, default=[1,2,4,10,20], help="Dimensions of latent variable. Default=1,2,4,10,20")

    # parser.add_argument(
    #     "--zdim", type=list, help="Dimension of latent variable"
    # )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, required=True, help="CTF parameters (.pkl)"
    )

    # parser.add_argument(
    #     "--mask", metavar="mrc", default=None, type=os.path.abspath, help="mask (.mrc)"
    # )

    parser.add_argument(
        "--mask", metavar="mrc", default=None, type=os.path.abspath, help="mask (.mrc)"
    )

    parser.add_argument(
        "--focus-mask", metavar="mrc", dest = "focus_mask", default=None, type=os.path.abspath, help="mask (.mrc)"
    )

    parser.add_argument(
        "--mask-option", metavar=str, default="input", help="mask options: from_halfmaps , input (default), sphere, none"
    )

    parser.add_argument(
        "--mask-dilate-iter", type=int, default=0, dest="mask_dilate_iter", help="mask options how many iters to dilate input mask (only used for input mask)"
    )

    parser.add_argument(
        "--correct-contrast",
        dest = "correct_contrast",
        action="store_true",
        help="estimate and correct for amplitude scaling (contrast) variation across images "
    )

    parser.add_argument(
        "--ignore-zero-frequency",
        dest = "ignore_zero_frequency",
        action="store_true",
        help="use if you want zero frequency to be ignored. If images have been normalized to 0 mean, this is probably a good idea"
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices",
    )

    group.add_argument(
        "--uninvert-data",
        dest="uninvert_data",
        default = "automatic",
        help="Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0",
    )

    # group.add_argument(
    #     "--rerescale",
    #     dest = "rerescale",
    #     action="store_true",
    # )

    # Should probably add these options
    # group.add_argument(
    #     "--no-window",
    #     dest="window",
    #     action="store_false",
    #     help="Turn off real space windowing of dataset",
    # )
    # group.add_argument(
    #     "--window-r",
    #     type=float,
    #     default=0.85,
    #     help="Windowing radius (default: %(default)s)",
    # )
    # group.add_argument(
    #     "--lazy",
    #     action="store_true",
    #     help="Lazy loading if full dataset is too large to fit in memory",
    # )

    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )
    group.add_argument(
            "--n-images",
            default = -1,
            dest="n_images",
            type=int,
            help="Number of images to use (should only use for quick run)",
        )
    
    group.add_argument(
            "--padding",
            type=int,
            default = 0,
            help="Real-space padding",
        )
    
    group.add_argument(
            "--halfsets",
            default = None,
            type=os.path.abspath,
            help="Path to a file with indices of split dataset (.pkl).",
        )

    ### CHANGE THESE TWO BACK!?!?!?!
    group.add_argument(
            "--keep-intermediate",
            dest = "keep_intermediate",
            action="store_true",
            help="saves some intermediate result. Probably only useful for debugging"
        )

    group.add_argument(
            "--noise-model",
            dest = "noise_model",
            default = "radial",
            help="what noise model to use. Options are radial (default) computed from outside the masks, and white computed by power spectrum at high frequencies"
        )

    group.add_argument(
            "--mean-fn",
            dest = "mean_fn",
            default = "triangular",
            help="which mean function to use. Options are triangular (default), old, triangular_reg"
        )

    group = parser.add_argument_group("Covariance estimation options")


    group.add_argument(
            "--covariance-fn",
            dest = "covariance_fn",
            default = "noisemask",
            help="noisemask (default), kernel"
        )

    group.add_argument(
            "--covariance-reg-fn",
            dest = "covariance_reg_fn",
            default = "old",
            help="old (default), new"
        )

    group.add_argument(
            "--covariance-left-kernel",
            dest = "covariance_left_kernel",
            default = "triangular",
            help="triangular (default), square"
        )

    group.add_argument(
            "--covariance-right-kernel",
            dest = "covariance_right_kernel",
            default = "triangular",
            help="triangular (default), square"
        )

    group.add_argument(
            "--covariance-left-kernel-width",
            dest = "covariance_left_kernel_width",
            default = 1,
            type=int,
        )

    group.add_argument(
            "--covariance-right-kernel-width",
            dest = "covariance_right_kernel_width",
            default = 2,
            type=int,
        )
    
    # options = {
    #     "covariance_fn": "noisemask",
    #     "reg_fn": "old",
    #     "left_kernel": "triangular",
    #     "right_kernel": "triangular",
    #     "left_kernel_width": 1,
    #     "right_kernel_width": 2,
    #     "shift_fsc": False,
    #     "substract_shell_mean": False,
    #     "grid_correct": True,
    #     "use_spherical_mask": True,
    #     "use_mask_in_fsc": False,
    #     "column_radius": 5,
    # }

    # group.add_argument(
    #         "--covariance-shift-fsc",
    #         dest = "covariance_shift_fsc",
    #         action="store_true",
    #     )


    # group.add_argument(
    #         "--covariance-substract-shell-mean",
    #         dest = "covariance_substract_shell_mean",
    #         action="store_true",
    #     )

    group.add_argument(
            "--covariance-grid-correct",
            dest = "covariance_substract_shell_mean",
            action="store_true",
        )



    group.add_argument(
            "--covariance-mask-in-fsc",
            dest = "covariance_mask_in_fsc",
            action="store_true",
        )


    group.add_argument(
            "--n-covariance-columns",
            dest = "covariance_reg_fn",
            default = "old",
            help="old (default), new"
        )

    group.add_argument(
            "--test-covar-options",
            dest = "test_covar_options",
            action="store_true",
        )

    group.add_argument(
            "--low-memory-option",
            dest = "low_memory_option",
            action="store_true",
        )


    group.add_argument(
            "--dont-use-image-mask",
            dest = "dont_use_image_mask",
            action="store_true",
        )

    group.add_argument(
            "--do-over-with-contrast",
            dest = "do_over_with_contrast",
            action="store_true",
        )
    return parser
    


def standard_recovar_pipeline(args):
    # import pdb; pdb.set_trace()
    st_time = time.time()

    if args.mask_option == 'input' and args.mask is None:
        raise ValueError("Mask option is input, but no mask provided. Provide a mask using --mask path/to/mask.mrc")

    o.mkdir_safe(args.outdir)
    logger.addHandler(logging.FileHandler(f"{args.outdir}/run.log"))
    logger.info(args)
    ind_split = dataset.figure_out_halfsets(args)

    dataset_loader_dict = dataset.make_dataset_loader_dict(args)
    options = utils.make_algorithm_options(args)

    cryos = dataset.get_split_datasets_from_dict(dataset_loader_dict, ind_split)
    cryo = cryos[0]
    gpu_memory = utils.get_gpu_memory_total()
    volume_shape = cryo.volume_shape
    disc_type = "linear_interp"

    batch_size = utils.get_image_batch_size(cryo.grid_size, gpu_memory)
    logger.info(f"image batch size: {batch_size}")
    logger.info(f"volume batch size: {utils.get_vol_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"column batch size: {utils.get_column_batch_size(cryo.grid_size, gpu_memory)}")
    logger.info(f"number of images: {cryos[0].n_images + cryos[1].n_images}")
    utils.report_memory_device(logger=logger)

    noise_var_from_hf, _ = noise.estimate_noise_variance(cryos[0], batch_size)

    # I need to rewrite the reweighted so it can use the more general noise distribution, but for now I'll go with that. 
    # cov_noise_init = cov_noise
    valid_idx = cryo.get_valid_frequency_indices()
    noise_model = args.noise_model

    # ## SETTING CONTRAST HE
    # print("SETTING CONTRAST HERE!!!")
    # path = '/projects/CRYOEM/singerlab/mg6942/simulated_empiar10180/volumes_256/vol/dataset_5_extra/contrast_qr_radial_new/'
    # from recovar import output
    # pipeline_output2 = output.PipelineOutput(path)
    # contrasts = pipeline_output2.get('contrasts')[10]
    # contrasts /= np.mean(contrasts)
    # embedding.set_contrasts_in_cryos(cryos, contrasts)
    # cryos_old = pipeline_output2.get('dataset')
    # import pdb; pdb.set_trace()
    
    n_repeats = 1
    if args.do_over_with_contrast:
        n_repeats = 2
    else:
        n_repeats = 1

    for repeat in range(n_repeats):

        if repeat == 1:
            if 10 in options['zs_dim_to_test']:
                ndim = 10
            else:
                ndim = np.median(options['zs_dim_to_test'])
            logger.warning(f"repeating with contrast of zdim={ndim}")
            contrasts_for_second = est_contrasts[ndim]
            contrasts_for_second /= np.mean(contrasts_for_second) # normalize to have mean 1
            embedding.set_contrasts_in_cryos(cryos, contrasts_for_second)
            options["contrast"] = "none"
        else:
            contrasts_for_second = None

        # Compute mean
        if args.mean_fn == 'old':
            means, mean_prior, _, _ = homogeneous.get_mean_conformation(cryos, 5*batch_size, noise_var_from_hf , valid_idx, disc_type, use_noise_level_prior = False, grad_n_iter = 5)
        elif args.mean_fn == 'triangular':
            means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = noise_var_from_hf,  use_regularization = False)
        elif args.mean_fn == 'triangular_reg':
            means, mean_prior, _, _  = homogeneous.get_mean_conformation_relion(cryos, 5*batch_size, noise_variance = noise_var_from_hf,  use_regularization = True)
        else:
            raise ValueError(f"mean function {args.mean_fn} not recognized")
        utils.report_memory_device(logger=logger)

        mean_real = ftu.get_idft3(means['combined'].reshape(cryos[0].volume_shape))

        ## DECIDE IF WE SHOULD UNINVERT DATA
        uninvert_check = np.sum((mean_real.real**3 * cryos[0].get_volume_radial_mask(cryos[0].grid_size//3).reshape(cryos[0].volume_shape))) < 0
        if args.uninvert_data == 'automatic':
            # Check if in real space, things towards the middle are mostly positive or negative
            if uninvert_check:
            # if np.sum(mean_real.real**3 * cryos[0].get_volume_mask() ) < 0:
                for key in ['combined', 'init0', 'init1', 'corrected0', 'corrected1']:
                    if key in means:
                        means[key] =- means[key]
                for cryo in cryos:
                    cryo.image_stack.mult = -1 * cryo.image_stack.mult
                args.uninvert_data = "true"
                logger.warning('sum(mean) < 0! swapping sign of data (uninvert-data = true)')
            else:
                logger.info('setting (uninvert-data = false)')
                args.uninvert_data = "false"
        elif uninvert_check:
            logger.warning('sum(mean) < 0! Data probably needs to be inverted! set --uninvert-data=true (or automatic)')
        ## END OF THIS - maybe move this block of code somewhere else?


        if means['combined'].dtype != cryo.dtype:
            logger.warning(f"mean estimate is in type: {means['combined'].dtype}")
            means['combined'] = means['combined'].astype(cryo.dtype)

        logger.info(f"mean computed in {time.time() - st_time}")

        # Compute mask
        volume_mask, dilated_volume_mask= mask.masking_options(args.mask_option, means, volume_shape, args.mask, cryo.dtype_real, args.mask_dilate_iter)

        if args.focus_mask is not None:
            focus_mask, _= mask.masking_options(args.mask_option, means, volume_shape, args.focus_mask, cryo.dtype_real, args.mask_dilate_iter)
        else:
            focus_mask = volume_mask

        # Let's see?
        
        noise_time = time.time()
        # Probably should rename all of this...
        masked_image_PS, std_masked_image_PS, image_PS, std_image_PS =  noise.estimate_radial_noise_statistic_from_outside_mask(cryo, dilated_volume_mask, batch_size)

        if args.mask_option is not None:
            radial_noise_var_outside_mask, _,_ =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)

            white_noise_var_outside_mask = noise.estimate_white_noise_variance_from_mask(cryo, dilated_volume_mask, batch_size)
            # white_noise_var_outside_mask = white_noise_var_outside_mask.copy()
        else:
            radial_noise_var_outside_mask = noise_var_from_hf * np.ones(cryos[0].grid_size//2 -1, dtype = np.float32)
            white_noise_var_outside_mask = noise_var_from_hf
            # radial_noise_var_outside_mask = noise_var_from_hf * np.ones_like(noise_var_outside_mask)

        logger.info(f"time to estimate noise is {time.time() - noise_time}")

        # radial_noise_var_outside_mask = np.where(radial_noise_var_outside_mask < 0, image_PS / 10, radial_noise_var_outside_mask)

        logger.info(f"time to estimate noise is {time.time() - noise_time}")

        # I believe that some versino of this is how relion/cryosparc infer the noise, but it seems like it would only be correct for homogeneous datasets
        # ub_noise_var, std_ub_noise_var, _, _ =  noise.estimate_radial_noise_upper_bound_from_inside_mask(cryo, means['combined'], dilated_volume_mask, batch_size)

        radial_ub_noise_var, _,_ =  noise.estimate_radial_noise_upper_bound_from_inside_mask_v2(cryo, means['combined'], dilated_volume_mask, batch_size)

        # noise_var_outside_mask, per_pixel_noise =  noise.estimate_noise_variance_from_outside_mask_v2(cryo, dilated_volume_mask, batch_size)

        noise_time = time.time()
        logger.info(f"time to upper bound noise is {time.time() - noise_time}")
        radial_noise_var_ubed = np.where(radial_noise_var_outside_mask >  radial_ub_noise_var, radial_ub_noise_var, radial_noise_var_outside_mask)
        # logger.warning("doing funky noise business")
        # noise_var = np.where(noise_var_outside_mask >  noise_var_from_hf, noise_var_outside_mask, np.ones_like(noise_var_from_hf))

        # noise_var_ = np.where(noise_var_outside_mask >  ub_noise_var, ub_noise_var, noise_var_outside_mask)

        # noise_var = noise_var_outside_mask
        # Noise statistic
        if noise_model == "white":
            noise_var_used = np.ones_like(radial_noise_var_ubed) * white_noise_var_outside_mask
        else:
            noise_var_used = radial_noise_var_ubed
        
        if (noise_var_used <0).any():
            logger.warning("Negative noise variance detected. Setting to image power spectrum / 10")

        noise_var_used = np.where(noise_var_used < 0, image_PS / 10, noise_var_used)

        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        from recovar import covariance_estimation
        variance_est, variance_prior, variance_fsc, lhs, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], batch_size//2, dilated_volume_mask, noise_variance = image_cov_noise,  use_regularization = True, disc_type = 'cubic')
        print('using regul in variance est?!?')


        rad_grid = np.array(ftu.get_grid_of_radial_distances(cryos[0].volume_shape).reshape(-1))
        # Often low frequency noise will be overestiated. This can be bad for the covariance estimation. This is a way to upper bound noise in the low frequencies by noise + variance .
        n_shell_to_ub = np.min([32, cryos[0].grid_size//2 -1])
        ub_noise_var_by_var_est = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_5_pc = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_median = np.zeros(n_shell_to_ub, dtype = np.float32)

        for k in range(n_shell_to_ub):
            if np.sum(rad_grid==k) >0:
                ub_noise_var_by_var_est[k] = np.percentile(noise_p_variance_est[rad_grid==k], 5)
                ub_noise_var_by_var_est[k] = np.max([0, ub_noise_var_by_var_est[k]])
                variance_est_low_res_5_pc[k] = np.percentile(variance_est['combined'][rad_grid==k], 5)
                variance_est_low_res_median[k] = np.median(variance_est['combined'][rad_grid==k])

        if np.any(ub_noise_var_by_var_est >  noise_var_used[:n_shell_to_ub]):
            logger.warning("Estimated noise greater than upper bound. Bounding noise using estimated upper obund")


        if np.any(variance_est_low_res_5_pc < 0):
            logger.warning("Estimated variance resolutino is < 0. This probably means that the noise was incorrectly estimated. Setting to 0")
            print("5 percentile:", variance_est_low_res_5_pc)
            print("5 percentile/median over low shells:", variance_est_low_res_5_pc/variance_est_low_res_median)

        noise_var_used[:n_shell_to_ub] = np.where( noise_var_used[:n_shell_to_ub] > ub_noise_var_by_var_est, ub_noise_var_by_var_est, noise_var_used[:n_shell_to_ub])

        noise_var_used = noise_var_used.astype(cryos[0].dtype_real)

        if noise_model == "mixed":
            # Noise at very low resolution is difficult to estimate. This is a heuristic to avoid some issues.
            fixed_resolution_shell = 32
            # Take min of PS and noise variance at fixed shell
            noise_var_used[:fixed_resolution_shell] = np.where(image_PS[:fixed_resolution_shell]> noise_var_used[fixed_resolution_shell], noise_var_used[fixed_resolution_shell], image_PS[:fixed_resolution_shell])

        ## DELETE FROM HERE?
        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        from recovar import covariance_estimation
        variance_est, variance_prior, variance_fsc, lhs, noise_p_variance_est = covariance_estimation.compute_variance(cryos, means['combined'], batch_size//2, dilated_volume_mask, noise_variance = image_cov_noise,  use_regularization = True, disc_type = 'cubic')
        print('using regul in variance est?!?')


        rad_grid = np.array(ftu.get_grid_of_radial_distances(cryos[0].volume_shape).reshape(-1))
        # Often low frequency noise will be overestiated. This can be bad for the covariance estimation. This is a way to upper bound noise in the low frequencies by noise + variance .
        n_shell_to_ub = np.min([32, cryos[0].grid_size//2 -1])
        ub_noise_var_by_var_est = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_5_pc = np.zeros(n_shell_to_ub, dtype = np.float32)
        variance_est_low_res_median = np.zeros(n_shell_to_ub, dtype = np.float32)

        for k in range(n_shell_to_ub):
            if np.sum(rad_grid==k) >0:
                ub_noise_var_by_var_est[k] = np.percentile(noise_p_variance_est[rad_grid==k], 5)
                ub_noise_var_by_var_est[k] = np.max([0, ub_noise_var_by_var_est[k]])
                variance_est_low_res_5_pc[k] = np.percentile(variance_est['combined'][rad_grid==k], 5)
                variance_est_low_res_median[k] = np.median(variance_est['combined'][rad_grid==k])

        if np.any(ub_noise_var_by_var_est >  noise_var_used[:n_shell_to_ub]):
            logger.warning("Estimated noise greater than upper bound. Bounding noise using estimated upper obund")


        if np.any(variance_est_low_res_5_pc < 0):
            logger.warning("Estimated variance resolutino is < 0. This probably means that the noise was incorrectly estimated. Setting to 0")
            print("5 percentile:", variance_est_low_res_5_pc)
            print("5 percentile/median over low shells:", variance_est_low_res_5_pc/variance_est_low_res_median)
        ## DELETE END HERE


        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))


        from recovar import covariance_estimation
        # test_covar_options = False
        if args.test_covar_options:
            tests = [ 
                {},
                {
                "mask_images_in_proj": False,
                "mask_images_in_H_B": False,
                },
                # {'column_sampling_scheme': 'high_snr_from_var_est', 'sampling_avoid_in_radius': 3 },
                # {'column_sampling_scheme': 'high_snr_from_var_est','sampling_n_cols':50,  'sampling_avoid_in_radius': 3, 'randomized_sketch_size' : 100, 'n_pcs_to_compute' : 100},
                # {'column_sampling_scheme': 'high_snr_from_var_est', 'sampling_avoid_in_radius': 1 },
                # {'column_sampling_scheme': 'high_snr_from_var_est', 'sampling_avoid_in_radius': 4 },
                ]
            idx = 0
            for test in tests:
                output_folder = args.outdir + '/output/' 
                # Compute principal components
                covariance_options = covariance_estimation.get_default_covariance_computation_options()
                for key in test:
                    covariance_options[key] = test[key]
        
                u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, noise_var_used, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model, covariance_options = covariance_options, variance_estimate = variance_est['combined'])
                from recovar import output
                output.mkdir_safe(output_folder)
                utils.pickle_dump({
                    'options':test, 'u' :u['rescaled'][:,:20], 's' :s['rescaled'][:20]
                }, output_folder + f'test_{idx}.pkl')
                del u, s, covariance_cols, picked_frequencies, column_fscs
                idx = idx + 1
                print('done with', idx, test)


        utils.report_memory_device(logger=logger)

        covariance_options = covariance_estimation.get_default_covariance_computation_options()
        if args.low_memory_option:
            covariance_options['sampling_n_cols'] = 50
            covariance_options['randomized_sketch_size'] = 100
            covariance_options['n_pcs_to_compute'] = 100
            covariance_options['sampling_avoid_in_radius'] = 3

        if args.dont_use_image_mask:
            # "mask_images_in_proj": True,
            # "mask_images_in_H_B": True,
            covariance_options['mask_images_in_proj'] = False
            covariance_options['mask_images_in_H_B'] = False


        # Compute principal components
        u,s, covariance_cols, picked_frequencies, column_fscs = principal_components.estimate_principal_components(cryos, options, means, mean_prior, noise_var_used, focus_mask, dilated_volume_mask, valid_idx, batch_size, gpu_memory_to_use=gpu_memory,noise_model=noise_model, covariance_options = covariance_options, variance_estimate = variance_est['combined'])

        if options['ignore_zero_frequency']:
            # Make the noise in 0th frequency gigantic. Effectively, this ignore this frequency when fitting.
            logger.info('ignoring zero frequency')
            noise_var_used[0] *=1e16

        image_cov_noise = np.asarray(noise.make_radial_noise(noise_var_used, cryos[0].image_shape))

        if not args.keep_intermediate:
            del u['real']
            if 'rescaled_no_contrast' in u:
                del u['rescaled_no_contrast']
            covariance_cols = None

        # Compute embeddings
        zs = {}; cov_zs = {}; est_contrasts = {}        
        for zdim in options['zs_dim_to_test']:
            z_time = time.time()
            zs[zdim], cov_zs[zdim], est_contrasts[zdim] = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] , zdim,
                                                                    image_cov_noise, cryos, volume_mask, gpu_memory, 'linear_interp',
                                                                    contrast_grid = None, contrast_option = options['contrast'],
                                                                    ignore_zero_frequency = options['ignore_zero_frequency'] )
            logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time}")

    zs_cont = {}; cov_zs_cont = {}; est_contrasts_cont = {}        
    # if args.correct_contrast:
    #     for zdim in options['zs_dim_to_test']:
    #         # contrast = est_contrasts[zdim]
    #         contrast_var = np.var(est_contrasts[zdim])
    #         z_time = time.time()
    #         zs_cont[zdim], cov_zs_cont[zdim], est_contrasts_cont[zdim] = embedding.get_per_image_embedding(means['combined'], u['rescaled'], s['rescaled'] , zdim,
    #                                                                 image_cov_noise, cryos, volume_mask, gpu_memory, covariance_options['disc_type'],
    #                                                                 contrast_grid = None, contrast_option = options['contrast'],
    #                                                                 ignore_zero_frequency = options['ignore_zero_frequency'], contrast_variance = contrast_var )
    #         logger.info(f"embedding time for zdim={zdim}: {time.time() - z_time} with contrast")


    n_images_to_test = np.round((cryos[0].n_images + cryos[1].n_images) * 0.01).astype(int)
    var_metrics, all_estimators, all_lhs = principal_components.test_different_embeddings_from_variance(cryos, zs, cov_zs, image_cov_noise, zdims= np.array(options['zs_dim_to_test']), n_images = n_images_to_test, tau = means['prior'])


    zdim = np.max(options['zs_dim_to_test'])
    noise_var_from_het_residual, _,_ = noise.estimate_noise_from_heterogeneity_residuals_inside_mask_v2(cryos[0], dilated_volume_mask, means['combined'], u['rescaled'][:,:zdim], est_contrasts[zdim], zs[zdim], batch_size//10, disc_type = covariance_options['disc_type'] )


    # ### END OF DEL

    logger.info(f"embedding time: {time.time() - st_time}")

    utils.report_memory_device()

    # Compute latent space density
    # Precompute the density on the 4D grid. This is the most expensive part of computing trajectories, and can be reused across trajectories. 
    logger.info(f"starting density computation")
    density_z = 10 if 10 in zs else options['zs_dim_to_test'][0]
    density, latent_space_bounds  = latent_density.compute_latent_space_density(zs[density_z], cov_zs[density_z], pca_dim_max = 4, num_points = 50, density_option = 'kde')
    logger.info(f"ending density computation")

    # Dump results to file
    output_model_folder = args.outdir + '/model/'
    o.mkdir_safe(args.outdir)
    o.mkdir_safe(output_model_folder)

    logger.info(f"peak gpu memory use {utils.get_peak_gpu_memory_used(device =0)}")

    if args.halfsets is None:
        pickle.dump(ind_split, open(output_model_folder + 'halfsets.pkl', 'wb'))
        args.halfsets = output_model_folder + 'halfsets.pkl'
    
    result = { 's' : s['rescaled'],'s_all': s,
                'input_args' : args,
                'latent_space_bounds' : np.array(latent_space_bounds), 
                'density': np.array(density),
                'noise_var_from_hf': noise_var_from_hf,
                'radial_noise_var_outside_mask' : np.array(radial_noise_var_outside_mask),
                'radial_ub_noise_var' : np.array(radial_ub_noise_var),
                'white_noise_var_outside_mask' : np.array(white_noise_var_outside_mask),
                'image_PS' : np.array(image_PS),
                'std_image_PS' : np.array(std_image_PS),
                'masked_image_PS' : np.array(masked_image_PS),
                'std_masked_image_PS' : np.array(std_masked_image_PS),
                'noise_var_from_het_residual' : np.array(noise_var_from_het_residual),
                'noise_var_used' : np.array(noise_var_used),
                'column_fscs': column_fscs, 
                'covariance_cols': covariance_cols, 
                'picked_frequencies' : picked_frequencies, 'volume_shape': volume_shape, 'voxel_size': cryos[0].voxel_size, 'pc_metric' : var_metrics['filt_var'],
                'variance_est': variance_est, 'variance_fsc': variance_fsc, 'noise_p_variance_est': noise_p_variance_est, 'ub_noise_var_by_var_est': ub_noise_var_by_var_est, 'covariance_options': covariance_options,
                'contrasts_for_second': contrasts_for_second}

    output_folder = args.outdir + '/output/' 
    o.mkdir_safe(output_folder)
    o.save_covar_output_volumes(output_folder, means['combined'], u['rescaled'], s, volume_mask, volume_shape,  voxel_size = cryos[0].voxel_size)
    o.save_volume(volume_mask, output_folder + 'volumes/' + 'mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    o.save_volume(dilated_volume_mask, output_folder + 'volumes/' + 'dilated_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)
    o.save_volume(focus_mask, output_folder + 'volumes/' + 'focus_mask', volume_shape, from_ft = False,  voxel_size = cryos[0].voxel_size)

    o.save_volume(means['corrected0'], output_folder + 'volumes/' + 'mean_half1_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)
    o.save_volume(means['corrected1'], output_folder + 'volumes/' + 'mean_half2_unfil', volume_shape, from_ft = True,  voxel_size = cryos[0].voxel_size)

    utils.pickle_dump(covariance_cols, output_model_folder + 'covariance_cols.pkl')
    utils.pickle_dump(result, output_model_folder + 'params.pkl')
    utils.pickle_dump({ 'zs': zs, 'cov_zs' : cov_zs , 'contrasts': est_contrasts, 'zs_cont' : zs_cont, 'cov_zs_cont' : cov_zs_cont, 'contrasts_cont' : est_contrasts_cont}, output_model_folder + 'embeddings.pkl')


    logger.info(f"Dumped results to file:, {output_model_folder}results.pkl")
    
    logger.info(f"total time: {time.time() - st_time}")
    
    # from analyze import analyze
    # analyze(args.outdir, output_folder = None, zdim=  np.max(options['zs_dim_to_test']), n_clusters = 40, n_paths= 2, skip_umap = False, q=None, n_std=None )

    return means, u, s, volume_mask, dilated_volume_mask, noise_var_used 


if __name__ == "__main__":
    # import jax
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    standard_recovar_pipeline(args)
