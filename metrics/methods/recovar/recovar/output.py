import logging
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib, pickle, os, json

import recovar.latent_density as ld
from recovar.fourier_transform_utils import fourier_transform_utils

ftu = fourier_transform_utils(jnp)
from recovar import embedding, linalg, trajectory, utils, dataset, regularization
import time

logger = logging.getLogger(__name__)


def get_resampled_distances(gt_vols):
    return trajectory.get_cum_curvelength(gt_vols)


def resample_trajectory(gt_vols, n_vols_along_path=6):
    distances_between_volumes = get_resampled_distances(gt_vols)

    # n_volumes at approximately equispaced points
    x = np.linspace(0, distances_between_volumes[-1], n_vols_along_path)
    gt_vols_x = np.interp(
        x,
        distances_between_volumes,
        np.arange(gt_vols.shape[0]),
        left=None,
        right=None,
        period=None,
    )

    indices_along_path = np.round(gt_vols_x).astype(int)
    return indices_along_path


def mkdir_safe(folder):
    os.makedirs(folder, exist_ok=True)


def save_volume(vol, path, volume_shape=None, from_ft=True, voxel_size=None):
    volume_shape = (
        3 * [utils.guess_grid_size_from_vol_size(vol.size)]
        if volume_shape is None
        else volume_shape
    )
    if from_ft:
        vol = np.real(ftu.get_idft3(vol.reshape(volume_shape)))
    else:
        vol = np.real(vol.reshape(volume_shape))
    utils.write_mrc(path + ".mrc", vol.astype(np.float32), voxel_size=voxel_size)


def save_volumes(
    volumes, save_path, volume_shape=None, from_ft=True, index_offset=0, voxel_size=None
):
    grid_size = np.round((volumes[0].shape[0]) ** (1 / 3)).astype(int)
    volume_shape = 3 * [grid_size] if volume_shape is None else volume_shape
    for v_idx, vol in enumerate(volumes):
        save_volume(
            vol,
            save_path + format(index_offset + v_idx, "03d"),
            volume_shape,
            from_ft=from_ft,
            voxel_size=voxel_size,
        )


def plot_on_same_scale(
    cs,
    xs,
    labels,
    plot_folder,
):
    plt.figure(figsize=(7, 5))
    k = 0

    for curve in cs:
        plt.plot(
            np.linspace(0, 1, curve.size), curve / np.max(curve), label=labels[k], lw=4
        )
        k += 1
    plt.legend()
    save_filepath = plot_folder + "path_density.png"
    # if save_to_file:
    plt.savefig(save_filepath, bbox_inches="tight")


def plot_two_twings_with_diff_scale(cs, xs, labels, plot_folder=None):
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)

    # Create some mock data
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_ylabel(labels[0], color=color, fontsize=20)
    x = np.linspace(0, 1, cs[0].size) if xs[0] is None else xs[0] / np.max(xs[0])
    ax1.plot(x, cs[0], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, np.max(cs[0]))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        labels[1], color=color, fontsize=20
    )  # we already handled the x-label with ax1
    x = np.linspace(0, 1, cs[1].size) if xs[1] is None else xs[1] / np.max(xs[1])
    ax2.plot(x, cs[1], color=color)
    ax2.set_ylim(0, np.max(cs[1]))

    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if plot_folder is not None:
        save_filepath = plot_folder + "path_density_t.png"
        # if save_to_file:
        plt.savefig(save_filepath, bbox_inches="tight")
    plt.show()


def sum_over_other(x, use_axis=[0, 1]):
    other_axes = []
    for k in range(x.ndim):
        if k not in use_axis:
            other_axes.append(k)

    xk = np.sum(x, axis=tuple(other_axes))
    return xk


def plot_trajectories_over_density_from_result(results, trajectories, subsampled, zdim):
    latent_space_bounds = ld.compute_latent_space_bounds(results["zs"][zdim])
    plot_trajectories_over_density(
        results["density"],
        trajectories,
        latent_space_bounds,
        subsampled=subsampled,
        colors=None,
        plot_folder=None,
        cmap="inferno",
        same_st_end=True,
        zs=results["zs"][zdim],
        cov_zs=results["cov_zs"][zdim],
    )
    return


def plot_trajectories_over_density(
    density,
    trajectories,
    latent_space_bounds,
    subsampled=None,
    colors=None,
    plot_folder=None,
    cmap="inferno",
    same_st_end=True,
    zs=None,
    cov_zs=None,
):
    colors = ["k", "cornflowerblue"] if colors is None else colors
    path_exists = trajectories is not None

    compute_density = False
    if density is None:
        assert zs is not None
        assert cov_zs is not None
        compute_density = True

    if compute_density:
        num_points = 200
    else:
        num_points = density.shape[0]
    grid_to_z, z_to_grid = ld.get_grid_z_mappings(latent_space_bounds, num_points)

    # path_grid = z_to_grid(path_z[:,:low_dim])
    # g_st = z_to_grid(z_st[:low_dim])
    # g_end = z_to_grid(z_end[:low_dim])

    def plot_traj_along_axes(axes, save_to_file=False):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]
        if compute_density:
            density_pl, _ = ld.compute_latent_space_density_on_2_axes(
                zs, cov_zs, axes=axes, num_points=num_points
            )
        else:
            density_pl = sum_over_other(density, axes)

        if axis_x > axis_y:
            density_pl = density_pl.T

        ax.imshow(
            (density_pl.T), origin="lower", cmap=cmap, interpolation="bilinear"
        )  # [...,25,25])
        # plt.colorbar()

        if path_exists:
            # path_grid = z_to_grid(path)
            for traj_idx, traj in enumerate(trajectories):

                # if compute_density:
                # traj = trajectory.copy()
                traj = z_to_grid(traj)

                plt.plot(traj[:, axis_x], traj[:, axis_y], "-", c="w", linewidth=6)
                plt.plot(
                    traj[:, axis_x],
                    traj[:, axis_y],
                    "--",
                    c=colors[traj_idx],
                    dashes=[3],
                    linewidth=6,
                )
                if subsampled is not None:
                    # subs = subsampled[traj_idx]
                    subs = z_to_grid(subsampled[traj_idx].copy())
                    plt.scatter(
                        subs[:, axis_x],
                        subs[:, axis_y],
                        marker="o",
                        c=colors[traj_idx],
                        edgecolors="w",
                        s=500,
                        zorder=2,
                    )

                if not same_st_end or traj_idx == 0:
                    g_st = traj[0]
                    g_end = traj[-1]
                    plt.scatter(
                        g_st[axis_x],
                        g_st[axis_y],
                        marker="*",
                        c="w",
                        edgecolors=colors[traj_idx],
                        s=1800,
                        zorder=2,
                    )
                    plt.scatter(
                        g_end[axis_x],
                        g_end[axis_y],
                        marker="s",
                        c="w",
                        edgecolors=colors[traj_idx],
                        s=600,
                        zorder=2,
                    )

        ax.axis("off")

        if plot_folder is not None:
            save_filepath = (
                plot_folder + "density_" + str(axes[0]) + str(axes[1]) + ".png"
            )
            plt.savefig(save_filepath, bbox_inches="tight")

    traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    for k1 in range(np.min([traj_dim, 3])):
        for k2 in range(k1 + 1, traj_dim):
            plot_traj_along_axes([k1, k2])


def save_covar_output_volumes(
    output_folder,
    mean,
    u,
    s,
    mask,
    volume_shape,
    us_to_save=50,
    us_to_var=[4, 10, 20],
    voxel_size=None,
):

    mkdir_safe(output_folder + "volumes/")
    save_volumes(
        [u[..., k] for k in range(us_to_save)],
        output_folder + "volumes/" + "eigen_pos",
        volume_shape=volume_shape,
        voxel_size=voxel_size,
    )
    save_volumes(
        [-u[..., k] for k in range(us_to_save)],
        output_folder + "volumes/" + "eigen_neg",
        volume_shape=volume_shape,
        voxel_size=voxel_size,
    )
    save_volume(
        mean,
        output_folder + "volumes/" + "mean",
        volume_shape=volume_shape,
        voxel_size=voxel_size,
    )

    grid_size = np.round((mean.shape[0]) ** (1 / 3)).astype(int)
    vol_batch_size = int((2**24) / (grid_size**3))
    for n_eigs in us_to_var:
        u_real = linalg.batch_idft3(u[..., :n_eigs], volume_shape, vol_batch_size)
        variance_real = utils.estimate_variance(u_real.T, s["rescaled"][:n_eigs])
        save_volume(
            variance_real,
            output_folder + "volumes/" + "variance" + str(n_eigs),
            volume_shape,
            from_ft=False,
            voxel_size=voxel_size,
        )


# def kmeans_analysis_from_dict(output_folder, pipeline_output, cryos, likelihood_threshold,  n_clusters = 20, generate_volumes = True, zdim =-1, compute_reproj = False):
#     from recovar import dataset

#     if cryos is None:

#         cryos = pipeline_output.get('dataset')#(results['input_args']) if cryos is None else cryos
#         embedding.set_contrasts_in_cryos(cryos, pipeline_output.get('contrasts')[zdim])

#     return kmeans_analysis(output_folder, cryos, pipeline_output.get('mean'), results['u']['rescaled'], results['zs'][zdim], results['cov_zs'][zdim], results['cov_noise'], likelihood_threshold,  n_clusters = n_clusters, generate_volumes = generate_volumes, compute_reproj = compute_reproj)


def kmeans_analysis(output_folder, zs, n_clusters=20):

    import cryodrgn.analysis as cryodrgn_analysis

    # key = 'zs12'
    # zs = results[key]
    # cov_zs = results['cov_' + key]
    reorder = zs.shape[1] != 1
    labels, centers = cryodrgn_analysis.cluster_kmeans(zs, n_clusters, reorder=reorder)

    import os

    try:
        os.mkdir(output_folder)
    except:
        pass

    def plot_axes(axes=[0, 1]):
        fig, ax = cryodrgn_analysis.scatter_annotate(
            zs[:, axes[0]],
            zs[:, axes[1]],
            centers=centers[:, axes],
            centers_ind=None,
            annotate=True,
            labels=None,
            alpha=0.1,
            s=1,
        )
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(
                output_folder + "centers_" + str(axes[0]) + str(axes[1]) + ".png"
            )

        fig, ax = cryodrgn_analysis.scatter_annotate(
            zs[:, axes[0]],
            zs[:, axes[1]],
            centers=centers[:, axes],
            centers_ind=None,
            annotate=False,
            labels=None,
            alpha=0.1,
            s=2,
        )
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(
                output_folder
                + "centers_"
                + str(axes[0])
                + str(axes[1])
                + "no_annotate.png"
            )

    for k in range(1, zs.shape[-1]):
        plot_axes(axes=[0, k])
    if zs.shape[-1] > 2:
        plot_axes(axes=[1, 2])

    return centers, labels


def plot_umap(output_folder, zs, centers):
    import cryodrgn.analysis as cryodrgn_analysis

    def plot_axes(axes=[0, 1]):
        fig, ax = cryodrgn_analysis.scatter_annotate(
            zs[:, axes[0]],
            zs[:, axes[1]],
            centers=centers[:, axes],
            centers_ind=None,
            annotate=True,
            labels=None,
            alpha=0.1,
            s=1,
        )
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + "centers_.png")

        fig, ax = cryodrgn_analysis.scatter_annotate(
            zs[:, axes[0]],
            zs[:, axes[1]],
            centers=centers[:, axes],
            centers_ind=None,
            annotate=False,
            labels=None,
            alpha=0.1,
            s=2,
        )
        fig.set_figheight(6)
        fig.set_figwidth(6)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if output_folder is not None:
            plt.savefig(output_folder + "centers_no_annotate.png")

        import seaborn as sns

        g = sns.jointplot(x=zs[:, 0], y=zs[:, 1], alpha=0.1, s=1)
        g.set_axis_labels("UMAP1", "UMAP2")
        if output_folder is not None:

            plt.savefig(output_folder + "sns.png")

        g = sns.jointplot(x=zs[:, 0], y=zs[:, 1], kind="hex")
        g.set_axis_labels("UMAP1", "UMAP2")
        if output_folder is not None:
            plt.savefig(output_folder + "sns_hex.png")

    plot_axes(axes=[0, 1])


def compute_and_save_reweighted(
    cryos,
    path_subsampled,
    zs,
    cov_zs,
    noise_variance,
    output_folder,
    B_factor,
    n_bins=30,
    n_min_images=100,
    embedding_option="cov_dist",
    save_all_estimates=False,
):

    # batch_size =

    mkdir_safe(output_folder)
    new_volume_generation = True
    if new_volume_generation:
        from recovar import heterogeneity_volume, latent_density

        for k in range(path_subsampled.shape[0]):
            output_folder_this = (
                os.path.join(output_folder, "vol" + format(k, "03d")) + "/"
            )
            mkdir_safe(output_folder_this)
            ndim = zs.shape[-1]
            # n_bins = 30
            latent_points = path_subsampled[k][None]

            if embedding_option == "llh":
                log_likelihoods = latent_density.compute_latent_log_likelihood(
                    latent_points, zs, cov_zs
                )[..., 0]
                heterogeneity_distances = log_likelihoods - np.min(log_likelihoods)
            elif embedding_option == "cov_dist":
                cov_zs = cov_zs  # *0 + np.eye(dim)
                heterogeneity_distances = (
                    latent_density.compute_latent_quadratic_forms_in_batch(
                        latent_points, zs, cov_zs
                    )[..., 0]
                )
            elif embedding_option == "dist":
                cov_zs = cov_zs * 0 + np.eye(ndim)
                heterogeneity_distances = latent_density.compute_latent_log_likelihood(
                    latent_points, zs, cov_zs
                )[..., 0]
            else:
                raise ValueError("Unknown embed option")

            # latent_points = path_subsampled[k][None]
            # log_likelihoods = latent_density.compute_latent_quadratic_forms_in_batch(latent_points[:,:ndim], zs, cov_zs)[...,0]
            heterogeneity_distances = [
                heterogeneity_distances[: cryos[0].n_images],
                heterogeneity_distances[cryos[0].n_images :],
            ]

            # heterogeneity_volume.make_volumes_kernel_estimate_local(heterogeneity_distances, cryos, noise_variance, output_folder_this, -1, n_bins, B_factor, tau = None, n_min_images = 300, metric_used = "locres_auc")
            from recovar import noise

            noise_variance = noise.make_radial_noise(
                noise_variance, cryos[0].image_shape
            )
            locres_maskrad = cryos[0].grid_size * cryos[0].voxel_size / 20
            logger.info(
                f"Setting locres_maskrac = locres_sampling = box_size * voxel_size / 20 = {locres_maskrad:.1f} Angstroms"
            )

            heterogeneity_volume.make_volumes_kernel_estimate_local(
                heterogeneity_distances,
                cryos,
                noise_variance,
                output_folder_this,
                ndim,
                n_bins,
                B_factor,
                tau=None,
                n_min_images=n_min_images,
                locres_sampling=locres_maskrad,
                locres_maskrad=locres_maskrad,
                locres_edgwidth=0,
                upsampling_for_ests=1,
                use_mask_ests=False,
                grid_correct_ests=False,
                save_all_estimates=save_all_estimates,
                metric_used="locshellmost_likely",
            )
            logger.info(
                f"Done with volume generation {k} stored in {output_folder_this}"
            )

    # memory_to_use = utils.get_gpu_memory_total() - path_subsampled.shape[0] * cryos[0].volume_size * 8 / 1e9 * 8
    # assert memory_to_use > 0, "reduce number of volumes computed at once"
    # batch_size = 2 * utils.get_image_batch_size(cryos[0].grid_size, memory_to_use)
    # logger.info(f"batch size in reweighting: {batch_size}")

    # else:
    #     trajectory_prior, halfmaps = embedding.generate_conformation_from_reweighting(cryos, means, noise_variance, zs, cov_zs, path_subsampled, batch_size = batch_size, disc_type = 'linear_interp', likelihood_threshold = likelihood_threshold, recompute_prior = recompute_prior, volume_mask = volume_mask, adaptive=adaptive)
    #     save_volumes(trajectory_prior, output_folder +  'reweight_')
    #     save_volumes(halfmaps[0], output_folder +  'halfmap0_reweight_')
    #     save_volumes(halfmaps[1], output_folder +  'halfmap1_reweight_')


# def compute_and_save_volumes_from_z(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, adaptive = False ):
#     mkdir_safe(output_folder)
#     compute_and_save_reweighted(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, adaptive = adaptive)

# if compute_reproj:
#     n_eigs = zs.shape[1]
#     trajectory_reproj = embedding.generate_conformation_from_reprojection(path_subsampled, means['combined'], u[:,:n_eigs] )
#     save_volumes(trajectory_reproj, output_folder +  'reproj_')


def plot_loglikelihood_over_scatter(
    path_subsampled, zs, cov_zs, save_path, likelihood_threshold=1e-5
):
    scale_zs = ld.compute_det_cov_xs(cov_zs)
    likelihoods = ld.compute_likelihood_of_latent_given_image(
        path_subsampled, zs, cov_zs, scale_zs
    )

    likelihoods = ld.compute_latent_quadratic_forms(path_subsampled, zs, cov_zs)
    vmax = np.max(likelihoods)
    vmin = np.max([np.min(likelihoods), 1e-8 * np.max(likelihoods)])
    # print(vmax, vmin)
    plt.ioff()
    for k in range(likelihoods.shape[1]):
        fig, ax = plt.subplots(figsize=(8, 8))
        greater_x = likelihoods[:, k] > vmin
        plt.scatter(
            zs[~greater_x, 0],
            zs[~greater_x, 1],
            c="black",
            alpha=0.1,
            s=2,
            edgecolors="none",
        )

        plt.scatter(
            zs[greater_x, 0],
            zs[greater_x, 1],
            c=likelihoods[greater_x, k],
            cmap="rainbow",
            s=5,
            alpha=1,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True),
            edgecolors="none",
        )

        plt.xticks([], [])
        plt.yticks([], [])
        save_filepath = (
            save_path + format(k, "03d") + ".png"
        )  # output_folder + 'plots/' + 'vol_weights' +str(k) + '.png'
        plt.savefig(save_filepath, bbox_inches="tight")
        if k > 0:
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


def load_results(datadir, option_str="from_halfmap"):
    model_folder = datadir + "model" + option_str + "/"
    output_folder = datadir + "output" + option_str + "/"
    with open(model_folder + "results.pkl", "rb") as f:
        results = pickle.load(f)
    results["dataset_loader_dict"]["compute_ground_truth"] = False
    # dataset_loader = dataset.get_dataset_loader_from_dict(results['dataset_loader_dict'])
    grid_to_z, z_to_grid = ld.get_grid_z_mappings(
        results["latent_space_bounds"], results["density"].shape[0]
    )
    return (
        results,
        results["dataset_loader_dict"],
        grid_to_z,
        z_to_grid,
        output_folder,
    )  # ['mean'], results['u'], results['s'], results['volume_mask'], results['dilated_volume_mask'],


def load_results_new(datadir):
    model_folder = datadir + "model" + "/"
    output_folder = datadir + "output" + "/"
    with open(model_folder + "results.pkl", "rb") as f:
        results = pickle.load(f)
    # results['dataset_loader_dict']['compute_ground_truth'] = False
    # dataset_loader = dataset.get_dataset_loader_from_dict(results['dataset_loader_dict'])
    # grid_to_z, z_to_grid = ld.get_grid_z_mappings(results['latent_space_bounds'], results['density'].shape[0])
    return results  # , results['dataset_loader_dict'], grid_to_z, z_to_grid


class PipelineOutput:
    def __init__(self, result_path):
        self.params = utils.pickle_load(result_path + "model/params.pkl")
        self.embedding = None
        self.embedding_loaded = False
        self.result_path = result_path

    def get(self, key):
        if key in self.params:
            return self.params[key]
        elif key in [
            "zs",
            "cov_zs",
            "contrasts",
            "zs_cont",
            "cov_zs_cont",
            "est_contrasts_cont",
        ]:
            if self.embedding_loaded:
                return self.embedding[key]
            else:
                self.embedding = utils.pickle_load(
                    self.result_path + "model/" + "embeddings" + ".pkl"
                )
                self.embedding_loaded = True
                return self.embedding[key]
        elif key == "u" or key == "u_real":
            n_pcs = 50
            u = np.zeros([n_pcs, *(self.params["volume_shape"])])
            for i in range(n_pcs):
                u[i] = utils.load_mrc(
                    self.result_path
                    + "output/volumes/"
                    + "eigen_pos"
                    + format(i, "03d")
                    + ".mrc"
                )
            if key == "u_real":
                return u
            else:
                # return self.params['volume_shape'], 10).reshape(n_pcs, -1)
                return ftu.get_dft3(u).reshape(n_pcs, -1)

        elif key == "mean":
            return ftu.get_dft3(
                utils.load_mrc(self.result_path + "output/volumes/" + "mean" + ".mrc")
            ).reshape(-1)

        elif key == "mean_halfmaps":
            half1 = ftu.get_dft3(
                utils.load_mrc(
                    self.result_path + "output/volumes/" + "mean_half1_unfil" + ".mrc"
                )
            ).reshape(-1)
            half2 = ftu.get_dft3(
                utils.load_mrc(
                    self.result_path + "output/volumes/" + "mean_half2_unfil" + ".mrc"
                )
            ).reshape(-1)
            return half1, half2
        elif key == "image_snr":
            vol_shape = self.get("volume_shape")
            PS = regularization.average_over_shells(
                np.abs((self.get("mean").reshape(self.get("volume_shape")))) ** 2,
                self.get("volume_shape"),
            )
            noise_level = self.get("noise_var_used")
            snr = utils.make_radial_image(PS / noise_level, vol_shape[:2])
            return snr

        elif key == "volume_mask":
            return utils.load_mrc(
                self.result_path + "output/volumes/" + "mask" + ".mrc"
            )
        elif key == "dilated_volume_mask":
            return utils.load_mrc(
                self.result_path + "output/volumes/" + "dilated_mask" + ".mrc"
            )
        elif key == "covariance_cols":
            return utils.load_pickle(
                self.result_path + "model/" + "covariance_cols" + ".pkl"
            )
        elif key == "dataset":
            return dataset.load_dataset_from_args(self.params["input_args"], lazy=False)
        elif key == "lazy_dataset":
            return dataset.load_dataset_from_args(self.params["input_args"], lazy=True)
        else:
            assert False, "key not found"

    def keys(self):
        keys = list(self.params.keys())
        keys += [
            "zs",
            "cov_zs",
            "contrasts",
            "u",
            "u_real",
            "mean",
            "volume_mask",
            "dilated_volume_mask",
            "covariance_cols",
            "dataset",
            "lazy_dataset",
        ]
        return keys


def load_results_newest(datadir):
    model_folder = datadir + "model" + "/"
    output_folder = datadir + "output" + "/"
    with open(model_folder + "results.pkl", "rb") as f:
        results = pickle.load(f)
    # results['dataset_loader_dict']['compute_ground_truth'] = False
    # dataset_loader = dataset.get_dataset_loader_from_dict(results['dataset_loader_dict'])
    # grid_to_z, z_to_grid = ld.get_grid_z_mappings(results['latent_space_bounds'], results['density'].shape[0])
    return results  # , results['dataset_loader_dict'], grid_to_z, z_to_grid


def make_trajectory_plots_from_results(
    pipeline_output,
    basis_size,
    output_folder,
    cryos=None,
    z_st=None,
    z_end=None,
    gt_volumes=None,
    n_vols_along_path=6,
    plot_llh=False,
    compute_reproj=False,
    likelihood_threshold=None,
    adaptive=False,
):

    assert ((z_st is not None) and (z_end is not None)) or (
        gt_volumes is not None
    ), "either z_st and z_end should be passed, or gt_volumes"

    # results = load_results_new(results['output_dir'])
    if cryos is None:
        cryos = pipeline_output.get("dataset") if cryos is None else cryos
        embedding.set_contrasts_in_cryos(
            cryos, pipeline_output.get("contrasts")[basis_size]
        )

    # cryos = dataset.load_dataset_from_args(results['input_args']) if cryos is None else cryos
    latent_space_bounds = ld.compute_latent_space_bounds(
        pipeline_output.get("zs")[basis_size]
    )

    return make_trajectory_plots(
        pipeline_output.get("density"),
        pipeline_output.get("zs")[basis_size],
        pipeline_output.get("cov_zs")[basis_size],
        z_st,
        z_end,
        latent_space_bounds,
        output_folder,
        gt_volumes=None,
        n_vols_along_path=n_vols_along_path,
        plot_llh=plot_llh,
    )


def make_trajectory_plots(
    density,
    zs,
    cov_zs,
    z_st,
    z_end,
    latent_space_bounds,
    output_folder,
    gt_volumes=None,
    n_vols_along_path=6,
    plot_llh=False,
    use_input_density=False,
):
    st_time = time.time()

    # likelihood_threshold = ld.get_log_likelihood_threshold(k = zs.shape[-1]) if likelihood_threshold is None else likelihood_threshold
    gt_volumes = None
    # if gt_volumes is not None:
    #     gt_subs_idx = resample_trajectory(gt_volumes, n_vols_along_path = n_vols_along_path)

    #     # Find ground truth volumes in latent coordinates
    #     save_volumes(tuple(gt_volumes),   output_folder + '/gt' , volume_shape = None, from_ft = True  )
    #     save_volumes(tuple(gt_volumes[gt_subs_idx]),   output_folder + '/gt_subs' , volume_shape = None, from_ft = True  )
    #     json.dump(gt_subs_idx.tolist(), open(output_folder + '/gt_index_resampled.json', 'w'))

    #     z_vols = vol_to_z(gt_volumes, u, means['combined'], basis_size)
    #     z_st = z_vols[0]
    #     z_end = z_vols[-1]

    #     gt_volumes_z = vol_to_z(gt_volumes, u, means['combined'], basis_size)
    #     z_st = gt_volumes_z[0]
    #     z_end = gt_volumes_z[-1]
    #     # gt_coords_grid = z_to_grid(gt_volumes_z)

    #     same_st_end = True

    basis_size = zs.shape[1]
    zs = zs[:, :basis_size]
    latent_space_bounds = ld.compute_latent_space_bounds(zs)
    cov_zs = cov_zs[:, :basis_size, :basis_size]
    st_time = time.time()

    mkdir_safe(output_folder + "density/")
    if basis_size > 1:
        if not use_input_density:
            path_z = trajectory.compute_high_dimensional_path(
                zs,
                cov_zs,
                z_st,
                z_end,
                density_low_dim=density,
                density_eps=1e-5,
                max_dim=basis_size,
                percentile_bound=1,
                num_points=50,
                use_log_density=False,
            )
        else:
            path_z = trajectory.compute_fixed_dimensional_path(
                z_st,
                z_end,
                density,
                latent_space_bounds,
                density_eps=1e-5,
                debug_plot=False,
                density_option="kde",
                use_log_density=False,
            )

        path_subsampled = trajectory.subsample_path(path_z, n_pts=n_vols_along_path)

        logger.info(f"after path {time.time() - st_time}")
        # trajectory.subsample_path(path_z, n_pts = n_vols_along_path)
        # plot_trajectories_over_density(density, None,latent_space_bounds,  colors = None, plot_folder = output_folder + 'density_nopath/', cmap = 'inferno')
        inp_dens = density if use_input_density else None

        if gt_volumes is not None:
            plot_trajectories_over_density(
                inp_dens,
                [gt_volumes_z, path_z],
                latent_space_bounds,
                subsampled=[gt_volumes_z[gt_subs_idx][1:-1], path_subsampled[1:-1]],
                colors=["k", "cornflowerblue"],
                plot_folder=output_folder,
                cmap="inferno",
                zs=zs,
                cov_zs=cov_zs,
            )
        else:
            plot_trajectories_over_density(
                inp_dens,
                [path_z],
                latent_space_bounds,
                subsampled=[path_subsampled[1:-1]],
                colors=["cornflowerblue"],
                plot_folder=output_folder,
                cmap="inferno",
                same_st_end=False,
                zs=zs,
                cov_zs=cov_zs,
            )
    else:
        path_z = np.linspace(z_st, z_end, n_vols_along_path)[..., 0]
        path_subsampled = path_z
        path_subsampled = path_subsampled

    st_time = time.time()
    # compute_and_save_reweighted(dataset_loader, path_subsampled, zs, cov_zs, noise_variance, output_folder, B_factor, n_bins = 30)
    # logger.info(f"vol time {time.time() - st_time}")

    if use_input_density:
        grid_to_z, z_to_grid = ld.get_grid_z_mappings(
            latent_space_bounds, num_points=density.shape[0]
        )
        path_grid = z_to_grid(path_z)
        import jax.scipy

        density_on_path = jax.scipy.ndimage.map_coordinates(
            density, path_grid.T, order=1
        )

        # density_on_path = ld.compute_latent_space_density_at_pts(path_z, zs, cov_zs)
    else:
        density_on_path = ld.compute_latent_space_density_at_pts(path_z, zs, cov_zs)
    densities = {
        "density": density_on_path.tolist(),
        "path": path_z.tolist(),
        "path_subsampled": path_subsampled.tolist(),
    }
    json.dump(densities, open(output_folder + "/path.json", "w"))

    if plot_llh:
        plot_loglikelihood_over_scatter(
            path_subsampled,
            zs,
            cov_zs,
            save_path=output_folder,
            likelihood_threshold=None,
        )

    logger.info(f"after all plots {time.time() - st_time}")
    return path_z, path_subsampled


def vol_to_z(gt_volumes, u, mean, basis_size):
    coords = ((np.conj(u[:, :basis_size].T) @ (gt_volumes - mean).T).T).real
    return coords


def plot_trajectories_over_scatter(
    trajectories,
    subsampled=None,
    colors=None,
    plot_folder=None,
    cmap="inferno",
    same_st_end=True,
    zs=None,
    cov_zs=None,
):
    colors = ["k", "cornflowerblue"] if colors is None else colors
    path_exists = trajectories is not None

    def plot_traj_along_axes(axes, save_to_file=False):
        axes = tuple(axes)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_frame_on(True)

        axis_x = axes[0]
        axis_y = axes[1]

        if axis_x > axis_y:
            density_pl = density_pl.T

        ax.scatter(zs[:, axis_x], zs[:, axis_y], s=1, alpha=0.05)

        if path_exists:
            # path_grid = z_to_grid(path)
            for traj_idx, traj in enumerate(trajectories):

                plt.plot(
                    traj[:, axis_x],
                    traj[:, axis_y],
                    "-o",
                    c=colors[traj_idx],
                    linewidth=3,
                    zorder=3,
                )
                # plt.plot(traj[:,axis_x], traj[:,axis_y], '--', c=colors[traj_idx], dashes=[3], linewidth=6)

                if subsampled is not None:
                    subs = subsampled[traj_idx]
                    if subs is not None:
                        plt.scatter(
                            subs[:, axis_x],
                            subs[:, axis_y],
                            marker=">",
                            c=colors[traj_idx],
                            edgecolors="w",
                            s=200,
                            zorder=3,
                        )

                # for traj_idx, traj in enumerate(trajectories):
                if not same_st_end or traj_idx == 0:
                    g_st = traj[0]
                    g_end = traj[-1]
                    plt.scatter(
                        g_st[axis_x],
                        g_st[axis_y],
                        marker="*",
                        c="w",
                        edgecolors=colors[traj_idx],
                        s=1800,
                        zorder=2,
                    )
                    plt.scatter(
                        g_end[axis_x],
                        g_end[axis_y],
                        marker="s",
                        c="w",
                        edgecolors=colors[traj_idx],
                        s=600,
                        zorder=2,
                    )

        # ax.axis("off")

        if plot_folder is not None:
            save_filepath = (
                plot_folder + "density_" + str(axes[0]) + str(axes[1]) + ".png"
            )
            plt.savefig(save_filepath, bbox_inches="tight")

    traj_dim = trajectories[0].shape[1] if trajectories is not None else 4
    # print(traj_dim)
    for k1 in range(np.min([traj_dim, 3])):
        for k2 in range(k1 + 1, traj_dim):
            plot_traj_along_axes([k1, k2])


def umap_latent_space(zs):
    import umap

    st_time = time.time()
    n_components = np.min([zs.shape[1], 2])
    mapper = umap.UMAP(n_components=n_components).fit(zs)
    # umap.plot.points(mapper)
    logger.info(f"time to umap: {time.time() - st_time}")
    return mapper
