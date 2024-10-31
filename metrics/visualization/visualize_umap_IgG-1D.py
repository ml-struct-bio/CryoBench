import os
import sys
import argparse
import pickle
import json
import glob
import numpy as np
from cryodrgn import analysis
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "fsc"))
from utils import volumes


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "result_path",
        type=os.path.abspath,
        help="umap & latent folder before method name (e.g. /scratch/gpfs/ZHONGE/mj7341/CryoBench/results/IgG-1D/snr0.01)",
    )
    parser.add_argument(
        "-o",
        type=os.path.abspath,
        required=True,
        help="Output folder to save the UMAP plot",
    )

    parser.add_argument("--num_imgs", type=int, help="number of images")
    parser.add_argument("--num_classes", type=int, default=20, help="number of classes")
    parser.add_argument("--num_vols", type=int, default=100, help="number of classes")

    return parser


def parse_class_assignments(path_for_label, path_for_model, K):
    # for labels
    cs = np.load(path_for_label)
    keys = ["alignments_class3D_{}/class_posterior".format(i) for i in range(K)]
    classes = [[x[k] for k in keys] for x in cs]
    classes = np.asarray(classes)
    class_id = classes.argmax(axis=1)

    # for models
    model_lst = []
    path_for_model.split("/")[-2]
    cs_model_path = [
        path_for_model
        + path_for_model.split("/")[-2]
        + "_passthrough_particles_class_{}.cs".format(i)
        for i in range(K)
    ]
    for i in range(K):
        cs_model = np.load(cs_model_path[i])
        for j in range(len(cs_model)):
            num_vol = cs_model[j][1].split(b"_")[1].decode("utf-8")
            model_lst.append(num_vol)
    model_id = np.asarray(model_lst)

    return class_id, model_id


def parse_class_abinit_assignments(path_for_label, path_for_model, K):
    # for labels
    cs = np.load(path_for_label, allow_pickle=True)
    keys = ["alignments_class_{}/class_posterior".format(i) for i in range(K)]
    classes = [[x[k] for k in keys] for x in cs]
    classes = np.asarray(classes)
    class_id = classes.argmax(axis=1)

    # for models
    model_lst = []
    path_for_model.split("/")[-2]
    cs_model_path = [
        path_for_model
        + path_for_model.split("/")[-2]
        + "_class_{:02d}_final_particles.cs".format(i)
        for i in range(K)
    ]
    for i in range(K):
        cs_model = np.load(cs_model_path[i])
        for j in range(len(cs_model)):
            num_vol = cs_model[j][1].split(b"_")[1].decode("utf-8")
            model_lst.append(num_vol)
    model_id = np.asarray(model_lst)

    return class_id, model_id


def plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04):

    x = np.repeat(dihedral_angles, 1000)
    xx = np.cos(x / 180 * np.pi)
    yy = np.sin(x / 180 * np.pi)

    xx_jittered = xx + np.random.randn(len(xx)) * jitter
    yy_jittered = yy + np.random.randn(len(yy)) * jitter

    colorList = plt.cm.tab20.colors
    cmap = matplotlib.colors.ListedColormap(colorList[: args.num_classes])

    plt.scatter(
        xx_jittered,
        yy_jittered,
        cmap=cmap,
        s=1,
        alpha=0.1,
        c=labels_3dcls,
        vmin=0,
        vmax=args.num_classes,
        rasterized=True,
    )
    plt_umap_labels()
    plt.savefig(
        f"{args.o}/{args.method}/{args.method}_{args.num_classes}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plt_umap_labels():
    plt.xticks([])
    plt.yticks([])


def plot_methods(args, method, v, dihedral_angles, is_umap=True, use_axis=False):
    fig, ax = plt.subplots(figsize=(4, 4))
    # Whole
    plot_dim = (0, 1)
    # Dihedral angles
    c_all = np.repeat(dihedral_angles, args.num_imgs)
    c = c_all
    plot_args = dict(alpha=0.1, s=1, cmap="gist_rainbow", vmin=0, vmax=356.4)
    ax.scatter(v[:, plot_dim[0]], v[:, plot_dim[1]], c=c, rasterized=True, **plot_args)

    if is_umap:
        if use_axis:
            fig.savefig(os.path.join(args.o, f"{method}_umap.pdf"), bbox_inches="tight")
        else:
            plt_umap_labels()
            fig.savefig(
                os.path.join(args.o, f"{method}_umap_no_axis.pdf"),
                bbox_inches="tight",
            )

    else:
        if use_axis:
            fig.savefig(
                os.path.join(args.o, f"{method}_latent.pdf"), bbox_inches="tight"
            )
        else:
            plt_umap_labels()
            fig.savefig(
                os.path.join(args.o, f"{method}_latent_no_axis.pdf"),
                bbox_inches="tight",
            )

    plt.close()


def main(args):
    dihedral_angles = np.linspace(0, 356.4, 100)
    input_files = os.listdir(args.result_path)
    os.makedirs(args.o, exist_ok=True)

    if "config.yaml" in input_files:
        umap_pkl = os.path.join(args.result_path, "analyze.19", "umap.pkl")
        umap_pkl = open(umap_pkl, "rb")
        umap_pkl = pickle.load(umap_pkl)
        plot_methods(args, "cryoDRGN", umap_pkl, dihedral_angles, is_umap=True)

    elif "config.pkl" in input_files:
        umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"

        umap_pkl = open(umap_pkl, "rb")
        umap_pkl = pickle.load(umap_pkl)
        # UMap
        plot_methods(args, "OPUS-DSD", umap_pkl, dihedral_angles, is_umap=True)

    elif "reordered_z.npy" in input_files:
        latent_path = os.path.join(args.result_path, args.method, "reordered_z.npy")
        latent_z = np.load(latent_path)

        umap_path = f"{args.result_path}/{args.method}/reordered_z_umap.npy"
        umap_pkl = analysis.run_umap(latent_z)  # v: latent space
        np.save(umap_path, umap_pkl)

        plot_methods(args, "RECOVAR", umap_pkl, dihedral_angles, is_umap=True)

    elif "job.json" in input_files:
        cryosparc_path, cryosparc_job = os.path.split(
            os.path.normpath(args.result_path)
        )
        with open(os.path.join(args.input_dir, "job.json")) as f:
            configs = json.load(f)

        if configs["type"] == "class_3D":
            file_pattern = "*.mrc"
            files = glob.glob(
                os.path.join(
                    args.result_path,
                    args.method,
                    "cls_" + str(args.num_classes),
                    file_pattern,
                )
            )
            pred_dir = sorted(files, key=volumes.natural_sort_key)
            cryosparc_num = os.path.split(pred_dir[0])[-1].split(".")[0].split("_")[3]
            print("cryosparc_num:", cryosparc_num)
            print("cryosparc_job:", cryosparc_job)
            path_for_label = os.path.join(
                args.result_path, f"{cryosparc_job}_{cryosparc_num}_particles.cs"
            )
            path_for_model = os.path.join(cryosparc_path, cryosparc_job)
            labels_3dcls, models_3dcls = parse_class_assignments(
                path_for_label, path_for_model, args.num_classes
            )
            plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04)

        elif args.method == "3dcls_abinit":
            file_pattern = "*.mrc"
            files = glob.glob(
                os.path.join(
                    args.result_path,
                    args.method,
                    "cls_" + str(args.num_classes),
                    file_pattern,
                )
            )
            pred_dir = sorted(files, key=volumes.natural_sort_key)
            cryosparc_job = pred_dir[0].split("/")[-1].split(".")[0].split("_")[0]
            print("cryosparc_job:", cryosparc_job)
            path_for_label = f"{args.cryosparc_path}/{cryosparc_job}/{cryosparc_job}_final_particles.cs"
            path_for_model = f"{args.cryosparc_path}/{cryosparc_job}/"
            labels_3dcls, models_3dcls = parse_class_abinit_assignments(
                path_for_label, path_for_model, args.num_classes
            )
            plot_3dcls(args, dihedral_angles, labels_3dcls, jitter=0.04)

        elif args.method == "3dva":
            path = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_particles.cs"

            x = np.load(path)
            v = np.empty((len(x), 3))  # component_0,1,2
            for i in range(3):
                v[:, i] = x[f"components_mode_{i}/value"]
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)
            # UMap
            umap_path = f"{args.o}/{args.method}/{args.method}_umap.npy"
            if not os.path.exists(umap_path):
                umap_latent = analysis.run_umap(v)  # v: latent space
                np.save(umap_path, umap_latent)
            else:
                umap_latent = np.load(umap_path)
            plot_methods(args, "3DVA", umap_latent, dihedral_angles, is_umap=True)

        elif args.method == "3dflex":
            path = f"{args.cryosparc_path}/{args.cryosparc_job_num}/{args.cryosparc_job_num}_latents_011200.cs"
            x = np.load(path)

            v = np.empty((len(x), 2))
            for i in range(2):
                v[:, i] = x[f"components_mode_{i}/value"]
            latent_path = f"{args.o}/{args.method}/{args.method}_latents.npy"
            np.save(latent_path, v)
            # Latent
            plot_methods(args, "3DFlex", v, dihedral_angles, is_umap=True)

    elif (
        "out" in input_files
        and os.path.isdir(os.path.join(args.input_dir, "out"))
        and "config.yaml" in os.listdir(os.path.join(args.input_dir, "out"))
    ):
        umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"
        umap_pkl = open(umap_pkl, "rb")
        umap_pkl = pickle.load(umap_pkl)
        plot_methods(args, "DRGN-AI", umap_pkl, dihedral_angles, is_umap=True)

    else:
        raise ValueError(
            f"Unrecognized output folder format found in `{args.result_path}`!"
            f"Does not match for any known methods: "
            "cryoDRGN, DRGN-AI, OPUS-DSD, 3dflex, 3DVA, RECOVAR"
        )


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
    print("done!")
