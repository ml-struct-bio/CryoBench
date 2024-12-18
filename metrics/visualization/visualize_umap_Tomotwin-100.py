import numpy as np
from cryodrgn import analysis

import pickle
import os
import re
import argparse
import matplotlib.pyplot as plt

log = print

import glob


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, help="type of methods")
    parser.add_argument("--is_cryosparc", action="store_true", help="cryosparc or not")
    parser.add_argument("--num_classes", type=int, default=20, help="number of classes")
    parser.add_argument("--num_vols", type=int, default=100, help="number of classes")
    parser.add_argument("--num_imgs", type=int, default=1000, help="number of images")
    parser.add_argument(
        "-o",
        type=os.path.abspath,
        required=True,
        help="Output folder to save the UMAP plot",
    )
    parser.add_argument(
        "--result-path",
        type=os.path.abspath,
        required=True,
        help="umap & latent folder before method name (e.g. /scratch/gpfs/ZHONGE/mj7341/CryoBench/results/IgG-1D/snr0.01)",
    )
    parser.add_argument(
        "--cryosparc_path", type=os.path.abspath, help="cryosparc folder path"
    )
    parser.add_argument("--cryosparc_job_num", type=str, help="cryosparc job number")

    return parser


def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split("([0-9]+)", s)

    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])

    return parts


def plt_umap_labels():
    plt.xticks([])
    plt.yticks([])


def plot_methods(args, v, is_umap=True, use_axis=False):
    mass_for_class = [
        156,
        162,
        168,
        174,
        179,
        180,
        191,
        193,
        197,
        200,
        204,
        205,
        206,
        206,
        215,
        226,
        231,
        233,
        240,
        247,
        249,
        250,
        251,
        257,
        258,
        260,
        266,
        280,
        280,
        285,
        291,
        296,
        313,
        313,
        316,
        331,
        333,
        359,
        375,
        377,
        382,
        383,
        393,
        399,
        410,
        422,
        424,
        439,
        456,
        464,
        468,
        478,
        488,
        490,
        491,
        493,
        502,
        511,
        515,
        518,
        518,
        529,
        547,
        551,
        556,
        574,
        588,
        590,
        591,
        595,
        597,
        602,
        607,
        622,
        629,
        632,
        633,
        652,
        663,
        671,
        681,
        727,
        732,
        838,
        847,
        864,
        865,
        877,
        881,
        881,
        899,
        921,
        956,
        957,
        1014,
        1023,
        1023,
        1057,
        1066,
        1131,
    ]
    mass_for_classes = np.repeat(mass_for_class, args.num_imgs)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_dim = (0, 1)

    plt.scatter(
        v[:, plot_dim[0]],
        v[:, plot_dim[1]],
        alpha=0.1,
        s=1,
        c=mass_for_classes,
        cmap="rainbow",
        rasterized=True,
    )

    if is_umap:
        if use_axis:
            plt.savefig(
                f"{args.o}/{args.method}/{args.method}_umap.pdf", bbox_inches="tight"
            )
        else:
            plt_umap_labels()
            plt.savefig(
                f"{args.o}/{args.method}/{args.method}_umap_no_axis_rainbow.pdf",
                bbox_inches="tight",
            )
    else:
        if use_axis:
            plt.savefig(
                f"{args.o}/{args.method}/{args.method}_latent.pdf", bbox_inches="tight"
            )
        else:
            plt_umap_labels()
            plt.savefig(
                f"{args.o}/{args.method}/{args.method}_latent_no_axis_rainbow.pdf",
                bbox_inches="tight",
            )
    plt.close()


def main(args):
    if args.is_cryosparc:
        if args.method == "3dva":
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
            plot_methods(args, umap_latent, is_umap=True)

    else:
        if args.method == "cryodrgn":

            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"

            umap_pkl = open(umap_pkl, "rb")
            umap_pkl = pickle.load(umap_pkl)
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == "cryodrgn2":
            umap_pkl = f"{args.result_path}/{args.method}/analyze.29/umap.pkl"

            umap_pkl = open(umap_pkl, "rb")
            umap_pkl = pickle.load(umap_pkl)
            # UMap
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == "drgnai_fixed":
            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"

            umap_pkl = open(umap_pkl, "rb")
            umap_pkl = pickle.load(umap_pkl)
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == "drgnai_abinit":
            umap_pkl = f"{args.result_path}/{args.method}/out/analysis_100/umap.pkl"

            umap_pkl = open(umap_pkl, "rb")
            umap_pkl = pickle.load(umap_pkl)
            # UMap
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == "opus-dsd":
            umap_pkl = f"{args.result_path}/{args.method}/analyze.19/umap.pkl"

            umap_pkl = open(umap_pkl, "rb")
            umap_pkl = pickle.load(umap_pkl)
            # UMap
            plot_methods(args, umap_pkl, is_umap=True)

        elif args.method == "recovar":
            latent_path = os.path.join(args.result_path, args.method, "reordered_z.npy")
            latent_z = np.load(latent_path)

            umap_path = f"{args.result_path}/{args.method}/reordered_z_umap.npy"
            umap_pkl = analysis.run_umap(latent_z)  # v: latent space
            np.save(umap_path, umap_pkl)
            plot_methods(args, umap_pkl, is_umap=True)


if __name__ == "__main__":
    args = parse_args().parse_args()
    if not os.path.exists(args.o + "/" + args.method):
        os.makedirs(args.o + "/" + args.method)

    main(args)
    print("done!")
