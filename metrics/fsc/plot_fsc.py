"""Visualize FSCs between conformations matched across a volume model's latent space.

Example usage
-------------
$ python metrics/fsc/plot_fsc.py cryobench_output/

"""
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from utils import volumes


def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outdir",
        type=os.path.abspath,
        help="Input directory containing outputs of FSC per conformation analysis.",
    )
    parser.add_argument("--Apix", type=float, default=3.0, help="pixel size")

    return parser


def main(args: argparse.Namespace) -> None:
    fsc_dirs = [
        d
        for d in os.listdir(args.outdir)
        if d.startswith("fsc_") and os.path.isdir(os.path.join(args.outdir, d))
    ]

    for fsc_lbl in fsc_dirs:
        subdir = os.path.join(args.outdir, fsc_lbl)
        fsc_files = sorted(
            glob(os.path.join(subdir, "*.txt")), key=volumes.numfile_sortkey
        )
        fsc_list = list()
        auc_lst = list()

        freq = np.arange(1, 6) * 0.1
        res = ["1/{:.1f}".format(val) for val in ((1 / freq) * args.Apix)]
        res_text = res

        for i, fsc_file in enumerate(fsc_files):
            fsc = pd.read_csv(fsc_file, sep=" ")
            plt.plot(fsc.pixres, fsc.fsc, label=i)
            plt.xticks(np.arange(1, 6) * 0.1, res_text, fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel("Spatial frequency (1/Å)", fontsize=20)
            plt.ylabel("Fourier shell correlation", fontsize=20)

            fsc_list.append(fsc.assign(vol=i))
            auc_lst.append(auc(fsc.pixres, fsc.fsc))

        plt.ylim((0, 1))
        auc_avg_np = np.nanmean(auc_lst)
        auc_std_np = np.nanstd(auc_lst)
        auc_med_np = np.nanmedian(auc_lst, 0)
        plt.title(
            f"AUC: {auc_avg_np:.3f}\u00B1{auc_std_np:.3f}; median: {auc_med_np:.3f}",
            fontsize=15,
        )

        auc_str = str()
        for i, auc_val in enumerate(auc_lst):
            auc_str += f"{i:>7}: AUC {auc_val:.4f}"
            if i < (len(auc_lst) - 1) and i % 4 == 3:
                auc_str += "\n"

        auc_str += "\n-------------------------------------------------\n"
        auc_str += f"AUC_avg: {auc_avg_np:.5f}, std: {auc_std_np:.5f}, "
        auc_str += f"AUC_med: {auc_med_np:.5f}"
        print(auc_str)

        plt.tight_layout()
        pltfile = os.path.join(args.outdir, f"{fsc_lbl}.png")
        plt.savefig(pltfile, dpi=1200, bbox_inches="tight")
        plt.clf()

        fsc_df = pd.concat(fsc_list).reset_index(drop=True)
        sns.set_style("ticks")
        sns.set_palette(sns.color_palette("muted"))
        palette = sns.color_palette("GnBu_r", n_colors=100)
        palette = ["red"] + palette[:-1]
        g = sns.lineplot(data=fsc_df, x="pixres", y="fsc")
        plt.xticks(np.arange(1, 6) * 0.1, res_text, fontsize=15)
        g.figure.axes[0].set(xlabel="Spatial frequency (1/Å)")
        g.figure.axes[0].set(ylabel="Fourier shell correlation")
        g.figure.axes[0].set(xlim=(0, 0.5))
        g.figure.axes[0].set(ylim=(0, 1.0))
        plt.hlines(xmin=0, xmax=0.5, y=0.5, color="k", linestyle="--", linewidth=1)
        plt.grid(True)
        plt.savefig(
            os.path.join(args.outdir, f"{fsc_lbl}_means.png"),
            dpi=1200,
            bbox_inches="tight",
        )
        plt.clf()

        print(f"`{fsc_lbl}` plots saved!\n")


if __name__ == "__main__":
    main(create_args().parse_args())