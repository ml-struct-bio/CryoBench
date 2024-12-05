"""Visualize FSCs between conformations matched across a volume model's latent space.

The CryoBench output directory used as the argument to this script should contain at
least one folder with the prefix "fsc_" already produced by a FSC analysis script such
as `cdrgn.py`.

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
        for i, fsc_file in enumerate(fsc_files):
            fsc = pd.read_csv(fsc_file, sep=" ")
            plt.plot(fsc.pixres, fsc.fsc, label=i)
            fsc_list.append(fsc.assign(vol=i))
            auc_lst.append(auc(fsc.pixres, fsc.fsc))

        freq = np.arange(0, 6) * 0.1
        res_text = [
            f"{int(k / (2. * args.Apix) * 1000)/1000. if k > 0 else 'DC'}"
            for k in np.linspace(0, 1, 6)
        ]
        plt.xlim((0, 0.5))
        plt.ylim((0, 1))
        plt.grid(True, linewidth=0.53)
        plt.xticks(freq, res_text, fontsize=11)
        plt.yticks(fontsize=11)
        plt.xlabel("Spatial frequency (1/Å)", fontsize=14, weight="semibold")
        plt.ylabel("Fourier shell correlation", fontsize=14, weight="semibold")

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
        g = sns.lineplot(data=fsc_df, x="pixres", y="fsc", color="red", ci="sd")

        plt.xticks(freq, res_text, fontsize=11)
        plt.yticks(fontsize=11)
        g.figure.axes[0].set(xlim=(0, 0.5))
        g.figure.axes[0].set(ylim=(0, 1.0))
        plt.grid(True, linewidth=0.53)
        plt.xlabel("Spatial frequency (1/Å)", fontsize=14, weight="semibold")
        plt.ylabel("Fourier shell correlation", fontsize=14, weight="semibold")

        pltfile = os.path.join(args.outdir, f"{fsc_lbl}_means.png")
        plt.savefig(pltfile, dpi=1200, bbox_inches="tight")
        plt.clf()

        print(f"`{fsc_lbl}` plots saved!\n")


if __name__ == "__main__":
    main(create_args().parse_args())
