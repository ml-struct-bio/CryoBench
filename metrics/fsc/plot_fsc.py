"""Visualize FSCs between conformations matched across a volume model's latent space.

Example usage
-------------
$ python metrics/fsc/plot_fsc.py cryobench_output/

"""
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
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
        auc_lst = list()

        freq = np.arange(1, 6) * 0.1
        res = ["1/{:.1f}".format(val) for val in ((1 / freq) * args.Apix)]
        res_text = res

        for i, fsc_file in enumerate(fsc_files):
            fsc = pd.read_csv(fsc_file, sep=" ")
            plt.plot(fsc.pixres, fsc.fsc, label=i)
            plt.xticks(np.arange(1, 6) * 0.1, res_text, fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel("1/resolution (1/Å)", fontsize=20)
            plt.ylabel("Fourier shell correlation", fontsize=20)
            auc_lst.append(auc(fsc.pixres, fsc.fsc))

        plt.ylim((0, 1))
        auc_total_np = np.array(auc_lst)
        auc_avg_np = np.nanmean(auc_total_np)
        auc_std_np = np.nanstd(auc_total_np)
        auc_med_np = np.nanmedian(auc_total_np, 0)
        plt.title(
            f"AUC: {auc_avg_np:.3f}\u00B1{auc_std_np:.3f}; median: {auc_med_np:.3f}",
            fontsize=15,
        )

        for i in range(len(auc_total_np)):
            print(f"{i}: AUC {auc_total_np[i]}")
        print(f"AUC_avg: {auc_avg_np}, std: {auc_std_np}, AUC_med: {auc_med_np}")

        plt.tight_layout()
        pltfile = os.path.join(args.outdir, f"{fsc_lbl}.png")
        plt.savefig(pltfile, dpi=1200, bbox_inches="tight")
        print(f"{fsc_lbl} plot saved!")
        plt.clf()


if __name__ == "__main__":
    main(create_args().parse_args())
