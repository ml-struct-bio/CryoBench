import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import utils


def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fscdir",
        type=os.path.abspath,
        help="Input directory containing outputs of FSC per conformation analysis.",
    )
    parser.add_argument("--Apix", type=float, default=3.0, help="pixel size")

    return parser


def main(args: argparse.Namespace) -> None:
    auc_lst = []

    for sublbl in ["fsc", "fsc_flipped", "fsc_no_mask", "fsc_flipped_no_mask"]:
        subdir = os.path.join(args.fscdir, sublbl)
        if not os.path.isdir(subdir):
            continue

        fsc_files = sorted(
            glob(os.path.join(subdir, "*.txt")), key=utils.numfile_sortkey
        )
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
            "auc:"
            + str(round(auc_avg_np, 3))
            + "+-"
            + str(round(auc_std_np, 3))
            + "/med:"
            + str(round(auc_med_np, 3)),
            fontsize=15,
        )

        for i in range(len(auc_total_np)):
            print(f"{i}: AUC {auc_total_np[i]}")
        print(f"AUC_avg: {auc_avg_np}, std: {auc_std_np}, AUC_med: {auc_med_np}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.fscdir, f"fsc_{sublbl}.png"),
            dpi=1200,
            bbox_inches="tight",
        )
        print("plot saved!")


if __name__ == "__main__":
    main(create_args().parse_args())
