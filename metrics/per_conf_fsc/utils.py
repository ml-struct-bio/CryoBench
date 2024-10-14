import argparse
import os
import re
from glob import glob
import logging
import numpy as np
import pandas as pd
import torch
from cryodrgn import analysis, mrcfile
from cryodrgn.commands_utils.fsc import get_fsc_curve

logger = logging.getLogger(__name__)


def add_calc_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="dir contains weights, config, z")
    parser.add_argument("-o", help="Output directory")
    parser.add_argument(
        "--epoch", default=19, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--num-vols",
        default=100,
        type=int,
        help="Number of total reconstructed volumes",
    )
    parser.add_argument("--Apix", default=3.0, type=float)
    parser.add_argument(
        "--num-imgs",
        default=1000,
        type=int,
        help="Number of images per model (structure)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="override auto label for type of method, used for output subfolder",
    )
    parser.add_argument(
        "--mask",
        default=None,
        type=os.path.abspath,
        help="Path to mask .mrc to compute the masked metric",
    )
    parser.add_argument("--gt-dir", help="Directory of gt volumes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", type=int, default=1)
    parser.add_argument("--cuda-device", default=0, type=int)

    return parser


def get_fsc_cutoff(fsc_curve: pd.DataFrame, t: float) -> float:
    """Find the resolution at which the FSC curve first crosses a given threshold."""
    fsc_indx = np.where(fsc_curve.fsc < t)[0]
    return fsc_curve.pixres[fsc_indx[0]] ** -1 if len(fsc_indx) > 0 else 2.0


def numfile_sort(s):
    # Convert the string to a list of text and numbers
    parts = re.split("([0-9]+)", s)

    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])

    return parts


def get_nearest_z_array(zmat: np.ndarray, num_vols: int, num_imgs: int) -> np.ndarray:
    z_lst = []
    z_mean_lst = []
    for i in range(num_vols):
        z_nth = zmat[i * num_imgs : (i + 1) * num_imgs]
        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)

    nearest_z_lst = []
    centers_ind_lst = []
    for i in range(num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind)

    return np.array(nearest_z_lst)


def get_fsc_curves(outdir: str, args: argparse.Namespace) -> None:
    gt_volfiles = sorted(glob(os.path.join(args.gt_dir, "*.mrc")), key=numfile_sort)
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)

    outlbl = "fsc" if args.mask is not None else "fsc_no_mask"
    os.makedirs(os.path.join(outdir, outlbl), exist_ok=True)
    fsc_curves = dict()

    for ii, gt_volfile in enumerate(gt_volfiles):
        if ii % args.fast != 0:
            continue

        out_fsc = os.path.join(outdir, outlbl, f"{ii}.txt")
        vol_file = os.path.join(outdir, "vols", f"vol_{ii:03d}.mrc")
        vol1 = torch.tensor(mrcfile.parse_mrc(gt_volfile)[0])
        vol2 = torch.tensor(mrcfile.parse_mrc(vol_file)[0])
        maskvol = None
        if args.mask is not None:
            maskvol = torch.tensor(mrcfile.parse_mrc(args.mask)[0])

        if os.path.exists(out_fsc) and not args.overwrite:
            logger.info("FSC exists, loading from file...")
            fsc_curves[ii] = pd.read_csv(out_fsc, sep=" ")
        else:
            fsc_curves[ii] = get_fsc_curve(vol1, vol2, maskvol, out_file=out_fsc)

    # Summary statistics
    fsc143 = [get_fsc_cutoff(x, 0.143) for x in fsc_curves.values()]
    fsc5 = [get_fsc_cutoff(x, 0.5) for x in fsc_curves.values()]
    logger.info("cryoDRGN FSC=0.143")
    logger.info("Mean: {}".format(np.mean(fsc143)))
    logger.info("Median: {}".format(np.median(fsc143)))
    logger.info("cryoDRGN FSC=0.5")
    logger.info("Mean: {}".format(np.mean(fsc5)))
    logger.info("Median: {}".format(np.median(fsc5)))
