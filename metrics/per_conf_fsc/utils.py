import os
import re
from glob import glob
from collections.abc import Iterable
import logging
from typing import Optional
import numpy as np
import pandas as pd
import torch
from cryodrgn import analysis, mrcfile
from cryodrgn.commands_utils.fsc import get_fsc_curve

logger = logging.getLogger(__name__)


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


def pad_mrc_vols(mrc_volfiles: Iterable[str], new_D: int) -> None:
    for mrc_file in mrc_volfiles:
        v, header = mrcfile.parse_mrc(mrc_file)
        x, y, z = v.shape
        assert new_D >= x
        assert new_D >= y
        assert new_D >= z

        new = np.zeros((new_D, new_D, new_D), dtype=np.float32)

        i = (new_D - x) // 2
        j = (new_D - y) // 2
        k = (new_D - z) // 2

        new[i : (i + x), j : (j + y), k : (k + z)] = v

        # adjust origin
        apix = header.apix
        xorg, yorg, zorg = header.origin
        xorg -= apix * k
        yorg -= apix * j
        zorg -= apix * i

        mrcfile.write_mrc(mrc_file, new)


def get_fsc_curves(
    outdir: str,
    gt_dir: str,
    mask_file: Optional[str] = None,
    fast: int = 1,
    overwrite: bool = False,
) -> None:
    gt_volfiles = sorted(glob(os.path.join(gt_dir, "*.mrc")), key=numfile_sort)
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)

    outlbl = "fsc" if mask_file is not None else "fsc_no_mask"
    os.makedirs(os.path.join(outdir, outlbl), exist_ok=True)
    fsc_curves = dict()

    for ii, gt_volfile in enumerate(gt_volfiles):
        if ii % fast != 0:
            continue

        out_fsc = os.path.join(outdir, outlbl, f"{ii}.txt")
        vol_file = os.path.join(outdir, "vols", f"vol_{ii:03d}.mrc")
        vol1 = torch.tensor(mrcfile.parse_mrc(gt_volfile)[0])
        vol2 = torch.tensor(mrcfile.parse_mrc(vol_file)[0])
        maskvol = None
        if mask_file is not None:
            maskvol = torch.tensor(mrcfile.parse_mrc(mask_file)[0])

        if os.path.exists(out_fsc) and not overwrite:
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
