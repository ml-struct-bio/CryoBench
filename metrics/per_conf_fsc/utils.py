"""Utility functions used across pipelines for calculating FSCs across conformations."""

import os
import re
from glob import glob
from collections.abc import Iterable
import logging
from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
import torch
from cryodrgn import analysis, fft, mrc

logger = logging.getLogger(__name__)


# Ribosembly number of images per Ribosembly structure (total 16 structures)
RIBOSEMBLY_NUM_IMGS = [
    9076,
    14378,
    23547,
    44366,
    30647,
    38500,
    3915,
    3980,
    12740,
    11975,
    17988,
    5001,
    35367,
    37448,
    40540,
    5772,
]


def get_fsc_cutoff(fsc_curve: pd.DataFrame, t: float) -> float:
    """Find the resolution at which the FSC curve first crosses a given threshold."""
    fsc_indx = np.where(fsc_curve.fsc < t)[0]
    return fsc_curve.pixres[fsc_indx[0]] ** -1 if len(fsc_indx) > 0 else 2.0


def numfile_sortkey(s):
    """Get the numeric part of a filepath that contains an integer in the file name."""

    # Split the filepath according to before, after, and the numeric part itself, and
    # then convert the numeric part to an integer object for proper numeric comparison
    parts = re.split("([0-9]+)", s)
    parts[1::2] = map(int, parts[1::2])

    return parts


def get_nearest_z_array(
    zmat: np.ndarray, num_vols: int, num_imgs: Union[int, str]
) -> np.ndarray:
    z_lst = []
    z_mean_lst = []
    for i in range(num_vols):
        if isinstance(num_imgs, int):
            z_nth = zmat[(i * num_imgs) : ((i + 1) * num_imgs)]
        elif num_imgs == "ribo":
            z_nth = zmat[
                sum(RIBOSEMBLY_NUM_IMGS[:i]) : sum(RIBOSEMBLY_NUM_IMGS[: (i + 1)])
            ]
        else:
            raise ValueError(f"{num_imgs=}")

        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)

    nearest_z_lst = []
    centers_ind_lst = []
    num_img_for_centers = 0
    for i in range(num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind + num_img_for_centers)

        if num_imgs == "ribo":
            num_img_for_centers += RIBOSEMBLY_NUM_IMGS[i]

    return np.array(nearest_z_lst)


def pad_mrc_vols(mrc_volfiles: Iterable[str], new_D: int) -> None:
    for mrc_file in mrc_volfiles:
        v, header = mrc.parse_mrc(mrc_file)
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
        apix = header.get_apix()
        xorg, yorg, zorg = header.get_origin()
        xorg -= apix * k
        yorg -= apix * j
        zorg -= apix * i

        mrc.write(mrc_file, new, header)


# These functions for calculating FSCs were originally copied from cryoDRGN v3.4.1
# CryoBench methods depend on older versions of cryoDRGN that don't have these methods!


def get_fftn_center_dists(box_size: int) -> np.array:
    """Get distances from the center (and hence the resolution) for FFT co-ordinates."""

    x = np.arange(-box_size // 2, box_size // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    dists = (coords**2).sum(-1) ** 0.5
    assert dists[box_size // 2, box_size // 2, box_size // 2] == 0.0

    return dists


def calculate_fsc(
    v1: Union[np.ndarray, torch.Tensor], v2: Union[np.ndarray, torch.Tensor]
) -> float:
    """Calculate the Fourier Shell Correlation between two complex vectors."""
    var = (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5

    return float((np.vdot(v1, v2) / var).real) if var else 1.0


def get_fsc_curve(
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    mask_file: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate the FSCs between two volumes across all available resolutions."""

    maskvol = None
    if mask_file is not None:
        maskvol = torch.tensor(mrc.parse_mrc(mask_file)[0])

    # Apply the given mask before applying the Fourier transform
    maskvol1 = vol1 * maskvol if maskvol is not None else vol1.clone()
    maskvol2 = vol2 * maskvol if maskvol is not None else vol2.clone()
    box_size = vol1.shape[0]
    dists = get_fftn_center_dists(box_size)
    maskvol1 = fft.fftn_center(maskvol1)
    maskvol2 = fft.fftn_center(maskvol2)

    prev_mask = np.zeros((box_size, box_size, box_size), dtype=bool)
    fsc = [1.0]
    for i in range(1, box_size // 2):
        mask = dists < i
        shell = np.where(mask & np.logical_not(prev_mask))
        fsc.append(calculate_fsc(maskvol1[shell], maskvol2[shell]))
        prev_mask = mask

    return pd.DataFrame(
        dict(pixres=np.arange(box_size // 2) / box_size, fsc=fsc), dtype=float
    )


def get_fsc_curves(
    outdir: str,
    gt_dir: str,
    vol_dir: Optional[str] = None,
    mask_file: Optional[str] = None,
    fast: int = 1,
    overwrite: bool = False,
    vol_fl_function: Callable[[int], str] = lambda i: f"vol_{i:03d}.mrc",
) -> None:
    """Calculate FSC curves across conformations compared to ground truth volumes."""

    gt_volfiles = sorted(glob(os.path.join(gt_dir, "*.mrc")), key=numfile_sortkey)
    if vol_dir is None:
        vol_dir = os.path.join(outdir, "vols")

    outlbl = "fsc" if mask_file is not None else "fsc_no_mask"
    os.makedirs(os.path.join(outdir, outlbl), exist_ok=True)
    fsc_curves = dict()

    for ii, gt_volfile in enumerate(gt_volfiles):
        if ii % fast != 0:
            continue

        out_fsc = os.path.join(outdir, outlbl, f"{ii}.txt")
        vol_file = os.path.join(vol_dir, vol_fl_function(ii))
        vol1 = torch.tensor(mrc.parse_mrc(gt_volfile)[0])
        vol2 = torch.tensor(mrc.parse_mrc(vol_file)[0])

        if os.path.exists(out_fsc) and not overwrite:
            if ii % 20 == 0:
                logger.info(f"FSC {ii} exists, loading from file...")
            fsc_curves[ii] = pd.read_csv(out_fsc, sep=" ")
        else:
            fsc_curves[ii] = get_fsc_curve(vol1, vol2, mask_file)
            if ii % 20 == 0:
                logger.info(f"Saving FSC {ii} values to {out_fsc}")
            fsc_curves[ii].to_csv(out_fsc, sep=" ", header=True, index=False)

    # Summary statistics
    fsc143 = [get_fsc_cutoff(x, 0.143) for x in fsc_curves.values()]
    fsc5 = [get_fsc_cutoff(x, 0.5) for x in fsc_curves.values()]
    logger.info("cryoDRGN FSC=0.143")
    logger.info("Mean: {}".format(np.mean(fsc143)))
    logger.info("Median: {}".format(np.median(fsc143)))
    logger.info("cryoDRGN FSC=0.5")
    logger.info("Mean: {}".format(np.mean(fsc5)))
    logger.info("Median: {}".format(np.median(fsc5)))
