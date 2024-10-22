"""Utility functions used across pipelines for calculating FSCs across conformations.

Many of these functions for calculating FSCs were originally copied from cryoDRGN v3.4.1
CryoBench methods depend on older versions of cryoDRGN that don't have these methods!

"""
import os
import subprocess
import time
import re
from glob import glob
import logging
from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
import torch
from cryodrgn import fft, mrc

logger = logging.getLogger(__name__)


CHIMERAX_PATH = os.environ["CHIMERAX_PATH"]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def numfile_sortkey(s):
    """Get the numeric part of a filepath that contains an integer in the file name."""

    # Split the filepath according to before, after, and the numeric part itself, and
    # then convert the numeric part to an integer object for proper numeric comparison
    parts = re.split("([0-9]+)", s)
    parts[1::2] = map(int, parts[1::2])

    return parts


def align_volumes_multi(vol_dir: str, gt_dir: str) -> None:
    gt_vols = sorted(glob(os.path.join(gt_dir, "*.mrc")), key=numfile_sortkey)
    matching_vols = sorted(glob(os.path.join(vol_dir, "*.mrc")), key=numfile_sortkey)

    os.makedirs(os.path.join(vol_dir, "aligned"), exist_ok=True)
    os.makedirs(os.path.join(vol_dir, "flipped_aligned"), exist_ok=True)
    align_jobs = list()

    for i, file_path in enumerate(matching_vols):
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        new_filename = base_filename + ".mrc"
        destination_path = os.path.join(vol_dir, "aligned", new_filename)
        ref_path = gt_vols[i]
        tmp_file = os.path.join(vol_dir, "aligned", f"temp_{i:03d}.txt")

        align_cmd = (
            f"sbatch -t 10 -J align_{i} -o {tmp_file} --wrap='{CHIMERAX_PATH} --nogui "
            f"--script \" {os.path.join(ROOT_DIR, 'utils', 'align.py')} {ref_path} "
            f"{os.path.join(vol_dir, new_filename)} -o {destination_path} "
            f"-f {tmp_file} \" ' "
        )
        if i % 20 == 0:
            print(align_cmd)

        align_out = subprocess.run(align_cmd, shell=True, capture_output=True)
        assert align_out.stderr.decode("utf8") == "", align_out.stderr.decode("utf8")
        align_out = align_out.stdout.decode("utf8")
        align_jobs.append(align_out.strip().split("Submitted batch job ")[1])

    jobs_left = len(align_jobs)
    while jobs_left > 0:
        print(f"Waiting for {jobs_left} jobs to finish...")
        time.sleep(30)
        jobs_left = (
            subprocess.run(
                f"squeue -h -j {','.join(align_jobs)}", shell=True, capture_output=True
            )
            .stdout.decode("utf8")
            .count("\n")
        )


def get_fsc_cutoff(fsc_curve: pd.DataFrame, t: float) -> float:
    """Find the resolution at which the FSC curve first crosses a given threshold."""
    fsc_indx = np.where(fsc_curve.fsc < t)[0]
    return fsc_curve.pixres[fsc_indx[0]] ** -1 if len(fsc_indx) > 0 else 2.0


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
