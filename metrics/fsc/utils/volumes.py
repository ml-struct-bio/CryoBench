"""Utility functions used across pipelines for calculating FSCs across conformations.

Many of these functions for calculating FSCs were originally copied from cryoDRGN v3.4.1
CryoBench methods depend on older versions of cryoDRGN that don't have these methods!

"""
import os
import subprocess
import time
import yaml
import re
from glob import glob
import logging
from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
import torch
from cryodrgn import fft, models, mrc

logger = logging.getLogger(__name__)


CHIMERAX_PATH = os.environ["CHIMERAX_PATH"]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def numfile_sortkey(s: str) -> list:
    """Split a filepath into a list that can be used to sort files by numeric order."""
    parts = re.split("([0-9]+)", s)
    parts[1::2] = map(int, parts[1::2])

    return parts


def get_volume_generator(
    config_path: str, checkpoint_path: str
) -> Callable[[torch.Tensor], np.ndarray]:
    """Create a latent space volume generator using a saved cryoDRGN model."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    norm = [float(x) for x in cfg["dataset_args"]["norm"]]
    model, lattice = models.HetOnlyVAE.load(cfg, checkpoint_path, device="cuda:0")
    model.eval()

    return lambda z: model.decoder.eval_volume(
        lattice.coords, lattice.D, lattice.extent, norm, z
    )


def align_volumes_multi(
    vol_paths: Union[str, list[str]],
    gt_paths: Union[str, list[str]],
    outdir: Optional[str] = None,
    flip: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    if isinstance(vol_paths, str):
        if os.path.isdir(vol_paths):
            matching_vols = sorted(
                glob(os.path.join(vol_paths, "*.mrc")), key=numfile_sortkey
            )
        else:
            raise ValueError(
                "Single value given for `vol_paths` must be a path to a directory "
                "containing .mrc volumes to be aligned!"
            )
    elif isinstance(vol_paths, list):
        matching_vols = vol_paths
    else:
        raise ValueError(
            f"Unrecognized type given for argument "
            f"`vol_paths`: {type(vol_paths).__name__} !"
        )

    if isinstance(gt_paths, str):
        if os.path.isdir(gt_paths):
            gt_vols = sorted(glob(os.path.join(gt_paths, "*.mrc")), key=numfile_sortkey)
        else:
            raise ValueError(
                "Single value given for `gt_paths` must be a path to a directory "
                "containing .mrc volumes to be aligned against!"
            )
    elif isinstance(gt_paths, list):
        gt_vols = gt_paths
    else:
        raise ValueError(
            f"Unrecognized type given for argument "
            f"`gt_paths`: {type(gt_paths).__name__} !"
        )

    if outdir is None:
        if isinstance(vol_paths, str):
            outdir = vol_paths
        else:
            outdir = os.path.dirname(vol_paths[0])

    aligndir = "flipped_aligned" if flip else "aligned"
    os.makedirs(os.path.join(outdir, aligndir), exist_ok=True)
    flip_str = "--flip" if flip else ""
    seed_str = f" --seed {random_seed}" if random_seed is not None else ""
    align_jobs = list()

    for i, file_path in enumerate(matching_vols):
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        new_filename = base_filename + ".mrc"
        destination_path = os.path.join(outdir, aligndir, new_filename)
        ref_path = gt_vols[i]
        tmp_file = os.path.join(outdir, aligndir, f"temp_{i:03d}.txt")

        align_cmd = (
            f"sbatch -t 61 -J align_{i} -o {tmp_file} --wrap='{CHIMERAX_PATH} --nogui "
            f"--script \" {os.path.join(ROOT_DIR, 'utils', 'align.py')} {ref_path} "
            f"{os.path.join(outdir, new_filename)} -o {destination_path} "
            f"{flip_str}{seed_str} -f {tmp_file} \" ' "
        )
        if i % 20 == 0:
            print(align_cmd)

        align_out = subprocess.run(align_cmd, shell=True, capture_output=True)
        assert align_out.stderr.decode("utf8") == "", align_out.stderr.decode("utf8")
        align_out = align_out.stdout.decode("utf8")
        align_jobs.append(align_out.strip().split("Submitted batch job ")[1])

    jobs_left = len(align_jobs)
    while jobs_left > 0:
        if jobs_left > 1:
            print(f"Waiting for {jobs_left} volume alignment jobs to finish...")
        else:
            print(
                f"Waiting for one volume alignment job "
                f"(Slurm ID: {align_jobs[0]}) to finish..."
            )

        time.sleep(max(10, jobs_left / 1.7))
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
    vol_paths: Union[str, list[str]],
    gt_paths: Union[str, list[str]],
    outdir: Optional[str] = None,
    mask_file: Optional[str] = None,
    fast: int = 1,
    overwrite: bool = False,
    vol_fl_function: Callable[[int], str] = lambda i: f"vol_{i:03d}",
) -> dict[int, pd.DataFrame]:
    """Calculate FSC curves across conformations compared to ground truth volumes."""

    if isinstance(gt_paths, str):
        if os.path.isdir(gt_paths):
            gt_vols = sorted(glob(os.path.join(gt_paths, "*.mrc")), key=numfile_sortkey)
        else:
            raise ValueError(
                "Single value given for `gt_paths` must be a path to a directory "
                "containing .mrc volumes to be aligned against!"
            )
    elif isinstance(gt_paths, list):
        gt_vols = gt_paths
    else:
        raise ValueError(
            f"Unrecognized type given for argument "
            f"`gt_paths`: {type(gt_paths).__name__} !"
        )

    if isinstance(vol_paths, str):
        if os.path.isdir(vol_paths):
            vol_files = [
                os.path.join(vol_paths, f"{vol_fl_function(i)}.mrc")
                for i in range(len(gt_vols))
            ]
        else:
            raise ValueError(
                "Single value given for `vol_paths` must be a path to a directory "
                "containing .mrc volumes to be aligned!"
            )
    elif isinstance(vol_paths, list):
        vol_files = vol_paths
    else:
        raise ValueError(
            f"Unrecognized type given for argument "
            f"`vol_paths`: {type(vol_paths).__name__} !"
        )

    if mask_file is not None:
        outlbl = f"fsc_{os.path.splitext(os.path.basename(mask_file))[0]}"
    else:
        outlbl = "fsc_no_mask"

    if outdir is None:
        if isinstance(vol_paths, str):
            outdir = vol_paths
        else:
            outdir = os.path.dirname(vol_paths[0])

    os.makedirs(os.path.join(outdir, outlbl), exist_ok=True)
    fsc_curves = dict()
    for ii, gt_volfile in enumerate(gt_vols):
        if ii % fast != 0:
            continue

        out_fsc = os.path.join(outdir, outlbl, f"{ii:03d}.txt")
        vol1 = torch.tensor(mrc.parse_mrc(gt_volfile)[0])
        vol2 = torch.tensor(mrc.parse_mrc(vol_files[ii])[0])

        if os.path.exists(out_fsc) and not overwrite:
            if ii % 20 == 0:
                logger.info(f"FSC {ii} exists, loading from file...")
            fsc_curves[ii] = pd.read_csv(out_fsc, sep=" ")
        else:
            fsc_curves[ii] = get_fsc_curve(vol1, vol2, mask_file)
            if ii % 20 == 0:
                logger.info(f"Saving FSC {ii} values to {out_fsc}")
            fsc_curves[ii].round(7).clip(0, 1).to_csv(
                out_fsc, sep=" ", header=True, index=False
            )

    # Print summary statistics on max resolutions satisfying particular FSC thresholds
    fsc143 = [get_fsc_cutoff(x, 0.143) for x in fsc_curves.values()]
    logger.info(
        f"cryoDRGN FSC=0.143  —  "
        f"Mean: {np.mean(fsc143):.4g} \t Median {np.median(fsc143):.4g}"
    )
    fsc5 = [get_fsc_cutoff(x, 0.5) for x in fsc_curves.values()]
    logger.info(
        f"cryoDRGN FSC=0.5    —  "
        f"Mean: {np.mean(fsc5):.4g} \t Median {np.median(fsc5):.4g}"
    )

    return fsc_curves
