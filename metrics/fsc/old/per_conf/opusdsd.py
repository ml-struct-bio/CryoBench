"""Calculate FSCs between conformations matched across an OPUS-DSD model latent space.

See github.com/alncat/opusDSD for the source code for this method, and
www.nature.com/articles/s41592-023-02031-6 for its publication.

Example usage
-------------
$ python metrics/fsc/old/per_conf/opusdsd.py opusdsd-outputs/001_base/ \
            --epoch 19 -o opusdsd-outputs/001_base/cryobench.10/ \
            --gt-dir IgG-1D/vols/128_org/ --mask IgG-1D/init_mask/mask.mrc \
            --num-imgs 1000 --num-vols 100 --Apix=3.0

# We sometimes need to pad the opusDSD volumes to a larger box size
$ python metrics/fsc/old/per_conf/opusdsd.py opusdsd-outputs/001_base/ \
            --epoch 19 -o opusdsd-outputs/001_base/cryobench.10/ \
            --gt-dir IgG-1D/vols/128_org/ --mask IgG-1D/init_mask/mask.mrc \
            --num-imgs 1000 --num-vols 100 --Apix=3.0 -D 256

"""
import os
import sys
import argparse
import subprocess
from glob import glob
import logging
import numpy as np
import torch

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOTDIR, "fsc"))
from utils import volumes, conformations, interface

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-D", default=128, type=int)

    return parser


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by OPUS-DSD."""

    cfg_file = os.path.join(args.input_dir, "config.pkl")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find opusDSD config file {cfg_file} "
            f"— is {args.input_dir=} a folder opusDSD output folder?"
        )

    epoch_str = "" if args.epoch == -1 else f".{args.epoch}"
    weights_fl = os.path.join(args.input_dir, f"weights{epoch_str}.pkl")
    if not os.path.exists(weights_fl):
        raise ValueError(
            f"Could not find opusDSD model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.input_dir, f"z{epoch_str}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find opusDSD latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )

    logger.info(f"Putting output under: {args.outdir} ...")
    voldir = os.path.join(args.outdir, "vols")
    os.makedirs(voldir, exist_ok=True)
    z = torch.load(z_path)["mu"].cpu().numpy()
    num_imgs = int(args.num_imgs) if z.shape[0] == 100000 else "ribo"
    nearest_z_array = conformations.get_nearest_z_array(z, args.num_vols, num_imgs)

    eval_vol_cmd = os.path.join(
        ROOTDIR,
        "methods",
        "opusDSD",
        "cryodrgn",
        "commands",
        "eval_vol.py",
    )
    out_zfile = os.path.join(args.outdir, "zfile.txt")
    logger.info(out_zfile)
    cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_device}; "
    cmd += f"python {eval_vol_cmd} --load {weights_fl} -c {cfg_file} "
    cmd += f"--zfile {out_zfile} -o {voldir} --Apix {args.Apix}; "

    logging.basicConfig(level=logging.INFO)
    logger.info(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(voldir, args.gt_dir)

    conformations.pad_mrc_vols(sorted(glob(os.path.join(voldir, "*.mrc"))), args.D)
    if args.align_vols:
        volumes.align_volumes_multi(voldir, args.gt_dir)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            voldir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=lambda i: f"reference{i}",
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                os.path.join(voldir, "aligned"),
                args.gt_dir,
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
                vol_fl_function=lambda i: f"reference{i}",
            )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
