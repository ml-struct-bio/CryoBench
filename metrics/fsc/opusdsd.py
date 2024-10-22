"""Calculate FSCs between conformations matched across opusDSD model latent spaces.

See github.com/alncat/opusDSD for the source code for this method, and
www.nature.com/articles/s41592-023-02031-6 for its publication.

Example usage
-------------
$ python metrics/per_conf_fsc/opusdsd.py results/opusdsd \
            --epoch 19 --Apix 3.0 -o output --gt-dir ./gt_vols --mask ./mask.mrc \
            --num-imgs 1000 --num-vols 100

# We sometimes need to pad the opusDSD volumes to a larger box size
$ python metrics/per_conf_fsc/opusdsd.py results/opusdsd \
            --epoch 19 --Apix 3.0 -o output --gt-dir ./gt_vols --mask ./mask.mrc \
            --num-imgs 1000 --num-vols 100 -D 256

"""
import argparse
import os
import subprocess
from glob import glob
import logging
import numpy as np
import torch
from metrics.utils import utils
from metrics.per_conf_fsc.utils import interface

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-D", default=128, type=int)

    return parser


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by opusDSD."""

    cfg_file = os.path.join(args.input_dir, "config.pkl")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find opusDSD config file {cfg_file} "
            f"— is {args.input_dir=} a folder opusDSD output folder?"
        )
    weights_fl = os.path.join(args.input_dir, f"weights.{args.epoch}.pkl")
    if not os.path.exists(weights_fl):
        raise ValueError(
            f"Could not find opusDSD model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.input_dir, f"z.{args.epoch}.pkl")
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
    nearest_z_array = utils.get_nearest_z_array(z, args.num_vols, num_imgs)

    eval_vol_cmd = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
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

    logger.info(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    utils.pad_mrc_vols(sorted(glob(os.path.join(voldir, "*.mrc"))), args.D)

    if args.calc_fsc_vals:
        utils.get_fsc_curves(
            args.outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=lambda i: f"reference{i}.mrc",
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
