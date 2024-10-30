"""Calculate FSCs between conformations matched across a cryoDRGN model latent space.

Example usage
-------------
$ python metrics/fsc/cdrgn.py results/cryodrgn --epoch 19 --Apix 3.0 \
                              -o output --gt-dir ./gt_vols --mask ./mask.mrc

# Also align output volumes to grund truth volumes with ChimeraX before computing FSCs
$ python metrics/fsc/cdrgn.py results/cryodrgn --epoch 19 --Apix 3.0 \
                              -o output --gt-dir ./gt_vols --mask ./mask.mrc

"""
import os
import sys
import argparse
import subprocess
import logging
import numpy as np
import cryodrgn.utils

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from utils import volumes, conformations, interface

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by cryoDRGN."""

    cfg_file = os.path.join(args.input_dir, "config.yaml")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find cryoDRGN config file {cfg_file} "
            f"— is {args.input_dir=} a folder cryoDRGN output folder?"
        )

    epoch_str = "" if args.epoch == -1 else f".{args.epoch}"
    weights_fl = os.path.join(args.input_dir, f"weights{epoch_str}.pkl")
    if not os.path.exists(weights_fl):
        raise ValueError(
            f"Could not find cryoDRGN model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.input_dir, f"z{epoch_str}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find cryoDRGN latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )

    logger.info(f"Putting output under: {args.outdir} ...")
    voldir = os.path.join(args.outdir, "vols")
    os.makedirs(voldir, exist_ok=True)
    z = cryodrgn.utils.load_pkl(z_path)
    num_imgs = int(args.num_imgs) if z.shape[0] == 100000 else "ribo"
    nearest_z_array = conformations.get_nearest_z_array(z, args.num_vols, num_imgs)

    # Generate volumes at these cryoDRGN latent space coordinates using the model tool
    out_zfile = os.path.join(args.outdir, "zfile.txt")
    logger.info(out_zfile)
    cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_device}; cryodrgn eval_vol {weights_fl} "
    cmd += f"-c {cfg_file} --zfile {out_zfile} -o {voldir} --Apix {args.Apix} "

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

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            voldir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                os.path.join(voldir, "aligned"),
                args.gt_dir,
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
