import argparse
import os
import subprocess
from glob import glob
import logging
import numpy as np
import torch
import utils
import interface

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-D", default=128, type=int)

    return parser


def main(args: argparse.Namespace) -> None:
    if args.method is None:
        logger.info('No method label specified, using "opusDSD" as default...')
        method_lbl = "opusDSD"
    else:
        method_lbl = str(args.method)

    config = os.path.join(args.input_dir, "config.pkl")
    if not os.path.exists(config):
        raise ValueError(
            f"Could not find opusDSD config file {config} "
            f"— is {args.input_dir=} a folder opusDSD output folder?"
        )
    weights = os.path.join(args.input_dir, f"weights.{args.epoch}.pkl")
    if not os.path.exists(weights):
        raise ValueError(
            f"Could not find opusDSD model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did model finishing running?"
        )
    z_path = os.path.join(args.input_dir, f"z.{args.epoch}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find opusDSD latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did model finishing running?"
        )

    outdir = str(os.path.join(args.o, method_lbl, "per_conf_fsc"))
    logger.info(f"Putting output under: {outdir} ...")
    z = torch.load(z_path)["mu"].cpu().numpy()
    gt = np.repeat(np.arange(0, args.num_vols), args.num_imgs)
    assert len(gt) == len(z)
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)
    nearest_z_array = utils.get_nearest_z_array(z, args.num_vols, args.num_imgs)

    out_zfile = os.path.join(outdir, "zfile.txt")
    logger.info(out_zfile)
    cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_device}; pip uninstall -y cryodrgn; "
    cmd += f"dsd eval_vol --load {weights} -c {config} --zfile {out_zfile} "
    cmd += f"-o {os.path.join(outdir, 'vols')} --Apix {args.Apix}; "
    cmd += "pip install cryodrgn; "

    logger.info(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    utils.pad_mrc_vols(sorted(glob(os.path.join(outdir, "vols", "*.mrc"))), args.D)

    if args.calc_fsc_vals:

        def vol_fl_function(i: int):
            return f"reference{i}.mrc"

        utils.get_fsc_curves(
            outdir, args.gt_dir, args.mask, args.fast, args.overwrite, vol_fl_function
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
