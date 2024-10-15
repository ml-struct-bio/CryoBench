"""Calculate FSCs between conformations matched across cryoDRGN model latent spaces.

Example usage
-------------
$ python metrics/per_conf_fsc/drgnai_fixed.py results/drgnai_fixed \
                                       --epoch 19 --Apix 3.0 \
                                       -o output --gt-dir ./gt_vols --mask ./mask.mrc \
                                       --num-imgs 1000 --num-vols 100

"""
import argparse
import os
import subprocess
import logging
import numpy as np
import interface
import utils
import cryodrgn.utils

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    if args.method is None:
        logger.info('No method label specified, using "drgnai_fixed" as default...')
        method_lbl = "drgnai_fixed"
    else:
        method_lbl = str(args.method)

    z_path = os.path.join(args.input_dir, "out", f"conf.{args.epoch}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find cryoDRGN latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did model finishing running?"
        )

    outdir = str(os.path.join(args.o, method_lbl, "per_conf_fsc"))
    logger.info(f"Putting output under: {outdir} ...")
    z = cryodrgn.utils.load_pkl(z_path)
    gt = np.repeat(np.arange(0, args.num_vols), args.num_imgs)
    assert len(gt) == len(z)
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)
    nearest_z_array = utils.get_nearest_z_array(z, args.num_vols, args.num_imgs)

    out_zfile = os.path.join(outdir, "zfile.txt")
    logger.info(out_zfile)
    cmd = "CUDA_VISIBLE_DEVICES={} drgnai analyze {} --volume-metrics --z-values-txt {} --epoch {} --invert -o {}/{}/per_conf_fsc/vols --Apix {}".format(
        args.cuda_device,
        args.input_dir,
        out_zfile,
        args.epoch,
        args.o,
        method_lbl,
        args.Apix,
    )
    logger.info(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    if args.calc_fsc_vals:
        utils.get_fsc_curves(
            outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
