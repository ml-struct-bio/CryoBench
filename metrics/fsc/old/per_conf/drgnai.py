"""Calculate FSCs between conformations matched across DRGN-AI model latent spaces.

Example usage
-------------
$ python metrics/per_conf/drgnai.py results/drgnai_fixed \
            --epoch 19 --Apix 3.0 -o output/drgnai_fixed --gt-dir ./gt_vols \
            --mask ./mask.mrc --num-imgs 1000 --num-vols 100

"""
import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
import cryodrgn.utils
from cryodrgnai.analyze import VolumeGenerator
from cryodrgnai.lattice import Lattice
from cryodrgnai import models

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from utils import volumes, conformations, interface

logging.basicConfig(
    level=logging.INFO,
    format="(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by DRGN-AI."""

    cfg_file = os.path.join(args.input_dir, "out", "drgnai-configs.yaml")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find DRGN-AI configuration parameter file 'out/config.pkl' "
            f"in given folder {args.input_dir=} — is this a DRGN-AI output folder?"
        )

    epoch_str = "" if args.epoch == -1 else f".{args.epoch}"
    weights_fl = os.path.join(args.input_dir, "out", f"weights{epoch_str}.pkl")
    if not os.path.exists(weights_fl):
        raise ValueError(
            f"Could not find DRGN-AI model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.input_dir, "out", f"conf{epoch_str}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find DRGN-AI latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )

    logger.info(f"Putting output under: {args.outdir} ...")
    os.makedirs(args.outdir, exist_ok=True)
    z = cryodrgn.utils.load_pkl(z_path)
    num_imgs = int(args.num_imgs) if z.shape[0] == 100000 else "ribo"
    nearest_z_array = conformations.get_nearest_z_array(z, args.num_vols, num_imgs)

    with open(cfg_file, "r") as f:
        configs = yaml.safe_load(f)
    checkpoint = torch.load(weights_fl)
    hypervolume_params = checkpoint["hypervolume_params"]
    hypervolume = models.HyperVolume(**hypervolume_params)
    hypervolume.load_state_dict(checkpoint["hypervolume_state_dict"])
    hypervolume.eval()
    hypervolume.to(args.cuda_device)

    lattice = Lattice(
        checkpoint["hypervolume_params"]["resolution"],
        extent=0.5,
        device=args.cuda_device,
    )
    z_dim = checkpoint["hypervolume_params"]["z_dim"]
    radius_mask = (
        checkpoint["output_mask_radius"] if "output_mask_radius" in checkpoint else None
    )
    vol_generator = VolumeGenerator(
        hypervolume,
        lattice,
        z_dim,
        True,
        radius_mask,
        data_norm=(configs["data_norm_mean"], configs["data_norm_std"]),
    )

    out_zfile = os.path.join(args.outdir, "zfile.txt")
    logger.info(out_zfile)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            vol_generator.gen_volumes(args.outdir, nearest_z_array)

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(args.outdir, args.gt_dir)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            args.outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                os.path.join(args.outdir, "aligned"),
                args.gt_dir,
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
