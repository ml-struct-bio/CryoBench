"""Calculate FSCs between conformations matched across DRGN-AI model latent spaces.

Example usage
-------------
$ python metrics/per_conf_fsc/drgnai.py results/drgnai_fixed \
            --epoch 19 --Apix 3.0 -o output/drgnai_fixed --gt-dir ./gt_vols \
            --mask ./mask.mrc --num-imgs 1000 --num-vols 100

"""
import argparse
import os
import logging
import numpy as np
import torch
from metrics.utils import utils
from metrics.per_conf_fsc.utils import interface
import cryodrgn.utils
from cryodrgnai.analyze import VolumeGenerator
from cryodrgnai.lattice import Lattice
from cryodrgnai import models

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by DRGN-AI."""

    cfg_file = os.path.join(args.input_dir, "out", "drgnai-configs.yaml")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find DRGN-AI configuration parameter file 'out/config.pkl' "
            f"in given folder {args.input_dir=} — is this a DRGN-AI output folder?"
        )
    weights_fl = os.path.join(args.input_dir, "out", f"weights.{args.epoch}.pkl")
    if not os.path.exists(weights_fl):
        raise ValueError(
            f"Could not find DRGN-AI model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.input_dir, "out", f"conf.{args.epoch}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find drgnAI latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did the model finishing running?"
        )

    logger.info(f"Putting output under: {args.outdir} ...")
    voldir = os.path.join(args.outdir, "vols")
    os.makedirs(voldir, exist_ok=True)
    z = cryodrgn.utils.load_pkl(z_path)
    num_imgs = int(args.num_imgs) if z.shape[0] == 100000 else "ribo"
    nearest_z_array = utils.get_nearest_z_array(z, args.num_vols, num_imgs)

    configs = cryodrgn.utils.load_yaml(cfg_file)
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
            vol_generator.gen_volumes(voldir, nearest_z_array)

    if args.calc_fsc_vals:
        utils.get_fsc_curves(
            args.outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
