import argparse
import numpy as np
import os
import re
from glob import glob
import subprocess
import logging
from cryodrgn import analysis, mrcfile, utils
from cryodrgn.commands_utils.fsc import get_fsc_curve
import torch

logger = logging.getLogger(__name__)


def add_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="dir contains weights, config, z")
    parser.add_argument("-o", help="Output directory")
    parser.add_argument(
        "--epoch", default=19, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--num-vols",
        default=100,
        type=int,
        help="Number of total reconstructed volumes",
    )
    parser.add_argument("--Apix", default=3.0, type=float)
    parser.add_argument(
        "--num-imgs",
        default=1000,
        type=int,
        help="Number of images per model (structure)",
    )
    parser.add_argument(
        "--method", type=str, help="type of methods (Each method folder name)"
    )
    parser.add_argument(
        "--mask", default=None, help="Use mask to compute the masked metric"
    )
    parser.add_argument("--gt-dir", help="Directory of gt volumes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", type=int, default=1)
    parser.add_argument("--cuda-device", default=0, type=int)

    return parser


def get_cutoff(fsc, t):
    w = np.where(fsc[:, 1] < t)
    logger.info(w)
    if len(w[0]) >= 1:
        x = fsc[:, 0][w]
        return 1 / x[0]
    else:
        return 2


def numfile_sort(s):
    # Convert the string to a list of text and numbers
    parts = re.split("([0-9]+)", s)

    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])

    return parts


def main(args):
    logger.info(f"method: {args.method}")

    config = os.path.join(args.input_dir, "config.yaml")
    if not os.path.exists(config):
        raise ValueError(
            f"Could not find cryoDRGN config file {config} "
            f"— is {args.input_dir=} a folder cryoDRGN output folder?"
        )
    weights = os.path.join(args.input_dir, f"weights.{args.epoch}.pkl")
    if not os.path.exists(weights):
        raise ValueError(
            f"Could not find cryoDRGN model weights for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did model finishing running?"
        )
    z_path = os.path.join(args.input_dir, f"z.{args.epoch}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find cryoDRGN latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.input_dir=} — did model finishing running?"
        )

    z = utils.load_pkl(z_path)
    gt = np.repeat(np.arange(0, args.num_vols), args.num_imgs)
    assert len(gt) == len(z)

    z_lst = []
    z_mean_lst = []
    for i in range(args.num_vols):
        z_nth = z[i * args.num_imgs : (i + 1) * args.num_imgs]
        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)
    nearest_z_lst = []
    centers_ind_lst = []
    for i in range(args.num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind)
    nearest_z_array = np.array(nearest_z_lst)

    gt_volfiles = sorted(glob(os.path.join(args.gt_dir, "*.mrc")), key=numfile_sort)
    outdir = str(os.path.join(args.o, args.method, "per_conf_fsc"))
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)

    # Generate cdrgn volumes
    out_zfile = os.path.join(outdir, "zfile.txt")
    logger.info(out_zfile)
    cmd = "CUDA_VISIBLE_DEVICES={} cryodrgn eval_vol {} -c {} --zfile {} -o {}/{}/per_conf_fsc/vols --Apix {}".format(
        args.cuda_device, weights, config, out_zfile, args.o, args.method, args.Apix
    )

    logger.info(cmd)
    if os.path.exists(out_zfile) and not args.overwrite:
        logger.info("Z file exists, skipping...")
    else:
        if not args.dry_run:
            np.savetxt(out_zfile, nearest_z_array)
            subprocess.check_call(cmd, shell=True)

    # Compute FSC cdrgn
    outlbl = "fsc" if args.mask is not None else "fsc_no_mask"
    os.makedirs(os.path.join(outdir, outlbl), exist_ok=True)
    for ii, gt_volfile in enumerate(gt_volfiles):
        if ii % args.fast != 0:
            continue

        out_fsc = os.path.join(outdir, outlbl, f"{ii}.txt")
        vol_file = os.path.join(outdir, "vols", f"vol_{ii:03d}.mrc")
        vol1 = torch.tensor(mrcfile.parse_mrc(gt_volfile)[0])
        vol2 = torch.tensor(mrcfile.parse_mrc(vol_file)[0])
        maskvol = None
        if args.mask is not None:
            maskvol = torch.tensor(mrcfile.parse_mrc(args.mask)[0])

        if os.path.exists(out_fsc) and not args.overwrite:
            logger.info("FSC exists, skipping...")
        else:
            get_fsc_curve(vol1, vol2, maskvol, out_file=out_fsc)

    # Summary statistics
    fsc = [np.loadtxt(x) for x in glob(os.path.join(outdir, outlbl, "*.txt"))]
    fsc143 = [get_cutoff(x, 0.143) for x in fsc]
    fsc5 = [get_cutoff(x, 0.5) for x in fsc]
    logger.info("cryoDRGN FSC=0.143")
    logger.info("Mean: {}".format(np.mean(fsc143)))
    logger.info("Median: {}".format(np.median(fsc143)))
    logger.info("cryoDRGN FSC=0.5")
    logger.info("Mean: {}".format(np.mean(fsc5)))
    logger.info("Median: {}".format(np.median(fsc5)))


if __name__ == "__main__":
    main(add_args().parse_args())
