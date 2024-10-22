"""Calculate FSCs between conformations matched across cryoSPARC 3D classifications.

Example usage
-------------
$ python metrics/per_conf_fsc/cryosparc_3dcls.py results/CS-cryobench/J5 \
            -o cBench/cBench-out_3Dcls/ --gt-dir vols/128_org/ --mask bproj_0.005.mrc \
            --num-classes 10

"""
import argparse
import os
import json
import numpy as np
from glob import glob
import logging
from metrics.utils import utils
from metrics.per_conf_fsc.utils import interface

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--num-classes", default=10, type=int)

    return parser


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across conformations produced by cryoSPARC."""

    cfg_file = os.path.join(args.input_dir, "job.json")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find cryoSPARC job info file `job.json` in given folder "
            f"{args.input_dir=} — did a cryoSPARC job use this as the output path?"
        )
    with open(cfg_file) as f:
        configs = json.load(f)

    if configs["type"] != "class_3D":
        raise ValueError(
            f"Given folder {args.input_dir=} contains cryoSPARC job type "
            f"`{configs['type']=}`; this script is for ab-initio jobs (`class_3D`)!"
        )

    outdir = str(os.path.join(args.outdir, "per_conf_fsc"))
    file_pattern = "*.mrc"
    files = [
        f for f in glob(os.path.join(args.input_dir, file_pattern)) if "mask" not in f
    ]
    pred_dir = sorted(files, key=utils.numfile_sortkey)
    print("pred_dir[0]:", pred_dir[0])
    csparc_num = pred_dir[0].split("/")[-1].split(".")[0].split("_")[3]
    csparc_job = pred_dir[0].split("/")[-1].split(".")[0].split("_")[0]
    print("cryosparc_num:", csparc_num)
    print("cryosparc_job:", csparc_job)

    lst = []
    for cls in range(args.num_classes):
        class_fl = f"{csparc_job}_passthrough_particles_class_{cls}.cs"
        cs = np.load(os.path.join(args.input_dir, class_fl))
        cs_new = cs[:: args.num_imgs]
        print(f"class {cls}: {len(cs_new)}")

        for cs_i in range(len(cs_new)):
            path = cs_new[cs_i]["blob/path"].decode("utf-8")
            gt = path.split("/")[-1].split("_")[1]
            lst.append((int(cls), int(gt)))

    if args.calc_fsc_vals:
        utils.get_fsc_curves(
            outdir,
            args.gt_dir,
            vol_dir=args.input_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=(
                lambda i: f"{csparc_job}_class_{lst[i][0]:02d}_{csparc_num}_volume.mrc"
            ),
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
