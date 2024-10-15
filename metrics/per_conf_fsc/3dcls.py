import argparse
import numpy as np
import os
from glob import glob
import logging
import utils
import interface

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--num-classes", default=10, type=int)

    return parser


def main(args):
    if args.method is None:
        logger.info('No method label specified, using "3Dcls" as default...')
        method_lbl = "3Dcls"
    else:
        method_lbl = str(args.method)

    outdir = str(os.path.join(args.o, method_lbl, "per_conf_fsc"))
    file_pattern = "*.mrc"
    files = [
        f for f in glob(os.path.join(args.input_dir, file_pattern)) if "mask" not in f
    ]
    pred_dir = sorted(files, key=utils.numfile_sort)
    print("pred_dir[0]:", pred_dir[0])
    cryosparc_num = pred_dir[0].split("/")[-1].split(".")[0].split("_")[3]
    cryosparc_job = pred_dir[0].split("/")[-1].split(".")[0].split("_")[0]
    print("cryosparc_num:", cryosparc_num)
    print("cryosparc_job:", cryosparc_job)

    lst = []
    for cls in range(args.num_classes):
        cs = np.load(
            os.path.join(
                args.input_dir, f"{cryosparc_job}_passthrough_particles_class_{cls}.cs"
            )
        )
        cs_new = cs[:: args.num_imgs]
        print(f"class {cls}: {len(cs_new)}")

        for cs_i in range(len(cs_new)):
            path = cs_new[cs_i]["blob/path"].decode("utf-8")
            gt = path.split("/")[-1].split("_")[1]
            lst.append((int(cls), int(gt)))

    if args.calc_fsc_vals:

        def vol_fl_function(i: int):
            return f"{cryosparc_job}_class_{lst[i][0]:02d}_{cryosparc_num}_volume.mrc"

        utils.get_fsc_curves(
            outdir,
            args.gt_dir,
            vol_dir=args.input_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=vol_fl_function,
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
