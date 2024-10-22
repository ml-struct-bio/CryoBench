"""Calculate FSCs across cryoSPARC 3D Flex Train model conformations.

Example usage
-------------
$ python metrics/per_conf_fsc/cryosparc_3dflex.py results/CS-cryobench/J9 \
            -o cBench/cBench-out_3Dflex/ --gt-dir vols/128_org/ --mask bproj_0.005.mrc

"""
import os
import json
from metrics.utils import utils
from metrics.per_conf_fsc.utils import interface


def main(args):
    cfg_file = os.path.join(args.input_dir, "job.json")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find cryoSPARC job info file `job.json` in given folder "
            f"{args.input_dir=} — did a cryoSPARC job use this as the output path?"
        )
    with open(cfg_file) as f:
        configs = json.load(f)

    if configs["type"] != "flex_train":
        raise ValueError(
            f"Given folder {args.input_dir=} contains cryoSPARC job type "
            f"`{configs['type']=}`; this script is for 3D Flex "
            f"Train jobs (`flex_train`)!"
        )

    outdir = str(os.path.join(args.outdir, "per_conf_fsc"))
    os.makedirs(os.path.join(outdir, "vols"), exist_ok=True)

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
