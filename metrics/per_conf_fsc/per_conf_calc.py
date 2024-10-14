"""Calculate FSCs between conformations matched across cryoDRGN model latent spaces.

Example usage
-------------
$ python metrics/per_conf_fsc/cdrgn.py results/cryodrgn --epoch 19 --Apix 3.0 \
                                       -o output --gt-dir ./gt_vols --mask ./mask.mrc \
                                       --num-imgs 1000 --num-vols 100

"""
import argparse
import os
import utils
from cdrgn import main as run_cdrgn
from drgnai_fixed import main as run_drgnai_fixed


def main(args: argparse.Namespace):
    input_files = os.listdir(args.input_dir)

    if "config.yaml" in input_files:
        run_cdrgn(args)
    elif "out" in input_files and "config.yaml" in os.listdir(
        os.path.join(args.input_dir, "out")
    ):
        run_drgnai_fixed(args)
    else:
        raise ValueError(
            f"Unrecognized output folder format found in `{args.input_dir}`!"
            f"Does not match for any known methods: "
            "cryoDRGN, DRGN-AI, opusDSD, 3dflex, 3DVA, RECOVAR"
        )


if __name__ == "__main__":
    main(utils.add_calc_args().parse_args())
