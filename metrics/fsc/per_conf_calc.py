"""Calculate FSCs between conformations matched across a model's latent spaces.

Example usage
-------------
$ python metrics/per_conf_fsc/per_conf_calc results/cryodrgn --epoch 19 --Apix 3.0 \
                                       -o output --gt-dir ./gt_vols --mask ./mask.mrc \
                                       --num-imgs 1000 --num-vols 100

"""
import argparse
import os
from metrics.utils import utils
from cdrgn import main as run_cdrgn
from drgnai import main as run_drgnai
from opusdsd import main as run_opusdsd
from re_covar import main as run_recovar
from cryosparc_3dcls import main as run_cryosparc_3dcls


def main(args: argparse.Namespace):
    input_files = os.listdir(args.input_dir)

    if "config.yaml" in input_files:
        run_cdrgn(args)
    elif "config.pkl" in input_files:
        run_opusdsd(args)
    elif "reordered_z.npy" in input_files:
        run_recovar(args)
    elif "job.json" in input_files:
        run_cryosparc_3dcls(args)
    elif (
        "out" in input_files
        and os.path.isdir(os.path.join(args.input_dir, "out"))
        and "config.yaml" in os.listdir(os.path.join(args.input_dir, "out"))
    ):
        run_drgnai(args)
    else:
        raise ValueError(
            f"Unrecognized output folder format found in `{args.input_dir}`!"
            f"Does not match for any known methods: "
            "cryoDRGN, DRGN-AI, opusDSD, 3dflex, 3DVA, RECOVAR"
        )


if __name__ == "__main__":
    main(utils.add_calc_args().parse_args())
