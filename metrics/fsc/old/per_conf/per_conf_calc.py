"""Calculate FSCs between conformations matched across a volume model's latent space.

This script is an alternative to the method-specific FSC calculation scripts found in
this folder; it can automatically detect the method used to generate
the output folder given.

Example usage
-------------
$ python metrics/fsc/old/per_conf/per_conf_calc results/cryodrgn \
                --epoch 19 --Apix 3.0 -o output --gt-dir ./gt_vols --mask ./mask.mrc

"""
import os
import sys
import argparse
import json
from cdrgn import main as run_cdrgn
from drgnai import main as run_drgnai
from opusdsd import main as run_opusdsd
from re_covar import main as run_recovar
from cryosparc_3dcls import main as run_cryosparc_3dcls
from cryosparc_abinitio import main as run_cryosparc_abinitio
from cryosparc_3dva import main as run_cryosparc_3dva
from cryosparc_3dflex import main as run_cryosparc_3dflex

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from utils import interface


def main(args: argparse.Namespace):
    input_files = os.listdir(args.input_dir)

    if "config.yaml" in input_files:
        run_cdrgn(args)
    elif "config.pkl" in input_files:
        run_opusdsd(args)
    elif "reordered_z.npy" in input_files:
        run_recovar(args)

    elif "job.json" in input_files:
        with open(os.path.join(args.input_dir, "job.json")) as f:
            configs = json.load(f)

        if configs["type"] == "var_3D":
            run_cryosparc_3dva(args)
        elif configs["type"] == "homo_abinit":
            run_cryosparc_abinitio(args)
        elif configs["type"] == "flex_test":
            run_cryosparc_3dflex(args)
        elif configs["type"] == "class_3D":
            run_cryosparc_3dcls(args)
        else:
            raise RuntimeError(f"Unrecognized cryoSPARC job type `{configs['type']}` !")

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
    main(interface.add_calc_args().parse_args())
