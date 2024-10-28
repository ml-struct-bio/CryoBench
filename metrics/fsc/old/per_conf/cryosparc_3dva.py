"""Calculate FSCs across cryoSPARC 3D Variability model conformations.

Example usage
-------------
$ python metrics/fsc/cryosparc_3dva.py results/CS-cryobench/J11 \
            -o cBench/cBench-out_3Dvar/ --gt-dir vols/128_org/ --mask bproj_0.005.mrc

"""
import argparse
import os
import json
import numpy as np
from glob import glob
import logging
from utils import volumes, interface
from cryodrgn import analysis, mrc

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Script to get FSCs across conformations produced by cryoSPARC 3D Variability."""

    cfg_file = os.path.join(args.input_dir, "job.json")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find cryoSPARC job info file `job.json` in given folder "
            f"{args.input_dir=} — did a cryoSPARC job use this as the output path?"
        )
    with open(cfg_file) as f:
        configs = json.load(f)

    if configs["type"] != "var_3D":
        raise ValueError(
            f"Given folder {args.input_dir=} contains cryoSPARC job type "
            f"`{configs['type']=}`; this script is for 3D Variability jobs (`var_3D`)!"
        )

    voldir = os.path.join(args.outdir, "vols")
    os.makedirs(voldir, exist_ok=True)
    file_pattern = "*.mrc"
    files = [
        f for f in glob(os.path.join(args.input_dir, file_pattern)) if "mask" not in f
    ]
    pred_dir = sorted(files, key=volumes.numfile_sortkey)
    print("pred_dir[0]:", pred_dir[0])
    csparc_job = pred_dir[0].split("/")[-1].split(".")[0].split("_")[0]
    print("cryosparc_job:", csparc_job)

    # weights z_ik
    cs_path = os.path.join(args.input_dir, f"{csparc_job}_particles.cs")
    map_mrc_path = os.path.join(args.input_dir, f"{csparc_job}_map.mrc")

    # reference
    v_0 = mrc.parse_mrc(map_mrc_path)[0]
    x = np.load(cs_path)
    component_mrc_path = os.path.join(args.input_dir, f"{csparc_job}_component_0.mrc")
    v_k1 = mrc.parse_mrc(component_mrc_path)[0]  # [128 128 128]
    component_mrc_path = os.path.join(args.input_dir, f"{csparc_job}_component_1.mrc")
    v_k2 = mrc.parse_mrc(component_mrc_path)[0]  # [128 128 128]
    component_mrc_path = os.path.join(args.input_dir, f"{csparc_job}_component_2.mrc")
    v_k3 = mrc.parse_mrc(component_mrc_path)[0]  # [128 128 128]

    for i in range(args.num_vols):
        components_1 = x["components_mode_0/value"]
        components_2 = x["components_mode_1/value"]
        components_3 = x["components_mode_2/value"]

        start_i, end_i = i * args.num_imgs, (i + 1) * args.num_imgs
        z_1 = components_1[start_i:end_i].reshape(args.num_imgs, 1)
        z_2 = components_2[start_i:end_i].reshape(args.num_imgs, 1)
        z_3 = components_3[start_i:end_i].reshape(args.num_imgs, 1)

        z1_nth_avg = z_1.mean(axis=0)
        z1_nth_avg = z1_nth_avg.reshape(1, -1)
        z2_nth_avg = z_2.mean(axis=0)
        z2_nth_avg = z2_nth_avg.reshape(1, -1)
        z3_nth_avg = z_3.mean(axis=0)
        z3_nth_avg = z3_nth_avg.reshape(1, -1)

        nearest_z1, centers_ind1 = analysis.get_nearest_point(z_1, z1_nth_avg)
        nearest_z2, centers_ind2 = analysis.get_nearest_point(z_2, z2_nth_avg)
        nearest_z3, centers_ind3 = analysis.get_nearest_point(z_3, z3_nth_avg)
        vol = v_0 + (nearest_z1 * (v_k1) + nearest_z2 * (v_k2) + nearest_z3 * (v_k3))
        mrc.write(os.path.join(voldir, f"vol_{i:03d}.mrc"), vol.astype(np.float32))

    # Align output conformation volumes to ground truth volumes using ChimeraX
    if args.align_vols:
        volumes.align_volumes_multi(voldir, args.gt_dir)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            args.gt_dir,
            voldir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
        )

        if args.align_vols:
            volumes.get_fsc_curves(
                args.gt_dir,
                vol_dir=os.path.join(voldir, "aligned"),
                mask_file=args.mask,
                fast=args.fast,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
