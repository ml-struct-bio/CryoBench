"""Calculate FSCs across cryoSPARC 3D Flex Train model conformations.

Example usage
-------------
$ python metrics/fsc/old/per_conf/cryosparc_3dflex.py results/CS-cryobench/J9 \
            -o cBench/cBench-out_3Dflex/ --gt-dir vols/128_org/ --mask bproj_0.005.mrc
            --project-num P564 --job_num J13

"""
import numpy as np
import os
import sys
import argparse
import zipfile
from cryosparc.tools import CryoSPARC
from cryodrgn import analysis

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from utils import volumes, interface

# replace these as necessary with your CryoSPARC credentials
license_id = None
email = None
password = None
host = None
run_lane = None


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--project-num", required=True)
    parser.add_argument("--job-num", required=True)
    parser.add_argument("--base-port", default=39000, type=int)

    return parser


def main(args):
    """Script to get FSCs across conformations output by cryoSPARC 3D Flex Generate."""
    num_vols = 100

    cs = CryoSPARC(
        license=license_id,
        email=email,
        password=password,
        host=host,
        base_port=args.base_port,
    )
    project = cs.find_project(args.project_num)
    flex_job = cs.find_job(args.project_num, args.job_num)  # Flex train
    particles = flex_job.load_output("particles")
    latents_job = project.create_external_job("W1", "Custom Latents")
    latents_job.connect("particles", args.job_num, "particles", slots=["components"])

    v = np.empty((len(particles), 2))
    for i in range(2):
        v[:, i] = particles[f"components_mode_{i}/value"]

    z_lst = []
    z_mean_lst = []
    for i in range(num_vols):
        z_nth = v[i * args.num_imgs : (i + 1) * args.num_imgs]
        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)

    nearest_z_lst = []
    centers_ind_lst = []
    for i in range(num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind)
    latent_pts = np.array(nearest_z_lst)

    slots = [
        {"prefix": "components_mode_%d" % k, "dtype": "components", "required": True}
        for k in range(2)
    ]
    latents_dset = latents_job.add_output(
        type="particle",
        name="latents",
        slots=slots,
        title="Latents",
        alloc=len(latent_pts),
    )

    for k in range(2):
        latents_dset["components_mode_%d/component" % k] = k
        latents_dset["components_mode_%d/value" % k] = latent_pts[:, k]

    with latents_job.run():
        latents_job.save_output("latents", latents_dset)

    gen_job = project.create_job("W1", "flex_generate")
    gen_job.connect("flex_model", args.job_num, "flex_model")
    gen_job.connect("volume", args.job_num, "volume")
    gen_job.connect("latents", latents_job.uid, "latents")

    gen_job.queue(lane=run_lane)
    gen_job.wait_for_done(error_on_incomplete=True)
    zip_outs = [fl for fl in gen_job.list_files() if os.path.splitext(fl)[1] == ".zip"]
    assert len(zip_outs) == 1
    cryosparc_dir = os.path.join(os.path.split(os.path.realpath(args.input_dir))[0])
    zip_outs = os.path.join(cryosparc_dir, gen_job.uid, zip_outs[0])

    voldir = os.path.join(args.outdir, "vols")
    os.makedirs(voldir, exist_ok=True)
    with zipfile.ZipFile(zip_outs, "r") as zip_ref:
        zip_ref.extractall(voldir)

    if args.calc_fsc_vals:
        volumes.get_fsc_curves(
            voldir,
            args.gt_dir,
            outdir=args.outdir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=lambda i: f"{gen_job.uid}_series_000_frame_{i:03d}",
        )


if __name__ == "__main__":
    main(add_args(interface.add_calc_args()).parse_args())
