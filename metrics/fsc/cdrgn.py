"""Calculate FSCs between volumes from images' mappings in a cryoDRGN latent space.

Example usage
-------------
$ python metrics/fsc/imgfsc_cryodrgn.py \
    CryoBench/001_IgG-1D/ data/2024/cryobench/IgG-1D/gt_latents.pkl \
    --gt_dir=data/2024/cryobench/IgG-1D/vols/128_org/ \
    -o cBench-output_test/cdrgn-imgfsc_001 -n 100 --apix 3.0

"""
import os
import argparse
import subprocess
import pickle
import yaml
from glob import glob
from time import time
import logging
from typing import Callable
from utils import volumes
import numpy as np
import torch
from sklearn.metrics import auc
from cryodrgn import mrc, models, utils

logger = logging.getLogger(__name__)
CHIMERAX_PATH = os.environ["CHIMERAX_PATH"]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN_PATH = os.path.join(ROOT_DIR, "utils", "align.py")


def parse_args() -> argparse.Namespace:
    """Create and parse command line arguments for this script (see `main` below)."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "traindir",
        help="Path to folder with output from a cryoDRGN train_vae or abinit_het model",
    )
    parser.add_argument("-o", required=True)

    parser.add_argument(
        "labels", help=".pkl file with ground truth class index per particle"
    )
    parser.add_argument(
        "--gt_paths", help=".pkl file with path to ground truth volume per particle"
    )
    parser.add_argument(
        "--gt_dir", help="path to folder with ground truth .mrc volumes per particle"
    )
    parser.add_argument("-n", required=True, type=int, help="Number of vols to sample")
    parser.add_argument("--apix", required=True, type=float)
    parser.add_argument("--epoch", type=int, default=-1, help="epoch (default: last)")
    parser.add_argument(
        "--no-align",
        action="store_false",
        dest="align_volumes",
        help="Skip alignment of volumes, in the case of fixed pose reconstruction",
    )
    parser.add_argument(
        "--multi-align",
        action="store_true",
        help="Align volumes in parallel using a compute cluster",
    )
    parser.add_argument("--no-fscs", action="store_false", dest="calc_fsc_vals")

    return parser.parse_args()


def prep_generator(
    config_path: str, checkpoint_path: str
) -> Callable[[torch.Tensor], np.ndarray]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    norm = [float(x) for x in cfg["dataset_args"]["norm"]]
    model, lattice = models.HetOnlyVAE.load(cfg, checkpoint_path, device="cuda:0")
    model.eval()

    return lambda z: model.decoder.eval_volume(
        lattice.coords, lattice.D, lattice.extent, norm, z
    )


def align(vol_path: str, ref_path: str, apix: float = 1.0, flip: bool = True) -> None:
    data, header = mrc.parse_mrc(vol_path)
    header.update_origin(0.0, 0.0, 0.0)
    header.update_apix(apix)
    mrc.write(vol_path, data, header)

    flip_str = "--flip" if flip else ""
    log_file = os.path.splitext(vol_path)[0] + ".txt"
    cmd = f'{CHIMERAX_PATH} --nogui --script "{ALIGN_PATH} {ref_path} {vol_path} '
    cmd += f'{flip_str} -o {vol_path} -f {log_file} " > {log_file} '

    subprocess.check_call(cmd, shell=True)


def main(args: argparse.Namespace) -> None:
    """Running the script to get FSCs across cryoDRGN image-wise conformations."""

    cfg_file = os.path.join(args.traindir, "config.yaml")
    if not os.path.exists(cfg_file):
        raise ValueError(
            f"Could not find cryoDRGN config file {cfg_file} "
            f"— is {args.traindir=} a folder cryoDRGN output folder?"
        )
    cfg = os.path.join(args.traindir, "config.yaml")

    if not (args.gt_paths is None) ^ (args.gt_dir is None):
        raise ValueError("Must provide exactly one of --gt_paths or --gt_dir!")
    if not args.align_volumes and args.multi_align:
        raise ValueError(
            "Cannot use parallelized volume alignment when using --no-align!"
        )

    labels = pickle.load(open(args.labels, "rb"))
    if args.gt_paths is not None:
        gt_paths = pickle.load(open(args.gt_paths, "rb"))

        if len(labels) != len(gt_paths):
            raise ValueError(
                f"Mismatch between size of labels {len(labels)} "
                f"and volume paths {len(gt_paths)} !"
            )

    else:
        gt_files = sorted(
            glob(os.path.join(args.gt_dir, "*.mrc")), key=volumes.numfile_sortkey
        )
        gt_paths = [gt_files[i] for i in labels]

    N = len(labels)
    particle_idxs = np.arange(0, N, N // args.n)
    os.makedirs(args.o, exist_ok=True)

    epoch_str = "" if args.epoch == -1 else f".{args.epoch}"
    checkpoint = os.path.join(args.traindir, f"weights{epoch_str}.pkl")
    if not os.path.exists(checkpoint):
        raise ValueError(
            f"Could not find cryoDRGN model weights for epoch {args.epoch} "
            f"in output folder {args.traindir=} — did the model finishing running?"
        )
    z_path = os.path.join(args.traindir, f"z{epoch_str}.pkl")
    if not os.path.exists(z_path):
        raise ValueError(
            f"Could not find cryoDRGN latent space coordinates for epoch {args.epoch} "
            f"in output folder {args.traindir=} — did the model finishing running?"
        )

    z = utils.load_pkl(z_path)
    generator = prep_generator(cfg, checkpoint)
    log_interval = max(round((len(particle_idxs) // 1000), -2), 5)
    gen_paths = list()

    for vol_i, particle_i in enumerate(particle_idxs):
        if vol_i % log_interval == 0:
            logger.info(
                f"Generating volume {vol_i + 1}/{len(particle_idxs)}  "
                f"(vol_{vol_i:03d}.mrc) ..."
            )

        gt_path = gt_paths[particle_i]
        gen_paths.append(os.path.join(args.o, f"vol_{vol_i:03d}.mrc"))
        gen_vol = generator(z[particle_i, :])
        mrc.write(gen_paths[-1], gen_vol.astype(np.float32))
        if not os.path.isabs(gt_path) and args.gt_paths is not None:
            gt_path = os.path.join(os.path.dirname(args.gt_paths), gt_path)

        if args.align_volumes and not args.multi_align:
            if vol_i % log_interval == 0:
                logger.info(f"Aligning volume (vol_{vol_i:03d} ...")
            align(gen_paths[-1], gt_path, args.apix)

    if args.multi_align:
        volumes.align_volumes_multi(gen_paths, gt_paths)

    if args.calc_fsc_vals:
        fsc_curves = volumes.get_fsc_curves(gt_paths, gen_paths, outdir=args.o)

        auc_vals = {
            particle_idxs[i]: auc(fsc_df.pixres, fsc_df.fsc.abs())
            for i, fsc_df in fsc_curves.items()
        }
        aucs = {class_idx: [] for class_idx in np.unique(labels)}
        for i, auc_val in auc_vals.items():
            aucs[labels[i]].append(auc_val)

        logger.info(
            "\n".join(
                [""]
                + [
                    f"No Images in Class {class_idx} "
                    if len(class_aucs) == 0
                    else f"Num Images Class {class_idx}: {len(class_aucs)} \n"
                    f"AU-FSC Class {class_idx}: "
                    f"{np.mean(class_aucs):.5f} +/- {np.std(class_aucs):.3f}"
                    for class_idx, class_aucs in aucs.items()
                ]
            )
        )
        all_aucs = [a for auc_list in aucs.values() for a in auc_list]
        logger.info(
            f"AU-FSC Overall: {np.mean(all_aucs):.5f} +/- {np.std(all_aucs):.3f}"
        )


if __name__ == "__main__":
    args = parse_args()
    s = time()
    main(args)
    logger.info(f"Completed in {(time()-s):.5g} seconds")
