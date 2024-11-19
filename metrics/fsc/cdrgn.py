"""Calculate FSCs between volumes from images' mappings in a cryoDRGN latent space.

Example usage
-------------
# See zenodo.org/records/11629428 for the Conf-het dataset used in these examples

$ python metrics/fsc/cdrgn.py cryodrgn_output/train_vae/001_IgG-1D/ \
                              IgG-1D/gt_latents.pkl --gt_dir=IgG-1D/vols/128_org/ \
                              -o cryobench_output/cdrgn_train-vae_001/ \
                              -n 100 --Apix 3.0

# Sample more volumes and align before computing FSCs in parallel using compute cluster
$ python metrics/fsc/cdrgn.py cryodrgn_output/abinit_het/001_IgG-1D/ \
                              IgG-1D/gt_latents.pkl --gt_dir=IgG-1D/vols/128_org/ \
                              -o cryobench_output/cdrgn_abinit-het_001/ \
                              -n 1000 --Apix 3.0 --parallel-align

"""
import os
import argparse
import subprocess
import pickle
from glob import glob
from time import time
import logging
import numpy as np
from sklearn.metrics import auc
from utils import volumes
from cryodrgn import mrc, utils

logging.basicConfig(
    level=logging.INFO,
    format="(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
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
    parser.add_argument(
        "labels",
        type=os.path.abspath,
        help=".pkl file with ground truth class index per particle",
    )
    parser.add_argument(
        "-o",
        type=os.path.abspath,
        required=True,
        help="Path to folder where output will be saved",
    )
    parser.add_argument("-n", required=True, type=int, help="Number of vols to sample")
    parser.add_argument("--Apix", required=True, type=float)

    parser.add_argument(
        "--gt-paths", help=".pkl file with path to ground truth volume per particle"
    )
    parser.add_argument(
        "--gt-dir", help="path to folder with ground truth .mrc volumes per particle"
    )

    parser.add_argument(
        "--mask",
        default=None,
        type=os.path.abspath,
        help="Path to mask .mrc to compute the masked metric",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-generated volumes instead of reusing them",
    )

    parser.add_argument("--epoch", type=int, default=-1, help="epoch (default: last)")
    parser.add_argument("--cuda-device", default=0, type=int)
    parser.add_argument("--no-fscs", action="store_false", dest="calc_fsc_vals")

    parser.add_argument(
        "--serial-align",
        action="store_true",
        help="Align volumes in one after the other on the local compute.",
    )
    parser.add_argument(
        "--parallel-align",
        action="store_true",
        help="Align volumes in parallel using a compute cluster",
    )

    return parser.parse_args()


def align_volumes(
    vol_path: str, ref_path: str, apix: float = 1.0, flip: bool = True
) -> None:
    """Align a volume in a .mrc file to another .mrc volume using ChimeraX."""

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
    if args.serial_align and args.parallel_align:
        raise ValueError(
            "Cannot use parallelized volume alignment when using --serial-align!"
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
    generator = volumes.get_volume_generator(cfg, checkpoint)
    log_interval = max(round((len(particle_idxs) // 1000), -2), 5)
    gen_paths = list()

    for vol_i, particle_i in enumerate(particle_idxs):
        gen_file = os.path.join(args.o, f"vol_{vol_i:03d}.mrc")
        gen_paths.append(gen_file)

        if os.path.exists(gen_file) and not args.overwrite:
            continue

        if vol_i % log_interval == 0:
            logger.info(
                f"Generating volume {vol_i + 1}/{len(particle_idxs)}  "
                f"(vol_{vol_i:03d}.mrc) ..."
            )

        gt_path = gt_paths[particle_i]
        gen_vol = generator(z[particle_i, :])
        mrc.write(gen_paths[-1], gen_vol.astype(np.float32))
        if not os.path.isabs(gt_path) and args.gt_paths is not None:
            gt_path = os.path.join(os.path.dirname(args.gt_paths), gt_path)

        if args.serial_align:
            if vol_i % log_interval == 0:
                logger.info(f"Aligning volume (vol_{vol_i:03d} ...")
            align_volumes(gen_paths[-1], gt_path, args.Apix)

    if args.parallel_align:
        volumes.align_volumes_multi(gen_paths, gt_paths)

    if args.calc_fsc_vals:
        fsc_curves = volumes.get_fsc_curves(
            gt_paths, gen_paths, mask_file=args.mask, outdir=args.o
        )

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
                    else f"Class {class_idx} ({len(class_aucs)} image) — "
                    f"AU-FSC: {np.mean(class_aucs):.5f}"
                    if len(class_aucs) == 1
                    else f"Class {class_idx} ({len(class_aucs)} images) — "
                    f"AU-FSC: {np.mean(class_aucs):.5f} +/- {np.std(class_aucs):.3f}"
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
