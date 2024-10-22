import os
import argparse
import subprocess
import pickle
from time import time
import logging
from typing import Callable
from utils import volumes
import numpy as np
import torch
from sklearn.metrics import auc
from cryodrgn import config, mrc, models

logger = logging.getLogger(__name__)
CHIMERAX_PATH = os.environ["CHIMERAX_PATH"]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN_PATH = os.path.join(ROOT_DIR, "utils", "align.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traindir")
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

    return parser.parse_args()


def prep_generator(
    config_path: str, checkpoint_path: str
) -> Callable[[torch.Tensor], np.ndarray]:
    cfg = config.load(config_path)
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
    if not (args.gt_paths is None) ^ (args.gt_dir is None):
        raise ValueError("Must provide exactly one of --gt_paths or --gt_dir!")

    labels = pickle.load(open(args.labels, "rb"))
    if args.gt_paths is not None:
        gt_paths = pickle.load(open(args.gt_paths, "rb"))

        if len(labels) != len(gt_paths):
            raise ValueError(
                f"Mismatch between size of labels {len(labels)} "
                f"and volume paths {len(gt_paths)} !"
            )

    else:
        gt_paths = [os.path.join(args.gt_dir, f"{i:03d}.mrc") for i in labels]

    N = len(labels)
    particle_idxs = np.arange(0, N, N // args.n)
    os.makedirs(args.o, exist_ok=True)

    if args.epoch == -1:
        z = pickle.load(open(os.path.join(args.traindir, "z.pkl"), "rb"))
        checkpoint = os.path.join(args.traindir, "weights.pkl")
    else:
        z = pickle.load(open(os.path.join(args.traindir, f"z.{args.epoch}.pkl"), "rb"))
        checkpoint = os.path.join(args.traindir, f"weights.{args.epoch}.pkl")

    cfg = os.path.join(args.traindir, "config.yaml")
    generator = prep_generator(cfg, checkpoint)
    aucs = {class_idx: [] for class_idx in np.unique(labels)}

    for vol_i, particle_i in enumerate(particle_idxs):
        if vol_i % 20 == 0:
            logger.info(f"Generating volume #{vol_i:03d} ...")

        gen_path = os.path.join(args.o, f"vol_{vol_i:03d}.mrc")
        gen_vol = generator(z[particle_i, :])
        mrc.write(gen_path, gen_vol.astype(np.float32))

        fsc_path = os.path.join(args.o, f"fsc_{vol_i:03d}.txt")
        gt_path = gt_paths[particle_i]
        if not os.path.isabs(gt_path) and args.gt_paths is not None:
            gt_path = os.path.join(os.path.dirname(args.gt_paths), gt_path)

        if args.align_volumes:
            if vol_i % 20 == 0:
                logger.info(f"Aligning volume #{vol_i:03d} ...")
            align(gen_path, gt_path, args.apix)

        gt_vol = mrc.parse_mrc(gt_path)[0]
        fsc_df = volumes.get_fsc_curve(torch.tensor(gt_vol), torch.tensor(gen_vol))
        auc_val = auc(fsc_df.pixres, fsc_df.fsc.abs())
        aucs[labels[particle_i]].append(auc_val)
        np.savetxt(fsc_path, fsc_df.values)

    for class_idx, class_aucs in aucs.items():
        logger.info(
            f"Num Images Class {class_idx}: {len(class_aucs)} \n"
            f"AUC Class {class_idx}: "
            f"{np.mean(class_aucs):05f} +/- {np.std(class_aucs):05f}"
        )

    logger.info(
        f"AUC Overall: {np.mean(np.concatenate(list(aucs.values())))} "
        f"+/- {np.std(np.concatenate(list(aucs.values())))}"
    )


if __name__ == "__main__":
    args = parse_args()
    s = time()
    main(args)
    print(f"Completed in {(time()-s):.5g} seconds")
