"""Align and compute distance between two series of rotation matrices.

Example usage
-------------
$ python metrics/pose_error/rot_error.py cryobench_input/003_IgG-1D_cdrgn2/ \
            datasets/IgG-1D/combined_poses.pkl --labels datasets/IgG-1D/gt_latents.pkl \
            --save-err cryobench_output/cdrgn2_003_rot-error/

"""
import os
import argparse
import logging
import numpy as np
from datetime import datetime as dt
import torch
import cryodrgn.utils
from cryodrgn import lie_tools
from scipy.linalg import logm

logger = logging.getLogger(__name__)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("traindir", help="Results directory")
    parser.add_argument("true_poses", help=".pkl file with GT poses")
    parser.add_argument("--save-err", help="save error with .npy")
    parser.add_argument("--epoch", type=int, default=-1, help="epoch (default: last)")
    parser.add_argument(
        "--labels", help=".pkl file with ground truth class index per particle"
    )
    parser.add_argument("--ind", type=str)
    parser.add_argument(
        "--pred-labels",
        help="Provide if best class pose needs to be selected from predicted poses",
    )
    parser.add_argument(
        "-N", type=int, default=30, help="Number of particles to attempt to align on"
    )
    parser.add_argument("--data", type=str, help="name of the data")
    parser.add_argument("--seed", type=int, default=0)

    return parser


# will give very close result to ang_dist()
def ang_dist_2(A, B):
    diff_rot = np.zeros(len(A))
    for i in range(len(diff_rot)):
        diff_rot[i] = np.sum(logm(np.dot(A[i].T, B[i])) ** 2) ** 0.5

    return np.rad2deg(diff_rot) / np.sqrt(2)


def ang_dist_oop(A, B):
    unitvec_gt = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)
    out_of_planes_gt = np.sum(A * unitvec_gt, axis=-2)
    out_of_planes_gt /= np.linalg.norm(out_of_planes_gt, axis=-1, keepdims=True)
    out_of_planes_pred = np.sum(B * unitvec_gt, axis=-2)
    out_of_planes_pred /= np.linalg.norm(out_of_planes_pred, axis=-1, keepdims=True)
    diff_angle = (
        np.arccos(np.clip(np.sum(out_of_planes_gt * out_of_planes_pred, -1), -1.0, 1.0))
        * 180.0
        / np.pi
    )

    return diff_angle


def ang_dist(A, B):
    diff_rot = np.zeros_like(A)
    for i in range(len(diff_rot)):
        diff_rot[i, ...] = A[i, ...].T @ B[i, ...]
    diff_angle = np.arccos(
        np.clip((np.trace(diff_rot, axis1=1, axis2=2) - 1) / 2, -1, 1)
    )
    diff_angle = np.abs(np.rad2deg(diff_angle))

    return diff_angle


def rot_diff(A, B):
    return np.matmul(np.swapaxes(B, -1, -2), A)


def _flip(rot):
    return np.matmul(np.diag([1, 1, -1]).astype(rot.dtype), rot)


def align_rot(rotA, rotB, N, flip=False):
    if flip:
        rotB = _flip(rotB)

    best_rot, best_medse = None, 1e9

    for i in np.random.choice(len(rotA), min(len(rotA), N), replace=False):
        mean_rot = np.dot(rotB[i].T, rotA[i])
        rotB_hat = np.matmul(rotB, mean_rot)
        medse = np.median(np.sum((rotB_hat - rotA) ** 2, axis=(1, 2)))
        if medse < best_medse:
            best_medse = medse
            best_rot = mean_rot

    # align B into A's reference frame
    rotA_hat = np.matmul(rotA, best_rot.T).astype(rotA.dtype)
    rotB_hat = np.matmul(rotB, best_rot).astype(rotB.dtype)
    dist2 = np.sum((rotB_hat - rotA) ** 2, axis=(1, 2))
    if flip:
        rotA_hat = _flip(rotA_hat)

    return rotA_hat, rotB_hat, best_rot, dist2


def align_rot_flip(rotA, rotB, N):
    ret1 = align_rot(rotA, rotB, N, flip=False)
    ret2 = align_rot(rotA, rotB, N, flip=True)

    if np.median(ret1[-1]) < np.median(ret2[-1]):
        return ret1
    else:
        return ret2


def err_format(err: float, digits: int = 5) -> str:
    if err < 1:
        return format(err, f".{digits}f")
    else:
        return format(format(err, f"{digits}g"), f"<0{digits + 2}")


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    if not os.path.exists(os.path.join(args.save_err)):
        os.makedirs(os.path.join(args.save_err))

    t1 = dt.now()

    basedir, job = os.path.split(os.path.normpath(args.traindir))

    def train_path(f: str) -> str:
        return os.path.join(args.traindir, f)

    if os.path.exists(train_path(f"{job}_final_particles.cs")):
        method = "cryosparc"
        particle_info = np.load(train_path(f"{job}_final_particles.cs"))
        highest_prob = np.zeros(len(particle_info))
        rot1 = np.zeros((len(particle_info), 3))
        cl_idx = 0

        while f"alignments_class_{cl_idx}/class_posterior" in particle_info.dtype.names:
            new_best = (
                particle_info[f"alignments_class_{cl_idx}/class_posterior"]
                > highest_prob
            )
            highest_prob[new_best] = particle_info[
                f"alignments_class_{cl_idx}/class_posterior"
            ][new_best]
            rot1[new_best] = particle_info[f"alignments_class_{cl_idx}/pose"][new_best]
            cl_idx += 1

        rot1 = lie_tools.expmap(torch.tensor(rot1))
        rot1 = rot1.cpu().numpy()
        rot1 = np.array([x.T for x in rot1])

    elif os.path.isfile(args.traindir) and os.path.splitext(args.traindir)[1] == ".pkl":
        method = ".pkl"
        rot1 = cryodrgn.utils.load_pkl(args.traindir)

    else:
        # if args.data == "Tomotwin-100":
        #     rot1 = load_pkl(args.traindir)
        # else:
        if os.path.isdir(train_path("out")):
            method = "drgnai"
            args.traindir = os.path.join(args.traindir, "out")
        else:
            method = "cryodrgn"

        if args.epoch == -1:
            args.epoch = max(
                int(f.split(".")[1])
                for f in os.listdir(args.traindir)
                if f.startswith("pose.") and len(f.split(".")) == 3
            )

        rot1 = cryodrgn.utils.load_pkl(train_path(f"pose.{args.epoch}.pkl"))

    rot2 = cryodrgn.utils.load_pkl(args.true_poses)

    if isinstance(rot1, tuple):
        rot1 = rot1[0]
    if isinstance(rot2, tuple):
        rot2 = rot2[0]

    if args.pred_labels:
        pred_labels = cryodrgn.utils.load_pkl(args.pred_labels)
        rot1 = np.take_along_axis(rot1, pred_labels[:, None, None, None], 1).squeeze()
    if args.ind:
        ind = cryodrgn.utils.load_pkl(args.ind)
        rot2 = rot2[ind]

    assert rot1.shape == rot2.shape
    logger.info(f"data and method: {args.data}, {method}")
    errors_lst = []

    if args.labels:
        labels = np.array(cryodrgn.utils.load_pkl(args.labels), dtype=int)
        if args.ind:
            labels = labels[ind]

        uniq_lbls = np.unique(labels)
        cls_space = int(np.log10(len(uniq_lbls) - 1)) + 1
        print("\n[mean; median] rotation errors:")
        print(f"{' ' * (7 + cls_space)}|    Frobenius     |    Geodesic")
        print("-" * (45 + cls_space))

        frob_means, frob_meds, counts = list(), list(), list()
        geo_means, geo_meds = list(), list()
        for i in np.unique(labels):
            mask = labels == i
            # print('mask:',mask.shape)
            counts.append(mask.sum())
            rot1_i = rot1[mask]
            rot2_i = rot2[mask]
            # print('rot1_i:',rot1_i.shape)
            # print('rot2_i:',rot2_i.shape)
            r1, r2, rot, dist2 = align_rot_flip(rot1_i, rot2_i, args.N)
            # print('r1:',r1.shape)
            # print('dist2:',dist2.shape)
            fmean, fmed = np.mean(dist2), np.median(dist2)
            frob_means.append(fmean)
            frob_meds.append(fmed)

            ang_dists = ang_dist(r1, rot2_i)
            # print('ang_dists:',ang_dists.shape)
            gmean, gmed = np.mean(ang_dists), np.median(ang_dists)
            geo_means.append(gmean)
            geo_meds.append(gmed)
            errors_lst.append(ang_dists)

            fstr = f"{err_format(fmean)}; {err_format(fmed)}"
            gstr = f"{err_format(gmean)}; {err_format(gmed)}"
            print(f"Class {i:<{cls_space}} | {fstr} | {gstr}")

        logger.info(f"Class average Mean squared error: {np.mean(geo_means)}")
        logger.info(f"Class average Median squared error: {np.mean(geo_meds)}")
        w_mean = np.sum(np.array(geo_means) * np.array(counts)) / len(labels)
        w_med = np.sum(np.array(geo_meds) * np.array(counts)) / len(labels)
        logger.info(f"Weighted class average Mean squared error: {w_mean}")
        logger.info(f"Weighted class average Median squared error: {w_med}")

    else:
        r1, r2, rot, dist2 = align_rot_flip(rot1, rot2, args.N)

        logger.info(f"Mean squared error: {np.mean(dist2)}")
        logger.info(f"Median squared error: {np.median(dist2)}")

        ang_dists = ang_dist(r1, rot2)
        mean, med = np.mean(ang_dists), np.median(ang_dists)
        logger.info(f"Mean Geodesic: {mean}")
        logger.info(f"Median Geodesic: {med}")
        errors_lst.append(ang_dists)

    npy_name = f"errs_{method}_rot.npy"
    err_np_path = os.path.join(args.save_err, npy_name)
    errors_npy = np.array(errors_lst)

    with open(err_np_path, "wb") as f:
        np.save(f, errors_npy)

    tottime = dt.now() - t1
    logger.info(f"Finished in {tottime}  ({tottime / args.N} per particle) ")


if __name__ == "__main__":
    main(parse_args().parse_args())
