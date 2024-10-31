"""Compute error between two series of particle shifts"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cryodrgn.utils


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("trans1", help="Input translations")
    parser.add_argument("trans2", help="Input translations")
    parser.add_argument("--save-err", help="save error with .npy")
    parser.add_argument("--rot-pred", action="store_true")
    parser.add_argument(
        "--labels", help=".pkl file with ground truth class index per particle"
    )
    parser.add_argument("--ind1", help="Index filter for trans1")
    parser.add_argument("--ind2", help="Index filter for trans2")
    parser.add_argument("--ind-rot", help="Index filter for rot")
    parser.add_argument(
        "--rot", help="Input rotations, to adjust for translation shift between models"
    )
    parser.add_argument("--s1", type=float, default=1.0, help="Scale for trans1")
    parser.add_argument("--s2", type=float, default=1.0, help="Scale for trans2")
    parser.add_argument("--show", action="store_true", help="Show histogram")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")

    return parser


def adjust_translations_for_offset(args, rot, trans1, trans2):
    rot = torch.from_numpy(rot).float()
    trans1 = torch.from_numpy(trans1).float()
    trans2 = torch.from_numpy(trans2).float()
    model_offset = torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32))
    optimizer = torch.optim.Adam([model_offset], lr=1e0)

    # normalize the translations to avoid ill-conditioning
    # FIXME: didn't deal with mean
    std = float((trans1.std() + trans2.std()) / 2)
    trans1 = trans1 / std
    trans2 = trans2 / std

    for i in range(512):
        optimizer.zero_grad()
        rotated_offset = (rot @ model_offset.unsqueeze(-1)).squeeze(-1)[:, :2]
        loss = ((trans2 - trans1 + rotated_offset) ** 2).sum(-1).mean()
        loss.backward()
        optimizer.step()

        if i & (i - 1) == 0 and args.verbose:
            print(
                f"epoch {i} RMSE: {loss**0.5:.4f}  Offset: {model_offset.data.numpy()}"
            )

    adj_trans2 = trans2 + rotated_offset.detach()
    return adj_trans2.numpy() * std, model_offset.detach().numpy() * std


def main(args: argparse.Namespace) -> None:
    trans1 = cryodrgn.utils.load_pkl(args.trans1)
    if isinstance(trans1, tuple):
        trans1 = trans1[1]
    trans1 *= args.s1

    if args.verbose:
        print(trans1.shape)
        print(trans1)

    trans2 = cryodrgn.utils.load_pkl(args.trans2)
    if isinstance(trans2, tuple):
        trans2 = trans2[1]

    trans2 *= args.s2
    if args.verbose:
        print(trans2.shape)
        print(trans2)

    if args.ind1:
        trans1 = trans1[cryodrgn.utils.load_pkl(args.ind1).astype(int)]
    if args.ind2:
        trans2 = trans2[cryodrgn.utils.load_pkl(args.ind2).astype(int)]

    assert trans1.shape == trans2.shape
    if args.verbose:
        print(np.mean(trans1, axis=0))
        print(np.mean(trans2, axis=0))

    errors_lst = []
    if args.labels:
        labels = np.array(cryodrgn.utils.load_pkl(args.labels), dtype=int)
        means, meds, counts = [], [], []
        if args.rot:
            rot = cryodrgn.utils.load_pkl(args.rot)

            if isinstance(rot, tuple):
                rot = rot[0]
            if args.ind_rot:
                rot = rot[cryodrgn.utils.load_pkl(args.ind_rot).astype(int)]

        for i in np.unique(labels):
            mask = labels == i
            counts.append(mask.sum())

            trans1_i = trans1[mask]
            trans2_i = trans2[mask]
            dists_i = np.sum((trans1_i - trans2_i) ** 2, axis=1) ** 0.5

            print(dists_i.shape)
            print(f"Class {i} Mean error: {np.mean(dists_i)}")
            print(f"Class {i} Median error: {np.median(dists_i)}")

            if args.rot:
                rot_i = rot[mask]
                trans2_i, offset_3d = adjust_translations_for_offset(
                    args, rot_i, trans1_i, trans2_i
                )
                dists_i = np.sum((trans1_i - trans2_i) ** 2, axis=1) ** 0.5
                mean, med = np.mean(dists_i), np.median(dists_i)
                if args.verbose:
                    print("offset3d: {}".format(offset_3d))
                    print(f"Class {i} Mean error after adjustment: {mean}")
                    print(f"Class {i} Median error after adjustment: {med}")

                means.append(mean)
                meds.append(med)
                errors_lst.append(dists_i)

        dists = np.array(errors_lst)

    else:
        dists = np.sum((trans1 - trans2) ** 2, axis=1) ** 0.5
        if args.verbose:
            print(dists.shape)

        print(f"Mean error: {np.mean(dists):.7g}")
        print(f"Median error: {np.median(dists):.7g}")

        if args.rot:
            rot = cryodrgn.utils.load_pkl(args.rot)
            if isinstance(rot, tuple):
                rot = rot[0]
            if args.ind_rot:
                rot = rot[cryodrgn.utils.load_pkl(args.ind_rot).astype(int)]

            trans2, offset_3d = adjust_translations_for_offset(
                args, rot, trans1, trans2
            )
            dists = np.sum((trans1 - trans2) ** 2, axis=1) ** 0.5
            print(f"offset3d: {np.round(offset_3d, 6)}")
            print(f"Mean error after adjustment: {np.mean(dists):.7g}")
            print(f"Median error after adjustment: {np.median(dists):.7g}")

    if args.rot_pred:
        npy_name = "errs_trans_rot_pred_new.npy"
    else:
        npy_name = "errs_trans_rot_gt_new.npy"

    err_np_path = os.path.join(args.save_err, npy_name)
    os.makedirs(args.save_err, exist_ok=True)
    # errors_npy = np.array(dists)
    with open(err_np_path, "wb") as f:
        np.save(f, dists)

    if args.show:
        plt.figure(1)
        plt.hist(dists)
        plt.figure(2)
        plt.scatter(trans1[:, 0], trans1[:, 1], s=1, alpha=0.1)
        plt.figure(3)
        plt.scatter(trans2[:, 0], trans2[:, 1], s=1, alpha=0.1)
        plt.figure(4)
        d = trans1 - trans2
        plt.scatter(d[:, 0], d[:, 1], s=1, alpha=0.1)
        plt.show()


if __name__ == "__main__":
    args = parse_args().parse_args()
    verbose = args.verbose
    main(args)
