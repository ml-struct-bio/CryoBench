"""Compute FSC between two volumes"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from cryodrgn import fft
from cryodrgn.source import ImageSource
import os 
import glob, re
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vol1", help="input vols folder")
    parser.add_argument("vol2", help="gt volumes (all no-noise volumes) folder")
    parser.add_argument("--mask", default=None, help="mask mrc")
    parser.add_argument("--method", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--Apix", type=float, default=1)
    parser.add_argument("-o", help="output folder (fsc.txt will be saved)")
    return parser

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args, cls_path, gt_path, output_file):
    vol1 = ImageSource.from_file(cls_path)
    vol2 = ImageSource.from_file(gt_path)

    vol1 = vol1.images()
    vol2 = vol2.images()

    # assert isinstance(vol1, np.ndarray)
    # assert isinstance(vol2, np.ndarray)

    if args.mask is not None:
        mask_path = os.path.join(args.mask)
        mask = ImageSource.from_file(mask_path)
        mask = mask.images()

        # assert isinstance(mask, np.ndarray)
        vol1 *= mask
        vol2 *= mask

    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing="ij")
    coords = np.stack((x0, x1, x2), -1)
    r = (coords**2).sum(-1) ** 0.5

    assert r[D // 2, D // 2, D // 2] == 0.0

    vol1 = fft.fftn_center(vol1)
    vol2 = fft.fftn_center(vol2)

    # logger.info(r[D//2, D//2, D//2:])
    prev_mask = np.zeros((D, D, D), dtype=bool)
    fsc = [1.0]
    for i in range(1, D // 2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1, v2) / (np.vdot(v1, v1) * np.vdot(v2, v2)) ** 0.5
        fsc.append(float(p.real))
        prev_mask = mask
    fsc = np.asarray(fsc)
    x = np.arange(D // 2) / D

    res = np.stack((x, fsc), 1)
    if args.o:
        np.savetxt(output_file, res)
    else:
        logger.info(res)

    w = np.where(fsc < 0.5)
    if w:
        logger.info("0.5: {}".format(1 / x[w[0]] * args.Apix))

    w = np.where(fsc < 0.143)
    if w:
        logger.info("0.143: {}".format(1 / x[w[0]] * args.Apix))

    if args.plot:
        plt.plot(x, fsc)
        plt.ylim((0, 1))
        plt.show()


if __name__ == "__main__":
    args = parse_args().parse_args()
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    if not os.path.exists(os.path.join(args.o, "no_mask")):
        os.makedirs(os.path.join(args.o, "no_mask"))
    if not os.path.exists(os.path.join(args.o, "mask")):
        os.makedirs(os.path.join(args.o, "mask"))

    file_pattern = "*.mrc"
    class_files = glob.glob(os.path.join(args.vol1, file_pattern))
    class_files = sorted(class_files, key=natural_sort_key)
    gt_files = glob.glob(os.path.join(args.vol2, file_pattern))
    gt_files = sorted(gt_files, key=natural_sort_key)
    
    for cls_path in class_files:
        if args.method == '3dcls':
            cls = cls_path.split('/')[-1].split('.')[0].split('_')[2]
        elif '3dcls_abinit' in args.method:
            cls = cls_path.split('/')[-1].split('.')[0].split('_')[2]
        elif args.method == 'opus-dsd' or args.method == 'opus-dsd_2':
            # print('cls_path:',cls_path)
            cls = cls_path.split('/')[-1].split('.')[0][9:]
        else:
            cls = cls_path.split('/')[-1].split('.')[0].split('_')[-1]
        print('cls:',cls)
        cls = int(cls)
        assert isinstance(cls, int)
        for gt_path in gt_files:
            gt = gt_path.split('/')[-1].split('.')[0].split('_')[0]
            gt = int(gt)
            assert isinstance(gt, int)
            if args.mask is not None:
                output_file = os.path.join(args.o, "mask", f"masked_fsc_cls_{cls}_gt_{gt}.txt")
            else:
                output_file = os.path.join(args.o, "no_mask", f"fsc_cls_{cls}_gt_{gt}.txt")
    
            main(args, cls_path, gt_path, output_file)
