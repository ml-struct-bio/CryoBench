"""Utility functions used across FSC pipelines for handling conformation outputs."""

from collections.abc import Iterable
import logging
from typing import Union
import numpy as np
from cryodrgn import analysis, mrc

logger = logging.getLogger(__name__)


# Ribosembly number of images per Ribosembly structure (total 16 structures)
RIBOSEMBLY_NUM_IMGS = [
    9076,
    14378,
    23547,
    44366,
    30647,
    38500,
    3915,
    3980,
    12740,
    11975,
    17988,
    5001,
    35367,
    37448,
    40540,
    5772,
]


def get_nearest_z_array(
    zmat: np.ndarray, num_vols: int, num_imgs: Union[int, str]
) -> np.ndarray:
    z_lst = []
    z_mean_lst = []
    for i in range(num_vols):
        if isinstance(num_imgs, int):
            z_nth = zmat[(i * num_imgs) : ((i + 1) * num_imgs)]
        elif num_imgs == "ribo":
            z_nth = zmat[
                sum(RIBOSEMBLY_NUM_IMGS[:i]) : sum(RIBOSEMBLY_NUM_IMGS[: (i + 1)])
            ]
        else:
            raise ValueError(f"{num_imgs=}")

        z_nth_avg = z_nth.mean(axis=0)
        z_nth_avg = z_nth_avg.reshape(1, -1)
        z_lst.append(z_nth)
        z_mean_lst.append(z_nth_avg)

    nearest_z_lst = []
    centers_ind_lst = []
    num_img_for_centers = 0
    for i in range(num_vols):
        nearest_z, centers_ind = analysis.get_nearest_point(z_lst[i], z_mean_lst[i])
        nearest_z_lst.append(nearest_z.reshape(nearest_z.shape[-1]))
        centers_ind_lst.append(centers_ind + num_img_for_centers)

        if num_imgs == "ribo":
            num_img_for_centers += RIBOSEMBLY_NUM_IMGS[i]

    return np.array(nearest_z_lst)


def pad_mrc_vols(mrc_volfiles: Iterable[str], new_D: int) -> None:
    for mrc_file in mrc_volfiles:
        v, header = mrc.parse_mrc(mrc_file)
        x, y, z = v.shape
        assert new_D >= x
        assert new_D >= y
        assert new_D >= z

        new = np.zeros((new_D, new_D, new_D), dtype=np.float32)
        i = (new_D - x) // 2
        j = (new_D - y) // 2
        k = (new_D - z) // 2
        new[i : (i + x), j : (j + y), k : (k + z)] = v

        # adjust origin
        apix = header.get_apix()
        xorg, yorg, zorg = header.get_origin()
        xorg -= apix * k
        yorg -= apix * j
        zorg -= apix * i

        mrc.write(mrc_file, new, mrc.MRCHeader.make_default_header(new, Apix=apix))
