import argparse
import os
import logging
import interface
import utils

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    if args.method is None:
        logger.info('No method label specified, using "recovar" as default...')
        method_lbl = "recovar"
    else:
        method_lbl = str(args.method)

    outdir = str(os.path.join(args.o, method_lbl, "per_conf_fsc"))
    logger.info(f"Putting output under: {outdir} ...")

    if args.calc_fsc_vals:

        def vol_fl_function(i: int):
            return os.path.join(format(i, "03d"), "ml_optimized_locres_filtered.mrc")

        utils.get_fsc_curves(
            outdir,
            args.gt_dir,
            mask_file=args.mask,
            fast=args.fast,
            overwrite=args.overwrite,
            vol_fl_function=vol_fl_function,
        )


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
