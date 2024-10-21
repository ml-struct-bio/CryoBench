import os
import interface


def main(args):
    print("method:", args.method)
    if not os.path.exists(os.path.join(args.o, args.method, "per_conf_fsc")):
        os.makedirs(os.path.join(args.o, args.method, "per_conf_fsc"))

    # Compute FSC cdrgn
    if not os.path.exists("{}/{}/per_conf_fsc/fsc".format(args.o, args.method)):
        os.makedirs("{}/{}/per_conf_fsc/fsc".format(args.o, args.method))
    if not os.path.exists("{}/{}/per_conf_fsc/fsc_no_mask".format(args.o, args.method)):
        os.makedirs("{}/{}/per_conf_fsc/fsc_no_mask".format(args.o, args.method))


if __name__ == "__main__":
    main(interface.add_calc_args().parse_args())
