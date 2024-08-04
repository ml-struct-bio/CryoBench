import matplotlib.pyplot as plt
import numpy as np
import os, glob, re
from sklearn.metrics import auc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input")
    parser.add_argument("--method", help="Input")
    parser.add_argument("--flip", default="False", help="Input")
    parser.add_argument("--mask", default="True", help="Input")
    parser.add_argument("--apix", type=float, default=3.0, help="pixel size")
    return parser

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args):
    auc_lst = []
    file_pattern = "*.txt"
    if args.mask == "True":
        if args.flip == "False":
            fsc_files = glob.glob(os.path.join(args.input, args.method, "per_conf_fsc", "fsc", file_pattern))
        else:
            fsc_files = glob.glob(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_flipped", file_pattern))
    else:
        if args.flip == "False":
            fsc_files = glob.glob(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_no_mask", file_pattern))
        else:
            fsc_files = glob.glob(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_flipped_no_mask", file_pattern))

    fsc_files = sorted(fsc_files, key=natural_sort_key)
    freq = np.arange(1, 6) * 0.1
    res = ["1/{:.1f}".format(val) for val in ((1 / freq) * args.apix)]
    res_text = res
    for i, fsc_file in enumerate(fsc_files):
        fsc = np.loadtxt(fsc_file)
        plt.plot(fsc[:,0], fsc[:,1], label=i)
        plt.xticks(np.arange(1, 6) * 0.1, res_text, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("1/resolution (1/Å)", fontsize=20)
        plt.ylabel("Fourier shell correlation", fontsize=20)
        auc_lst.append(auc(fsc[:,0], fsc[:,1]))
    plt.ylim((0, 1))
    auc_total_np = np.array(auc_lst)
    auc_avg_np = np.nanmean(auc_total_np)
    auc_std_np = np.nanstd(auc_total_np)
    auc_med_np = np.nanmedian(auc_total_np, 0)
    for i in range(len(auc_total_np)):
        print(f"{i}: AUC {auc_total_np[i]}")
    print(f"AUC_avg: {auc_avg_np}, std: {auc_std_np}, AUC_med: {auc_med_np}")

    plt.title('auc:'+str(round(auc_avg_np,3))+'+-'+str(round(auc_std_np,3)) +'/med:'+str(round(auc_med_np, 3)), fontsize=15)
    plt.tight_layout()
    if args.mask == "True":
        if args.flip == "False":
            plt.savefig(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_auc.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_auc_flipped.png"), bbox_inches='tight')
    else:
        if args.flip == "False":
            plt.savefig(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_auc_no_mask.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(args.input, args.method, "per_conf_fsc", "fsc_auc_flipped_no_mask.png"), bbox_inches='tight')
    print('plot saved!')


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
