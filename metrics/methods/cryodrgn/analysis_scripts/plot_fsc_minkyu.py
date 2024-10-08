"""Plot FSC txtfile"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob, re


# Load data from file
def load_data(file):
    data = np.loadtxt(file)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


# Plot data
def plot_data(x, y, label):
    plt.plot(x, y, label=label)

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def parse_args():
    parser = argparse.ArgumentParser(description="Plot FSC data.")
    parser.add_argument(
        "-i", "--input", help="input cryoDRGN fsc text files", required=True
    )
    parser.add_argument(
        "-a", "--angpix", type=float, default=0, help="physical pixel size in angstrom"
    )
    parser.add_argument("-o", "--output", type=str, help="output file name")
    return parser


def main(args):
    # Create a subplot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Load and plot data from each file
    file_pattern = "*.txt"
    all_fscs = glob.glob(os.path.join(args.input, file_pattern))
    all_fscs = sorted(all_fscs, key=natural_sort_key)
    print('len(all_fscs):',len(all_fscs))
    for file in all_fscs:
        x, y = load_data(file)
        plot_data(x, y, os.path.basename(file))

    ax.set_aspect(0.3)  # Set the aspect ratio on the plot specifically

    if args.angpix != 0:
        freq = np.arange(1, 6) * 0.1
        res = ["1/{:.1f}".format(val) for val in ((1 / freq) * args.angpix)]
        print(res)
        res_text = res
        plt.xticks(np.arange(1, 6) * 0.1, res_text)
        plt.xlabel("1/resolution (1/Å)")
        plt.ylabel("Fourier shell correlation")
    else:
        plt.xlabel("Spatial Frequency")
        plt.ylabel("Fourier shell correlation")

    plt.ylim(0, 1.0)
    plt.xlim(0, 0.5)

    # Create the legend on the figure, not the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": 6})

    plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        plt.savefig(os.path.join(args.output,"fsc_plots.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main(parse_args().parse_args())
