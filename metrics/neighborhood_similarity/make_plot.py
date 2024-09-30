import numpy as np
import matplotlib.pyplot as plt

# Function to read data from text file
def read_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    error = data[:, 2]
    return x, y, error


embd_names = [
    "cryosparc_3dflex_embeddings",
    "cryosparc_3dva_embeddings",
    "recovar_embeddings",
    "cryodrgn_embeddings",
    "cryodrgn2_embeddings",
    "drgnai_abinit_embeddings",
    "drgnai_fixed_embeddings",
    "opusdsd_mu_embeddings",
]

# embd_names = ['CryoDRGN', 'DrgnAI-fixed','Opus-DSD','3DFlex', '3DVA', 'RECOVAR','CryoDRGN2','DrgnAI-abinit']


mapping = {
    "cryosparc_3dflex_embeddings": "3DFlex",
    "cryosparc_3dva_embeddings": "3DVA",
    "drgnai_fixed_embeddings": "DrgnAI-fixed",
    "recovar_embeddings": "RECOVAR",
    "cryodrgn_embeddings": "CryoDRGN",
    "cryodrgn2_embeddings": "CryoDRGN2",
    "drgnai_abinit_embeddings": "DrgnAI-abinit",
    "opusdsd_mu_embeddings": "Opus-DSD",
}

# Transforming the list using the mapping dictionary

color_map = {
    "cryodrgn_embeddings": "#6190e6",
    "drgnai_fixed_embeddings": "#88B4E6",
    "opusdsd_mu_embeddings": "#b0e0e6",
    "cryosparc_3dflex_embeddings": "#98fb98",
    "cryosparc_3dva_embeddings": "#f4a460",
    "recovar_embeddings": "#f08080",
    "cryodrgn2_embeddings": "#7b68ee",
    "drgnai_abinit_embeddings": "#a569bd",
    #    '3D Class': '#d8bfd8',
    #    '3D Class (abinit)': '#da70d6',
    #    'G.T': '#bfbfbf'
}

# Read data from each file and create plots
for name in embd_names:
    # Read data from text file
    filename = f"{name}_output.txt"
    x, y, error = read_data(filename)

    # Create plot
    plt.errorbar(
        x / 200.0,
        y / x * 100,
        yerr=error / x * 100,
        fmt="o",
        markersize=8,
        label=name,
        color=color_map.get(name, "black"),
    )
    plt.plot(
        x / 200.0,
        y / x * 100,
        linestyle="-",
        color=color_map.get(name, "black"),
        linewidth=2.5,
    )

# Set plot title and labels
# plt.title('Embedding Neighborhood Similarity',fontsize=20)
plt.xlabel("Neighborhood Radius [%]", fontsize=20)
plt.ylabel("% of Matching Neighbors", fontsize=20)
# plt.legend(fontsize=8)
plt.legend().set_visible(False)
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set tab20 colormap for the plot
plt.set_cmap("tab20")

# Save plot as a high-resolution PDF
plt.tight_layout()
plt.savefig("neighbor_conensus-conf-het-1.pdf", dpi=1200, bbox_inches="tight")

# Show the plot
# plt.show()
