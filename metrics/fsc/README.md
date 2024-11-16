# FSCs: Tools for comparing similarity of models' reconstructed volumes to ground truth

This folder contains scripts to calculate Fourier shell correlations between volumes at reconstruction model latent
space co-ordinates corresponding to individual particle images, as well as to visualize these FSC results.
It also contains a folder `old/per_conf` for the same analyses done using an older method of reconstructing volumes at
class average latent space points.


## Installation instructions

We recommend using conda environments to install the dependencies for calculating these metrics, and to use a separate
environment for each method **as well as for running the reconstruction model and for CryoBench analyses**.
This is necessary as many of the methods have overlapping dependencies — especially cryoDRGN, which forms
the basis for several of the example methods and is also used by CryoBench itself.

We show here how to install these environments for benchmarking cryoDRGN; instructions for the other example methods
can be found at
[our manual](https://app.gitbook.com/o/gYlX75MBAfjzRuXIYbKH/s/QwtxcduDAIdbCB0vBNnT/getting-started/installation-instructions).

Start by cloning the CryoBench git repository; note that we also have to fetch the codebases for the
example methods through their submodules:
```bash
$ git clone --recurse-submodules git@github.com:ml-struct-bio/CryoBench.git --branch='refactor' --recurse-submodules
```

You will also have to install ChimeraX, which can be done by downloading the correct version for your operating system
from [their website](https://www.cgl.ucsf.edu/chimerax/download.html). Also create an environment variable pointing to
this installation:
```bash
$ export CHIMERAX_PATH="/myhome/software/chimerax-1.6.1/bin/ChimeraX"
```
This variable has to be re-defined every time the environment is loaded unless it is e.g. saved in your `.bashrc` file.

Create an environment for running cryoDRGN models.
Here we specify a recent version to use for producing reconstruction output:
```bash
$ conda create --name cryodrgn_model python=3.10
$ conda activate cryodrgn_model
(cryodrgn_model)$ pip install 'cryodrgn==3.4.1'
```

Next, create an environment for running CryoBench analyses on cryoDRGN output.
Here we instead install an older version of cryoDRGN, and also downgrade its dependencies to account for updates
since this older version of cryoDRGN was released:
```bash
$ conda create --name cryodrgn_bench python=3.10
$ conda activate cryodrgn_bench
(cryodrgn_bench)$ pip install git+https://github.com/ml-struct-bio/cryodrgn.git@2.0.0-beta
(cryodrgn_bench)$ pip install 'numpy<1.27'
```


## Generating conformations, calculating FSCs, and plotting results

For CryoDRGN, we show here how to:
1) Download the IgG-1D Conf-het CryoBench dataset
2) Run both the fixed-pose and ab-initio versions of the CryoDRGN reconstruction model on the IgG-1D dataset
2) Apply the CryoBench tools to calculate FSCs between CryoDRGN model conformation volumes and IgG-1D ground truth
3) Visualize the results of the FSC analysis

Corresponding instructions for the other example methods can be found at
[our manual](https://app.gitbook.com/o/gYlX75MBAfjzRuXIYbKH/s/QwtxcduDAIdbCB0vBNnT/~/changes/3/getting-started/running-reconstruction-models)


### Downloading a CryoBench dataset
Although you can also download the IgG-1D dataset through e.g. a web browser by navigating to the
[Zenodo portal](https://zenodo.org/records/11629428), the below demonstrates how to download the data via the
command-line:
```bash
$ curl "https://zenodo.org/records/11629428/files/IgG-1D.zip?download=1" --output IgG-1D.zip
$ unzip IgG-1D.zip
```

The commands below are assumed to be run from the same directory in which `IgG-1D/` created by `unzip` above is located.

### cryoDRGN with fixed poses

We first run the reconstruction algorithm. This command took 3h 20min using 4 Tesla V100 GPUs:
```bash
  $ conda activate cryodrgn_model

  (cryodrgn_model)$ cryodrgn train_vae IgG-1D/images/snr0.01/sorted_particles.128.txt -n 20 --zdim 8 \
                                       --ctf IgG-1D/combined_ctfs.pkl --poses IgG-1D/combined_poses.pkl \
                                       -o cBench_input/IgG-1D/cryodrgn_fixed/
```

We then run the CryoBench script for generating image volumes and comparing them to ground truth volumes. Because
cryoDRGN output volumes are 0-indexed, the last volume from our model lasting twenty epochs is numbered `19`:
```bash
  $ conda activate cryodrgn_bench

  # Compute per image FSC
  (cryodrgn_bench)$ python metrics/fsc/cdrgn.py cBench_input/IgG-1D/cryodrgn_fixed/ --epoch 19 --Apix 3.0 -n 100 \
                                                --gt-dir IgG-1D/vols/128_org/ \
                                                --mask IgG-1D/init_mask/mask.mrc \
                                                -o cBench_output/IgG-1D/cryodrgn_fixed/

  # Plot FSCs
  (cryodrgn_bench)$ python metrics/fsc/per_conf_plot.py cBench-output/IgG-1D/cryodrgn_fixed/
```

### cryoDRGN with ab-initio poses
```bash
  $ conda activate cryodrgn_model

  (cryodrgn_model)$ cryodrgn abinit_het IgG-1D/images/snr0.005/sorted_particles.128.txt -n 30 --zdim 8 \
                                        --ctf IgG-1D/combined_ctfs.pkl -o cBench_input/IgG-1D/cryodrgn_abinit/
```

```bash
  $ conda activate cryodrgn_bench

  # Compute per image FSC
  (cryodrgn_bench)$ python metrics/fsc/cdrgn.py cBench_input/IgG-1D/cryodrgn_abinit/ --epoch 29 --Apix 3.0 -n 100 \
                                                --gt-dir IgG-1D/vols/128_org/ \
                                                --mask IgG-1D/init_mask/mask.mrc \
                                                -o cBench_output/IgG-1D/cryodrgn_abinit/

  # Plot FSCs
  (cryodrgn_bench)$ python metrics/fsc/per_conf_plot.py cBench_output/IgG-1D/cryodrgn_abinit/
```
