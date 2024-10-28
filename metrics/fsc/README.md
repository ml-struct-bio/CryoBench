# FSCs: Tools for comparing similarity of models' latent conformation volumes to ground truth

This folder contains scripts to calculate Fourier shell correlations between volumes at particular points in a given
reconstruction method's latent conformation space, as well as to visualize these FSC results.


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

For cryoDRGN we show here how to run both the fixed-pose and ab-inition versions of the reconstruction model on
CryoBench datasets to generate volumes, and then how to apply the CryoBench tools to calculate FSCs across model
conformation volumes and visualize the results.

Corresponding instructions for the other example methods can be found at
[our manual](https://app.gitbook.com/o/gYlX75MBAfjzRuXIYbKH/s/QwtxcduDAIdbCB0vBNnT/~/changes/3/getting-started/running-reconstruction-models)


### cryoDRGN with fixed poses

```bash
  $ conda activate cryodrgn_model

  (cryodrgn_model)$ cryodrgn train_vae IgG-1D/images/snr0.005/sorted_particles.128.txt \
                                       -n 50 --zdim 8 --ctf IgG-1D/combined_ctfs.pkl --poses IgG-1D/combined_poses.pkl \
                                       -o cBench-input/cryodrgn_fixed/IgG-1D/

  (cryodrgn_model)$ conda activate cryodrgn_bench

  # Compute per conformation FSC
  (cryodrgn_bench)$ python metrics/fsc/cdrgn.py cBench-input/cryodrgn_fixed/IgG-1D/ --epoch 49 --Apix 3.0 \
                                                --gt-dir IgG-1D/vols/128_org/ \
                                                --mask IgG-1D/init_mask/backproj_0.005.mrc \
                                                -o cBench-output/IgG-1D/cryodrgn_fixed/

  # Plot FSCs
  (cryodrgn_bench)$ python metrics/fsc/per_conf_plot.py cBench-output/IgG-1D/cryodrgn_fixed/
```

### cryoDRGN with ab-initio poses
```bash
  $ conda activate cryodrgn_model

  (cryodrgn_model)$ cryodrgn abinit_het IgG-1D/images/snr0.005/sorted_particles.128.txt -n 50 --zdim 8 \
                                        --ctf IgG-1D/combined_ctfs.pkl -o cBench-input/cryodrgn_abinit/IgG-1D

  (cryodrgn_model)$ conda activate cryodrgn_bench

  # Compute per conformation FSC
  (cryodrgn_bench)$ python metrics/fsc/cdrgn.py cBench-input/cryodrgn_abinit/IgG-1D/ --epoch 49 --Apix 3.0 \
                                                --gt-dir IgG-1D/vols/128_org/ \
                                                --mask IgG-1D/init_mask/backproj_0.005.mrc \
                                                -o cBench-output/IgG-1D/cryodrgn_abinit/

  # Plot FSCs
  (cryodrgn_bench)$ python metrics/fsc/per_conf_plot.py cBench-output/IgG-1D/cryodrgn_abinit/
```
