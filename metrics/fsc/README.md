# FSCs: Tools for comparing similarity of models' latent conformation volumes to ground truth

This folder contains scripts to calculate Fourier shell correlations between volumes at particular points in a given
reconstruction method's latent conformation space, as well as to visualize these FSC results.


## Installation instructions

We recommend using conda environments to install the dependencies for calculating these metrics, and to use a separate
environment for each method **as well as for running the reconstruction model and for CryoBench analyses**.
This is necessary as many of the methods have overlapping dependencies — especially cryoDRGN, which forms
the basis for several of the example methods and is also used by CryoBench itself.

We show here how to install these environments for becnhmarking cryoDRGN; instructions for the other example methods
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

Create an environment for running cryoDRGN models. Here we specify a recent version:
```bash
$ conda create --name cryodrgn_model python=3.10
$ conda activate cryodrgn_model
(cryodrgn_model)$ pip install 'cryodrgn==3.4.1'
```

Next, create an environment for running CryoBench analyses on cryoDRGN output. Here we instead install an older version
of cryoDRGN used by CryoBench through the git submodule:
```bash
$ conda create --name cryodrgn_bench python=3.10
$ conda activate cryoDRGN_bench
(cryodrgn_bench)$ 'cryodrgn==3.4.1'
```


## Generating conformations, calculating FSCs, and plotting results

For cryoDRGN we show here how to run both the fixed-pose and ab-inition version of the reconstruction model on
CryoBench datasets to generate volumes, and then apply our set of CryoBench tools to calculate FSCs across model
conformation volumes and visualize the results.

Corresponding instructions for the other example methods can be found at
[our manual](https://app.gitbook.com/o/gYlX75MBAfjzRuXIYbKH/s/QwtxcduDAIdbCB0vBNnT/~/changes/3/getting-started/running-reconstruction-models)

Take special care to make sure you are running each step in the correct conda environment as described above!

### cryoDRGN: fixed poses

```bash
  $ conda activate cryoDRGN_env

  $ cryodrgn train_vae IgG-1D/images/snr0.005/sorted_particles.128.txt -n 50 --zdim 8 \
                       --ctf IgG-1D/combined_ctfs.pkl --poses IgG-1D/combined_poses.pkl -o cBench-input/cryoDRGN/IgG-1D

  # Compute per conformation FSC
  $ python metrics/fsc/cdrgn.py cBench-input/cryoDRGN/IgG-1D --epoch 49 --Apix 3.0 --gt-dir IgG-1D/vols/128_org/ \
                                --mask IgG-1D/init_mask/backproj_0.005.mrc -o cBench-output/IgG-1D/cryoDRGN_fixed

  # Plot FSCs
  $ python metrics/fsc/per_conf_plot.py output
```

### DRGN-AI: fixed poses
```bash
  $ conda activate drgnai_env

  $ drgnai setup cBench-input/drgnai/IgG-1D/ --particles IgG-1D/images/snr0.005/sorted_particles.128.txt \
                 --ctf IgG-1D/combined_ctfs.pkl --pose IgG-1D/combined_poses.pkl \
                 --reconstruction-type het --pose-estimation fixed
  $ drgnai train cBench-input/cryoDRGN/IgG-1D/

  # Compute per conformation FSC
  $ python metrics/fsc/drgnai.py cBench-input/drgnai/IgG-1D/ --epoch 20 --Apix 3.0 --gt-dir IgG-1D/vols/128_org/ \
                                 --mask IgG-1D/init_mask/backproj_0.005.mrc -o cBench-output/IgG-1D/drgnai_fixed/

  # Plot FSCs
  $ python metrics/fsc/per_conf_fsc_plot.py cBench-output/IgG-1D/drgnai_fixed/
```

### OPUS-DSD
```bash
  $ conda activate opusdsd_env

  $ dsd parse_pose_star IgG-1D/images/snr0.005/snr0.005.star -o snr0.005_pose.pkl --Apix 3.0 -D 128 --relion31
  $ dsd train_cv IgG-1D/images/snr0.005/sorted_particles.128.txt --ctf IgG-1D/combined_ctfs.pkl \
                 --poses snr0.005_pose.pkl --lazy-single --pe-type vanilla --encode-mode grad --template-type conv \
                 -n 20 -b 12 --zdim 12 --lr 1.e-4 --num-gpus 4 --multigpu --beta-control 2. --beta cos \
                 -o cBench-input/opusdsd/IgG-1D/ -r IgG-1D/init_mask/backproj_0.005.mrc --downfrac 0.75 --valfrac 0.25 \
                 --lamb 2. --split sp-split.pkl --bfactor 4. --templateres 224

  # Compute per conformation FSC
  $ python metrics/fsc/opusdsd.py cBench-input/opusdsd/IgG-1D/ --epoch 19 -o cBench-output/IgG-1D/opusdsd/ --Apix 3.0 \
                                  --gt-dir IgG-1D/vols/128_org/ --mask IgG-1D/init_mask/backproj_0.005.mrc \
                                  -o cBench-output/IgG-1D/opusdsd/

  # Plot FSCs
  $ python metrics/fsc/per_conf_plot.py cBench-output/IgG-1D/opusdsd/
```

### cryoSPARC: 3DFlex Train
* Get 3DFlex reconstructed volumes by using `3dflex_per-conf-fsc.ipynb` before computing FSC, then move the generated volumes to `output/3dflex/per_conf_fsc/vols/`.
```bash
  $ conda activate cryodrgn_env

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dflex.py -o output --method 3dflex --gt-dir ./gt_vols --cryosparc-job J130 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output -method 3dflex
```

### cryoSPARC: 3D Variability
```bash
  $ conda activate cryodrgn_env

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J60 --mask ./mask.mrc --Apix 3.0 --num-imgs 1000 --num-vols 100

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dva
```

### RECOVAR
```bash
  $ conda activate recovar_env

  # Generate volumes before computing FSC
  $ python metrics/methods/recovar/gen_reordered_z.py --recovar-result-dir results/recovar
  $ python metrics/methods/recovar/gen_vol_for_per_conf_fsc.py results/recovar -o output/recovar/per_conf_fsc/vols --zdim 10 --num-imgs 1000 --num-vols 100

  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/recovar.py output/recovar/per_conf_fsc/vols -o output --method recovar --gt-dir ./gt_vols --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method recovar
```

### cryoSPARC: 3D Classification
* Move each class volume to `results/3dcls` before computing metric.
```bash
  $ conda activate cryoDRGN_env

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls.py results/3dcls -o output --method 3dcls --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --num-classes 10 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls
```

### cryoDRGN: Ab-initio reconstruction
```bash
  $ conda activate cryoDRGN_env

  # Generate volumes before computing FSC
  $ python metrics/per_conf_fsc/cdrgn2_gen_vol.py results/cryodrgn2 --epoch 29 -o output --method cryodrgn2 --gt-dir ./gt_vols

  # Align reconstructed volumes with ground truth volumes
  $ python metrics/utils/align_multi.py output/cryodrgn2/per_conf_fsc/vols --Apix 3.0 --org-vol ./gt_vols

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/cdrgn2_after_align.py -o output --method cryodrgn2 --gt-dir ./gt_vols --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn2
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn2 --flip True
```

### DRGN-AI: Ab-initio reconstruction
```bash
  $ conda activate drgnai_env

  # Generate volumes before computing FSC
  $ python metrics/per_conf_fsc/drgnai_abinit_gen_vol.py results/drgnai_abinit --epoch 100 --Apix 3.0 -o output --method drgnai_abinit --gt-dir ./gt_vols --num-vols 100 --num-imgs 1000


  # Align reconstructed volumes with ground truth volumes
  $ python metrics/per_conf_fsc/align_multi.py output/drgnai_abinit/per_conf_fsc/vols --Apix 3.0 --org-vol ./gt_vols

  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/drgnai_abinit_after_align.py  -o output --method drgnai_abinit --gt-dir ./gt_vols --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method drgnai_abinit
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method drgnai_abinit --flip True
```

### cryoSPARC: 3D Ab-Initio Classification
```bash
  $ conda activate cryoDRGN_env

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls_abinit.py results/3dcls_abinit -o output --method 3dcls_abinit --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J45 --num-classes 20 --mask ./mask.mrc --num-classes 20 --num-imgs 1000

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --flip True
```
