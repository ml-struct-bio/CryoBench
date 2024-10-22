# Per Conformation FSC: Tools for comparing models' conformation volumes' similarity to ground truth volumes

This folder contains scripts to calculate Fourier shell correlations between volumes at particular points in a given
reconstruction method's latent conformation space, as well as to visualize these per conformation FSCs.

We have included here example scripts to perform this analysis for cryoDRGN, DRGN-AI, opusDSD, RECOVAR, as well as four
cryoSPARC reconstruction methods (3D Classification, Ab-Initio, 3D Variability, and 3D Flex). These are designed to be
run on the output of these methods when applied to the example datasets found at:


## Installation instructions

We recommend using conda environments to install the dependencies for calculating these metrics.
Shown below are instructions for creating environments that can also be used to run the reconstruction learning methods
that create the input for the per conformation pipelines.

Start by cloning the CryoBench git repository; note that we also have to fetch the dependency codebases drgnai,
opusDSD, and RECOVAR through their submodules:
```bash
git clone --recurse-submodules git@github.com:ml-struct-bio/CryoBench.git --branch='refactor'
```

We recommend creating a separate environment to install each tested method, as many of the methods have
overlapping dependencies — especially cryoDRGN, which forms the basis for many of the methods and is also used by
CryoBench in its own analyses. We install an older version of cryoDRGN that also necessitates specifying versions for
cryoDRGN dependencies (`pip install 'torch<=2.4.0' 'numpy<1.27' 'matplotlib<3.7' 'cryodrgn<3'`).

### cryoDRGN and cryoSPARC

The simplest installation process is for testing cryoDRGN outputs; we can also use the same environment for testing
cryoSPARC outputs which requires no additional dependencies:
```bash
conda create --name cryoDRGN_env python=3.10
conda activate cryoDRGN_env
pip install 'torch<=2.4.0' 'numpy<1.27' 'matplotlib<3.7' 'cryodrgn<3'
```

### DRGN-AI

Here we have to additionally install DRGN-AI via GitHub using `pip`:
```bash
conda create --name drgnai_env python=3.10
conda activate drgnai_env
pip install git+https://github.com/ml-struct-bio/drgnai.git
pip install 'torch<=2.4.0' 'numpy<1.27' 'matplotlib<3.7' 'cryodrgn<3'
```

### opusDSD

For this package we have to use their custom dependency list, and to install the package from
the clone repository using `pip install -e`:
```bash
conda env create --name opusdsd_env -f CryoBench/metrics/methods/opusDSD/environmentcu11torch11.yml
conda activate opusdsd_env
cd CryoBench/metrics/methods/opusDSD
git checkout 1.0.0
cd ../../../../
pip install -e CryoBench/metrics/methods/opusDSD
pip install 'torch<=2.4.0' 'numpy<1.27' 'matplotlib<3.7' 'cryodrgn<3'
```

### RECOVAR

In the case of RECOVAR, we have to install a specific version of `jax` on top of their custom requirements list, along
with setting some bash environment variables and creating a special `ipykernel` installation:
```bash
conda create --name recovar_env python=3.11
conda activate recovar_env
pip install -U "jax[cuda12_pip]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --no-deps -r CryoBench/metrics/methods/recovar/recovar_install_requirements.txt
python -m ipykernel install --user --name=recovar

# can also be added to .bashrc
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```


## Example usage

Here are some example commands used for each method to create model output, calculate FSCs against ground truth
conformations, and visualize the results. Additional examples can be found in the documentation in each script.

### cryoDRGN

```bash
  # Compute per conformation FSC
  (cryoBench_env)$ python metrics/per_conf_fsc/cdrgn.py results/cryodrgn --epoch 19 --Apix 3.0 -o output --method cryodrgn --gt-dir ./gt_vols --mask ./mask.mrc --num-imgs 1000 --num-vols 100

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn
```

### Example usage (DRGN-AI-fixed):
```bash
  $ conda activate drgnai

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/drgnai_fixed.py results/drgnai_fixed --epoch 100 -o output --method drgnai_fixed --gt-dir ./gt_vols --mask ./mask.mrc --Apix 3.0 --num-vols 100 --num-imgs 1000

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_fsc_plot.py output --method drgnai_fixed
```

### Example usage (Opus-DSD):
```bash
  $ conda activate dsd

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/opus-dsd.py results/opus-dsd --epoch 19 -o output --method opus-dsd --gt-dir ./gt_vols --mask ./mask.mrc --Apix 3.0 --num-vols 100 --num-imgs 1000

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method opus-dsd
```

### Example usage (3DFlex):
* Get 3DFlex reconstructed volumes by using `3dflex_per-conf-fsc.ipynb` before computing FSC, then move the generated volumes to `output/3dflex/per_conf_fsc/vols/`.
```bash
  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dflex.py -o output --method 3dflex --gt-dir ./gt_vols --cryosparc-job J130 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output -method 3dflex
```

### Example usage (3DVA):
```bash
  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J60 --mask ./mask.mrc --Apix 3.0 --num-imgs 1000 --num-vols 100

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dva
```

### Example usage (RECOVAR):
```bash
  $ conda activate recovar

  # Generate volumes before computing FSC
  $ python metrics/methods/recovar/gen_reordered_z.py --recovar-result-dir results/recovar
  $ python metrics/methods/recovar/gen_vol_for_per_conf_fsc.py results/recovar -o output/recovar/per_conf_fsc/vols --zdim 10 --num-imgs 1000 --num-vols 100

  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/recovar.py output/recovar/per_conf_fsc/vols -o output --method recovar --gt-dir ./gt_vols --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method recovar
```

### Example usage (3D Class):
* Move each class volume to `results/3dcls` before computing metric.
```bash
  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls.py results/3dcls -o output --method 3dcls --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --num-classes 10 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls
```

### Example usage (CryoDRGN2):
```bash
  $ conda activate cryodrgn

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

### Example usage (DRGN-AI):
```bash
  $ conda activate drgnai

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

### Example usage (3D Class abinit):
```bash
  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls_abinit.py results/3dcls_abinit -o output --method 3dcls_abinit --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J45 --num-classes 20 --mask ./mask.mrc --num-classes 20 --num-imgs 1000

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --flip True
```
