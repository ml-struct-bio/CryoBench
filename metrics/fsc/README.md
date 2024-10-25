# FSCs: Tools for comparing models' conformation volumes' similarity to ground truth volumes

This folder contains scripts to calculate Fourier shell correlations between volumes at particular points in a given
reconstruction method's latent conformation space, as well as to visualize these FSC results.

We have included here example scripts to perform this analysis for
[cryoDRGN](https://github.com/ml-struct-bio/cryodrgn), [DRGN-AI](https://github.com/ml-struct-bio/drgnai),
[OPUS-DSD](https://github.com/alncat/opusDSD), [RECOVAR](https://github.com/ma-gilles/recovar),
as well as four [cryoSPARC](https://guide.cryosparc.com/)
reconstruction methods (3D Classification, Ab-Initio, 3D Variability, and 3D Flex).
These are designed to be run on the output of these methods when applied to the example datasets
found at [https://zenodo.org/records/11629428](https://zenodo.org/records/11629428).


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


## Generating conformations, calculating FSCs, and plotting results

For our example methods we show here how to run the corresponding model on CryoBench datasets to generate
reconstructions, and then apply our set of CryoBench tools to calculate FSCs across model conformation volumes and
visualize the results. Additional examples can be found in the documentation in the scripts within this folder.

Each set of analyses should be run in the same conda environment that was created for each method as described above.
Note that we use ChimeraX throughout these analyses; you must first install ChimeraX locally and then create a bash
environment variable pointing to the installation location as shown in the examples below.

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
