# Per Conformation FSC: Tools for computing per conformation FSC
### Dependencies:
* cryodrgn version 3.4.0
* drgnai version 1.0.0
### Example usage (CryoDRGN):
```
  $ conda activate cryodrgn
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/cdrgn.py results/cryodrgn --epoch 19 --Apix 3.0 -o output --method cryodrgn --gt-dir ./gt_vols --mask ./mask.mrc --num-imgs 1000 --num-vols 100
  
  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn
```

### Example usage (DRGN-AI-fixed):
```
  $ conda activate drgnai
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/drgnai_fixed.py results/drgnai_fixed --epoch 100 -o output --method drgnai_fixed --gt-dir ./gt_vols --mask ./mask.mrc --Apix 3.0 --num-vols 100 --num-imgs 1000
  
  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_fsc_plot.py output --method drgnai_fixed
```

### Example usage (Opus-DSD):
```
  $ conda activate dsd
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/opus-dsd.py results/opus-dsd --epoch 19 -o output --method opus-dsd --gt-dir ./gt_vols --mask ./mask.mrc --Apix 3.0 --num-vols 100 --num-imgs 1000
  
  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method opus-dsd
```

### Example usage (3DFlex):
* Get 3DFlex reconstructed volumes by using `3dflex_per-conf-fsc.ipynb` before computing FSC, then move the generated volumes to `output/3dflex/per_conf_fsc/vols/`.
```
  $ conda activate cryodrgn
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dflex.py -o output --method 3dflex --gt-dir ./gt_vols --cryosparc-job J130 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output -method 3dflex
```

### Example usage (3DVA):
```
  $ conda activate cryodrgn
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J60 --mask ./mask.mrc --Apix 3.0 --num-imgs 1000 --num-vols 100

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dva
```

### Example usage (RECOVAR):
```
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
```
  $ conda activate cryodrgn
  
  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls.py results/3dcls -o output --method 3dcls --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --num-classes 10 --mask ./mask.mrc

  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls
```

### Example usage (CryoDRGN2):
```
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
```
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
```
  $ conda activate cryodrgn

  # Compute per conformation FSC
  $ python metrics/per_conf_fsc/3dcls_abinit.py results/3dcls_abinit -o output --method 3dcls_abinit --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J45 --num-classes 20 --mask ./mask.mrc --num-classes 20 --num-imgs 1000
  
  # Plot FSCs
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit
  $ python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --flip True
```
