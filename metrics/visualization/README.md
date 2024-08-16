# Latent visualization: Tools for visualizing latent space colored by ground truth
### Dependencies:
* cryodrgn version 3.4.0
### Example usage (IgG-1D):
```
  $ conda activate cryodrgn
  
  # 3D Class
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dcls --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results  --num_imgs 1000 --num_classes 10 --num_vols 100

  # 3D Class abinit
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dcls_abinit --cryosparc_job_num J45 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results  --num_imgs 1000 --num_classes 20 --num_vols 100

  # 3DVA
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dva --cryosparc_job_num J60 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_vols 100

  # 3DFlex
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dflex --cryosparc_job_num J78 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_vols 100

  # Get reordered latent of recovar
  $ python metrics/methods/recovar/gen_reordered_z.py --recovar-result-dir results/recovar # put gen_reordered_z.py in recovar official directory

  # Other methods
  $ for method in cryodrgn cryodrgn2 drgnai_fixed drgnai_abinit opus-dsd recovar
  do
    python metrics/visualization/visualize_umap_IgG-1D.py --method ${method} -o output/visualize_umap_igg1d --result-path results --num_imgs 1000 --num_vols 100
  done
```