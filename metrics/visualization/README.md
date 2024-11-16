# Latent visualization: Tools for visualizing latent space colored by ground truth
### Dependencies:
* cryodrgn version 3.4.0
* recovar

### Example usage (IgG-1D):
* `result-path`: A path to the folder that contains UMAP and latent files before the method name (e.g., /scratch/gpfs/ZHONGE/mj7341/CryoBench/results/IgG-1D/snr0.01).
```
  $ conda activate cryodrgn

  # 3D Class
  * Copy 3dcls job to `results/3dcls/cls_{num classes}` (e.g., JXX_class_XX_XXXXX_volume.mrc)
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dcls --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_classes 10 --num_vols 100

  # 3D Class abinit
  * Copy 3dcls_abinit job to `results/3dcls_abinit/cls_{num classes}` (e.g., JXX_class_XX_final_volume.mrc)
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dcls_abinit --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_classes 20 --num_vols 100

  # 3DVA
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dva --cryosparc_job_num JXX --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_vols 100

  # 3DFlex
  $ python metrics/visualization/visualize_umap_IgG-1D.py --method 3dflex --cryosparc_job_num JXX --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_vols 100

  # Get reordered latent of recovar
  $ conda activate recovar
  $ python metrics/methods/recovar/gen_reordered_z.py --recovar-result-dir results/recovar

  # Other methods
  $ for method in cryodrgn cryodrgn2 drgnai_fixed drgnai_abinit opus-dsd recovar
  do
    python metrics/visualization/visualize_umap_IgG-1D.py --method ${method} -o output/visualize_umap_igg1d --result-path results --num_imgs 1000 --num_vols 100
  done
```
