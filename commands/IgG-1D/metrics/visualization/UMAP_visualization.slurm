conda activate cryodrgn

python metrics/visualization/visualize_umap_dihed.py --method 3dcls --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results  --num_imgs 1000 --num_classes 10 --num_vols 100
python metrics/visualization/visualize_umap_dihed.py --method 3dcls_abinit --cryosparc_job_num J45 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results  --num_imgs 1000 --num_classes 20 --num_vols 100

python metrics/visualization/visualize_umap_dihed.py --method 3dva --cryosparc_job_num J60 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_classes 20 --num_vols 100
python metrics/visualization/visualize_umap_dihed.py --method 3dflex --cryosparc_job_num J78 --is_cryosparc -o output/visualize_umap_igg1d --cryosparc_path cryosparc/CS-IgG1D --result-path results --num_imgs 1000 --num_classes 20 --num_vols 100

python metrics/gen_reordered_z.py --recovar-result-dir results/recovar # put gen_reordered_z.py in recovar official directory

for method in cryodrgn cryodrgn2 drgnai_fixed drgnai_abinit opus-dsd recovar
do
    python metrics/visualization/visualize_umap_dihed.py --method ${method} -o output/visualize_umap_igg1d --result-path results --num_imgs 1000 --num_classes 20 --num_vols 100
done