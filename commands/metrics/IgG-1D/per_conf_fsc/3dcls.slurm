module purge
module load anaconda3/2023.3

conda activate tomodrgn

python metrics/per_conf_fsc/3dcls.py --input-dir /scratch/gpfs/ZHONGE/mj7341/NeurIPS/results/conf-het/dihedral/snr001 -o output --method 3dcls --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --num-classes 10 --mask ./mask.mrc
python metrics/per_conf_fsc/3dcls.py --input-dir /scratch/gpfs/ZHONGE/mj7341/NeurIPS/results/conf-het/dihedral/snr001 -o output --method 3dcls --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --num-classes 10
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls --mask False