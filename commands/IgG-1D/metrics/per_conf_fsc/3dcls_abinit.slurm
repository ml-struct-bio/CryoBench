conda activate cryodrgn

python metrics/per_conf_fsc/3dcls_abinit.py --input-dir results/3dcls_abinit -o output --method 3dcls_abinit --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J45 --num-classes 20 --mask ./mask.mrc --num-classes 20 --num-vols 100 --num-imgs 1000
python metrics/per_conf_fsc/3dcls_abinit.py --input-dir results/3dcls_abinit -o output --method 3dcls_abinit --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J45 --num-classes 20 --num-vols 100 --num-imgs 1000

python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --flip True

python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --mask False
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dcls_abinit --flip True --mask False