conda activate cryodrgn
python metrics/per_conf_fsc/3dflex.py -o output --method 3dflex --gt-dir ./gt_vols --cryosparc-job J130 --mask ./mask.mrc --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output -method 3dflex

python metrics/per_conf_fsc/3dflex.py -o output --method 3dflex --gt-dir ./gt_vols --cryosparc-job J130 --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output -method 3dflex --mask False