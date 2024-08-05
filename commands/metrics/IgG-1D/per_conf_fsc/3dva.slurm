conda activate cryodrgn

# Use Mask
python metrics/per_conf_fsc/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J60 --mask ./mask.mrc --apix 3.0 --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dva

# No Mask
python metrics/per_conf_fsc/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-IgG1D --cryosparc-job J60 --apix 3.0 --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output --method 3dva --mask False