conda activate cryodrgn

# Use Mask
python metrics/per_conf_fsc_Ribosembly/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-Ribosembly --cryosparc-job JXX --mask ./mask.mrc --apix 3.0 --num-vols 100
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method 3dva

# No Mask
python metrics/per_conf_fsc_Ribosembly/3dva.py -o output --method 3dva --gt-dir ./gt_vols --cryosparc-dir cryosparc/CS-Ribosembly --cryosparc-job JXX --apix 3.0 --num-vols 16
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method 3dva --mask False