conda activate cryodrgn
python metrics/per_conf_fsc_Ribosembly/cdrgn2_after_align.py -o output --method cryodrgn2 --gt-dir ./gt_vols --mask ./mask.mrc
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn2
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn2 --flip True

python metrics/per_conf_fsc_Ribosembly/cdrgn2_after_align.py -o output --method cryodrgn2 --gt-dir ./gt_vols
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn2 --mask False
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn2 --flip True --mask False