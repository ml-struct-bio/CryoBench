conda activate cryodrgn
python metrics/per_conf_fsc_Ribosembly/drgnai_abinit_after_align.py  -o output --method drgnai_abinit --gt-dir ./gt_vols --mask ./mask.mrc
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method drgnai_abinit
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method drgnai_abinit --flip True

python metrics/per_conf_fsc_Ribosembly/drgnai_abinit_after_align.py -o output --method drgnai_abinit --gt-dir ./gt_vols
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method drgnai_abinit --mask False
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method drgnai_abinit --flip True --mask False