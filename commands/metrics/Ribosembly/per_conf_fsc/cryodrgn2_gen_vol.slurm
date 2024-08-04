conda activate cryodrgn
python metrics/per_conf_fsc_Ribosembly/cdrgn2_gen_vol.py --input-dir results/cryodrgn2 --epoch 29 -o output --method cryodrgn2 --gt-dir ./gt_vols
python metrics/per_conf_fsc_Ribosembly/align_multi.py output/cryodrgn2/per_conf_fsc/vols --apix 3.0 --org-vol ./gt_vols