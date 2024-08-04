conda activate cryodrgn
python metrics/per_conf_fsc_Ribosembly/cdrgn.py --input-dir results/cryodrgn --epoch 19 --apix 3.0 -o  output --method cryodrgn --gt-dir ./gt_vols --mask ./mask.mrc --num-vols 16
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn

python metrics/per_conf_fsc_Ribosembly/cdrgn.py --input-dir results/cryodrgn --epoch 19 --apix 3.0 -o  output --method cryodrgn --gt-dir ./gt_vols --num-vols 16
python metrics/per_conf_fsc_Ribosembly/per_conf_plot.py output --method cryodrgn --mask False