conda activate cryodrgn
python metrics/per_conf_fsc/cdrgn.py --input-dir results/cryodrgn --epoch 19 --apix 3.0 -o  output --method cryodrgn --gt-dir ./gt_vols --mask ./mask.mrc --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn

python metrics/per_conf_fsc/cdrgn.py --input-dir results/cryodrgn --epoch 19 --apix 3.0 -o  output --method cryodrgn --gt-dir ./gt_vols --num-imgs 1000 --num-vols 100
python metrics/per_conf_fsc/per_conf_plot.py output --method cryodrgn --mask False