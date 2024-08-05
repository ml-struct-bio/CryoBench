conda activate dsd
python metrics/per_conf_fsc/opus-dsd.py --input-dir results/opus-dsd --epoch 19 -o output --method opus-dsd --gt-dir ./gt_vols --mask ./mask.mrc --apix 3.0 --num-vols 100 --num-imgs 1000

conda activate cryodrgn
python metrics/per_conf_fsc/per_conf_plot.py output --method opus-dsd

conda activate dsd
python metrics/per_conf_fsc/opus-dsd.py --input-dir results/opus-dsd --epoch 19 -o output --method opus-dsd --gt-dir ./gt_vols

conda activate cryodrgn
python metrics/per_conf_fsc/per_conf_plot.py output --method opus-dsd --mask False