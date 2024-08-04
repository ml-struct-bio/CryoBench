conda activate drgnai
python metrics/per_conf_fsc/drgnai_fixed.py --input-dir results/drgnai_fixed --epoch 100 -o output --method drgnai_fixed --gt-dir ./gt_vols --mask ./mask.mrc --apix 3.0 --num-vols 100 --num-imgs 1000

conda activate cryodrgn
python metrics/per_conf_fsc/drgnai_fixed_fsc.py -o output --method drgnai_fixed --gt-dir ./gt_vols --mask ./mask.mrc
python metrics/per_conf_fsc/per_conf_fsc_plot.py output --method drgnai_fixed

python metrics/per_conf_fsc/drgnai_fixed_fsc.py -o output --method drgnai_fixed --gt-dir ./gt_vols
python metrics/per_conf_fsc/per_conf_fsc_plot.py output --method drgnai_fixed --mask False