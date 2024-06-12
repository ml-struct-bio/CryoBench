conda activate drgnai
python metrics/option_c/drgnai_abinit_gen_vol.py --input-dir results/drgnai_abinit --epoch 100 --apix 3.0 -o output --method drgnai_abinit --gt-dir ./gt_vols --num-vols 100 --num-imgs 1000
python metrics/option_c/align_multi.py output/drgnai_abinit/option_c/vols --apix 3.0 --org-vol ./gt_vols