conda activate recovar
python metrics/recovar/gen_reordered_z.py --recovar-result-dir ./recovar_results
python metrics/recovar/gen_vol_for_per_conf_fsc.py recovar_results -o output/recovar/per_conf_fsc/vols --zdim 10 --num-imgs 1000 --num-vols 100
conda deactivate recovar

conda activate cryodrgn
python metrics/per_conf_fsc/recovar.py --input-dir output/recovar/per_conf_fsc/vols -o output --method recovar --gt-dir ./gt_vols --mask ./mask.mrc
python metrics/per_conf_fsc/recovar.py --input-dir output/recovar/per_conf_fsc/vols -o output --method recovar --gt-dir ./gt_vols 
python metrics/per_conf_fsc/per_conf_plot.py output --method recovar --mask False
python metrics/per_conf_fsc/per_conf_plot.py output --method recovar