#!/bin/bash
#SBATCH --job-name=information_imbalance
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=24:00:00

python compute_information_imbalance.py --method conf_het_1   --subset_size 2000   --input_latents_fname confhet1_wrangled_latents.npz      --output_information_imbalance_fname confhet1_information_imbalance.csv     --uniq_ks 1,3,10,30,100,300  1> stdout.txt 2> stderr.txt
python compute_information_imbalance.py --method conf_het_2   --subset_size 2000   --input_latents_fname confhet2_wrangled_latents.npz      --output_information_imbalance_fname confhet2_information_imbalance.csv     --uniq_ks 1,3,10,30,100,300  1> stdout.txt 2> stderr.txt
python compute_information_imbalance.py --method assemble_het --subset_size 2000   --input_latents_fname assemblehet_wrangled_latents.npz   --output_information_imbalance_fname assemblehet_information_imbalance.csv  --uniq_ks 1,3,10,30,100,300  1> stdout.txt 2> stderr.txt
python compute_information_imbalance.py --method mix_het      --subset_size 2000   --input_latents_fname mixhet_wrangled_latents.npz        --output_information_imbalance_fname mixhet_information_imbalance.csv       --uniq_ks 1,3,10,30,100,300  1> stdout.txt 2> stderr.txt
python compute_information_imbalance.py --method md           --subset_size 2000   --input_latents_fname MD_wrangled_latents.npz            --output_information_imbalance_fname MD_information_imbalance.csv           --uniq_ks 1,3,10,30,100,300  1> stdout.txt 2> stderr.txt
