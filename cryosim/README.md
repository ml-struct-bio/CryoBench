# cryosim: Tools for generating synthetic cryo-EM images
This repository is built upon https://github.com/ml-struct-bio/cryosim/tree/main
### Dependencies:
* cryodrgn version 3.4.0

### Example usage:
```
  # Generate 1k projection images of a volume
  $ python project3d.py input.mrc -N 1000 -o ./dataset/IgG-1D/3d_projected/mrcs --out-pose ./dataset/IgG-1D/3d_projected/poses --t-extent 20 -b 50 --out-png ./dataset/IgG-1D/3d_projected/pngs

  # No noise addition, CTF values added
  $ python add_ctf.py --particles ./dataset/IgG-1D/3d_projected/mrcs --ctf-pkl ./dataset/IgG-1D/ctfs --Apix 1.5 --s1 0 --s2 0 -o ./dataset/IgG-1D/add_ctf/mrcs --out-png ./dataset/IgG-1D/add_ctf/pngs
  
  # Add gaussian noise to SNR of 0.01
  $ python add_noise.py --mrcs ./dataset/IgG-1D/add_ctf/mrcs -o ./dataset/IgG-1D/add_noise/256/snr0.01/mrcs --invert --snr 0.01 --out-png ./dataset/IgG-1D/add_noise/256/snr0.01/pngs

  # Downsample to 128 (from 256)
  $ for i in {0..99}
    do
        formatted_i=$(printf "%03d" "$i")
        cryodrgn downsample ./dataset/IgG-1D/add_noise/256/snr0.01/mrcs/${formatted_i}_particles.mrcs -D 128 -o ./dataset/IgG-1D/add_noise/128/snr0.01/mrcs/${formatted_i}_particles_128.mrcs
    done
  
  # Integrate pickle files
  $ python integrate_files.py --poses-dir ./dataset/IgG-1D/3d_projected/poses/ --integrated-pose ./dataset/IgG-1D/combined_poses.pkl --mrcs ./dataset/IgG-1D/add_noise/128/snr0.01/mrcs/
```

### Generating CTF parameters
```
  # Subsample 100k CTF parameters from an experimental dataset without replacement
  $ python subsample_ctf.py experimental_ctf.pkl -o ctf.pkl -N 100000 --Apix 1.5 -D 256 --seed 0
  
  # Or create 100 separate files for each GT conformation and combine
  $ for i in {0..99}; do python subsample_ctf.py experimental_ctf.pkl -N 1000 -D 256 --Apix 1.5 --seed $i -o ctf.${i}.pkl; done 
  $ python integrate_files.py $(for i in {0..99}; do echo ctf.${i}.pkl; done) -o ctf.combined.pkl 
```

### Generate projection images of a volume
```
  # Generate 1k projection images from a volume
  $ python project3d.py input.mrc -N 1000 -o output_projections.mrcs --out-pose poses.pkl --t-extent 20

  # Or generate 1k projection images for 100 volumes (Total 100k images)
  $ for i in {0..99}; do python project3d.py input.${i}.mrc -N 1000 -o output_projections.${i}.mrcs --out-pose poses.${i}.pkl --t-extent 20; done 
  $ python integrate_files.py $(for i in {0..99}; do echo output_projections.${i}.mrcs; done) -o particles.txt
```