# cryosim: Tools for generating synthetic cryo-EM images
This repository is built upon https://github.com/ml-struct-bio/cryosim/tree/main
### Dependencies:
* cryodrgn version 3.4.0

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
  $ python integrate_files.py $(for i in {0..99}; do echo output_projections.${i}.mrcs; done) -o projection_particles.txt

  # Integrate all poses to make one pkl file
  $ python integrate_files.py $(for i in {0..99}; do echo pose.${i}.pkl; done) -o pose.combined.pkl 
```

### No noise addition, CTF values added
```
  # Apply CTF to one mrcs
  $ python add_ctf.py input_particles.mrcs --Apix 1.5 --ctf-pkl ctf.pkl --s1 0 --s2 0 -o output_particles_w_ctf.mrcs

  # Or generate 100 CTF added mrcs files
  $ for i in {0..99}; do python add_ctf.py input_particles.${i}.mrcs --Apix 1.5 --ctf-pkl ctf.${i}.pkl; done 
  $ python integrate_files.py $(for i in {0..99}; do echo output_particles_w_ctf.${i}.mrcs; done) -o CTF_added_particles.txt
```

### Add gaussian noise to SNR of 0.01
```
  # Add gaussian noise to SNR of 0.01
  $ python add_noise.py --mrcs input_noiseless_particles.mrcs -o output_noisy_particles.mrcs --snr 0.01

  # Or generate 100 noise added mrcs files
  $ for i in {0..99}; do python add_noise.py input_noiseless_particles.${i}.mrcs -o output_noisy_particles.mrcs --snr 0.01; done 
  $ python integrate_files.py $(for i in {0..99}; do echo output_noisy_particles.${i}.mrcs; done) -o noise_added_particles.txt
```

### Downsample particles
```
# Downsample from D = 256 to D = 128
$ for i in {0..99}
  do
      formatted_i=$(printf "%03d" "$i")
      cryodrgn downsample output_noisy_particles.${formatted_i}.mrcs -D 128 -o output_noisy_particles.${formatted_i}.128.mrcs
  done
```