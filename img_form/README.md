# cryosim: Tools for generating synthetic cryo-EM images
This repository is built upon https://github.com/ml-struct-bio/cryosim/tree/main
### Dependencies:
* cryodrgn version 3.4.0

### Example usage:
```
  # Sample CTF from experimental dataset without replacement
  $ python sampling_ctf.py --ctf-dir ./dataset/IgG-1D/ctfs --ctf-file experimental_ctf.pkl -o ./dataset/IgG-1D/combined_ctfs.pkl --apix 1.5 --img-size 256
  
  # Generate 1k projection images of a volume
  $ python project3d.py --mrc ./dataset/IgG-1D/vols/256 -N 1000 -o ./dataset/IgG-1D/3d_projected/mrcs --out-pose ./dataset/IgG-1D/3d_projected/poses --t-extent 20 -b 50 --out-png ./dataset/IgG-1D/3d_projected/pngs

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