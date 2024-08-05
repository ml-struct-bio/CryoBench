# CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM

Cryo-electron microscopy (cryo-EM) is a powerful technique for determining high-resolution 3D biomolecular structures from imaging data. As this technique can capture dynamic biomolecular complexes, 3D reconstruction methods are increasingly being developed to resolve this intrinsic structural heterogeneity. However, the absence of standardized benchmarks with ground truth structures and validation metrics limits the advancement of the field. Here, we propose CryoBench, a suite of datasets, metrics, and performance benchmarks for heterogeneous reconstruction in cryo-EM. We propose five datasets representing different sources of heterogeneity and degrees of difficulty. These include conformational heterogeneity generated from simple motions and random configurations of antibody complexes and from tens of thousands of structures sampled from a molecular dynamics simulation. We also design datasets containing compositional heterogeneity from mixtures of ribosome assembly states and 100 common complexes present in cells. We then perform a comprehensive analysis of state-of-the-art heterogeneous reconstruction tools including neural and non-neural methods and their sensitivity to noise, and propose new metrics for quantitative comparison of methods. We hope that this benchmark will be a foundational resource for analyzing existing methods and new algorithmic development in both the cryo-EM and machine learning communities.

## Documentation:

The latest documentation for CryoBench is available [homepage](https://cryobench.cs.princeton.edu/).

For any feedback, questions, or bugs, please file a Github issue, start a Github discussion, or email.

## Installation:
To run the metrics, you have to install `cryodrgn`.
`cryodrgn` may be installed via `pip`, and we recommend installing `cryodrgn` in a clean conda environment.

    # Create and activate conda environment
    (base) $ conda create --name cryodrgn python=3.9
    (cryodrgn) $ conda activate cryodrgn

    # install cryodrgn
    (cryodrgn) $ pip install cryodrgn

More installation instructions are found in the [documentation](https://ez-lab.gitbook.io/cryodrgn/installation).

## Metrics

### 1. Per-Conformation FSC
**Note** Example commands are in the `commands/metrics/IgG-1D/per_conf_fsc` and `commands/metrics/Ribosembly/per_conf_fsc`

Python files in `metrics/per_conf_fsc/` contain the code for computing the Per-Conformation FSC for each method. 
<details><summary><code>$ metric/per_conf_fsc</code></summary>

    usage for CryoDRGN: python metric/per_conf_fsc/cdrgn.py --input-dir INPUT --epoch EPOCH --apix APIX 
	-o OUTPUT --method METHOD --gt-dir GT [--mask MASK] --num-imgs N-IMGS --num-vols N-VOLS

    Sample the CTF from the experimental data, and set the apix and image size

    arguments:
      --input-dir INPUT  Directory contains weights, config, z for each method
      --epoch EPOCH		 Number of training epochs 
      --apix APIX    	 Path to save the integrated ctf file
	  -o OUTPUT			 Output directory
	  --method METHOD	 Type of methods (each method folder name)
	  --gt-dir GT		 Directory of gt volumes
	  --mask MASK (optional)
	  					 Use mask to compute the masked metric
	 --num-imgs N-IMGS	 Number of images per model (structure)
	 --num-vols N-VOLS	 Number of total reconstructed volumes

</details>
Example usage to compute Per-Conformation FSC for CryoDRGN.

    $ python metrics/per_conf_fsc/cdrgn.py --input-dir results/cryodrgn --epoch 19 --apix 3.0 -o output --method cryodrgn --gt-dir ./dataset/IgG-1D/vols --mask ./mask.mrc --num-imgs 1000 --num-vols 100

**Note** For the ab initio methods, you should align each reconstructed volume to the corresponding ground truth volume before computing the Per-Conformation FSC.

### 2. UMAP visualization
**Note** Example command is in the `commands/metrics/visualization/visualize_umap_IgG-1D.py`

<details><summary><code>$ metric/visualization/visualize_umap_IgG-1D</code></summary>

    usage for CryoDRGN: python metric/visualization/visualize_umap_IgG-1D.py --method METHOD -o OUTPUT --result-path RESULTS --num_imgs N-IMGS --num_vols N-VOLS

    arguments:
      --method METHOD  	 Method name -- folder name that contains UMAP (e.g. cryodrgn)
      -o O				 Output folder to save the UMAP plot
	  --result-path RESULTS
	  					 Path for the folder contains umap and latent before the method name
	  --num-imgs N-IMGS	 Number of images per model (structure)
	  --num-vols N-VOLS	 Number of total reconstructed volumes

</details>
Example usage to visualize UMAP colored by G.T for CryoDRGN.

    $ python metrics/visualization/visualize_umap_IgG-1D.py --method cryodrgn -o output/visualize_umap_igg1d --result-path results/IgG-1D --num_imgs 1000 --num_vols 100

## Image Formation
**Note:** Example command is in the `commands/IgG-1D_image_form.slurm`.

### 1. Sample CTF

First sample CTF from the experimental data using the `sampling_ctf` command:

<details><summary><code>$ img_form/sampling_ctf</code></summary>

    usage: python img_form/sampling_ctf.py --ctf-dir CTFS --ctf-file EXPERIMENTAL_CTF -o COMBINED_CTF [--N N] [--apix APIX]
                               [--img-size IMAGE_SIZE]
                               [--num-ctfs NUM_CTFS]

    Sample the CTF from the experimental data, and set the apix and image size

    positional arguments:
      --ctf-dir CTFS     Directory to save the sampled ctfs
      --ctf-file EXPERIMENTAL_CTF
                         Experimental ctf that we will sample from
      -o COMBINED_CTF    Path to save the integrated ctf file

    optional arguments:
      --N N              Number of models (default: 100)
      --apix APIX        A/PIX (default: 1.5)
      --img-size         Size of image (default: 256)
      --num-ctfs         Number of CTFs per model (= the number of image) (default:1000)

</details>

    $ python img_form/sampling_ctf.py --ctf-dir ctfs --ctf-file experimental_ctf.pkl -o combined_ctfs.pkl --apix 1.5 --img-size 256


### 2. Project 3D Volumes into 2D images

To create cryo-EM images, we rotate a volume, project it to 2D plane, and then translate the images. 
<details><summary><code>$ img_form/project3d</code></summary>

    usage: python img_form/project3d.py --mrc MRC [-N N] -o PARTICLES --out-pose POSES [--t-extent T] [-b B] [--out-png PNGS] [--apix APIX]

    positional arguments:
      --mrc MRC     	 Directory of input volumes (.mrc)
      -o PARTICLES		 Path to save the output projection stacks
      --out-pose POSES   Path to save the output poses

    optional arguments:
	  -N N				 Number of random projections
      --t-extent T		 Extent of image translation in pixels (default: +/-(default)s)
      --b B		         Minibatch size (default: 100)

</details>

Example usage to project volumes to images:

    $ python img_form/project3d.py --mrc vols -N 1000 -o 3d_projected --out-pose poses --t-extent 20 -b 50 --out-png pngs

### 3. Apply CTF

After projecting each volume, we apply CTF to each cryo-EM image.
Example usage to apply CTF:

    $ python img_form/add_ctf.py --particles 3d_projected --ctf-pkl ctfs --Apix 1.5 --s1 0 --s2 0 -o ctf_applied --out-png pngs

The `--particles` argument is the folder contain the projected images above and `--ctf-pkl` argument is directory of sampled ctfs saved in `sampling_ctf.py`.

### 4. Test pose/CTF parameters parsing

Add noise to CTF-applied images.
Example usage to add noise:

    $ python img_form/add_noise.py --mrcs add_ctf -o snr0.01 --snr 0.01 --out-png pngs --apix 1.5

The `--mrcs` argument is the folder contain the ctf-applied images above.

### 5. Downsample images

Resize your particle images using the `cryodrgn downsample` command:

<details><summary><code>$ cryodrgn downsample -h</code></summary>

    usage: cryodrgn downsample [-h] -D D -o MRCS [--is-vol] [--chunk CHUNK]
                               [--datadir DATADIR]
                               mrcs

    Downsample an image stack or volume by clipping fourier frequencies

    positional arguments:
      mrcs               Input images or volume (.mrc, .mrcs, .star, .cs, or .txt)

    optional arguments:
      -h, --help         Show this help message and exit
      -D D               New box size in pixels, must be even
      -o MRCS            Output image stack (.mrcs) or volume (.mrc)
      --is-vol           Flag if input .mrc is a volume
      --chunk CHUNK      Chunksize (in # of images) to split particle stack when
                         saving
      --relion31         Flag for relion3.1 star format
      --datadir DATADIR  Optionally provide path to input .mrcs if loading from a
                         .star or .cs file
      --max-threads MAX_THREADS
                         Maximum number of CPU cores for parallelization (default: 16)
      --ind PKL          Filter image stack by these indices

</details>

We recommend first downsampling images to 128x128 since larger images can take much longer to train:

    $ cryodrgn downsample [input particle stack] -D 128 -o particles.128.mrcs
	
## References:

For a complete description of the method, see:

* CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks
Ellen D. Zhong, Tristan Bepler, Bonnie Berger*, Joseph H Davis*
Nature Methods 2021, https://doi.org/10.1038/s41592-020-01049-4 [pdf](https://ezlab.princeton.edu/assets/pdf/2021_cryodrgn_nature_methods.pdf)

An earlier version of this work appeared at ICLR 2020:

* Reconstructing continuous distributions of protein structure from cryo-EM images
Ellen D. Zhong, Tristan Bepler, Joseph H. Davis*, Bonnie Berger*
ICLR 2020, Spotlight, https://arxiv.org/abs/1909.05215


## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.
