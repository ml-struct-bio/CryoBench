# CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM

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

## Image Formation
**Note** Look at the repo [cryosim](https://github.com/ml-struct-bio/CryoBench/tree/main/cryosim).

## Metrics

### 1. Per-Conformation FSC
**Note** Look at the repo [metrics/per_conf_fsc](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/per_conf_fsc) and [metrics/per_conf_fsc/Ribosembly](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/per_conf_fsc_Ribosembly).

### 2. UMAP visualization
**Note** Look at the repo [metrics/visualization](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/visualization)

	
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
