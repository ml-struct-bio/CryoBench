# CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM

## Documentation

The latest documentation for CryoBench is available at our [homepage](https://cryobench.cs.princeton.edu/) and also at
our [manual](https://ez-lab.gitbook.io/cryobench).

For any feedback, questions, or bugs, please file a Github issue, start a Github discussion, or email.

## Installation
To run the metrics, you have to install `cryodrgn`.
`cryodrgn` may be installed via `pip`, and we recommend installing `cryodrgn` in a clean conda environment.

    # Create and activate conda environment
    (base) $ conda create --name cryodrgn python=3.9
    (cryodrgn) $ conda activate cryodrgn

    # install cryodrgn
    (cryodrgn) $ pip install cryodrgn

More installation instructions are found in the [documentation](https://ez-lab.gitbook.io/cryodrgn/installation).

Datasets are available for download at Zenodo.

1. Conf-het: [https://zenodo.org/records/11629428](https://zenodo.org/records/11629428).
2. Comp-het: [https://zenodo.org/records/12528292](https://zenodo.org/records/12528292).
3. Spike-MD: [https://zenodo.org/records/14902250](https://zenodo.org/records/14902250).

## Image Formation
Look at the repo [cryosim](https://github.com/ml-struct-bio/CryoBench/tree/main/cryosim).

## Metrics

### 1. Per-image FSCs
Look at the repo [metrics/fsc](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/fsc)

### 2. UMAP visualization
Look at the repo [metrics/visualization](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/visualization)


## References:

Jeon, Minkyu, et al. "CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM." arXiv preprint arXiv:2408.05526 (2024) [paper](https://arxiv.org/abs/2408.05526).

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.
