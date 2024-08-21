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
* Zhong, Ellen D., et al. "CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks." Nature methods 18.2 (2021): 176-185. [github](https://github.com/ml-struct-bio/cryodrgn)

* Levy, Axel, et al. "Revealing biomolecular structure and motion with neural ab initio cryo-EM reconstruction." bioRxiv (2024): 2024-05. [github](https://github.com/ml-struct-bio/drgnai)

* Luo, Zhenwei, et al. "OPUS-DSD: deep structural disentanglement for cryo-EM single-particle analysis." Nature Methods 20.11 (2023): 1729-1738. [github](https://github.com/alncat/opusDSD)

* Punjani, Ali, and David J. Fleet. "3DFlex: determining structure and motion of flexible proteins from cryo-EM." Nature Methods 20.6 (2023): 860-870. [cryosparc](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-3d-flexible-refinement)

* Punjani, Ali, and David J. Fleet. "3D variability analysis: Resolving continuous flexibility and discrete heterogeneity from single particle cryo-EM." Journal of structural biology 213.2 (2021): 107702. [cryosparc](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-3d-variability-analysis-part-one)

* Gilles, Marc Aurele, and Amit Singer. "A Bayesian framework for cryo-EM heterogeneity analysis using regularized covariance estimation." bioRxiv (2023). [github](https://github.com/ma-gilles/recovar)

* Punjani, Ali, et al. "cryoSPARC: algorithms for rapid unsupervised cryo-EM structure determination." Nature methods 14.3 (2017): 290-296.


## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.
