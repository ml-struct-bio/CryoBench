# CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM

## Documentation:

The latest documentation for CryoBench is available [homepage](https://cryobench.cs.princeton.edu/).

For any feedback, questions, or bugs, please file a Github issue, start a Github discussion, or email.

## Installation:

To run the script that calculates the neighborhood similarity, please first install [JAX](https://jax.readthedocs.io/en/latest/installation.html).

    # install jax
    $ pip install -U jax


## Neighborhood Similarity
The neighborhood similarity quantifies the percentage of matching neighbors with respect to the ground truth that are found within a neighborhood radius `k`.


### Example 
In the repo [metrics/neighborhood_similarity](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/neighborhood_similarity) , you can find `cal_neighb_hit_werror.py` that calculates the neighborhood similarity between ground truth embeddings and several sets of embeddings from reconstruction algorithms found in  `conf-het-1_wrangled_latents.npz`:

	$ python cal_neighb_hit_werror.py

The output files are the neighborhood similarity as a function of the neighborhood radius for each reconstruction algorithm. 

## References:

Jeon, Minkyu, et al. "CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM." arXiv preprint arXiv:2408.05526 (2024) [paper](https://arxiv.org/abs/2408.05526).

Boggust, Angie et a. "Embedding comparator: Visualizing differences in global structure and local neighborhoods via small multiples." In 27th international
conference on intelligent user interfaces, pages 746–766, 2022.

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.
