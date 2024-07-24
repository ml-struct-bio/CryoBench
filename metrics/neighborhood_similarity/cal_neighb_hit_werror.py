"""
Code to compute the neighborhood similarity

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

import pickle

import jax.numpy as jnp
import jax

def compute_loss(n_neighs,dist_points1_int, dist_points2_int):
    """
    Computes product of nearest neighbors matrices for two sets of points.

    Arguments
    ----------
    dist_points1_int: jax Array
        int 1 if neighbor 0 otherwise for the first set of points.
    dist_points2_int: jax Array
        int 1 if neighbor 0 otherwise for the second set of points.

    Returns
    --------
    loss: int
        Number of neighbors that match for the two sets of points

    """
    loss = jnp.sum(dist_points1_int*dist_points2_int)
    return loss

def compare_points_index(index, n_neighs, points1, points2):
    """
    Auxliary function for vmapping `compare_points`. This function computes the euclidean distances for each set of points with the reference point being points[index]. The distances are arg_sorted twice to obtain the rank. Neighbor matrix are used to computed_loss.

    If the number of neighbors is an Array, then it computes the loss for each number of neighbors by vmapping over compute_loss.

    Arguments
    ---------
    index: int
        index of the reference point (to compute distances)
    n_neighs int | Array:
        number of neighbors to consider (can be an iterable)
    points1: Array
        First set of points
    points2: Array
        Second set of points

    Returns
    --------
    loss: array with the loss for each number of neighbors

    """
    dist_points1 = jnp.argsort(jnp.argsort(jnp.linalg.norm(points1 - points1[index], axis=1))) < n_neighs[0] +1
    dist_points2 = jnp.argsort(jnp.argsort(jnp.linalg.norm(points2 - points2[index], axis=1))) < n_neighs[0] +1
   # print(dist_points1)
    dist_points1_int = dist_points1.astype(int)
    dist_points2_int = dist_points2.astype(int)

    compute_loss_ = jax.vmap(compute_loss, in_axes=(0, None, None))
    
    return compute_loss_(n_neighs,dist_points1_int, dist_points2_int)

def compare_points(n_neighs, points1, points2):
    """
    Computes the losses between points1 and points2.
    The loss is defined as the number of neighbors that match between the sets of points.
    
    Arguments
    ----------
    n_neighs: Array
        Number of neighbors to be considered 
    points1: Array
        First set of points
    points2: Array
        Second set of points

    Returns
    --------
    loss: Array["number of neighbors", "number of points"]
        The loss for each point, for different number of neighbors considered
    """

    assert points1.shape[0] == points2.shape[0], "The number of points must be equal"

    comp_points_map_index = jax.vmap(compare_points_index, in_axes=(0, None, None, None))

    return comp_points_map_index(np.arange(points1.shape[0]), n_neighs, points1, points2).T

def calculate_neigh_hits_k(start, points1, points2, k_neigh_range):
    """
    Computes the number of matching neighbors as a function of k

    Arguments
    ----------
    start: int
        Starting point to take embedding subsets. This is to calculate the error.
    points1: Array
        First set of points
    points2: Array
        Second set of points

    Returns
    --------
    neigh_hit_k: Array
        The mean number of matched neighbors for k 
    """
    points_gt = points1[start::5]
    points_embd = points2[start::5]

    neigh_hit_k = []

    for k in k_neigh_range:
        k_neighs = jnp.array([k])
        losses = compare_points(k_neighs,points_gt, points_embd)
        mean=losses.mean(1)[0]
        neigh_hit_k.append(mean)

    return neigh_hit_k


data = np.load('conf-het-1_wrangled_latents.npz')

embd_names = data.files

print(embd_names)

points_gt = data['gt_s1_embeddings']
points_gt = jnp.array(points_gt.copy())

k_neigh_range = np.arange(200, 4100, 200) 

for name in embd_names:

    points_embd = data[name]
    print(name,points_gt.shape,points_embd.shape)

    points_embd = jnp.array(points_embd.copy())

    neigh_hit_diff_start_k = []

    for start in range(0,5):
        neigh_hit = calculate_neigh_hits_k(start,points_gt,points_embd,k_neigh_range)
        neigh_hit_diff_start_k.append(neigh_hit)

    mean_neigh_hit_k = np.array(neigh_hit_diff_start_k)
    i = -1

    with open(f'{name}_output.txt', 'w') as file:
        for k in k_neigh_range:
            i += 1
            file.write(f"{k} {mean_neigh_hit_k.mean(0)[i]} {mean_neigh_hit_k.std(0)[i]}\n")

