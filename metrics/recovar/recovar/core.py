import jax
import jax.numpy as jnp
import numpy as np
import functools
from jax import vjp
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jax.numpy)

## Low level functions for cryo-EM. Handling of indexing, slicing, adjoint of slicing, CTF, etc.

# Three different representation. E.g. for a 2x2x2 grid  
# vol_indices are [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]
# vec_indices are 0, 1, 2, 3, 4, 5, 6, 7
# frequencies are [-1, -1, -1], [-1, -1, 0], etc ...

@functools.partial(jax.jit, static_argnums=[1])
def vol_indices_to_vec_indices(vol_indices, vol_shape):
    # good_idx = check_vol_indices_in_bound(vol_indices,vol_shape[0])
    og_shape = vol_indices.shape
    vol_indices = vol_indices.reshape(-1, vol_indices.shape[-1])
    vec_indices = jnp.ravel_multi_index(vol_indices.T, vol_shape, mode = 'clip').reshape(og_shape[:-1])

    return vec_indices#jnp.where(good_idx, vec_indices, -1)
    # return jnp.ravel_multi_index(vol_indices.T, vol_shape, mode = 'clip')

def vec_indices_to_vol_indices(vec_indices, vol_shape):
    vol_indices = jnp.unravel_index(vec_indices, vol_shape)
    vol_indices = jnp.stack(vol_indices, axis = -1)
    return vol_indices

def vol_indices_to_frequencies(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    # ((vol_shape - (vol_shape%2 ==1))//2 * 1)
    mid_grid =  ((vol_shape - (vol_shape%2 ==1))//2 * 1).astype(int)
    return vol_indices - mid_grid

def frequencies_to_vol_indices(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid =  ((vol_shape - (vol_shape%2 ==1))//2 * 1).astype(int)
    return vol_indices + mid_grid

def vec_indices_to_frequencies(vec_indices, vol_shape):
    return vol_indices_to_frequencies(vec_indices_to_vol_indices(vec_indices, vol_shape), vol_shape)

def frequencies_to_vec_indices(frequencies, vol_shape):
    return vol_indices_to_vec_indices(frequencies_to_vol_indices(frequencies, vol_shape), vol_shape)

def check_frequencies_in_bound(frequencies,grid_size):
    return jnp.prod((frequencies >= -grid_size/2 ) * (frequencies < grid_size/2), axis = -1).astype(bool)

def check_vol_indices_in_bound(vol_indices,grid_size):
    return jnp.prod((vol_indices >= 0 ) * (vol_indices < grid_size ), axis = -1).astype(bool)

def check_vec_indices_in_bound(vec_indices,grid_size):
    return ((vec_indices < grid_size**3 ) * vec_indices >= 0).astype(bool)


# This function is meant to handle the heterogeneity case.
def multi_vol_indices_to_multi_vec_indices(vol_indices, het_indices, vol_shape, het_shape):
    og_shape = vol_indices.shape
    vol_indices = jnp.concatenate( [het_indices, vol_indices.reshape(-1, vol_indices.shape[-1]) ], axis=-1)
    multi_vol_shape = (het_shape, *vol_shape)
    vec_indices = jnp.ravel_multi_index(vol_indices.T, multi_vol_shape, mode = 'clip').reshape(og_shape[:-1])

    return





## Some similar function used for adaptive discretization: find nearby gridpoints
@functools.partial(jax.jit, static_argnums=[1])
def find_frequencies_within_grid_dist(coords, max_grid_dist: int ):
    # k 
    dim = coords.shape[-1]

    # Closest ind
    # closest_point = round_to_int(coord)
    if dim ==2:
        neighbors = jnp.mgrid[-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1]
    else:
        neighbors = jnp.mgrid[-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1,-max_grid_dist:max_grid_dist+1]

    neighbors = neighbors.reshape(dim, -1).T

    coords_ndim = coords.ndim
    neighbors = neighbors.reshape( (coords_ndim-1) * [1]  + list(neighbors.shape))
    neighbors += coords[...,None,:]
    if coords.dtype.kind == 'f':
        neighbors = round_to_int(neighbors)
    return neighbors

## This could be improved...
# @functools.partial(jax.jit, static_argnums=[1])
# def find_nearest_k_frequencies(coord, k: int ):
#     # (1 + n)^2 (1 + 4 n + 2 n^2)
#     # 1/2 (-1 + k^(1/3))
#     #  
#     neighbors = find_frequencies_within_grid_dist(coord, max_grid_dist: int )
#     distances = jnp.linalg.norm(neighbors - coord, axis = 0)
#     # Now find k smallest distances
#     distances_at_best, indices = jax.lax.top_k(-distances, k)
    # return

batch_find_frequencies_within_grid_dist= jax.vmap(find_frequencies_within_grid_dist, in_axes = (0, None))
batch_batch_find_frequencies_within_grid_dist= jax.vmap(batch_find_frequencies_within_grid_dist, in_axes = (0, None))


# def find_frequencies_within_dist(coord, max_grid_dist):
#     dim = coord.shape[-1]
#     k = max_grid_dist * 2**dim - 1
#     return find_k_nearest_frequencies(coord, k )

def distance_to_max_grid_dist(dist):
    return np.ceil(dist).astype(int)


# There are two ways to slice volumes. One fast which just uses nearest gridpoints, and one slow which calls map_coordinates, which can use either both nearest and linear interpolation.
@jax.jit
def slice_volume_by_nearest(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid] 

# Used to project the mean
batch_slice_volume_by_nearest = jax.vmap(slice_volume_by_nearest, (None, 0))

def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    elif disc_type == "nearest":
        return 0
    elif disc_type == "cubic":
        return 3
    else:
        raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")

@functools.partial(jax.jit, static_argnums = [2,3,4])    
def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):    
    order = decide_order(disc_type)
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order)
    

@functools.partial(jax.jit, static_argnums=[2,3,4])
def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type):  
    volume_size = np.prod(volume_shape)
    f = lambda volume : slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    _, u = vjp(f,jnp.zeros(volume_size, dtype = slices.dtype ))
    return u(slices)[0]

# ## This is about twice faster. It doesn't do any checking, I guess? 
# def slice_volume_by_map_nomap(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type):
#     grid_point_indices = batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape, grid_size )
#     return volume[grid_point_indices] 


# Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
# @functools.partial(jax.jit, static_argnums=0)
# def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids):
#     volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
#     volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
#     return volume_vec

# This is what people called the backprojection
# Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
@functools.partial(jax.jit, static_argnums=0)
def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids, volume_vec = None):
    if volume_vec is None:
        volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
    volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
    return volume_vec

batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))


# def batch_over_vol_summed_adjoint_slice_by_nearest_add():
#     return jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))
# batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None, -1), out_axes = ( -1))


# # Computes \sum_i S_i v_i where S_i: N^2 -> N^3 is sparse, v_i \in N^2
# @functools.partial(jax.jit, static_argnums=0)
# def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids):
#     volume_vec = jnp.zeros(volume_size, dtype = image_vecs.dtype)
#     volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
#     return volume_vec


# batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, -1,None), out_axes = ( -1))

nosummed_adjoint_slice_by_nearest = jax.vmap( summed_adjoint_slice_by_nearest, in_axes = (None, 0,0)) 

@functools.partial(jax.jit, static_argnums=0)
def sum_adj_forward_model(volume_size, images, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return summed_adjoint_slice_by_nearest(volume_size, images * jnp.conj(CTF_val_on_grid_stacked), plane_indices_on_grid_stacked)


@jax.jit
def forward_model(volume_vec, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return batch_slice_volume_by_nearest(volume_vec, plane_indices_on_grid_stacked) * CTF_val_on_grid_stacked


@jax.jit
def translate_single_image(image, translation, lattice):
    phase_shift = jnp.exp( 1j * -2 * jnp.pi * (lattice @ translation[:,None] ) )[...,0]
    return image.reshape(-1) * phase_shift


batch_translate = jax.vmap(translate_single_image, in_axes = (0,0, None))

@functools.partial(jax.jit, static_argnums=2)
def translate_images(image, translation, image_shape):
    twod_lattice = get_unrotated_plane_coords(image_shape, voxel_size =1, scaled = True )[:,:2]
    return batch_translate(image, translation, twod_lattice)#.astype(image.dtype)


batch_trans_translate_images = jax.vmap(translate_images, in_axes = (None,-2, None), out_axes = (-2))


def get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor = 1):
    # This is used to handle discretization of projection 
    unrotated_plane_indices =  ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = 1, scaled = False)
    unrotated_plane_indices = jnp.concatenate( [unrotated_plane_indices, jnp.zeros(unrotated_plane_indices.shape[0], dtype = unrotated_plane_indices.dtype )[...,None] ], axis = -1)
    # If we are 
    # 
    return unrotated_plane_indices * three_d_upsampling_factor


def get_unrotated_plane_coords(image_shape, voxel_size, scaled =True ):
    # These are scaled with voxel size. This is only used for CTF and translations
    plane_coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size = voxel_size, scaled = scaled)
    plane_coords = jnp.concatenate( [plane_coords, jnp.zeros(plane_coords.shape[0], dtype = plane_coords.dtype)[...,None] ], axis = -1)
    return plane_coords



## JITTING THIS MAY CAUSE WEIRD ERRORS...
# @functools.partial(jax.jit, static_argnums=[1,2,3])
def get_nearest_gridpoint_indices(rotation_matrix, image_shape, volume_shape):
    rotated_plane = get_gridpoint_coords(rotation_matrix, image_shape, volume_shape)
    rotated_indices = round_to_int(rotated_plane)
    rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_indices


@functools.partial(jax.jit, static_argnums=[1,2])
def get_gridpoint_coords(rotation_matrix, image_shape, volume_shape):

    three_d_upsampling_factor = volume_shape[0] // image_shape[0]
    unrotated_plane_indices = get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor = three_d_upsampling_factor)
    rotated_plane = jnp.matmul(unrotated_plane_indices, rotation_matrix, precision = jax.lax.Precision.HIGHEST)
    # I have no idea why there is a random +1 here
    rotated_coords = rotated_plane + jnp.floor(1.0 * volume_shape[0]/2)
    
    # rotated_indices = vol_indices_to_vec_indices(rotated_indices, volume_shape)
    # Note I am being very sloppy with out of bound stuff. JAX just ignores them, which is what we want, but this doomed to cause bugs eventually.
    # indices_in_grid = jnp.prod((rotated_indices >= 0 ) * (rotated_indices < grid_size), axis = -1).astype(bool)
    return rotated_coords

@jax.jit
def round_to_int(array):
    return jax.lax.round(array).astype(jnp.int32)


batch_get_nearest_gridpoint_indices = jax.vmap(get_nearest_gridpoint_indices, in_axes =(0, None, None) ) 
batch_get_gridpoint_coords = jax.vmap(get_gridpoint_coords, in_axes =(0, None, None) ) 


@functools.partial(jax.jit, static_argnums=[3,4,5,6,7])
def forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):
    slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)
    return slices

batch_forward_model_from_map = jax.vmap(forward_model_from_map, in_axes = (0, 0, 0, None, None, None, None, None) )



@functools.partial(jax.jit, static_argnums=[3,4,5,6,7])
def forward_model_from_map_and_return_adjoint(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type)
    slices, f_adj = vjp(f,volume)
    return slices, f_adj


# A JAXed version of the adjoint. This is actually slightly slower but will run with disc_type = 'linear_interp'. I probably should just write out an explicit adjoint of linear interpolation...
@functools.partial(jax.jit, static_argnums=[3,4,5,6,7])
def adjoint_forward_model_from_map(slices, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):  
    volume_size = np.prod(volume_shape)
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type)
    _, u = vjp(f,jnp.zeros(volume_size, dtype = slices.dtype ))
    return u(slices)[0]


@functools.partial(jax.jit, static_argnums=[3,4,5,6,7])
def adjoint_forward_model_from_trilinear(slices, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):  

    return adjoint_slice_volume_by_trilinear(slices * CTF_fun( CTF_params, image_shape, voxel_size), rotation_matrices, image_shape, volume_shape, volume = None)




# Compute A^TAx (the forward, then its adjoint). For JAX reasons, this should be about 2x faster than doing each call separately.
@functools.partial(jax.jit, static_argnums=[3,4,5,6,7])
def compute_A_t_Av_forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type):    
    f = lambda volume : forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type)
    y, u = vjp(f,volume)
    return u(y)



    
# @functools.partial(jax.jit, static_argnums = [2,3,4])    
# def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order):
#     # import pdb; pdb.set_trace()
#     # batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape, grid_size )
#     # batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
#     # batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T
#     batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    
#     # slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)

#     if order ==3:
#         from recovar import cryojax_map_coordinates
#         slices = cryojax_map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
#     else:
#         slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)

#     return slices


def rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape):
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T
    return batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape


# from recovar import jax_map_squared
# @functools.partial(jax.jit, static_argnums = [2,3,4])    
# def map_coordinates_squared_on_slices(volume, rotation_matrices, image_shape, volume_shape, disc_type):
#     order = 1 if disc_type == "linear_interp" else 0
#     batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
#     slices = jax_map_squared.map_coordinates_squared(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
#     return slices

# # Not sure this should be used...
# def slice_by_linear_interp_explicit(volume, rotation_matrices, image_shape, volume_shape, order):
#     batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape )
#     # Get all 8 grid-points    
#     return

# # def get_nearby_grid
# def mesh_grid_one(x_grid):
#     z = jnp.meshgrid(low_high, low_high)
#     return jnp.concatenate()
# mesh_grid_coords = jax.vmap(mesh_grid_coords, in_axes=(-1))

def get_stencil(dim):
    if dim==2:
        return jnp.array([[0, 0], [0,1], [1,0], [1,1]], dtype = int)
    if dim==3:
        return jnp.array([[0, 0, 0], [0, 0,1], [0, 1,0], [0, 1,1], \
                         [1, 0, 0], [1, 0,1], [1, 1,0], [1, 1,1]] , dtype = int)


#
def get_trilinear_weights_and_vol_indices(grid_coords, volume_shape):
    # lower_grid_points = jnp.floor(grid_points).astype(int)
    lower_points_ndim = grid_coords.ndim-1
    all_grid_points = jnp.floor(grid_coords)[...,None,:] + get_stencil(grid_coords.shape[-1]).reshape( [*(lower_points_ndim * [1]), 8,3])

    all_weights = (1 - jnp.abs(grid_coords[...,None,:] - all_grid_points )).astype(np.float32)
    all_weights = jnp.where(all_weights > 0, all_weights, 0)
    all_weights = jnp.prod(all_weights, axis=-1)

    # all_weights = jnp.where(all_weights > 0, all_weights, 0)
    #**2
    # all_weights /= jnp.linalg.norm(all_weights, axis=-1, keepdims=True)

    # Zero-out out of bound for good measure.
    good_points = check_vol_indices_in_bound(all_grid_points, volume_shape[0])
    all_weights *= good_points
    # assert False

    return all_grid_points.astype(jnp.int32), all_weights

# @functools.partial(jax.jit, static_argnums = [2,3,4])    
def slice_volume_by_trilinear(volume, rotation_matrices, image_shape, volume_shape):    
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices( grid_points, volume_shape)
    sliced_volume = jnp.sum(volume[grid_vec_indices.reshape(-1)].reshape(grid_vec_indices.shape) * weights, axis=-1)
    return sliced_volume.reshape(grid_coords_og_shape[:-1]).astype(volume.dtype)


# @functools.partial(jax.jit, static_argnums = [2,3,4])    
## UNTESTED
def adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volume = None):    
    grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype = images.dtype)

    weights *= images.reshape(-1,1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))

    return volume




@functools.partial(jax.jit, static_argnums = [2,3,4])    
def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order):
    # import pdb; pdb.set_trace()
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape )
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1,3).T

    if order ==3:
        from recovar import cryojax_map_coordinates
        slices = cryojax_map_coordinates.map_coordinates_with_cubic_spline(volume, batch_grid_pt_vec_ind_of_images, mode = 'fill', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)
    else:
        slices = jax.scipy.ndimage.map_coordinates(volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, order = order, mode = 'constant', cval = 0.0).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1] ).astype(volume.dtype)

    return slices

# batch volumes
batch_vol_rot_slice_volume_by_map = jax.vmap(slice_volume_by_map, in_axes = (0, 0, None, None, None) )

batch_translate_images = jax.vmap(translate_images, in_axes = (0, 0, None) )

# TODO: Should it be residual of masked?
# Residual will be 4 dimensional
# volumes_batch x images_batch x rotations_batch x translations_batch x  
# @functools.partial(jax.jit, static_argnums = [7,8,9,10,11,12])    
def compute_residuals_many_poses(volumes, images, rotation_matrices, translations, CTF_params, noise_variance, voxel_size, volume_shape, image_shape, disc_type, CTF_fun ):
    

    assert(rotation_matrices.shape[0] == volumes.shape[0])
    assert(rotation_matrices.shape[1] == images.shape[0])

    assert(translations.shape[0] == volumes.shape[0])
    assert(translations.shape[1] == images.shape[0])


    # n_vols x rotations x image_size
    projected_volumes = batch_vol_rot_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type)
    projected_volumes = projected_volumes * CTF_fun( CTF_params, image_shape, voxel_size)

    translated_images = translate_images(images, translations, image_shape)

    norm_res_squared = jnp.linalg.norm((projected_volumes - translated_images) / jnp.sqrt(noise_variance), axis = (-1))
    return norm_res_squared





## CTF functions.
@jax.jit
def evaluate_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor):
    '''
    Adapted for JAX from cryoDRGN: https://github.com/zhonge/cryodrgn
    
    
    Compute the 2D CTF
   
    Input: 
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees 
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * jnp.pi / 180
    phase_shift = phase_shift * jnp.pi / 180
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2

    # LAMBDA IN CRYODRGN
    # lam = 12.2639 / (volt + 0.97845e-6 * volt**2)**.5
    # LAMBDA IN RELION
    lam = 12.2642598 / jnp.sqrt(volt * (1. + volt * 9.78475598e-7))
    # 	lambda=12.2643247 / sqrt(local_kV * (1. + local_kV * 0.978466e-6)); // See http://en.wikipedia.org/wiki/Electron_diffraction

    # This causes a CTF difference of about 0.6% at 300 kV, which seems kinda big.

    x = freqs[...,0]    
    y = freqs[...,1]
    ang = jnp.arctan2(y,x)
    s2 = x**2 + y**2
    df = .5*(dfu + dfv + (dfu-dfv)*jnp.cos(2*(ang-dfang)))
    gamma = 2*jnp.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    ctf = (1-w**2)**.5*jnp.sin(gamma) - w*jnp.cos(gamma) 
    if bfactor is not None:
        ctf *= jnp.exp(-bfactor/4*s2)
    return ctf

@jax.jit
def evaluate_ctf_packed(freqs, CTF):
    return evaluate_ctf(freqs, CTF[0], CTF[1], CTF[2], CTF[3], CTF[4], CTF[5], CTF[6], CTF[7]) * CTF[8]

batch_evaluate_ctf = jax.vmap(evaluate_ctf_packed, in_axes = (None, 0))

CTF_FUNCTION_OPTION = "cryodrgn"

def evaluate_ctf_wrapper(CTF_params, image_shape, voxel_size, CTF_FUNCTION_OPTION ):
    return cryodrgn_CTF(CTF_params, image_shape, voxel_size)

    # if CTF_FUNCTION_OPTION == "dynamight":
    #     return dynamight_CTF_wrapper(CTF_params, image_shape, voxel_size, 0)
    # if CTF_FUNCTION_OPTION == "cryodrgn_antialiasing":

    #     grid_size = image_shape[0]
    #     large_image_shape = (image_shape[0]*2, image_shape[1]*2)
    #     psi = get_unrotated_plane_coords(large_image_shape, voxel_size*1, scaled = True)[...,:2]
    #     CTF_large_grid = batch_evaluate_ctf(psi, CTF_params).reshape([CTF_params.shape[0], *large_image_shape])

    #     CTF_large_grid_ft = ftu.get_dft2(CTF_large_grid)

    #     # CTF_ft_windowed = padding.unpad_images_spatial_domain(CTF_ft_windowed, image_shape)
    #     # CTF_ft_windowed = CTF_large_grid_ft[:,grid_size//2:-grid_size//2,grid_size//2:-grid_size//2]
    #     o = 2
    #     CTF_ft_windowed = equinox.nn.AvgPool2d(o+o//2,o)(CTF_large_grid_ft)

    #     CTF = ftu.get_idft2(CTF_ft_windowed) / 4

    #     CTF2 = CTF_large_grid[:,grid_size//2:-grid_size//2,grid_size//2:-grid_size//2]

    #     psi = get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2]
    #     CTF3 = batch_evaluate_ctf(psi, CTF_params).reshape([CTF_params.shape[0], *image_shape])

    #     # Downsample
    #     from recovar import padding

    #     # CTF_large_grid_ft = padding.unpad_images_spatial_domain(CTF_large_grid_ft, image_shape)
    #     # CTF_large_grid_ft = ftu.get_dft2(CTF_large_grid, large_image_shape)
    #     # CTF = padding.unpad_images_fourier_domain(CTF_large_grid, large_image_shape, image_shape[0])
    #     # import pdb; pdb.set_trace()
    #     return CTF
    
    # if CTF_FUNCTION_OPTION == "cryodrgn_zeroed":
    #     CTF = cryodrgn_CTF(CTF_params, image_shape, voxel_size)
    #     CTF = CTF.reshape([CTF_params.shape[0], *image_shape])
    #     CTF = CTF.at[:,1::2,:].set(0)
    #     CTF = CTF.at[:,:,1::2].set(0)
    #     return CTF.reshape([CTF_params.shape[0], -1])
    # return cryodrgn_CTF(CTF_params, image_shape, voxel_size)

def cryodrgn_CTF(CTF_params, image_shape, voxel_size):
    psi = get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2]
    return batch_evaluate_ctf(psi, CTF_params)

# def evaluate_ctf_wrapper(CTF_params, image_shape, voxel_size):
#     if CTF_FUNCTION_OPTION == "dynamight":
#         return dynamight_CTF_wrapper(CTF_params, image_shape, voxel_size, 1)
    
#     psi = get_unrotated_plane_coords(image_shape, voxel_size, scaled = True)[...,:2]
#     return batch_evaluate_ctf(psi, CTF_params)




### CTF functions taken from dynamight and JAXed up, which does anti-aliasing business.
# See https://github.com/3dem/DynaMight/blob/616360b790febf56edf08aef5d4c414058194376/dynamight/data/handlers/ctf.py#L16

# class ContrastTransferFunction:
#     def __init__(
#             self,
#             voltage: float,
#             spherical_aberration: float = 0.,
#             amplitude_contrast: float = 0.,
#             phase_shift: float = 0.,
#             b_factor: float = 0.,
#     ) -> None:
#         np = jnp
#         """
#         Initialization of the CTF parameter for an optics group.
#         :param voltage: Voltage
#         :param spherical_aberration: Spherical aberration
#         :param amplitude_contrast: Amplitude contrast
#         :param phase_shift: Phase shift
#         :param b_factor: B-factor
#         """

#         self.voltage = voltage
#         self.spherical_aberration = spherical_aberration
#         self.amplitude_contrast = amplitude_contrast
#         self.phase_shift = phase_shift
#         self.b_factor = b_factor

#         # Adjust units
#         spherical_aberration = spherical_aberration * 1e7
#         voltage = voltage * 1e3

#         # Relativistic wave length
#         # See http://en.wikipedia.org/wiki/Electron_diffraction
#         # lambda = h/sqrt(2*m*e) * 1/sqrt(V*(1+V*e/(2*m*c^2)))
#         # h/sqrt(2*m*e) = 12.2642598 * 10^-10 meters -> 12.2642598 Angstrom
#         # e/(2*m*c^2)   = 9.78475598 * 10^-7 coulombs/joules
#         lam = 12.2642598 / np.sqrt(voltage * (1. + voltage * 9.78475598e-7))

#         # Some constants
#         self.c1 = -np.pi * lam
#         self.c2 = np.pi / 2. * spherical_aberration * lam ** 3
#         self.c3 = phase_shift * np.pi / 180.
#         self.c4 = -b_factor/4.
#         self.c5 = \
#             np.arctan(
#                 amplitude_contrast / np.sqrt(1-amplitude_contrast**2)
#             )

#         self.xx = {}
#         self.yy = {}
#         self.xy = {}
#         self.n4 = {}


#     def __call__(
#             self,
#             grid_size: int,
#             pixel_size: float,
#             u,
#             v,
#             angle,
#             h_sym: bool = False,
#             antialiasing: int = 0
#     ) :
#         """
#         Get the CTF in an numpy array, the size of freq_x or freq_y.
#         Generates a Numpy array or a Torch tensor depending on the object type
#         on freq_x and freq_y passed to the constructor.
#         :param u: the U defocus
#         :param v: the V defocus
#         :param angle: the azimuthal angle defocus (degrees)
#         :param antialiasing: Antialiasing oversampling factor (0 = no antialiasing)
#         :param grid_size: the side of the box
#         :param pixel_size: pixel size
#         :param h_sym: Only consider the hermitian half
#         :return: Numpy array or Torch tensor containing the CTF
#         """
#         torch = jnp
        
#         freq_x, freq_y = self._get_freq(grid_size, pixel_size, h_sym, antialiasing)
#         xx = freq_x**2
#         yy = freq_y**2
#         xy = freq_x * freq_y
#         n4 = (xx + yy)**2  

#         angle = angle * np.pi / 180
#         acos = torch.cos(angle)
#         asin = torch.sin(angle)
#         acos2 = torch.square(acos)
#         asin2 = torch.square(asin)

#         """
#         Out line of math for following three lines of code
#         Q = [[sin cos] [-sin cos]] sin/cos of the angle
#         D = [[u 0] [0 v]]
#         A = Q^T.D.Q = [[Axx Axy] [Ayx Ayy]]
#         Axx = cos^2 * u + sin^2 * v
#         Ayy = sin^2 * u + cos^2 * v
#         Axy = Ayx = cos * sin * (u - v)
#         defocus = A.k.k^2 = Axx*x^2 + 2*Axy*x*y + Ayy*y^2
#         """

#         xx_ = (acos2 * u + asin2 * v)[:, None, None] * xx[None, :, :]
#         yy_ = (asin2 * u + acos2 * v)[:, None, None] * yy[None, :, :]
#         xy_ = (acos * asin * (u - v))[:, None, None] * xy[None, :, :]

#         gamma = self.c1 * (xx_ + 2. * xy_ + yy_) + self.c2 * n4[None, :, :] - self.c3 - self.c5
#         ctf = -torch.sin(gamma)
#         if self.c4 > 0:
#             ctf *= torch.exp(self.c4 * n4)

#         if antialiasing > 0:
#             o = 2**antialiasing
#             # ctf = ctf.unsqueeze(1)  # Add singleton channel
#             # ctf = torch.nn.functional.avg_pool2d(ctf, kernel_size=o+o//2, stride=o)
#             # ctf = ctf.squeeze(1)  # Remove singleton channel
#             import equinox
#             ctf = equinox.nn.AvgPool2d(o+o//2,o)(ctf)



#         return ctf


#     @staticmethod
#     def _get_freq(
#             grid_size: int,
#             pixel_size: float,
#             h_sym: bool = False,
#             antialiasing: int = 0,
#     ):
#         """
#         Get the inverted frequencies of the Fourier transform of a square or cuboid grid.
#         Can generate both Torch tensors and Numpy arrays.
#         TODO Add 3D
#         :param antialiasing: Antialiasing oversampling factor (0 = no antialiasing)
#         :param grid_size: the side of the box
#         :param pixel_size: pixel size
#         :param h_sym: Only consider the hermitian half
#         :return: two or three numpy arrays or tensors,
#                  containing frequencies along the different axes
#         """
#         np = jnp
#         if antialiasing > 0:
#             o = 2**antialiasing
#             grid_size *= o
#             y_ls = np.linspace(
#                 -(grid_size + o) // 2,
#                 (grid_size - o) // 2,
#                 grid_size + o//2
#             )
#             x_ls = y_ls if not h_sym else np.linspace(0, grid_size // 2, grid_size // 2 + o + 1)
#         else:
#             y_ls = np.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size)
#             x_ls = y_ls if not h_sym else np.linspace(0, grid_size // 2, grid_size // 2 + 1)

#         y, x = np.meshgrid((y_ls), (x_ls), indexing = 'ij')
#         freq_x = x / (grid_size * pixel_size)
#         freq_y = y / (grid_size * pixel_size)

#         return freq_x, freq_y

# @functools.partial(jax.jit, static_argnums=[1,2,3])
# def dynamight_CTF_wrapper(CTF_params, image_shape, voxel_size, antialiasing ):
    
#     # assert (jnp.linalg.norm(CTF_params[:, 3:6] - CTF_params[0, 3:6]) < 1e-6)
#         # assert False
    
#     ctf_obj = ContrastTransferFunction(
#         CTF_params[0,3], 
#         CTF_params[0,4], 
#         CTF_params[0,5], 
#     )
    
#     zz = ctf_obj(image_shape[0], voxel_size, CTF_params[:,0], CTF_params[:,1], CTF_params[:,2], False, antialiasing)

#     return -zz.reshape(zz.shape[0], -1)



# These functions below are not really used anymore. Could probably delete them. They could useful for preconditioning of Ewald

# # This squared the entries of the forward model. Useful to compute the diagonal of P_i^T_i
# @functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8])
# def forward_model_squared_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):    
#     # slices = map_coordinates_squared_on_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)**2
#     slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)**2
#     return slices


# # This squared the entries of the forward model. Useful to compute the diagonal of P_i^T_i
# @functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8])
# def forward_model_squared_from_map_and_return_adjoint(volume, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):    
#     f = lambda volume : forward_model_squared_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
#     slices, f_adj = vjp(f,volume)
#     return slices, f_adj

#     # slices = map_coordinates_squared_on_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)**2
#     # return slices


# @functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8])
# def adjoint_forward_model_squared_from_map(slices, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):  
#     # Not clear to me whether this has to be done everytime I want to adjoint
#     volume_size = np.prod(volume_shape)
#     f = lambda volume : forward_model_squared_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type)
#     _, u = vjp(f,jnp.zeros(volume_size, dtype = slices.dtype ))
#     return u(slices)[0]


# @functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8])
# def forward_model_from_map(volume, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):
#     slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)
#     return slices


# def forward_model_from_map(mean, CTF_params, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size, CTF_fun, disc_type):
#     slices = slice_volume_by_map(mean, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF_fun( CTF_params, image_shape, voxel_size)
#     return slices