import logging
import jax.numpy as jnp
import numpy as np
import jax, time
from scipy.stats import chi2
from recovar import utils

logger = logging.getLogger(__name__)

def get_log_likelihood_threshold(k = 4, q=0.954499736104):
    return chi2.ppf(q,df=k)

def compute_weights_of_conformation_2(latent_points, zs, cov_zs,likelihood_threshold ):
    log_likelihoods = compute_latent_quadratic_forms_in_batch(latent_points, zs, cov_zs)
    weights = np.array(1.0 * ( log_likelihoods <= likelihood_threshold))
    return weights

# Handling change between latent space and the grid 

def pca_coord_to_grid(x, bounds, num_points, to_int = False):
    v =  (x - bounds[:,0] ) / ( bounds[:,1]  - bounds[:,0] ) * (num_points - 1)    
    if to_int:
        return np.round(v).astype(int)   
    else:
        return v

def grid_to_pca_coord(v, bounds, num_points):
    x =  v * ( bounds[:,1]  - bounds[:,0] ) / (num_points - 1)  + bounds[:,0]
    return x

def get_grid_to_z(bounds, num_points ):
    def grid_to_z(x):
        return grid_to_pca_coord(x, bounds = bounds, num_points = num_points)        
    return grid_to_z

def get_z_to_grid(bounds, num_points ):
    def z_to_grid(x, to_int = False):
        return pca_coord_to_grid(x, bounds = bounds, num_points = num_points, to_int = to_int)        
    return z_to_grid

def get_grid_z_mappings(bounds, num_points):
    return get_grid_to_z(bounds, num_points ), get_z_to_grid(bounds, num_points )

def make_latent_space_grid_from_bounds(latent_space_bounds, num_points):
    # latent_space_bounds = compute_latent_space_bounds(zs, percentile = 1)
    # FIND BOUNDS ON SPACE TO DISCRETIZE
    coord_pca_1D = []
    latent_space_bounds = np.array(latent_space_bounds)
    for pca_dim in range(latent_space_bounds.shape[0]):
        coord_pca = np.linspace(latent_space_bounds[pca_dim][0], latent_space_bounds[pca_dim][1], num_points)
        coord_pca_1D.append(coord_pca)
    grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
    grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 
    return grids_flat

# Computes density in pca_dim_max dimensions on grid
def compute_latent_space_density(zs, cov_zs, pca_dim_max = 4, num_points = 50, density_option = "kde"):
    
    if density_option == "kde":
        return compute_latent_space_density_kde(zs, cov_zs, pca_dim_max = pca_dim_max, num_points = num_points)
    elif density_option != "old":
        raise ValueError("Density option not recognized")

    if zs.shape[1] != pca_dim_max:
        zs = zs[:,:pca_dim_max]
        cov_zs = cov_zs[:,:pca_dim_max,:pca_dim_max]        
        
    latent_space_bounds = compute_latent_space_bounds(zs, percentile = 1)
    grids_flat = make_latent_space_grid_from_bounds(latent_space_bounds, num_points)
    # # DISCRETIZE LATENT SPACE
    # latent_space_bounds = compute_latent_space_bounds(zs, percentile = 1)
    # coord_pca_1D = []
    # # FIND BOUNDS ON SPACE TO DISCRETIZE
    # for pca_dim in range(pca_dim_max):
    #     coord_pca = np.linspace(latent_space_bounds[pca_dim][0], latent_space_bounds[pca_dim][1], num_points)
    #     coord_pca_1D.append(coord_pca)

    # # Numpy's meshgrid seems to be randomly incredibly slow. Not sure why. See https://stackoverflow.com/questions/76058225/numpys-meshgrid-is-discontinuously-slow
    # # grids = np.meshgrid(*coord_pca_1D, indexing="ij")
    # # grids_flat = np.transpose(np.vstack([np.reshape(g, -1) for g in grids])).astype(np.float32) 

    # grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
    # grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 
    # grids_inv_pca = grids_flat
    
    st_time = time.time()    
    summed_probs = compute_probs_in_batch(grids_flat, zs, cov_zs)
    summed_probs_sq = summed_probs.reshape(grids[0].shape)
    end_time = time.time()
    logger.info(f"latent space computation:, {end_time - st_time}")
    
    return summed_probs_sq, latent_space_bounds



def compute_latent_space_density_on_2_axes(zs, cov_zs, axes = [0,1], num_points = 50):
    return compute_latent_space_density(zs[:,axes], cov_zs[:,axes][:,:,axes], pca_dim_max = 2, num_points = num_points)


def compute_latent_space_density_at_pts(test_pts, zs, cov_zs):
    return compute_probs_in_batch(test_pts, zs, cov_zs)

def compute_probs_in_batch(test_pts, zs, cov_zs):
    scale_zs = np.array(compute_det_cov_xs(cov_zs))
    summed_probs = jnp.zeros_like(test_pts[:,0])
    
    n_images = zs.shape[0]
    batch_size_x = np.max([int(15 / (utils.get_size_in_gb(test_pts) * cov_zs.shape[1]**2)), 1])
    
    logger.info(f"batch size in latent computation: {batch_size_x}")

    for k in range(0, int(np.ceil(n_images/batch_size_x))):
        batch_st, batch_end = utils.get_batch_of_indices(n_images, batch_size_x, k)
        summed_probs += compute_sum_exp_residuals( test_pts, zs[batch_st:batch_end].real, cov_zs[batch_st:batch_end], scale_zs[batch_st:batch_end] )
    return summed_probs
    

def compute_latent_space_bounds(zs, percentile = 1):
    pca_bounds = []
    # FIND BOUNDS ON SPACE TO DISCRETIZE
    for pca_dim in range(zs.shape[-1]):
        k = pca_dim
        x = zs[:,k]
        min_x = np.percentile(x, percentile) 
        max_x = np.percentile(x, 100-percentile) 
        pca_bounds.append([min_x, max_x])
    return np.array(pca_bounds)


def compute_latent_space_density_on_curve(zs, cov_zs, path,  latent_space_bounds, pca_dim = None, num_points = 50, density_option = "old", gauss_kde = None, normalize = True):

    if density_option == "kde" and gauss_kde is None:
        utils.logger.info("Computing Gaussian KDE with SIlverman bandwidth")
        gauss_kde = jax.scipy.stats.gaussian_kde(zs.T, 'silverman')


    type_used = np.float32
    # pca_dim = zs.shape[1] if pca_dim is None # Should add one more dimension to the path
    assert zs.shape[1] == pca_dim+1

    # Computes a 2D density on a [path(t) x zs[path.shape[1]]] grid.
    for k in [pca_dim]:
        min_x = latent_space_bounds[k][0]
        max_x = latent_space_bounds[k][1]
        coord_pca = np.linspace(min_x, max_x, num_points)
    
    # Form 2D grid of index x coord pc1
    grids = np.meshgrid( np.arange(path.shape[0]), coord_pca, indexing="ij")
    grids_flat = np.transpose(np.vstack([np.reshape(g, -1) for g in grids])).astype(type_used) 

    current_path_dim = path.shape[1]
    grids_flat_with_path = np.zeros([grids_flat.shape[0], path.shape[-1] + 1 ])
    grids_flat_with_path[:,:current_path_dim] = path[grids_flat[:,0].astype(int)]
    grids_flat_with_path[:,-1] = grids_flat[:,-1]
    grids_flat = grids_flat_with_path

    st_time = time.time()
    grids_flat = jnp.array(grids_flat)
    n_images = zs.shape[0]    
    if density_option == "kde":
        summed_probs = compute_kde_density(grids_flat, gauss_kde, normalize = normalize)
    elif density_option == "old":
        summed_probs = compute_probs_in_batch(grids_flat, zs, cov_zs)
    else:
        raise ValueError("Density option not recognized")

    summed_probs_sq = summed_probs.reshape(grids[0].shape)
    end_time = time.time()
    
    return summed_probs_sq


def compute_kde_density(points, gauss_kde, normalize = True):
    if normalize:
        logpdfs = gauss_kde.logpdf(points.T)
        logpdfs = logpdfs - jnp.max(logpdfs)
        pdfs = jnp.exp(logpdfs)
    else:
        pdfs = gauss_kde.pdf(points.T)
    return pdfs


# def compute_latent_space_density_at_zs(zs, cov_zs):   
#     return compute_probs_in_batch(zs, zs, cov_zs)

# DENSITY HELPER FUNCTIONS
@jax.jit
def compute_residuals_single(test_pt, image_latent_means, image_latent_covs):
    diff_to_mean = test_pt - image_latent_means
    res = 0.5 * (jnp.conj(diff_to_mean).T @ (image_latent_covs @ diff_to_mean) ).real 
    return res


batch0_compute_residuals = jax.vmap(compute_residuals_single, in_axes = (0, None,None) )
batch01_compute_residuals = jax.vmap(batch0_compute_residuals, in_axes = (None, 0,0) )

@jax.jit
def compute_sum_exp_residuals(test_pts, xs, cov_xs, scale_xs):
    #import pdb; pdb.set_trace()
    return jnp.sum( scale_xs[...,None] *  jnp.exp(- batch01_compute_residuals(test_pts, xs, cov_xs) ), axis = 0)

@jax.jit
def compute_likelihood_of_latent_given_image(test_pts, xs, cov_xs, scale_xs):
    return scale_xs[...,None] *  jnp.exp(- batch01_compute_residuals(test_pts, xs, cov_xs))

@jax.jit
def compute_unscaled_log_likelihood_of_latent_given_image(test_pts, xs, cov_xs, scale_xs):
    return batch01_compute_residuals(test_pts, xs, cov_xs)

@jax.jit
def compute_latent_quadratic_forms(test_pts, xs, cov_xs):
    # The two because there is a 0.5 up there?
    return 2 * batch01_compute_residuals(test_pts, xs, cov_xs)


def compute_latent_quadratic_forms_in_batch(test_pts, zs, cov_zs):
    assert zs.shape[1] == test_pts.shape[1]
    assert zs.shape[1] == cov_zs.shape[1]
    assert test_pts.ndim == 2
    assert cov_zs.ndim == zs.ndim + 1

    quads = np.zeros([zs.shape[0], test_pts.shape[0]] )
    n_images = zs.shape[0]
    utils
    batch_size_x = utils.get_latent_density_batch_size(test_pts, zs.shape[-1], utils.get_gpu_memory_total() ) 
    logger.info(f"batch size in latent computation: {batch_size_x}")
    logger.warning("SHOULD THIS BE SCALED?")
    for k in range(0, utils.get_number_of_index_batch(n_images, batch_size_x)):
        batch_st, batch_end = utils.get_batch_of_indices(n_images, batch_size_x, k)
        quads[batch_st:batch_end,:] = compute_latent_quadratic_forms( test_pts.real, zs[batch_st:batch_end].real, cov_zs[batch_st:batch_end])

    return quads

def compute_latent_log_likelihood(test_pts, zs, cov_zs):
    assert zs.shape[1] == test_pts.shape[1]
    assert zs.shape[1] == cov_zs.shape[1]
    assert test_pts.ndim == 2
    assert cov_zs.ndim == zs.ndim + 1

    det_cov_zs = compute_log_det_cov(cov_zs)
    quads = np.zeros([zs.shape[0], test_pts.shape[0]] )
    n_images = zs.shape[0]
    
    batch_size_x = utils.get_latent_density_batch_size(test_pts, zs.shape[-1], utils.get_gpu_memory_total() ) 
    logger.info(f"batch size in latent computation: {batch_size_x}")
    logger.warning("SHOULD THIS BE SCALED?")
    for k in range(0, utils.get_number_of_index_batch(n_images, batch_size_x)):
        batch_st, batch_end = utils.get_batch_of_indices(n_images, batch_size_x, k)
        quads[batch_st:batch_end,:] = 0.5 * (compute_latent_quadratic_forms( test_pts.real, zs[batch_st:batch_end].real, cov_zs[batch_st:batch_end]) + det_cov_zs[batch_st:batch_end][...,None])
    # import pdb; pdb.set_trace()

    return quads

@jax.jit
def compute_log_det_cov(cov_xs):
    vs = jnp.sum(jnp.log(jax.numpy.linalg.eigvalsh(cov_xs)), axis =-1)
    # Determinants are exp(vs) now..., but only care about the ratio so:
    # we exp(vs_i) / exp(vs_j) = exp(vs_i - vs_j)
    return vs


@jax.jit
def compute_det_cov_xs(cov_xs):
    vs = compute_log_det_cov(cov_xs)
    # Determinants are exp(vs) now..., but only care about the ratio so:
    # we exp(vs_i) / exp(vs_j) = exp(vs_i - vs_j)
    print("is this right?!?")
    vs_subs_min = (vs - jnp.max(vs))
    return jnp.exp(vs_subs_min)



def compute_latent_space_density_kde(zs, cov_zs, pca_dim_max = 4, num_points = 50, gauss_kde = None):
    
    if zs.shape[1] != pca_dim_max:
        zs = zs[:,:pca_dim_max]
        cov_zs = cov_zs[:,:pca_dim_max,:pca_dim_max]        
        
    gauss_kde = jax.scipy.stats.gaussian_kde(zs.T, 'silverman') if gauss_kde is None else gauss_kde

    # DISCRETIZE LATENT SPACE
    latent_space_bounds = compute_latent_space_bounds(zs, percentile = 1)
    coord_pca_1D = []
    # FIND BOUNDS ON SPACE TO DISCRETIZE
    for pca_dim in range(pca_dim_max):
        coord_pca = np.linspace(latent_space_bounds[pca_dim][0], latent_space_bounds[pca_dim][1], num_points)
        coord_pca_1D.append(coord_pca)
        
    grids = jnp.meshgrid(*coord_pca_1D, indexing="ij")
    grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32) 
    
    st_time = time.time()    
    
    batch_size = 10000
    n_pts = grids_flat.shape[0]
    probs = np.zeros(grids_flat.shape[0])
    option = 'new'
    if option == 'old':
        for k in range(0, int(np.ceil(n_pts/batch_size ))):
            batch_st, batch_end = utils.get_batch_of_indices(n_pts, batch_size , k)
            probs[batch_st:batch_end] = gauss_kde.evaluate(grids_flat[batch_st:batch_end].T)        
    else:
        for k in range(0, int(np.ceil(n_pts/batch_size ))):
            batch_st, batch_end = utils.get_batch_of_indices(n_pts, batch_size , k)
            # logpdfs = gauss_kde.logpdf(points.T)
            probs[batch_st:batch_end] = gauss_kde.logpdf(grids_flat[batch_st:batch_end].T)        
        probs = probs - jnp.max(probs)
        probs = jnp.exp(probs)

    summed_probs_sq = probs.reshape(grids[0].shape)
    end_time = time.time()
    logger.info(f"latent space computation:, {end_time - st_time}")
    return summed_probs_sq, latent_space_bounds#, grids_flat.reshape(*grids[0].shape, pca_dim_max )
