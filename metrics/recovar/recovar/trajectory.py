import logging

import numpy as np
import skfmm
import scipy.ndimage
import matplotlib.pyplot as plt

from recovar import latent_density
logger = logging.getLogger(__name__)



def subsample_path(path, n_pts):
    pts_along_coor = np.round(np.linspace(0, path.shape[0]-1, n_pts)).astype(int)
    return path[pts_along_coor]

def subsample_path_indices(path, n_pts):
    pts_along_coor = np.round(np.linspace(0, path.shape[0]-1, n_pts)).astype(int)
    return pts_along_coor

def resample_at_uniform_pts(gt_vols, n_vols_along_path = 6):
    distances_between_volumes = get_cum_curvelength(gt_vols)
    # n_volumes at approximately equispaced points 
    x = np.linspace(0, distances_between_volumes[-1], n_vols_along_path, endpoint=True)
    gt_vols_x = np.zeros([n_vols_along_path, gt_vols.shape[-1]])
    for k in range(gt_vols.shape[-1]):
        gt_vols_x[:,k] = np.interp(x, distances_between_volumes, gt_vols[:,k], left=None, right=None, period=None)
     
    return gt_vols_x

def get_cum_curvelength(gt_vols):
    distances_between_volumes = np.linalg.norm(gt_vols[1:,...] - gt_vols[:-1,...], axis =1)
    distances_between_volumes = np.append([0], np.cumsum(distances_between_volumes))
    return distances_between_volumes


## TRAJECTORY FUNCTIONS
def find_trajectory_in_grid(density, g_st, g_end, latent_space_bounds, eps = 1e-6, use_log_density = False, debug = False):
    # use_log_density = True
    density_p_eps = density + np.max(density) * eps
    if use_log_density:
        normalized_dens = density / np.max(density*(1+eps))
        # dens = e^(-kB * T * energy)
        # log(dens) = - kB * T* energy
        # energy  = - log(dens)/kB*T
        density_p_eps = 1/(-np.log(normalized_dens ) + eps) + eps
        # Cost should be 1/ energy in this case.
    
    travel_time = compute_travel_time(density_p_eps, g_st, latent_space_bounds)
    
    max_steps = np.linalg.norm(density.shape) * 50
    dx = get_grid_spacing(latent_space_bounds, density)
    # logger.info(f"dx {dx}")
    path = gradient_descent_nd(travel_time, g_st, g_end, dx,  step_size = 0.25, n_theta = 10, max_steps = max_steps )
    debug = True
    if debug:
        plt.imshow(density, aspect = density.shape[1]/ density.shape[0]); plt.colorbar(); plt.show()
        plt.imshow(np.log(travel_time), aspect = density.shape[1]/ density.shape[0]); plt.colorbar(); plt.show()

    # if density.ndim == 2:
    #     plt.imshow(density, aspect = density.shape[1]/ density.shape[0]); plt.colorbar(); plt.show()
    #     plt.imshow(np.log(travel_time), aspect = density.shape[1]/ density.shape[0]); plt.colorbar(); plt.show()
    
    while path is None:
        if eps > 0.1:
            logger.warning(f"Failed to find path, and eps>0.1. Probably a bug. Exiting.")
            break
        
        eps *= 10
        density_p_eps = density + np.max(density) * eps
        travel_time = compute_travel_time(density_p_eps, g_st, latent_space_bounds)
        path = gradient_descent_nd(travel_time, g_st, g_end, dx,  step_size = 0.25, n_theta = 10, max_steps = max_steps )

    return path

def find_trajectory_in_latent_space(density, z_st, z_end, z_to_grid, grid_to_z, latent_space_bounds, density_eps = 1e-5):
    
    def check_in_bound(g):
        for k in range(g.size):
            g[k] = np.max([0, g[k]])
            g[k] = np.min([g[k], density.shape[k]-1])
        return g
        
    
    g_st = z_to_grid(z_st, to_int = True) # Start needs to be on a grid point
    g_st = check_in_bound(g_st)
    g_end = z_to_grid(z_end)
    g_end = check_in_bound(g_end)
    
    path_g = find_trajectory_in_grid(density, g_st, g_end, latent_space_bounds, eps = density_eps)
    return grid_to_z(path_g)

def evaluate_function_off_grid(density, pts):
    return scipy.ndimage.map_coordinates(density, pts.T, order = 1, cval = np.finfo(np.float64).max )

# NOTE that this is not used for optimization purposes. It is used to find the curve which is orthogonal to the level curves of the solution of the Eikonal equation.
def gradient_descent_nd(travel_time, x_st, x_end, dx, step_size = 0.25, n_theta = 10, max_steps = 2000, ):
    
    def f_lambda(pts):
        return evaluate_function_off_grid(travel_time, pts)#scipy.ndimage.map_coordinates(travel_time, pts.T, order = 1)
    
    x_grid = np.linspace(-1, 1.01, n_theta)
    grids = np.meshgrid( *(x_st.shape[0] * [ x_grid]), copy=True, sparse=False, indexing='xy')
    directions = np.stack( [ g.reshape(-1) for g in grids] , axis =-1) 
    directions /= np.linalg.norm(directions, axis =-1)[:,None]
    
    directions /= dx / np.mean(dx)
    #  
    
    path = [x_end]
    x_cur = x_end
    k = 0 
    distances = []
    #if within one cell, end.
    while np.linalg.norm(x_cur - x_st) >  np.sqrt(travel_time.ndim):#np.sqrt(travel_time.ndim):#step_size:

        x_next = x_cur + directions * step_size
        f_x_next = f_lambda(x_next)
        x_cur_idx = np.argmin(f_x_next)
        x_cur = x_next[x_cur_idx]
        path.append(x_cur)
        k += 1
        distances.append(np.linalg.norm(x_cur - x_st))
        
        if k > max_steps:
            cur_path = np.flip(np.stack(path), axis =0)
            plt.scatter(cur_path[:,0], cur_path[:,1])
            plt.show()
            logger.info(f"Failed to find path. Increasing minimum density")
            import pdb; pdb.set_trace()
            return None
            
    path.append(x_st)
    
    return np.flip(np.stack(path), axis =0)


def get_grid_spacing(latent_space_bounds, density):
    dx = []
    for k in range(len(latent_space_bounds)):
        dx.append(
            (latent_space_bounds[k][1] - latent_space_bounds[k][0]) / density.shape[k]
            )
    return dx
    
def compute_travel_time(density, g_st, latent_space_bounds):
    phi = np.ones_like(density)
    phi[tuple(g_st)] = -1
    dx = get_grid_spacing(latent_space_bounds, density)
    travel_time = skfmm.travel_time(phi, speed = density, dx = dx )
    travel_time[tuple(g_st)] = 0 
    return travel_time


def compute_fixed_dimensional_path(z_st, z_end, density_low_dim, latent_space_bounds, density_eps = 1e-5, debug_plot = False, density_option = "kde", use_log_density = False):

    assert z_st.shape[-1] == density_low_dim.ndim, "Start point should be in the same dimension as density"
    assert np.isclose(np.array(density_low_dim.shape) -  density_low_dim.shape[0], 0).all(), "Density should be on square grid"

    max_dim = zs.shape[-1] if max_dim is None else max_dim
    num_points = density_low_dim.shape[0]
    grid_to_z, z_to_grid = latent_density.get_grid_z_mappings(latent_space_bounds, num_points)

    # Start needs to be on a grid point
    g_st = z_to_grid(z_st, to_int = True) 
    g_end = z_to_grid(z_end)

    def check_in_bound(g, num_points):
        for k in range(g.size):
            g[k] = np.max([0, g[k]])
            g[k] = np.min([g[k], num_points-1])
        return g
    ## This is not used.
    g_st_in_bound = check_in_bound(g_st, num_points)
    g_end_in_bound = check_in_bound(g_end, num_points)


    current_path_grid = find_trajectory_in_grid(density_low_dim,
                                            g_st_in_bound,
                                            g_end_in_bound,
                                            latent_space_bounds,
                                            eps = density_eps, 
                                            use_log_density = use_log_density)
    
    return current_path_grid


def compute_high_dimensional_path(zs, cov_zs, z_st, z_end, density_low_dim, density_eps = 1e-5, max_dim = None, percentile_bound = 1, num_points = 50, use_log_density = False, debug_plot = False, density_option = "kde"):

    assert np.isclose(np.array(density_low_dim.shape) -  density_low_dim.shape[0], 0).all(), "Density should be on square grid"
    max_dim = zs.shape[-1] if max_dim is None else max_dim
    latent_space_bounds = latent_density.compute_latent_space_bounds(zs, percentile = percentile_bound)


    low_dim = density_low_dim.ndim
    if low_dim > max_dim: # Hmmm, this is a bit of a hack.
        density_low_dim, _  = latent_density.compute_latent_space_density(zs, cov_zs, pca_dim_max = max_dim, num_points = 100, density_option = density_option)
        logger.info(f"Recomputed density on {max_dim} dimensions")
        low_dim = max_dim

    num_points = density_low_dim.shape[0]
    grid_to_z, z_to_grid = latent_density.get_grid_z_mappings(latent_space_bounds, num_points)

    # Start needs to be on a grid point
    g_st = z_to_grid(z_st, to_int = True) 
    g_end = z_to_grid(z_end)

    def check_in_bound(g, num_points):
        for k in range(g.size):
            g[k] = np.max([0, g[k]])
            g[k] = np.min([g[k], num_points-1])
        return g
    ## This is not used.
    g_st_in_bound = check_in_bound(g_st, num_points)
    g_end_in_bound = check_in_bound(g_end, num_points)

    current_path_grid = find_trajectory_in_grid(density_low_dim,
                                            g_st_in_bound[:low_dim],
                                            g_end_in_bound[:low_dim],
                                            latent_space_bounds[:low_dim], 
                                            eps = density_eps, 
                                            use_log_density = use_log_density,debug = debug_plot)

    grid_to_z_curr_dim, z_to_grid_curr_dim = latent_density.get_grid_z_mappings(latent_space_bounds[:low_dim], num_points)
    current_path_z = grid_to_z_curr_dim(current_path_grid)
    # resample.
    current_path_z = resample_at_uniform_pts(current_path_z, n_vols_along_path = current_path_z.shape[0])#int(current_path_z.shape[0] * 1.2))
    
    for dim in range(low_dim, max_dim):
        print("here?")
        num_points = 200
        grid_to_z, z_to_grid = latent_density.get_grid_z_mappings(latent_space_bounds, num_points)
        g_st = z_to_grid(z_st, to_int = True) 
        g_end = z_to_grid(z_end)

        # Is this necessary? computed 
        def check_in_bound(g, num_points):
            for k in range(g.size):
                g[k] = np.max([0, g[k]])
                g[k] = np.min([g[k], num_points-1])
            return g
        g_st_in_bound = check_in_bound(g_st, num_points)
        g_end_in_bound = check_in_bound(g_end, num_points)

        # Compute density
        density = latent_density.compute_latent_space_density_on_curve(zs[:,:dim+1], 
                                cov_zs[:,:dim+1,:dim+1], current_path_z,  latent_space_bounds, pca_dim = dim, num_points = num_points, density_option = density_option)
        if debug_plot:
            plt.imshow(density, aspect = density.shape[1]/ density.shape[0]); plt.colorbar(); plt.show()

        g_st_on_slice = np.array([ 0 , g_st[dim] ])
        distances = get_cum_curvelength(current_path_z)
        g_end_on_slice = np.array([ distances.size-1, g_end[dim] ])

        latent_space_bounds_slice = np.array([ [distances[0], distances[-1]], latent_space_bounds[dim]])

        if debug_plot:
            logger.debug('latent_space_bounds_slice', latent_space_bounds_slice)

        ## IN GRADIENT DESCENT, FIX DIFFERENT DX IN DIRECTIONS
        ## Is this fixed?

        slice_path_grid = find_trajectory_in_grid(density, g_st_on_slice, g_end_on_slice, latent_space_bounds_slice, eps = density_eps, use_log_density = use_log_density)

        grid_to_z_on_slice, z_to_grid_on_slice = latent_density.get_grid_z_mappings(latent_space_bounds_slice, np.array(density.shape))
        if debug_plot:
            plt.plot(slice_path_grid / np.max(slice_path_grid, axis =0)); plt.show()
        slice_path_z = grid_to_z_on_slice(slice_path_grid)
        
        # Now reconstruct path in dim-dimensions.
        new_path_z = np.zeros([slice_path_z.shape[0], dim+1])
        # First dim-1 are interpolated
        for k in range(dim):
            new_path_z[:,k] = np.interp(slice_path_grid[:,0], np.arange(current_path_z.shape[0]), current_path_z[:,k], left=None, right=None, period=None)
        
        # Tack on new path dimension
        new_path_z[:,-1] = slice_path_z[:,-1]
        
        # Resample uniformly, and make sure size doesn't blow up.
        # these aren't quite equidistant because of discretization. Not sure if it should be changed.
        if dim == max_dim-1:
            current_path_z = resample_at_uniform_pts(new_path_z, n_vols_along_path = new_path_z.shape[0])# int(current_path_z.shape[0] * 1.2))
        else:
            current_path_z = resample_at_uniform_pts(new_path_z, n_vols_along_path = 200)# int(current_path_z.shape[0] * 1.2))

    return current_path_z