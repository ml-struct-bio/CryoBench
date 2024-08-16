import logging, functools
import jax
import jax.numpy as jnp
import numpy as np
import mrcfile, os , psutil, pickle
from recovar.fourier_transform_utils import fourier_transform_utils
from recovar import core
ftu = fourier_transform_utils(jax.numpy)
    
logger = logging.getLogger(__name__)

@functools.partial(jax.jit, static_argnums = [1,2])    
def make_radial_image(average_image_PS, image_shape, extend_last_frequency = True):
    if extend_last_frequency:
        last_noise_band = average_image_PS[-1]
        average_image_PS = jnp.concatenate( [average_image_PS, last_noise_band * jnp.ones_like(average_image_PS) ] )
    radial_distances = ftu.get_grid_of_radial_distances(image_shape, scaled = False, frequency_shift = 0).astype(int).reshape(-1)
    prior = jnp.asarray(average_image_PS)[radial_distances]
    return prior

batch_make_radial_image = jax.vmap(make_radial_image, in_axes = (0,None,None))


def find_angle_between_subspaces(v1,v2, max_rank):
    ss = np.conj(v1[:,:max_rank]).T @ v2[:,:max_rank]
    s,v,d = np.linalg.svd(ss)
    if np.any(v > 1.2):
        print('v too big!')
    v = np.where(v < 1, v, 1)
    return np.sqrt( 1 - v[-1]**2)


def subspace_angles(u ,v, max_rank = None):
    max_rank = u.shape[-1] if max_rank is None else max_rank
    corr = np.zeros(max_rank)
    for k in range(1,max_rank+1):
        if k > u.shape[-1]:
            corr[k-1] = 1
        else:
            corr[k-1] = find_angle_between_subspaces(u[:,:k], v[:,:k], max_rank = k )
    return corr  


def estimate_variance(u, s):
    var = np.sum(np.abs(u)**2 * s[...,None], axis = 0)
    return var

# inner psutil function
def get_process_memory_used():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return int(mem_info.rss / 1e9)

def get_gpu_memory_total(device =0):
    if jax_has_gpu():
        return int(jax.local_devices()[device].memory_stats()['bytes_limit']/1e9)
    else:
        logger.warning("GPU not found. Using default value of 80GB for batching computation on CPU.")
        return int(80)
    


def get_gpu_memory_used(device =0):
    return int(jax.local_devices()[device].memory_stats()['bytes_in_use']/1e9)

def get_peak_gpu_memory_used(device =0):
    return int(jax.local_devices()[device].memory_stats()['peak_bytes_in_use']/1e9)

def report_memory_device(device=0, logger=None):
    output_str = f"GPU mem in use:{get_gpu_memory_used(device)}; peak:{get_peak_gpu_memory_used(device)}; total available:{get_gpu_memory_total(device)}, process mem in use:{get_process_memory_used()}"
    if logger is None:
        print(output_str)
    else:
        logger.info(output_str)

def get_size_in_gb(x):
    return x.size * x.itemsize / 1e9
    
def write_mrc(file, ar, voxel_size = None):
    # This is to agree with the cryosparc/cryoDRGN convention
    if ar.ndim == 3 and np.isclose(ar.shape, ar.shape[0]).all():
        ar = np.transpose(ar, (2,1,0))
        
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(ar.real.astype(np.float32))
        if voxel_size is not None:
            mrc.voxel_size = voxel_size

def load_mrc(filepath, return_voxel_size = False):
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
        if return_voxel_size:
            voxel_size = mrc.voxel_size

    # This is to agree with the cryosparc/cryoDRGN convention
    if data.ndim == 3 and np.isclose(data.shape, data.shape[0]).all():
        data = np.transpose(data, (2,1,0))

    # in order not to break rest of code...    
    if return_voxel_size:
        return data, voxel_size 
    
    return data
        
def symmetrize_ft_volume(vol, volume_shape):
    og_volume_shape = vol.shape
    vol = vol.reshape(volume_shape)
    vol = vol.at[1:,1:,1:].set( 0.5 * (np.conj(np.flip(vol[1:,1:,1:])) + vol[1:,1:,1:]) )
    return vol.reshape(og_volume_shape)

# def symmetrize_ft_image(vol, volume_shape):
#     og_volume_shape = vol.shape
#     vol = vol.reshape(volume_shape)
#     vol = vol.at[1:,1:].set( 0.5 * (np.conj(np.flip(vol[1:,1:])) + vol[1:,1:]) )
#     return vol.reshape(og_volume_shape)


def get_all_dataset_indices(cryos):
    return np.concatenate([cryo.dataset_indices for cryo in cryos])

def get_inverse_dataset_indices(cryos):
    return np.argsort(np.concatenate([cryo.dataset_indices for cryo in cryos]))

def guess_grid_size_from_vol_size(vol_size):
    return np.round((vol_size)**(1/3)).astype(int)
        
def guess_vol_shape_from_vol_size(vol_size):
    return tuple(3*[guess_grid_size_from_vol_size(vol_size)])

# These should probably be set more intelligently
# Sometimes, memory can grow like O(vol_batch_size * image_batch_size)
def get_image_batch_size(grid_size, gpu_memory):
    return int(2*(2**24)/ (grid_size**2)  * gpu_memory / 38)

def get_vol_batch_size(grid_size, gpu_memory):
    return int(25 * (256 / grid_size)**3 * gpu_memory / 38) 

def get_column_batch_size(grid_size, gpu_memory):
    return int(50 * ((256/grid_size)**3) * gpu_memory / 38)

def get_latent_density_batch_size(test_pts,zdim, gpu_memory):
    return np.max([int(gpu_memory /(3 * (get_size_in_gb(test_pts) * zdim**2))), 1])

def get_embedding_batch_size(basis, image_size, contrast_grid, zdim, gpu_memory):

    left_over_memory = ( gpu_memory - get_size_in_gb(basis))
    # assert left_over_memory > 0, "GPU memory too small?"

    batch_size = int(left_over_memory/ ( (image_size  * np.max([zdim, 4]) + contrast_grid.size * zdim**2 ) *8/1e9 )/ 20)

    if batch_size < 1:
        logger.warning('GPU may be too small for the default parameters. Trying anyway')
        return 1

    return batch_size


def make_algorithm_options(args):
    options = {'volume_mask_option': args.mask_option,
    'zs_dim_to_test': args.zdim,
    'contrast' : "contrast_qr" if args.correct_contrast else "none",
    'ignore_zero_frequency' : args.ignore_zero_frequency ,
    'keep_intermediate' : args.keep_intermediate ,
    }
    return options


def pickle_dump(object, file):
    with open(file, "wb") as f:
        pickle.dump(object, f)

def pickle_load( file):
    with open(file, "rb") as f:
        return pickle.load(f)

    
def get_variances(covariance_cols, picked_frequencies = None):
    # picked_frequencies = np.array(covariance_core.get_picked_frequencies(volume_shape, radius = constants.COLUMN_RADIUS, use_half = True))

    volume_shape = guess_vol_shape_from_vol_size(covariance_cols.shape[-1])
    # freqs = core.vec_indices_to_frequencies(picked_frequencies, volume_shape)

    # Probably a better way to do this...
    variances = np.zeros(picked_frequencies.size, covariance_cols.dtype)
    for k in range(picked_frequencies.size):
        variances[k] = covariance_cols[picked_frequencies[k], k]

    return variances

def get_number_of_index_batch(n_images, batch_size):
    return int(np.ceil(n_images/batch_size))

def get_batch_of_indices(n_images, batch_size, k):
    batch_st = int(k * batch_size)
    batch_end = int(np.min( [(k+1) * batch_size, n_images] ))
    return batch_st, batch_end

def get_batch_of_indices_arange(n_images, batch_size, k):
    batch_st = int(k * batch_size)
    batch_end = int(np.min( [(k+1) * batch_size, n_images] ))
    return np.arange(batch_st, batch_end)


def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

def dtype_to_real(rvs_dtype):
    return rvs_dtype.type(0).real.dtype


import starfile
import pandas as pd
def write_starfile(CTF_params, rotation_matrices, translations, voxel_size, grid_size, particles_file, output_filename, halfset_indices = None):

    # Stored like this in CTF_params
    # dfu (float or Bx1 tensor): DefocusU (Angstrom)
    # dfv (float or Bx1 tensor): DefocusV (Angstrom)
    # dfang (float or Bx1 tensor): DefocusAngle (degrees)
    # volt (float or Bx1 tensor): accelerating voltage (kV)
    # cs (float or Bx1 tensor): spherical aberration (mm)
    # w (float or Bx1 tensor): amplitude contrast ratio
    # phase_shift (float or Bx1 tensor): degrees 
    # bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    
    keys = ['rlnOpticsGroup', 'rlnOpticsGroupName', 'rlnAmplitudeContrast',
       'rlnSphericalAberration', 'rlnVoltage', 'rlnImagePixelSize',
       'rlnImageSize', 'rlnImageDimensionality']
    dtype = np.float64
    values = [ 1, 'opticsGroup1', CTF_params[0, 5].astype(dtype),  CTF_params[0, 4].astype(dtype), CTF_params[0, 3].astype(dtype), voxel_size, grid_size, 2]
    optic_df = pd.DataFrame.from_dict({ 0: values}, orient='index',
                       columns=keys)
    n_images = CTF_params.shape[0]
    image_names = [ f"{k+1}@{particles_file}" for k in range(n_images) ]
    micrograph_names = [ f"{k+1}" for k in range(n_images) ]
    optics_group = np.ones(n_images).astype(int)
    
    keys = ['rlnImageName', 'rlnMicrographName', 'rlnDefocusU', 'rlnDefocusV',
       'rlnDefocusAngle', 'rlnOpticsGroup']
    
    values = [ image_names, micrograph_names, CTF_params[:,0].astype(dtype), CTF_params[:,1].astype(dtype), CTF_params[:,2].astype(dtype), optics_group ]

    if rotation_matrices is not None:
        keys += [ 'rlnAngleRot',
                'rlnAngleTilt', 
                'rlnAnglePsi', 
                'rlnOriginXAngst', 
                'rlnOriginYAngst',] 
        import cryodrgn.utils as cryodrgn_utils
        rots = cryodrgn_utils.R_to_relion_scipy(rotation_matrices)
        values += [ rots[:,0], rots[:,1], rots[:,2], translations[:,0], translations[:,1] ]
    

    if halfset_indices is not None:
        keys += ['rlnRandomSubset']
        values += [halfset_indices]
    
    d = dict(zip(keys, values))
    particles_df = pd.DataFrame(d)
    star_df = { 'optics' : optic_df, 'particles' : particles_df }
    starfile.write(star_df, output_filename)
    return



def write_starfile_from_cryodrgn_format(ctf_path, pose_path, particles_file_path, output_filename, halfset_indices = None):
    ctf = pickle_load(ctf_path)
    poses = pickle_load(pose_path)
    rots = poses[0]
    trans = poses[1]
    # particles = load_mrc(particles_file_path)
    # import pdb; pdb.set_trace()
    write_starfile(ctf[:,2:], rots, trans, ctf[0,1], ctf[0,0], particles_file_path, output_filename, halfset_indices = None)

# make_starfile(cryo_dataset.CTF_params, cryo_dataset.rotation_matrices, cryo_dataset.translations, cryo_dataset.voxel_size, cryo_dataset.volume_shape[0], dataset_dict['particles_file'], output_filename = output_folder + 'sim_newest.star', halfset_indices = array_indices )
