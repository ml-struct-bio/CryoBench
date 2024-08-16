import logging
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import tensorflow as tf

from recovar import plot_utils, core, mask
import recovar.padding as pad
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)

logger = logging.getLogger(__name__)

# Maybe should take out these dependencies?
import torch
from cryodrgn import mrc, ctf, dataset
from cryodrgn.pose import PoseTracker

class MRCDataMod(torch.utils.data.Dataset):
    # Adapted from cryoDRGN
    '''
    Class representing an .mrcs stack file -- images preloaded
    '''
    def __init__(self, mrcfile, datadir=None, ind = None, padding = 0, uninvert_data = False, mask = None ):
        
        if ind is not None:
            # particles = dataset.load_particles(mrcfile, True, datadir=datadir)
            # particles = np.array([particles[i].get() for i in ind])
            particles = dataset.load_particles(mrcfile, False, datadir=datadir)[ind]
        else:
            particles = dataset.load_particles(mrcfile, False, datadir=datadir)
        N, ny, nx = particles.shape
        
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even. Is this a preprocessed dataset? Use the --preprocessed flag if so."
        
        
        self.n_images = N
        self.D = (nx + padding) 
        self.image_size = self.D * self.D
        self.image_shape = (nx + padding, ny + padding)

        self.dtype = np.complex64 # ???
        self.unpadded_D = nx
        self.unpadded_image_shape = (nx, ny)
        self.mask = set_standard_mask(self.unpadded_D, self.dtype)
        self.mult = -1 if uninvert_data else 1
        
        # Maybe should do do this on CPU?
        self.particles = particles #np.array(padded_dft(particles, self.mask, self.image_size))
        self.padding = padding
        
    def get(self, i):
        return self.particles[i]

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        return self.get(index), index

    def process_images(self, images, apply_image_mask = False):
        if apply_image_mask:
            images = images * self.mask
        # logger.warning("CHANGE BACK USE MASK TO FALSE")
        images = pad.padded_dft(images * self.mult,  self.image_size, self.padding)
        return images

    def get_dataset_generator(self, batch_size, num_workers = 0):
        return tf.data.Dataset.from_tensor_slices((self.particles,np.arange(self.n_images))).batch(batch_size, num_parallel_calls = tf.data.AUTOTUNE).as_numpy_iterator()

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        return tf.data.Dataset.from_tensor_slices((self.particles[subset_indices], subset_indices)).batch(batch_size, num_parallel_calls = tf.data.AUTOTUNE).as_numpy_iterator()



class LazyMRCDataMod(torch.utils.data.Dataset):
    # Adapted from cryoDRGN
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, datadir=None, ind = None, padding = 0, uninvert_data = False ):
        
        particles = dataset.load_particles(mrcfile, True, datadir=datadir)
        if ind is not None:
            particles = [particles[x] for x in ind]
            
        N = len(particles)
        ny, nx = particles[0].get().shape
        
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even. Is this a preprocessed dataset? Use the --preprocessed flag if so."
        

        self.mrcfile = mrcfile
        self.datadir = datadir
        self.n_images = N
        self.D = (nx + padding) 
        self.image_size = self.D * self.D
        self.image_shape = (nx + padding, ny + padding)
        self.dtype = np.complex64 

        self.mean_mask = set_standard_mask(self.D, self.dtype)
        self.mask = set_standard_mask(self.D, self.dtype)
        self.mult = -1 if uninvert_data else 1
        
        self.unpadded_D = nx
        self.unpadded_image_shape = (nx, ny)

        # Maybe should do do this on CPU?
        self.particles = particles #np.array(padded_dft(particles, self.mask, self.image_size))
        self.padding = padding
        
    def get(self, i):
        return self.particles[i].get()

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        return self.get(index), index

    def process_images(self, images, apply_image_mask = False):
        
        if apply_image_mask:
            images = images * self.mask

        images = pad.padded_dft(images * self.mult, self.image_size, self.padding)
        return images

    def get_dataset_generator(self, batch_size, num_workers = 0):
        return NumpyLoader(self, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        # raise NotImplementedError
        # Maybe this would work?
        NumpyLoader(torch.utils.data.Subset(self, subset_indices), batch_size=batch_size, shuffle=False, num_workers = num_workers)
        # torch.utils.data.Subset(self, subset_indices)
    
    
def get_num_images_in_dataset(mrc_path):
    particles = dataset.load_particles(mrc_path, True)
    return len(particles)

def set_standard_mask(D, dtype):
    return mask.window_mask(D, 0.85, 0.99)
    # return np.ones([D,D], dtype = dtype).real
    
# Image loader functions - supposed to give quick access to images to GPU

# I don't remember why I use two different loaders here.

# This might work. 
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return jnp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return jnp.array(batch)

class NumpyLoader(torch.utils.data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    

# A dataset class, that includes images and all other information
class CryoEMDataset:

    def __init__(self, image_stack, voxel_size, rotation_matrices, translations, CTF_params, CTF_fun = core.evaluate_ctf_wrapper, dtype = np.complex64, rotation_dtype = np.float64, dataset_indices = None, grid_size = None, volume_upsampling_factor = 1  ):
        
        if image_stack is not None:
            grid_size = image_stack.D
        elif grid_size is None:
            raise ValueError("Must specify grid_size if image_stack is None")
        

        self.voxel_size = voxel_size
        self.grid_size = grid_size

        self.volume_upsampling_factor = volume_upsampling_factor
        self.upsampled_grid_size = self.grid_size * volume_upsampling_factor
        self.grid_size = grid_size

        self.volume_shape = tuple(3*[self.grid_size ])
        self.volume_size = np.prod(self.volume_shape)

        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

        # self.original_volume_shape = tuple(3*[image_stack.unpadded_D])
        # self.volume_shape = tuple(3*[image_stack.unpadded_D])
        # Allows for passing None as image_stack (for simulation)
        if image_stack is None:
            self.image_stack = None
            self.image_shape = tuple(2*[self.grid_size])
            self.image_size = np.prod(self.image_shape)
            self.n_images = CTF_params.shape[0]
            self.padding = 0 
        else:
            self.image_stack = image_stack
            self.image_shape = tuple(image_stack.image_shape)
            self.image_size = np.prod(image_stack.image_shape)
            self.n_images = image_stack.n_images
            self.padding = image_stack.padding

        self.CTF_FUNCTION_OPTION = "cryodrgn"
        self.CTF_fun_inp = CTF_fun
        self.hpad = self.padding//2
        self.volume_mask_threshold = 4 * self.grid_size / 128 # At around 128 resolution, 4 seems good, so scale up accordingly. This probably should have a less heuristic value here. This is assuming the mask is scaled between [0,1]

        self.dtype = dtype
        self.dtype_real = dtype(0).real.dtype
        self.CTF_dtype = self.dtype_real # this might changed in the future for Ewald sphere
        # Note that images are stored in float 32 but rotations are stored in float 64.
        # There seems to be a JAX-bug with float 32 when doing the nearest neighbor approximation...
        self.rotation_dtype = rotation_dtype
        self.rotation_matrices = np.array(rotation_matrices.astype(rotation_dtype))
        self.translations = np.array(translations)

        '''
            0 - dfu (float or Bx1 tensor): DefocusU (Angstrom)
            1 - dfv (float or Bx1 tensor): DefocusV (Angstrom)
            2 - dfang (float or Bx1 tensor): DefocusAngle (degrees)
            3 - volt (float or Bx1 tensor): accelerating voltage (kV)
            4 - cs (float or Bx1 tensor): spherical aberration (mm)
            5 - w (float or Bx1 tensor): amplitude contrast ratio
            6 - phase_shift (float or Bx1 tensor): degrees 
            7 - bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
            8 - per-particle scale
        '''
        self.CTF_params = np.array(CTF_params.astype(self.CTF_dtype))

        self.dataset_indices = dataset_indices

    def delete(self):
        del self.image_stack.particles
        del self.image_stack
        del self.rotation_matrices
        del self.CTF_params
        del self.translations

    def update_volume_upsampling_factor(self, volume_upsampling_factor):

        self.volume_upsampling_factor = volume_upsampling_factor
        self.upsampled_grid_size = self.grid_size * volume_upsampling_factor
        
        self.upsampled_volume_shape = tuple(3*[self.grid_size * volume_upsampling_factor ])
        self.upsampled_volume_size = np.prod(self.upsampled_volume_shape)

        return


    def get_dataset_generator(self, batch_size, num_workers = 0):
        return self.image_stack.get_dataset_generator(batch_size,num_workers = num_workers)
    
    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        return self.image_stack.get_dataset_subset_generator(batch_size, subset_indices, num_workers = num_workers)


    def CTF_fun(self,*args):
        # Force dtype
        return self.CTF_fun_inp(*args, CTF_FUNCTION_OPTION = self.CTF_FUNCTION_OPTION).astype(self.CTF_dtype)

    def get_valid_frequency_indices(self,rad = None):
        rad = self.grid_size//2 -1 if rad is None else rad
        return np.array(self.get_volume_radial_mask(rad))

    def get_valid_upsampled_frequency_indices(self,rad = None):
        rad = self.upsampled_grid_size//2 -1 if rad is None else rad
        return np.array(self.get_upsampled_volume_radial_mask(rad))


    #### All functions below are only just for plotting/debugging

    def compute_CTF(self, CTF_params):
        return self.CTF_fun(CTF_params, self.image_shape, self.voxel_size)

    def get_CTF(self, indices):
        return self.compute_CTF(self.CTF_params[indices])

    def get_volume_radial_mask(self, radius = None):
        return mask.get_radial_mask(self.volume_shape, radius = radius).reshape(-1)

    def get_upsampled_volume_radial_mask(self, radius = None):
        return mask.get_radial_mask(self.upsampled_volume_shape, radius = radius).reshape(-1)


    def get_image_radial_mask(self, radius = None):        
        return mask.get_radial_mask(self.image_shape, radius = radius).reshape(-1)

    def get_proj(self, X, to_real = np.real, axis = 0, hide_padding = True):
        im = to_real(ftu.get_idft2(jnp.take(X.reshape(self.volume_shape), self.grid_size//2, axis = axis)))
        if hide_padding:
            im = im[self.hpad:self.image_stack.unpadded_D + self.hpad,self.hpad:self.image_stack.unpadded_D + self.hpad]
        return im

    def get_slice(self, X, to_real_fn = np.abs, axis = 0):
        # zero_th freq
        z_freq = self.grid_size//2 +1
        im = to_real_fn(jnp.take(X.reshape(self.volume_shape), z_freq, axis = axis))
        return im

    def get_slice_real(self, X, to_real_fn = np.real, axis = 0):
        im = to_real_fn(ftu.get_idft3(X.reshape(self.volume_shape)))
        im2 = jnp.take(im, self.grid_size//2, axis = axis)
        return to_real_fn(im2)

    def get_image(self, i ):
        image = self.image_stack.get(i)[None]
        processed_image = self.image_stack.process_images(image)
        return processed_image.reshape(self.image_shape)

    def get_CTF_image(self, i ):
        return self.get_CTF(np.array([i])).reshape(self.image_shape)

    def get_image_real(self,i, to_real= np.real, hide_padding = True):
        hpad= self.image_stack.padding//2
        if hide_padding:
            return to_real(ftu.get_idft2(self.get_image(i))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        else:
            return to_real(ftu.get_idft2(self.get_image(i)))


    def get_denoised_image(self,i, to_real= np.real, hide_padding = True, weiner_param =1):
        batch_image_ind = np.array([i])
        CTFs = self.CTF_fun(self.CTF_params[batch_image_ind], self.image_shape, self.voxel_size) # Compute CTF
        images = self.image_stack.get(i)[None]
        images = self.image_stack.process_images(images) # Compute DFT, masking
        images = (CTFs / (CTFs**2 + weiner_param)) * images  # CTF correction
        images = images.reshape(self.image_shape)
        # if hide_padding:
        #     return to_real(ftu.get_idft2(self.get_image(i))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        # else:
        return to_real(ftu.get_idft2(images))


    def plot_FSC(self, image1 = None, image2 = None, filename = None, threshold = 0.5, curve = None, ax = None):
        score = plot_utils.plot_fsc_new(image1, image2, self.volume_shape, self.voxel_size,  curve = curve, ax = ax, threshold = threshold, filename = filename)
        return score
    
    def get_image_mask(self, indices, mask, binary = True, soften = 5):
        indices = np.asarray(indices).astype(int)
        from recovar import covariance_core # Not sure I want this depency to exist... Could make some circular imports
        return covariance_core.get_per_image_tight_mask(mask, self.rotation_matrices[indices], self.image_stack.mask, self.volume_mask_threshold, self.image_shape, self.volume_shape, self.grid_size, self.padding, disc_type = 'linear_interp',  binary = binary, soften = soften)


def subsample_cryoem_dataset(dataset, indices):

    import copy
    image_stack = copy.copy(dataset.image_stack)

    if type(image_stack.particles) is list:
        image_stack.particles = [dataset.image_stack.particles[i] for i in indices]
        image_stack.n_images = len(image_stack.particles)#.shape[0]
    else:
        image_stack.particles = dataset.image_stack.particles[indices]
        image_stack.n_images = image_stack.particles.shape[0]

    return CryoEMDataset( image_stack, dataset.voxel_size, dataset.rotation_matrices[indices], dataset.translations[indices], dataset.CTF_params[indices], CTF_fun = dataset.CTF_fun_inp, dtype = dataset.dtype, rotation_dtype = dataset.rotation_dtype, dataset_indices = dataset.dataset_indices[indices] , volume_upsampling_factor= dataset.volume_upsampling_factor)



# Loads dataset that are stored in the cryoDRGN format
def load_cryodrgn_dataset(particles_file, poses_file, ctf_file, datadir = None, n_images = None, ind = None, lazy = True, padding = 0, uninvert_data = False):
    
    # if ind is None:
    #     ind = None if n_images is None else jnp.arange(n_images) 
    
    if lazy:
        dataset = LazyMRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data)
    else:
        dataset = MRCDataMod(particles_file, ind = ind, datadir = datadir, padding = padding, uninvert_data = uninvert_data)
        
    ctf_params = np.array(ctf.load_ctf_for_training(dataset.unpadded_D, ctf_file))
        
    ctf_params = ctf_params if ind is None else ctf_params[ind]
    
    # Initialize bfactor == 0
    ctf_params = np.concatenate( [ctf_params, np.zeros_like(ctf_params[:,0][...,None])], axis =-1)
    
    # Initialize constrast == 1
    ctf_params = np.concatenate( [ctf_params, np.ones_like(ctf_params[:,0][...,None])], axis =-1)
    
    posetracker = PoseTracker.load( poses_file, dataset.n_images, dataset.unpadded_D, ind = ind) #,   None, ind, device=device)
    
    # Translation might NOT BE PROPERLY SCALED!!!
    
    voxel_sizes = ctf_params[:,0]
    assert np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0))
    voxel_size = float(voxel_sizes[0])
    CTF_fun = core.evaluate_ctf_wrapper
    return CryoEMDataset( dataset, voxel_size,
                              np.array(posetracker.rots), np.array(posetracker.trans), ctf_params[:,1:], CTF_fun = CTF_fun, dataset_indices = ind)

def get_split_datasets_from_dict(dataset_loader_dict, ind_split, lazy = False):
    return get_split_datasets(**dataset_loader_dict, ind_split=ind_split, lazy =lazy)

def get_split_datasets(particles_file, poses_file, ctf_file, datadir,
                                  uninvert_data = False, ind_file = None,
                                  padding = 0, n_images = None, ind_split = None, lazy = False):
    
    cryos = []
    for ind in ind_split:
        cryos.append(load_cryodrgn_dataset(particles_file, poses_file, ctf_file , datadir = datadir, n_images = n_images, ind = ind, lazy = lazy, padding = padding, uninvert_data = uninvert_data))
    
    return cryos


def get_split_indices(particles_file, ind_file = None):

    if ind_file is None:
        n_images = get_num_images_in_dataset(particles_file)
        indices = np.arange(n_images)
    else:
        if isinstance(ind_file, np.ndarray):
            indices = ind_file
        else:
            # Get indf
            with open( ind_file,'rb') as f:
                indices = pickle.load(f)  
  
    split_indices = split_index_list(indices)
    return split_indices

def split_index_list( all_valid_image_indices, split_random_seed = 0 ):
    np.random.seed(split_random_seed)

    half_ind_size = all_valid_image_indices.size //2
    shuffled_ind = np.arange(all_valid_image_indices.size)

    np.random.shuffle(shuffled_ind)
    ind_split = [
                all_valid_image_indices[shuffled_ind[:half_ind_size]], 
                all_valid_image_indices[shuffled_ind[half_ind_size:]],
                ]
    return ind_split
        

def make_dataset_loader_dict(args):
    dataset_loader_dict = { 'particles_file' : args.particles,
                            'ctf_file': args.ctf ,
                            'poses_file' : args.poses,
                            'datadir': args.datadir,
                            'n_images' : args.n_images,
                            'ind_file': args.ind,
                            'padding' : args.padding }
    
    
    if args.uninvert_data == "automatic" or  args.uninvert_data == "false":
        dataset_loader_dict['uninvert_data'] = False
    elif args.uninvert_data == "true":
        dataset_loader_dict['uninvert_data'] = True
    else:
        raise Exception("input uninvert-data option is wrong. Should be automatic, true or false ")
    
    return dataset_loader_dict

def figure_out_halfsets(args):
    if args.halfsets == None:
        logging.info("Randomly splitting dataset into halfsets")
        # ind_split = dataset.get_split_indices(args.particles_file, ind_file = args.ind)
        # # pickle.dump(ind_split, open(args.out))
        halfsets = get_split_indices(args.particles, ind_file = args.ind)
    else:
        logging.info("Loading halfset from file")
        halfsets = pickle.load(open(args.halfsets, 'rb'))
    if args.n_images > 0:
        halfsets = [ halfset[:args.n_images//2] for halfset in halfsets]
        logging.info(f"using only {args.n_images} images")
    return halfsets


def load_dataset_from_args(args, lazy = False):
    ind_split = figure_out_halfsets(args)
    dataset_loader_dict = make_dataset_loader_dict(args)
    return get_split_datasets_from_dict(dataset_loader_dict, ind_split, lazy = lazy)



# Only used in recovar_coding_example.ipynb  
def get_default_dataset_option():
    dataset_loader_dict = { 'particles_file' : None,
                            'ctf_file': None ,
                            'poses_file' : None,
                            'datadir': None,
                            'n_images' : -1,
                            'ind': None,
                            'padding' : 0,
                            # 'lazy': False,
                            'uninvert_data' : False}
    return dataset_loader_dict

def load_dataset_from_dict(dataset_loader_dict, lazy = True):
    # if dataset_loader_dict['lazy']:
    #     return load_cryodrgn_dataset(**dataset_loader_dict, lazy = lazy)
    return load_cryodrgn_dataset(**dataset_loader_dict, lazy = lazy)


def reorder_to_original_indexing(arr, cryos ):
    if type(arr) is list:
        arr = np.concatenate(arr)
    dataset_indices = np.concatenate([ cryo.dataset_indices for cryo in cryos])
    inv_argsort = np.argsort(dataset_indices)
    return arr[inv_argsort]
