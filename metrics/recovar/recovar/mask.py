import logging
import jax.numpy as jnp
import numpy as np
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)
from recovar import utils 
import skimage
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

def masking_options(volume_mask_option, means, volume_shape, input_mask, dtype_real = np.float32, mask_dilation_iter = 0):
    dilation_iterations = np.ceil(6 * volume_shape[0] / 128).astype(int)
    if isinstance(volume_mask_option, str):
        if volume_mask_option == 'input' or input_mask is not None :
            assert input_mask is not None, 'set volume_mask_option = input, but no mask passed'
            input_mask = utils.load_mrc(input_mask) 
            logger.info('Using input mask')

            if mask_dilation_iter > 0:
                logger.info('thresholding and dilating input mask')
                input_mask = input_mask > 0.99
                input_mask = binary_dilation(input_mask,iterations=mask_dilation_iter)       

            if input_mask.shape[0] != volume_shape[0]:
                input_mask = skimage.transform.rescale( input_mask, volume_shape[0]/input_mask.shape[0])

            kernel_size = 3
            logger.info('Thresholding mask at 0.5 and softening cosine kernel of radius 3 pixels')
            input_mask = input_mask > 0.5
            volume_mask = soften_volume_mask(input_mask, kernel_size)
            dilated_volume_mask = binary_dilation(input_mask,iterations=dilation_iterations)
            dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
        elif volume_mask_option == 'from_halfmaps':
            volume_mask = make_mask_from_half_maps_from_means_dict(means, smax = 3 )
            kernel_size = 3
            logger.info('Softening mask')

            dilated_volume_mask = binary_dilation(volume_mask,iterations=dilation_iterations)
            volume_mask = soften_volume_mask(volume_mask, kernel_size)
            dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
            logger.info('using mask computed from mean')
        elif volume_mask_option == 'sphere':
            volume_mask = get_radial_mask(volume_shape)
            dilated_volume_mask = get_radial_mask(volume_shape)
            logger.info('using spherical mask')
        elif volume_mask_option == 'none':
            volume_mask = np.ones(volume_shape)
            dilated_volume_mask = volume_mask
            logger.info('using no mask')
    else:
        assert False, 'mask option not recognized'
    return np.array(volume_mask.astype(dtype_real)), np.array(dilated_volume_mask.astype(dtype_real))

def make_mask_from_half_maps_from_means_dict(means, smax = 3 ):
    # from emda.ext.maskmap_class import MaskedMaps
    # ftu = fourier_transform_utils(np)
    # x = MaskedMaps()
    # x.smax = smax
    vol_shape = utils.guess_vol_shape_from_vol_size(means['corrected0'].size)
    halfmap1 = ftu.get_idft3(means['corrected0'].reshape(vol_shape)).real
    halfmap2 = ftu.get_idft3(means['corrected1'].reshape(vol_shape)).real
    return make_mask_from_half_maps(halfmap1, halfmap2, smax = smax )


def make_mask_from_half_maps(halfmap1, halfmap2, smax = 3 ):
    # from emda.ext.maskmap_class import MaskedMaps
    # ftu = fourier_transform_utils(np)
    x = MaskedMaps()
    x.smax = smax
    x.arr1 = halfmap1
    x.arr2 = halfmap2
    x.iter = int(6 * halfmap1.shape[0] // 128)
    x.generate_mask()
    return x.mask


def make_mask_from_gt(gt_map_ft, smax = 3, iter = 10, from_ft = True ):
    # from emda.ext.maskmap_class import MaskedMaps
    ftu = fourier_transform_utils(np)
    x = MaskedMaps()
    x.smax = smax
    if iter is not None:
        x.iter = iter
    # x.iter = x.iter +5
    vol_shape = utils.guess_vol_shape_from_vol_size(gt_map_ft.size)
    if from_ft:
        x.arr1 = ftu.get_idft3(gt_map_ft.reshape(vol_shape)).real
    else:
        x.arr1 = gt_map_ft.reshape(vol_shape)
    # x.arr2 = ftu.get_idft3(means['corrected1'].reshape(vol_shape)).real
    x.generate_mask_from_gt()
    return x.mask


def create_soft_edged_kernel_pxl(r1, shape):
    # Create soft-edged-kernel. r1 is the radius of kernel in pixels
    # This implementation is adapted from EMDA - https://gitlab.com/ccpem/emda
    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1
    
    # Are these offset by 1 pixel ? or 1/2 or something
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))]) + 1
    distances =  jnp.linalg.norm(volume_coords, axis =-1)
    half_boxsize = boxsize // 2
    r1 = half_boxsize
    r0 = r1 - 2
    
    kern_sphere_soft = jnp.where((distances < r0), jnp.ones_like(distances), jnp.zeros_like(distances))
    # 1
    kern_sphere_soft = jnp.where((distances <= r1) * (distances >= r0),
                                 (1 + jnp.cos(jnp.pi * (distances - r0) / (r1 - r0))) / 2.0,
                                 kern_sphere_soft )
    return kern_sphere_soft / jnp.sum(kern_sphere_soft)

def create_hard_edged_kernel_pxl(r1, shape):
    
    # Are these offset by 1 pixel ? or 1/2 or something
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))]) + 1
    distances =  jnp.linalg.norm(volume_coords, axis =-1)
    
    kern_sphere_soft = jnp.where((distances <= r1), jnp.ones_like(distances), jnp.zeros_like(distances))

    return kern_sphere_soft / jnp.sum(kern_sphere_soft)



# def soften_volume_mask(volume_mask, kernel_size):

#     image_shape = volume_mask.shape
#     # Soft mask
#     soft_edge_kernel = create_soft_edged_kernel_pxl(kernel_size, image_shape)
#     import jax.scipy
#     volume_mask = jax.scipy.ndimage.convolve(volume_mask, soft_edge_kernel, mode='full', cval=0.0)

#     # Convolve
#     # soft_edge_kernel_ft = ftu.get_dft3(soft_edge_kernel)
#     # volume_mask_ft = ftu.get_dft3(volume_mask)
    
#     # volume_mask_ft = volume_mask_ft * soft_edge_kernel_ft
#     # volume_mask = ftu.get_idft3(volume_mask_ft).real
#     return np.array(volume_mask)


def get_radial_mask(shape, radius = None):
    radius = shape[0]//2-1 if radius is None else radius
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))])
    zero_out_outside_sphere_small =  jnp.linalg.norm(volume_coords, axis =-1) < radius + 1e-7
    return zero_out_outside_sphere_small



# Standard image masking (Other masking are used, too)
def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32), 
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask


## What is below is copy-pasted from EMDA (https://emda.readthedocs.io/en/latest/)
# It was stripped out because it caused dependency issues.

class MaskedMaps:
    def __init__(self, hfmap_list=None):
        self.hfmap_list = hfmap_list
        self.mask = None
        self.uc = None
        self.arr1 = None
        self.arr2 = None
        self.origin = None
        self.iter = 3
        self.smax = 9 # kernel radius in pixels
        self.prob = 0.99
        self.dthresh = None

    def generate_mask(self):
        kern = create_soft_edged_kernel_pxl(self.smax, self.arr1.shape)
        self.arr1 = threshold_map(arr=self.arr1, prob=self.prob, dthresh=self.dthresh)
        self.arr2 = threshold_map(arr=self.arr2, prob=self.prob, dthresh=self.dthresh)
        halfcc3d = get_3d_realspcorrelation(self.arr1, self.arr2, kern)
        
        # radial mask stuff
        halfcc3d *= get_radial_mask(self.arr1.shape)

        #self.mask = self.histogram2(halfcc3d, prob=self.prob)
        self.mask = self.thereshol_ccmap(ccmap=halfcc3d)


    ## Mine not EMDA. Could be bad.
    def generate_mask_from_gt(self):
        kern = create_soft_edged_kernel_pxl(self.smax, self.arr1.shape)
        self.arr1 = threshold_map(arr=self.arr1, prob=self.prob, dthresh=self.dthresh)
        self.arr1 = self.arr1> 0
        dilate = binary_dilation(self.arr1, iterations=self.iter)
        self.mask = soften_volume_mask(dilate, kern_rad=2)
        # import pdb; pdb.set_trace()


    def thereshol_ccmap(self, ccmap):
        from scipy.ndimage.morphology import binary_dilation

        ccmap_binary = (ccmap >= 1e-3).astype(int)
        dilate = binary_dilation(ccmap_binary, iterations=self.iter)
        mask = soften_volume_mask(dilate, kern_rad=2)
        return mask * (mask >= 1e-3)      


def threshold_map(arr, prob = 0.99, dthresh=None):
    if dthresh is None:
        X2 = np.sort(arr.flatten())
        F2 = np.array(range(len(X2))) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)


def smooth_circular_mask(image_size, radius, thickness):
    # Copy pasted from Dynamight https://github.com/3dem/DynaMight/blob/main/dynamight/data/handlers/grid.py
    y, x = np.meshgrid(
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
    )
    r = np.sqrt(x ** 2 + y ** 2)
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = np.zeros((image_size, image_size))
    mask[r < radius] = 1
    mask[band_mask] = np.cos(np.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask



def raised_cosine_mask( volume_shape, radius, radius_p, offset):
    # adapted from relion
    grid = ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size = 1, scaled = False)
    grid -= offset

    distances =  jnp.linalg.norm(grid, axis =-1)
    # mask = jnp.zeros(volume_shape)
    mask = jnp.where(distances < radius, 1, 0)
    mask = jnp.where((distances >= radius) * (distances < radius_p ), 
                    0.5 - 0.5 * jnp.cos(np.pi * (radius_p - distances) / (radius_p - radius)),
                     mask)

    return mask.reshape(volume_shape)


import numpy as np
# This is the RELION function translated by chatgpt
# https://github.com/3dem/relion/blob/e5c4835894ea7db4ad4f5b0f4861b33269dbcc77/src/mask.cpp
def soft_mask_outside_map(vol, radius=-1, cosine_width=3, Mnoise=None):
    # vol = np.roll(vol, -np.array(vol.shape) // 2)  # Assuming vol.setXmippOrigin() adjusts the origin

    vol = jnp.asarray(vol)
    if radius < 0:
        radius = np.max(np.array(vol.shape) // 2)

    radius_p = radius + cosine_width
    shape = vol.shape

    # Not very clear whether this should be 0 or 1
    volume_coords =  ftu.get_k_coordinate_of_each_pixel(shape, voxel_size = 1, scaled = False).reshape(list(shape) + [len(list(shape))]) + 0

    # r, i, j = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    r = jnp.linalg.norm(volume_coords, axis =-1)
    mask1 = r <= radius
    mask2 = (r > radius) * (r <= radius_p)
    mask3 = r > radius_p
    raised_cos = 0.5 + 0.5 * jnp.cos(jnp.pi * (radius_p - r) / cosine_width)
    mask = jnp.zeros_like(vol).real
    # mask = mask.at[mask1].set(1)
    mask = jnp.where(mask1, 1, mask)
    # mask = mask.at[mask2].set(1 - raised_cos[mask2])
    mask = jnp.where(mask2, 1 - raised_cos, mask)


    if Mnoise is None:
        sum_bg = jnp.sum((vol * mask) * (mask3 + mask2))
        sum = jnp.sum((mask) * (mask3 + mask2))
        avg_bg = sum_bg / sum
    else:
        avg_bg = None

    if Mnoise is None:
        # vol = vol.at[mask3].set(avg_bg)
        vol = jnp.where(mask3, avg_bg, vol)
    else:
        # vol = vol.at[mask3].set(Mnoise[mask3])
        vol = jnp.where(mask3, Mnoise, vol)

    add = Mnoise if Mnoise is not None else avg_bg
    vol = mask * vol + (1 - mask) * add
    # vol[mask2] = (1 - raised_cos[mask2]) * vol[mask2] + raised_cos[mask2] * add

    return vol, mask


def soften_volume_mask(dilated_mask, kern_rad=3):
    # convoluting with gaussian sphere
    import scipy.signal
    kern_sphere = create_soft_edged_kernel_pxl(kern_rad, dilated_mask.shape)
    return scipy.signal.fftconvolve(dilated_mask, kern_sphere, "same")



## from realsp_local.py in EMDA
def get_3d_realspcorrelation(half1, half2, kern, mask=None):
    import scipy.signal
    from scipy.stats import mode

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")
    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A ** 2
    var3_B = loc3_B2 - loc3_B ** 2
    # regularization
    #reg_a = mode(var3_A)[0][0][0][0] / 100
    #reg_b = mode(var3_B)[0][0][0][0] / 100
    reg_a = np.max(var3_A) / 1000
    reg_b = np.max(var3_B) / 1000
    var3_A = np.where(var3_A < reg_a, reg_a, var3_A)
    var3_B = np.where(var3_B < reg_b, reg_b, var3_B)
    halfmaps_cc = cov3_AB / np.sqrt(var3_A * var3_B) 
    return halfmaps_cc


