import jax.numpy as jnp
import jax
import numpy as np
import functools

from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)


# PADDING FUNCTIONS 
    
@functools.partial(jax.jit, static_argnums = [1,2])    
def padded_dft(images, image_size, padding : int):
    n_images = images.shape[0]
    images_shape = images.shape[-2:]
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    images_big = jnp.zeros_like(images, shape = [n_images, padded_image_x, padded_image_y] )
    images_big = images_big.at[...,padding//2:images_shape[0] + padding//2, padding//2:images_shape[1] + padding//2].set(images)    
    padded_image_size = padded_image_x * padded_image_y
    return ftu.get_dft2(images_big).reshape([n_images, padded_image_size])

def pad_images_spatial_domain(images, padding):
    n_images = images.shape[0]
    images_shape = images.shape[-2:]
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    images_big = jnp.zeros_like(images, shape = [n_images, padded_image_x, padded_image_y] )
    images_big = images_big.at[...,padding//2:images_shape[0] + padding//2, padding//2:images_shape[1] + padding//2].set(images)
    return images_big


def pad_volume_spatial_domain(images, padding):
    images_shape = images.shape
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    padded_image_z = images_shape[2] + padding
    images_big = jnp.zeros_like(images, shape = [padded_image_x, padded_image_y, padded_image_z] )
    images_big = images_big.at[padding//2:images_shape[0] + padding//2, padding//2:images_shape[1] + padding//2,padding//2:images_shape[2] + padding//2].set(images)
    return images_big

def unpad_volume_spatial_domain(volume, padding):
    return volume[padding//2:volume.shape[0] - padding//2, padding//2:volume.shape[1] - padding//2, padding//2:volume.shape[2] - padding//2]

def unpad_volume_fourier_domain(volume, padded_image_shape, padding):
    volume = volume.reshape(list(padded_image_shape))
    volume = ftu.get_idft3(volume)
    unpadded_volume = unpad_volume_spatial_domain(volume, padding)
    unpadded_volume = ftu.get_dft3(unpadded_volume).reshape( -1)
    return unpadded_volume


def pad_images_fourier_domain(images, image_shape, padding):
    images = images.reshape([-1] + list(image_shape))
    images = ftu.get_idft2(images)
    padded_images = pad_images_spatial_domain(images, padding)
    return ftu.get_dft2(padded_images).reshape( [images.shape[0], -1])

def unpad_images_spatial_domain(images, padding):
    return images[..., padding//2:images.shape[-2] - padding//2, padding//2:images.shape[-1] - padding//2]

def unpad_images_fourier_domain(images, padded_image_shape, padding):
    images = images.reshape([images.shape[0]] + list(padded_image_shape))
    images = ftu.get_idft2(images)
    unpadded_images = unpad_images_spatial_domain(images, padding)
    unpadded_images = ftu.get_dft2(unpadded_images).reshape( [images.shape[0], -1])
    return unpadded_images


### PROJECTIONS BY NUFFT
