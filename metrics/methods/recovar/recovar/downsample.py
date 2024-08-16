# JUST USE THE CRYODRGN ONE.

# import jax.numpy as jnp
# import numpy as np
# import dataset
# from recovar.fourier_transform_utils import fourier_transform_utils
# ftu = fourier_transform_utils(jnp)

# from cryodrgn import mrc
# def downsample_image_stack(image_stack, D, batch_size):

#     oldD = image_stack.D
#     start = int(oldD/2 - D/2)
#     stop = int(oldD/2 + D/2)
#     n_images = image_stack.n_images
#     new = np.empty((n_images, D, D), dtype=np.float32)

#     def downsample_images2(imgs):
#         oldft = ftu.get_dft2(imgs)
#         newft = oldft[:, start:stop, start:stop]
#         newft = newft.at[:,0].set(0)
#         newft = newft.at[:,:,0].set(0)
#         new = ftu.get_idft2(newft).real
#         return new

#     data_generator = dataset.NumpyLoader(image_stack, batch_size=batch_size, shuffle=False)
#     for batch, batch_image_ind in data_generator:
#         new[batch_image_ind] = np.array(downsample_images2(batch))
#         print("batch done")

#     return new
    
# def downsample_mrc(input_particle_file, output_particle_file, D, datadir = "", output_dir = "",  batch_size = 8192):
    
#     image_stack = dataset.LazyMRCDataMod( input_particle_file, ind = None, datadir = datadir, padding = 0)
#     print("done loading old")
#     new_images =  downsample_image_stack(image_stack, D, batch_size)    
#     mrc.write(output_dir + output_particle_file, new_images.astype(np.float32), is_vol=False)

# def main():    

#     datadir = "/scratch/gpfs/mg6942/10180/data/"
#     output_dir = "/scratch/gpfs/mg6942/cryodrgn_empiar/empiar10180/inputs/"
#     input_particle_file = "consensus_data.star"
        
#     datadir = "/scratch/gpfs/mg6942/10180/data/"
#     output_dir = "/scratch/gpfs/mg6942/cryodrgn_empiar/empiar10180/inputs/"
#     input_particle_file = "consensus_data.star"

#     datadir = '/tigress/CRYOEM/singerlab/mg6942/uniform/'
#     output_dir =  datadir
#     input_particle_file = "particles.128.mrcs"
#     new_grid_size = 16
#     output_particle_file = "particles."+ str(new_grid_size) + ".mrcs"
    
#     downsample_mrc(output_dir + input_particle_file, output_particle_file, new_grid_size, datadir = datadir, output_dir = output_dir,  batch_size = int(8192/16))


# if __name__ == '__main__':
#     main()
