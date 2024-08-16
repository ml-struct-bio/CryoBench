import numpy 

DEFAULT_FFT_NORM = "backward"

# I did this a long time ago I could swap backends between numpy and jax.numpy easily.
# Unclear whether it's of any use anymore, and it makes the import very wonky.

class fourier_transform_utils:
    def __init__(self, numpy_backend = numpy) -> None:
        self.np = numpy_backend


    def get_1d_frequency_grid(self, N, voxel_size = 1, scaled = False):
        if N % 2 == 0:
            grid =  self.np.linspace( - N/2, N/2 - 1 , N) 
        else:
            grid =  self.np.linspace( - (N - 1)/2, (N- 1)/2 , N)

        if scaled:
            grid = grid / ( N * voxel_size)
        
        return grid.astype(self.np.float32)

    def get_k_coordinate_of_each_pixel(self,image_shape, voxel_size, scaled = True):
        one_D_grids = [ self.get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape ]

        grids = self.np.meshgrid(*one_D_grids, indexing="xy")
        return self.np.transpose(self.np.vstack([self.np.reshape(g, -1) for g in grids])).astype(one_D_grids[0].dtype)  


    def get_k_coordinate_of_each_pixel_3d(self,image_shape, voxel_size, scaled = True):
        # Very unfortunate change from xy to ij. FIX THIS 
        one_D_grids = [ self.get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape ]
        
        grids = self.np.meshgrid(*one_D_grids, indexing="ij")
        return self.np.transpose(self.np.vstack([self.np.reshape(g, -1) for g in grids])).astype(one_D_grids[0].dtype)  


    def get_grid_of_radial_distances(self, image_shape, voxel_size = 1, scaled = False, frequency_shift = 0, rounded = True ):
        result = self.np.linalg.norm(self.get_k_coordinate_of_each_pixel(image_shape, voxel_size = voxel_size, scaled = scaled) - frequency_shift, axis = -1)
        if rounded:
            return self.np.round(result).astype(int)
        else:
            return result


    def get_dft(self, img, norm = DEFAULT_FFT_NORM):
        # The First FFTSHIFT accounts for the phase shift difference between DFT and continuous FT.
        return self.np.fft.fftshift(self.np.fft.fft(self.np.fft.fftshift(img, axes = (-1)), norm = norm ), axes = ( -1))

    def get_idft(self, img, norm = DEFAULT_FFT_NORM):
        return self.np.fft.ifftshift(self.np.fft.ifft(self.np.fft.ifftshift(img, axes = (-1)), norm = norm ), axes = (-1 ))

    
    def get_dft2(self, img, norm = DEFAULT_FFT_NORM):
        # The First FFTSHIFT accounts for the phase shift difference between DFT and continuous FT.
        return self.np.fft.fftshift(self.np.fft.fft2(self.np.fft.fftshift(img, axes = (-2,-1)), norm = norm), axes = (-2, -1))

    def get_idft2(self, img, norm = DEFAULT_FFT_NORM):
        return self.np.fft.ifftshift(self.np.fft.ifft2(self.np.fft.ifftshift(img, axes = (-2, -1)), norm =norm), axes = (-2,-1 ))


    def get_dft3(self, img, norm = DEFAULT_FFT_NORM):
        # The First FFTSHIFT accounts for the phase shift difference between DFT and continuous FT.
        # return self.np.fft.fftshift(self.np.fft.fftn(self.np.fft.fftshift(img, axes = (-3, -2,-1)), axes = (-3, -2,-1 )), axes = (-3, -2, -1))
        return self.get_dft3_weird(img, norm = norm)

    # Jax ifftn is broken...
    def get_dft3_weird(self, u_res, norm = DEFAULT_FFT_NORM):
        u_res = self.np.fft.fftshift(u_res, axes = (-3, -2, -1))
        u_res = self.np.fft.fft(u_res, axis = -3, norm = norm)
        u_res = self.np.fft.fft(u_res, axis = -2, norm = norm)
        u_res = self.np.fft.fft(u_res, axis = -1, norm = norm)
        u_res = self.np.fft.fftshift(u_res, axes = (-3, -2, -1))
        return u_res

        
    def get_idft3(self, img, norm = DEFAULT_FFT_NORM):
        # return self.np.fft.ifft2(self.np.fft.ifftshift(img, axes = (-2, -1)))
        return self.get_idft3_weird(img, norm = norm)#self.np.fft.ifftshift(self.np.fft.ifftn(self.np.fft.ifftshift(img, axes = (-3, -2, -1)), axes = (-3, -2,-1 )), axes = (-3, -2,-1 ))

    # Jax ifftn is broken...
    def get_idft3_weird(self, u_res, norm = DEFAULT_FFT_NORM):
        u_res = self.np.fft.ifftshift(u_res, axes = (-3, -2, -1))
        u_res = self.np.fft.ifft(u_res, axis = -3, norm = norm)
        u_res = self.np.fft.ifft(u_res, axis = -2, norm = norm)
        u_res = self.np.fft.ifft(u_res, axis = -1, norm = norm)
        u_res = self.np.fft.ifftshift(u_res, axes = (-3, -2, -1))
        return u_res

    # # These are possibly broken in JAX, but not used in the code.
    # def get_dftn(self, img, norm = DEFAULT_FFT_NORM):
    #     return self.np.fft.fftshift(self.np.fft.fftn(self.np.fft.fftshift(img)))

    # def get_idftn(self, img, norm = DEFAULT_FFT_NORM):
    #     return self.np.fft.ifftshift(self.np.fft.ifftn(self.np.fft.ifftshift(img)))


    ### FUNCTIONS BELOW ARE OLD, BELOW COULD THAKE THEM OUT

    def get_grid_of_radial_distances(self, image_shape, voxel_size = 1, scaled = False, frequency_shift = 0 ):
        
        one_D_grids = [ self.get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape ]
        
        ## THIS IS I,J, BECAUSE... REASONS??? FOR 3D, IJ is correct and for 2D xy is correct somehow??
        grids = self.np.stack(self.np.meshgrid(*one_D_grids, indexing="ij"), axis =-1).astype(one_D_grids[0].dtype)   
        # grids = self.get_k_coordinate_of_each_pixel(image_shape, voxel_size = voxel_size, scaled = scaled)
        r = self.np.linalg.norm(grids - frequency_shift, axis = -1)
        if scaled:
            return r
        else:
            return self.np.round(r).astype(int)
        
    # def DFT_to_FT_scaling_vector(self, N, voxel_size):
    #     frequencies = self.get_1d_frequency_grid(N, voxel_size = 1, scaled = False)
    #     return self.np.exp(1j*self.np.pi*frequencies)

    # def DFT_to_FT_scaling(self, img_ft, voxel_size):
    #     N = img_ft.shape[0]
    #     scaling_vec = self.DFT_to_FT_scaling_vector(N, voxel_size)
        
    #     if img_ft.ndim ==1:
    #         phase_shifted_img_ft = (img_ft * scaling_vec) 
    #     if img_ft.ndim ==2:
    #         phase_shifted_img_ft = ((img_ft * scaling_vec[:,None]) * scaling_vec[None, :])
    #     if img_ft.ndim ==3:
    #         phase_shifted_img_ft = ((img_ft * scaling_vec[:,None,None]) * scaling_vec[None, :,None])* scaling_vec[None,None,:]
    #     return phase_shifted_img_ft
        

    # def FT_to_DFT_scaling(self, img_ft, voxel_size):
    #     N = img_ft.shape[0]
    #     scaling_vec = self.DFT_to_FT_scaling_vector(N, voxel_size)
        
    #     if img_ft.ndim ==1:
    #         phase_shifted_img_ft = (img_ft / scaling_vec) 
    #     if img_ft.ndim ==2:
    #         phase_shifted_img_ft = ((img_ft / scaling_vec[:,None]) / scaling_vec[None, :]) 
    #     if img_ft.ndim ==3:
    #         phase_shifted_img_ft = ((img_ft / scaling_vec[:,None,None]) / scaling_vec[None, :,None]) / scaling_vec[None,None,:]
        
    #     return phase_shifted_img_ft


    # def DFT_to_FT(self, img_dft, voxel_size):
    #     return self.DFT_to_FT_scaling(self.np.fft.fftshift(img_dft), voxel_size)
        
    # def FT_to_DFT(self, img_ft, voxel_size):
    #     return self.np.fft.ifftshift(self.FT_to_DFT_scaling(img_ft, voxel_size))

    # def compute_index_dict(self, img_shape):
    #     r = self.get_grid_of_radial_distances(img_shape)
    #     from collections import defaultdict
    #     r_dict = defaultdict(list)
    #     for idx, ri in enumerate(r.flatten()):
    #         r_dict[ri].append(idx)    
    #     return r_dict


    # def compute_spherical_average_from_index_dict(self, img_ft, r_dict, use_abs = False):
    #     max_freq = self.np.max(list(r_dict.keys()))
    #     spherical_average = self.np.zeros(max_freq + 1, dtype = img_ft.dtype)
    #     img_ft_flat = img_ft.flatten()
    #     if use_abs:
    #         img_ft_flat = self.np.abs(img_ft_flat)
    #     for key, value in r_dict.items():
    #         spherical_average[key] = self.np.mean(img_ft_flat[value])
    #     return spherical_average


    # def compute_spherical_average(self, img_ft, r_dict = None, use_abs = False):
    #     r_dict = self.compute_index_dict(img_ft.shape) if r_dict is None else r_dict
    #     return self.compute_spherical_average_from_index_dict(img_ft, r_dict, use_abs = use_abs)

    
    # def make_spherical_image(self, spherical_average, image_shape ):
    #     r_dict = self.compute_index_dict(image_shape)
    #     image_flatten = self.np.zeros(self.np.prod(image_shape), dtype = spherical_average.dtype)

    #     for key, value in r_dict.items():
    #         if key <= (image_shape[0]//2):            
    #             image_flatten[value] = spherical_average[key]

    #     return image_flatten.reshape(image_shape)

