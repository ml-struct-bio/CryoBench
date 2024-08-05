# import recovar.config
from importlib import reload
import numpy as np
from cryodrgn import analysis
from cryodrgn import utils
from cryodrgn import ctf
from recovar import plot_utils
from recovar import output, dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from recovar import simulator
reload(simulator)
import jax
import warnings
# warnings.filterwarnings("error")
grid_size =128
for log_n in [6]:
    # output_folder ='/home/mg6942/mytigress/spike256/../'
    volume_folder_input =  f"/tigress/CRYOEM/singerlab/mg6942/simulated_empiar10180/volumes_{grid_size}_small/vol"
    output_folder = volume_folder_input+ f"/dataset_{log_n}_bump_3/"
    outlier_file_input = "/home/mg6942/mytigress/6vxx_256.mrc"
    n_images = int(10**(log_n))
    voxel_size = 4.25 * 128 / grid_size #f"{output_folder}../spike{grid_size}_small/0000.mrc"
    output.mkdir_safe(output_folder)
    volume_distribution = np.zeros(300)
    first_k = 300
    volume_distribution[:first_k] = 1/first_k

    ## Triple bump
    import numpy as np
    from scipy.stats import norm
    grid = np.linspace(0,1,300)
    means = [ 0.25, 0.5, 0.75]
    std_used = 0.05 
    weights = [ 0.5, 0.25, 0.5]
    volume_distribution = norm.pdf(grid, means[0], std_used) * weights[0]
    volume_distribution += norm.pdf(grid, means[1], std_used) * weights[1]
    volume_distribution += norm.pdf(grid, means[2], std_used) * weights[2]
    volume_distribution = volume_distribution / volume_distribution.sum()

    image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, n_images,
                                                                 outlier_file_input = outlier_file_input, grid_size = grid_size,
                                   volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level =3,
                                   noise_model = "radial1", put_extra_particles = False, percent_outliers = 0.00, 
                                   volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0.2 * 0, contrast_std =0.2 *0  , disc_type = 'nufft')
    print(f"Finished generating dataset {output_folder}")
    