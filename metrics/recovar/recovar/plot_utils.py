from matplotlib import colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib
import warnings
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import pandas as pd
import jax.numpy as jnp
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
from recovar import regularization, utils
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# I copy pasted this from an older project. Most of this is probably useless

names_to_show = { "diagonal": "diagonal", "wilson": "Wilson", "diagonal masked": "diagonal masked", "wilson masked": "Wilson masked" }
colors_name = { "diagonal": "cornflowerblue", "wilson": "lightsalmon", "diagonal masked": "blue", "wilson masked": "orangered"  }
plt.rcParams['text.usetex'] = True

def plot_power_spectrum(volume, ax = None):
    input_ax_is_none = ax is None
    if input_ax_is_none:
        plt.figure(figsize=(6, 5))
        ax = plt.gca() 
    avg = utils.average_over_shells(volume, volume.shape)
    ax.semilogy(avg)
    return avg

def plot_noise_profile(pipeline_output, yscale = 'linear'):
    plt.figure(figsize = (8,8))
    results = pipeline_output.params
    yy = results['noise_var']
    if results['input_args'].ignore_zero_frequency:
        yy[0] =0 
    plt.errorbar(x = np.arange(results['noise_var'].size), y=yy, yerr=2*results['std_noise_var'], capsize=3, alpha = 0.5, label = 'estimated noise power spectrum')
    plt.errorbar(x = np.arange(results['image_PS'].size), y=results['image_PS'], yerr=2*results['std_image_PS'], capsize=3, alpha = 0.5, label = 'image power spectrum')
    plt.plot(np.arange(results['image_PS'].size), results['cov_noise']*np.ones_like(results['image_PS']))
    plt.yscale(yscale)
    plt.legend()
    
    return

def compare_two_volumes(cryo, vol1, vol2, from_ft_inp = True):
    import matplotlib
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 4
    fig, axs = plt.subplots(n_plots, 6, figsize = ( 6*3, n_plots * 3))#, 6*3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = ftu.get_dft3(vol.reshape(cryo.volume_shape)).reshape(-1)
        global is_first
        
        axs[n_plot,0].set_ylabel(name)

        for k in range(3):

            img = cryo.get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:            
                axs[n_plot,k].set_title(f"projection {k}")

                # axs[n_plot,k].set_ylabel(name)#f"projection {k}")#, fontsize = 20)

        for k in range(3,6):

            img = cryo.get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            
            if is_first:            
                axs[n_plot,k].set_title(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return


    plot_vol(vol1, 0, from_ft = from_ft_inp, name = 'vol1')
    plot_vol(vol2, 1, from_ft = from_ft_inp,name = 'vol2')
    vol_diff = ftu.get_idft3((vol1-vol2).reshape(cryo.volume_shape)).reshape(-1)
    plot_vol(vol_diff,2, from_ft = not from_ft_inp,name = 'diff')
    plot_vol(np.abs(vol_diff),3, from_ft = not from_ft_inp,name = '||diff||')
    
    plt.subplots_adjust(wspace=0, hspace=0)

    return

def plot_summary_t(pipeline_output,cryos, n_eigs = 3, u_key = "rescaled"):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 2 + n_eigs
    fig, axs = plt.subplots(n_plots, 6, figsize = ( 6*3, n_plots * 3))#, 6*3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = ftu.get_dft3(vol.reshape(cryos[0].volume_shape)).reshape(-1)
        global is_first
        
        axs[n_plot,0].set_ylabel(name)

        for k in range(3):

            img = cryos[0].get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:            
                axs[n_plot,k].set_title(f"projection {k}")

                # axs[n_plot,k].set_ylabel(name)#f"projection {k}")#, fontsize = 20)

        for k in range(3,6):

            img = cryos[0].get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            
            if is_first:            
                axs[n_plot,k].set_title(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return


    plot_vol(pipeline_output.get('mean'), 0, from_ft = True, name = 'mean')
    plot_vol(pipeline_output.get('volume_mask'), 1, from_ft = False,name = 'mask')
    u = pipeline_output.get('u_real')
    for k in range(n_eigs):
        plot_vol(u[k], k+2, from_ft = False, cmap = 'seismic' ,name = f"PC {k}", symmetric = True)

    plt.subplots_adjust(wspace=0, hspace=0)

    return

def plot_summary(results,cryos, n_eigs = 3):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 2 + n_eigs
    fig, axs = plt.subplots(6, n_plots, figsize = (6*3, n_plots * 3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = ftu.get_dft3(vol.reshape(cryos[0].volume_shape)).reshape(-1)
        global is_first
        axs[0, n_plot].set_title(name)
        for k in range(3):
            img = cryos[0].get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            if is_first:
                axs[k, n_plot].set_ylabel(f"projection {k}")#, fontsize = 20)

        for k in range(3,6):
            img = cryos[0].get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            
            if is_first:
                axs[k, n_plot].set_ylabel(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return

    plot_vol(results['means']['combined'], 0, from_ft = True, name = 'mean')
    plot_vol(results['volume_mask'], 1, from_ft = False,name = 'mask')
    for k in range(n_eigs):
        plot_vol(results['u']['rescaled'][:,k], k+2, from_ft = True, cmap = 'seismic' ,name = f"eigen{k}", symmetric = True)

    plt.subplots_adjust(wspace=0, hspace=0)

    
    return
    
def plot_volume_sequence(volumes,cryos):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = len(volumes)
    fig, axs = plt.subplots(6, n_plots, figsize = (6*3, n_plots * 3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = ftu.get_dft3(vol.reshape(cryos[0].volume_shape)).reshape(-1)
        global is_first
        axs[0, n_plot].set_title(name)
        for k in range(3):
            img = cryos[0].get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            if is_first:
                axs[k, n_plot].set_ylabel(f"projection {k}")#, fontsize = 20)

        for k in range(3,6):
            img = cryos[0].get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            
            if is_first:
                axs[k, n_plot].set_ylabel(f"slice {k-3}")#, fontsize = 20)

        is_first = False
            
    for vol_idx,vol in enumerate(volumes):
        plot_vol(results['means']['combined'], vol_idx, from_ft = True, name = f"vol{vol_idx}")

    plt.subplots_adjust(wspace=0, hspace=0)
    
    return
    
    


def plot_cov_results(u,s, max_eig = 40, savefile = None):
    
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    m = np.tile(["o", "s", "D", "*", '<','8', '>'], 5)

    m_idx = 0 
    #s['parallel_analysis'] *= 0.8
    plt.figure(figsize = (6,6))
    
    names_to_plot = dict(zip(list(s.keys()), list(s.keys())))
    names_to_plot['real'] = 'init'            
    names_to_plot['rescaled'] = 'SVD'
                        
    for key in s.keys():
        # for key in used_keys:
        if "+" in key:
            continue

        plt.plot(np.arange(1,s[key][:max_eig].size+1), s[key][:max_eig],  "-"+m[m_idx], label = names_to_plot[key], alpha = 0.5, ms = 15)
        m_idx = (m_idx + 1) % len(m)
        plt.yscale('log')
        plt.legend()
        # plt.savefig(output_folder + 'plots/eigenvals.pdf')
        
    if 'parallel_analysis' in s:
        plt.semilogy(np.ones_like(s[key][:max_eig]) * s['parallel_analysis'][0], "-"+m[m_idx], label = "par0", alpha = 0.5, ms = 15)

    plt.title('eigenvalues')
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile + 's.png')

    gt_key = "gt"
    angles ={}
    if gt_key in u:
        plt.figure(figsize = (6,6))
        m = np.repeat(["o", "s", "D", "*", 'x', '>'], 3, 0)
        m_idx = 0
        for key in u.keys():
            if key == gt_key:
                continue
            pkey = key  

            max_subspace_size = np.min([15, u[key].shape[-1]])
            angles[key] = utils.subspace_angles(u[key], u[gt_key], max_subspace_size)
            plt.plot( np.arange(1,1+len(angles[key])), angles[key], "-"+m[m_idx], label = pkey, alpha = 0.5,  ms = 15)
            m_idx = (m_idx + 1) % len(m)
        plt.ylim([-0.05,1.05])
        plt.legend()
        if savefile is not None:
            plt.savefig(savefile + 'u.png')

    return angles

def plot_mean_fsc(pipeline_output,cryos):

    halfmap1, halfmap2 = pipeline_output.get('mean_halfmaps')

    ax, score = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = None, threshold = 1/7, filename = None, name = "unmasked")

    ax, score_masked = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = ax, threshold = 1/7, filename = None, volume_mask = pipeline_output.get('volume_mask'), name = "masked")
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}

    ax.set_title("mean estimation", fontsize=20)
    return ax
    
def plot_fsc(cryo, vol1, vol2, mask = None, threshold = 1/7, ax = None, voxel_size= None, volume_shape= None, name = "unmasked", fmat = ""):
    voxel_size = cryo.voxel_size if voxel_size is None else voxel_size
    volume_shape = cryo.volume_shape if volume_shape is None else volume_shape

    ax, score = plot_fsc_new(vol1, vol2, volume_shape, voxel_size,  curve = None, ax = ax, threshold = threshold, filename = None, name = name, volume_mask = mask, fmat = fmat)
    return ax
    
    

def plot_mean_result(cryo, means, cov_noise):
    # Check power spectrums
    volume_shape = cryo.volume_shape
    plt.figure(figsize = (6,6))

    for mean_key in means.keys():
        if "1" in mean_key or 'prior' in mean_key:
            continue
        PS = regularization.average_over_shells( np.abs(means[mean_key])**2 , volume_shape)
        plt.semilogy( PS, '-o', label = mean_key, ms = 15, alpha = 0.3)

    noise_level = np.ones_like(PS) * cov_noise
    plt.semilogy( noise_level , '--', label = "noise", ms =15 )

    PS_prior = regularization.average_over_shells(  means["prior"] , volume_shape)
    plt.semilogy( PS_prior , '-.', label = "prior", ms = 15)
    plt.legend()

    # GSFSC
    plot_fsc_function_paper_simple()

    score = cryo.plot_FSC(means["corrected0"], means["corrected1"], threshold = 1/7 )
    
    # GSFSC
    if "ground_truth" in means:
        cryo.plot_FSC(means["combined"], means["ground_truth"], threshold = 1/2)
        
    # projection images
    axis = 0
    plt.figure()
    plt.imshow(cryo.get_proj(means["combined"], axis = axis))
    plt.colorbar()
    plt.title('combined')

    if "ground_truth" in means:
        plt.figure()
        plt.imshow(cryo.get_proj(means["ground_truth"], axis = axis))
        plt.title('ground truth')
        plt.colorbar()

        
        
def plot_fsc_new(image1, image2, volume_shape, voxel_size,  curve = None, ax = None, threshold = 1/7, filename = None, volume_mask = None, name = "", fmat = ""):
    grid_size = volume_shape[0]
    input_ax_is_none = ax is None
    if input_ax_is_none:
        plt.figure(figsize=(6, 5))
        ax = plt.gca() 

    if volume_mask is not None:
        image1 = ftu.get_idft3(image1.reshape(volume_shape))
        image2 = ftu.get_idft3(image2.reshape(volume_shape))
        image1 = ftu.get_dft3(image1 * volume_mask)
        image2 = ftu.get_dft3(image2 * volume_mask)

    # get_fsc_gpu
    if curve is None:
        curve = FSC(np.array(image1).reshape(volume_shape), np.array(image2).reshape(volume_shape))
    
    # import pdb; pdb.set_trace()
    # Huuuh why is there a 1/2 here??
    freq = ftu.get_1d_frequency_grid(grid_size, voxel_size = voxel_size, scaled = True)
    freq = freq[freq >= 0 ]
    freq = freq[:grid_size//2 ]
    max_idx = min(curve.size, freq.size)
    line, = ax.plot(freq[:max_idx], curve[:max_idx],  fmat, linewidth = 2 )
    color = line.get_color()
    
    if threshold is not None:
        score = fsc_score(curve, grid_size, voxel_size, threshold = threshold)

        label = name + " "+ "{:.2f}".format(1 / score)+ "\AA"
        n_dots_in_line = 20
        ax.plot(np.ones(n_dots_in_line) * score, np.linspace(0,1, n_dots_in_line), "-", color = color, label= label)
        ax.plot(freq, threshold * np.ones(freq.size), "k--")
    else:
        score = None

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    if input_ax_is_none:

        plt.ylim([0, 1.02])
        plt.xlim([0, np.max(freq)])
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=10) 
        # plt.legend("AA{-1}) = "+"{:.2f}".format(threshold))
        # plt.title("FSC("  + "{:.2f}".format(1 / score)  + "AA{-1}) = "+ "{:.2f}".format(threshold))
    if threshold is not None:  
        ax.plot(freq, threshold * np.ones(freq.size), "k--")
        ax.legend()

    if filename is not None:
        plt.savefig(filename )
    return ax, score


def plotly_scatter(points_to_plot, opacity = 0.1):

    import plotly
    import plotly.graph_objects as go



    fig = go.Figure(data=[go.Scatter3d(x=points_to_plot[0][:,0], y=points_to_plot[0][:,1], z=points_to_plot[0][:,2],
                                    mode='markers', opacity = opacity)])

    for k in range(1, len(points_to_plot)):
        fig.add_trace(go.Scatter3d(x=points_to_plot[k][:,0], y=points_to_plot[k][:,1], z=points_to_plot[k][:,2],
                                        mode='markers', opacity = opacity))


    fig.update_layout(scene_aspectmode='cube')


    fig.show()

def compute_and_plot_fsc(image1, image2, volume_shape = None, voxel_size =1):
    if volume_shape is not None:
        image1 = image1.reshape(volume_shape)
        image2 = image2.reshape(volume_shape)
        grid_size = volume_shape[0]
    else:
        volume_shape = image1.shape
        grid_size = volume_shape[0]

    fsc = FSC(np.array(image1), np.array(image2), r_dict = None)
    plot_fsc_function_paper_simple([fsc], grid_size, voxel_size)
    return fsc
    
    
### PLOTTING STUFF
def FSC(image1, image2, r_dict = None):
    from recovar import regularization
    fsc = regularization.get_fsc_gpu(image1, image2, image1.shape, False, frequency_shift = 0 )
    return fsc



def fsc_score(fsc_curve, grid_size, voxel_size, threshold = 0.5 ):
    # First index below 0.5
    freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
    freq = freq[freq >= 0 ]

    freq = freq[1:]
    fsc_curve = fsc_curve[1:]

    # First index above threshold
    above_threshold = fsc_curve >= threshold
    if np.all(above_threshold):
        return freq[-1]
    
    if np.all(~above_threshold):
        return freq[0]


    idx = int(np.max([np.argmin(above_threshold), 0])) 
    if idx >= grid_size//2 -1:
        return freq[-1]
    if idx == 0:
        return freq[idx]
    idx = idx-1
        
    # Linearly interpolate
    from scipy import interpolate
    fsc_curve = np.where(np.isnan(fsc_curve), 0, fsc_curve)
    f = interpolate.interp1d( np.array([fsc_curve[idx], fsc_curve[idx+1]]), np.array([freq[idx], freq[idx+1]]) )
    
    return np.min([f(threshold), 2 * voxel_size])


def plot_all_images_for_paper(images_to_plot, n_dirs, global_name, scale_image_name, voxel_size, save_to_file):
    for (key, image) in images_to_plot.items():
        if save_to_file:
            import mrcfile
            with mrcfile.new(global_name + key + ".mrc", overwrite=True) as mrc:
                mrc.set_data(image.astype(np.float32))

    for idx in range(n_dirs):
        projections = {}
        for (key, image) in images_to_plot.items():
            projections[key] = np.real(np.sum(image, axis = idx))
        plot_images_on_same_scale(projections, global_name + "proj" + str(idx), scale_image_name,voxel_size, save_to_file)

        slices = {}
        for (key, image) in images_to_plot.items():
            slices[key] = np.real(image.take(image.shape[0]//2, axis = idx))
        plot_images_on_same_scale(slices, global_name + "slice" + str(idx), scale_image_name, voxel_size,save_to_file)

def plot_images_on_same_scale(images_to_plot, global_name, scale_image_name,  voxel_size, save_to_file):
    min_val_all = np.inf
    max_val_all = - np.inf
    for img in images_to_plot.values():
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    for (name, image) in images_to_plot.items():
        plot_function_with_scale(image, global_name + name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = False)
    if scale_image_name in images_to_plot:
        plot_function_with_scale(images_to_plot[scale_image_name],  global_name + scale_image_name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = True)
    return
    

def plot_function_with_scale(image, name, vmin, vmax, voxel_size, save_to_file = False, show_colorbar = False, show_scalebar = True):
        # Create subplot
        fig, ax = plt.subplots()
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
        
        if show_scalebar:
            # Create scale bar
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)

        if show_colorbar:
            fig.colorbar(pos, ax = ax)
        if save_to_file:
            plt.savefig(name + ".pdf", bbox_inches='tight')

            
def plot_fsc_function_paper_simple(fsc_curves, grid_size, voxel_size):
    return plot_fsc_function_paper(fsc_curves, "", "", grid_size, voxel_size, save_to_file = False)
                        
            
def plot_fsc_function_paper(fsc_curves, global_name, names, grid_size, voxel_size, save_to_file = False):
    plot_fsc_flag = True
    if plot_fsc_flag:
        plt.figure(figsize=(9, 8))
        ax = plt.gca()
        plt.rcParams['text.usetex'] = False
        freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
        freq = freq[freq >= 0 ]
        freq = freq[:grid_size//2 ]

        lines = []
        line_names = []
        colors_plotted  = []
        names_plotted  = []
        scores = {}
        for name in names:
            curve = fsc_curves[name]
            max_idx = min(curve.size, freq.size)
            plt.plot(freq[:max_idx], curve[:max_idx], color = colors[colors_name[name]] , label = names_to_show[name] , linewidth = 2 )
            colors_plotted.append(colors[colors_name[name]])
            names_plotted.append(names_to_show[name])

            curve = fsc_curves[name + " masked"]
            max_idx = min(curve.size, freq.size)
            line = plt.plot(freq[:max_idx], curve[:max_idx],color =  colors[colors_name[name + " masked"]], label = names_to_show[name + " masked"], linewidth = 2 )
            colors_plotted.append(colors[colors_name[name +" masked"]])
            names_plotted.append(names_to_show[name + " masked"])

            scores[name] = fsc_score(curve, grid_size, voxel_size)

        n_dots_in_line = 20
        for name in names:
            plt.plot(np.ones(n_dots_in_line) * scores[name], np.linspace(0,1, n_dots_in_line), "k-")

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        plt.plot(freq, 0.5 * np.ones(freq.size), "k--")
        plt.ylim([0, 1.02])
        plt.xlim([0, np.max(freq)])
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=20) 

        if save_to_file:
            global_name = global_name + "fsc.pdf"
            plt.savefig(global_name, bbox_inches='tight')

        plt.figure()
        f = lambda m,c: plt.plot([],[], color=c, ls="-")[0]
        handles = [f("-", colors_plotted[i]) for i in range(len(colors_plotted))]
        labels = names_plotted
        legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True, prop={'size': 20})
        if save_to_file:
            export_legend(legend, filename = global_name + "legend.pdf")
        plt.show()


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)



# Plotting for sample figures
def plot_samples_on_same_scale(images_to_plot, save_filepath, voxel_size, save_to_file):
    min_val_all = 0
    max_val_all = -np.inf
    for img in images_to_plot:
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    idx = 0 
    for image in images_to_plot:
        plot_sample_with_scale(image, save_filepath + str(idx), voxel_size, min_val_all, max_val_all, save_to_file = save_to_file, show_colorbar = True)
        idx +=1 
    return

def plot_sample_with_scale(image, save_filepath, voxel_size, vmin , vmax , save_to_file = False, show_colorbar = True, show_scalebar = True):
        fig, ax = plt.subplots(figsize = (4,4))
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
    
        if show_scalebar:
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)
        plt.show()

        if save_to_file:
            plt.savefig(save_filepath, bbox_inches='tight')

        if show_colorbar:
            fig, ax = plt.subplots(figsize = (4,4))
            plt.gca().set_visible(False)
            cbar = fig.colorbar(pos, orientation="horizontal")
            cbar.ax.tick_params(labelsize=14)
            plt.savefig(save_filepath + "colorbar" , bbox_inches='tight')
