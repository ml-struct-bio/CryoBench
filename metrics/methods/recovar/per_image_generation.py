import pickle
import numpy as np


def main():
    recovar_result_dir = "/home/mg6942/mytigress/10345/recovar_data/radial/"

    embeddings = pickle.load(open(recovar_result_dir + "model/embeddings.pkl", "rb"))
    zdim = 10
    zs = embeddings["zs"][zdim]

    # Sort by first coordinate?
    sorted_idx = np.argsort(zs[:, 0])
    zs = zs[sorted_idx]

    num_images = zs.shape[0]
    # Take one hundred images, evenly spaced in number of zs
    indices = (np.linspace(0, num_images, 100)).astype(int)
    target_zs = zs[indices]

    # If in the original indexing, things are sorted by the ground truth labels, and you want to sample every 100 in the ground truth label, then you can run this:

    ### NOTE that since imports recovar and it is not pip installed, you should recovar this file to recovar folder to run it.

    # from recovar import output, dataset
    # pipeline_output = output.PipelineOutput(recovar_result_dir + '/')
    # cryos = pipeline_output.get('lazy_dataset')
    # zs = pipeline_output.get('zs')[zdim]
    # zs_reordered = dataset.reorder_to_original_indexing(zs, cryos )
    # num_images  = zs.shape[0]
    # # Take one hundred images, evenly spaced in number of zs
    # indices = (np.linspace(0, num_images, 100)).astype(int)
    # target_zs = zs_reordered[indices]

    output_dir = "zs_to_eval.txt"
    np.savetxt(output_dir, target_zs)

    return


if __name__ == "__main__":
    main()
