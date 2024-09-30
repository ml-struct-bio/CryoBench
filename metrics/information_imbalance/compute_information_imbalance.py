import numpy as np
import pandas as pd
import argparse

from dadapy.metric_comparisons import MetricComparisons


def compute_information_imbalance(embedding_1, embedding_2, **kwargs):
    _, embedding_1_unique_idx = np.unique(embedding_1, axis=0, return_index=True)
    _, embedding_2_unique_idx = np.unique(embedding_2, axis=0, return_index=True)

    unique_idx = list(set(embedding_1_unique_idx) & set(embedding_2_unique_idx))

    N = len(unique_idx)
    d = MetricComparisons(embedding_1[unique_idx], maxk=N - 1)

    result = d.return_information_imbalace(embedding_2[unique_idx], **kwargs)
    information_imbalance = np.stack(result, axis=0)
    return information_imbalance, unique_idx


def compute_results(
    embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
):
    assert len(embeddings_1) == len(uniq_labels_1)
    assert len(embeddings_2) == len(uniq_labels_2)
    information_imbalances = []
    unique_len = []
    ks = []
    labels_1 = []
    labels_2 = []
    for label_1, embedding_1 in zip(uniq_labels_1, embeddings_1):
        for label_2, embedding_2 in zip(uniq_labels_2, embeddings_2):
            for k in uniq_ks:
                print(
                    "Computing information imbalance between {} and {} for k={}".format(
                        label_1, label_2, k
                    )
                )
                if label_1 == label_2:
                    continue
                information_imbalance, unique_idx = compute_information_imbalance(
                    embedding_1, embedding_2, k=k, subset_size=subset_size
                )
                information_imbalances.append(information_imbalance[:, 0])
                unique_len.append(len(unique_idx))
                ks.append(k)
                labels_1.append(label_1)
                labels_2.append(label_2)
    results_df = pd.concat(
        [
            pd.DataFrame(
                information_imbalances, columns=["infor_imb_12", "infor_imb_21"]
            ),
            pd.DataFrame(
                {"uniq": unique_len, "label_1": labels_1, "label_2": labels_2, "k": ks}
            ),
        ],
        axis=1,
    )
    return results_df


def conf_het_1(
    input_latents_fname, uniq_ks, subset_size, output_information_imbalance_fname
):

    data = np.load(input_latents_fname)

    ctf_zscore_embeddings = data["ctf_zscore_embeddings"]
    drgnai_fixed_pose_embeddings = data["drgnai_fixed_pose_embeddings"]

    gt_heterogeneity_angle_embeddings = data["gt_heterogeneity_angle_embeddings"]
    gt_heterogeneity_s1_embeddings = data["gt_heterogeneity_s1_embeddings"]
    gt_heterogeneity_angle_withnoise_embeddings = data[
        "gt_heterogeneity_angle_withnoise_embeddings"
    ]
    gt_heterogeneity_s1_withnoise_embeddings = data[
        "gt_heterogeneity_s1_withnoise_embeddings"
    ]

    cryosparc_3dflex_embeddings = data["cryosparc_3dflex_embeddings"]
    cryosparc_3dva_embeddings = data["cryosparc_3dva_embeddings"]
    cryosparc_3d_embeddings = data["cryosparc_3d_embeddings"]
    cryosparc_3dabinit_embeddings = data["cryosparc_3dabinit_embeddings"]

    recovar_embeddings = data["recovar_embeddings"]
    cryodrgn_embeddings = data["cryodrgn_embeddings"]
    cryodrgn2_embeddings = data["cryodrgn2_embeddings"]
    drgnai_abinit_embeddings = data["drgnai_abinit_embeddings"]
    drgnai_fixed_embeddings = data["drgnai_fixed_embeddings"]
    opusdsd_mu_embeddings = data["opusdsd_mu_embeddings"]

    embeddings_1 = [
        gt_heterogeneity_s1_withnoise_embeddings,
        drgnai_fixed_pose_embeddings,
        ctf_zscore_embeddings,
    ]
    uniq_labels_1 = [
        "ground-truth-heterogeneity-s1-smear",
        "ground-truth-pose-R3x3",
        "ground-truth-ctf",
    ]

    embeddings_2 = [
        recovar_embeddings,
        cryodrgn_embeddings,
        cryodrgn2_embeddings,
        drgnai_abinit_embeddings,
        drgnai_fixed_embeddings,
        opusdsd_mu_embeddings,
        cryosparc_3dflex_embeddings,
        cryosparc_3dva_embeddings,
        cryosparc_3d_embeddings,
        cryosparc_3dabinit_embeddings,
    ]
    uniq_labels_2 = [
        "recovar",
        "cryodrgn",
        "cryodrgn2",
        "drgnai-abinit",
        "drgnai-fixed",
        "opusdsd_mu",
        "cryosparc-3dflex",
        "cryosparc-3dva",
        "cryosparc-3d",
        "cryosparc-3dabinit",
    ]

    results_df = compute_results(
        embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
    )

    results_df.to_csv(output_information_imbalance_fname, index=False)
    return results_df


def conf_het_2(
    input_latents_fname, uniq_ks, subset_size, output_information_imbalance_fname
):

    data = np.load(input_latents_fname)

    ctf_zscore_embeddings = data["ctf_zscore_embeddings"]
    drgnai_fixed_pose_embeddings = data["drgnai_fixed_pose_embeddings"]
    gt_heterogeneity_withnoise_embeddings = data[
        "gt_heterogeneity_withnoise_embeddings"
    ]

    cryosparc_3dflex_embeddings = data["cryosparc_3dflex_embeddings"]
    cryosparc_3dva_embeddings = data["cryosparc_3dva_embeddings"]
    cryosparc_3dabinit_embeddings = data["cryosparc_3dabinit_embeddings"]
    cryosparc_3d_embeddings = data["cryosparc_3d_embeddings"]

    recovar_embeddings = data["recovar_embeddings"]
    cryodrgn_embeddings = data["cryodrgn_embeddings"]
    cryodrgn2_embeddings = data["cryodrgn2_embeddings"]
    drgnai_abinit_embeddings = data["drgnai_abinit_embeddings"]
    drgnai_fixed_embeddings = data["drgnai_fixed_embeddings"]
    opusdsd_mu_embeddings = data["opusdsd_mu_embeddings"]

    embeddings_1 = [
        gt_heterogeneity_withnoise_embeddings,
        drgnai_fixed_pose_embeddings,
        ctf_zscore_embeddings,
    ]
    uniq_labels_1 = [
        "ground-truth-heterogeneity-smear",
        "ground-truth-pose-R3x3",
        "ground-truth-ctf",
    ]

    embeddings_2 = [
        cryosparc_3dabinit_embeddings,
        cryosparc_3d_embeddings,
        recovar_embeddings,
        cryodrgn_embeddings,
        cryodrgn2_embeddings,
        drgnai_abinit_embeddings,
        drgnai_fixed_embeddings,
        opusdsd_mu_embeddings,
        cryosparc_3dflex_embeddings,
        cryosparc_3dva_embeddings,
    ]
    uniq_labels_2 = [
        "cryosparc-3dabinit",
        "cryosparc-3d",
        "recovar",
        "cryodrgn",
        "cryodrgn2",
        "drgnai-abinit",
        "drgnai-fixed",
        "opusdsd_mu",
        "cryosparc-3dflex",
        "cryosparc-3dva",
    ]

    results_df = compute_results(
        embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
    )
    results_df.to_csv(output_information_imbalance_fname, index=False)

    return results_df


def assemble_het(
    input_latents_fname, uniq_ks, subset_size, output_information_imbalance_fname
):

    data = np.load(input_latents_fname)
    ctf_zscore_embeddings = data["ctf_zscore_embeddings"]
    drgnai_fixed_pose_embeddings = data["drgnai_fixed_pose_embeddings"]

    gt_heterogeneity_voxel_withnoinse_embeddings = data[
        "gt_heterogeneity_voxel_withnoinse_embeddings"
    ]
    gt_heterogeneity_ranksize_withnoise_embeddings = data[
        "gt_heterogeneity_ranksize_withnoise_embeddings"
    ]

    cryosparc_3dva_embeddings = data["cryosparc_3dva_embeddings"]
    cryosparc_3dabinit_embeddings = data["cryosparc_3dabinit_embeddings"]
    cryosparc_3d_embeddings = data["cryosparc_3d_embeddings"]

    recovar_embeddings = data["recovar_embeddings"]
    cryodrgn_embeddings = data["cryodrgn_embeddings"]
    cryodrgn2_embeddings = data["cryodrgn2_embeddings"]
    drgnai_abinit_embeddings = data["drgnai_abinit_embeddings"]
    drgnai_fixed_embeddings = data["drgnai_fixed_embeddings"]
    opusdsd_mu_embeddings = data["opusdsd_mu_embeddings"]

    embeddings_1 = [
        gt_heterogeneity_voxel_withnoinse_embeddings,
        gt_heterogeneity_ranksize_withnoise_embeddings,
        drgnai_fixed_pose_embeddings,
        ctf_zscore_embeddings,
    ]
    uniq_labels_1 = [
        "ground-truth-heterogeneity-voxel-smear",
        "ground-truth-heterogeneity-ranksize-smear",
        "ground-truth-pose-R3x3",
        "ground-truth-ctf",
    ]

    embeddings_2 = [
        cryosparc_3dabinit_embeddings,
        cryosparc_3d_embeddings,
        recovar_embeddings,
        cryodrgn_embeddings,
        cryodrgn2_embeddings,
        drgnai_abinit_embeddings,
        drgnai_fixed_embeddings,
        opusdsd_mu_embeddings,
        cryosparc_3dva_embeddings,
    ]
    uniq_labels_2 = [
        "cryosparc-3dabinit",
        "cryosparc-3d",
        "recovar",
        "cryodrgn",
        "cryodrgn2",
        "drgnai-abinit",
        "drgnai-fixed",
        "opusdsd_mu",
        "cryosparc-3dva",
    ]

    results_df = compute_results(
        embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
    )
    results_df.to_csv(output_information_imbalance_fname, index=False)

    return results_df


def mix_het(
    input_latents_fname, uniq_ks, subset_size, output_information_imbalance_fname
):

    data = np.load(input_latents_fname)

    ctf_zscore_embeddings = data["ctf_zscore_embeddings"]
    drgnai_fixed_pose_embeddings = data["drgnai_fixed_pose_embeddings"]

    gt_heterogeneity_sizestep_withnoise_embeddings = data[
        "gt_heterogeneity_sizestep_withnoise_embeddings"
    ]
    gt_heterogeneity_onehot_withnoise_embeddings = data[
        "gt_heterogeneity_onehot_withnoise_embeddings"
    ]

    cryosparc_3dva_embeddings = data["cryosparc_3dva_embeddings"]
    cryosparc_3dabinit_embeddings = data["cryosparc_3dabinit_embeddings"]
    cryosparc_3d_embeddings = data["cryosparc_3d_embeddings"]

    recovar_embeddings = data["recovar_embeddings"]
    cryodrgn_embeddings = data["cryodrgn_embeddings"]
    cryodrgn2_embeddings = data["cryodrgn2_embeddings"]
    drgnai_abinit_embeddings = data["drgnai_abinit_embeddings"]
    drgnai_fixed_embeddings = data["drgnai_fixed_embeddings"]
    opusdsd_mu_embeddings = data["opusdsd_mu_embeddings"]

    embeddings_1 = [
        gt_heterogeneity_sizestep_withnoise_embeddings,
        gt_heterogeneity_onehot_withnoise_embeddings,
        drgnai_fixed_pose_embeddings,
        ctf_zscore_embeddings,
    ]
    uniq_labels_1 = [
        "ground-truth-heterogeneity-sizestep-smear",
        "ground-truth-heterogeneity-onehot-smear",
        "ground-truth-pose-R3x3",
        "ground-truth-ctf",
    ]

    embeddings_2 = [
        recovar_embeddings,
        cryodrgn_embeddings,
        cryodrgn2_embeddings,
        drgnai_abinit_embeddings,
        drgnai_fixed_embeddings,
        opusdsd_mu_embeddings,
        cryosparc_3dva_embeddings,
        cryosparc_3dabinit_embeddings,
        cryosparc_3d_embeddings,
    ]
    uniq_labels_2 = [
        "recovar",
        "cryodrgn",
        "cryodrgn2",
        "drgnai-abinit",
        "drgnai-fixed",
        "opusdsd_mu",
        "cryosparc-3dva",
        "cryosparc-3dabinit",
        "cryosparc-3d",
    ]

    results_df = compute_results(
        embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
    )
    results_df.to_csv(output_information_imbalance_fname, index=False)

    return results_df


def md(input_latents_fname, uniq_ks, subset_size, output_information_imbalance_fname):

    data = np.load(input_latents_fname)

    ctf_zscore_embeddings = data["ctf_zscore_embeddings"]
    gt_pose_embeddings = data["gt_pose_embeddings"]

    cryosparc_3dva_embeddings = data["cryosparc_3dva_embeddings"]
    cryosparc_3dflex_embeddings = data["cryosparc_3dflex_embeddings"]

    gt_heterogeneity_withnoise_embeddings = data[
        "gt_heterogeneity_withnoise_embeddings"
    ]
    recovar_embeddings = data["recovar_embeddings"]
    cryodrgn_embeddings = data["cryodrgn_embeddings"]
    cryodrgn2_embeddings = data["cryodrgn2_embeddings"]
    drgnai_abinit_embeddings = data["drgnai_abinit_embeddings"]
    drgnai_fixed_embeddings = data["drgnai_fixed_embeddings"]
    opusdsd_mu_embeddings = data["opusdsd_mu_embeddings"]

    embeddings_1 = [
        gt_heterogeneity_withnoise_embeddings,
        gt_pose_embeddings,
        ctf_zscore_embeddings,
    ]
    uniq_labels_1 = [
        "ground-truth-heterogeneity-smear",
        "ground-truth-pose-R3x3",
        "ground-truth-ctf",
    ]

    embeddings_2 = [
        recovar_embeddings,
        cryodrgn_embeddings,
        cryodrgn2_embeddings,
        drgnai_abinit_embeddings,
        drgnai_fixed_embeddings,
        cryosparc_3dva_embeddings,
        cryosparc_3dflex_embeddings,
        opusdsd_mu_embeddings,
    ]
    uniq_labels_2 = [
        "recovar",
        "cryodrgn",
        "cryodrgn2",
        "drgnai-abinit",
        "drgnai-fixed",
        "cryosparc-3dva",
        "cryosparc-3dflex",
        "opusdsd_mu",
    ]

    results_df = compute_results(
        embeddings_1, embeddings_2, uniq_labels_1, uniq_labels_2, uniq_ks, subset_size
    )
    results_df.to_csv(output_information_imbalance_fname, index=False)

    return results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="conf_het_1")
    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument(
        "--input_latents_fname",
        type=str,
        default="conf-het-1/snr001/wrangled_latents.npz",
    )
    parser.add_argument(
        "--output_information_imbalance_fname",
        type=str,
        default="conf-het-1/snr001/information_imbalance.csv",
    )

    def parse_list(string):
        return [int(item) for item in string.split(",")]

    parser.add_argument("--uniq_ks", type=parse_list, default=[1, 3])
    args = parser.parse_args()
    print(args)

    if args.method == "conf_het_1":
        run_dataset = conf_het_1
    elif args.method == "conf_het_2":
        run_dataset = conf_het_2
    elif args.method == "assemble_het":
        run_dataset = assemble_het
    elif args.method == "mix_het":
        run_dataset = mix_het
    elif args.method == "md":
        run_dataset = md
    else:
        assert False, "chose a dataset to run"

    results_df = run_dataset(
        args.input_latents_fname,
        args.uniq_ks,
        args.subset_size,
        args.output_information_imbalance_fname,
    )
