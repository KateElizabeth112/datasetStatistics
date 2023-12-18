# contains functions for printing results in latex format
import numpy as np
from scipy import stats
import os
import copy


def significanceThreshold(p):
    # Test significance and return *, **, or blank
    if p <= 0.01:
        sig = "**"
    elif p <= 0.05:
        sig = "*"
    else:
        sig = ""

    return sig


def printDeltas(g1_ex1, g2_ex1, g1_ex2, g2_ex2, g1_ex3, g2_ex3, organ_dict, label, file_path):
    # Calculate the deltas from the baseline experiment
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    for j in range(1, n_channels):

        organ = organs[j]

        if label == "HD95":
            i = copy.deepcopy(j-1)
        else:
            i = copy.deepcopy(j)

        # Delete NaNs
        g1_ex1_i = g1_ex1[:, i][np.isfinite(g1_ex1[:, i])]
        g2_ex1_i = g2_ex1[:, i][np.isfinite(g2_ex1[:, i])]
        g1_ex2_i = g1_ex2[:, i][np.isfinite(g1_ex2[:, i])]
        g2_ex2_i = g2_ex2[:, i][np.isfinite(g2_ex2[:, i])]
        g1_ex3_i = g1_ex3[:, i][np.isfinite(g1_ex3[:, i])]
        g2_ex3_i = g2_ex3[:, i][np.isfinite(g2_ex3[:, i])]

        # Baseline results
        av_g1_ex1 = np.mean(g1_ex1_i)
        av_g2_ex1 = np.mean(g2_ex1_i)

        # look at difference between group 1 and group 2
        delta_1 = ((av_g1_ex1 - av_g2_ex1) / np.mean((av_g1_ex1, av_g2_ex1))) * 100
        (_, p_1) = stats.ttest_ind(g1_ex1_i, g2_ex1_i, equal_var=False)

        # Experiment 2 (all group 1 training set)
        av_g1_ex2 = np.mean(g1_ex2_i)
        av_g2_ex2 = np.mean(g2_ex2_i)

        delta_g1_ex2 = -((av_g1_ex1 - av_g1_ex2) / np.mean((av_g1_ex1, av_g1_ex2))) * 100
        delta_g2_ex2 = -((av_g2_ex1 - av_g2_ex2) / np.mean((av_g2_ex1, av_g2_ex2))) * 100

        (_, p_g1_ex2) = stats.ttest_ind(g1_ex1_i, g1_ex2_i, equal_var=False)
        (_, p_g2_ex2) = stats.ttest_ind(g2_ex1_i, g2_ex2_i, equal_var=False)

        # Experiment 3 (group 2 training set)
        av_g1_ex3 = np.mean(g1_ex3_i)
        av_g2_ex3 = np.mean(g2_ex3_i)

        delta_g1_ex3 = -((av_g1_ex1 - av_g1_ex3) / np.mean((av_g1_ex1, av_g1_ex3))) * 100
        delta_g2_ex3 = -((av_g2_ex1 - av_g2_ex3) / np.mean((av_g2_ex1, av_g2_ex3))) * 100

        (_, p_g1_ex3) = stats.ttest_ind(g1_ex1_i, g1_ex3_i, equal_var=False)
        (_, p_g2_ex3) = stats.ttest_ind(g1_ex1_i, g2_ex3_i, equal_var=False)

        # Get significance thresholds as *'s
        sig_1 = significanceThreshold(p_1)
        sig_g1_ex2 = significanceThreshold(p_g1_ex2)
        sig_g2_ex2 = significanceThreshold(p_g2_ex2)
        sig_g1_ex3 = significanceThreshold(p_g1_ex3)
        sig_g2_ex3 = significanceThreshold(p_g2_ex3)

        with open(file_path, "a") as myfile:
            myfile.write(organ + " & {0:.2f} {1} & {2:.2f} {3} & {4:.2f} {5} & {6:.2f} {7} & {8:.2f} {9}".format(delta_1,
                                                                                                      sig_1,
                                                                                                      delta_g1_ex2,
                                                                                                      sig_g1_ex2,
                                                                                                      delta_g1_ex3,
                                                                                                      sig_g1_ex3,
                                                                                                      delta_g2_ex2,
                                                                                                      sig_g2_ex2,
                                                                                                      delta_g2_ex3,
                                                                                                      sig_g2_ex3) +
                         r" \\" + "\n")


def printVolumes(vol_g1, vol_g2, organ_dict, file_path):
    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    organs = list(organ_dict.keys())

    for i in range(1, len(organs)):
        organ = organs[i]

        # Calculate averages
        v_av_g1 = np.mean(vol_g1[:, i])
        v_av_g2 = np.mean(vol_g2[:, i])

        # difference in average (mm3)
        v_diff = v_av_g1 - v_av_g2
        v_diff_prop = (v_diff / np.mean((v_av_g1, v_av_g2))) * 100

        # perform t-test
        res = stats.ttest_ind(vol_g1[:, i], vol_g2[:, i], equal_var=False)

        # save difference in mean, difference in mean as a proportion of the average volume, and p-value
        if res[1] < 0.01:
            sig = "**"
        elif res[1] < 0.05:
            sig = "*"
        else:
            sig = ""

        with open(file_path, "a") as myfile:
            myfile.write("{0} & {1:.0f} {2} & {3:.2f} {4} ".format(organ, v_diff, sig, v_diff_prop, sig) + r"\\" + "\n")


def printVolumeErrors(vol_err_g1_ex1, vol_err_g2_ex1, vol_err_g1_ex2, vol_err_g2_ex2, vol_err_g1_ex3, vol_err_g2_ex3, organ_dict, save_path):
    organs = list(organ_dict.keys())

    for i in range(1, len(organs)):
        organ = organs[i]

        # Experiment 1
        g1_ex1_mean = np.mean(vol_err_g1_ex1[:, i])
        g1_ex1_med = np.median(vol_err_g1_ex1[:, i])
        g2_ex1_mean = np.mean(vol_err_g2_ex1[:, i])
        g2_ex1_med = np.median(vol_err_g2_ex1[:, i])

        # Experiment 2
        g1_ex2_mean = np.mean(vol_err_g1_ex2[:, i])
        g1_ex2_med = np.median(vol_err_g1_ex2[:, i])
        g2_ex2_mean = np.mean(vol_err_g2_ex2[:, i])
        g2_ex2_med = np.median(vol_err_g2_ex2[:, i])

        # Experiment 3
        g1_ex3_mean = np.mean(vol_err_g1_ex3[:, i])
        g1_ex3_med = np.median(vol_err_g1_ex3[:, i])
        g2_ex3_mean = np.mean(vol_err_g2_ex3[:, i])
        g2_ex3_med = np.median(vol_err_g2_ex3[:, i])

        print("{0} & {1:.0f} & {2:.0f} & {3:.0f} & {4:.0f} & {5:.0f} & {6:.0f}".format(organ, g1_ex1_mean, g2_ex1_mean,
                                                                                       g1_ex2_mean, g2_ex2_mean,
                                                                                       g1_ex3_mean, g2_ex3_mean) + r"\\" + "\n")


def printCrossDeltas(ds1_self, ds2_self, ds1_cross, ds2_cross, organ_dict, file_path):
    # Calculate the deltas from the baseline experiment for cross-dataset generalisation
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    for i in range(1, n_channels):
        organ = organs[i]

        # Delete NaNs
        ds1_self_i = ds1_self[:, i][np.isfinite(ds1_self[:, i])]
        ds2_self_i = ds2_self[:, i][np.isfinite(ds2_self[:, i])]
        ds1_cross_i = ds1_cross[:, i][np.isfinite(ds1_cross[:, i])]
        ds2_cross_i = ds2_cross[:, i][np.isfinite(ds2_cross[:, i])]

        # Baseline results
        av_ds1_self = np.mean(ds1_self_i)
        av_ds2_self = np.mean(ds2_self_i)

        # look at difference between group 1 and group 2
        delta_1 = ((av_ds1_self - av_ds2_self) / np.mean((av_ds1_self, av_ds2_self))) * 100
        (_, p_1) = stats.ttest_ind(ds1_self_i, ds2_self_i, equal_var=False)

        # Experiment 2 (all group 1 training set)
        av_ds1_cross = np.mean(ds1_cross_i)
        av_ds2_cross = np.mean(ds2_cross_i)

        delta_ds1_cross = -((av_ds1_self - av_ds1_cross) / np.mean((av_ds1_self, av_ds1_cross))) * 100
        delta_ds2_cross = -((av_ds2_self - av_ds2_cross) / np.mean((av_ds2_self, av_ds2_cross))) * 100

        (_, p_ds1_cross) = stats.ttest_ind(ds1_self_i, ds1_cross_i, equal_var=False)
        (_, p_ds2_cross) = stats.ttest_ind(ds2_self_i, ds2_cross_i, equal_var=False)

        # Get significance thresholds as *'s
        sig_1 = significanceThreshold(p_1)
        sig_ds1_cross = significanceThreshold(p_ds1_cross)
        sig_ds2_cross = significanceThreshold(p_ds2_cross)

        with open(file_path, "a") as myfile:
            myfile.write(organ + " & {0:.2f} {1} & {2:.2f} {3} & {4:.2f} {5}".format(delta_1,
                                                                                     sig_1,
                                                                                     delta_ds1_cross,
                                                                                     sig_ds1_cross,
                                                                                     delta_ds2_cross,
                                                                                     sig_ds2_cross) +
                         r" \\" + "\n")