# contains functions for printing results in latex format
import numpy as np
from scipy import stats
import os
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl

lblu = "#add9f4"
lred = "#f36860"
lgrn = "#94c267"
lgry = "#f3efec"


def significanceThreshold(p):
    # Test significance and return *, **, or blank
    if p <= 0.01:
        sig = "**"
    elif p <= 0.05:
        sig = "*"
    else:
        sig = ""

    return sig


def printDeltas(g1_ex1, g2_ex1, g1_ex2, g2_ex2, g1_ex3, g2_ex3, organ_dict, label, file_path, dataset, variable):
    # Calculate the deltas from the baseline experiment
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # create a numpy array to store results
    dice_table = np.zeros((n_channels - 1, 5))
    colour_table = np.zeros((n_channels - 1, 5))

    for j in range(1, n_channels):

        organ = organs[j]

        if (label == "HD95") or (label == "HD"):
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
            if organ == "left kidney":
                myfile.write("{0} & {1} & ".format(dataset, variable) + organ + " & {0:.2f} {1} & {2:.2f} {3} & {4:.2f} {5} & {6:.2f} {7} ".format(
                                                                                                          delta_g1_ex2,
                                                                                                          sig_g1_ex2,
                                                                                                          delta_g1_ex3,
                                                                                                          sig_g1_ex3,
                                                                                                          delta_g2_ex2,
                                                                                                          sig_g2_ex2,
                                                                                                          delta_g2_ex3,
                                                                                                          sig_g2_ex3) +
                                                                                                          r" \\" + "\n")
            else:
                myfile.write("&  & " + organ + " & {0:.2f} {1} & {2:.2f} {3} & {4:.2f} {5} & {6:.2f} {7}".format(
                                                                                                          delta_g1_ex2,
                                                                                                          sig_g1_ex2,
                                                                                                          delta_g1_ex3,
                                                                                                          sig_g1_ex3,
                                                                                                          delta_g2_ex2,
                                                                                                          sig_g2_ex2,
                                                                                                          delta_g2_ex3,
                                                                                                          sig_g2_ex3) +
                                                                                                          r" \\" + "\n")


        # Store values if significant
        if not (sig_1 == ""):
            dice_table[j - 1, 0] = delta_1
            colour_table[j - 1, 0] = delta_1
        else:
            dice_table[j - 1, 0] = delta_1
            colour_table[j - 1, 0] = 0

        if not (sig_g1_ex2 == ""):
            dice_table[j - 1, 1] = delta_g1_ex2
            colour_table[j - 1, 1] = delta_g1_ex2
        else:
            dice_table[j - 1, 1] = delta_g1_ex2
            colour_table[j - 1, 1] = 0

        if not (sig_g1_ex3 == ""):
            dice_table[j - 1, 2] = delta_g1_ex3
            colour_table[j - 1, 2] = delta_g1_ex3
        else:
            dice_table[j - 1, 2] = delta_g1_ex3
            colour_table[j - 1, 2] = 0

        if not (sig_g2_ex2 == ""):
            dice_table[j - 1, 3] = delta_g2_ex2
            colour_table[j - 1, 3] = delta_g2_ex2
        else:
            dice_table[j - 1, 3] = delta_g2_ex2
            colour_table[j - 1, 3] = 0

        if not (sig_g2_ex3 == ""):
            dice_table[j - 1, 4] = delta_g2_ex3
            colour_table[j - 1, 4] = delta_g2_ex3
        else:
            dice_table[j - 1, 4] = delta_g2_ex3
            colour_table[j - 1, 4] = 0


        # Force values for an example
        #dice_table[j - 1, 0] = 0
        #dice_table[j - 1, 1] = 1
        #dice_table[j - 1, 2] = -1
        #dice_table[j - 1, 3] = -1
        #dice_table[j - 1, 4] = 1

    # Now plot a heatmap
    if label == "Dice":
        cmap = mpl.colors.ListedColormap([lred, lgry, lgrn])
        #cmap.set_over('0.25')
        #cmap.set_under('0.75')

        bounds = [-100, -0.0001, 0.00001, 100]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.clf()
        plt.figure()
        plt.imshow(colour_table[:, 1:], cmap=cmap, norm=norm)
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

        # populate the image pixels with values
        #for (j, i), label in np.ndenumerate(dice_table[:, 1:]):
        #    ax.text(i, j, "{0:.2f}".format(label), ha='center', va='center')
        plt.yticks([])
        plt.xticks([])
        plt.axvline(1.5, color='k')
        #plt.show()
        plt.savefig(os.path.join("/Users/katecevora/Documents/PhD/results", "info_graphic", "{}_{}.png".format(dataset, variable)))



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


def printCrossDeltas(ds1_self, ds2_self, ds1_cross, ds2_cross, organ_dict, label, file_path):
    # Calculate the deltas from the baseline experiment for cross-dataset generalisation
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # create a numpy array to store results
    dice_table = np.zeros((n_channels - 1, 3))

    for j in range(1, n_channels):
        organ = organs[j]

        if (label == "HD95") or (label == "HD"):
            i = copy.deepcopy(j-1)
        else:
            i = copy.deepcopy(j)

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

        if not (sig_1 == ""):
            dice_table[j - 1, 0] = delta_1
        else:
            dice_table[j - 1, 0] = 0

        if not (sig_ds1_cross == ""):
            dice_table[j - 1, 1] = delta_ds1_cross
        else:
            dice_table[j - 1, 1] = 0

        if not (sig_ds2_cross == ""):
            dice_table[j - 1, 2] = delta_ds2_cross
        else:
            dice_table[j - 1, 2] = 0



    if False:
        # Store values if significant
        print(sig_1, sig_ds2_cross, sig_ds1_cross)
        cmap = mpl.colors.ListedColormap([lblu, lgry, lred])
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        bounds = [-100, -0.0001, 0.00001, 100]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.clf()
        plt.figure()
        plt.imshow(dice_table, cmap=cmap, norm=norm)
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, 3, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)
        plt.yticks([])
        plt.xticks([])
        plt.show()


def printDeltas2(g1_ex2, g2_ex2, g1_ex3, g2_ex3, organ_dict, label, file_path, dataset, variable):
    organs = list(organ_dict.keys())
    n_channels = len(organs)

    # Remove old file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    for j in range(1, n_channels):

        organ = organs[j]

        if (label == "HD95") or (label == "HD"):
            i = copy.deepcopy(j-1)
        else:
            i = copy.deepcopy(j)

        # Delete NaNs
        g1_ex2_i = g1_ex2[:, i][np.isfinite(g1_ex2[:, i])]
        g2_ex2_i = g2_ex2[:, i][np.isfinite(g2_ex2[:, i])]
        g1_ex3_i = g1_ex3[:, i][np.isfinite(g1_ex3[:, i])]
        g2_ex3_i = g2_ex3[:, i][np.isfinite(g2_ex3[:, i])]

        av_g1_ex2 = np.mean(g1_ex2_i)
        av_g2_ex2 = np.mean(g2_ex2_i)

        av_g1_ex3 = np.mean(g1_ex3_i)
        av_g2_ex3 = np.mean(g2_ex3_i)

        # Group 1
        # Difference between Tr=G1 (ex2), Ts=G1 and Tr=G2 (ex3) and Ts=G1
        delta_g1 = -(av_g1_ex2 - av_g1_ex3) * 100 / np.mean((av_g1_ex2, av_g1_ex3))
        (_, p_g1) = stats.ttest_ind(g1_ex2_i, g1_ex3_i, equal_var=False)

        # Group 2
        delta_g2 = -(av_g2_ex3 - av_g2_ex2) * 100 / np.mean((av_g2_ex3, av_g2_ex2))
        (_, p_g2) = stats.ttest_ind(g2_ex3_i, g2_ex2_i, equal_var=False)

        sig_g1 = significanceThreshold(p_g1)
        sig_g2 = significanceThreshold(p_g2)

        with open(file_path, "a") as myfile:
            if organ == "left kidney":
                myfile.write("{0} & {1} & ".format(dataset, variable) + organ + "& {0:.2f} {1} & {2:.2f} {3}".format(delta_g1,
                                                                      sig_g1,
                                                                      delta_g2,
                                                                      sig_g2) +
                                                                      r" \\" + "\n")
            else:
                myfile.write("& & " + organ + " & {0:.2f} {1} & {2:.2f} {3}".format(delta_g1,
                                                                                      sig_g1,
                                                                                      delta_g2,
                                                                                      sig_g2) +
                                                                                      r" \\" + "\n")

