# How does this work?
# 1. For each dataset and variable, iterate over the five folds of the experiments and build a combined list of
# training and test ids for group 1 and group 2
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress


root_dir = "/Users/katecevora/Documents/PhD/data"
results_dir = "/Users/katecevora/Documents/PhD/results"

lblu = "#add9f4"
lred = "#f36860"
lgrn = "#94c267"
lgry = "#f3efec"
lorg = "#e8a32c"

organ_names = ["left kidney", "right kidney", "liver", "pancreas"]
colours = [lred, lblu, lgrn, lorg]

def getTrainingAndTestSet(dataset, variable):
    """
    Iterates over the five folds of each experiment and builds a list of training IDs and test set IDs for Group 1 and
    Group 2
    :param dataset: TS
    :param variable:
    :return:
    """
    # For each experiment, iterate over the folds and recombine training IDs
    combo_train_ids = []
    g1_train_ids = []
    g2_train_ids = []
    test_ids = []

    for fold in range(0, 5):
        if variable == "Age":
            f = open(os.path.join(root_dir, "training_splits", dataset, "fold_{}_age.pkl".format(fold)), 'rb')
            [set_1_ids, set_2_ids, set_3_ids] = pkl.load(f)
            f.close()
        elif variable == "Sex":
            f = open(os.path.join(root_dir, "training_splits", dataset, "fold_{}.pkl".format(fold)), 'rb')
            [set_1_ids, set_2_ids, set_3_ids] = pkl.load(f)
            f.close()

        combo_train_ids += list(set_1_ids["train"])
        g1_train_ids += list(set_2_ids["train"])
        g2_train_ids += list(set_3_ids["train"])
        test_ids += list(set_2_ids["test"])

    combo_train_ids = np.unique(np.array(combo_train_ids))
    g1_train_ids = np.unique(np.array(g1_train_ids))
    g2_train_ids = np.unique(np.array(g2_train_ids))
    test_ids = np.array(test_ids)

    # save
    f = open(os.path.join(root_dir, "training_splits", dataset, "combined_{}.pkl".format(variable)), "wb")
    pkl.dump([g1_train_ids, g2_train_ids, test_ids, combo_train_ids], f)
    f.close()


def checkVolumes():
    """
    Open the volume measurements (for the full dataset) for TotalSegmentator and AMOS and look at the distributions
    :return:
    """
    # open one of the volume pickle files
    f = open(os.path.join(root_dir, "TotalSegmentator", "volumes.pkl"), "rb")
    [subjects, volumes1] = pkl.load(f)
    f.close()

    f = open(os.path.join(root_dir, "AMOS_3D", "volumes.pkl"), "rb")
    [subjects, volumes2] = pkl.load(f)
    f.close()

    for organ in range(0, 4):
        # calculate mean and standard deviation

        u1 = np.mean((volumes1[:, organ] / 1000))
        std1 = np.std((volumes1[:, organ] / 1000))
        u2 = np.mean((volumes2[:, organ]  / 1000))
        std2 = np.std((volumes2[:, organ] / 1000))

        # calculate the bins
        # First get the minimum
        min1 = np.nanmin((volumes1[:, organ] / 1000))
        min2 = np.nanmin((volumes2[:, organ] / 1000))
        min = np.nanmin((min1, min2))

        # Then the maximum
        max1 = np.nanmax((volumes1[:, organ] / 1000))
        max2 = np.nanmax((volumes2[:, organ] / 1000))
        max = np.nanmax((max1, max2))

        # Now calculate the bin boundaries
        bins = np.linspace(min, max, 20)

        plt.clf()
        plt.hist(volumes1[:, organ] / 1000,
                 label="TS u={0:.0f}, std={1:.0f}".format(u1, std1),
                 density=True,
                 alpha=0.6,
                 color=lblu,
                 bins=bins)
        plt.hist(volumes2[:, organ] / 1000,
                 label="AMOS u={0:.0f}, std={1:.0f}".format(u2, std2),
                 density=True,
                 alpha=0.6,
                 color=lred,
                 bins=bins)
        plt.legend()
        plt.title(organ_names[organ])
        plt.xlabel("milliliters")
        plt.show()


def analyseVolumes(dataset, variable):
    """
    For each dataset and each variable, use the training and test IDs per group from getTrainAndTestIDs and volumes
    from the whole dataset to build list of volumes/ids for group 1 and group 2 and store
    :param dataset:
    :param variable:
    :return:
    """
    # open the volumes
    if dataset == "TS":
        f = open(os.path.join(root_dir, "TotalSegmentator", "volumes.pkl"), "rb")
        [subjects, volumes] = pkl.load(f)
        f.close()
    elif dataset == "AMOS":
        f = open(os.path.join(root_dir, "AMOS_3D", "volumes.pkl"), "rb")
        [subjects, volumes] = pkl.load(f)
        f.close()

    f = open(os.path.join(root_dir, "training_splits", dataset, "combined_{}.pkl".format(variable)), "rb")
    [g1_train_ids, g2_train_ids, test_ids, combo_ids] = pkl.load(f)
    f.close()

    if g1_train_ids.shape[0] != g2_train_ids.shape[0]:
        raise Exception

    #if combo_ids.shape[0] != g2_train_ids.shape[0]:
    #    raise Exception

    # iterate over the training ids and store the volumes
    volumes_combo = []
    volumes_g1 = []
    volumes_g2 = []

    for i in range(g1_train_ids.shape[0]):
        volumes_g1.append(volumes[subjects == g1_train_ids[i]])
        volumes_g2.append(volumes[subjects == g2_train_ids[i]])
        volumes_combo.append(volumes[subjects == combo_ids[i]])

    volumes_g1 = np.array(volumes_g1).squeeze()
    volumes_g2 = np.array(volumes_g2).squeeze()
    volumes_combo = np.array(volumes_combo).squeeze()

    if volumes_g1.shape[0] != g1_train_ids.shape[0]:
        raise Exception

    if volumes_g2.shape[0] != g2_train_ids.shape[0]:
        raise Exception

    #if volumes_combo.shape[0] != g2_train_ids.shape[0]:
    #    raise Exception

    # save
    results_dict = {"g1_train_ids": g1_train_ids,
                    "g2_train_ids": g2_train_ids,
                    "combo_ids": combo_ids,
                    "g1_volumes": volumes_g1,
                    "g2_volumes": volumes_g2,
                    "combo_volumes": volumes_combo}

    f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "wb")
    pkl.dump(results_dict, f)
    f.close()


def plotVolStdAgainstDice():
    """
    For each organ, plot the volume standard deviation for the training set against the average dice score for the test
    set.
    We assume that the larger the standard deviation for the training set (implying greater diversity) the better the
    score on the test set.
    :return:
    """
    # create lists to store all results
    volume_std = []
    dice_mean = []

    for dataset in ["TS", "AMOS"]:
        for variable in ["Age", "Sex"]:

            if dataset == "TS":
                input_map = [1, 2, 3, 4]
                ds_name = "TotalSegmentator"
            elif dataset == "AMOS":
                input_map = [2, 3, 6, 10]
                ds_name = "AMOS_3D"

            # open the volumes
            f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "rb")
            results_dict = pkl.load(f)
            f.close()

            volumes_g1 = results_dict["g1_volumes"]
            volumes_g2 = results_dict["g2_volumes"]

            # open the dice scores
            f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_1.pkl".format(variable)), 'rb')
            results_ex2 = pkl.load(f)
            f.close()

            dice_ex2 = results_ex2["dice"].reshape((-1, np.array(results_ex2["dice"]).shape[-1]))[:, input_map]

            f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_2.pkl".format(variable)), 'rb')
            results_ex3 = pkl.load(f)
            f.close()

            dice_ex3 = results_ex3["dice"].reshape((-1, np.array(results_ex3["dice"]).shape[-1]))[:, input_map]

            # calculates means for Dice and mean and std for volume
            dice_ex2_mean = np.nanmean(dice_ex2, axis=0)
            dice_ex3_mean = np.nanmean(dice_ex3, axis=0)

            volumes_g1_mean = np.mean(volumes_g1, axis=0) / 1000
            volumes_g2_mean = np.mean(volumes_g2, axis=0) / 1000

            volumes_g1_std = np.std(volumes_g1, axis=0) / 1000
            volumes_g2_std = np.std(volumes_g2, axis=0) / 1000

            volume_std.append(volumes_g1_std)
            dice_mean.append(dice_ex2_mean)

            volume_std.append(volumes_g2_std)
            dice_mean.append(dice_ex3_mean)

    # process data for cross dataset experiments
    # open the dice scores for AMOS
    f = open(os.path.join(root_dir, "AMOS_3D", "inference", "results_combined_cross.pkl"), 'rb')
    results_cross = pkl.load(f)
    f.close()

    amos_dice_cross = results_cross["dice"].reshape((-1, np.array(results_cross["dice"]).shape[-1]))[:, 1:]
    amos_dice_mean = np.nanmean(amos_dice_cross, axis=0)

    # open the dice scores for TS
    f = open(os.path.join(root_dir, "TotalSegmentator", "inference", "results_combined_cross.pkl"), 'rb')
    results_cross = pkl.load(f)
    f.close()

    ts_dice_cross = results_cross["dice"].reshape((-1, np.array(results_cross["dice"]).shape[-1]))[:, 1:]
    ts_dice_mean = np.nanmean(ts_dice_cross, axis=0)

    # open the corresponding training volumes
    f = open(os.path.join(results_dir, "ids_and_volumes_AMOS_age.pkl"), "rb")
    results_dict = pkl.load(f)
    f.close()

    volumes_train_amos = results_dict["combo_volumes"]
    volumes_train_amos_std = np.nanstd(volumes_train_amos, axis=0) / 1000

    f = open(os.path.join(results_dir, "ids_and_volumes_TS_age.pkl"), "rb")
    results_dict = pkl.load(f)
    f.close()

    volumes_train_ts = results_dict["combo_volumes"]
    volumes_train_ts_std = np.nanstd(volumes_train_ts, axis=0) / 1000

    # append
    volume_std.append(volumes_train_amos_std)
    dice_mean.append(ts_dice_mean)

    volume_std.append(volumes_train_ts_std)
    dice_mean.append(amos_dice_mean)

    # convert to np arrays
    volume_std = np.array(volume_std)
    dice_mean = np.array(dice_mean)

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    axes = [ax1, ax2, ax3, ax4]

    for organ in range(0, 4):
        # correlation coefficient
        correlation_matrix = np.corrcoef(volume_std[:, organ], dice_mean[:, organ])
        correlation_coefficient = correlation_matrix[0, 1]

        # line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(volume_std[:, organ], dice_mean[:, organ])
        line = slope * volume_std[:, organ] + intercept
        axes[organ].plot(volume_std[:, organ], line, color=colours[organ], label="corr. {0:.2f}".format(correlation_coefficient))

        axes[organ].scatter(volume_std[:, organ], dice_mean[:, organ], label=organ_names[organ], color=colours[organ])
        axes[organ].legend(loc='upper right')
        axes[organ].set_ylabel("Average Dice (Test Set)")
        axes[organ].set_xlabel("Volume Standard Deviation (ml)")

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()
    plt.show()


def printVolStdGapAgainstDiceGap():
    """
    For each organ, print the gap in volume standard deviation between the two groups against the performance gap.
    :return:
    """
    # create lists to store all results
    volume_std = []
    dice_mean = []

    for variable in ["Sex", "Age"]:
        if variable == "Sex":
            group1 = "Female"
            group2 = "Male"
        elif variable == "Age":
            group1 = "U50"
            group2 = "O70"

        print(r"\textbf{Dataset} & \textbf{Organ} & \textbf{$g_1$=" + group1 + r"} & \textbf{$g_2$=" + group2 + r"}  & \textbf{$g_1$-$g_2$} \\")

        for dataset in ["TS", "AMOS"]:
            if dataset == "TS":
                input_map = [1, 2, 3, 4]
                ds_name = "TotalSegmentator"
            elif dataset == "AMOS":
                input_map = [2, 3, 6, 10]
                ds_name = "AMOS_3D"

            # open the volumes
            f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "rb")
            results_dict = pkl.load(f)
            f.close()

            volumes_g1 = results_dict["g1_volumes"]
            volumes_g2 = results_dict["g2_volumes"]

            volumes_g1_mean = np.mean(volumes_g1, axis=0) / 1000
            volumes_g2_mean = np.mean(volumes_g2, axis=0) / 1000

            volumes_g1_std = np.std(volumes_g1, axis=0) / 1000
            volumes_g2_std = np.std(volumes_g2, axis=0) / 1000

            # iterate over organs and print in latex format
            for o in range(0, 4):
                print(" & {0} & {1:.0f} & {2:.0f} & {3:.0f}".format(organ_names[o],
                                                                    volumes_g1_std[o],
                                                                    volumes_g2_std[o],
                                                                    volumes_g1_std[o] - volumes_g2_std[o]) + r"\\")

    # open the corresponding training volumes
    f = open(os.path.join(results_dir, "ids_and_volumes_AMOS_age.pkl"), "rb")
    results_dict = pkl.load(f)
    f.close()

    volumes_train_amos = results_dict["combo_volumes"]
    volumes_train_amos_std = np.nanstd(volumes_train_amos, axis=0) / 1000

    f = open(os.path.join(results_dir, "ids_and_volumes_TS_age.pkl"), "rb")
    results_dict = pkl.load(f)
    f.close()

    volumes_train_ts = results_dict["combo_volumes"]
    volumes_train_ts_std = np.nanstd(volumes_train_ts, axis=0) / 1000

    # append
    volume_std.append(volumes_train_amos_std)
    #dice_mean.append(ts_dice_mean)

    volume_std.append(volumes_train_ts_std)
    #dice_mean.append(amos_dice_mean)

    # convert to np arrays
    volume_std = np.array(volume_std)
    #dice_mean = np.array(dice_mean)

    # iterate over organs and print in latex format
    print(r"\textbf{Dataset} & \textbf{Organ} & \textbf{$g_1$=AMOS} & \textbf{$g_2$=TS}  & \textbf{$g_1$-$g_2$} \\")
    for o in range(0, 4):
        print(" & {0} & {1:.0f} & {2:.0f} & {3:.0f}".format(organ_names[o],
                                                            volumes_train_amos_std[o],
                                                            volumes_train_ts_std[o],
                                                            volumes_train_amos_std[o] - volumes_train_ts_std[o]) + r"\\")



def getLowVolumeIDs(ds, organ_idx):
    # Plot GT dice score against volume against dice score in the cross dataset experiment
    # open the dice scores for TS
    f = open(os.path.join(root_dir, ds, "inference", "results_combined_cross.pkl"), 'rb')
    results_cross = pkl.load(f)
    f.close()

    dice_cross = results_cross["dice"].reshape((-1, np.array(results_cross["dice"]).shape[-1]))[:, organ_idx]
    vol = results_cross["vol_gt"].reshape((-1, np.array(results_cross["vol_gt"]).shape[-1]))[:, organ_idx] / 1000

    f = open(os.path.join(root_dir, ds, "inference", "results_Age_0.pkl"), 'rb')
    results_self = pkl.load(f)
    f.close()

    dice_self = results_self["dice"].reshape((-1, np.array(results_self["dice"]).shape[-1]))[:, organ_idx]

    plt.clf()
    plt.scatter(vol, dice_cross, label="cross")
    plt.scatter(vol, dice_self, label="self", marker='x')
    plt.xlabel("{} Volume (ml)".format(organ_names[organ_idx-1]))
    plt.ylabel("Dice Score")
    plt.legend()
    plt.title(ds)
    plt.show()


def plotVolumes(variable, dataset):
    if dataset == "TS":
        input_map = [1, 2, 3, 4]
        ds_name = "TotalSegmentator"
    elif dataset == "AMOS":
        input_map = [2, 3, 6, 10]
        ds_name = "AMOS_3D"

    if variable == "Age":
        group1 = "Under 50"
        group2 = "Over 70"
        colours = [lgrn, lorg]
        print("Variable is age.")
    elif variable == "Sex":
        group1 = "Female"
        group2 = "Male"
        colours = [lred, lblu]
        print("Variable is sex.")

    # open the volumes
    f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "rb")
    results_dict = pkl.load(f)
    f.close()

    volumes_g1 = results_dict["g1_volumes"]
    volumes_g2 = results_dict["g2_volumes"]

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)
    ax1, ax2, ax3, ax4 = axes.flatten()

    axes = [ax1, ax2, ax3, ax4]

    for i in range(0, 4):
        # calculate mean and standard deviation
        u1 = np.mean((volumes_g1[:, i] / 1000))
        std1 = np.std((volumes_g1[:, i] / 1000))
        u2 = np.mean((volumes_g2[:, i] / 1000))
        std2 = np.std((volumes_g2[:, i] / 1000))

        # calculate the bins
        # First get the minimum
        min1 = np.nanmin((volumes_g1[:, i] / 1000))
        min2 = np.nanmin((volumes_g2[:, i] / 1000))
        min = np.nanmin((min1, min2))

        # Then the maximum
        max1 = np.nanmax((volumes_g1[:, i] / 1000))
        max2 = np.nanmax((volumes_g2[:, i] / 1000))
        max = np.nanmax((max1, max2))

        # Now calculate the bin boundaries
        bins = np.linspace(min, max, 20)

        axes[i].hist(volumes_g1[:, i] / 1000,
                 label="{0}, u={1:.0f}, std={2:.0f}".format(group1, u1, std1),
                 density=True,
                 alpha=0.6,
                 color=colours[0],
                 bins=bins)
        axes[i].hist(volumes_g2[:, i] / 1000,
                 label="{0}, u={1:.0f}, std={2:.0f}".format(group2, u2, std2),
                 density=True,
                 alpha=0.6,
                 color=colours[1],
                 bins=bins)
        axes[i].legend()
        axes[i].set_title(organ_names[i])
        axes[i].set_xlabel("milliliters")

    plt.tight_layout()
    fig.suptitle("{}".format(dataset))
    plt.savefig(os.path.join("/Users/katecevora/Documents/PhD/", "results", "datasets", "{}_{}_volume_histogram.png".format(dataset, variable)))


def plotDiversityAgainstDice(variable):
    """
    For each organ, plot the volume standard deviation for the training set against the average dice score for the test
    set.
    We assume that the larger the standard deviation for the training set (implying greater diversity) the better the
    score on the test set.
    :return:
    """
    # create lists to store all results
    volume_std = []
    dice_mean_g1 = []
    dice_mean_g2 = []

    for dataset in ["TS", "AMOS"]:

        if dataset == "TS":
            input_map = [1, 2, 3, 4]
            ds_name = "TotalSegmentator"
            if variable == "Sex":
                label_g1 = "Female"
                label_g2 = "Male"
            elif variable == "Age":
                label_g1 = "Under 50"
                label_g2 = "Over 70"
                age_1 = 50
                age_2 = 70
        elif dataset == "AMOS":
            input_map = [2, 3, 6, 10]
            ds_name = "AMOS_3D"
            if variable == "Sex":
                label_g1 = "Female"
                label_g2 = "Male"
            elif variable == "Age":
                label_g1 = "Under 50"
                label_g2 = "Over 70"
                age_1 = 40
                age_2 = 65


        # open the volumes
        f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "rb")
        results_dict = pkl.load(f)
        f.close()

        volumes_g1 = results_dict["g1_volumes"]
        volumes_g2 = results_dict["g2_volumes"]

        # open the dice scores
        f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_1.pkl".format(variable)), 'rb')
        results_ex2 = pkl.load(f)
        f.close()

        sex = results_ex2["sex"].flatten()
        age = results_ex2["age"].flatten()

        dice_ex2 = results_ex2["dice"].reshape((-1, np.array(results_ex2["dice"]).shape[-1]))[:, input_map]

        f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_2.pkl".format(variable)), 'rb')
        results_ex3 = pkl.load(f)
        f.close()

        dice_ex3 = results_ex3["dice"].reshape((-1, np.array(results_ex3["dice"]).shape[-1]))[:, input_map]

        # split the dice scores by group
        # sort by characteristic of interest
        if variable == "Age":
            dice_g1_ex2 = dice_ex2[age <= age_1, :]
            dice_g2_ex2 = dice_ex2[age >= age_2, :]
            dice_g1_ex3 = dice_ex3[age <= age_1, :]
            dice_g2_ex3 = dice_ex3[age >= age_2, :]
        elif variable == "Sex":
            dice_g1_ex2 = dice_ex2[sex == 1, :]
            dice_g2_ex2 = dice_ex2[sex == 0, :]
            dice_g1_ex3 = dice_ex3[sex == 1, :]
            dice_g2_ex3 = dice_ex3[sex == 0, :]

        # calculates means for Dice and mean and std for volume
        dice_g1_ex2_mean = np.nanmean(dice_g1_ex2, axis=0)
        dice_g2_ex2_mean = np.nanmean(dice_g2_ex2, axis=0)

        dice_g1_ex3_mean = np.nanmean(dice_g1_ex3, axis=0)
        dice_g2_ex3_mean = np.nanmean(dice_g2_ex3, axis=0)

        volumes_g1_mean = np.mean(volumes_g1, axis=0) / 1000
        volumes_g2_mean = np.mean(volumes_g2, axis=0) / 1000

        volumes_g1_std = np.std(volumes_g1, axis=0) / 1000
        volumes_g2_std = np.std(volumes_g2, axis=0) / 1000

        # Experiment 1 (train with g1)
        volume_std.append(volumes_g1_std)
        dice_mean_g1.append(dice_g1_ex2_mean)
        dice_mean_g2.append(dice_g2_ex2_mean)

        # Experiment 2 (train with g2)
        volume_std.append(volumes_g2_std)
        dice_mean_g1.append(dice_g1_ex3_mean)
        dice_mean_g2.append(dice_g2_ex3_mean)

    volume_std = np.array(volume_std)
    dice_mean_g1 = np.array(dice_mean_g1)
    dice_mean_g2 = np.array(dice_mean_g2)
    # plot
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    axes = [ax1, ax2, ax3, ax4]

    for o in range(0, 4):
        axes[o].scatter(volume_std[:, o], dice_mean_g1[:, o], label=label_g1, color=colours[0])
        axes[o].scatter(volume_std[:, o], dice_mean_g2[:, o], label=label_g2, color=colours[1])
        axes[o].legend(loc='upper right')
        axes[o].set_ylabel("Average Dice (Test Set)")
        axes[o].set_xlabel("Volume Standard Deviation (ml)")
        axes[o].set_title(organ_names[o])

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()
    plt.show()

def plotAnotherThing():
    """
    For each organ, plot the volume standard deviation for the training set against the average dice score for the test
    set.
    We assume that the larger the standard deviation for the training set (implying greater diversity) the better the
    score on the test set.
    :return:
    """
    # plot
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
    #ax1, ax2, ax3, ax4 = axes.flatten()
    #axes = [ax1, ax2, ax3, ax4]

    for dataset, k in zip(["TS", "AMOS"], [0,1]):
        for variable in ["Sex", "Age"]:
            volume_std = []
            dice_mean_g1 = []
            dice_mean_g2 = []

            if dataset == "TS":
                input_map = [1, 2, 3, 4]
                ds_name = "TotalSegmentator"
                if variable == "Sex":
                    label_g1 = "Female"
                    label_g2 = "Male"
                    c = [lred, lblu]
                elif variable == "Age":
                    label_g1 = "Under 50"
                    label_g2 = "Over 70"
                    age_1 = 50
                    age_2 = 70
                    c = [lgrn, lorg]
            elif dataset == "AMOS":
                input_map = [2, 3, 6, 10]
                ds_name = "AMOS_3D"
                if variable == "Sex":
                    label_g1 = "Female"
                    label_g2 = "Male"
                    c = [lred, lblu]
                elif variable == "Age":
                    label_g1 = "Under 50"
                    label_g2 = "Over 70"
                    age_1 = 40
                    age_2 = 65
                    c = [lgrn, lorg]

            # open the volumes
            f = open(os.path.join(results_dir, "ids_and_volumes_{}_{}.pkl".format(dataset, variable)), "rb")
            results_dict = pkl.load(f)
            f.close()

            volumes_g1 = results_dict["g1_volumes"]
            volumes_g2 = results_dict["g2_volumes"]

            # open the dice scores
            f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_1.pkl".format(variable)), 'rb')
            results_ex2 = pkl.load(f)
            f.close()

            sex = results_ex2["sex"].flatten()
            age = results_ex2["age"].flatten()

            dice_ex2 = results_ex2["dice"].reshape((-1, np.array(results_ex2["dice"]).shape[-1]))[:, input_map]

            f = open(os.path.join(root_dir, ds_name, "inference", "results_{}_2.pkl".format(variable)), 'rb')
            results_ex3 = pkl.load(f)
            f.close()

            dice_ex3 = results_ex3["dice"].reshape((-1, np.array(results_ex3["dice"]).shape[-1]))[:, input_map]

            # split the dice scores by group
            # sort by characteristic of interest
            if variable == "Age":
                dice_g1_ex2 = dice_ex2[age <= age_1, :]
                dice_g2_ex2 = dice_ex2[age >= age_2, :]
                dice_g1_ex3 = dice_ex3[age <= age_1, :]
                dice_g2_ex3 = dice_ex3[age >= age_2, :]
            elif variable == "Sex":
                dice_g1_ex2 = dice_ex2[sex == 1, :]
                dice_g2_ex2 = dice_ex2[sex == 0, :]
                dice_g1_ex3 = dice_ex3[sex == 1, :]
                dice_g2_ex3 = dice_ex3[sex == 0, :]

            # calculates means for Dice and mean and std for volume
            dice_g1_ex2_mean = np.nanmean(dice_g1_ex2, axis=0)
            dice_g2_ex2_mean = np.nanmean(dice_g2_ex2, axis=0)

            dice_g1_ex3_mean = np.nanmean(dice_g1_ex3, axis=0)
            dice_g2_ex3_mean = np.nanmean(dice_g2_ex3, axis=0)

            volumes_g1_mean = np.mean(volumes_g1, axis=0) / 1000
            volumes_g2_mean = np.mean(volumes_g2, axis=0) / 1000

            volumes_g1_std = np.std(volumes_g1, axis=0) / 1000
            volumes_g2_std = np.std(volumes_g2, axis=0) / 1000

            # Experiment 1 (train with g1)
            volume_std.append(volumes_g1_std)
            dice_mean_g1.append(dice_g1_ex2_mean)
            dice_mean_g2.append(dice_g2_ex2_mean)

            # Experiment 2 (train with g2)
            volume_std.append(volumes_g2_std)
            dice_mean_g1.append(dice_g1_ex3_mean)
            dice_mean_g2.append(dice_g2_ex3_mean)

            volume_std = np.array(volume_std)
            dice_mean_g1 = np.array(dice_mean_g1)
            dice_mean_g2 = np.array(dice_mean_g2)

            for o in range(0, 4):
                #if o == 3:
                #    axes[k, o].set_ylim(0.8, 0.9)
                #else:
                #    axes[k, o].set_ylim(0.91, 0.99)
                axes[k, o].scatter(volume_std[:, o], dice_mean_g1[:, o], label=label_g1, color=c[0])
                axes[k, o].plot(volume_std[:, o], dice_mean_g1[:, o], color=c[0])
                axes[k, o].scatter(volume_std[:, o], dice_mean_g2[:, o], label=label_g2, color=c[1])
                axes[k, o].plot(volume_std[:, o], dice_mean_g2[:, o], color=c[1])
                #axes[k, o].set_yticks([])
                #axes[k, o].set_yticklabels([])

                #axes[o].legend(loc='upper right')
                #axes[o].set_ylabel("Average Dice (Test Set)")
                #axes[o].set_xlabel("Volume Standard Deviation (ml)")
                if k == 0:
                    axes[k, o].set_title(organ_names[o])
                if k == 1 and o == 3:
                    axes[k, o].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Adjust space between subplots
    #plt.subplots_adjust(hspace=0.1)  # Adjust horizontal space
    fig.supxlabel("Training Set Volume Standard Deviation (ml)")
    fig.supylabel("Test Set Average Dice")
    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()
    plt.show()


def main():
    #checkVolumes()
    #for dataset in ["TS", "AMOS"]:
    #    for variable in ["Age", "Sex"]:
    #        getTrainingAndTestSet(dataset, variable)
    #        analyseVolumes(dataset, variable)

    #plotVolStdGapAgainstDiceGap()
    #plotVolStdAgainstDice()
    #plotDiversityAgainstDice("Age")
    plotAnotherThing()
    #getLowVolumeIDs("AMOS_3D", 3)
    #plotVolumes("Sex", "TS")
    #plotVolumes("Age", "TS")
    #plotVolumes("Sex", "AMOS")
    #plotVolumes("Age", "AMOS")


if __name__ == "__main__":
    main()