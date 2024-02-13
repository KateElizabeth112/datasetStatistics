import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt


root_dir = "/Users/katecevora/Documents/PhD/data"
results_dir = "/Users/katecevora/Documents/PhD/results"

dataset = "TS"
variable = "Sex"

lblu = "#add9f4"
lred = "#f36860"
lgrn = "#94c267"
lgry = "#f3efec"
lorg = "#e8a32c"

organ_names = ["left kidney", "right kidney", "liver", "pancreas"]
colours = [lred, lblu, lgrn, lorg]

def getTrainingAndTestSet(dataset, variable):
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
    # open one of the volume pickle files
    f = open(os.path.join(root_dir, "TotalSegmentator", "volumes.pkl"), "rb")
    [subjects, volumes1] = pkl.load(f)
    f.close()

    f = open(os.path.join(root_dir, "AMOS_3D", "volumes.pkl"), "rb")
    [subjects, volumes2] = pkl.load(f)
    f.close()

    for organ in range(0, 4):
        # calculate mean and standard deviation
        u1 = np.mean((volumes1[:, organ][volumes1[:, organ] > 0] / 1000))
        std1 = np.std((volumes1[:, organ][volumes1[:, organ] > 0] / 1000))
        u2 = np.mean((volumes2[:, organ][volumes2[:, organ] > 0] / 1000))
        std2 = np.std((volumes2[:, organ][volumes2[:, organ] > 0] / 1000))

        plt.clf()
        plt.hist(volumes1[:, organ][volumes1[:, organ] > 0] / 1000, label="u={0:.0f}, std={1:.0f}".format(u1, std1), density=True)
        plt.hist(volumes2[:, organ][volumes2[:, organ] > 0] / 1000, label="u={0:.0f}, std={1:.0f}".format(u2, std2), density=True)
        plt.legend()
        plt.title(organ_names[organ])
        plt.xlabel("milliliters")
        plt.show()


def analyseVolumes(dataset, variable):
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


def collateResults():
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
        axes[organ].scatter(volume_std[:, organ], dice_mean[:, organ], label=organ_names[organ], color=colours[organ])
        axes[organ].legend(loc='upper right')
        axes[organ].set_ylabel("Average Dice")
        axes[organ].set_xlabel("Volume Standard Deviation (ml)")

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    plt.show()


def main():
    #for dataset in ["TS", "AMOS"]:
    #    for variable in ["Age", "Sex"]:
    #        getTrainingAndTestSet(dataset, variable)
    #        analyseVolumes(dataset, variable)

    collateResults()




if __name__ == "__main__":
    main()