import numpy as np
import pickle as pkl
import os
import numpy as np
from plotResults import plotDistribution, plotVolumeDistribution
from printResults import printDeltas2, printVolumes, printVolumeErrors

root_dir = "/Users/katecevora/Documents/PhD/data"
output_dir = "/Users/katecevora/Documents/PhD/results"

#dataset = "TotalSegmentator"
dataset = "AMOS_3D"

variable = "Age"
#variable = "Sex"

organ_dict = {"background": 0,
              "right kidney": 1,
              "left kidney": 2,
              "liver": 3,
              "pancreas": 4}

if dataset == "TotalSegmentator":

    if variable == "Sex":
        label_g1 = "Male"
        label_g2 = "Female"
    elif variable == "Age":
        label_g1 = "Under 50"
        label_g2 = "Over 70"
        age_1 = 50
        age_2 = 70

    input_map = [0, 1, 2, 3, 4]
    input_map_hd = [0, 1, 2, 3]

elif dataset == "AMOS_3D":

    if variable == "Sex":
        label_g1 = "Male"
        label_g2 = "Female"
    elif variable == "Age":
        label_g1 = "Under 50"
        label_g2 = "Over 70"
        age_1 = 40
        age_2 = 65

    input_map = [0, 2, 3, 6, 10]
    input_map_hd = [1, 2, 5, 9]

else:
    print("Dataset not recognised")


def createFolders(output_dir, dataset, variable):
    if not os.path.exists(os.path.join(output_dir, dataset)):
        os.mkdir(os.path.join(output_dir, dataset))

    if not os.path.exists(os.path.join(output_dir, dataset, variable)):
        os.mkdir(os.path.join(output_dir, dataset, variable))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "plots")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "plots"))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "text")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "text"))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "plots", "dice")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "plots", "dice"))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "plots", "hd")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "plots", "hd"))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "plots", "hd95")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "plots", "hd95"))

    if not os.path.exists(os.path.join(output_dir, dataset, variable, "plots", "volume")):
        os.mkdir(os.path.join(output_dir, dataset, variable, "plots", "volume"))


def getVolumeErrors(vol_gt, vol_pred_ex1, vol_pred_ex2, vol_pred_ex3):
    # calculate the percentage errors in volume

    vol_err_ex1 = vol_pred_ex1 - vol_gt
    vol_err_ex2 = vol_pred_ex2 - vol_gt
    vol_err_ex3 = vol_pred_ex3 - vol_gt

    return vol_err_ex1, vol_err_ex2, vol_err_ex3


def main():
    #createFolders(output_dir, dataset, variable)

    f = open(os.path.join(root_dir, dataset, "inference", "results_{}_0.pkl".format(variable)), 'rb')
    results_ex1 = pkl.load(f)
    f.close()

    case_id = results_ex1["case_id"].flatten()
    sex = results_ex1["sex"].flatten()
    age = results_ex1["age"].flatten()

    dice_ex1 = results_ex1["dice"].reshape((-1, np.array(results_ex1["dice"]).shape[-1]))[:, input_map]
    hd95_ex1 = np.array(results_ex1["hd95"]).reshape((-1, np.array(results_ex1["hd95"]).shape[-1]))[:, input_map_hd]
    hd_ex1 = np.array(results_ex1["hd"]).reshape((-1, np.array(results_ex1["hd"]).shape[-1]))[:, input_map_hd]
    vol_pred_ex1 = results_ex1["vol_pred"].reshape((-1, np.array(results_ex1["vol_pred"]).shape[-1]))[:, input_map]
    vol_gt = results_ex1["vol_gt"].reshape((-1, np.array(results_ex1["vol_gt"]).shape[-1]))[:, input_map]

    f = open(os.path.join(root_dir, dataset, "inference", "results_{}_1.pkl".format(variable)), 'rb')
    results_ex2 = pkl.load(f)
    f.close()

    dice_ex2 = results_ex2["dice"].reshape((-1, np.array(results_ex2["dice"]).shape[-1]))[:, input_map]
    hd95_ex2 = np.array(results_ex2["hd95"]).reshape((-1, np.array(results_ex2["hd95"]).shape[-1]))[:, input_map_hd]
    hd_ex2 = np.array(results_ex2["hd"]).reshape((-1, np.array(results_ex2["hd"]).shape[-1]))[:, input_map_hd]
    vol_pred_ex2 = results_ex2["vol_pred"].reshape((-1, np.array(results_ex2["vol_pred"]).shape[-1]))[:, input_map]

    f = open(os.path.join(root_dir, dataset, "inference", "results_{}_2.pkl".format(variable)), 'rb')
    results_ex3 = pkl.load(f)
    f.close()

    dice_ex3 = results_ex3["dice"].reshape((-1, np.array(results_ex3["dice"]).shape[-1]))[:, input_map]
    hd95_ex3 = np.array(results_ex3["hd95"]).reshape((-1, np.array(results_ex3["hd95"]).shape[-1]))[:, input_map_hd]
    hd_ex3 = np.array(results_ex3["hd"]).reshape((-1, np.array(results_ex3["hd"]).shape[-1]))[:, input_map_hd]
    vol_pred_ex3 = results_ex3["vol_pred"].reshape((-1, np.array(results_ex3["vol_pred"]).shape[-1]))[:, input_map]

    # sort by characteristic of interest
    if variable == "Age":
        dice_g1_ex1 = dice_ex1[age <= age_1, :]
        dice_g2_ex1 = dice_ex1[age >= age_2, :]
        dice_g1_ex2 = dice_ex2[age <= age_1, :]
        dice_g2_ex2 = dice_ex2[age >= age_2, :]
        dice_g1_ex3 = dice_ex3[age <= age_1, :]
        dice_g2_ex3 = dice_ex3[age >= age_2, :]
    elif variable == "Sex":
        dice_g1_ex1 = dice_ex1[sex == 0, :]
        dice_g2_ex1 = dice_ex1[sex == 1, :]
        dice_g1_ex2 = dice_ex2[sex == 0, :]
        dice_g2_ex2 = dice_ex2[sex == 1, :]
        dice_g1_ex3 = dice_ex3[sex == 0, :]
        dice_g2_ex3 = dice_ex3[sex == 1, :]


    # Plot Dice Scores
    #save_path = os.path.join(output_dir, dataset, variable, "plots", "dice")
    #plotDistribution(dice_g1_ex1, dice_g2_ex1, dice_g1_ex2, dice_g2_ex2, dice_g1_ex3, dice_g2_ex3, label_g1, label_g2,
    #                 organ_dict, "Dice Scores", save_path)

    # Print dice score gaps
    file_path = os.path.join(output_dir, dataset, variable, "text", "dice_short.txt")
    printDeltas2(dice_g1_ex2, dice_g2_ex2, dice_g1_ex3, dice_g2_ex3, organ_dict, "Dice", file_path)

    # HD95
    # sort by characteristic of interest
    if variable == "Age":
        hd95_g1_ex1 = hd95_ex1[age <= age_1, :]
        hd95_g2_ex1 = hd95_ex1[age >= age_2, :]
        hd95_g1_ex2 = hd95_ex2[age <= age_1, :]
        hd95_g2_ex2 = hd95_ex2[age >= age_2, :]
        hd95_g1_ex3 = hd95_ex3[age <= age_1, :]
        hd95_g2_ex3 = hd95_ex3[age >= age_2, :]
    elif variable == "Sex":
        hd95_g1_ex1 = hd95_ex1[sex == 0, :]
        hd95_g2_ex1 = hd95_ex1[sex == 1, :]
        hd95_g1_ex2 = hd95_ex2[sex == 0, :]
        hd95_g2_ex2 = hd95_ex2[sex == 1, :]
        hd95_g1_ex3 = hd95_ex3[sex == 0, :]
        hd95_g2_ex3 = hd95_ex3[sex == 1, :]

    # Plot HD95 distribution for each organ
    save_path = os.path.join(output_dir, dataset, variable, "plots", "hd95")
    plotDistribution(hd95_g1_ex1, hd95_g2_ex1, hd95_g1_ex2, hd95_g2_ex2, hd95_g1_ex3, hd95_g2_ex3,
                     label_g1, label_g2, organ_dict, "HD95", save_path)

    # Print HD95 gaps
    file_path = os.path.join(output_dir, dataset, variable, "text", "hd95.txt")
    printDeltas(hd95_g1_ex1, hd95_g2_ex1, hd95_g1_ex2, hd95_g2_ex2, hd95_g1_ex3, hd95_g2_ex3, organ_dict, "HD95", file_path)


    # HD
    # sort by characteristic of interest
    if variable == "Age":
        hd_g1_ex1 = hd_ex1[age <= age_1, :]
        hd_g2_ex1 = hd_ex1[age >= age_2, :]
        hd_g1_ex2 = hd_ex2[age <= age_1, :]
        hd_g2_ex2 = hd_ex2[age >= age_2, :]
        hd_g1_ex3 = hd_ex3[age <= age_1, :]
        hd_g2_ex3 = hd_ex3[age >= age_2, :]
    elif variable == "Sex":
        hd_g1_ex1 = hd_ex1[sex == 0, :]
        hd_g2_ex1 = hd_ex1[sex == 1, :]
        hd_g1_ex2 = hd_ex2[sex == 0, :]
        hd_g2_ex2 = hd_ex2[sex == 1, :]
        hd_g1_ex3 = hd_ex3[sex == 0, :]
        hd_g2_ex3 = hd_ex3[sex == 1, :]

    # Plot HD distribution for each organ
    save_path = os.path.join(output_dir, dataset, variable, "plots", "hd")
    plotDistribution(hd_g1_ex1, hd_g2_ex1, hd_g1_ex2, hd_g2_ex2, hd_g1_ex3, hd_g2_ex3,
                     label_g1, label_g2, organ_dict, "HD", save_path)

    # Print HD gaps
    file_path = os.path.join(output_dir, dataset, variable, "text", "hd.txt")
    printDeltas(hd_g1_ex1, hd_g2_ex1, hd_g1_ex2, hd_g2_ex2, hd_g1_ex3, hd_g2_ex3, organ_dict, "HD",
                file_path)

    # VOLUME
    vol_err_ex1, vol_err_ex2, vol_err_ex3 = getVolumeErrors(vol_gt, vol_pred_ex1, vol_pred_ex2, vol_pred_ex3)

    if variable == "Age":
        vol_err_g1_ex1 = vol_err_ex1[age <= age_1, :]
        vol_err_g2_ex1 = vol_err_ex1[age >= age_2, :]
        vol_err_g1_ex2 = vol_err_ex2[age <= age_1, :]
        vol_err_g2_ex2 = vol_err_ex2[age >= age_2, :]
        vol_err_g1_ex3 = vol_err_ex3[age <= age_1, :]
        vol_err_g2_ex3 = vol_err_ex3[age >= age_2, :]

        vol_gt_g1 = vol_gt[age <= age_1]
        vol_gt_g2 = vol_gt[age >= age_2]
    elif variable == "Sex":
        vol_err_g1_ex1 = vol_err_ex1[sex == 0, :]
        vol_err_g2_ex1 = vol_err_ex1[sex == 1, :]
        vol_err_g1_ex2 = vol_err_ex2[sex == 0, :]
        vol_err_g2_ex2 = vol_err_ex2[sex == 1, :]
        vol_err_g1_ex3 = vol_err_ex3[sex == 0, :]
        vol_err_g2_ex3 = vol_err_ex3[sex == 1, :]

        vol_gt_g1 = vol_gt[sex == 0]
        vol_gt_g2 = vol_gt[sex == 1]

    # Plot distribution of volume errors
    save_path = os.path.join(output_dir, dataset, variable, "plots", "volume")
    plotDistribution(vol_err_g1_ex1, vol_err_g2_ex1, vol_err_g1_ex2, vol_err_g2_ex2, vol_err_g1_ex3, vol_err_g2_ex3,
                     label_g1, label_g2, organ_dict, "Volume Error", save_path)

    # Print volume for each organ
    file_path = os.path.join(output_dir, dataset, variable, "text", "volumes.txt")
    printVolumes(vol_gt_g1, vol_gt_g2, organ_dict, file_path)

    # Plot distribution of GT volumes for all organs at once
    save_path = os.path.join(output_dir, dataset, variable, "plots", "volume")
    plotVolumeDistribution(vol_gt_g1, vol_gt_g2, label_g1, label_g2, organ_dict, save_path)

    #printVolumeErrors(vol_err_g1_ex1, vol_err_g2_ex1, vol_err_g1_ex2, vol_err_g2_ex2, vol_err_g1_ex3, vol_err_g2_ex3,
    #                  organ_dict, save_path)


if __name__ == "__main__":
    main()