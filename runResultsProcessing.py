import numpy as np
import pickle as pkl
import os
import numpy as np
from plotResults import plotDistribution, plotVolumeDistribution
from printResults import printPerformanceGap, printDice

import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="TotalSegmentator", help="Dataset")
parser.add_argument("-v", "--variable", default="Age", help="Variable of interest")
args = vars(parser.parse_args())

root_dir = "/Users/katecevora/Documents/PhD/data"
output_dir = "/Users/katecevora/Documents/PhD/results"


dataset = args["dataset"]
variable = args["variable"]

organ_dict = {"background": 0,
              "right kidney": 1,
              "left kidney": 2,
              "liver": 3,
              "pancreas": 4}

if dataset == "TotalSegmentator":

    if variable == "Sex":
        label_g1 = "Female"
        label_g2 = "Male"
    elif variable == "Age":
        label_g1 = "Under 50"
        label_g2 = "Over 70"
        age_1 = 50
        age_2 = 70

    input_map = [0, 1, 2, 3, 4]
    input_map_hd = [0, 1, 2, 3]

elif dataset == "AMOS_3D":

    if variable == "Sex":
        label_g1 = "Female"
        label_g2 = "Male"
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
    createFolders(output_dir, dataset, variable)

    f = open(os.path.join(root_dir, dataset, "inference", "results_{}_0.pkl".format(variable)), 'rb')
    results_ex2 = pkl.load(f)
    f.close()

    sex = results_ex2["sex"].flatten()
    age = results_ex2["age"].flatten()
    dice_ex2 = results_ex2["dice"].reshape((-1, np.array(results_ex2["dice"]).shape[-1]))[:, input_map]
    hd95_ex2 = np.array(results_ex2["hd95"]).reshape((-1, np.array(results_ex2["hd95"]).shape[-1]))[:, input_map_hd]

    f = open(os.path.join(root_dir, dataset, "inference", "results_{}_1.pkl".format(variable)), 'rb')
    results_ex3 = pkl.load(f)
    f.close()

    dice_ex3 = results_ex3["dice"].reshape((-1, np.array(results_ex3["dice"]).shape[-1]))[:, input_map]
    hd95_ex3 = np.array(results_ex3["hd95"]).reshape((-1, np.array(results_ex3["hd95"]).shape[-1]))[:, input_map_hd]

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

    file_path = os.path.join(output_dir, dataset, variable, "text", "dice_gap.txt")
    printPerformanceGap(dice_g1_ex2, dice_g2_ex2, dice_g1_ex3, dice_g2_ex3, organ_dict, "Dice", file_path, dataset, variable)

    file_path = os.path.join(output_dir, dataset, variable, "text", "dice_raw.txt")
    printDice(dice_g1_ex2, dice_g2_ex2, dice_g1_ex3, dice_g2_ex3, organ_dict, "Dice", file_path, dataset, variable)

    # HD95
    # sort by characteristic of interest
    if variable == "Age":
        hd95_g1_ex2 = hd95_ex2[age <= age_1, :]
        hd95_g2_ex2 = hd95_ex2[age >= age_2, :]
        hd95_g1_ex3 = hd95_ex3[age <= age_1, :]
        hd95_g2_ex3 = hd95_ex3[age >= age_2, :]
    elif variable == "Sex":
        hd95_g1_ex2 = hd95_ex2[sex == 1, :]
        hd95_g2_ex2 = hd95_ex2[sex == 0, :]
        hd95_g1_ex3 = hd95_ex3[sex == 1, :]
        hd95_g2_ex3 = hd95_ex3[sex == 0, :]

    file_path = os.path.join(output_dir, dataset, variable, "text", "hd95_gap.txt")
    printPerformanceGap(hd95_g1_ex2, hd95_g2_ex2, hd95_g1_ex3, hd95_g2_ex3, organ_dict, "HD95", file_path, dataset, variable)


if __name__ == "__main__":
    main()