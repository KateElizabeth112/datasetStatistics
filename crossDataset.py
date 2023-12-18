# Examine cross dataset degradation
import numpy as np
import pickle as pkl
import os
from plotResults import plotDualDistribution, plotVolumeDistribution
from printResults import printCrossDeltas, printVolumes

root_dir = "/Users/katecevora/Documents/PhD/data"
output_dir = "/Users/katecevora/Documents/PhD/results"
dataset1 = "TotalSegmentator"
dataset2 = "AMOS_3D"

organ_dict = {"background": 0,
              "right kidney": 1,
              "left kidney": 2,
              "liver": 3,
              "pancreas": 4}


def main():
    f = open(os.path.join(root_dir, dataset1, "inference", "results_combined_cross.pkl"), 'rb')
    results_cross = pkl.load(f)
    f.close()

    d1_case_id = results_cross["case_id"].flatten()
    d1_sex = results_cross["sex"].flatten()
    d1_age = results_cross["age"].flatten()

    d1_dice_cross = results_cross["dice"].reshape((-1, np.array(results_cross["dice"]).shape[-1]))
    d1_hd_cross = np.array(results_cross["hd"]).reshape((-1, np.array(results_cross["hd"]).shape[-1]))
    d1_vol_pred_cross = results_cross["vol_pred"].reshape((-1, np.array(results_cross["vol_pred"]).shape[-1]))
    d1_vol_gt = results_cross["vol_gt"].reshape((-1, np.array(results_cross["vol_gt"]).shape[-1]))

    f = open(os.path.join(root_dir, dataset1, "inference", "results_age_0.pkl"), 'rb')
    results_self = pkl.load(f)
    f.close()

    d1_dice_self = results_self["dice"].reshape((-1, np.array(results_self["dice"]).shape[-1]))
    d1_hd_self = np.array(results_self["hd"]).reshape((-1, np.array(results_self["hd"]).shape[-1]))
    d1_vol_pred_self = results_self["vol_pred"].reshape((-1, np.array(results_self["vol_pred"]).shape[-1]))
    d1_vol_gt = results_self["vol_gt"].reshape((-1, np.array(results_self["vol_gt"]).shape[-1]))

    save_path = os.path.join(output_dir, "cross_dataset", "inference", "plots", "TS_dice.png")
    plotDualDistribution(d1_dice_self, d1_dice_cross, "Self", "Cross", "Dice Score", organ_dict, save_path)

    save_path = os.path.join(output_dir, "cross_dataset", "inference", "plots", "TS_hd.png")
    plotDualDistribution(d1_hd_self, d1_hd_cross, "Self", "Cross", "HD95", organ_dict, save_path)

    # open the results from the second dataset
    f = open(os.path.join(root_dir, dataset2, "inference", "results_combined_cross.pkl"), 'rb')
    results_cross = pkl.load(f)
    f.close()

    d2_case_id = results_cross["case_id"].flatten()
    d2_sex = results_cross["sex"].flatten()
    d2_age = results_cross["age"].flatten()

    d2_dice_cross = results_cross["dice"].reshape((-1, np.array(results_cross["dice"]).shape[-1]))
    d2_hd_cross = np.array(results_cross["hd"]).reshape((-1, np.array(results_cross["hd"]).shape[-1]))
    d2_vol_pred_cross = results_cross["vol_pred"].reshape((-1, np.array(results_cross["vol_pred"]).shape[-1]))
    d2_vol_gt = results_cross["vol_gt"].reshape((-1, np.array(results_cross["vol_gt"]).shape[-1]))

    f = open(os.path.join(root_dir, dataset2, "inference", "results_age_0.pkl"), 'rb')
    results_self = pkl.load(f)
    f.close()

    d2_dice_self = results_self["dice"].reshape((-1, np.array(results_self["dice"]).shape[-1]))
    d2_hd_self = np.array(results_self["hd"]).reshape((-1, np.array(results_self["hd"]).shape[-1]))
    d2_vol_pred_self = results_self["vol_pred"].reshape((-1, np.array(results_self["vol_pred"]).shape[-1]))
    d2_vol_gt = results_self["vol_gt"].reshape((-1, np.array(results_self["vol_gt"]).shape[-1]))

    save_path = os.path.join(output_dir, "cross_dataset", "inference", "plots", "AMOS_dice.png")
    plotDualDistribution(d2_dice_self, d2_dice_cross, "Self", "Cross", "Dice Score", organ_dict, save_path)

    save_path = os.path.join(output_dir, "cross_dataset", "inference", "plots", "AMOS_hd.png")
    plotDualDistribution(d2_hd_self, d2_hd_cross, "Self", "Cross", "HD95", organ_dict, save_path)

    # print and plot the two datasets together
    file_path = os.path.join(output_dir, "cross_dataset", "inference", "text", "dice.txt")
    printCrossDeltas(d1_dice_self, d2_dice_self, d1_dice_cross, d2_dice_cross, organ_dict, file_path)

    # volume distribution
    save_path = os.path.join(output_dir, "cross_dataset", "inference", "plots")
    plotVolumeDistribution(d1_vol_gt, d2_vol_gt, "TS", "AMOS", organ_dict, save_path)

    file_path = os.path.join(output_dir, "cross_dataset", "inference", "text", "volume.txt")
    printVolumes(d1_vol_gt, d2_vol_gt, organ_dict, file_path)


if __name__ == "__main__":
    main()