# script for combining the results from multiple folds into a single file
import pickle as pkl
import numpy as np
import os
import argparse

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--rootdir", default="/rds/general/user/kc2322/home/data/AMOS_3D", help="full path to root directory")
parser.add_argument("-v", "--variable", default="Age", help="Variable of interest.")
args = vars(parser.parse_args())

root_dir = args["rootdir"]
variable = args["variable"]

folds = [0, 1, 2, 3, 4]

if variable == "Age":
    s = "Age"
elif variable == "Sex":
    s = "Fold"
else:
    print("Variable {} not recognised".format(variable))


def main():
    # iterate over 3 experiments
    for ex in range(0, 3):
        case_id_all = []
        sex_all = []
        age_all = []
        dice_all = []
        hd95_all = []
        hd_all = []
        vol_pred_all = []
        vol_gt_all = []

        # iterate over folds and combine
        for fold in folds:
            ds = "Dataset{}0{}_{}{}".format(5 + fold, ex+1, s, fold)
            f = open(os.path.join(root_dir, "inference", ds, "all", "results.pkl"), "rb")
            results = pkl.load(f)
            f.close()

            print(np.array(results["dice"]).shape)

            case_id_all.append(list(results["case_id"]))
            sex_all.append(list(results["sex"]))
            age_all.append(list(results["age"]))
            dice_all.append(list(results["dice"]))
            hd_all.append(results["hd"])
            hd95_all.append(results["hd95"])
            vol_pred_all.append(list(results["vol_pred"]))
            vol_gt_all.append(list(results["vol_gt"]))

        print(np.array(dice_all).shape)

        f = open(os.path.join(root_dir, "inference", "results_{}_{}.pkl".format(s, ex)), 'wb')
        pkl.dump({"case_id": np.array(case_id_all),
                  "sex": np.array(sex_all),
                  "age": np.array(age_all),
                  "dice": np.array(dice_all),
                  "hd": np.array(hd_all),
                  "hd95": np.array(hd95_all),
                  "vol_pred": np.array(vol_pred_all),
                  "vol_gt": np.array(vol_gt_all),
                  }, f)
        f.close()


if __name__ == "__main__":
    main()