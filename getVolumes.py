# iterate over the full datasets and store the organ volumes alongside the patient ID in a pickle
import os
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
from plottingFunctions import plot3Dmesh

import argparse

parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="TS", help="Task to evaluate")
args = vars(parser.parse_args())

dataset = args["dataset"]

if dataset == "TS":
    root_dir = "/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator"
    gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset300_Full", "labelsTr")
    organs = [1, 2, 3, 4]

elif dataset == "AMOS":
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"
    gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset200_AMOS", "labelsTr")
    organs = [2, 3, 6, 10]

    labels = {"background": 0,
              "spleen": 1,
              "right kidney": 2,
              "left kidney": 3,
              "gallbladder": 4,
              "esophagus": 5,
              "liver": 6,
              "stomach": 7,
              "aorta": 8,
              "inferior vena cava": 9,
              "pancreas": 10,
              "right adrenal gland": 11,
              "left adrenal gland": 12,
              "duodenum": 13,
              "bladder": 14,
              "prostate/uterus": 15}


def calculate_volumes():
    # create containers to store the volumes
    volumes_all = []
    subjects = []

    # get a list of the files in the gt seg folder
    f_names = os.listdir(gt_seg_dir)

    for f in f_names:
        if f.endswith(".nii.gz"):
            # load image
            gt_nii = nib.load(os.path.join(gt_seg_dir, f))

            # get the volume of 1 voxel in mm3
            sx, sy, sz = gt_nii.header.get_zooms()
            volume = sx * sy * sz

            # find the number of voxels per organ in the ground truth image
            gt = gt_nii.get_fdata()
            volumes = []

            for i in organs:
                voxel_count = np.sum(gt == i)
                volumes.append(voxel_count * volume)

            volumes_all.append(np.array(volumes))

            subjects.append(f[5:9])

            if dataset == "AMOS":
                # ground truth needs to have number of channels reduced
                input_map = [2, 3, 6, 10]
                output_map = [1, 2, 3, 4]

                gt_full = gt_nii.get_fdata()
                gt = np.zeros(gt_full.shape)
                for q in range(len(input_map)):
                    gt[gt_full == input_map[q]] = output_map[q]

            # plot the organs as a 3D mesh
            save_path = os.path.join(root_dir, "volume_plots")
            plot3Dmesh(gt, save_path, f[5:9])

    # Save the volumes ready for further processing
    f = open(os.path.join(root_dir, "volumes.pkl"), "wb")
    pkl.dump([np.array(subjects), np.array(volumes_all)], f)
    f.close()


def main():
    calculate_volumes()


if __name__ == "__main__":
    main()