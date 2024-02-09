# Create metadata dictionary for the full dataset
# Create a pkl file that contains dataset information in a dictionary of arrays
# Include only images that have 4 organs of interest labelled
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pkl
import argparse
import numpy as np


# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--local", default=False, help="are we running locally or on hcp clusters")
args = vars(parser.parse_args())

root_folder = "/Users/katecevora/Documents/PhD/data/TotalSegmentator"

def create():

    meta = pd.read_csv(os.path.join(root_folder, "meta.csv"), sep=";")
    patients = meta["image_id"].values
    genders = meta["gender"].values
    age = meta["age"].values
    institute = meta["institute"].values
    study_type = meta["study_type"].values

    genders = np.array(genders)
    sex = np.zeros(genders.shape[0])
    sex[genders == "f"] = 1

    # Save lists
    info = {"id": np.array(patients),
            "sex": np.array(sex),
            "age": np.array(age),
            "institute": np.array(institute),
            "study_type": np.array(study_type)}

    f = open(os.path.join(root_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()



def main():
    create()


if __name__ == "__main__":
    main()