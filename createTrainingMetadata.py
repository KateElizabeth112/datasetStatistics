# Create metadata files specifically fot each experimental group of subjects
import numpy as np
import pickle as pkl
import os

root_dir = "/Users/katecevora/Documents/PhD/data"
output_dir = "/Users/katecevora/Documents/PhD/results"

#dataset = "TotalSegmentator"
dataset = "AMOS_3D"

ts_metadata = os.path.join(root_dir, "TotalSegmentator", "info.pkl")
amos_metadata = os.path.join(root_dir, "AMOS_3D", "info.pkl")

paths = [ts_metadata, amos_metadata]


def sortMetadataAMOS(variable):
    # open the results
    f = open(os.path.join(root_dir, "AMOS_3D", "inference", "results_{}_0.pkl".format(variable)), 'rb')
    results_ex1 = pkl.load(f)
    f.close()

    id = results_ex1["case_id"].flatten()

    # open the metadata
    f = open(paths[1], "rb")
    metadata = pkl.load(f)
    f.close()

    id_all = np.array(metadata["id"])
    age_all = np.array(metadata["age"])
    sex_all = np.array(metadata["sex"])
    site_all = np.array(metadata["site"])
    scanner_all = np.array(metadata["scanner"])

    # Empty lists to store sorted info
    age = []
    sex = []
    site = []
    scanner = []

    # Cycle over all IDs
    for i in id:
        age.append(age_all[id_all == i][0])
        sex.append(sex_all[id_all == i][0])
        site.append(site_all[id_all == i][0])
        scanner.append(scanner_all[id_all == i][0])


    # Store result
    new_metadata = {"id": id,
                    "age": age,
                    "sex": sex,
                    "site": site,
                    "scanner": scanner}

    f = open(os.path.join(root_dir, "AMOS_3D", "inference", "metadata_{}.pkl".format(variable)), 'wb')
    pkl.dump(new_metadata, f)
    f.close()

    print("done")


def sortMetadataTS(variable):
    # open the results
    f = open(os.path.join(root_dir, "TotalSegmentator", "inference", "results_{}_0.pkl".format(variable)), 'rb')
    results_ex1 = pkl.load(f)
    f.close()

    id = results_ex1["case_id"].flatten()

    # open the metadata
    f = open(paths[0], "rb")
    metadata = pkl.load(f)
    f.close()

    id_all = np.array(metadata["id"])
    age_all = np.array(metadata["age"])
    sex_all = np.array(metadata["sex"])
    institute_all = np.array(metadata["institute"])
    study_type_all = np.array(metadata["study_type"])

    # Empty lists to store sorted info
    age = []
    sex = []
    institute = []
    study_type = []

    # Cycle over all IDs
    for i in id:
        age.append(age_all[id_all == "s" + i][0])
        sex.append(sex_all[id_all == "s" + i][0])
        institute.append(institute_all[id_all == "s" + i][0])
        study_type.append(study_type_all[id_all == "s" + i][0])


    # Store result
    new_metadata = {"id": id,
                    "age": age,
                    "sex": sex,
                    "institute": institute,
                    "study_type": study_type}

    f = open(os.path.join(root_dir, "TotalSegmentator", "inference", "metadata_{}.pkl".format(variable)), 'wb')
    pkl.dump(new_metadata, f)
    f.close()

    print("done")




def main():
    #sortMetadataAMOS("Age")
    #sortMetadataAMOS("Sex")
    sortMetadataTS("Age")
    sortMetadataTS("Sex")

if __name__ == "__main__":
    main()