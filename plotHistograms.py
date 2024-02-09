import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import pandas as pd
from scipy.stats import chisquare, ttest_ind, chi2_contingency
import string

root_dir = "/Users/katecevora/Documents/PhD/data"
kits_metadata = os.path.join(root_dir, "KITS19", "info.pkl")
#ts_metadata = os.path.join(root_dir, "TotalSegmentator", "info.pkl")
#amos_metadata = os.path.join(root_dir, "AMOS_3D", "info.pkl")



#paths = [ts_metadata, amos_metadata]
labels = ["totalSegmentator", "AMOS"]


lblu = "#add9f4"
lred = "#f36860"
lgrn = "#94c267"
lgry = "#f3efec"
lorg = "#e8a32c"

colors = [lblu, lgrn]


def totalSegmentatorPlots(variable):
    f = open(os.path.join(root_dir, "TotalSegmentator", "inference", "metadata_{}.pkl".format(variable)), "rb")
    metadata = pkl.load(f)
    f.close()

    if variable == "Age":
        group1 = "Under 50"
        group2 = "Over 70"
        colours = [lgrn, lorg]
        print("Variable is age.")
    elif variable == "Sex":
        group1 = "Male"
        group2 = "Female"
        colours = [lblu, lred]
        print("Variable is sex.")

    age = np.array(metadata["age"])
    sex = np.array(metadata["sex"])
    institute = np.array(metadata["institute"])
    study_type = np.array(metadata["study_type"])

    # Split institutes by variable of interest
    institutes = np.unique(institute)
    institute_group_1 = []
    institute_group_2 = []

    if variable == "Sex":
        for s in institutes:
            institute_group_1.append(institute[(sex == 1) & (institute == s)].shape[0])
            institute_group_2.append(institute[(sex == 0) & (institute == s)].shape[0])

    elif variable == "Age":
        for s in institutes:
            institute_group_1.append(institute[(age <= 50) & (institute == s)].shape[0])
            institute_group_2.append(institute[(age >= 70) & (institute == s)].shape[0])

    else:
        print("Variable not recognised.")

    # Perform chi-square test for independence
    chi2_stat, p_value1, dof, expected = chi2_contingency(np.array((np.array(institute_group_1), np.array(institute_group_2))))
    print("P-value for institutes: {}".format(p_value1))

    # Split study_types by variable of interest
    study_types = np.unique(study_type)
    study_type_group_1 = []
    study_type_group_2 = []
    if variable == "Sex":
        for s in study_types:
            study_type_group_1.append(study_type[(sex == 1) & (study_type == s)].shape[0])
            study_type_group_2.append(study_type[(sex == 0) & (study_type == s)].shape[0])
    elif variable == "Age":
        for s in study_types:
            study_type_group_1.append(study_type[(age <= 50) & (study_type == s)].shape[0])
            study_type_group_2.append(study_type[(age >= 70) & (study_type == s)].shape[0])

    # Perform chi-square test for independence
    chi2_stat, p_value2, dof, expected = chi2_contingency(np.array((np.array(study_type_group_1), np.array(study_type_group_2))))
    print("P-value for study_type: {}".format(p_value2))

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2))
    ax1, ax2, ax3 = axes.flatten()

    d = {"institute": institutes,
         group1: np.array(institute_group_1) / np.sum(np.array(institute_group_1)),
         group2: np.array(institute_group_2) / np.sum(np.array(institute_group_2))}

    df = pd.DataFrame(data=d)
    df.plot.bar(x='institute', ax=ax1, color=colours)
    ax1.set_yticks([])
    ax1.set_ylabel("")
    ax1.set_title("P={0:.3f}".format(p_value1))

    d = {"study_type": list(string.ascii_uppercase)[:len(study_types)],
         group1: np.array(study_type_group_1) / np.sum(np.array(study_type_group_1)),
         group2: np.array(study_type_group_2) / np.sum(np.array(study_type_group_2))}

    df = pd.DataFrame(data=d)
    df.plot.bar(x='study_type', ax=ax2, color=colours)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_title("P={0:.3f}".format(p_value2))

    if variable == "Age":
        sex_groups = np.unique(sex)
        sex_group_1 = []
        sex_group_2 = []

        for s in sex_groups:
            sex_group_1.append(sex[(age <= 50) & (sex == s)].shape[0])
            sex_group_2.append(sex[(age >= 70) & (sex == s)].shape[0])

        # Perform chi-square test for independence
        chi2_stat, p_value3, dof, expected = chi2_contingency(np.array((np.array(sex_group_1), np.array(sex_group_2))))
        print("P-value for sex: {}".format(p_value3))

        d = {"Sex": sex_groups,
             group1: np.array(sex_group_1) / np.sum(np.array(sex_group_1)),
             group2: np.array(sex_group_2) / np.sum(np.array(sex_group_2))}

        df = pd.DataFrame(data=d)
        df.plot.bar(x='Sex', ax=ax3, color=colours)
        ax3.set_xticklabels(["M", "F"])
        ax3.set_yticks([])
        ax3.set_ylabel("")
        ax3.set_title("P={0:.3f}".format(p_value3))

    elif variable == "Sex":
        age_group_1 = age[sex == 0]
        age_group_2 = age[sex == 1]

        (_, p_value3) = ttest_ind(age_group_1, age_group_2, equal_var=False)
        print("P-value for age: {}".format(p_value3))

        # df.plot.bar(x='Sex', ax=ax3, color=colours)
        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ax3.hist(age_group_1, color=lblu, label=group1, bins=bins)
        ax3.hist(age_group_2, color=lred, label=group2, bins=bins)
        ax3.legend()
        ax3.set_xlabel("Age")
        ax3.set_yticks([])
        ax3.set_ylabel("")
        ax3.set_title("P={0:.3f}".format(p_value3))

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join("/Users/katecevora/Documents/PhD/", "results", "datasets", "TS_{}_chisquare.png".format(variable)))


def AMOSPlots(variable):
    f = open(os.path.join(root_dir, "AMOS_3D", "inference", "metadata_{}.pkl".format(variable)), "rb")
    metadata = pkl.load(f)
    f.close()

    if variable == "Age":
        group1 = "Under 50"
        group2 = "Over 70"
        colours = [lgrn, lorg]
        print("Variable is age.")
    elif variable == "Sex":
        group1 = "Male"
        group2 = "Female"
        colours = [lblu, lred]
        print("Variable is sex.")

    age = np.array(metadata["age"])
    sex = np.array(metadata["sex"])
    site = np.array(metadata["site"])
    scanner = np.array(metadata["scanner"])

    # Split sites by variable of interest
    sites = np.unique(site)
    site_group_1 = []
    site_group_2 = []

    if variable == "Sex":
        for s in sites:
            site_group_1.append(site[(sex == 1) & (site == s)].shape[0])
            site_group_2.append(site[(sex == 0) & (site == s)].shape[0])

    elif variable == "Age":
        for s in sites:
            site_group_1.append(site[(age < 45) & (site == s)].shape[0])
            site_group_2.append(site[(age > 60) & (site == s)].shape[0])

    else:
        print("Variable not recognised.")

    # Perform chi-square test for goodness of fit
    chi2_stat, p_value1, dof, expected = chi2_contingency(np.array((np.array(site_group_1), np.array(site_group_2))))
    print("P-value for sites: {}".format(p_value1))

    # Split scanners by variable of interest
    scanners = np.unique(scanner)
    scanner_group_1 = []
    scanner_group_2 = []
    if variable == "Sex":
        for s in scanners:
            scanner_group_1.append(scanner[(sex == 1) & (scanner == s)].shape[0])
            scanner_group_2.append(scanner[(sex == 0) & (scanner == s)].shape[0])
    elif variable == "Age":
        for s in scanners:
            scanner_group_1.append(scanner[(age < 45) & (scanner == s)].shape[0])
            scanner_group_2.append(scanner[(age > 60) & (scanner == s)].shape[0])

    # Perform chi-square test for independence
    chi2_stat, p_value2, dof, expected = chi2_contingency(np.array((np.array(scanner_group_1), np.array(scanner_group_2))))
    print("P-value for scanner: {}".format(p_value2))

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2))
    ax1, ax2, ax3 = axes.flatten()

    d = {"Site": ["C", "P"],
         group1: np.array(site_group_1) / np.sum(np.array(site_group_1)),
         group2: np.array(site_group_2) / np.sum(np.array(site_group_2))}

    df = pd.DataFrame(data=d)
    df.plot.bar(x='Site', ax=ax1, color=colours)
    ax1.set_yticks([])
    ax1.set_ylabel("")
    ax1.set_title("P={0:.3f}".format(p_value1))

    d = {"Scanner": list(string.ascii_uppercase)[:len(scanners)],
         group1: np.array(scanner_group_1) / np.sum(np.array(scanner_group_1)),
         group2: np.array(scanner_group_2) / np.sum(np.array(scanner_group_2))}

    df = pd.DataFrame(data=d)
    df.plot.bar(x='Scanner', ax=ax2, color=colours)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_title("P={0:.3f}".format(p_value2))

    if variable == "Age":
        sex_groups = np.unique(sex)
        sex_group_1 = []
        sex_group_2 = []

        for s in sex_groups:
            sex_group_1.append(sex[(age < 45) & (sex == s)].shape[0])
            sex_group_2.append(sex[(age > 60) & (sex == s)].shape[0])

        # Perform chi-square test for goodness of fit
        chi2_stat, p_value3, dof, expected = chi2_contingency(np.array((np.array(sex_group_1), np.array(sex_group_2))))
        print("P-value for sex: {}".format(p_value3))

        d = {"Sex": sex_groups,
             group1: np.array(sex_group_1) / np.sum(np.array(sex_group_1)),
             group2: np.array(sex_group_2) / np.sum(np.array(sex_group_2))}

        df = pd.DataFrame(data=d)
        df.plot.bar(x='Sex', ax=ax3, color=colours)
        ax3.set_xticklabels(["M", "F"])
        ax3.set_yticks([])
        ax3.set_ylabel("")
        ax3.set_title("P={0:.3f}".format(p_value3))

    elif variable == "Sex":
        age_group_1 = age[sex == 0]
        age_group_2 = age[sex == 1]

        (_, p_value3) = ttest_ind(age_group_1, age_group_2, equal_var=False)
        print("P-value for age: {}".format(p_value3))

        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ax3.hist(age_group_1, color=lblu, label=group1, bins=bins)
        ax3.hist(age_group_2, color=lred, label=group2, bins=bins)
        ax3.legend()
        ax3.set_xlabel("Age")
        ax3.set_yticks([])
        ax3.set_ylabel("")
        ax3.set_title("P={0:.3f}".format(p_value3))

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join("/Users/katecevora/Documents/PhD/", "results", "datasets", "AMOS_{}_chisquare.png".format(variable)))


def main():
    totalSegmentatorPlots("Age")
    totalSegmentatorPlots("Sex")
    AMOSPlots("Age")
    AMOSPlots("Sex")

if __name__ == "__main__":
    main()