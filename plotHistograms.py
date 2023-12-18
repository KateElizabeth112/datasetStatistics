import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl


root_dir = "/Users/katecevora/Documents/PhD/data"
kits_metadata = os.path.join(root_dir, "KITS19", "info.pkl")
ts_metadata = os.path.join(root_dir, "TotalSegmentator", "info.pkl")
amos_metadata = os.path.join(root_dir, "AMOS_3D", "info.pkl")

paths = [kits_metadata, ts_metadata, amos_metadata]
labels = ["kits", "totalSegmentator", "AMOS"]


def plot_age():
    # check proportions
    g1_max = 50
    g2_min = 70

    plt.clf()
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for i in range(len(paths)):
        f = open(paths[i], "rb")
        metadata = pkl.load(f)
        f.close()

        age = metadata["age"]

        g1_count = np.sum(np.array(age) <= g1_max)
        g2_count = np.sum(np.array(age) >= g2_min)

        print(labels[i])
        print("g1: {}".format(g1_count))
        print("g2: {}".format(g2_count))

        plt.hist(age, label=labels[i], bins=bins, alpha=0.6, density=True)

    plt.legend()
    plt.show()






def main():
    plot_age()


if __name__ == "__main__":
    main()