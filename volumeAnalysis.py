import numpy as np
import pickle as pkl
import os

root_dir = "/Users/katecevora/Documents/PhD/data"

dataset = "TS"
variable = "Sex"

def getTrainingAndTestSet(dataset, variable):
    # For each experiment, iterate over the folds and recombine training IDs

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

        g1_train_ids += list(set_2_ids["train"])
        g2_train_ids += list(set_3_ids["train"])
        test_ids += list(set_2_ids["test"])

    g1_train_ids = np.unique(np.array(g1_train_ids))
    g2_train_ids = np.unique(np.array(g2_train_ids))
    test_ids = np.array(test_ids)

    # save
    f = open(os.path.join(root_dir, "training_splits", dataset, "combined_{}.pkl".format(variable)), "wb")
    pkl.dump([g1_train_ids, g2_train_ids, test_ids], f)
    f.close()


def main():
    getTrainingAndTestSet("TS", "Sex")
    getTrainingAndTestSet("TS", "Age")
    getTrainingAndTestSet("AMOS", "Sex")
    getTrainingAndTestSet("AMOS", "Age")


if __name__ == "__main__":
    main()