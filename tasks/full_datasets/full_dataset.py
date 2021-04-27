import os
import numpy as np

class FeltFullDataset():
    def write_to_tsv(self, lst, out_file):
        with open(out_file, "w") as fout:
            for line in lst:
                fout.write("{}\t{}\n".format(line[0], line[1]))

class FeltFullClassificationDataset(FeltFullDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def generate_train_dev_test_data(self, seed, n_train=None, n_dev=1024, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        if n_train is None:
            k_shot_train = train_lines[:-n_dev]
        else:
            assert n_train + n_dev <= len(train_lines)
            k_shot_train = train_lines[:n_train]

        k_shot_dev = train_lines[-n_dev:]

        k_shot_test = test_lines

        # save to path
        if path:
            os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
            prefix = os.path.join(path, self.hf_identifier, "{}_full_{}".format(self.hf_identifier, seed))
            self.write_to_tsv(k_shot_train, prefix + "_train.tsv")
            self.write_to_tsv(k_shot_dev, prefix + "_dev.tsv")
            self.write_to_tsv(k_shot_test, prefix + "_test.tsv")

        return k_shot_train, k_shot_dev, k_shot_test

class FeltFullTextToTextDataset(FeltFullDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def generate_train_dev_test_data(self, seed, n_train=None, n_dev=1024, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        if n_train is None:
            k_shot_train = train_lines[:-n_dev]
        else:
            assert n_train + n_dev <= len(train_lines)
            k_shot_train = train_lines[:n_train]

        k_shot_dev = train_lines[-n_dev:]

        k_shot_test = test_lines

        # save to path
        if path:
            os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
            prefix = os.path.join(path, self.hf_identifier, "{}_full_{}".format(self.hf_identifier, seed))
            self.write_to_tsv(k_shot_train, prefix + "_train.tsv")
            self.write_to_tsv(k_shot_dev, prefix + "_dev.tsv")
            self.write_to_tsv(k_shot_test, prefix + "_test.tsv")

        return k_shot_train, k_shot_dev, k_shot_test