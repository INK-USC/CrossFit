import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class YelpPolarity(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "yelp_polarity"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"negative",
            1:"positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")

        np.random.seed(42)
        np.random.shuffle(test_lines)
        n = len(test_lines)
        test_lines = test_lines[:int(0.2*n)]
        # using 20% of test cases, otherwise it's too slow to do evaluation

        return train_lines, test_lines
        
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["text"].replace("\\n", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('yelp_polarity')

def main():
    dataset = YelpPolarity()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()