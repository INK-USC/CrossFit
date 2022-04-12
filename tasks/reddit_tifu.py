import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset
from utils import clean

class Reddit_TIFU(FewshotGymTextToTextDataset):

    def __init__(self, subset):
        self.hf_identifier = "reddit_tifu-" + subset
        self.subset = subset
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("summarize: " + clean(datapoint["documents"]), clean(datapoint[self.subset])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("reddit_tifu", "long")

def main():
    dataset = Reddit_TIFU("tldr")

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

    dataset = Reddit_TIFU("title")

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()