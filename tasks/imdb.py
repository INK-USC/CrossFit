import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset
from utils import clean

class IMDB(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "imdb"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "negative",
            1: "positive",
        }

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")
        
        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((clean(datapoint["text"]), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('imdb')

def main():
    dataset = IMDB()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()