import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class YelpReviewFull(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "yelp_review_full"

        self.task_type = "regression"


    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")

        return train_lines, test_lines
        
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["text"].replace("\\n", " "), str(datapoint["label"] + 1)))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('yelp_review_full')

def main():
    dataset = YelpReviewFull()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()