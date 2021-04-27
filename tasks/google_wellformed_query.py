import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class GoogleWellformedQuery(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "google_wellformed_query"
        self.task_type = "regression"
        self.license = "unknown"

        self.label = {
            0: "not well-formed",
            1: "well-formed"
        }

    
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["rating"] < 0.4:
                lines.append((datapoint["content"].strip(), self.label[0]))
            elif datapoint["rating"] > 0.6:
                lines.append((datapoint["content"].strip(), self.label[1]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('google_wellformed_query')

def main():
    dataset = GoogleWellformedQuery()
    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()