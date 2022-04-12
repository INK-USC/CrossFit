import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset
from utils import clean

class Spider(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "spider"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((clean(datapoint["question"]), clean(datapoint["query"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("spider")

def main():
    dataset = Spider()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()