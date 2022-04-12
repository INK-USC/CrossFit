import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset
from utils import clean

class Kilt_Fever(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "kilt_fever"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((clean(datapoint["input"]), "\t".join(list(set([item["answer"] for item in datapoint["output"]])))))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('kilt_tasks','fever')

def main():
    dataset = Kilt_Fever()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()