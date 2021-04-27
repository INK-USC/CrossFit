import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Kilt_ELI5(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "kilt_eli5"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["input"].replace("\n", " "), datapoint["output"][0]["answer"].replace("\n", " ")))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('kilt_tasks','eli5')

def main():
    dataset = Kilt_ELI5()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()