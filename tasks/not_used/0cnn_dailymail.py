import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class CNNDailyMail(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "cnn_dailymail"
        self.task_type = "text to text"
        self.license = "apache-2.0"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("summarize: " + datapoint["article"], datapoint["highlights"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('cnn_dailymail', '3.0.0')

def main():
    dataset = CNNDailyMail()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()