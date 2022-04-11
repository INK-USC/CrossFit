import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class AESLC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "aeslc"
        self.task_type = "text to text"
        self.license = "cc-by-sa-4.0"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("summarize: " + datapoint["email_body"].replace("\n", " ").replace("\r", " ").replace("\t", " "), datapoint["subject_line"].replace("\n", " ").replace("\r", " ").replace("\t", " ")))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("aeslc")

def main():
    dataset = AESLC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()