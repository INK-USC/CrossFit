import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset
from utils import clean

class SciCite(FewshotGymClassificationDataset):

    def __init__(self):
        self.hf_identifier = "scicite"
        self.task_type = "classification"
        self.license = "unknown"

        self.label = {
            0:"method",
            1:"background",
            2:"result",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = datapoint["string"][datapoint["citeStart"]: datapoint["citeEnd"]] + " [SEP] " + datapoint["string"].replace("\n", " ") + " [SEP] " + datapoint["sectionName"]
            lines.append((clean(input_text), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("scicite")

def main():
    dataset = SciCite()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()