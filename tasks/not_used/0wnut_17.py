import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}

class WNut17(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "wnut_17"
        self.task_type = "sequence tagging"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):

        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = " ".join(datapoint["tokens"])
            output_text = " ".join([str(item) for item in datapoint["ner_tags"]])
            lines.append((input_text, output_text))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("wnut_17")

def main():
    dataset = WNut17()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()