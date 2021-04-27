import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class JNLPBA(FewshotGymTextToTextDataset):

    # 0:"B-long", 1:"B-short", 2:"I-long", 3:"I-short", 4:"O"

    def __init__(self):
        self.hf_identifier = "jnlpba"
        self.task_type = "sequence tagging"
        self.license = ""

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            assert len(datapoint["ner_tags"]) == len(datapoint["tokens"])
            lines.append((" ".join(datapoint["tokens"]), " ".join(datapoint["ner_tags"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('jnlpba')

def main():
    dataset = JNLPBA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()