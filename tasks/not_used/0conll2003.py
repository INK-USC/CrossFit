import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class CoNLL2003(FewshotGymTextToTextDataset):

    def __init__(self, subset_identifier):
        self.hf_identifier = "conll2003-" + subset_identifier
        self.subset_identifier = subset_identifier
        self.task_type = "sequence tagging"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        key = self.subset_identifier + "_tags"
        for datapoint in hf_dataset[split_name]:
            assert len(datapoint[key]) == len(datapoint["tokens"])
            lines.append((" ".join(datapoint["tokens"]), " ".join([str(item) for item in datapoint[key]])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('conll2003')

def main():
    for tag in ["ner", "pos", "chunk"]:
        dataset = CoNLL2003(tag)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()