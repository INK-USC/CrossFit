import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class NCBI_Disease(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ncbi_disease"
        self.task_type = "sequence tagging"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            assert len(datapoint["ner_tags"]) == len(datapoint["tokens"])
            lines.append((" ".join(datapoint["tokens"]), " ".join([str(item) for item in datapoint["ner_tags"]])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('ncbi_disease')

def main():
    dataset = NCBI_Disease()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()