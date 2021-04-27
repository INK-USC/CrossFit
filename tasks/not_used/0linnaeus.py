import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Linnaeus(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "linnaeus"
        self.task_type = "sequence tagging"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if len(datapoint["tokens"]) < 10:
                continue
            assert len(datapoint["tokens"]) == len(datapoint["ner_tags"])
            datapoint["ner_tags"] = [str(item) for item in datapoint["ner_tags"]]
            lines.append((" ".join(datapoint["tokens"]), " ".join(datapoint["ner_tags"])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('linnaeus')

def main():
    dataset = Linnaeus()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()