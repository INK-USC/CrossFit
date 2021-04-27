import os
import datasets
import numpy as np

from full_dataset import FeltFullClassificationDataset

class Glue_MNLI(FeltFullClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-mnli"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        if split_name == "validation":
            split_name = "validation_matched"
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("premise: " + datapoint["premise"] + " [SEP] hypothesis: " + datapoint["hypothesis"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'mnli')

def main():
    dataset = Glue_MNLI()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_train_dev_test_data(seed=seed, path="../../data_more_shots/full")

if __name__ == "__main__":
    main()
