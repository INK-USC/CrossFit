import os
import datasets
import numpy as np

from full_dataset import FeltFullTextToTextDataset

class ROPES(FeltFullTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ropes"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = "question: " + datapoint["question"] + " [SEP] situation: " + datapoint["situation"] + " [SEP] background: " + datapoint["background"]
            output_text = "\t".join(datapoint["answers"]["text"])
            lines.append((input_text.replace("\n", " "), output_text))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("ropes")

def main():
    dataset = ROPES()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_train_dev_test_data(seed=seed, path="../../data_more_shots/full")


def main_more_shots():
    dataset = ROPES()

    for shot in [2048, 4096]:
        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_train_dev_test_data(seed=seed, n_train=shot, path="../../data_more_shots/{}_shot-1024dev".format(shot))


if __name__ == "__main__":
    # main()
    main_more_shots()