import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Fever(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "fever"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"REFUTES",
            1:"SUPPORTS",
        }
    
    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "labelled_dev")

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if datapoint["label"] == "SUPPORTS":
                # line[0]: input; line[1]: output
                lines.append((datapoint["claim"].strip(), "Supports"))
            elif datapoint["label"] == "REFUTES":
                lines.append((datapoint["claim"].strip(), "Refutes"))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('fever', 'v1.0')

def main():
    dataset = Fever()
    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()