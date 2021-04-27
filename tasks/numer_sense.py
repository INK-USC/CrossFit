import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class NumerSense(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "numer_sense"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_train_test_lines(self, dataset):

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["sentence"].replace("<mask>", "[MASK]"), datapoint["target"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('numer_sense')

def main():
    dataset = NumerSense()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()