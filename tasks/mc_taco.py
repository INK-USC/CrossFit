import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class MC_TACO(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "mc_taco"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "no",
            1: "yes",
        }
    
    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "validation")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            input_text = datapoint["sentence"] + " [SEP] " + datapoint["question"] + " [SEP] " + datapoint["answer"]
            lines.append((input_text, self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('mc_taco')

def main():
    dataset = MC_TACO()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()