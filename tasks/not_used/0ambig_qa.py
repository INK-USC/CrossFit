import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymTextToTextDataset

class AmbigQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ambig_qa"
        self.task_type = "text to text"
        self.license = "cc-by-sa-3.0"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            if len(datapoint["annotations"]["qaPairs"]) != 1:
                continue

            concat_answer = ""
            if len(datapoint["annotations"]["qaPairs"][0]["answer"]) == 0:
                continue
            for ans in datapoint["annotations"]["qaPairs"][0]["answer"]:
                concat_answer += ans[0] + " [SEP] "

            lines.append((datapoint["question"].replace("\n", " "), concat_answer[:-7])) # remove the last [SEP]
        return lines

    def load_dataset(self):
        return datasets.load_dataset("ambig_qa", "light")

def main():
    dataset = AmbigQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()