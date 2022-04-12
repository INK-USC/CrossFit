import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset
from utils import clean

class TweetQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "tweet_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input_text = "question: " + datapoint["Question"] + " context: " + datapoint["Tweet"]
            output_text = "\t".join([clean(item) for item in datapoint["Answer"]])
            lines.append((clean(input_text), output_text))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("tweet_qa")

def main():
    dataset = TweetQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()