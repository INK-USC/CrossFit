import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class WikiMovies(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "wiki_movies"
        self.task_type = "text to text"
        self.license = "unknown"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((datapoint["question"], datapoint["answer"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("wiki_movies")

def main():
    dataset = WikiMovies()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()