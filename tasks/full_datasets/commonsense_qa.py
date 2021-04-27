import os
import datasets
import numpy as np

from full_dataset import FeltFullTextToTextDataset

class CommonsenseQA(FeltFullTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "commonsense_qa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["choices"]["label"])):
            if datapoint["choices"]["label"][i] == answer_index:
                answer_string = datapoint["choices"]["text"][i]
            choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("commonsense_qa")

def main():
    dataset = CommonsenseQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_train_dev_test_data(seed=seed, path="../../data_more_shots/full")

def main_more_shots():
    dataset = CommonsenseQA()

    for shot in [2048, 4096]:
        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_train_dev_test_data(seed=seed, n_train=shot, path="../../data_more_shots/{}_shot-1024dev".format(shot))

if __name__ == "__main__":
    # main()
    main_more_shots()