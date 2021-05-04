import pandas as pd
import numpy as np
import argparse
import os

# python collect_results.py --logs_dir ../models/bart-base-meta-dev-apr4 --output_file ./results_summary/bart-base-meta-dev-apr4.csv
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logs_dir", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()

    df = pd.DataFrame(columns=["task", "entry", "dev_performance", "test_performance"])

    directories = os.listdir(args.logs_dir)
    for directory in sorted(directories):
        if directory.startswith("singletask"):
            task = directory[11:]
        else:
            task = directory

        if not os.path.exists(os.path.join(args.logs_dir, directory, "result.csv")):
            print("Something wrong with task {}\n\n".format(task))
            continue

        df0 = pd.read_csv(os.path.join(args.logs_dir, directory, "result.csv"))
        
        devs, tests = [], []
        
        for idx, row in df0.iterrows():
            if row["prefix"].endswith("_best"):
                df.loc[len(df.index)] = [task, row["prefix"][:-5], row["dev_performance"], row["test_performance"]]
                # print(row["prefix"], row["dev_performance"], row["test_performance"])
                devs.append(row["dev_performance"])
                tests.append(row["test_performance"])
        
        if len(devs) > 0:
            df.loc[len(df.index)] = [task, "mean", np.mean(devs), np.mean(tests)]
            df.loc[len(df.index)] = [task, "std", np.std(devs), np.std(tests)]
            df.loc[len(df.index)] = ["", "", "", ""]

    df.to_csv(args.output_file)
    # print(directories)

if __name__ == "__main__":
    main()