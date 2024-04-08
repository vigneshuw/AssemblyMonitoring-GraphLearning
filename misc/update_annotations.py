import os
import sys

import pandas as pd


if __name__ == "__main__":

    annotations_path = "/home/vselvaraj/projects/action_localization/DATA/AssemblyDemo/cycles/testing"

    for file_name in os.listdir(annotations_path):

        # Only for CSV files
        if file_name.split(".")[-1] != "csv":
            continue

        # Read the information and remove the column
        df = pd.read_csv(os.path.join(annotations_path, file_name), usecols=["task", "startTime", "endTime"])

        df.to_csv(os.path.join(annotations_path, file_name), index=False)

    print("All files has been processed")
