from pathlib import Path

from preprocess import label_sequences
from lime import *

if __name__ == "__main__":
    dataset_ind_path = Path("data/GM128-Ind.txt")
    dataset_path= Path("data/GM12878.txt")

    with open(dataset_ind_path) as f1, open(dataset_path) as f2:
        dataset_ind_unprocessed = f1.read()
        dataset_unprocessed = f2.read()
    dataset_ind_processed = label_sequences(dataset_ind_unprocessed)
    dataset_processed = label_sequences(dataset_unprocessed)