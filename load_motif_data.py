import numpy as np
import os


def get_length_of_sequence(cell_line, train_or_test):
    f = open("./data/" + cell_line + "_x_" + train_or_test + ".fasta", 'r')
    sequence = []
    num = []
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                seq = line.upper().strip('\n')
                sequence.append(line.upper().strip('\n'))
                num.append(len(seq))
    return  np.array(num).reshape(-1, 1)


def normalize(Vcount, cell_line, train_or_test):
    Ls = get_length_of_sequence(cell_line, train_or_test)
    return Vcount/Ls
    

def load_dataset(cell_line):
    try:
        train_motif_file = f"./motif_data/{cell_line}_train_motif.txt"
        y_train_file = f"./data/{cell_line}_y_train.txt"
        test_motif_file = f"./motif_data/{cell_line}_test_motif.txt"
        y_test_file = f"./data/{cell_line}_y_test.txt"

        # Check if files exist
        for file_path in [train_motif_file, y_train_file, test_motif_file, y_test_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load and process data
        x_train = normalize(np.loadtxt(train_motif_file), cell_line, "train")
        y_train = np.loadtxt(y_train_file)
        x_test = normalize(np.loadtxt(test_motif_file), cell_line, "test")
        y_test = np.loadtxt(y_test_file)

        # Validate shapes
        if x_train.shape[1] != 401:
            raise ValueError(f"x_train has an unexpected shape: {x_train.shape}")
        if x_test.shape[1] != 401:
            raise ValueError(f"x_test has an unexpected shape: {x_test.shape}")
        if y_train.shape[0] != x_train.shape[0]:
            raise ValueError(f"Mismatch between x_train and y_train: {x_train.shape[0]} != {y_train.shape[0]}")
        if y_test.shape[0] != x_test.shape[0]:
            raise ValueError(f"Mismatch between x_test and y_test: {x_test.shape[0]} != {y_test.shape[0]}")

        return x_train, y_train, x_test, y_test

    except Exception as e:
        print(f"Error loading dataset for cell line '{cell_line}': {e}")
        raise  # Re-raise the exception to ensure the program exits or can handle it at a higher level

if __name__ == "__main__":
    cell_line = "GM12878"
    x_train, y_train, x_test, y_test = load_dataset(cell_line)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)