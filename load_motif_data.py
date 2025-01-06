import numpy as np


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
    x_train = normalize(np.loadtxt("./motif_data/" + cell_line + "_train_motif.txt"), cell_line, "train")
    y_train = np.loadtxt("./data/"+ cell_line + "_y_train.txt")
    x_test= normalize(np.loadtxt("./motif_data/" + cell_line + "_test_motif.txt"), cell_line, "test")
    y_test= np.loadtxt("./data/"+ cell_line + "_y_test.txt")
    assert x_train.shape == (x_train.shape[0], 401)
    assert x_test.shape == (x_test.shape[0], 401)
    assert y_train.shape == (x_train.shape[0], )
    assert y_test.shape == (x_test.shape[0], )
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    cell_line = "GM12878"
    x_train, y_train, x_test, y_test = load_data(cell_line)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)