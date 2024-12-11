from pathlib import Path
import re 
import numpy as np
import itertools
from collections import Counter 

def parse_sequences(text):
    pattern_pos = r">Positive.*\n([A|T|C|G|a|t|c|g]*)"
    pos_matches = re.findall(pattern_pos, text)
    pattern_neg = r">Negative.*\n([A|T|C|G|a|t|c|g]*)"
    neg_matches = re.findall(pattern_neg, text)

    labels = ["Positive"]*len(pos_matches) + ["Negative"] * len(neg_matches)
    return pos_matches+neg_matches, labels

def load_motives():
    with open("data/PWMs_motif_mono_human.txt") as f:
        pwms_data = f.read()
    regex = r">(.*)_HUMAN.*\n((?:[-|\d|\.|e]*\t[-|\d|\.|e]*\t[-|\d|\.|e]*\t[-|\d|\.|e]*\n)*)"
    matches = re.findall(regex, pwms_data)
    motives = []
    pwms=[]
    for match in matches:
        motives.append(match[0])
        rows = match[1].split("\n")
        m = [list(map(float, row.split("\t"))) for row in rows if row.strip() != '']
        pwms.append(np.matrix(m))
    return motives, pwms


def motif_frequency(pwm_matrices, sequence):
    Vcount = np.zeros(401)
    Pvalue = 10^-4
    # map the base to a position in the PWM matrix
    base_map = {"A":0, "C":1, "G":2, "T":3}
    Ls = len(sequence)
    # For each PWM matix equivalent to a motiv
    for z, pwm in enumerate(pwm_matrices):
        # get the motif length
        Lm = len(pwm)
        i=0
        # sliding window with size Lm and stride 1 over the sequence 
        while i + Lm <= Ls:
            Qvalue=0
            # segment to test if it matches a motif
            segment = sequence[i:i+Lm] 
            # calculate Q value as the sum of the PWM entries depending on position and base
            for j,base in enumerate(segment):
                Qvalue += pwm[j, base_map[base]]
            # if Q > P the sequence matches the motif
            if Qvalue > Pvalue:
                Vcount[z] +=1
            i+=1
    Vfrequency = Vcount/Ls
    return Vfrequency



if __name__ == "__main__":
    dataset_ind_path = Path("data/GM128-Ind.txt")
    dataset_path= Path("data/GM12878.txt")

    with open(dataset_ind_path) as f1, open(dataset_path) as f2:
        dataset_ind_unprocessed = f1.read()
        dataset_unprocessed = f2.read()
    train_dataset, train_labels = parse_sequences(dataset_unprocessed)
    independent_dataset, ind_labels = parse_sequences(dataset_ind_unprocessed)
    
    motives, pwm_matrices = load_motives()
    train_dataset_encoded = []
    for sequence in itertools.islice(train_dataset,1,3):
        train_dataset_encoded.append((motif_frequency(pwm_matrices, sequence)))
    np.savez("data/GM12878_encoded", data=train_dataset_encoded, labels=train_labels)

    independent_dataset_encoded = []
    for sequence in itertools.islice(independent_dataset,1,3):
        independent_dataset_encoded.append((motif_frequency(pwm_matrices, sequence)))
    np.savez("data/GM128-Ind_encoded", data=independent_dataset_encoded, labels=ind_labels)
    


