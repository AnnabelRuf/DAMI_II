from pathlib import Path
import re 
from sklearn.model_selection import train_test_split
from collections import Counter

def parse_sequences(text):
    pattern = r">chr.*\n([A|T|C|G|a|t|c|g]*)"
    matches = re.findall(pattern, text)
    assert len(matches) in {742, 1484}, f"Unexpected length: {len(matches)}"
    for sequence in matches:
        assert len(sequence) == 200, f"Unexpected sequence length: {len(matches)}"
    return matches

def label_data(data, label):
    dict = {}
    for seq in data:
        dict[seq] = label
    return dict 

def create_test_train_split(strong_enhancers_file, weak_enhancers_file, non_enhancers_file):
    strong_enhancers = parse_sequences(strong_enhancers_file)
    weak_enhancers = parse_sequences(weak_enhancers_file)
    enhancers = strong_enhancers + weak_enhancers
    non_enhancers = parse_sequences(non_enhancers_file)
    assert len(enhancers) == len(non_enhancers)
    labeled_enhancer = label_data(enhancers,"Enhancer")
    labeled_non_enhancer = label_data(non_enhancers,"Non")
    dataset = {**labeled_enhancer, **labeled_non_enhancer}
    return train_test_split(list(dataset.keys()), list(dataset.values()), test_size=0.2, random_state=42)

if __name__ == "__main__":
    non_enhancer_path = Path("data/iEnhancer_2L_non_enhancers.txt")
    strong_enhancer_path= Path("data/iEnhancer_2L_strong_enhancers.txt")
    weak_enhancer_path= Path("data/iEnhancer_2L_weak_enhancers.txt")

    with open(non_enhancer_path) as f1, open(strong_enhancer_path) as f2, open(weak_enhancer_path) as f3:
        non_enhancers_file = f1.read()
        strong_enhancers_file = f2.read()
        weak_enhancers_file = f3.read()
    X_train, X_test, y_train, y_test = create_test_train_split(strong_enhancers_file, weak_enhancers_file, non_enhancers_file)

