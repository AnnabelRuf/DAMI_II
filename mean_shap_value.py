import shap
import pandas as pd
import numpy as np
import re

def get_motif_labels():
    with open("./motif_data/HOCOMOCOv11_core_pwms_HUMAN_mono.txt") as f:
        pwms = f.read()
    pattern = r"\>(.*)_HUMAN"
    labels = re.findall(pattern, pwms)
    assert len(labels) == 401
    return labels
if __name__ == "__main__":
    NN =False 
    TOP_X = 20
    shap_values = np.load("./XGB_output/Standard_shap_values.npy")
    if NN:
        shap_values = shap_values.squeeze(axis=-1)
    abs_shap_values = np.abs(shap_values)

    # Calculate the mean of the absolute SHAP values across all classes and instances
    mean_abs_shap_values = np.mean(abs_shap_values, axis=0)
    labels = get_motif_labels()
    # Compute the 10 highest values and their indices in the 401D mean vector
    top_indices = np.argsort(mean_abs_shap_values)[-TOP_X:]  # Get indices of the 10 largest values
    top_values = mean_abs_shap_values[top_indices]     # Get the corresponding value
    # Combine indices and values, then sort in descending order
    top_sorted = sorted(zip(top_indices, top_values), key=lambda x: x[1], reverse=True)
    # Create a dictionary with indices as keys and values as the mean values
    top_dict_descending = {labels[int(index)]: round(float(value),3) for index, value in top_sorted}
    # print as such that it can be added to the table
    for key,val in top_dict_descending.items():
        print(f"{key} ({val})")