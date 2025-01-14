
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

def mean_shap_values(shap_values, n, flatten=False):
    #print(shap_values.shape)
    if flatten:
        shap_values = shap_values.squeeze(axis=-2)
    abs_shap_values = np.abs(shap_values)

    # Calculate the mean of the absolute SHAP values across all classes and instances
    mean_abs_shap_values = np.mean(abs_shap_values, axis=0)
    labels = get_motif_labels()
    # Compute the 10 highest values and their indices in the 401D mean vector
    top_indices = np.argsort(mean_abs_shap_values)[-n:]  # Get indices of the 10 largest values
    top_values = mean_abs_shap_values[top_indices]     # Get the corresponding value
    # Combine indices and values, then sort in descending order
    top_sorted = sorted(zip(top_indices, top_values), key=lambda x: x[1], reverse=True)
    # Create a dictionary with indices as keys and values as the mean values
    top_dict_descending = {labels[int(index)]: round(float(value),3) for index, value in top_sorted}
    return top_dict_descending

def mean_shap_values_binary(shap_values, n):
    #print(shap_values.shape)
    # Step 1: Compute the absolute values
    abs_shap_values = np.abs(shap_values)

    # Step 2: Average over samples (axis=0)
    mean_abs_shap_per_sample = np.mean(abs_shap_values, axis=0)  # Shape: (401, 2)

    # Step 3: Sum (or average) over outputs (axis=1) to get one value per feature
    mean_abs_shap_per_feature = np.mean(mean_abs_shap_per_sample, axis=1)  # Shape: (401,)
    labels = get_motif_labels()
    # Compute the 10 highest values and their indices in the 401D mean vector
    top_indices = np.argsort(mean_abs_shap_per_feature)[-n:]  # Get indices of the 10 largest values
    top_values = mean_abs_shap_per_feature[top_indices]     # Get the corresponding value
    # Combine indices and values, then sort in descending order
    top_sorted = sorted(zip(top_indices, top_values), key=lambda x: x[1], reverse=True)
    # Create a dictionary with indices as keys and values as the mean values
    top_dict_descending = {labels[int(index)]: round(float(value),3) for index, value in top_sorted}
    return top_dict_descending
if __name__ == "__main__":
    shap_values = np.load("./Random_Forest_output/Best_Config_shap_values.npy")
    #mean_shap_values(shap_values, 10)
    top_dict_descending = mean_shap_values_binary(shap_values, 10)
    for key,val in top_dict_descending.items():
        print(f"{key} ({val})")
