import shap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from load_motif_data import load_dataset
from mean_shap_value import get_motif_labels
import sys

sys.path.append('../DAMI_II')

DATASET = "GM12878"

X_train, y_train, X_test, y_test = load_dataset(DATASET)
labels = get_motif_labels()

XGB_optimized= np.load(f"./XGB_output/{DATASET}/Best_Config_shap_values.npy")
XGB_default =  np.load(F"./XGB_output/{DATASET}/Standard_shap_values.npy")

RF_optimized= np.load(f"./Random_Forest_output/{DATASET}/Best_Config_shap_values.npy")
RF_default =  np.load(F"./Random_Forest_output/{DATASET}/Standard_shap_values.npy")
#Flatten RF data 
flattened_RF_optimized = np.mean(RF_optimized, axis=-1)
flattened_RF_default = np.mean(RF_default, axis=-1)

LR_optimized= np.load(f"./Logistic_Regression_output/{DATASET}/Best_Config_shap_values.npy")
LR_default =  np.load(F"./Logistic_Regression_output/{DATASET}/Standard_shap_values.npy")

NN_small = np.load(f"./NN_output/{DATASET}/Small_shap_values.npy")
NN_big =  np.load(F"./NN_output/{DATASET}/Big_shap_values.npy")
#flatten NN data
flattened_NN_small = np.squeeze(NN_small, axis=-1)#
flattened_NN_big = np.squeeze(NN_big, axis=-1)

models = [
    (XGB_optimized, "XGB Optimized Configuration"),
    (XGB_default, "XGB Default Configuration"),
    (flattened_RF_optimized, "RF Optimized Configuration"),
    (flattened_RF_default, "RF Default Configuration"),
    (LR_optimized, "LR Optimized Configuration"),
    (LR_default, "LR Default Configuration"),
    (flattened_NN_small, "NN Small Model"),
    (flattened_NN_big, "NN Big Model")
]

# List of model names for use in file names
model_names = [
    "XGB_optimized", "XGB_default", "RF_optimized", "RF_default", 
    "LR_optimized", "LR_default", "NN_small", "NN_big"
]

# Create individual SHAP summary plots and save them as separate images
for (model, title), model_name in zip(models, model_names):
    # Create a new figure for each SHAP plot
    plt.figure(figsize=(12, 6))  # Adjust the size for each individual plot
    
    # Generate the SHAP summary plot for the model
    if model_name == "NN_small" or model_name == "NN_big":
        shap.summary_plot(model, shap.sample(X_test, 500), feature_names=labels, show=False)
    else:
        shap.summary_plot(model, X_test, feature_names=labels, show=False)
    
    plt.title(title)
    plt.xlabel("SHAP value")
    
    # Save the figure
    plt.savefig(f"shap_analysis/{DATASET}_shap_plot_{model_name}.png", bbox_inches='tight', dpi=300)
    
    # Close the plot to avoid overlap when generating the next plot
    plt.close()