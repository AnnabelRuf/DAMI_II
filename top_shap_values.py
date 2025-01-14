import numpy as np
from mean_shap_value import *
from collections import Counter
import re

N = 10

XGB_shap_dict = {
    "GM12878 best config": mean_shap_values(np.load("./XGB_output/GM12878/Best_Config_shap_values.npy"), N),
    "GM12878 standard config": mean_shap_values(np.load("./XGB_output/GM12878/Standard_shap_values.npy"),N),
    "HMEC best config": mean_shap_values(np.load("./XGB_output/HMEC/Best_Config_shap_values.npy"), N),
    "HMEC standard config": mean_shap_values(np.load("./XGB_output/HMEC/Standard_shap_values.npy"), N)
}
RF_shap_dict = {
    "GM12878 best config": mean_shap_values_binary(np.load("./Random_Forest_output/GM12878/Best_Config_shap_values.npy"), N),
    "GM12878 standard config": mean_shap_values_binary(np.load("./Random_Forest_output/GM12878/Standard_shap_values.npy"), N),
    "HMEC best config": mean_shap_values_binary(np.load("./Random_Forest_output/HMEC/Best_Config_shap_values.npy"), N),
    "HMEC standard config": mean_shap_values_binary(np.load("./Random_Forest_output/HMEC/Standard_shap_values.npy"), N)
}
LR_shap_dict = {
    "GM12878 best config": mean_shap_values(np.load("./Logistic_Regression_output/GM12878/Best_Config_shap_values.npy"), N),
    "GM12878 standard config": mean_shap_values(np.load("./Logistic_Regression_output/GM12878/Standard_shap_values.npy"), N),
    "HMEC best config": mean_shap_values(np.load("./Logistic_Regression_output/HMEC/Best_Config_shap_values.npy"), N),
    "HMEC standard config": mean_shap_values(np.load("./Logistic_Regression_output/HMEC/Standard_shap_values.npy"), N)
}
GM12878_shap_values = []
GM12878_pattern = r"GM12878.*"
HMEC_shap_values = []
HMEC_pattern = r"HMEC.*"

XGB_shap_values = []
for name, dict in XGB_shap_dict.items():
    XGB_shap_values += dict.keys()
    if re.match(GM12878_pattern, name):
        GM12878_shap_values+= dict.keys()
    if re.match(HMEC_pattern, name):
        HMEC_shap_values+= dict.keys()
    
assert len(XGB_shap_values) == len(XGB_shap_dict.keys()) * N
XGB_counter = Counter(XGB_shap_values)

RF_shap_values = []
for name, dict in RF_shap_dict.items():
    RF_shap_values+= dict.keys()
    if re.match(GM12878_pattern, name):
        GM12878_shap_values+= dict.keys()
    if re.match(HMEC_pattern, name):
        HMEC_shap_values+= dict.keys()
assert len(RF_shap_values) == len(RF_shap_dict.keys()) * N
RF_counter = Counter(RF_shap_values)

LR_shap_values = []
for name, dict in LR_shap_dict.items():
    LR_shap_values+= dict.keys()
    if re.match(GM12878_pattern, name):
        GM12878_shap_values+= dict.keys()
    if re.match(HMEC_pattern, name):
        HMEC_shap_values+= dict.keys()
assert len(LR_shap_values) == len(LR_shap_dict.keys()) * N
LR_counter = Counter(LR_shap_values)

total_counter = Counter(XGB_shap_values + RF_shap_values + LR_shap_values)

GM12878_counter = Counter(GM12878_shap_values)
HMEC_counter = Counter(HMEC_shap_values)


with open("shap_analysis/top_shap_vals.txt", "wt") as f:
    f.write("Total motifs:\n")
    f.write(str(total_counter))
    f.write("\nXGB motifs:\n")
    f.write(str(XGB_counter))
    f.write("\nRF motifs:\n")
    f.write(str(RF_counter))
    f.write("\nLR motifs:\n")
    f.write(str(LR_counter))
    f.write("\nMotifs in GM12878:\n")
    f.write(str(GM12878_counter))
    f.write("\nMotifs in HMEC:\n")
    f.write(str(HMEC_counter))