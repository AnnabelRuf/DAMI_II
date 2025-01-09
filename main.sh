#!/bin/bash

# Check if a parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 cell_line"
    exit 1
fi

# Assign the parameter to a variable
CELL_LINE=$1

# Execute the Python scripts with the parameter
python3 logistic_regression.py "$CELL_LINE"
python3 random_forest.py "$CELL_LINE"
python3 XGBoost.py "$CELL_LINE"
python3 NN.py "$CELL_LINE"
python3 SVM.py "$CELL_LINE"
