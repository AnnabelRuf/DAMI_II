import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from itertools import product
import os
import numpy as np
import openpyxl

from load_motif_data import load_dataset

def is_combination_valid(params):
    """Check if the combination of hyperparameters is valid."""
    penalty = params["penalty"]
    solver = params["solver"]

    # Incompatible penalty-solver combinations
    incompatible_combinations = [
        ("l1", ["sag"]),                # 'sag' does not support 'l1'
        ("none", ["liblinear", "sag"]), # 'none' not supported by these solvers
        ("lbfgs", ["liblinear", "sag"]) # 'lbfgs' is not a valid penalty
    ]

    for pen, invalid_solvers in incompatible_combinations:
        if penalty == pen and solver in invalid_solvers:
            return False
    return True

def run_cross_validation_with_logistic_regression(dataset):
    """Run cross-validation with compatible hyperparameter combinations."""
    # Load dataset
    x_train, y_train, x_test, y_test = load_dataset(dataset)  # Assumes this function is already implemented

    # Define hyperparameter grid
    param_grid = {
        "penalty": ["l1", "l2", "none", "lbfgs"],
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "sag"],
        "max_iter": [100, 500, 1000],
    }

    # param_grid = {
    #     "penalty": ["l1"],
    #     "C": [0.01],
    #     "solver": ["liblinear"],
    #     "max_iter": [2],
    # }

    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid["penalty"],
        param_grid["C"],
        param_grid["solver"],
        param_grid["max_iter"]
    ))

    results = []

    for combination in param_combinations:
        params = {
            "penalty": combination[0],
            "C": combination[1],
            "solver": combination[2],
            "max_iter": combination[3],
        }
        print ("Doing: ")
        print(params)

        # Check if combination is valid
        if not is_combination_valid(params):
            results.append({**params, "mean_cv_accuracy": "INCOMPATIBLE", "std_cv_accuracy": None})
            continue

        try:
            # Create the model
            model = LogisticRegression(
                penalty=params["penalty"],
                C=params["C"],
                solver=params["solver"],
                max_iter=params["max_iter"],
                random_state=42
            )

            # Perform cross-validation on the training set
            cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")

            # Log the results
            results.append({
                **params,
                "mean_cv_accuracy": np.mean(cv_scores),
                "std_cv_accuracy": np.std(cv_scores),
            })

        except Exception as e:
            # Log any errors
            results.append({**params, "mean_cv_accuracy": "ERROR", "std_cv_accuracy": None})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to an Excel file
    output_dir = "Logistic_Regression_Output"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "cross_validation_results.xlsx")
    results_df.to_excel(results_path, index=False)

    print(f"Results saved to {results_path}")
    return results_df

# Main function call
if __name__ == "__main__":
    run_cross_validation_with_logistic_regression("GM12878")
