import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from itertools import product
import os
import numpy as np
from load_motif_data import load_dataset

def run_cross_validation_with_random_forest(dataset):
    """Run cross-validation with compatible hyperparameter combinations."""
    # Load dataset
    x_train, y_train, x_test, y_test = load_dataset(dataset)
    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],          # Number of trees in the forest
        "max_depth": [None, 10, 20, 30],        # Maximum depth of the tree
        "min_samples_split": [2, 5, 10],        # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4],          # Minimum number of samples required to be at a leaf node
        "max_features": ["sqrt", "log2", None], # Number of features to consider for the best split
        "bootstrap": [False],             # Whether bootstrap samples are used when building trees
    }

    # param_grid = {
    #     "n_estimators": [5],          # Number of trees in the forest
    #     "max_depth": [10],        # Maximum depth of the tree
    #     "min_samples_split": [2],        # Minimum number of samples required to split an internal node
    #     "min_samples_leaf": [8],          # Minimum number of samples required to be at a leaf node
    #     "max_features": ["sqrt"], # Number of features to consider for the best split
    #     "bootstrap": [False],             # Whether bootstrap samples are used when building trees
    # }

    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_split"],
        param_grid["min_samples_leaf"],
        param_grid["max_features"],
        param_grid["bootstrap"],
    ))

    results = []

    for combination in param_combinations:
        params = {
            "n_estimators": combination[0],
            "max_depth": combination[1],
            "min_samples_split": combination[2],
            "min_samples_leaf": combination[3],
            "max_features": combination[4],
            "bootstrap": combination[5],
        }
        print("Doing")
        print(params)
        try:
            # Create the model
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                bootstrap=params["bootstrap"],
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
    output_dir = "Random_Forest_Output"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "cross_validation_results.xlsx")
    results_df.to_excel(results_path, index=False)

    print(f"Results saved to {results_path}")
    return results_df

# Main function call
if __name__ == "__main__":
    run_cross_validation_with_random_forest("GM12878")
