from evaluate import evaluate_model
from load_motif_data import load_dataset

from sklearn.linear_model import LogisticRegression
import sys



if __name__ == "__main__":
    cell_line = sys.argv[1]
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset(cell_line)

    # Model to train and evaluate
    LR_configs = {
        'Standard' : LogisticRegression(max_iter=1000, random_state=42),
        "Best_Config" : LogisticRegression(max_iter=1000, C=100, solver="sag", penalty="l2")
    }


    for config_name, model in LR_configs.items():
        #Train model
        model.fit(X_train, y_train)
        evaluate_model(model=model, name="LR", config_name=config_name, X_train=X_train, X_test=X_test, y_test=y_test, output_dir=f"Logistic_Regression_output/{cell_line}")
