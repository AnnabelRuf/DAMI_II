from evaluate import evaluate_model
from load_motif_data import load_dataset

from sklearn.linear_model import LogisticRegression



if __name__ == "__main__":


    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset("GM12878")

    # Model to train and evaluate
    LR_configs = {
        'Standard' : LogisticRegression(max_iter=1000, random_state=42)
    }


    for config_name, model in LR_configs.items():
        #Train model
        model.fit(X_train, y_train)
        evaluate_model(model=model, name="LR", config_name=config_name, X_train=X_train, X_test=X_test, y_test=y_test, output_dir=f"Logistic_Regression_output")
