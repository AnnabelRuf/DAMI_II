from evaluate import evaluate_model
from load_motif_data import load_dataset

from sklearn.ensemble import RandomForestClassifier
import sys



if __name__ == "__main__":
    cell_line = sys.argv[1]
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset(cell_line)

    # Model to train and evaluate
    RF_configs = {
        'Standard': RandomForestClassifier(random_state=42),
        'Best_Config': RandomForestClassifier(random_state=42, n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=False )
    }


    for config_name, model in RF_configs.items():
        #Train model
        model.fit(X_train, y_train)
        evaluate_model(model=model, name="RF", config_name=config_name, X_train=X_train, X_test=X_test, y_test=y_test, output_dir=f"Random_Forest_output")
