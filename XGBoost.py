from evaluate import evaluate_model
from load_motif_data import load_dataset

from xgboost.sklearn import XGBClassifier
import sys



if __name__ == "__main__":
    cell_line = sys.argv[1]
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset(cell_line)
    model = XGBClassifier()
    LR_configs = {
        'Standard' : XGBClassifier(),
        "Best_Config" : XGBClassifier(eval_metric='mlogloss', learning_rate =  0.1, max_depth= 3, subsample = 1)
    }
    for config_name, model in LR_configs.items():
        model.fit(X_train, y_train)
        evaluate_model(model=model, name="XGB", config_name=config_name, X_train=X_train, X_test=X_test, y_test=y_test,output_dir=f"XGB_output/{cell_line}")