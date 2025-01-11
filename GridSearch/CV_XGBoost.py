from evaluate import evaluate_model
from load_motif_data import load_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def grid_search_xgboost(X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }
    
    model = XGBClassifier(eval_metric='mlogloss')
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)
    
    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    print(grid.best_params_)

if __name__ == "__main__":
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset("GM12878")

    # Run grid search for XGBoost
    #grid_search_xgboost(X_train, y_train, X_test, y_test)

    # result: {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1}
    model = XGBClassifier(eval_metric='mlogloss', learning_rate =  0.1, max_depth= 3, subsample = 1)
    model.fit(X_train, y_train)
    evaluate_model(model=model, name="XGB", config_name="Best_GS_XGBoost", X_train=X_train, X_test=X_test, y_test=y_test, output_dir=f"XGB_output", sample_size=500)

