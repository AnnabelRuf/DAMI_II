from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from load_motif_data import load_dataset



def grid_search(X_train, y_train, X_test, y_test):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']}
    model = SVC()
    grid = GridSearchCV(model,param_grid)
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)
    accuracy_score(y_test, pred)
    
    print(classification_report(y_test,pred))
    print(grid.best_params_)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset("GM12878")

    # Model to train and evaluate
    #grid_search(X_train, y_train, X_test, y_test)
    # result: {'C': 100, 'gamma': 1, 'kernel': 'rbf'}