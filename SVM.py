from evaluate import evaluate_model
from load_motif_data import load_dataset

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import sys

    


if __name__ == "__main__":
    cell_line = sys.argv[1]
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset(cell_line)

    # Model to train and evaluate
    #grid_search(X_train, y_train, X_test, y_test)
    # result: {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
    model = SVC(probability=True, C=100, gamma=1, kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    evaluate_model(model=model, name="SVM", config_name="C-100_gamma-1_kernel-rbf", X_train=X_train, X_test=X_test, y_test=y_test, output_dir=f"SVM_output/{cell_line}", sample_size=100, visualize=False)
