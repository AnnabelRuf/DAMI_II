
from evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import load_motif_data



if __name__ == "__main__":
    output_dir = "model_outputs"  # Directory to save outputs

    # Train-test split
    X_train, y_train, X_test, y_test = load_motif_data.load_dataset("GM12878")

    # Model to train and evaluate
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }


    for name, model in models.items():
        #Train model
        model.fit(X_train, y_train)
        evaluate_model(model=model, name=name, X_train=X_train, X_test=X_test, y_test=y_test, output_dir=output_dir)
