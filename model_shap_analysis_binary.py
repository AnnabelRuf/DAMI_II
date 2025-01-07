import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import load_motif_data


# Load dataset
def load_data(file_path):
    # Replace with actual file path and dataset
    data = pd.read_csv(file_path)  # Placeholder
    return data



# Train-test split
def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Train models and evaluate
def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save classification report to Excel
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(output_dir, f"{name}_classification_report.xlsx")
        report_df.to_excel(report_path, index=True)

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
        plt.savefig(plt_path)
        plt.close()

        print(f"Model: {name}")
        print(f"Accuracy: {acc}")
        print(classification_report(y_test, y_pred))

        # Perform SHAP analysis
        shap_analysis(model, X_train, output_dir, name)


# SHAP analysis
def shap_analysis(model, X_train, output_dir, model_name):
    # Use predict_proba for SVM
    if model_name == 'SVM':
        explainer = shap.Explainer(model.predict_proba, X_train, max_evals=2 * X_train.shape[1] + 1)

    else:
        explainer = shap.Explainer(model, X_train)

    
    shap_values = explainer(X_train)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SHAP values as a .npy file for numpy array
    shap_values_array = np.array([shap_values[i].values for i in range(len(shap_values))])
    np.save(os.path.join(output_dir, f"{model_name}_shap_values.npy"), shap_values_array)


    # Handle SHAP values and X_train compatibility
    if isinstance(X_train, np.ndarray):        
        column_names = [f'col{i+1}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=column_names) #,columns=model.feature_names_in_

    # For binary classification, select the SHAP values for class 1
    if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:  # Binary classification
        shap_values_binary = shap_values.values[..., 1]  # Use SHAP values for class 1
    else:
        shap_values_binary = shap_values.values

    # Save SHAP values to Excel
    shap_values_df = pd.DataFrame(shap_values_binary, columns=X_train.columns)
    shap_values_path = os.path.join(output_dir, f"{model_name}_shap_values.xlsx")
    shap_values_df.to_excel(shap_values_path, index=False)

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values_binary, X_train, show=False)
    shap_summary_path = os.path.join(output_dir, f"{model_name}_shap_summary_plot.jpg")
    plt.savefig(shap_summary_path)
    plt.close()

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values_binary, X_train, plot_type="bar", show=False)
    shap_bar_path = os.path.join(output_dir, f"{model_name}_shap_bar_plot.jpg")
    plt.savefig(shap_bar_path)
    plt.close()




# Main script
if __name__ == "__main__":
    output_dir = "model_outputs"  # Directory to save outputs

    # Train-test split
    x_train, y_train, x_test, y_test = load_motif_data.load_dataset("GM12878")

    # Train and evaluate models
    train_and_evaluate_models(x_train, x_test, y_train, y_test, output_dir)
