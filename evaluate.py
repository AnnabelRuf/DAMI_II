import shap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


# SHAP analysis
def shap_analysis(model, X_test, output_dir, model_name):
    # Use predict_proba for SVM
    if model_name == 'SVM':
        explainer = shap.Explainer(model.predict_proba, X_test, max_evals=2 * X_test.shape[1] + 1)

    else:
        explainer = shap.Explainer(model, X_test)

    
    shap_values = explainer(X_test)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SHAP values as a .npy file for numpy array
    shap_values_array = np.array([shap_values[i].values for i in range(len(shap_values))])
    np.save(os.path.join(output_dir, f"{model_name}_shap_values.npy"), shap_values_array)


    # Handle SHAP values and X_test compatibility
    if isinstance(X_test, np.ndarray):        
        column_names = [f'col{i+1}' for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=column_names) #,columns=model.feature_names_in_

    # For binary classification, select the SHAP values for class 1
    if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:  # Binary classification
        shap_values_binary = shap_values.values[..., 1]  # Use SHAP values for class 1
    else:
        shap_values_binary = shap_values.values

    # Save SHAP values to Excel
    shap_values_df = pd.DataFrame(shap_values_binary, columns=X_test.columns)
    shap_values_path = os.path.join(output_dir, f"{model_name}_shap_values.xlsx")
    shap_values_df.to_excel(shap_values_path, index=False)

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values_binary, X_test, show=False)
    shap_summary_path = os.path.join(output_dir, f"{model_name}_shap_summary_plot.jpg")
    plt.savefig(shap_summary_path)
    plt.close()

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values_binary, X_test, plot_type="bar", show=False)
    shap_bar_path = os.path.join(output_dir, f"{model_name}_shap_bar_plot.jpg")
    plt.savefig(shap_bar_path)
    plt.close()
    
def evaluate_model(model, name, X_train, X_test, y_test, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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
