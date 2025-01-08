import tensorflow as tf
import shap
from tensorflow.keras.layers import Dense, Dropout

from load_motif_data import load_dataset
from evaluate import evaluate_model

def create_model(size_l1, size_l2, size_l3,dropout):
    model = tf.keras.Sequential([
        Dense(size_l1, activation='relu'),
        Dropout(dropout),
        Dense(size_l3, activation='relu'),
        Dropout(dropout),
        Dense(size_l3, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = create_model(128,64,16,0.5)
    X_train, y_train, X_test, y_test = load_dataset("GM12878")
    model.fit(X_train,y_train,  epochs = 1, batch_size = 8)
    evaluate_model(model, "NN", "Standard", X_train, X_test, y_test, output_dir="NN_output", sample_size=500)
