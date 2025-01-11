import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import sys

from load_motif_data import load_dataset
from evaluate import evaluate_model

def create_model(size_l1, size_l2, size_l3):
    model = tf.keras.Sequential([
        Dense(size_l1, activation='relu'),
        Dense(size_l3, activation='relu'),
        Dense(size_l3, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])
    return model

if __name__ == "__main__":
    NN_params = {
        "Small": create_model(128,64,32),
        "Big": create_model(256,128,64)
    }
    cell_line = sys.argv[1]
    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset(cell_line)
    for name, model in NN_params.items():
        model.fit(X_train,y_train,  epochs = 30, batch_size = 8)
        evaluate_model(model, "NN", name, X_train, X_test, y_test, output_dir=f"NN_output/{cell_line}", sample_size=500)
