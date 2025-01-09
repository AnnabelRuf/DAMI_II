from load_motif_data import load_dataset
from evaluate import evaluate_model

from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam

# Implementation according to https://www.kaggle.com/code/phamvanvung/shap-for-lstm/notebook
def create_lstm(l1Nodes, l2Nodes, d1Nodes, d2Nodes, inputShape):
    # input layer
    lstm1 = LSTM(l1Nodes, input_shape=inputShape, return_sequences=True)
    lstm2 = LSTM(l2Nodes, return_sequences=True)
    flatten = Flatten()
    dense1 = Dense(d1Nodes)
    dense2 = Dense(d2Nodes)

    # output layer, changed from relu for classification
    outL = Dense(1, activation='sigmoid')
    # combine the layers
    layers = [lstm1, lstm2, flatten,  dense1, dense2, outL]
    # create the model
    model = Sequential(layers)
    opt = Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model
    

if __name__ == "__main__":

    # Train-test split
    X_train, y_train, X_test, y_test = load_dataset("GM12878")
    # Reshape to fit LSTMs
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = create_lstm(8, 8, 8, 4, (X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=8, epochs=1)
    evaluate_model(model, "LSTM", "Standard", X_train, X_test, y_test, output_dir="LSTM_output", sample_size=50)
    
    