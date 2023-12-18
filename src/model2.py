import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.saving import save_model, load_model
from collector import Collector


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

class Model:
    def __init__(self, data=Collector(), model_name="model_only_hands.keras"):
        self.data = data
        self.X = self.data.preprocess()[0]
        self.y = self.data.preprocess()[1]
        self.model = self.build_model()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.05)
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.model_name = model_name
        

    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.y.shape[1], activation='softmax'))
        return model

    def train(self):
        model = self.model
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        model.fit(self.X_train, self.y_train, validation_data= (self.X_test, self.y_test), epochs=2000, callbacks=[tb_callback])
        save_model(model, "model_only_hands.keras", overwrite=True, save_format="tf")
        return model
    
    def evaluate(self):
        model = load_model(self.model_name, compile=True)  # load_model('action.h5')
        model.summary()
        yhat = model.predict(self.X_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))
    