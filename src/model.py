import numpy as np
import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.saving import save_model, load_model
from collector import Collector


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

class Model:
    def __init__(self, data=Collector(), model_name="full_alpha_model.keras"):
        self.data = data
        self.X = self.data.preprocess()[0]
        self.y = self.data.preprocess()[1]
        self.label_map = {label:num for num, label in enumerate(self.data.actions)}
        self.model = self.build_model()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.05)
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

    def train(self, epochs):
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30)
        model = self.model
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        model.fit(self.X_train, self.y_train, validation_data= (self.X_test, self.y_test), epochs=epochs, callbacks=[tb_callback, lr_callback])
        save_model(model, self.model_name, overwrite=True, save_format="tf")
        return model
    
    def evaluate(self):
        model = load_model(self.model_name, compile=True)
        model.summary()
        yhat = model.predict(self.X_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))
        inverse_label_map = {value:key for key, value in self.label_map.items()}
        ytrue = [inverse_label_map[i] for i in ytrue]
        yhat = [inverse_label_map[i] for i in yhat]
        data = confusion_matrix(ytrue, yhat)
        df_cm = pd.DataFrame(data, columns=np.unique(ytrue), index=np.unique(ytrue)) # data, columns=self.data.actions, index = self.data.actions
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize = (10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
        plt.show()
    