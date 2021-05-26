from os import name
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Flatten, Dense, InputLayer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.math import confusion_matrix

def question1():
    (in_train, out_train), (in_test, out_test) = mnist.load_data()
    
    model = Sequential()
    model.add(InputLayer(input_shape=(28, 28)))
    model.add(Flatten())
    model.add(Dense(392, name="oculta1"))
    model.add(Dense(196, name="oculta2"))
    model.add(Dense(98, name="oculta3"))
    model.add(Dense(49, name="oculta4"))
    model.add(Dense(25, name="oculta5"))
    model.add(Dense(10, name="saida", activation="softmax"))

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(in_train,
                out_train,
                epochs=500,
                batch_size=256,
                validation_split = 0.2)
    
    model.save("lista4-q1.h5")
    
    #model = load_model("lista4-q1.h5")
    test_result_onehot = model.predict(in_test)
    test_result = list()

    for result_array in test_result_onehot:
        test_result.append(np.argmax(result_array))

    matrix = confusion_matrix(out_test, test_result)

    plt.title("Matriz de confusão dos resultados do teste")
    sn.heatmap(np.array(matrix), annot=True, cmap="Blues", fmt="g")
    plt.xlabel("Resultado da predição")
    plt.ylabel("Classificação correta")
    plt.savefig("lista4-mc.jpg")
    plt.clf()

    plt.title("Evolução da acurácia em cada dataset")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(['treino', 'validação'], loc='upper left')
    plt.savefig("lista4-acc.jpg")
    plt.clf()

    plt.title("Evolução do erro em cada dataset")
    plt.xlabel("Época")
    plt.ylabel("Erro")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['treino', 'validação'], loc='upper left')
    plt.savefig("lista4-loss.jpg")
    plt.clf()

if __name__ == "__main__":
    question1()