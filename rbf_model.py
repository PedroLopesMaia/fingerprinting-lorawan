import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from kmeans_initializer import InitCentersKMeans
from rbflayer import RBFLayer
from functions import getPerformance, normalizer, positive, getEvaluation, importarDados, transformarDados

seed = 260623
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISBLE_DEVICE'] = ''
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#Separação dos subconjuntos dos dados
data_1 = importarDados('lorawan_antwerp_2019_dataset_com_id.csv')
lista = list(data_1.columns[:44])
lista.append('sf')
X_1 = data_1[lista]
y_1 = data_1[['latitude', 'longitude']]

n_data = 120000
train_data, dev_data, test_data, y_train, y_dev, y_test = transformarDados(X_1.iloc[:n_data], y_1.iloc[:n_data], seed)

epochs=1
batch_size=500
learning_rate=0.0005
betas=0.75
n_units=1
initializer=InitCentersKMeans(train_data)

mae = tf.keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error")
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

n_inputs, n_outputs = train_data.shape[1], y_train.shape[1]

rbflayer = RBFLayer(n_units,
                    initializer=initializer,
                    betas=betas,
                    input_shape=(n_inputs,))

model = Sequential()
model.add(rbflayer)
model.add(Dense(n_outputs))
model.compile(loss="mean_absolute_error", optimizer=adam, metrics=[mae])

response = getPerformance(model, train_data, dev_data, y_train, y_dev, batch_size, epochs)
#model.save_weights("my_model.weights.h5")

history = response[5]
results = response[4]

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, epochs+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

response = getEvaluation(model, train_data, y_train)
print("R²: ", response[0])
print("Mean Distance:", response[1])
print("Median Distance:", response[2])

response = getEvaluation(model, dev_data, y_dev)
print("R²: ", response[0])
print("Mean Distance:", response[1])
print("Median Distance:", response[2])

response = getEvaluation(model, test_data, y_test)
print("R²: ", response[0])
print("Mean Distance:", response[1])
print("Median Distance:", response[2])