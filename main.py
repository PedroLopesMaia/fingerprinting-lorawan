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
from functions import getPerformance, normalizer, positive, evaluateModel

seed = 260623
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISBLE_DEVICE'] = ''
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#Separação dos subconjuntos dos dados
data = pd.read_csv('lorawan_antwerp_2019_dataset.csv')
df = data.copy().drop(["RX Time", "HDOP"], axis=1)
X = df.copy().drop(["Latitude", "Longitude"], axis=1)
y = df[["Latitude", "Longitude"]]
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.3, random_state=seed)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=seed)

#Préprocessamento
pca_transformer = PCA(n_components=40)

X_train_normalized = X_train.copy().apply(normalizer)
X_dev_normalized = X_dev.copy().apply(normalizer)
X_test_normalized = X_test.copy().apply(normalizer)

pca_transformer.fit(X_train_normalized)
train_data = pca_transformer.transform(X_train_normalized)
dev_data = pca_transformer.transform(X_dev_normalized)
test_data = pca_transformer.transform(X_test_normalized)

epochs = 2000
batch_size = 10
learning_rate = 0.0000125
betas = 0.75
n_units = 270
initializer = InitCentersKMeans(train_data)

mae = tf.keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error")
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

n_inputs, n_outputs = train_data.shape[1], y_train.shape[1]

rbflayer = RBFLayer(n_units,
                    initializer=initializer,
                    betas=betas,
                    input_shape=(n_inputs,))

model = Sequential()
model.add(rbflayer)
#model.add(Dense(1024))
#model.add(Dense(512))
model.add(Dense(n_outputs))
model.compile(loss="mean_absolute_error", optimizer=adam, metrics=[mae])

#model.load_weights("my_model.weights.h5")
response = getPerformance(model, train_data, dev_data, y_train, y_dev, batch_size, epochs)
model.save_weights("my_model.weights.h5")

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

init = 1000
end = 2000
loss_train = history.history['loss'][init:end]
loss_val = history.history['val_loss'][init:end]
epochs = range(init+1,end+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("R² (train): ", response[0])
print("R² (dev):", response[1])
print("Mean Distance (train):", response[2])
print("Mean Distance (dev):", response[3])
print("Median Distance (train):", response[6])
print("Median Distance (dev):", response[7])
print("Loss (train):", loss_train[-1])
print("Loss (dev):", loss_val[-1])

response = evaluateModel(model, test_data, y_test)
print()
print("R² (test): ", response[0])
print("Mean Distance (test): ", response[1])
print("Median Distance (test):", response[2])
# print()
# for i in range(len(loss_train)):
#   print("Epoch: {e}ª - loss: {l1} - val_loss: {l2}" .format(e=i+1, l1=loss_train[i], l2=loss_val[i]))