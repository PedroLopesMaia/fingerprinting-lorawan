import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from keras import backend as K
import numpy as np
from geopy import distance
from statistics import fmean, median

def positive(x):
  return x+200

def normalizer(x):
  return positive(x)/200

# def getMeanDistance(y_true, y_pred):
#     y_pred = pd.DataFrame(y_pred)
#     temp = y_true.copy()
#     temp.reset_index(drop=True, inplace=True)
#     temp = temp.rename(columns={"Latitude": 0, "Longitude": 1})
#     d = []
#     for i in range(len(y_true)):
#         d1 = (temp[0][i], temp[1][i])
#         d2 = (y_pred[0][i], y_pred[1][i])
#         d.append(distance.distance(d1, d2).m)
#     return sum(d)/len(d)

def getMeanDistance(y_true, y_pred):
    y_pred = pd.DataFrame(y_pred)
    temp = y_true.copy()
    temp.reset_index(drop=True, inplace=True)
    temp = temp.rename(columns={"Latitude": 0, "Longitude": 1})
    d = []
    for i in range(len(y_true)):
        d1 = (temp[0][i], temp[1][i])
        d2 = (y_pred[0][i], y_pred[1][i])
        d.append(distance.distance(d1, d2).m)
    return fmean(d), median(d)

def evaluateModel(model, X_test, y_test):
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  md, mdd = getMeanDistance(y_test, y_pred)
  return [r2, md, mdd]

def evaluate_model(model, X_train, X_dev, y_train, y_dev, batch_size, epochs):
    results = list()
    history = model.fit(X_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev))
    result = model.evaluate(X_dev, y_dev, verbose=0)
    results.append(result)
    return results, history

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def getPerformance(model, X_train, X_dev, y_train, y_dev, batch_size, epochs):
  results, history = evaluate_model(model, X_train, X_dev, y_train, y_dev, batch_size, epochs)
  y_pred1 = model.predict(X_train)
  y_pred2 = model.predict(X_dev)
  r2 = r2_score(y_train, y_pred1)
  r22 = r2_score(y_dev, y_pred2)
  md1, mdd1 = getMeanDistance(y_train, y_pred1)
  md2, mdd2 = getMeanDistance(y_dev, y_pred2)
  return [r2, r22, md1, md2, results, history, mdd1, mdd2]

def getMaxMeanDistancesClusters(data):
  distances = []
  km = KMeans(n_clusters=n_units, max_iter=100, verbose=0)
  km.fit(data)
  array = km.cluster_centers_
  for i in range(len(array)):
    if i != len(array)-1:
      c1 = array[i]
      for c2 in array[i+1:]:
        distances.append(np.linalg.norm(c1 - c2))
  distances = np.array(distances)
  return distances.max(), distances.sum()/distances.size

def getHeuristicBetas(data):
  dmax, davg = getMaxMeanDistancesClusters(data)
  bmax = dmax/np.sqrt(n_units)
  bavg = 2*davg
  return bmax, bavg

