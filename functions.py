import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import math

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras import backend as K
from geopy import distance
from statistics import fmean, median

def transformarDados(X, y, seed):
  X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.4, random_state=seed)
  X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=seed)
  pca_transformer = PCA(n_components=40)
  X_train_normalized = X_train.copy().apply(normalizer)
  X_dev_normalized = X_dev.copy().apply(normalizer)
  X_test_normalized = X_test.copy().apply(normalizer)
  pca_transformer.fit(X_train_normalized)
  train_data = pca_transformer.transform(X_train_normalized)
  dev_data = pca_transformer.transform(X_dev_normalized)
  test_data = pca_transformer.transform(X_test_normalized)
  return train_data, dev_data, test_data, y_train, y_dev, y_test

def importarDados(path):
  df = pd.read_csv(path)
  df = dropRows(df, ['3432333852377918', '3432333851378418', '3432333877377B17',
                     '3432333863376118', '3432333852378418', '343233386A376018',
                     '343233384B376D18', '343233384D378718', '3432333851376518',
                     '343233384F378B18'])
  df['RX Time'] = timeLista(df)
  df['RX Time'] = df['RX Time'].apply(convertStringToDate)
  df = df.set_index('RX Time').sort_index()
  return df

def dropRows(df, devices):
  for device in devices:
    df = df.copy().drop(df[df['dev_eui'] == device].index)
  return df

def exibeMensagensJanelaTempo(df, window=7):
  df_copy = df.copy()
  df_copy['RX Time'] = df_copy['RX Time'].apply(convertStringToDate)
  date_number = getMessagesOfTimeWindow(df_copy, window)
  date_number['Window'] = date_number.copy()['Date'].apply(lambda x : x.strftime('%d-%m-%Y'))
  date_number = date_number.set_index('Date').sort_index()
  fig, ax = plt.subplots(figsize=(10, 8))
  fig = sns.barplot(x=date_number.index, y='Number', data=date_number)
  ax.set_xticklabels(labels=date_number.Window, rotation=90, ha='right')
  plt.show()

def exibeDispositivoQuantidade(df):
  date_number = getMessagesForDevice(df)
  fig, ax = plt.subplots(figsize=(10, 8))
  fig = sns.barplot(x=date_number.index, y='Number', data=date_number)
  ax.set_xticklabels(labels=date_number.Device, rotation=90, ha='right')
  plt.show()

def timeLista(df):
  df = df[df.columns[176:220]]
  lista = []
  for index in range(len(df)):
    for time in df.iloc[index]:
      if time != '?':
        lista.append(time)
        break
  return lista

def convertStringToDate(date_str):
    date_str = date_str.split("T")
    date_str[-1] = date_str[-1].split(".")
    date_str = date_str[0]
    date_format = '%Y-%m-%d'
    date_obj = datetime.datetime.strptime(date_str, date_format)
    return date_obj

def createDicWindow(window):
  dic_time = {}
  primeiro = datetime.datetime(2018, 11, 16)
  ultimo = datetime.datetime(2019, 2, 11)
  dic_time[primeiro] = 0
  dic_time[ultimo] = 0
  m = 1
  while True:
    dias = datetime.timedelta(days=(window*m))
    if dias + primeiro > ultimo:
      break
    else:
      dic_time[dias+primeiro] = 0
      m += 1
  return dic_time


def getMessagesOfTimeWindow(df, window):
  dic_time = createDicWindow(window)
  list_time = dic_time.keys()
  for time in df['RX Time']:
      if time in list_time:
          dic_time[time] += 1
      else:
          i = window-1
          while True:
              if time - datetime.timedelta(days=i) in list_time:
                  dic_time[time - datetime.timedelta(days=i)] += 1
                  break;
              else:
                  i -= 1
  return pd.DataFrame(dic_time.items(), columns=['Date', 'Number'])

def getMessagesForDevice(df):
  dic_device = {}
  list_device = []
  for device in df['dev_eui']:
    if device in list_device:
      dic_device[device] += 1
    else:
      dic_device[device] = 1
      list_device.append(device)
  return pd.DataFrame(dic_device.items(), columns=['Device', 'Number'])

def positive(x):
  return x+200

def normalizer(x):
  return positive(x)/200

def powed(x,minimum=-200,b=math.exp(1)):
    positive_x= x-minimum
    numerator = positive_x.pow(b)
    denominator = (-minimum)**(b)
    powed_x = numerator/denominator
    final_x = powed_x
    return final_x

# def exponential(x,minimum=-200):
#     positive_x= x-minimum
#     numerator = np.exp(positive_x.div(a))
#     denominator = np.exp(-minimum/a)
#     exponential_x = numerator/denominator
#     exponential_x = exponential_x * 1000  #facilitating calculations
#     final_x = exponential_x
#     return final_x

def getMeanDistance(y_true, y_pred):
    y_pred = pd.DataFrame(y_pred)
    temp = y_true.copy()
    temp.reset_index(drop=True, inplace=True)
    temp = temp.rename(columns={"latitude": 0, "longitude": 1})
    d = []
    for i in range(len(y_true)):
        d1 = (temp[0][i], temp[1][i])
        d2 = (y_pred[0][i], y_pred[1][i])
        d.append(distance.distance(d1, d2).m)
    return fmean(d), median(d)

def evaluate_model(X_train, X_dev, y_train, y_dev, batch_size, model, epochs):
    results = list()
    history = model.fit(X_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev))
    result = model.evaluate(X_dev, y_dev, verbose=0)
    results.append(result)
    return results,history

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def getEvaluation(model, X_test, y_test):
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  md, mdd = getMeanDistance(y_test, y_pred)
  return [r2, md, mdd]

def getPerformance(model, X_train, X_dev, y_train, y_dev, batch_size, epochs):
  results, history = evaluate_model(X_train, X_dev, y_train, y_dev, batch_size, model, epochs)
  y_pred1 = model.predict(X_train)
  y_pred2 = model.predict(X_dev)
  r2 = r2_score(y_train, y_pred1)
  r22 = r2_score(y_dev, y_pred2)
  md1, mdd1 = getMeanDistance(y_train, y_pred1)
  md2, mdd2 = getMeanDistance(y_dev, y_pred2)
  return [r2, r22, md1, md2, results, history, mdd1, mdd2]

def getMaxMeanDistancesClusters(data, n_units):
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

def getHeuristicBetas(data, n_units):
  dmax, davg = getMaxMeanDistancesClusters(data)
  bmax = dmax/np.sqrt(n_units)
  bavg = 2*davg
  return bmax, bavg