import pandas as pd
import numpy as np
import os

source_root = "./quantquote_daily_sp500/daily/"

def process(file_path):
    a = pd.read_csv(file_path, names=['date', 'unknown', 'open', 'high', 'low', 'close', 'volume'],
                    usecols=[0, 2, 3, 4, 5, 6], header=None)
    c = a['close']
    a['sma5'] = c.rolling(window=5).mean()
    a['sma25'] = c.rolling(window=25).mean()
    a['sma50'] = c.rolling(window=50).mean()
    a['sma150'] = c.rolling(window=150).mean()
    a['ema5'] = c.ewm(span=5).mean()
    a['ema25'] = c.ewm(span=25).mean()
    a['ema50'] = c.ewm(span=50).mean()
    a['ema150'] = c.ewm(span=150).mean()

    # discard the first 200 days.
    return a.as_matrix().astype('float32')[200:]

def load(file_path):
  a = process(file_path)
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  window_size = 100
  for i in range(0, len(a) - window_size):
    sample_source = a[i : i + window_size]
    (x, y) = buildSample(sample_source)
    last_date = sample_source[-1][0]
    if last_date > 20120101:
      x_test.append(x)
      y_test.append(y)
    else:
      x_train.append(x)
      y_train.append(y)
  return x_train, y_train, x_test, y_test

def buildSample(a):
  c1 = a[-1-5][4]
  c2 = a[-1][4]
  if c2 <= c1:
    c = 0
  else:
    c = 1
  y = np.array([c])
  return (a.T, y)

def load_all():
  fs = os.listdir(source_root)
  fs = fs[:5]

  x_trains = []
  y_trains = []
  x_tests = []
  y_tests = []
  for f in fs:
    x_train, y_train, x_test, y_test = load(source_root + f)
    print("Loaded " + f)
    x_trains.extend(x_train)
    y_trains.extend(y_train)
    x_tests.extend(x_test)
    y_tests.extend(y_test)
  return np.array(x_trains), \
         np_utils.to_categorical(np.array(y_trains), 2), \
         np.array(x_tests), \
         np_utils.to_categorical(np.array(y_tests), 2)

if __name__ == '__main__':
  x_train, y_train, x_test, y_test = load_all()
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)
