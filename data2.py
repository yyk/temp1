import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
import sys

np.set_printoptions(precision=4, suppress=True)

source_root = "./quantquote_daily_sp500/daily/"
output_root = "./quantquote_daily_sp500/generated/"
x_train_file = output_root + "x_train"
y_train_file = output_root + "y_train"
x_test_file = output_root + "x_test"
y_test_file = output_root + "y_test"


def process(file_path):
    a = pd.read_csv(file_path, names=['date', 'unknown', 'open', 'high', 'low', 'close', 'volume'],
                    usecols=[0, 2, 3, 4, 5, 6], header=None)
    c = a['close']
    v = a['volume']

    # a['sma5'] = c.rolling(window=5).mean()
    # a['sma20'] = c.rolling(window=20).mean()
    # a['sma50'] = c.rolling(window=50).mean()
    # a['sma150'] = c.rolling(window=150).mean()

    # a['msd5'] = c.rolling(window=5).std()
    # a['msd20'] = c.rolling(window=20).std()

#     a['ema5'] = c.ewm(span=5).mean()
    # a['ema20'] = c.ewm(span=20).mean()
#     a['ema25'] = c.ewm(span=25).mean()
#     a['ema35'] = c.ewm(span=35).mean()
    # a['ema50'] = c.ewm(span=50).mean()
    # a['ema150'] = c.ewm(span=150).mean()

    a['roc1'] = roc(c, 1)
    # a['roc5'] = roc(c, 5)
    # a['roc25'] = roc(c, 25)
    # a['roc20'] = roc(c, 20)
    # a['roc50'] = roc(c, 50)

    # a['vroc1'] = roc(v, 1)
    # a['vroc5'] = roc(v, 5)
    # a['vroc25'] = roc(v, 25)
    # a['vroc20'] = roc(v, 20)
    # a['vroc50'] = roc(v, 50)

#     a = macd(a, 12, 26)

    a.drop('open', axis=1, inplace=True)
    a.drop('high', axis=1, inplace=True)
    a.drop('low', axis=1, inplace=True)
    a.drop('close', axis=1, inplace=True)
    a.drop('volume', axis=1, inplace=True)

    # b = c.rolling(window=5).mean()
    # b = roc(c,5)
    b = c.diff(50).apply(lambda x: 0 if x <= 0 else 1)
    # a['diff5'] = b
    # print(a.loc[:, ['close', 'roc5', 'diff5']])
    # sys.exit(0)

    # discard the first 200 days.
    days_to_disgard = 200
    return a.as_matrix(\
           # columns=['date', 'roc5'] \
           #  columns=['date', 'roc5', 'vroc5', 'macd', 'macds', 'macdh' ] \
            ).astype('float32')[days_to_disgard:], \
            b.as_matrix()[days_to_disgard:]


def roc(prices, days):
    return prices.diff(days) / prices.shift(days)


def macd(a, fast, slow):
    prices = a['close']
    ema_fast = prices.ewm(fast).mean()
    ema_slow = prices.ewm(slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = prices.ewm(9).mean()
    macd_historgram = macd - macd_signal
    a['macd'] = macd
    a['macds'] = macd_signal
    a['macdh'] = macd_historgram
    return a

def load(file_path):
    a, b = process(file_path)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    window_size = 100
    for i in range(0, len(a) - window_size):
        # print(a[0])
        # sys.exit(0)
        x = a[i: i + window_size][:, 1:]
        # print(x[-6][:2])
        # print(x[-1])
        # x = x.T
        y = b[i + window_size -1]
        last_date = a[i+window_size][0]
        # print(y)
        # print(last_date)
        if last_date > 20120101:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)
    # sys.exit(0)
    return x_train, y_train, x_test, y_test


def gen_all():
    fs = os.listdir(source_root)
    #   fs = fs[:5]

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    #   i = 0
    for f in fs:
        x_train, y_train, x_test, y_test = load(source_root + f)
        #     i += 1
        #     print("%d Loaded %s" % (i, f))
        x_trains.extend(x_train)
        y_trains.extend(y_train)
        x_tests.extend(x_test)
        y_tests.extend(y_test)
    print("Creating np arrays")
    return np.array(x_trains), \
           np.array(y_trains), \
           np.array(x_tests), \
           np.array(y_tests)


def load_all():
    return gen_all()
    print("Loading " + x_train_file + ".npy")
    x_train = np.load(x_train_file + ".npy")
    print("Loading " + y_train_file + ".npy")
    y_train = np.load(y_train_file + ".npy")
    print("Loading " + x_test_file + ".npy")
    x_test = np.load(x_test_file + ".npy")
    print("Loading " + y_test_file + ".npy")
    y_test = np.load(y_test_file + ".npy")
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = gen_all()
    #   print(x_test.dtype)

    # print("Shuffling")
    # s = np.random.permutation(len(x_train))
    # x_train = x_train[s]
    # y_train = y_train[s]
    # s = np.random.permutation(len(x_test))
    # x_test = x_test[s]
    # y_test = y_test[s]

    print("Start writing")
    print(x_train.shape)

    np.save(x_train_file, x_train)
    print(y_train.shape)
    np.save(y_train_file, y_train)
    print(x_test.shape)
    np.save(x_test_file, x_test)
    print(y_test.shape)
    np.save(y_test_file, y_test)
