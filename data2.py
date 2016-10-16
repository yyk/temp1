import pandas as pd
from multiprocessing.pool import Pool
import threading
import numpy as np
import os
import sys

np.set_printoptions(precision=4, suppress=True)

version = 9
source_root = "./quantquote_daily_sp500/preprocessed/"
output_root = "./quantquote_daily_sp500/generated/%d/" % version
x_train_file = output_root + "x_train"
y_train_file = output_root + "y_train"
x_test_file = output_root + "x_test"
y_test_file = output_root + "y_test"
window_size = 100

def process(file_path):
    a = pd.read_csv(file_path, header=0)

    b = a['close_roc10_ln'].shift(-10).apply(lambda x: 0 if x <= 0 else 1)
    # b = rsi(a['close'], period=14)
    # a['diff5'] = b
    # print(a.loc[:, ['close', 'roc5', 'diff5']])
    # sys.exit(0)

    # discard the first 200 days.
    days_to_disgard = 200
    # return a.as_matrix().astype('float32')[days_to_disgard:], b.as_matrix()[days_to_disgard:]
    return a[days_to_disgard:], b[days_to_disgard:]

def load(file_path):
    a, b = process(file_path)
    print("Loading %s" % (file_path, ))
    columns = ['date',
            'macd', 'macds', 'macdh',
               # 'close_roc1_ln_scaled',
               # 'close_roc5_ln_scaled',
               # 'close_roc10_ln_scaled',
               'rsi14',
               ]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    a = a[columns]
    a = a.as_matrix(columns=columns).astype('float32')
    for i in range(0, len(a) - window_size):
        # print(a[0])
        # sys.exit(0)
        x = a[i: i + window_size, 1:]
        y = b.iloc[i + window_size - 1]
        last_date = a[i+window_size-1][0]

        # print(x)
        # print(y)
        # sys.exit(1)

        # x['macd'] = x['macd'] / x['sma%d' % window_size].iloc[-1] * 100
        # macd = x['macd']
        # x['macds'] = macd.ewm(9).mean()
        # x['macdh'] = macd - x['macds']

        # for column in ['date', 'close', 'sma%d' % window_size]:
        #     x.drop(column, axis=1, inplace=True)

        if last_date > 20120101:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)
    return x_train, y_train, x_test, y_test

def gen_all(test=False):
    fs = os.listdir(source_root)
    if test:
        print("Only testing with 5 files")
        fs = fs[:5]
    # fs = fs[:1]
    # fs = ["table_agn.csv", ]
    async_results = {}
    pool = Pool(processes=12)
    for f in fs:
        if not f.endswith(".csv"):
            print("Skipping ", f)
            continue
        async_results[f] = (pool.apply_async(load, (source_root + f,)))
        if test:
            break

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    #   i = 0
    for key, result in async_results.items():
        try:
            x_train, y_train, x_test, y_test = result.get()
        except Exception as e:
            print(key)
            raise e

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
    if not os.path.exists(x_train_file + ".npy"):
        produce_all()
    print("Loading " + x_train_file + ".npy")
    x_train = np.load(x_train_file + ".npy")
    print("Loading " + y_train_file + ".npy")
    y_train = np.load(y_train_file + ".npy")
    print("Loading " + x_test_file + ".npy")
    x_test = np.load(x_test_file + ".npy")
    print("Loading " + y_test_file + ".npy")
    y_test = np.load(y_test_file + ".npy")
    return x_train, y_train, x_test, y_test


def produce_all(test=False):
    x_train, y_train, x_test, y_test = gen_all(test)
    print("Start writing")
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    print(x_train.shape)
    np.save(x_train_file, x_train)
    print(y_train.shape)
    np.save(y_train_file, y_train)
    print(x_test.shape)
    np.save(x_test_file, x_test)
    print(y_test.shape)
    np.save(y_test_file, y_test)


if __name__ == '__main__':
    # fs = os.listdir(source_root)
    # c0 = 0
    # c1 = 1
    # for f in fs:
    #     x0, x1 = calculate(source_root + f)
    #     c0 += x0
    #     c1 += x1
    # print(c0/(c0+c1), c1/(c0+c1))

    produce_all(len(sys.argv) > 1 and sys.argv[1] == 'test')
