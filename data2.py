from keras.utils import np_utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from multiprocessing.pool import Pool
import threading
import numpy as np
import os
import sys

np.set_printoptions(precision=4, suppress=True)

version = 2
source_root = "./quantquote_daily_sp500/preprocessed/"
output_root = "./quantquote_daily_sp500/generated/%d/" % version
x_train_file = output_root + "x_train"
y_train_file = output_root + "y_train"
x_test_file = output_root + "x_test"
y_test_file = output_root + "y_test"
train_sample_weight_file = output_root + "train_sample_weight"
window_size = 100

def process(file_path):
    a = pd.read_csv(file_path, header=0)

    b = a['close_roc10_ln'].shift(-10).apply(lambda x: 0 if x <= 0 else 1)

    w = a['close_roc10_ln'].shift(-10)
    # b = rsi(a['close'], period=14)
    # a['diff5'] = b
    # print(a.loc[:, ['close', 'roc5', 'diff5']])
    # sys.exit(0)

    # discard the first 200 days.
    days_to_disgard = 200
    # return a.as_matrix().astype('float32')[days_to_disgard:], b.as_matrix()[days_to_disgard:]
    return a[days_to_disgard:], b[days_to_disgard:], w[days_to_disgard:]

def load(file_path, columns):
    a, b, c = process(file_path)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_sample_weight = []

    a = a[columns]
    a = a.as_matrix(columns=columns).astype('float32')
    for i in range(0, len(a) - window_size):
        # print(a[0])
        # sys.exit(0)
        x = a[i: i + window_size, 1:]
        y = b.iloc[i + window_size - 1]
        sample_weight = c.iloc[i + window_size - 1]
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
            train_sample_weight.append(sample_weight)
    return x_train, y_train, x_test, y_test, train_sample_weight

def gen_all(test=False):
    fs = os.listdir(source_root)
    if test:
        print("Only testing with 5 files")
        fs = fs[:5]
    # fs = fs[:1]
    # fs = ["table_agn.csv", ]
    async_results = {}
    processes=12
    pool = Pool(processes)
    columns = ['date',
        # 'macd', 'macds', 'macdh',
           'ppo_12_26', 'ppos_12_26', 'ppoh_12_26',
           'close_roc1_ln_scaled',
           'close_roc5_ln_scaled',
           'close_roc20_ln_scaled',
           'rsi14', 'mfi14',
           'close_ln_scaled',
           'close_sma5_ln_scaled', 'close_sma20_ln_scaled',
           ]
    # columns.extend(['month_%d' % x for x in range(1, 13)])

    print("Loading files with %d processes..." % processes)
    print("Columns being used ", columns)
    for f in fs:
        if not f.endswith(".csv"):
            print("Skipping ", f)
            continue
        async_results[f] = (pool.apply_async(load, (source_root + f, columns)))
        if test:
            break

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    train_sample_weights = []
    i = 0
    shard_number = 0
    item_per_shard = 10
    total_number_of_train_samples = 0
    total_number_of_test_samples = 0
    for key, result in async_results.items():
        try:
            x_train, y_train, x_test, y_test, train_sample_weight = result.get()
        except Exception as e:
            print(key)
            raise e

        i += 1
        # print("%d Loaded %s" % (i, f))
        x_trains.extend(x_train)
        y_trains.extend(y_train)
        x_tests.extend(x_test)
        y_tests.extend(y_test)
        train_sample_weights.extend(train_sample_weight)

        total_number_of_train_samples += len(x_train)
        total_number_of_test_samples += len(x_test)

        if i % item_per_shard == 0:
            save_np_array(str(shard_number), x_trains, y_trains, x_tests, y_tests, train_sample_weights)
            shard_number += 1
            x_trains = []
            y_trains = []
            x_tests = []
            y_tests = []
            train_sample_weights = []

    print("Total number of train samples ", total_number_of_train_samples) # 1461397
    print("Total number of test samples ", total_number_of_test_samples) # 197463


def save_np_array(shard_number, x_trains, y_trains, x_tests, y_tests, train_sample_weights):
    print("Saving shard " + shard_number)
    x_train = np.array(x_trains)
    y_train = np.array(y_trains)
    x_test = np.array(x_tests)
    y_test = np.array(y_tests)
    train_sample_weight = np.array(train_sample_weights)

    np.save(x_train_file + "." + shard_number, x_train)
    np.save(y_train_file + "." + shard_number, y_train)
    np.save(x_test_file + "." + shard_number, x_test)
    np.save(y_test_file + "." + shard_number, y_test)
    np.save(train_sample_weight_file + "." + shard_number, train_sample_weight)


def load_shard0():
    print("Loading " + x_train_file + ".0.npy")
    x_train = np.load(x_train_file + ".0.npy")
    print("Loading " + y_train_file + ".0.npy")
    y_train = np.load(y_train_file + ".0.npy")
    print("Loading " + x_test_file + ".0.npy")
    x_test = np.load(x_test_file + ".0.npy")
    print("Loading " + y_test_file + ".0.npy")
    y_test = np.load(y_test_file + ".0.npy")
    print("Loading " + train_sample_weight_file + ".0.npy")
    train_sample_weight = np.load(train_sample_weight_file + ".0.npy")
    return x_train, y_train, x_test, y_test, train_sample_weight

def load_y_test():
    shard = 0
    y_test = np.load(y_test_file + ".%d.npy" % shard)
    while True:
        shard += 1
        if not os.path.exists(y_test_file + ".%d.npy" % shard):
            return y_test
        y_test_ = np.load(y_test_file + ".%d.npy" % shard)
        y_test = np.concatenate((y_test, y_test_), axis=0)


def produce_all(test=False):
    print("Start writing")
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    gen_all(test)


def train_sample_generator(batch_size):
    shard_number = 0
    train_sample_weight, x_train, y_train = load_train_shard(shard_number)

    i = 0
    while True:
        while i + batch_size > x_train.shape[0]:
            x_train = x_train[i:]
            y_train = y_train[i:]
            train_sample_weight = train_sample_weight[i:]

            shard_number += 1
            if not os.path.exists(x_train_file + ".%d.npy" % shard_number):
                shard_number = 0
            train_sample_weight_, x_train_, y_train_ = load_train_shard(shard_number)

            x_train = np.concatenate((x_train, x_train_), axis=0)
            y_train = np.concatenate((y_train, y_train_), axis=0)
            train_sample_weight = np.concatenate((train_sample_weight, train_sample_weight_), axis=0)
            i = 0

        yield x_train[i:i + batch_size], y_train[i:i + batch_size], train_sample_weight[i:i + batch_size]
        i += batch_size


def load_train_shard(shard_number):
    # print("Loading train shard %d" % shard_number)
    x_train = np.load(x_train_file + ".%d.npy" % shard_number)
    y_train = np.load(y_train_file + ".%d.npy" % shard_number)
    y_train = np_utils.to_categorical(np.array(y_train), 2)
    train_sample_weight = np.load(train_sample_weight_file + ".%d.npy" % shard_number)
    weight_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    train_sample_weight = weight_scaler.fit_transform(train_sample_weight.reshape(-1, 1))
    train_sample_weight = (np.absolute(train_sample_weight)) + 0.5
    train_sample_weight = train_sample_weight.flatten()
    return train_sample_weight, x_train, y_train


def test_sample_generator(batch_size):
    shard_number = 0
    x_test, y_test = load_test_shard(shard_number)

    i = 0
    while True:
        while i + batch_size > x_test.shape[0]:
            x_test = x_test[i:]
            y_test = y_test[i:]

            shard_number += 1
            if not os.path.exists(x_test_file + ".%d.npy" % shard_number):
                shard_number = 0
            x_test_, y_test_ = load_test_shard(shard_number)

            x_test = np.concatenate((x_test, x_test_), axis=0)
            y_test = np.concatenate((y_test, y_test_), axis=0)
            i = 0

        yield x_test[i:i + batch_size], y_test[i:i + batch_size]
        i += batch_size


def load_test_shard(shard_number):
    # print("Loading test shard %d" % shard_number)
    x_test = np.load(x_test_file + ".%d.npy" % shard_number)
    y_test = np.load(y_test_file + ".%d.npy" % shard_number)
    y_test = np_utils.to_categorical(np.array(y_test), 2)
    return x_test, y_test

def predict_sample_generator(batch_size):
    shard_number = 0
    x_test, y_test = load_test_shard(shard_number)

    i = 0
    while True:
        while i + batch_size > x_test.shape[0]:
            x_test = x_test[i:]

            shard_number += 1
            if not os.path.exists(x_test_file + ".%d.npy" % shard_number):
                shard_number = 0
            x_test_, y_test_ = load_test_shard(shard_number)

            x_test = np.concatenate((x_test, x_test_), axis=0)
            i = 0

        yield x_test[i:i + batch_size]
        i += batch_size


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
