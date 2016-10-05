import os
import os.path
import itertools
import numpy as np
import random
import pickle
from multiprocessing.pool import ThreadPool

random.seed(5)

num_days = 10
src_root = "/home/yyk/stockmarket/quantquote_daily_sp500/"
output_dir_root = "/data/stock/"
# output_dir = output_dir_root + str(num_days) + "-2percent"
output_dir = output_dir_root + str(num_days)
xtrain_file = output_dir + "/x_train"
ytrain_file = output_dir + "/y_train"
xtest_file = output_dir + "/x_test"
ytest_file = output_dir + "/y_test"

total_loaded = 0

def load_all(window_size, test_count):
    """
    returns a tensor of dimensions (number of samples, channels (6), length)
    """

    print("Loading " + xtrain_file + ".npy")
    x_train = np.load(xtrain_file + ".npy")
    print("Loading " + ytrain_file + ".npy")
    y_train = np.load(ytrain_file + ".npy")

    category = 0
    tests = {}
    while True:
        x = xtest_file + ".category" + str(category) + ".npy"
        y = ytest_file + ".category" + str(category) + ".npy"
        if os.path.exists(x):
            print("Loading " + x)
            x_test = np.load(x)
            print("Loading " + y)
            y_test = np.load(y)
            tests[category] = (x_test, y_test)
            category += 1
        else:
            break

    return (x_train, y_train), tests

def pickle_all(window_size, test_count):
    file_paths = os.listdir(src_root + "daily/")

#     file_paths = ('table_flir.csv',)
    # file_paths = file_paths[:5]

    (train, tests) = load_batch(file_paths, window_size, test_count)

    return train, tests


def load_batch(file_paths, window_size, test_count):
    train = []
    tests = {}
    for file_path in file_paths:
        train_, tests_ = load(file_path, window_size, test_count)
        train.extend(train_)
        for key, value in tests_.items():
            tests.setdefault(key, [])
            tests[key].extend(value)
    return train, tests

def load(file_path, window_size, test_count):
    print("Loading " + file_path)
    global total_loaded
    f = open(src_root + "/daily/" + file_path, "r")
    lines = [line.strip().split(',') for line in f]
    lines = [(float(l[0]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])) for l in lines]
    train = []
    tests = {}
    num_tests = 0
    for i in range(0, len(lines) - window_size):
        (X, Y) = build(lines[i : i + window_size])
        last_line = lines[i + window_size]
        last_date = last_line[0]
        if last_date > 20120101:
            tests.setdefault(Y[0], [])
            tests[Y[0]].append((X, Y))
            num_tests += 1
        else:
            train.append((X, Y))

    total_loaded += 1
    print("%d Loaded %s %d %d" % (total_loaded, file_path, len(train), num_tests))
    return train, tests

def build(lines):
    channels = []
    for i in range(0, 6):
        channels.append([])
    for line in lines[:-num_days]:
        for i in range(0,6):
            channels[i].append(line[i])
    xarr = []
    for i in range(0, 6):
        xarr.append(np.array(channels[i]))
    X = np.array(xarr)
    c1 = lines[-1-num_days][4]
    c2 = lines[-1][4]
#     print(lines)
#     if (c2-c1)/c1 < 0.02:
    if c2 <= c1:
        c = 0
    else:
        c = 1
    # print(ratio, c)
    Y = np.array([c])
    return (X, Y)

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train, tests = pickle_all(100, 100)

    random.shuffle(train)
    x_train = [ t[0] for t in train ]
    y_train = [ t[1] for t in train ]
    X1 = np.array(x_train)
    Y1 = np.array(y_train)
    np.save(xtrain_file, X1)
    np.save(ytrain_file, Y1)

    for category, test in tests.items():
        random.shuffle(test)
        x_test = [ t[0] for t in test ]
        y_test = [ t[1] for t in test ]
        X = np.array(x_test)
        Y = np.array(y_test)
        np.save(xtest_file + ".category" + str(category), X)
        np.save(ytest_file + ".category" + str(category), Y)
