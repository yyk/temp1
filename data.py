import os
import os.path
import itertools
import numpy as np
import random
import pickle
from multiprocessing.pool import ThreadPool

random.seed(5)

num_days = 3
dir_root = "/tmp/stock/"
suffix = "up_down_next_%d_days" % num_days
total_loaded = 0

def load_all(window_size, test_count):
    """
    returns a tensor of dimensions (number of samples, channels (6), length)
    """
    output_file=dir_root + "daily_%s_%s.pickle" % (window_size, suffix)

    shard = 0
    X_train = []
    while True:
        f = output_file + ".xtrain.shard%d" % shard
        if os.path.isfile(f):
            print("Loading %s" % f)
            X_train.extend(pickle.load(open(f, "rb")))
        else:
            break
        shard += 1

    Y_train = pickle.load(open(output_file + ".ytrain", "rb"))
    X_test = pickle.load(open(output_file + ".xtest", "rb"))
    Y_test = pickle.load(open(output_file + ".ytest", "rb"))
    return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))

def pickle_all(window_size, test_count):
    file_paths = os.listdir(dir_root + "daily/")

#     file_paths = file_paths[:5]

    # pool = ThreadPool(processes=50)
    # futures = []
    # for path in file_paths:
    #     futures.append(pool.apply_async(load, (path, window_size, test_count)))
    #
    # train = []
    # test = []
    # for future in futures:
    #     (a, b) = future.get()
    #     train.extend(a)
    #     test.extend(b)

    (train, test) = load_batch(file_paths, window_size, test_count)

    random.shuffle(train)
    x_train = [ t[0] for t in train ]
    y_train = [ t[1] for t in train ]
    random.shuffle(test)
    x_test = [ t[0] for t in test ]
    y_test = [ t[1] for t in test ]

#     return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
    return (x_train, y_train), (x_test, y_test)

def load_batch(file_paths, window_size, test_count):
    train = []
    test = []
    for file_path in file_paths:
        train_, test_ = load(file_path, window_size, test_count)
        train.extend(train_)
        test.extend(test_)
    return train, test

def load(file_path, window_size, test_count):
    global total_loaded
    f = open(dir_root + "/daily/" + file_path, "r")
    lines = [line.strip().split(',') for line in f]
    lines = [(float(l[0]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])) for l in lines]
    train = []
    test = []
    for i in range(0, len(lines) - window_size):
        (X, Y) = build(lines[i : i + window_size])
        last_line = lines[i + window_size]
        last_date = last_line[0]
        if last_date > 20120101:
            test.append((X, Y))
        else:
            train.append((X, Y))

    total_loaded += 1
    print("%d Loaded %s %d %d" % (total_loaded, file_path, len(train), len(test)))
    return train, test

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
    c1 = lines[-1-num_days][5]
    c2 = lines[-1][5]
    if c1 > c2:
        c = 0
    else:
        c = 1
    # print(ratio, c)
    Y = np.array([c])
    return (X, Y)

if __name__ == '__main__':
    (X1, Y1), (X2, Y2) = pickle_all(100, 100)
    print("writing")
    output_file = "/home/yyk/stockmarket/quantquote_daily_sp500/daily_100_%s.pickle" % suffix
    i = 0
    shard = 0
    step = 100000
    while i <= len(X1):
        end = len(X1) if i + step > len(X1) else i + step
        f = output_file + ".xtrain.shard%d" % shard
        print("Writing %s" % f)
        pickle.dump(X1[i:end], open(f, "wb"))
        i += step
        shard += 1
    pickle.dump(Y1, open(output_file + ".ytrain", "wb"))
    pickle.dump(X2, open(output_file + ".xtest", "wb"))
    pickle.dump(Y2, open(output_file + ".ytest", "wb"))

