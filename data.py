import os
import numpy as np

def load_all(window_size, test_count):
    symbols = [ s.split('.')[0].split('_')[1]
      for s in os.listdir("/home/yyk/stockmarket/quantquote_daily_sp500/daily/") ]
    symbols = symbols[:10]
    X_train = np.empty(shape = [0, window_size, 6])
    Y_train = np.empty(shape = [0, 1])
    X_test = np.empty(shape = [0, window_size, 6])
    Y_test = np.empty(shape = [0, 1])
    for s in symbols:
        (X1, Y1), (X2, Y2) = load(s, window_size, test_count)
    np.concatenate((X_train, X1))
    np.concatenate((Y_train, Y1))
    np.concatenate((X_test, X2))
    np.concatenate((Y_test, Y2))

    return (X_train, Y_train), (X_test, Y_test)

def load(symbol, window_size, test_count):
    f = open("./quantquote_daily_sp500/daily/table_%s.csv" % symbol, "r")
    lines = [line.strip() for line in f]
    X_train = np.empty(shape = [0, window_size, 6])
    Y_train = np.empty(shape = [0, 1])
    X_test = np.empty(shape = [0, window_size, 6])
    Y_test = np.empty(shape = [0, 1])
    for i in range(0, len(lines) - window_size):
        (X, Y) = build(lines[i : i + window_size + 1])
        if i + test_count > len(lines):
            np.append(X_test, X)
            np.append(Y_test, 3)
        else:
            np.append(X_train, X)
            np.append(Y_train, 3)

    print("Loaded %s" % symbol)
    return (X_train, Y_train), (X_test, Y_test)

def build(lines):
    X = np.empty(shape = [len(lines) - 1, 6])
    for line in lines[:-1]:
        (d, unknown, o, h, l, c, v) = [ float(s) for s in line.split(',') ]
        x = np.array([d, o, h, l, c, v], np.float32)
        np.append(X, x)
    c1 = float(lines[-2].split(',')[5])
    c2 = float(lines[-1].split(',')[5])
    ratio = (c2-c1)/c1
    if ratio < -0.05:
        y = 1
    elif ratio < 0.05:
        y = 2
    else:
        y = 3
    Y = np.array([3])
    return (X, Y)

if __name__ == '__main__':
    load_all(100, 100)

