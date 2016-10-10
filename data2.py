import pandas as pd
from multiprocessing.pool import Pool
import threading
import numpy as np
import os
import sys

np.set_printoptions(precision=4, suppress=True)

version = 8
source_root = "./quantquote_daily_sp500/daily/"
output_root = "./quantquote_daily_sp500/generated/%d/" % version
x_train_file = output_root + "x_train"
y_train_file = output_root + "y_train"
x_test_file = output_root + "x_test"
y_test_file = output_root + "y_test"
window_size = 50

def calculate(file_path):
    a = pd.read_csv(file_path, names=['date', 'unknown', 'open', 'high', 'low', 'close', 'volume'],
                    usecols=[0, 2, 3, 4, 5, 6], header=None)
    c = a['close']
    a = a[(a['date'] < 20120101)]
    # print(a)
    b = a['close'].diff(-5).apply(lambda x: 0 if x <= 0 else 1)
    b = b[:200]
    if len(b) == 0:
        return 0,0

    return b.value_counts()

def process(file_path):
    a = pd.read_csv(file_path, names=['date', 'unknown', 'open', 'high', 'low', 'close', 'volume'],
                    usecols=[0, 2, 3, 4, 5, 6], header=None)
    c = a['close']
    a['volume']
    v = a['volume']

    # a['close_ln'] = np.log(a['close'])
    # a['roc1_ln'] = a['close_ln'] / a['close_ln'].shift(1)
    # a['volume_ln'] = np.log(a['volume']/10000)
    # a['vroc1_ln'] = a['volume_ln'] / a['volume_ln'].shift(1)

    a['sma%d' % window_size] = c.rolling(window=window_size).mean()
    # a['sma5'] = c.rolling(window=5).mean()
    # a['sma20'] = c.rolling(window=20).mean()
    # a['sma50'] = c.rolling(window=50).mean()
    # a['sma150'] = c.rolling(window=150).mean()

    # a['msd5'] = c.rolling(window=5).std()
    # a['msd20'] = c.rolling(window=20).std()

    # a['ema5'] = c.ewm(span=5).mean()
    # a['ema20'] = c.ewm(span=20).mean()
    # a['ema25'] = c.ewm(span=25).mean()
#     a['ema35'] = c.ewm(span=35).mean()
    # a['ema50'] = c.ewm(span=50).mean()
    # a['ema150'] = c.ewm(span=150).mean()

    # a['roc1'] = roc(c, 1)
    # a['roc5'] = roc(c, 5)
    # a['roc25'] = roc(c, 25)
    # a['roc20'] = roc(c, 20)
    # a['roc50'] = roc(c, 50)

    # a['vroc1'] = roc(v, 1)
    # a['vroc5'] = roc(v, 5)
    # a['vroc25'] = roc(v, 25)
    # a['vroc20'] = roc(v, 20)
    # a['vroc50'] = roc(v, 50)

    a = macd(a, 12, 26)
    a['rsi14'] = rsi(a['close'], period=14) / 100
    a['mfi14'] = mfi(a, period=4) / 100

    # b = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    # b = c.ewm(span=26).mean()
    # b = roc(c,5)
    b = c.diff(-10).apply(lambda x: 0 if x <= 0 else 1)
    # b = rsi(a['close'], period=14)
    # a['diff5'] = b
    # print(a.loc[:, ['close', 'roc5', 'diff5']])
    # sys.exit(0)

    # discard the first 200 days.
    days_to_disgard = 200
    # return a.as_matrix().astype('float32')[days_to_disgard:], b.as_matrix()[days_to_disgard:]
    return a[days_to_disgard:], b[days_to_disgard:]


def roc(prices, days):
    return prices.pct_change(periods=days)


def macd(a, fast, slow):
    prices = a['close']
    # average = prices.mean()
    ema_fast = prices.ewm(fast).mean()
    ema_slow = prices.ewm(slow).mean()
    # normalize them
    macd = ema_fast - ema_slow
    a['macd'] = macd
    # a['macds'] = macd.ewm(9).mean()
    # a['macdh'] = macd - macd.ewm(9).mean()
    return a

def rsi(series, period=14):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
    """
    diff = series.diff(1) # diff against previous day
    gain = diff.apply(lambda x: x if x > 0 else 0) # gain against previous day
    loss = diff.apply(lambda x: -x if x < 0 else 0) # loss against previous day (as positive value)
    first_average_gain = gain.rolling(window=period).mean() # average gain in previous 14 days
    first_average_loss = loss.rolling(window=period).mean() # average loss in previous 14 days

    average_gain = (first_average_gain.shift(1) * (period - 1) + gain) / period
    average_loss = (first_average_loss.shift(1) * (period - 1) + loss) / period

    result =  100 - 100 / (1 + average_gain / average_loss)
    # print(result)
    return result

def mfi(a, period = 14):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    """
    typical = (a['close'] + a['low'] + a['high']) / 3
    raw_money_flow = typical * a['volume']
    diff = typical.diff(1)
    positive = diff.apply(lambda x: 1 if x > 0 else 0)
    negative = diff.apply(lambda x: 1 if x < 0 else 0)
    positive_money_flow = (raw_money_flow * positive).rolling(window=period).sum()
    negative_money_flow = (raw_money_flow * negative).rolling(window=period).sum()

    money_flow_ratio = positive_money_flow / negative_money_flow
    result = 100 - 100 / (1 + money_flow_ratio)
    return result

def load(file_path):
    global global_number_of_file_loaded
    a, b = process(file_path)
    print("Loading %s" % (file_path, ))
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for column in ['open', 'high', 'low', 'volume']:
        a.drop(column, axis=1, inplace=True)
    a = a.as_matrix(columns=[
            'date',
            # 'sma%d' % window_size,
            # 'macd', 'macds', 'macdh',
            'rsi14', 'mfi14',
            # 'close_ln', 'roc1_ln',
            # 'volume_ln', 'vroc1_ln'
        ]).astype('float32')
    for i in range(0, len(a) - window_size):
        # print(a[0])
        # sys.exit(0)
        x = a[i: i + window_size, 1:]
        y = b.iloc[i + window_size - 1]
        last_date = a[i+window_size-1][0]

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
    async_results = []
    pool = Pool(processes=12)
    for f in fs:
        async_results.append(pool.apply_async(load, (source_root + f,)))
        if test:
            break

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    #   i = 0
    for async_result in async_results:
        x_train, y_train, x_test, y_test = async_result.get()
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
