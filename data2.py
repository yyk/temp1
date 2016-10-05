import pandas as pd


def process(file_path):
    a = pd.read_csv(file_path, names=['date', 'unknown', 'open', 'high', 'low', 'close', 'volume'],
                    usecols=[2, 3, 4, 5, 6], header=None)
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
    return a.as_matrix().astype('float32')[200:].T

if __name__ == '__main__':
    print(process('./quantquote_daily_sp500/daily/table_abc.csv'))

