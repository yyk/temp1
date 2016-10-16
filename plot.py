import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dir_root = "./quantquote_daily_sp500/preprocessed/"
# d = pd.read_csv(dir_root + "table_nflx.csv")
d = pd.read_csv(dir_root + "table_aapl.csv")
# plt.plot(d['close_ln_scaled'][-200:])
# plt.plot(d['close_ema5_ln_scaled'][-200:])
# plt.plot(d['close_sma20_ln_scaled'][-200:])

# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(d['macd'])
plt.plot(d['macd'][-200:])
plt.plot(d['macds'][-200:])
plt.plot(d['macdh'][-200:])
#
# scaler = MinMaxScaler(feature_range=(-1, 1))
# plt.plot(scaler.fit_transform(d['close']))
# plt.plot(d['macds'])
# plt.plot(d['macdh'])
# plt.plot(d['rsi14'])
plt.show()