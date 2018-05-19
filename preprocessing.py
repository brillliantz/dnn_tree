# encoding: utf-8

import re
import os
import sys
import codecs
import bz2

import numpy as np
import pandas as pd

JAQS_DIR = r'G:\Working\Junzhi\OpenSource\JAQS'
sys.path.append(JAQS_DIR)

import jaqs.util as jutil
from jaqs.data import RemoteDataService


HIGH_VOL_SECONDS = 60
KNOWN_COLS = ['symbol', 'name', 'time', 'last', 'high',
              'low', 'volume_diff', 'volume', 'Unknown0', 'preoi',
              'oi', 'oi_diff', 'turnover', 'limit_up', 'limit_down',
              'close', 'preclose', 'askprice1', 'bidprice1', 'askvolume1',
              'bidvolume1', 'open']
DATA_ROOT = r'G:\Study\Finance.Economics\quantitative finance\Data\UndergraduateThesis'
# TODO: different trading session for different instruments
TRADING_SESSIONS = [(90000, 113000),
                    (133000, 150000),
                    (210000, 230000)]
MARKET_CLOSE = [150000, 230000]


######################################################
# Read file and elementary pre-process (for backtest)

def read_to_df(fn):
    """
    
    Parameters
    ----------
    fn : str
        File_name of the BZ2/CSV file.

    """
    if fn.endswith('bz2'):
        fbuffer = bz2.BZ2File(fn)
    elif fn.endswith('csv'):
        fbuffer = codecs.open(fn, mode='r', encoding='utf-8')
    else:
        raise NotImplementedError("file type {:s}".format(fn))
    df = pd.read_csv(fbuffer, header=-1)
    return df


def read_daily_csv(symbol, trade_date, data_root):
    """
    
    Parameters
    ----------
    symbol : str
        rb1801.SHF
    trade_date : int
        20180101

    Returns
    -------
    df : pd.DataFrame
        Raw DataFrame read from the file.

    """
    month = trade_date // 100
    symbol_no_exchange, exchange = symbol.split('.')
    inst = re.findall(r'[a-z,A-Z]+', symbol_no_exchange)[0]
    delivery_month = re.findall(r'\d+', symbol_no_exchange)[0]
    
    fn = "{}_{}.csv".format(symbol_no_exchange, trade_date)
    fp = os.path.join(data_root, str(month), inst, fn)
    # file can be in .csv or .csv.bz2 format
    if not os.path.exists(fp):
        fp = fp + '.bz2'
    
    df = read_to_df(fp)
    
    return df


def pre_process_df(df, col_names):
    """
    Remove unnecessary columns; add date, time, trade_date.
    output is just like what we received from the real-time market data parser.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    col_names : list of str

    Returns
    -------
    res : pd.DataFrame

    """
    n_col = len(col_names)
    res = df.iloc[:, :n_col]
    res.columns = col_names
    
    res = res.drop(['name', 'Unknown0'], axis=1)
    
    res.loc[:, 'datetime'] = pd.to_datetime(res['time'], format="%Y-%m-%d %H:%M:%S.%f")
    
    res.loc[:, 'time'] = jutil.convert_datetime_to_int_time(res['datetime'])
    res.loc[:, 'date'] = jutil.convert_datetime_to_int(res['datetime'])
    # TODO: time after 20:00 will be next trade_date
    res.loc[:, 'trade_date'] = res['date']
    
    res.loc[:, 'symbol'] = res['symbol'].str.lower()
    res = res.drop('datetime', axis=1)
    
    return res


class RemoteDataService2(RemoteDataService):
    def tick(self, symbol, trade_date):
        """
        
        Parameters
        ----------
        symbol : str
            rb1801.SHF
        trade_date : int
            20180101

        Returns
        -------

        """
        df = read_daily_csv(symbol, trade_date, data_root=DATA_ROOT)
        res = pre_process_df(df, KNOWN_COLS)
        msg = '0,'
        
        return res, msg


######################################################
# Advanced pre-process to clean data (for statistical learning)
'''
对数据集index为[0, 1, 2, ..., n].

若idx=k处数据不可用，则
  计算forward return需要用后面forward_len长度的数据：[k-forward_len, k]总长(forward_len + 1)的数据点都不可用
  学习时用到前面总长backward_len的数据，则[k, k+backward_len-1]总长backward_len的数据点都不可用

15:00和21:00的两个session由于有开盘竞价，所以不可连一起；
23:00和09:00的两个session由于夜里发生很多事件，所以不可连一起；
因此09:00-15:00, 21:00-23:00是两个“连续session”，分别处理。
每一段的idx=-1和idx=n+1两个点可以认为是不可用，
因而开头[0, backward-2]不可用，结尾[n+1-forward_len, n]不可用。

理想处理方法:
1. 把两个“连续session”之间增加一个dummy data point，并标记为不可用，
2. 将所有不可用点的前后也mask上
3. 循环的时候只循环可用index

改进处理方法 - 把第一步改为：
1. 直接把收盘的那个data point标记为不可用。（因为接近收盘都平仓了，不需要继续预测）


'''


def is_valid_trading_hour(time_):
    """
    
    Parameters
    ----------
    time_ : int
        140321, 90100, etc.

    Returns
    -------
    bool

    """
    global TRADING_SESSIONS
    flag = False
    for start, end in TRADING_SESSIONS:
        flag = flag or (start <= time_ <= end)
    return flag


def add_mid_price(df):
    df.loc[:, 'mid'] = (df['bidprice1'] + df['askprice1']) / 2.0
    return df
    

def pre_process_df2(df):
    valid_trading_hour_mask = (df['time'] // 1000).apply(is_valid_trading_hour)
    df = df.reindex(index=valid_trading_hour_mask.index[valid_trading_hour_mask.values])
    
    df = add_mid_price(df)
    return df


######################################################
# Transforms
# Do transforms on everyday data to get daily (features, labels)
# pair and combine them for further statistical learning

def dt2hms(dt):
    hour = dt.hour
    minute = dt.minute
    sec = dt.second
    res = hour * 10000 + minute * 100 + sec
    return res


def hms2dt(n):
    return pd.to_datetime(n, format="%H%M%S")


def calc_high_vol_time_range(trading_sessions, start_len=120, end_len=120):
    """
    
    Parameters
    ----------
    trading_sessions
    start_len : int
        Unit: seconds
    end_len : int
        Unit: seconds

    Returns
    -------
    list of (tuple ,tuple)

    """
    delta_start = pd.Timedelta(seconds=start_len)
    delta_end = pd.Timedelta(seconds=end_len)
    
    res = []
    for start, end in trading_sessions:
        start_dt = hms2dt(start)
        start_dt2 = start_dt + delta_start
        start2 = dt2hms(start_dt2)
        
        end_dt = hms2dt(end)
        end_dt2 = end_dt - delta_end
        end2 = dt2hms(end_dt2)
        
        res.append(((start, start2), (end2, end)))
    return res


def is_high_vol(t, high_vol_range):
    in_start = False
    in_end = False
    
    for (start1, start2), (end1, end2) in high_vol_range:
        in_start = in_start or (start1 <= t <= start2)
        in_end = in_end or (end1 <= t <= end2)
    res = in_start or in_end
    return res


def get_mask_high_vol(df, high_vol_sec=60):
    global TRADING_SESSIONS
    high_vol_range = calc_high_vol_time_range(TRADING_SESSIONS,
                                              start_len=high_vol_sec,
                                              end_len=high_vol_sec)
    
    mask = (df['time'] // 1000).apply(is_high_vol, high_vol_range=high_vol_range)
    return mask


def get_mask_limit_reach(df, before=0, after=0):
    """
    Return boolean Series. True means limit_reaching.
    
    Parameters
    ----------
    df : pd.DataFrame
    before : int
        Extra length of Data to mask before limit-reaching.
    after : int
        Extra length of Data to mask after limit-reaching.

    Returns
    -------
    pd.Series

    """
    # when reaching limit_up, bidprice1 will be zero.
    limit_up = (df['bidprice1'] >= df['limit_up']) | (df['askvolume1'] == 0)
    # when reaching limit_down, askprice1 will be zero.
    limit_down = (df['askprice1'] <= df['limit_down']) | (df['bidvolume1'] == 0)
    res = np.logical_or(limit_up, limit_down)
    if before:
        res = res.rolling(window=before + 1).apply(np.any).shift(-before)
    if after:
        res = res.rolling(window=after + 1).apply(np.any)
    
    # res = res.fillna(1.0)
    # res = res.astype(bool)
    return res


def get_mask_market_close(df):
    """
    df must be market data of a SINGLE natural/trade date.
    
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------

    """
    global MARKET_CLOSE
    
    res = pd.Series(index=df.index, data=False)
    for close_time in MARKET_CLOSE:
        mask = (df['time'] // 1000) <= close_time
        last_idx = mask.index[mask.values][-1]
        res.loc[last_idx] = True
        
    return res


######################################################
# Unit tests

def test_elementary_preprocess():
    from jaqs.data.basic import Quote
    
    df = read_daily_csv('rb1701.SHF', 20161012, data_root=DATA_ROOT)
    res = pre_process_df(df, KNOWN_COLS)
    
    ds = RemoteDataService2()
    res, _ = ds.tick('rb1701.SHF', 20161012)

    l = Quote.create_from_df(res)
    print(len(l))
    print(res.shape)
    print(res.columns)


def test_advanced_preprocess():
    df = read_daily_csv('rb1701.SHF', 20161012, data_root=DATA_ROOT)
    res = pre_process_df(df, KNOWN_COLS)
    print(res.shape)
    # print(res.columns)
    
    res = pre_process_df2(res)
    print(res.shape)
    # print(res.columns)


def process_daily_symbol_data(symbol, trade_date):
    global DATA_ROOT, HIGH_VOL_SECONDS
    
    df = read_daily_csv(symbol, trade_date, data_root=DATA_ROOT)
    market_data = pre_process_df(df, KNOWN_COLS)
    print(market_data.shape)
    # print(res.columns)
    
    valid_data = pre_process_df2(market_data)
    print(valid_data.shape)
    # print(res.columns)
    
    mask_high_vol = get_mask_high_vol(valid_data, high_vol_sec=HIGH_VOL_SECONDS)
    mask_limit_reach = get_mask_limit_reach(valid_data,
                                            before=0,
                                            after=0)
    mask_market_close = get_mask_market_close(valid_data)
    
    assert len(mask_high_vol) == len(mask_market_close)
    assert len(mask_high_vol) == len(mask_limit_reach)
    
    print("{:5d}, {:.2f}% of the whole data is cut off because of high volatility.".format(
            sum(mask_high_vol), sum(mask_high_vol) * 1.0 / len(mask_high_vol)
    ))
    print("{:5d}, {:.2f}% of the whole data is cut off because of limit reach.".format(
            sum(mask_limit_reach), sum(mask_limit_reach) * 1.0 / len(mask_limit_reach)
    ))
    print("{:5d} data points marked as market_close.".format(
            sum(mask_market_close)
    ))
    
    dirty_index = pd.DataFrame(data={'high_vol'    : mask_high_vol,
                                     'limit_reach' : mask_limit_reach,
                                     'market_close': mask_market_close}).any(axis=1)
    valid_data.loc[:, 'dirty_index'] = dirty_index
    # daily data pre-processing is done.
    # 到这里位置每天数据处理完毕，可把多日同一合约的处理后数据连起来，然后进行下方的before、after操作，
    # 即可得到clean_index

    # valid_data.to_hdf('{symbol:s}_{trade_date:8d}.hd5'.format(symbol=symbol,
    #                                                           trade_date=trade_date),
    #                   key='valid_data')
    # valid_data.to_msgpack('tmp.msgpk')
    
    return valid_data
    
    
def process_range_symbol_data(symbol_prefix, start_date, end_date,
                              days_to_list=30, front_months=None,
                              backward_len=224, forward_predict_len=60):
    """
    
    Parameters
    ----------
    symbol_prefix : str
        'rb'
    start_date : int
        20180104
    end_date : int
        20180104
    backward_len : int
    forward_predict_len : int

    Returns
    -------
    pd.DataFrame

    """
    # Configs
    from jaqs.data.continue_contract import get_map
    from example.eventdriven.config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    # trade_config = jutil.read_json(TRADE_CONFIG_PATH)

    # DataService
    ds = RemoteDataService()
    ds.init_from_config(data_config)

    # Get (trade_date -> front month contract) map
    if front_months is None:
        front_months = ['01', '05', '10']
    df_map = get_map(ds, symbol_prefix,
                     start_date=start_date, end_date=end_date,
                     days_to_delist=days_to_list,
                     front_months=front_months)
    df_map = df_map.set_index('trade_date')

    # Get daily pre-processed data and dirty_index
    valid_data_list = []
    count = 0
    for trade_date, row in df_map.iterrows():
        count += 1
        symbol = row['symbol']
        
        print("=> Processing {:10s} on {:8d}".format(symbol, trade_date))
        valid_data = process_daily_symbol_data(symbol=symbol, trade_date=trade_date)
        print(valid_data.shape)
        print("=> Processed  {:10s} on {:8d}".format(symbol, trade_date))
        
        valid_data_list.append(valid_data)

    valid_data = pd.concat(valid_data_list, axis=0)
    valid_data.index = np.arange(valid_data.shape[0])

    dirty_index = valid_data['dirty_index']
    print("Before roll: {:5d}, {:.2f}% data points are cut-off in total".format(
            sum(dirty_index), sum(dirty_index) * 1.0 / len(dirty_index)
    ))
    dirty_before = dirty_index.rolling(window=forward_predict_len + 1).apply(np.any).shift(-forward_predict_len)
    dirty_after = dirty_index.rolling(window=backward_len).apply(np.any)
    dirty_before = dirty_before.fillna(1.0).astype(bool)
    dirty_after = dirty_after.fillna(1.0).astype(bool)
    dirty_index_new = pd.DataFrame(data={'original': dirty_index,
                                         'before'  : dirty_before,
                                         'after'   : dirty_after}).any(axis=1)
    
    print(" After roll: {:5d}, {:.2f}% data points are cut-off in total".format(
            sum(dirty_index_new), sum(dirty_index_new) * 1.0 / len(dirty_index_new)
    ))
    print("Result data shape: {}, n_trade_days: {:d}".format(valid_data.shape, count))

    assert valid_data.shape[0] == dirty_index_new.shape[0]
    valid_data.loc[:, 'dirty_index'] = dirty_index_new
    
    # Dump to local file
    valid_data.to_hdf('{symbol_prefix:s}_{start_date:8d}_{end_date:8d}.hd5'.format(symbol_prefix=symbol_prefix,
                                                                                   start_date=start_date,
                                                                                   end_date=end_date),
                      key='valid_data')
    
    # Get clean_data
    # clean_data = valid_data.drop(index=dirty_index_new.index[dirty_index_new.values])
    # print(clean_data.shape)
    # print(res.columns)
    
    return valid_data


if __name__ == "__main__":
    import time
    t1 = time.time()
    
    n_loops = 1
    for _ in range(n_loops):
        # test_advanced_preprocess()
        # test_transforms()
        # process_daily_symbol_data('rb1701.SHF', 20161012)
        res = process_range_symbol_data('rb', 20160824, 20160831,
                                        days_to_list=30, front_months=['01', '05', '10'],
                                        backward_len=224, forward_predict_len=60)
    
    t = (time.time() - t1) / n_loops
    print("{:5d} loops in total. Time per loop: {: 4.3f} sec. ".format(n_loops, t))
