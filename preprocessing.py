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
    
    res = res.fillna(1.0)
    res = res.astype(bool)
    return res


######################################################
# Unit tests

def test_elementary_preprocess():
    from jaqs.data.basic import Quote
    
    df = read_daily_csv('rb1610.SHF', 20160912, data_root=DATA_ROOT)
    res = pre_process_df(df, KNOWN_COLS)
    
    ds = RemoteDataService2()
    res, _ = ds.tick('rb1610.SHF', 20160912)

    l = Quote.create_from_df(res)
    print(len(l))
    print(res.shape)
    print(res.columns)


def test_advanced_preprocess():
    df = read_daily_csv('rb1610.SHF', 20160912, data_root=DATA_ROOT)
    res = pre_process_df(df, KNOWN_COLS)
    print(res.shape)
    # print(res.columns)
    
    res = pre_process_df2(res)
    print(res.shape)
    # print(res.columns)


def test_transforms():
    df = read_daily_csv('rb1610.SHF', 20160912, data_root=DATA_ROOT)
    res = pre_process_df(df, KNOWN_COLS)
    print(res.shape)
    # print(res.columns)
    
    res = pre_process_df2(res)
    print(res.shape)
    # print(res.columns)
    
    '''
    calculation order:
        1. calculate y: forward return
        2. drop masked data points
        3. use BACKWARD_LEN data as input to model per data point
    '''
    BACKWARD_LEN = 250  # in length
    FORWARD_PREDICT_SEC = 60  # in seconds
    TICKS_PER_SECOND = 2
    
    HIGH_VOL_SECONDS = 60
    mask_high_vol = get_mask_high_vol(res, high_vol_sec=HIGH_VOL_SECONDS)
    mask_limit_reach = get_mask_limit_reach(res,
                                            before=TICKS_PER_SECOND * FORWARD_PREDICT_SEC,
                                            after=BACKWARD_LEN)

    print("{:.2f}% of the whole data is cut off because of high volatility.".format(
            sum(mask_high_vol) * 1.0 / len(mask_high_vol)
    ))
    print("{:.2f}% of the whole data is cut off because of limit reach.".format(
            sum(mask_limit_reach) * 1.0 / len(mask_limit_reach)
    ))
    mask_all = np.logical_and(mask_high_vol, mask_limit_reach)

    res = res.drop(index=mask_all.index[mask_all.values])
    print(res.shape)
    # print(res.columns)


if __name__ == "__main__":
    import time
    t1 = time.time()
    
    n_loops = 6
    for _ in range(n_loops):
        # test_advanced_preprocess()
        test_transforms()
    
    t = (time.time() - t1) / n_loops
    print("{:5d} loops in total. Time per loop: {: 4.3f} sec. ".format(n_loops, t))
