# coding: utf-8

import os

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns

import sklearn.linear_model as sklinear
import sklearn.tree as sktree
import sklearn.ensemble as skesb

from data_vendor import DataVendor


DATA_PATH = 'Data/future'
# DATA_PATH = r'G:\Study\Finance.Economics\quantitative finance\Data\中信建投Tick数据-毕业论文用\data'
TRADING_SESSIONS = [(90000, 113000),
                    (133000, 150000),
                    (210000, 230000)]


# ### 定义函数 用于切除数据集中每日开盘收盘附近的数据
# 
# **为什么去除**：
# 
# 因为计算price diff的时候会跨天计算，这部分数据要切除
# 
# **实现**：
# 
# 根据datetime index得到hour-minute-second表示，根据hms来判断True False

# In[19]:


# 函数定义 勿修改

def dt2hms(dt):
    hour = dt.hour
    minute = dt.minute
    sec = dt.second
    res = hour * 10000 + minute * 100 + sec
    return res


def hms2dt(n):
    return pd.to_datetime(n, format="%H%M%S")
    

def calc_high_vol_range(trading_sessions, start_len=120, end_len=120):
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


def is_high_vol(t, high_vol_range, start_len=120, end_len=120):
    in_start = False
    in_end = False
    
    for (start1, start2), (end1, end2) in high_vol_range:
        in_start = in_start or (t >= start1 and t <= start2)
        in_end = in_end or (t >= end1 and t <= end2)
    res = in_start or in_end
    return res

# ### 定义函数 用于去除涨跌停
# 
# **为什么要去除涨跌停**：
# 
# 涨跌停时bid ask price、qty会有为0的情况；
# 而FORWARD_WINDOW前的数据计算price move时，由于涨跌停价格失真，price move也是失真的，所以涨跌停前的部分也要去掉。
# 
# **实现**：
# 
# bid ask price与limit_up, limit_down比较，标注True False；用逻辑运算把之前的部分也标注。

# In[20]:


def mask_limit_reach(df, before=0, after=0):
    # when reaching limit_up, bidprice1 will be zero.
    limit_up = (df['bidprice1'] >= df['limit_up']) | (df['askqty1'] == 0)
    # when reaching limit_down, askprice1 will be zero.
    limit_down = (df['askprice1'] <= df['limit_down']) | (df['bidqty1'] == 0)
    res = np.logical_or(limit_up, limit_down)
    if before:
        res = res.rolling(window=before+1).apply(np.any).shift(-before)
    if after:
        res = res.rolling(window=after+1).apply(np.any)
        
    res = np.logical_not(res)
    return res


def add_ma(df, cols, lens=(3, 7)):
    dfs = []
    for n in lens:
        roll = df.reindex(columns=cols).rolling(window=n)
        df_ma = roll.mean()
        df_ma.columns = [s + '_ma{:d}'.format(n) for s in cols]
        dfs.append(df_ma)
    res = pd.concat(dfs, axis=1)
    return res


def add_diff(df, cols, lens=(3, 7)):
    dfs = []
    for n in lens:
        df_diff = df.reindex(columns=cols).diff(n)
        df_diff.columns = [s + '_diff{:d}'.format(n) for s in cols]
        dfs.append(df_diff)
    res = pd.concat(dfs, axis=1)
    return res


def train_test_split(arr, train_pct=0.65):
    len_train = int(len(x) * train_pct)
    arr_train = arr[: len_train]
    arr_test = arr[len_train: ]
    
    return arr_train, arr_test


def rolling_train_test(x, y, roll_len, train_pct):
    assert len(x) == len(y)
    tot_len = len(x)
    assert roll_len < tot_len
    
    train_len = int(roll_len * train_pct)
    test_len = roll_len - train_len
    
    for i in range(0, tot_len, test_len):
        x_train = x[i: (i + train_len)]
        y_train = y[i: (i + train_len)]
        x_test  = x[(i + train_len):  (i + train_len + test_len)]
        y_test  = y[(i + train_len):  (i + train_len + test_len)]
        yield i, x_train, y_train, x_test, y_test


def load_data():
    # ## 读取数据到`dataset`变量

    # In[4]:

    store = pd.HDFStore(os.path.join(DATA_PATH,
                        'rb1701_201608-201611_AddFeature.hd5'),
                        mode='r')
    dataset = store['dataset']
    store.close()

    assert dataset.shape[0] == 3245785

    # 预测时间跨度
    FORWARD_WINDOW = 60
    # 要使用的自变量
    X_COLS = [#'book_pressure_norm',
              #'datetime'
              'mid',
              'last', 'bidprice1', 'askprice1', 'bidqty1', 'askqty1', 'volume_diff', 'oi_diff',
             ]
    XY_COLS = X_COLS.copy()
    if 'mid' not in XY_COLS:
        XY_COLS.append('mid')

    xy = dataset.reindex(columns=XY_COLS)
    # 目标预测变量
    xy.loc[:, 'y'] = xy['mid'].diff(FORWARD_WINDOW).shift(-FORWARD_WINDOW)

    # In[14]:

    df_mas =  add_ma(dataset, ['volume_diff', 'oi_diff'], lens=[3, 7, 13, 29, 53])
    df_diffs = add_diff(dataset, ['mid'], lens=[3, 7, 13, 29, 53])

    # In[15]:

    X_COLS.extend(df_mas.columns)
    XY_COLS.extend(df_mas.columns)
    X_COLS.extend(df_diffs.columns)
    XY_COLS.extend(df_diffs.columns)
    xy = xy.join(df_mas)
    xy = xy.join(df_diffs)

    # In[16]:

    X_COLS.extend(['bp_aq', 'ap_bq'])
    XY_COLS.extend(['bp_aq', 'ap_bq'])
    xy['bp_aq'] = xy['bidprice1'] * xy['askqty1']
    xy['ap_bq'] = xy['askprice1'] * xy['bidqty1']

    store = pd.HDFStore(os.path.join(DATA_PATH,
                        'mask_all.hd5'),
                        mode='r')
    mask_all= store['mask_all']
    store.close()

    # ## 准备$X$, $Y$：mask后取出，并划分训练、测试集 (定义函数，按时间先后切)
    # rolling的意思是，每段训练、测试集是全集的一段，不断更新训练，这样更符合实际情况。
    #
    # 每隔长度为`test_len`长度的数据后，重新训练数据。每次训练、测试集长度不变

    # In[7]:


    # In[24]:


    # xym即xy_masked
    xym = xy.loc[mask_all]#.iloc[:]#1000000]

    # X, Y都是np.ndarray
    X = xym[X_COLS].values
    Y = xym['y'].values

    return X, Y


if __name__ == "__main__":
    # test
    import time
    t0 = time.time()

    x, y = load_data()

    t1 = time.time()

    print(x.shape, y.shape)
    print("Time to load data: {:.1f} sec.".format(t1 - t0))

