import pandas as pd
import numpy as np
import datetime as dt

from utils import convert_freq_to_day,log_error


def resample_low_freq(df,freq,is_index,stock_code,workday_list,error_list):
    """
    低频数据重采样函数
    
    对股票或指数的日频或更低频率数据进行重采样。适用于日线、周线等低频数据处理。
    处理停牌情况，通过前向填充确保连续性。
    
    参数:
    df (pd.DataFrame): 原始价格数据，包含'Price'列，index为日期时间
    freq (str): 重采样频率，如"1d"、"2w"等低频格式
    is_index (bool): 是否为指数数据（当前版本中此参数对低频数据处理无影响）
    stock_code (str): 股票代码，用于错误日志记录
    workday_list (list): 工作日列表，包含datetime.date对象
    error_list (list): 错误日期列表，包含有问题的日期
    
    返回:
    pd.DataFrame: 重采样后的数据，包含以下特性：
                  - index: pd.DatetimeIndex，按指定频率的日期
                  - columns: ['Price']，包含重采样后的价格数据
                  - 停牌期间使用前向填充的历史价格
    
    处理逻辑:
    1. 先按日频重采样并前向填充，处理停牌情况
    2. 根据指定频率从工作日列表中选择采样点
    3. 对缺失数据进行前向填充
    """
    
    #这两行主要是为了确保停牌的时候采集到的是历史中最近的一个交易日的数据
    df=df.resample('D').last()
    df=df.ffill()

    df_resampled = pd.DataFrame(index=workday_list[::convert_freq_to_day(freq)])
    df_resampled['Price'] = df.loc[df.index.isin(df_resampled.index)]['Price']
    df_resampled=df_resampled.ffill()
    df_resampled.index=pd.to_datetime(df_resampled.index)
    return df_resampled


def resample_high_freq(df,freq,is_index,stock_code,workday_list,error_list):
    """
    高频数据重采样函数
    
    处理分钟级和小时级的股票数据重采样，考虑中国股市的特殊交易时段结构。
    对个股数据进行分红除权调整，确保价格数据的连续性和准确性。
    
    参数:
    df (pd.DataFrame): 原始高频数据
                       - index: pd.DatetimeIndex，精确到分钟或更细粒度的时间戳
                       - columns: ['Price']，原始价格数据
    freq (str): 高频重采样频率，支持：
                - "3min", "5min", "10min", "15min", "30min": 分钟级频率
                - "12h": 日内连续交易时段（相当于一个交易日的总体）
    is_index (bool): 数据类型标识
                     - True: 指数数据，使用原始价格作为除权参考价
                     - False: 个股数据，需要进行分红除权价格调整
    stock_code (str): 股票数字代码（如"000001"），用于：
                      - 查找分红数据：dividend/{stock_code}.csv
                      - 错误日志记录和调试
    workday_list (list): 有效工作日列表，包含datetime.date对象
                         用于过滤非交易日数据
    error_list (list): 问题日期列表，包含数据质量有问题的日期
                       这些日期的数据会被排除
    
    返回:
    pd.DataFrame: 重采样后的高频数据，包含以下特征：
                  - index: pd.DatetimeIndex，按以下时间点组织：
                    * 09:15: 除权参考价（个股）或当日首价（指数）
                    * 09:25: 集合竞价结果价格
                    * 09:30-11:29, 13:00-14:59: 连续交易时段的重采样价格
                    * 15:00: 收盘价
                  - columns: ['Price']，经过以下处理的价格：
                    * 个股：已进行分红除权调整
                    * 指数：保持原始价格
                    * 缺失数据进行前向填充
                    * 异常值（<0.00001）设为NaN后前向填充
    
    处理逻辑:
    1. 关键时点价格提取：
       - 收盘价：每日最后一个价格
       - 除权参考价：个股前收盘价调整，指数为首个价格
       - 集合竞价价格：9:25前最后一个价格
    2. 连续交易时段重采样（避开集合竞价时段）
    3. 个股分红调整：读取dividend文件进行除权计算
    4. 数据清洗：过滤非交易日、异常值处理、前向填充
    
    分红调整公式（仅个股）:
    除权价 = (前收盘价 - 现金分红) / (1 + 股票分红比例 + 转增比例)
    
    注意事项:
    - 个股需要dividend/{stock_code}.csv文件存在
    - 时间戳已调整为标准交易时点
    - 每日最后一个连续交易价格被收盘价替代以确保准确性
    """
    df_stock = df.copy()
    # df_stock[freq] = df_stock.index.time

    
    # 处理收盘价（15:00）
    # 按日取最后一个价格作为收盘价
    df_close = df_stock.resample('D').last()
    # 注释掉这一行是因为如果当天有get_data的数据，则一定当天会形成价格。
    # df_close=df_close.ffill()
    

    #处理关键时间段数据first(除权后的初始数据)
    if is_index:
        df_first = df_stock.resample('D').first()
        
        df_open = df_stock[df_stock.index.time<pd.to_datetime("09:25:59").time()].resample('D').last()
    
    else:
        # 个股进行分红处理
        df_dividend=pd.read_csv(f'dividend/{stock_code}.csv')
        df_dividend.index=pd.to_datetime(df_dividend['Exdistdt'])
        df_dividend=df_dividend.fillna(0)
        df_first=df_close[['Price']].shift(1)
        # print(df_first)
        
        start=df_close.index[0]
        end=df_close.index[-1]
        ex_dates=df_dividend[(df_dividend.index>=start)&(df_dividend.index<=end)].index.unique().tolist()
        # print(ex_dates)
        for date in ex_dates:
            # print(date)
            # print(df_first.loc[date,'Price'])
            df_first.loc[date,'Price']=(df_first.loc[date,'Price']-(df_dividend.loc[date,'Btperdiv']).sum())/(1+df_dividend.loc[date,'Perspt'].sum()+df_dividend.loc[date,'Pertran'].sum())
            # print(df_dividend.loc[date,'Btperdiv'].sum())
            # print(df_dividend.loc[date,'Perspt'].sum())
            # print(df_dividend.loc[date,'Pertran'].sum())
            # print(df_first.loc[date,'Price'])

        df_open = df_stock[df_stock.index.time<pd.to_datetime("09:25:59").time()].resample('D').last()
    
        #对于当日集合竞价阶段没有成交的情况，补0，之后会前项填充
        lack_open_index=df_close.index.difference(df_open.index)
        log_error(f"stock_code:{stock_code},lack_open_index:{lack_open_index}",stock_code,base_dir="")
        for date in lack_open_index:
            df_open.loc[date, 'Price'] = 0
        # df_open.loc[lack_open_index,'Price']=0

    df_first.index = df_first.index + dt.timedelta(hours=9) + dt.timedelta(minutes=15)
    df_open.index= df_open.index + dt.timedelta(hours=9) + dt.timedelta(minutes=25)
    df_close.index = df_close.index + dt.timedelta(hours=15)
    # 处理连续交易时段的数据
    # 按指定频率重采样，取每个时间窗口的最后一个价格
    # print(df_open.index)
    df_continuous = df_stock.resample(freq).last()
    # print(df_continuous.index)
    df_continuous = df_continuous.ffill()
    

    # 筛选连续交易时段：
    # 上午: 9:30-11:30
    # 下午: 13:00-14:59 (避开收盘集合竞价14:57-15:00
    # 时间选择11:29:59会导致一部分微小的信息丢失问题
    df_continuous = df_continuous[((df_continuous.index.time >= pd.to_datetime("09:30:00").time()) & 
                                (df_continuous.index.time < pd.to_datetime("11:29:59").time())) |
                                  ((df_continuous.index.time >= pd.to_datetime("13:00:00").time()) & 
                                   (df_continuous.index.time < pd.to_datetime("14:59:59").time()))]
    
    # 丢弃每一天最后一个df_continuous,用close替代
    last_idx_per_day = df_continuous.groupby(df_continuous.index.date).tail(1).index
    df_continuous = df_continuous.drop(last_idx_per_day)
    # 合并所有时段的数据
    df_resampled = pd.concat([df_first,df_open,df_continuous,df_close])
    df_resampled.sort_index(inplace=True)

    df_date_index = df_resampled.index[df_resampled.index.map(lambda x: x.date() in workday_list and x.date() not in error_list)]
    df_resampled=df_resampled[df_resampled.index.isin(df_date_index)]
    df_resampled.loc[df_resampled['Price']<0.00001,'Price']=np.nan
    df_resampled.ffill(inplace=True)
    df_resampled.index=pd.to_datetime(df_resampled.index)
    # df_stock=df_stock[df_stock[freq].isin(t_range.time)]  # 原有过滤逻辑，已注释
    
    # 数据清洗：处理集合竞价未形成价格的情况
    # 该数据库对于个股而言，如果集合竞价没有形成价格，Price填充为0
    # 对于填充为0的Price，将其设为NaN以便后续前向填充
    # if (np.abs(df_resampled['Price'] - 0) < 0.000001).any():
    #     print(df_resampled[np.abs(df_resampled['Price'] - 0) < 0.000001])
    #     df_resampled.loc[np.abs(df_resampled['Price'] - 0) < 0.000001, 'Price'] = np.nan
    
    # 对于没有形成开盘价的股票和freq内没有交易的股票，前向填充前一次交易的价格
    # df_resampled['Price'] = df_resampled['Price'].ffill()
    


    return df_resampled
    

def resample(df,freq,is_index,stock_code,workday_list,error_list):
    """
    股票和指数数据重采样主函数
    
    根据频率类型自动选择高频或低频重采样方法。处理中国股市的特殊交易时段，
    包括早盘集合竞价、连续交易时段和收盘价，同时支持个股分红调整。
    
    参数:
    df (pd.DataFrame): 需要重采样的原始数据
                       - index: pd.DatetimeIndex，时间戳
                       - columns: ['Price']，价格数据
    freq (str): 重采样频率，支持以下格式：
                - 高频："3min", "5min", "10min", "15min", "30min", "12h"
                - 低频："1d", "2w" 等
    is_index (bool): 是否为指数数据
                     - True: 指数数据，无需分红调整
                     - False: 个股数据，需要进行分红除权处理
    stock_code (str): 股票数字代码（不含交易所前缀），用于：
                      - 查找分红数据文件 dividend/{stock_code}.csv
                      - 错误日志记录
    workday_list (list): 工作日列表，包含datetime.date对象
                         用于确定有效交易日期
    error_list (list): 错误日期列表，包含有问题的日期
                       这些日期的数据会被排除
    
    返回:
    pd.DataFrame: 重采样后的数据，具有以下特征：
                  - index: pd.DatetimeIndex，按指定频率的时间点
                  - columns: ['Price']，调整后的价格数据
                  - 对于高频数据：包含开盘价、连续交易时段价格和收盘价
                  - 对于个股：已进行分红除权调整
                  - 缺失数据已进行前向填充
    
    处理逻辑:
    - 高频数据（min/h结尾）：调用resample_high_freq处理日内数据
    - 低频数据（d/w结尾）：调用resample_low_freq处理日线以上数据
    """

    if freq.endswith("min") or freq.endswith("h"):
        return resample_high_freq(df,freq,is_index,stock_code,workday_list,error_list)
    else:
        return resample_low_freq(df,freq,is_index,stock_code,workday_list,error_list)

if __name__=="__main__":    
    from get_data import get_data
    import os
    from get_cache import get_cache_text
    # os.makedirs('test',exist_ok=True)
    # df_all=get_data(start=dt.datetime(2024,2,24),end=dt.datetime(2024,6,10),exg="SH",full_code="SH603392")
    df_all=[]
    df_all.append(pd.read_csv("test/SZ300481.csv",parse_dates=True,index_col="Time"))
    df_all[0].index=pd.to_datetime(df_all[0].index)
    df_all.extend(list(get_cache_text("SZ300481",dt.datetime(2010,1,5),dt.datetime(2025,6,30))))
    # print(df_all.index)
    df_r=resample(df_all[0],freq="5min",is_index=False,stock_code="300481",workday_list=df_all[1],error_list=df_all[2])
    df_r.to_csv(os.path.join("test","resample_test.csv"))