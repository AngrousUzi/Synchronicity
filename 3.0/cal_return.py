import pandas as pd
from get_data import get_data
from resample import resample
import datetime as dt
import warnings
import os
# 设置warnings过滤，将RuntimeWarning设置为ignore模式
warnings.filterwarnings('ignore', category=RuntimeWarning)

def cal_return(df,full_code) -> pd.Series:
    """
    计算股票收益率函数
    
    计算DataFrame按照指定频率重采样后的收益率，并删除隔夜收益率部分
    因为使用的是不复权数据，隔夜收益率无法准确计算
    
    参数:
    df (pd.DataFrame): 需要重采样的DataFrame，包含价格数据
    t_range: 数据中符合要求的时间范围
    freq (str): 重采样频率
    
    返回:
    pd.Series: 计算得到的收益率序列
    """
    
    # 旧版本的计算逻辑（已注释掉）
    # df_resampled = resample(df,t_range,freq)
    # # print(df_resampled)
    # df_resampled['Return']= df_resampled['Price'].pct_change()
    # 
    # # 这里需要丢弃昨日收盘到当日开盘区间的数据，因为该数据库所使用的数据都是不复权数据，隔夜收益率是无法计算的
    # # 而对于多日连续计算而言，当天第一个就是昨夜/今天
    # # 但是，对于第一天而言，没有昨日的数据，所以第一天的index理论上要保留
    # # 然而，对于该计算，第一天刚好是nan，因此也需要删除
    # first_day_index=df_resampled.resample('D').first()['original_time']
    # df_return=df_resampled.loc[~df_resampled['original_time'].isin(first_day_index),'Return']
    # return df_return
    
    # 对数据进行重采样
    # print(df_resampled)  # 调试用打印语句
    # df_resampled.to_excel('high_freq_data_raw.xlsx')  # 调试用导出数据
    
    # 计算价格变化率（收益率）

    df_return = df['Price'].pct_change()

    df_return = df_return[~(df_return.index.time == pd.Timestamp("09:15:00").time())]
    

    # 对于如果有null值的index，删除当天所有的数据
    # 修复bug：原代码试图用 difference 删除非空日期，但 drop 期望 index label，导致报错
    # 正确做法：删除所有含有NaN的日期（即这些日期的所有数据），而不是用 difference
    null_dates = df_return[df_return.isna()].index.normalize().unique()
    if len(null_dates) > 0:
        print(f"{full_code} with {len(null_dates)} null date(s): {list(null_dates.date)}")
        os.makedirs(f'error_list/return', exist_ok=True)
        with open(f'error_list/return/{full_code}.txt', 'a', encoding='utf-8') as f:
            for date in null_dates:
                f.write(f'{date.date()}\n')
        # 删除这些日期的所有数据
        mask = ~df_return.index.normalize().isin(null_dates)
        df_return = df_return[mask]
    return df_return

def get_complete_return(full_code:str,start:dt.datetime,end:dt.datetime,freq:str,workday_list:list=None,is_index:bool=False):
    exg=full_code[:2]
    num_code=full_code[2:]

    # 提前一天，不然会有first数据缺失，这里需要计算过程中自行注意，当遇到假期的时候依然有可能会失效
    start_tmp=pd.Timestamp(start)-dt.timedelta(days=4)
    
    df_get_data=get_data(start=start_tmp,end=end,exg=exg,full_code=full_code,workday_list=workday_list)
    
    df_stock=df_get_data[0]
    workday_list=df_get_data[1]
    error_list=df_get_data[2]
    
    if df_stock is None:
        return None,workday_list,error_list
    df_resample=resample(df_stock,freq=freq,is_index=is_index,stock_code=num_code,workday_list=workday_list,error_list=error_list)
    df_return=cal_return(df_resample,full_code)

    # 删除第一天
    df_return=df_return.loc[df_return.index.date>=start.date()]
    # workday_list = [day for day in workday_list if day >= start.date()]
    return df_return,workday_list,error_list


if __name__=="__main__":
    start=dt.datetime(2021,1,24)
    end=dt.datetime(2021,10,10)
    freq="12h"
    df_index_return,workday_list,error_list=get_complete_return(full_code="SH000001",start=start,end=end,freq=freq,workday_list=None,is_index=True)
    df_index_return.to_csv("index_return_test.csv")
    df_return,workday_list,error_list=get_complete_return(full_code="SH600905",start=start,end=end,freq=freq,workday_list=workday_list,is_index=False)
    # df_return.to_csv("return_test.csv")
    print(error_list)
    df_return.to_csv("return_test.csv")