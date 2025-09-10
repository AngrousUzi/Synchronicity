import pandas as pd
import numpy as np
from get_data import get_data,get_low_freq_data,get_high_freq_data
from resample import resample,convert_freq_to_day
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

    return df_return


def check_null(df,full_code, base_dir: str = ""):
    def _log(msg: str):
        error_dir = os.path.join(base_dir, 'error')
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
            f.write("cal_return: " + str(msg) + '\n')
    null_dates = df[df.isna()].index.normalize().unique()
    if len(null_dates) > 0:
        print(f"{full_code} with {len(null_dates)} null date(s): {list(null_dates.date)}")
        for date in null_dates:
            _log(f'Null return on {date.date()}')
        # 删除这些日期的所有数据
        mask = ~df.index.normalize().isin(null_dates)
        df = df[mask]
    return df

def deal_return(df_return,full_code,start,end,is_index,base_dir: str = ""):

    if is_index:
        df_return=index_check_overnight(df_return,full_code, base_dir=base_dir)

    df_return=check_null(df_return,full_code, base_dir=base_dir)

    # workday_list = [day for day in workday_list if day >= start.date()]
    return df_return

def get_complete_return(full_code:str,workday_list:list=None,is_index:bool=False, params=None):
    if params is None:
        raise ValueError("params is None")
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    base_dir=params["base_dir"]
    # 先在start.date()_end.date()/freq/中找
    date_str = f"{start.date()}_{end.date()}"
    search_dir = os.path.join(date_str, freq)
    file_path = os.path.join(search_dir, f"{full_code}.csv")

    if os.path.exists(file_path):
        df_return = pd.read_csv(file_path, index_col=0, parse_dates=True, squeeze=True)
        # 重新获取workday_list和error_list
        if workday_list is None or error_list is None:
            # 尝试读取txt文件
            txt_path = os.path.join(base_dir, date_str, "agg_raw", f"{full_code}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                workday_list = None
                error_list = None
                for line in lines:
                    if line.startswith("workday_list:"):
                        try:
                            workday_list = eval(line.split(":", 1)[1].strip())
                        except Exception:
                            workday_list = None
                    if line.startswith("error_list:"):
                        try:
                            error_list = eval(line.split(":", 1)[1].strip())
                        except Exception:
                            error_list = None
        return df_return, workday_list, error_list


    # 如果找不到，则重新计算
    exg=full_code[:2]
    num_code=full_code[2:]

    # 提前一天，不然会有first数据缺失，这里需要计算过程中自行注意，当遇到假期的时候依然有可能会失效

    if freq.endswith("min") or freq.endswith("h"):
        start_tmp=pd.Timestamp(start)-dt.timedelta(days=4)
        df_get_data=get_high_freq_data(start=start_tmp,end=end,exg=exg,full_code=full_code,workday_list=workday_list, base_dir=base_dir)
    else:
        start_tmp=pd.Timestamp(start)-dt.timedelta(days=convert_freq_to_day(freq))
        df_get_data=get_low_freq_data(start=start_tmp,end=end,exg=exg,full_code=full_code,workday_list=workday_list, base_dir=base_dir)

    # df_get_data=get_data(start=start_tmp,end=end,exg=exg,full_code=full_code,workday_list=workday_list, base_dir=base_dir)
    
    df_stock=df_get_data[0]
    workday_list=df_get_data[1]
    error_list=df_get_data[2]
    
    if df_stock is None:
        return None,workday_list,error_list
    df_resample=resample(df_stock,freq=freq,is_index=is_index,stock_code=num_code,workday_list=workday_list,error_list=error_list)
    df_return=cal_return(df_resample,full_code)

    # 删除start_tmp到start的日期
    df_return=df_return.loc[df_return.index.date>=start.date()]
    
    if freq.endswith("min") or freq.endswith("h"):
        df_return=deal_return(df_return,full_code,start,end,is_index,base_dir)

    return df_return,workday_list,error_list

def index_check_overnight(df,full_code, base_dir: str = ""):
    # 获取每日的第一个数据点
    first_of_day = df.groupby(df.index.date).head(1)
    
    # 检查哪些点的隔夜收益率缺失（近似为0）
    missing_overnight_mask = np.abs(first_of_day) < 0.00001
    
    # 获取需要更新的行的完整时间戳索引
    indices_to_update = first_of_day[missing_overnight_mask].index

    # 如果没有需要更新的行，直接返回
    if indices_to_update.empty:
        return df
    error_dir = os.path.join(base_dir, 'error')
    os.makedirs(error_dir, exist_ok=True)
    for date in indices_to_update:
        with open(os.path.join(error_dir, f'{full_code}.txt'),'a',encoding='utf-8') as f:
            f.write(f"Overnight return missing on {date.date()}\n")
    print(f"For {full_code}, overnight return from {indices_to_update[0].date()} to {indices_to_update[-1].date()} need to be replaced.")
    # 加载预先计算好的隔夜收益率数据
    return_overnight_code="H"+full_code[3:]
    df_overnight_recal=pd.read_csv(os.path.join("index","return",f"{return_overnight_code}.csv"))
    df_overnight_recal=df_overnight_recal.set_index("date")
    df_overnight_recal.index=pd.to_datetime(df_overnight_recal.index)
    df_overnight_recal.index=df_overnight_recal.index+pd.Timedelta(hours=9)+pd.Timedelta(minutes=25)
    
    # 使用精确的时间戳索引来获取替换值
    # 为了安全起见，我们只选择df_overnight_recal中与indices_to_update匹配的索引
    valid_indices = indices_to_update.intersection(df_overnight_recal.index)
    replacement_values = df_overnight_recal.loc[valid_indices, 'overnight_return']
    # print(valid_indices)
    # 更新原始的收益率序列
    df.loc[valid_indices] = replacement_values
    return df

if __name__=="__main__":
    start=dt.datetime(2019,1,24)
    end=dt.datetime(2025,6,10)
    freq="12h"
    df_index_return,workday_list,error_list=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=None,is_index=True)
    df_index_return.to_csv("index_return_test.csv")
    df_return,workday_list,error_list=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=workday_list,is_index=False)
    # df_return.to_csv("return_test.csv")
    print(error_list)
    df_return.to_csv("return_test.csv")