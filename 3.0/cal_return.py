import pandas as pd
import numpy as np
from get_data import get_data,get_low_freq_data,get_high_freq_data
from resample import resample
import datetime as dt
import warnings
import os
from get_cache import get_cache_text,get_cache_price,get_cache_return
from utils import convert_freq_to_day

# 设置warnings过滤，将RuntimeWarning设置为ignore模式
warnings.filterwarnings('ignore', category=RuntimeWarning)


def cal_return(df,full_code) -> pd.Series:
    """
    计算股票或指数收益率函数
    
    从重采样后的价格数据计算百分比收益率，自动排除分红除权收益率（09:15时点）。

    
    参数:
    df (pd.DataFrame): 重采样后的价格数据，必须包含以下特征：
                       - index: pd.DatetimeIndex，包含交易时间戳
                       - columns: ['Price']，调整后的价格数据
                       - 数据应来自resample函数的输出
    full_code (str): 完整股票或指数代码（如"SH000300"），用于调试和错误记录
    
    返回:
    pd.Series: 收益率时间序列，具有以下特征：
               - index: pd.DatetimeIndex，与输入数据对应但排除09:15时点
               - values: float，百分比收益率 (当期价格/前期价格 - 1)
               - 排除了隔夜收益率，仅包含日内交易时段的收益率
               - 第一个观测值为NaN（无法计算前一期收益率）
    
    处理逻辑:
    1. 使用pandas的pct_change()计算相邻时点的收益率
    2. 自动排除09:15时点的数据（隔夜跳空，不复权数据不准确）
    3. 保留所有其他时点：09:25集合竞价、日内连续交易、15:00收盘价
    
    注意事项:
    - 输入数据假设已经过除权调整（对于个股）

    - 返回的收益率可用于后续的回归分析和风险建模
    
    示例:
    >>> # 假设df是5分钟重采样后的价格数据
    >>> returns = cal_return(df, "SH000300")
    >>> print(returns.index.time.unique())  # 不包含09:15:00
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
    """
    检查并处理收益率数据中的空值
    
    识别包含NaN值的交易日期，记录问题并删除这些日期的所有数据。
    这样可以避免单日内部分时点缺失导致的分析偏差。
    
    参数:
    df (pd.Series): 收益率时间序列
                    - index: pd.DatetimeIndex，包含交易时间戳
                    - values: float，可能包含NaN的收益率数据
    full_code (str): 完整的股票或指数代码（如"SH000300"）
                     用于错误日志记录和调试输出
    base_dir (str, optional): 基础目录路径，默认为空字符串
                              错误日志将保存到 base_dir/error/{full_code}.txt
    
    返回:
    pd.Series: 清洗后的收益率数据，具有以下特征：
               - 删除了包含NaN值的完整交易日期
               - 保持原有的时间索引结构
               - 确保数据的完整性和分析的可靠性
    
    副作用:
    - 在控制台输出包含空值的日期信息
    - 在base_dir/error/{full_code}.txt中记录详细的空值日期
    - 自动创建错误日志目录（如果不存在）
    
    处理逻辑:
    1. 识别所有包含NaN值的日期（使用normalize()获取日期部分）
    2. 记录问题日期到错误日志
    3. 删除这些问题日期的所有时点数据
    4. 返回清洗后的数据
    
    设计理念:
    采用"全日删除"策略而非插值填充，确保分析结果的可靠性，
    避免因部分时点缺失而造成的日内收益率结构扭曲。
    """
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
    """
    对原始收益率数据进行后处理和清洗
    
    针对指数和个股数据的不同特点，进行相应的数据清洗和质量控制。
    特别处理指数数据中的隔夜收益率问题和所有数据的空值问题。
    
    参数:
    df_return (pd.Series): 原始收益率数据，来自cal_return函数的输出
                           - index: pd.DatetimeIndex，交易时间戳
                           - values: float，收益率数据（可能包含NaN）
    full_code (str): 完整的股票或指数代码（如"SH000300"）
                     用于错误日志和特殊处理
    start (datetime.datetime): 分析起始日期（当前未使用）
    end (datetime.datetime): 分析结束日期（当前未使用）
    is_index (bool): 数据类型标识
                     - True: 指数数据，需要进行隔夜收益率检查
                     - False: 个股数据，直接进行空值检查
    base_dir (str, optional): 基础目录路径，默认为空字符串
                              错误日志将保存到此目录下
    
    返回:
    pd.Series: 清洗后的收益率数据，具有以下特征：
               - 对于指数：隔夜收益率已经过修正（如需要）
               - 对于所有数据：已删除包含NaN值的完整交易日
               - 保持原有的时间索引结构
               - 数据质量符合后续分析要求
    
    处理流程:
    1. 如果是指数数据：调用index_check_overnight检查和修正隔夜收益率
    2. 对所有数据：调用check_null删除包含空值的日期
    3. 返回清洗后的数据供后续分析使用
    
    设计理念:
    - 针对指数和个股的不同特性进行专门化处理
    - 确保数据质量的同时保持处理流程的统一性
    - 为后续的风险建模和回归分析提供清洁的数据
    
    注意事项:
    - start和end参数在当前实现中未被使用
    - 该函数主要针对高频数据（分钟级和小时级）
    """

    if is_index:
        df_return=index_check_overnight(df_return,full_code, base_dir=base_dir)

    df_return=check_null(df_return,full_code, base_dir=base_dir)

    # workday_list = [day for day in workday_list if day >= start.date()]
    return df_return

def get_complete_return(full_code:str,workday_list:list=None,is_index:bool=False, params=None):
    """
    获取完整的收益率数据（缓存优先的智能加载函数）
    
    该函数是获取收益率数据的主要入口点，首先尝试从缓存加载，
    如果缓存不存在则从原始数据重新计算。支持高频和低频数据处理。
    
    参数:
    full_code (str): 完整的股票或指数代码（如"SH000300"）
                     用于识别交易所和数字代码
    workday_list (list, optional): 工作日列表
                                   如为None则在数据加载过程中自动生成
    is_index (bool, optional): 数据类型标识，默认False
                               - True: 指数数据，需要附加隔夜收益率检查
                               - False: 个股数据，需要分红调整
    params (dict): 参数字典，必须包含关键信息：
                   - "start" (datetime): 分析起始日期
                   - "end" (datetime): 分析结束日期
                   - "freq" (str): 重采样频率
                   - "base_dir" (str): 基础目录路径
    
    返回:
    tuple: 三元组包含完整的数据和辅助信息：
           - df_return (pd.Series or None): 收益率数据
             * 成功时：pd.Series，经过清洗和质量控制
             * 失败时：None（无法获取原始数据）
           - workday_list (list): 实际使用的工作日列表
             * list[datetime.date]，包含数据范围内的所有交易日
           - error_list (list): 问题日期列表
             * list[datetime.date]，数据缺失或有问题的日期
    
    处理流程:
    1. **缓存检查**: 调用get_cache_return尝试加载已存在的数据
    2. **原始数据加载**: 如缓存未命中，根据频率类型选择加载方式
       - 高频数据：调用get_high_freq_data
       - 低频数据：调用get_low_freq_data
    3. **数据重采样**: 使用resample函数进行时间频率转换
    4. **收益率计算**: 调用cal_return计算百分比收益率
    5. **数据裁剪**: 删除起始日期之前的数据
    6. **质量控制**: 对高频数据进行额外的清洗处理
    
    数据起始日期调整:
    - 高频数据：提前4天加载，确保除权参考价可用
    - 低频数据：按频率周期提前相应天数
    
    错误处理:
    - 如果params为None，抛出ValueError
    - 如果原始数据为None，返回(None, workday_list, error_list)
    - 所有错误都会被记录到相应的错误日志中
    
    性能优化:
    - 缓存机制显著减少重复计算时间
    - 按需加载，避免不必要的数据处理
    - 智能识别高频/低频，选择适合的加载策略
    """
    if params is None:
        raise ValueError("params is None")
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    base_dir=params["base_dir"]

    error_list = []

    # 先在start.date()_end.date()/freq/中找
    df_return, workday_list_temp, error_list_temp = get_cache_return(full_code,workday_list,is_index, params)
    if df_return is not None:
        return df_return, workday_list_temp, error_list_temp

    print(f"No complete return of {full_code} found, recompute")
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
    """
    检查和修正指数数据中的隔夜收益率问题
    
    指数数据中，由于数据源问题，09:25时点的隔夜收益率可能缺失或近似零。
    此函数识别这些问题并使用预先计算的准确隔夜收益率进行替换。
    
    参数:
    df (pd.Series): 指数收益率数据，必须包含09:25时点的数据
                    - index: pd.DatetimeIndex，包含交易日期和时间
                    - values: float，收益率数据
    full_code (str): 完整的指数代码（如"SH000300"）
                     用于构建替换数据文件路径和日志记录
    base_dir (str, optional): 基础目录路径，默认为空字符串
                              错误日志将保存到 base_dir/error/{full_code}.txt
    
    返回:
    pd.Series: 修正后的收益率数据，具有以下特征：
               - 问题的隔夜收益率已被替换为准确值
               - 保持原有的数据结构和时间索引
               - 其他数据点保持不变
    
    处理逻辑:
    1. **问题识别**: 检查每日首个数据点（通常为09:25）的收益率
    2. **阈值判断**: 绝对值小于0.00001的认为是问题数据
    3. **日志记录**: 记录所有问题日期到错误日志文件
    4. **数据替换**: 从预计算文件加载准确的隔夜收益率
    5. **精确匹配**: 仅替换时间戳完全匹配的数据点
    
    隔夜收益率数据源:
    - 文件路径: index/return/H{num_code}.csv
    - 数据结构: date列(日期)，overnight_return列(隔夜收益率)
    - 时间戳调整: 自动调整为09:25时点以匹配目标数据
    
    副作用:
    - 在控制台输出替换范围信息
    - 在错误日志中记录每个问题日期
    - 自动创建错误日志目录
    
    错误处理:
    - 如果没有问题数据，直接返回原数据
    - 如果替换数据文件不存在，会抛出FileNotFoundError
    - 仅替换能够精确匹配时间戳的数据点
    
    应用场景:
    - 仅适用于指数数据（is_index=True）
    - 主要在日内高频分析中使用
    - 确保隔夜跳空收益率的准确性
    
    注意事项:
    - 需要预先准备index/return/目录下的隔夜收益率数据
    - 代码转换规则：SH000300 -> H000300
    - 时间精度要求较高，必须精确到分钟
    """
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
    params={"start":start,"end":end,"freq":freq,"base_dir":"test"}
    df_index_return,workday_list,error_list=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=None,is_index=True)
    df_index_return.to_csv("index_return_test.csv")
    df_return,workday_list,error_list=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=workday_list,is_index=False)
    # df_return.to_csv("return_test.csv")
    print(error_list)
    df_return.to_csv("return_test.csv")