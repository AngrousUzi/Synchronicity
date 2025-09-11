# 导入必要的数据分析和可视化库
import pandas as pd                       # 用于数据处理和分析
import numpy as np                        # 用于数值计算
import statsmodels.api as sm              # 用于统计建模（回归分析）
# from statsmodels.tools.sm_exceptions import MissingDataError  # 处理缺失数据异常

# import matplotlib.pyplot as plt          # 用于数据可视化
import datetime as dt                     # 用于日期时间处理

from cal_return import get_complete_return

import os

def log_error(msg, full_code, base_dir: str = ""):
    """Append error message to unified log file base_dir/error/full_code.txt"""
    error_dir = os.path.join(base_dir, 'error')
    os.makedirs(error_dir, exist_ok=True)
    with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
        f.write("single_cal: "+str(msg) + '\n')

def parse_timedelta(time_str):
    """
    时间字符串解析函数

    将时间间隔字符串（如 "0.5D"、 "12H"、 "0.5min"）转换为 timedelta 对象。

    支持的时间单位：
    - W/week/weeks：周
    - D/day/days：天
    - H/hour/hours：小时
    - min/minute/minutes：分钟
    - S/sec/second/seconds：秒

    参数:
    time_str (str): 时间间隔字符串，如 "5min", "2H", "1D"等

    返回:
    datetime.timedelta: 对应的时间间隔对象
    """
    
    # 提取数值部分（从字符串开头提取数字和小数点）
    value_str = ""
    i = 0
    while i < len(time_str) and (time_str[i].isdigit() or time_str[i] == '.'):
        value_str += time_str[i]
        i += 1

    # Debug: 打印提取的数值部分
    # print(f"[DEBUG] Extracted value_str: '{value_str}'")

    if value_str == "":
        value = 1.0
    else:
        try:
            value = float(value_str)
        except Exception as e:
            # print(f"[DEBUG] Error converting value_str '{value_str}' to float: {e}")
            raise

    # 提取单位部分（去除空格并转为小写）
    unit = time_str[i:].strip().lower()
    # Debug: 打印提取的单位部分
    # print(f"[DEBUG] Extracted unit: '{unit}'")

    # 根据单位转换为对应的 timedelta 对象
    if unit in ['w', 'week', 'weeks']:  # 
        return dt.timedelta(weeks=value)
    if unit in ['d', 'day', 'days']:  # 天
        return dt.timedelta(days=value)
    elif unit in ['h', 'hour', 'hours']:  # 小时
        return dt.timedelta(hours=value)
    elif unit in ['min', 'minute', 'minutes']:  # 分钟
        return dt.timedelta(minutes=value)
    elif unit in ['s', 'sec', 'second', 'seconds']:  # 秒
        return dt.timedelta(seconds=value)
    else:
        # 不支持的时间单位，抛出异常
        raise ValueError(f"Unsupported period: {time_str}")

def convert_freq_to_min(str):
    if str.endswith("min"):
        return int(str[:-3])
    elif str=="12h":
        return 240
    else:
        raise ValueError(f"Unsupported freq: {str}")

def simple_cal(df_stock,df_X,full_code, base_dir: str = ""):
    # 无法回归
    # df_index=df_X.iloc[:,0]
    if df_X.shape[0] <= 1 or df_stock.shape[0]<=1:
        nan_series = pd.Series(np.nan, index=df_X.columns)
        return {"r2": np.nan, "betas": nan_series}
    # if len(X_names)>1:
        # df_industry=df_X.iloc[:,1]
    # 数据对齐
    # print(df_stock.index)
    # print(df_X.index)

    if df_stock.shape[0]<df_X.shape[0]:
        log_error(f'{full_code} 数据长度不一致,start:{df_X.index[0]},end:{df_X.index[-1]}',full_code, base_dir=base_dir)
        log_error(f'stock_length:{df_stock.shape[0]},index_length:{df_X.shape[0]}',full_code, base_dir=base_dir)
        missing_set=set(df_X.index.values)-set(df_stock.index.values)
        # if len(missing_set)<3:
        log_error(f'缺少的index为{missing_set}',full_code, base_dir=base_dir)
        # df_stock.to_csv(f'error_list/return/{full_code}.csv')

        df_X=df_X.loc[df_stock.index].copy()
        # df_industry=df_industry.loc[df_stock.index].copy() 
    elif df_stock.shape[0]>df_X.shape[0]:
        log_error(f'{full_code} 数据长度不一致,stock_length:{df_stock.shape[0]},index_length:{df_X.shape[0]}',full_code, base_dir=base_dir)
        missing_set=set(df_X.index.values)-set(df_stock.index.values)
        # if len(missing_set)<3:
        log_error(f'缺少的index为{missing_set}',full_code, base_dir=base_dir)
        df_X=df_X.loc[df_stock.index].copy()
        # df_industry=df_industry.loc[df_stock.index].copy()



    Y=df_stock
    X=df_X
    X=sm.add_constant(X)

    # print(Y,X)

    model=sm.OLS(Y,X)
    results=model.fit()
    r2=results.rsquared
    params=results.params
    # try:
    # except IndexError:
        # print(results.summary())
    # print(type(params))
    # print(r2,params)
    return {"r2":r2,"betas":params[1:]}

def simple_cross_section_cal(df_stock,df_X,full_code,total_num, base_dir: str = ""):
    r2s=pd.Series()

    X_cols=df_X.columns
    betas_dict={col:pd.Series() for col in X_cols}


    for nth in range(total_num):
        # for df in [df_stock_period,df_index_period,df_industry_period]:
            # 选取每天的第x个元素（例如第一个元素）
            # 这里以选取每天的第一个元素为例
        df_stock_nth= df_stock.groupby(df_stock.index.date).nth(nth)
        df_X_nth= df_X.groupby(df_X.index.date).nth(nth)
        # df_stock_period.to_csv(f'temp/{period_start.date()}_{period_end.date()}.csv')
        
        simple_cal_result=simple_cal(df_stock_nth,df_X_nth,full_code, base_dir=base_dir)
        # print(simple_cal_result)
        r2s[nth]=simple_cal_result["r2"]
        for col in X_cols:
            betas_dict[col][nth]=simple_cal_result["betas"][col]
    # r2s,beta_indexs,beta_industrys=simple_cal(full_code,df_index,df_industry,start,end,freq,workday_list,period,method="cross_section")
    return {"r2":r2s,"betas":betas_dict}

def single_periodic_cal(full_code,df_X,workday_list,params=None):
    ''''
        如果method=="

    '''
    if params is None:
        raise ValueError("params is None")
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]

    X_cols=df_X.columns

    df_stock,_,error_list=get_complete_return(full_code,workday_list,False, params=params)
    # df_stock.to_csv('temp_stock.csv')
    if df_stock is None:
        return None

    # os.makedirs(os.path.join(base_dir, 'temp', "returns"), exist_ok=True)
    # df_stock.to_csv(os.path.join(base_dir, 'temp', "returns", f'{full_code}_returns.csv'))

    workday_list=[date for date in workday_list if date>=start.date()]
    if period=="full":
        period=max(len(workday_list)-1,1)
    else:
        period=int(period)
    # print(period)
    if method=="simple":
        total_num=1
        # r2s=pd.DataFrame(columns=[0])
        # betas_dict={col:pd.DataFrame(columns=[0]) for col in X_cols}
    elif method=="cross_section":
        freq_num = convert_freq_to_min(freq)
        total_num = 240 // freq_num +1
    else:
        raise ValueError(f"method {method} not supported")
    r2s_all=pd.DataFrame(columns=range(total_num))
    betas_dict={col:pd.DataFrame(columns=range(total_num)) for col in X_cols}
    
    # print(workday_list[::period],workday_list[period::period])
    for period_start,period_end in zip(workday_list[::period],workday_list[period::period]):
        # print(period_start,period_end)
        # print(df_X)
        # print(df_stock)
        # 对于pd.date_range(start,end) 包含start和end 因此在选择的时候需要不包含end中的日期
        # period_end=dt.datetime.strptime(str(period_start),"%Y-%m-%d %H:%M:%S")+dt.timedelta(days=period)-dt.timedelta(seconds=1)
        # period_end=dt.datetime.strftime(period_end,"%Y-%m-%d %H:%M:%S")
        # print(period_start,period_end)
        df_X_period=df_X.loc[pd.Timestamp(period_start):pd.Timestamp(period_end)]
        #对于选取1天的情况，可能出现的问题
        if df_X_period.shape[0]==0: continue
        df_stock_period=df_stock.loc[pd.Timestamp(period_start):pd.Timestamp(period_end)]
        
        # df_stock_period.to_csv(f'temp/{period_start.date()}_{period_end.date()}.csv')
        if method=="simple":
            cal_result=simple_cal(df_stock_period,df_X_period,full_code, base_dir=base_dir)
            r2s_all.loc[period_start,0]=cal_result["r2"]
            for col in X_cols:
                betas_dict[col].loc[period_start,0]=cal_result["betas"][col]
        elif method=="cross_section":
            cal_result=simple_cross_section_cal(df_stock_period,df_X_period,full_code,total_num, base_dir=base_dir)
            r2s_all.loc[period_start]=cal_result["r2"]
            for col in X_cols:
                betas_dict[col].loc[period_start]=cal_result["betas"][col]
    
    return {"r2":r2s_all,"betas":betas_dict}
    

if __name__=="__main__":
    start=dt.datetime(2021,1,3)
    end=dt.datetime(2021,12,30)
    freq="12h"
    df_index,workday_list,_=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=None,is_index=True)
    # print(workday_list)
    # df_industry,_,error_list=get_complete_return(full_code="SH000070",start=start,end=end,freq=freq,workday_list=workday_list,is_index=True)

    
    method="cross_section"
    # for period in ["full","10"]:
    period="20"
    # for full_code in [""]
    full_code="SH600905"
    print(full_code)
    # results=single_periodic_cal(full_code=full_code,df_index=df_index,df_industry=df_industry,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method)
    # results.to_csv(f'test_results_{period}_{method}.csv')
        
    # method="cross_section"
    # for period in ["full","10d"]:
    
    # df_X=pd.concat([df_index,df_industry],axis=1)
    # df_X.columns=["index","industry"]
    # print(df_X)
    X_cols=["index"]
    df_X=pd.DataFrame(df_index)
    df_X.columns=X_cols

    results=single_periodic_cal(full_code=full_code,df_X=df_X,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method)
    os.makedirs("test",exist_ok=True)
    results["r2"].to_csv(os.path.join("test",f'test_results_{period}_{method}.csv'))
    results["betas"]["index"].to_csv(os.path.join("test",f'test_betas_index_{period}_{method}.csv'))
    # results["betas"]["industry"].to_csv(f'test_betas_industry_{period}_{method}.csv')