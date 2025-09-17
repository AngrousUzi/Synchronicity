import pandas as pd
import datetime as dt
import os

def generate_single_txt(code,base_path,workday_list):
    csv_path=os.path.join(base_path,f"{code}.csv")
    # if os.path.exists(csv_path):
    df_stock=pd.read_csv(csv_path,index_col=0,parse_dates=True)
    workday_list_stock = [d.date() for d in df_stock.index.normalize().tolist()]        
    error_list=list(set(workday_list)-set(workday_list_stock))
    error_list.sort()
    txt_path=os.path.join(base_path,f"{code}.txt")
    with open(txt_path,"w") as f:
        f.write(f"workday_list: {workday_list}\n")
        f.write(f"error_list: {error_list}\n")
    print(code)

def log_error(msg, full_code, base_dir: str = ""):
    """Append error message to unified log file base_dir/error/full_code.txt"""
    error_dir = os.path.join(base_dir, 'error')
    os.makedirs(error_dir, exist_ok=True)
    with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

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

def convert_freq_to_day(str):
    """
    将频率字符串转换为对应的天数
    
    将低频率的字符串表示（如"1d", "2w"）转换为对应的交易日天数。
    主要用于低频数据重采样时计算采样间隔。
    
    参数:
    str (str): 频率字符串，支持的格式：
               - "Nd" 表示N天，如"1d"表示1天
               - "Nw" 表示N周，如"2w"表示2周（转换为10个交易日）
    
    返回:
    int: 对应的自然日天数
         - 对于天数格式：直接返回天数
         - 对于周数格式：周数乘以5（按交易日计算）
    
    异常:
    ValueError: 当输入的频率字符串格式不受支持时抛出
    
    示例:
    >>> convert_freq_to_day("1d")
    1
    >>> convert_freq_to_day("2w")
    10
    """
    if str.endswith("d") or str.endswith('D'):
        return int(str[:-1])
    elif str.endswith("w") or str.endswith('W'):
        return int(str[:-1])*5
    else:
        raise ValueError(f"Unsupported freq: {str}")
