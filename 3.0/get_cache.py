
import os 
import pandas as pd
import datetime as dt
import warnings

   
def get_cache_text(full_code:str,start,end):
    date_str = f"{start.date()}_{end.date()}"
    txt_file_path = os.path.join(date_str,"agg_raw", f"{full_code}.txt")
    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("workday_list:"):
                    import datetime
                    workday_list = eval(line.split(":", 1)[1].strip(), {"__builtins__": {}, "datetime": datetime, "dt": dt})
                if line.startswith("error_list:"):
                    import datetime
                    error_list = eval(line.split(":", 1)[1].strip(), {"__builtins__": {}, "datetime": datetime, "dt": dt})
        return workday_list, error_list
    warnings.warn(f"Error in get_cache_text {full_code}: file not found")
    return [],[]

def get_cache_price(full_code:str,start,end):

    date_str = f"{start.date()}_{end.date()}"
    csv_file_path = os.path.join(date_str,"agg_raw", f"{full_code}.csv")    
    if os.path.exists(csv_file_path):
        df_stock = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
        # 重新获取workday_list和error_list
        workday_list,error_list=get_cache_text(full_code,start,end)
        return df_stock, workday_list,error_list
    warnings.warn(f"Error in get_cache_price {full_code}: file not found")
    return None, [], []

def get_cache_return(full_code:str,workday_list:list=None,is_index:bool=False, params=None):
    """
    从缓存中获取已计算的收益率数据
    
    在指定目录中查找已存在的收益率数据文件，并加载相关的辅助信息。
    用于加速重复计算，避免从原始数据重新处理。
    
    参数:
    full_code (str): 完整的股票或指数代码（如"SH000300"）
                     用于构建文件路径和缓存查找
    workday_list (list, optional): 工作日列表（当前未使用）
    is_index (bool, optional): 是否为指数数据（当前未使用）
    params (dict): 参数字典，必须包含以下键值：
                   - "start" (datetime): 开始日期
                   - "end" (datetime): 结束日期  
                   - "freq" (str): 频率参数
                   - "base_dir" (str): 基础目录路径
    
    返回:
    tuple: 三元组包含以下元素：
           - df_return (pd.Series or None): 缓存的收益率数据
             * 如果找到：pd.Series，index为DatetimeIndex，values为收益率
             * 如果未找到：None
           - workday_list (list): 工作日列表
             * 如果成功加载：list[datetime.date]，从辅助文件读取
             * 如果失败：空列表
           - error_list (list): 错误日期列表
             * 如果成功加载：list[datetime.date]，从辅助文件读取
             * 如果失败：空列表
    
    文件结构:
    - 主数据文件：{date_str}/{freq}/{full_code}.csv
      * date_str = "start_date_end_date" 格式
      * 存储pandas Series格式的收益率数据
    - 辅助信息文件：{date_str}/{freq}/agg_raw/{full_code}.txt
      * 包含workday_list和error_list的字符串表示
      * 使用eval()解析，带有安全限制
    
    异常处理:
    - 如果params为None，抛出ValueError
    - 文件不存在时返回(None, [], [])
    - 辅助文件解析失败时使用默认空列表

    """
    if params is None:
        raise ValueError("params is None")
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    base_dir=params["base_dir"]
    date_str = f"{start.date()}_{end.date()}"
    search_dir = os.path.join(date_str, freq)
    file_path = os.path.join(search_dir, f"{full_code}.csv")
    if os.path.exists(file_path):
        df_return = pd.read_csv(file_path, index_col=0, parse_dates=True).squeeze()
        # 重新获取workday_list和error_list
        workday_list,error_list=get_cache_text(full_code,start,end)
        return df_return, workday_list, error_list
    warnings.warn(f"Error in get_cache_return {full_code}: file not found")
    return None, [], []
