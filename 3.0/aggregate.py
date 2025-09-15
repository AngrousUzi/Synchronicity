from get_data import get_data
from resample import resample
from cal_return import cal_return,deal_return
from all_cal import get_composites,prerequisite
import os
import pandas as pd
import datetime as dt
import multiprocessing as mp
from functools import partial
from get_cache import get_cache_text,get_cache_price,get_cache_return
from utils import convert_freq_to_min, log_error


def check_dataframe_format(full_code,df,freq):
    freq_num = convert_freq_to_min(freq)
    total_num = 240 // freq_num +1
    if df.shape[0] % total_num != 0:
        df["date"] = df.index.normalize()
        count_per_day = df.groupby("date").size()
        abnormal_days = count_per_day[count_per_day != total_num]
        if not abnormal_days.empty:
            print(f"{full_code}在{freq}频率下存在数据数量异常的日期: {abnormal_days.index.tolist()}，对应数量: {abnormal_days.values.tolist()}")
            return False
            
    return True

def all_resample_return(df,index_code,workday_list,error_list,start,end,is_index,base_dir):
    # base_dir=os.path.join(base_dir,freq)
    for freq in ["3min","5min","10min","15min","30min","12h"]:
        _dir=os.path.join(base_dir,freq)
        os.makedirs(_dir,exist_ok=True)
        # os.makedirs(os.path.join(base_dir),exist_ok=True)
        df_resampled=resample(df,freq=freq,is_index=is_index,stock_code=index_code[2:],workday_list=workday_list,error_list=error_list)
        df_return=cal_return(df_resampled,index_code)       
        df_return=df_return.loc[df_return.index.date>=start.date()] 
        df_return=deal_return(df_return,index_code,start,end,is_index,_dir)
        # df_resampled.to_csv(os.path.join(base_dir,freq,f"{index_code}.csv"))
        check_dataframe_format(index_code,df_return,freq)
            
        
        df_return.to_csv(os.path.join(_dir,f"{index_code}.csv"))

def process_single_stock(args):
    """处理单个股票的函数，用于多进程"""
    full_code, start_tmp, end, workday_list, error_list, start, base_dir,is_index = args
    try:
        df_stock, _, error_list = get_data(start=start_tmp, end=end, exg=full_code[:2], 
                                           full_code=full_code, workday_list=workday_list, 
                                           base_dir=base_dir)
        df_stock.to_csv(os.path.join(base_dir, "agg_raw", f"{full_code}.csv"))
        with open(os.path.join(base_dir, "agg_raw", f"{full_code}.txt"), "w") as f:
            f.write(f"workday_list: {workday_list}\n")
            f.write(f"error_list: {error_list}\n")
        all_resample_return(df_stock, full_code, workday_list, error_list, start, end, is_index, base_dir)
        print(f"Successfully processed {full_code}")
        return full_code
    except Exception as e:
        print(f"Error processing {full_code}: {str(e)}")
        return full_code

def process_single_stock_with_cache(full_code, start, end, base_dir):
    try:
        index_data, workday_list, error_list = get_cache_price(full_code, start, end)
        all_resample_return(index_data, full_code, workday_list, error_list, start, end, False, base_dir)
        print(f"Successfully processed {full_code}")
    except Exception as e:
        print(f"Error in process_single_stock_with_cache {full_code}: {str(e)}")
        return full_code
    return full_code

def agg_return_with_cache(index_code, start, end, base_dir, cpu_parallel_num=60):
    industry_matching = pd.read_csv("industry/stock_index_match.csv")
    industry_code_list = industry_matching['index'].unique().tolist()

    composites = get_composites(index_code)
    composite_code_list = composites['full_code'].tolist()

    all_list = composite_code_list
    # 使用partial创建部分应用的函数
    process_func = partial(process_single_stock_with_cache, start=start, end=end, base_dir=base_dir)
    with mp.Pool(processes=cpu_parallel_num) as pool:
        results = pool.map(process_func, all_list)
    return results

def agg_all(index_code,start,end,base_dir,cpu_parallel_num=5):
    exg=index_code[:2]
    start_tmp=pd.Timestamp(start)-dt.timedelta(days=4)
    index_data,workday_list,error_list=get_data(start=start_tmp,end=end,exg=exg,full_code=index_code,workday_list=None,base_dir=base_dir)
    os.makedirs(os.path.join(base_dir,f"agg_raw"),exist_ok=True)
    index_data.to_csv(os.path.join(base_dir,"agg_raw",f"{index_code}.csv"))
    with open(os.path.join(base_dir,"agg_raw",f"{index_code}.txt"),"w") as f:
        f.write(f"workday_list: {workday_list}\n")
        f.write(f"error_list: {error_list}\n")
    all_resample_return(index_data,index_code,workday_list,error_list,start,end,True,base_dir)

        
    industry_matching=pd.read_csv("industry/stock_index_match.csv")
    industry_code_list=industry_matching['index'].unique().tolist()
    for industry_code in industry_code_list: 
        df_industry,_,error_list=get_data(start=start_tmp,end=end,exg=industry_code[:2],full_code=industry_code,workday_list=workday_list, base_dir=base_dir)
        df_industry.to_csv(os.path.join(base_dir,"agg_raw",f"{industry_code}.csv"))
        with open(os.path.join(base_dir,"agg_raw",f"{industry_code}.txt"),"w") as f:
            f.write(f"workday_list: {workday_list}\n")
            f.write(f"error_list: {error_list}\n")
        all_resample_return(df_industry,industry_code,workday_list,error_list,start,end,True,base_dir)

    composites=get_composites(index_code)

    # 使用多进程处理composites中的股票代码
    if len(composites["full_code"]) > 0:
        print(f"开始处理 {len(composites['full_code'])} 个股票代码...")
        # 准备多进程参数
        process_args = [(full_code, start_tmp, end, workday_list, error_list, start, base_dir,False) 
                       for full_code in composites["full_code"]]
        
        # 使用多进程池处理（进程数不超过任务数）
        actual_processes = min(len(composites["full_code"]), cpu_parallel_num)
        print(f"使用 {actual_processes} 个进程并行处理")
        with mp.Pool(processes=actual_processes) as pool:
            results = pool.map(process_single_stock, process_args)
        
    else:
        print("No composite stocks to process.")



def get_single_price(code,start,end,base_dir):
    try:
        df_stock,workday_list,error_list=get_data(start=start,end=end,exg=code[:2],full_code=code,workday_list=None,base_dir=base_dir)
        df_stock.to_csv(os.path.join(base_dir,"agg_raw",f"{code}.csv"))
        with open(os.path.join(base_dir,"agg_raw",f"{code}.txt"),"w") as f:
            f.write(f"workday_list: {workday_list}\n")
            f.write(f"error_list: {error_list}\n")
        print(f"Successfully processed {code}")
    except Exception as e:
        print(f"Error in get_single_price {code}: {str(e)}")
        return code
    # all_resample_return(df_stock,code,workday_list,error_list,start,end,False,base_dir)

def get_code_list():
        # date_str=f"{start.date()}_{end.date()}"
    index_list=["SH000905","SH000906","SH000852"]
    industry_matching=pd.read_csv("industry/stock_index_match.csv")
    industry_list=industry_matching['index'].unique().tolist()
    composite_list=get_composites("SH000300")['full_code'].tolist()
    files_list=find_all_files(os.path.join("data"))
    files_list=[file for file in files_list if file.endswith(".csv")]
    to_do_list=set(files_list)-set(index_list)-set(industry_list)-set(composite_list)
    return list(to_do_list)

def find_all_files(directory):
    """
    遍历指定目录（包括子目录），返回所有文件的文件名列表（不含路径）
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(file)
    return file_list

def agg_price_backup_with_no_composite(start,end,base_dir,cpu_parallel_num=60):
    to_do_list=get_code_list()
    process_func = partial(get_single_price, start=start, end=end, base_dir=base_dir)
    with mp.Pool(processes=cpu_parallel_num) as pool:
        results = pool.map(process_func, to_do_list)
    return results


if __name__=="__main__":
    start=dt.datetime(2010,1,5)
    end=dt.datetime(2025,6,30)
    index_code="SZ300481"
    date_str = f"{start.date()}_{end.date()}"
    # # param_dir = f"{index_code}_{date_str}_{freq}"
    # # base_dir = os.path.join("result", param_dir)
    # os.makedirs(date_str, exist_ok=True)
    # agg_return_with_cache(index_code=index_code,start=start,end=end,base_dir=date_str,cpu_parallel_num=60)
    # # agg_raw(index_code=index_code,start=start,end=end,base_dir=date_str,cpu_parallel_num=5)
    # # base_dir=os.path.join("backup",date_str)
    # # agg_price_backup_with_no_composite(start,end,base_dir,cpu_parallel_num=60)
    
    freq="5min"
    df=pd.read_csv(os.path.join(date_str,freq,f"{index_code}.csv"),index_col=0,parse_dates=True)
    print(check_dataframe_format(index_code,df,freq))
