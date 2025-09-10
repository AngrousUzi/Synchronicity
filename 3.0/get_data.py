import pandas as pd
import numpy as np
import datetime as dt
import os

def get_high_freq_data(start:dt.datetime=None,end:dt.datetime=None,exg:str=None,full_code:str=None,workday_list:list=None, base_dir: str = ""):
    df_stock=pd.DataFrame()
    error_list=[]

    def log_error(msg: str):
        error_dir = os.path.join(base_dir, 'error')
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
            f.write("get_data: " + str(msg) + '\n')

    if workday_list is None:
        workday_list=[]
    
        for date in pd.date_range(start=start,end=end,freq="D"):
            year=str(date).split("-")[0]
            month=str(date).split('-')[1]
            day=str(date).split('-')[2].split(' ')[0]

            path = os.path.join('data', f'ws{year+month+day}fb', exg, f'{full_code}.csv')
            # print(path)

            try:
                df_tmp=pd.read_csv(path)
                df_stock=pd.concat([df_stock,df_tmp])
                workday_list.append(date)
            except FileNotFoundError:
                continue
        
        if not workday_list:
            raise ValueError(f"No data found for {full_code} from {start} to {end}.")
        workday_list=[start_time.date() for start_time in workday_list]
    else:
        
        # 如果workday_list不为空，说明已经完成了工作日判定
        for date in workday_list:
            year=str(date).split("-")[0]
            month=str(date).split('-')[1]
            day=str(date).split('-')[2].split(' ')[0]
            path = os.path.join('data', f'ws{year+month+day}fb', exg, f'{full_code}.csv')
            try:
                df_tmp=pd.read_csv(path)
                df_stock=pd.concat([df_stock,df_tmp])
            except FileNotFoundError:
                error_list.append(date)
                continue
        if len(error_list)>0:
            print(f"{full_code} on {[str(date) for date in error_list]} is not found.")
            log_error(f"Missing data dates: {[str(date) for date in error_list]}")
    # df_stock.to_csv("SAMPL.csv")
    if df_stock.empty:
        print(f"{full_code} is not found.")
        log_error(f"{full_code} DataFrame is empty after reading raw files.")
        return None,workday_list,error_list
    
    df_stock['Time']=pd.to_datetime(df_stock['Time'])
    df_stock=df_stock.set_index('Time').sort_index(ascending=True)
    
    df_stock=df_stock[['Price']]

    return df_stock,workday_list,error_list

def get_low_freq_data(start:dt.datetime=None,end:dt.datetime=None,exg:str=None,full_code:str=None,workday_list:list=None, base_dir: str = ""):
    df=pd.read_csv(os.path.join(base_dir,f"{full_code}.csv"),index_col=0,parse_dates=True)
    df.index=pd.to_datetime(df.index)
    df=df[['Price']]
    error_list=list(set(workday_list)-set(df.index.date))
    return df,workday_list,error_list


def get_data(start:dt.datetime=None,end:dt.datetime=None,exg:str=None,full_code:str=None,workday_list:list=None, base_dir: str = ""):
    """
    获取这一段日期范围内的数据
    start: 开始日期
    end: 结束日期
    workday_list: 工作日列表
    exg: 交易所
    full_code: 指数代码，如SH000001
    :return:  一个DataFrame,共两列，共两列，index（’Time')是datetime格式，columns为Price
                workday_list，以list形式存储，element为TimeStamp.date()的工作日，如2024-01-01      
    """
        
    return get_high_freq_data(start,end,exg,full_code,workday_list, base_dir)

if __name__=="__main__":
    os.makedirs('test',exist_ok=True)

    df=get_data(start=dt.datetime(2024,1,1),end=dt.datetime(2024,12,31),exg="SH",full_code="SH000001")
    df[0].to_csv(os.path.join('test',"sample_index_1.csv"))
    workday_list=df[1]
    with open(os.path.join('test',"workday_list.txt"),'w') as t:
        for day in workday_list:
            t.write(str(day)+"\n")
    # df=get_data(start=dt.datetime(2023,12,1),end=dt.datetime(2024,1,31),exg="SH",full_code="SH000070",workday_list=workday_list)
    # df[0].to_csv("sample_index_2.csv")

