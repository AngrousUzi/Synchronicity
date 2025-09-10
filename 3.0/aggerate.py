from get_data import get_data
from resample import resample
from cal_return import cal_return,deal_return
from all_cal import get_composites,prerequisite
import os
import pandas as pd
import datetime as dt

def agg_raw(index_code,start,end,base_dir):
    exg=index_code[:2]
    start_tmp=pd.Timestamp(start)-dt.timedelta(days=4)
    index_data,workday_list,error_list=get_data(start=start_tmp,end=end,exg=exg,full_code=index_code,workday_list=None,base_dir=base_dir)
    os.makedirs(os.path.join(base_dir,f"agg_raw"),exist_ok=True)
    index_data.to_csv(os.path.join(base_dir,"agg_raw",f"{index_code}.csv"))
    with open(os.path.join(base_dir,"agg_raw",f"{index_code}.txt"),"w") as f:
        f.write(f"workday_list: {workday_list}\n")
        f.write(f"error_list: {error_list}\n")
    all_resample_return(index_data,index_code,workday_list,error_list,start,end,True,base_dir)


    composites=get_composites(index_code)

    for full_code in composites["full_code"]:
        df_stock,workday_list,_=get_data(start=start_tmp,end=end,exg=full_code[:2],full_code=full_code,workday_list=workday_list,base_dir=base_dir)
        df_stock.to_csv(os.path.join(base_dir,"agg_raw",f"{full_code}.csv"))
        with open(os.path.join(base_dir,"agg_raw",f"{full_code}.txt"),"w") as f:
            f.write(f"workday_list: {workday_list}\n")
            f.write(f"error_list: {error_list}\n")
        all_resample_return(df_stock,full_code,workday_list,error_list,start,end,False,base_dir)
    industry_matching=pd.read_csv("industry/stock_index_match.csv")
    industry_code_list=industry_matching['index'].unique().tolist()
    for industry_code in industry_code_list: 
        df_industry,_,error_list=get_data(start=start_tmp,end=end,exg=industry_code[:2],full_code=industry_code,workday_list=workday_list, base_dir=base_dir)
        df_industry.to_csv(os.path.join(base_dir,"agg_raw",f"{industry_code}.csv"))
        with open(os.path.join(base_dir,"agg_raw",f"{industry_code}.txt"),"w") as f:
            f.write(f"workday_list: {workday_list}\n")
            f.write(f"error_list: {error_list}\n")
        all_resample_return(df_industry,industry_code,workday_list,error_list,start,end,True,base_dir)

def all_resample_return(df,index_code,workday_list,error_list,start,end,is_index,base_dir):
    # base_dir=os.path.join(base_dir,freq)
    for freq in ["3min","5min","10min","15min","30min","12h"]:
        _dir=os.path.join(base_dir,freq)
        os.makedirs(_dir,exist_ok=True)
        # os.makedirs(os.path.join(base_dir),exist_ok=True)
        df_resampled=resample(df,freq=freq,is_index=is_index,stock_code=index_code[2:],workday_list=workday_list,error_list=error_list)
        df_return=cal_return(df_resampled,index_code)        
        df_return=deal_return(df_return,index_code,start,end,is_index,_dir)
        # df_resampled.to_csv(os.path.join(base_dir,freq,f"{index_code}.csv"))
        df_return.to_csv(os.path.join(_dir,f"{index_code}.csv"))


if __name__=="__main__":
    start=dt.datetime(2024,1,1)
    end=dt.datetime(2024,1,31)
    index_code="SH000300"
    date_str = f"{start.date()}_{end.date()}"
    # param_dir = f"{index_code}_{date_str}_{freq}"
    # base_dir = os.path.join("result", param_dir)
    os.makedirs(date_str, exist_ok=True)
    agg_raw(index_code=index_code,start=start,end=end,base_dir=date_str)

