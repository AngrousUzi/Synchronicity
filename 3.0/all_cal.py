from single_cal import single_periodic_cal
from cal_return import get_complete_return

import pandas as pd 
import datetime as dt
import numpy as np
import os
import warnings
warnings.filterwarnings('default', category=RuntimeWarning)

def get_composites(index_code):
    composites = pd.DataFrame()
    composites['full_code'] = pd.read_excel(f'index/{index_code}.xlsx')['证券代码']
    composites.drop(index=composites.index[-2:], inplace=True)
    composites['exg'] = composites['full_code'].str.split('.').str[1]      # 提取交易所代码
    composites['num_code'] = composites['full_code'].str.split('.').str[0]     # 提取股票代码
    composites['full_code'] = composites['exg'] + composites['num_code']   
    # composites.set_index('full_code',inplace=True)    # 重新组合为统一格式
    return composites

def prerequisite(index_code,start,end,freq):
    df_index,workday_list,_=get_complete_return(full_code="SH000300",start=start,end=end,freq=freq,workday_list=None,is_index=True)
    # print(workday_list)
    workday_list_truly_used=[date for date in workday_list if date>=start.date()]
    print(f'From {start.date()} to {end.date()} (included), there are {len(workday_list_truly_used)} workdays.')
    df_constant=df_index.rename(index_code)
    industry_matching=pd.read_csv("industry/stock_index_match.csv")
    industry_code_list=industry_matching['index'].unique().tolist()
    for industry_code in industry_code_list: 
        df_industry,_,error_list=get_complete_return(full_code=industry_code,start=start,end=end,freq=freq,workday_list=workday_list,is_index=True)

        df_constant=pd.concat([df_constant,df_industry.rename(industry_code)],axis=1)

    df_constant.to_csv("df_constant.csv")
    if df_constant.isnull().any().any():
        raise ValueError("df_constant has null values.")
    composites=get_composites(index_code)
    composites=composites.merge(industry_matching,on='full_code',how='left')
    
    return df_constant,composites,workday_list

def all_cal(index_code,df_constant,X_cols,composites,start,end,freq,workday_list,period,method):
    if method=="cross_section" and (period=="1" or period=="2"):
        warnings.warn(f"Cross-section method may result in all NaN or 1 for period={period}", RuntimeWarning)


    composite_list=composites['full_code'].tolist()
    final_r2_result={}
    final_betas_dict={}
    processing_count=0
    if not os.path.exists("temp/r2"):
        os.makedirs("temp/r2")
    for full_code in composite_list:
        if "industry" in X_cols:
            industry_code=composites.loc[composites['full_code']==full_code,'index'].values[0]
            # df_constant=pd.concat([df_constant,df_industry.rename(industry_code)],axis=1)
            df_constant_needed=df_constant[[index_code,industry_code]]
        else:
            df_constant_needed=df_constant[[index_code]]
        df_constant_needed.columns=X_cols
        single_result=single_periodic_cal(full_code=full_code,df_X=df_constant_needed,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method)

        if single_result is None:
            print(f"No data for {full_code}")
        df_r2=single_result["r2"]
        betas_dict=single_result["betas"]
        
        final_betas_dict[full_code]=betas_dict

        
        if df_r2 is not None:
            final_r2_result[full_code]=df_r2
            df_r2.to_csv(f"temp/r2/{full_code}.csv")

        processing_count+=1
        # if processing_count%10=z=5: 
        print(full_code,processing_count)
            # break
    summed_df = sum_df(final_r2_result)

    return summed_df

def sum_df(r2_result_dict,beta_index_result_dict=None,beta_industry_result_dict=None):
    summed_df = None
    no_data_list=[]
    count=0
    for key,df in r2_result_dict.items():
        if summed_df is None:
            summed_df = df.copy()
        else:
            if df is not None:
                summed_df = summed_df.add(df, fill_value=0)
            else:
                no_data_list.append(key)
    if no_data_list:
        print(f"No data when summing up: {no_data_list}")
    return summed_df

def continue_cal(index_code,df_constant,X_cols,composites,start,end,freq,workday_list,period,method,continue_code):
    # rangeIndex无法slide index
    continue_index=composites[composites['full_code']==continue_code].index[0]
    composites_continue=composites[continue_index:].copy()
    # print(composites_continue)
    all_cal(index_code=index_code,df_constant=df_constant,X_cols=X_cols,composites=composites_continue,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method)
    results_dict={}
    for full_code in composites["full_code"].tolist():
        try:
            results=pd.read_csv(f"temp/r2/{full_code}.csv")
            results_dict[full_code]=results
        except FileNotFoundError:
            print(f"temp/r2/{full_code}.csv not found")
            results_dict[full_code]=None
            continue
    return sum_df(results_dict)     

if __name__=="__main__":
    
    start=dt.datetime(2025,1,2)
    end=dt.datetime(2025,1,4)
    freq="12h"
    period="full"
    method="simple"
    X_cols=["index"]

    print(f"start: {start.date()}, end: {end.date()}, freq: {freq}, period: {period}, method: {method}, X_cols: {X_cols}")
    df_constant,composites,workday_list=prerequisite(index_code="SH000300",start=start,end=end,freq=freq)
    df_constant.to_csv("df_constant.csv")
    # print(composites)

    
#%%
    results=all_cal(index_code="SH000300",df_constant=df_constant,X_cols=X_cols,composites=composites,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method)
    results.to_csv('test_results_all.csv')
    print("all_cal done")
    results=continue_cal(index_code="SH000300",df_constant=df_constant,composites=composites,start=start,end=end,freq=freq,workday_list=workday_list,period=period,method=method,continue_code="SH600570")
    results.to_csv('test_results_continue.csv')
    print("continue_cal done")