from single_cal import single_periodic_cal
from cal_return import get_complete_return

import pandas as pd 
import datetime as dt
import numpy as np
import os
import warnings
warnings.filterwarnings('default', category=RuntimeWarning)
import multiprocessing as mp
from functools import partial
import argparse

def run_single_calculation(full_code, index_code, composites, df_constant, workday_list, params=None):
    """
    Worker function to perform calculation for a single stock.
    This function is designed to be called by multiprocessing.Pool.
    """
    if params is None:
        raise ValueError("params is None")
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]

    print(f"Processing {full_code}...")
    if "industry" in X_cols:
        industry_code=composites.loc[full_code,'index']
        df_constant_needed=df_constant[[index_code,industry_code]]
    else:
        df_constant_needed=df_constant[[index_code]]
    df_constant_needed.columns=X_cols
    single_result=single_periodic_cal(full_code=full_code,df_X=df_constant_needed,workday_list=workday_list,params=params)

    if single_result is None:
        print(f"No data for {full_code}")
        error_dir = os.path.join(base_dir, 'error')

        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
            f.write('all_cal: No data for this code.\n')
        return full_code, None
    
    df_r2=single_result["r2"]
    betas_dict=single_result["betas"]
    
    if not df_r2.empty:
        # Save intermediate result, useful for resuming
        r2_dir = os.path.join(base_dir, "temp", "r2")
        os.makedirs(r2_dir, exist_ok=True)
        df_r2.to_csv(os.path.join(r2_dir, f"{full_code}.csv"))
    else:
        error_dir = os.path.join(base_dir, 'error')
        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
            f.write('all_cal: No r2 data for this code.\n')
    if betas_dict is not None:
        for key,value in betas_dict.items():
            if not value.empty:
                beta_dir = os.path.join(base_dir, "temp", "betas", key)
                os.makedirs(beta_dir, exist_ok=True)
                value.to_csv(os.path.join(beta_dir, f"{full_code}.csv"))
            else:
                error_dir = os.path.join(base_dir, 'error')
                with open(os.path.join(error_dir, f'{full_code}.txt'), 'a', encoding='utf-8') as f:
                    f.write('all_cal: No betas data for this code.\n')

    return full_code, {"r2": df_r2, "betas": betas_dict}

def get_composites(index_code):
    composites = pd.DataFrame()
    composites['full_code'] = pd.read_excel(f'index/{index_code}.xlsx')['证券代码']
    composites.drop(index=composites.index[-2:], inplace=True)
    composites['exg'] = composites['full_code'].str.split('.').str[1]      # 提取交易所代码
    composites['num_code'] = composites['full_code'].str.split('.').str[0]     # 提取股票代码
    composites['full_code'] = composites['exg'] + composites['num_code']   
    # composites.set_index('full_code',inplace=True)    # 重新组合为统一格式
    return composites

def prerequisite(index_code,params):
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    X_cols=params["X_cols"]
    base_dir=params["base_dir"]
    
    composites=get_composites(index_code)

    
    df_index,workday_list,_=get_complete_return(full_code="SH000300",workday_list=None,is_index=True, params=params)
    # print(workday_list)
    workday_list_truly_used=[date for date in workday_list if date>=start.date()]
    print(f'From {start.date()} to {end.date()} (included), there are {len(workday_list_truly_used)} workdays.')
    df_constant=df_index.to_frame(name=index_code)
    if "industry" in X_cols:
        industry_matching=pd.read_csv("industry/stock_index_match.csv")
        industry_code_list=industry_matching['index'].unique().tolist()
        for industry_code in industry_code_list: 
            df_industry,_,error_list=get_complete_return(full_code=industry_code,workday_list=workday_list,is_index=True, params=params)
            df_constant=pd.concat([df_constant,df_industry.to_frame(name=industry_code)],axis=1)
        
        composites=composites.merge(industry_matching,on='full_code',how='left')
    
    df_constant_path = os.path.join(base_dir, "df_constant.csv")

    df_constant.to_csv(df_constant_path)
    if df_constant.isnull().any().any():
        raise ValueError("df_constant has null values.")
        
    
    return df_constant,composites,workday_list

def all_cal(index_code,df_constant,composites,workday_list,cpu_parallel_num=None, params=None):
    
    if method=="cross_section" :
        if not freq.endswith("min") and not freq.endswith("h"):
            raise ValueError("Cross-section method is only supported for minute or hour frequency.")
        if (period=="1" or period=="2"):
            warnings.warn(f"Cross-section method may result in all NaN or 1 for period={period}", RuntimeWarning)

    
    if params is None:
        raise ValueError("params is None")
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    


    composite_list=composites['full_code'].tolist()
    if not os.path.exists(os.path.join(base_dir, "temp", "r2")):
        os.makedirs(os.path.join(base_dir, "temp", "r2"))
    for col in X_cols:
        beta_dir = os.path.join(base_dir, "temp", "betas", col)
        if not os.path.exists(beta_dir):
            os.makedirs(beta_dir)

    if cpu_parallel_num is None:
        results=[]
        for full_code in composite_list:
            results.append(run_single_calculation(full_code=full_code,index_code=index_code,composites=composites,df_constant=df_constant,workday_list=workday_list,params=params))

    else:
        if cpu_parallel_num==0:
            num_processes = max(1,os.cpu_count()-1)
        else:
            num_processes = cpu_parallel_num

        # Use partial to create a new function with most arguments pre-filled.
        # This is needed to pass a single-argument function to pool.map.
        worker_func = partial(run_single_calculation, 
                            index_code=index_code, 
                            composites=composites, 
                            df_constant=df_constant, 
                            workday_list=workday_list, 
                            params=params)

        # Determine the number of processes to use. Leave one core free for system stability.
        print(f"Using {num_processes} processes for calculation.")
    
        # Use a pool of worker processes to parallelize the calculation.
        # The 'with' statement ensures the pool is properly closed.
        with mp.Pool(processes=num_processes) as pool:
        # map will distribute the composite_list to the worker_func and collect results.
            results = pool.map(worker_func, composite_list)

    final_r2_result={}
    final_betas_dict={}

    # Process the results returned from the workers
    for full_code, result_data in results:
        if result_data:
            final_r2_result[full_code] = result_data["r2"]
            final_betas_dict[full_code] = result_data["betas"]
        else:
            final_r2_result[full_code] = None
    
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

def continue_cal(index_code,df_constant,X_cols,composites,workday_list,cpu_parallel_num,continue_code, params=None):
    if params is None:
        raise ValueError("params is None")
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    # rangeIndex无法slide index
    continue_index=composites[composites['full_code']==continue_code].index[0]
    composites_continue=composites[continue_index:].copy()
    # print(composites_continue)
    all_cal(index_code=index_code,df_constant=df_constant,X_cols=X_cols,composites=composites_continue,workday_list=workday_list,cpu_parallel_num=cpu_parallel_num, params=params)
    results_dict={}
    for full_code in composites["full_code"].tolist():
        try:
            results=pd.read_csv(os.path.join(base_dir, "temp", "r2", f"{full_code}.csv"))
            results_dict[full_code]=results
        except FileNotFoundError:
            print(f"temp/r2/{full_code}.csv not found")
            results_dict[full_code]=None
            continue
    return sum_df(results_dict)     

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Run parallel calculations for stock data.")
    
    parser.add_argument('--start', type=str, default='2010-01-05', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default='2025-06-30', help='End date in YYYY-MM-DD format')
    parser.add_argument('--freq', type=str, default='12h', help='Frequency for calculation')
    parser.add_argument('--period', type=str, default='30', help='Period for calculation')
    parser.add_argument('--method', type=str, default='simple', help='Method for calculation (simple or cross_section)')
    parser.add_argument('--X_cols', nargs='+', default=['index'], help='List of columns for X')
    parser.add_argument('--index_code', type=str, default='SH000300', help='Index code')
    parser.add_argument('--cpu_parallel_num', type=int, default=25, help='Number of CPU cores to use')

    args = parser.parse_args()

    start=dt.datetime.strptime(args.start, '%Y-%m-%d')
    end=dt.datetime.strptime(args.end, '%Y-%m-%d')
    freq=args.freq
    period=args.period
    method=args.method
    X_cols=args.X_cols
    index_code = args.index_code
    cpu_parallel_num=args.cpu_parallel_num


    
    print(f"start: {start.date()}, end: {end.date()}, freq: {freq}, period: {period}, method: {method}, X_cols: {X_cols}, index_code: {index_code}")

    x_str = "_".join(X_cols)
    date_str = f"{start.date()}_{end.date()}"
    param_dir = f"{index_code}_{date_str}_{freq}_{period}_{method}_{x_str}"
    base_dir = os.path.join("result", param_dir)
    os.makedirs(base_dir, exist_ok=True)

    params={"start":start,"end":end,"freq":freq,"period":period,"method":method,"X_cols":X_cols,"base_dir":base_dir}

    df_constant,composites,workday_list=prerequisite(index_code=index_code,params=params)
    # print(composites)

    
#%%
    results=all_cal(index_code=index_code,df_constant=df_constant,composites=composites,workday_list=workday_list,cpu_parallel_num=cpu_parallel_num, params=params)
    results.to_csv(os.path.join(base_dir, 'test_results_all.csv'))
    print("all_cal done")

#%%
    # results=continue_cal(index_code="SH000300",df_constant=df_constant,composites=composites,workday_list=workday_list,cpu_parallel_num=None,continue_code="SH600570",params=params)
    # results.to_csv('test_results_continue.csv')
    # print("continue_cal done")