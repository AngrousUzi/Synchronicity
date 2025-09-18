import pandas as pd
import numpy as np 
import os 
import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import font_manager
from utils import convert_freq_to_min
import multiprocessing as mp


def generate_param_dir(params):
    index_code=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    x_str = "_".join(X_cols)
    date_str = f"{start.date()}_{end.date()}"
    param_dir = f"{index_code}_{date_str}_{freq}_{period}_{method}_{x_str}"
    return param_dir


def aggerate_simple_temp_result(params):
    '''
        简单叠加个股df，添加code列
    '''
    index_code=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    param_dir = generate_param_dir(params)
    base_dir_agg = os.path.join(base_dir, param_dir)

    temp_dir=os.path.join(base_dir_agg, "temp","r2")
    dfs = []
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        # 这里可以对每个文件进行后续处理
        # INSERT_YOUR_CODE
        # 合成一张大的表格
        df = pd.read_csv(filepath, index_col=0)
        df['code'] = os.path.splitext(filename)[0]
        dfs.append(df)
    if dfs:
        big_df = pd.concat(dfs, axis=0)
        big_df.reset_index(inplace=True)
        os.makedirs(os.path.join(base_dir, "result"), exist_ok=True)
        out_path = os.path.join(base_dir, "result", f"{param_dir}_agg_r2.csv")
        big_df.to_csv(out_path, index=False)
    else:
        print("没有找到可合成的csv文件。")

    return big_df


def generate_title_cn(params):
    index_code=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    index="沪深300" if index_code=="SH000300" else "中证500" if index_code=="SH000905" else "中证1000" if index_code=="SH000852" else "错误"
    X_cols_str="市场指数" if len(X_cols)==1 else "行业指数和市场指数" if len(X_cols)==2 else "错误"
    method_str="全时段回归" if method=="simple" else "单时段对应回归" if method=="cross_section" else "错误"
    return f"市场指数选择：{index}，回归方式：{X_cols_str}{method_str}，时间范围{start.date()}-{end.date()}，重采样频率：{freq}，每次回归计算时段长度：{period}个交易日"

def generate_label_list(params):
    index_code=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    # print(columns)
    if method=="simple":
        return ["每次回归时段完整回归"]
    elif method=="cross_section" and freq=="12h":
        return ["集合竞价时段","连续交易时段"]
    else:
        freq_num = convert_freq_to_min(freq)
        num=240//freq_num
        return ["集合竞价时段"]+[f"连续交易时段第{i+1}个{freq}" for i in range(num)]


def cal_mean_temp_result(big_df:pd.DataFrame=None,params=None):
    '''
        计算temp结果的平均值
    '''
    param_dir = generate_param_dir(params)
    base_dir=params["base_dir"]
    df_pivot_list=[]
    df_mean_list=[]
    for i in range(len(big_df.columns)-2):
        df_pivot=big_df.pivot_table(index="index",columns="code",values=[str(i)])
        # df_pivot.to_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_pivot_{i}.csv"))

        df_mean=df_pivot.mean(axis=1)
        df_mean.index=pd.to_datetime(df_mean.index)
        # df_mean.to_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean_{i}.csv"))
        df_pivot_list.append(df_pivot)
        df_mean_list.append(df_mean)
    df_pivot=pd.concat(df_pivot_list,axis=0)
    df_mean=pd.concat(df_mean_list,axis=1)
    df_pivot.to_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_pivot.csv"))
    df_mean.to_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.csv"))
    return df_pivot,df_mean
    
def simple_analyze_result(big_df=None,params=None):
    '''
        简单分析temp结果
    '''
    index_code=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    base_dir=params["base_dir"]
    param_dir=generate_param_dir(params)
    if big_df is None:
        big_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2.csv"))

    df_pivot,df_mean=cal_mean_temp_result(big_df,params)

    plt.figure(figsize=(40,6))
    plt.plot(df_mean,label=generate_label_list(params))
    plt.legend()
    plt.title(generate_title_cn(params))
    plt.savefig(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.png"))
    plt.close()

def analyze_diff_freq(params):
    '''
        分析不同频率的结果
    '''
    param_dir = generate_param_dir(params)
    base_dir=params["base_dir"]
    index=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq_input=params["freq"]
    period=params["period"]
    method=params["method"]
    freq_list=["10min","30min","3min","12h"]
    params_dir=generate_param_dir(params)
    # mean_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.csv"))
    # df_pivot,df_mean=cal_mean_temp_result(mean_df,params)
    plt.figure(figsize=(40,6))
    for freq in freq_list:
        params["freq"]=freq
        param_dir = generate_param_dir(params)
        try:
            mean_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.csv"),parse_dates=True,index_col=0)
        except FileNotFoundError:
            print(f"{param_dir} not found")
            continue

        plt.plot(mean_df,label=freq)
    plt.legend()
    freq_text = f"重采样频率：{freq_input}"
    plt.title(f"{generate_title_cn(params).replace(freq_text,'')}  不同采用频率比较")
    # replace_text=f"重采样频率：{freq_input}"
    params_dir=params_dir.replace(f"{freq_input}_","")
    plt.savefig(os.path.join(base_dir, "result", f"{params_dir}_agg_r2_mean_diff_freq.png"))
    plt.close()
    print(f"analyze_diff_freq done")

def analyze_diff_period(params):
    '''
        分析不同时段的结果
    '''
    param_dir = generate_param_dir(params)
    base_dir=params["base_dir"]
    index=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period_input=params["period"]
    method=params["method"]
    period_list=["1","5","10","30","60","90"]
    plt.figure(figsize=(40,6))
    params_dir=generate_param_dir(params)
    for period in period_list:
        params["period"]=period
        param_dir = generate_param_dir(params)
        try:
            mean_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.csv"),parse_dates=True,index_col=0)

        except FileNotFoundError:
            print(f"{param_dir} not found")
            continue
        plt.plot(mean_df,label=period)
    plt.legend()
    period_text = f"每次回归计算时段长度：{period_input}"
    plt.title(f"{generate_title_cn(params).replace(period_text,'')}  不同时段比较")
    params_dir=params_dir.replace(f"{period_input}_","")
    plt.savefig(os.path.join(base_dir, "result", f"{params_dir}_agg_r2_mean_diff_period.png"))
    plt.close()
    print(f"analyze_diff_period done")

def analyze_diff_index(params):
    '''
        分析不同指数的结果
    '''
    param_dir = generate_param_dir(params)
    base_dir=params["base_dir"]
    index_input=params["index_code"]
    X_cols=params["X_cols"]
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    period=params["period"]
    method=params["method"]
    index_list=["SH000300","SH000852"]
    plt.figure(figsize=(40,6))
    params_dir=generate_param_dir(params)
    for index in index_list:
        params["index_code"]=index
        param_dir = generate_param_dir(params)
        try:
            mean_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2_mean.csv"),parse_dates=True,index_col=0)
        except FileNotFoundError:
            print(f"{param_dir} not found")
            continue
        print(mean_df)
        plt.plot(mean_df,label=index)
    plt.legend()
    index_text = f"市场指数选择：{index_input}"
    plt.title(f"{generate_title_cn(params).replace(index_text,'')}  不同指数比较")
    params_dir=params_dir.replace(f"{index_input}_","")
    plt.savefig(os.path.join(base_dir, "result", f"{params_dir}_agg_r2_mean_diff_index.png"))
    plt.close()
    print(f"analyze_diff_index done")

def exam_params(params):
    path=generate_param_dir(params)
    if not os.path.exists(os.path.join(base_dir, path)):
        return False
    return True

def read_big_df(params):
    param_dir = generate_param_dir(params)
    base_dir=params["base_dir"]
    big_df=pd.read_csv(os.path.join(base_dir, "result", f"{param_dir}_agg_r2.csv"),parse_dates=True)
    return big_df


def single_func(params):
    index_code = params["index_code"]
    freq = params["freq"]
    period = params["period"]
    method = params["method"]
    big_df = read_big_df(params)
    simple_analyze_result(big_df, params)
    print(f"index_code: {index_code}, freq: {freq}, period: {period}, method: {method} done")


if __name__=="__main__":
    import argparse
    base_dir=argparse.ArgumentParser()
    base_dir.add_argument("--base_dir", type=str, default=r"D:\科研\Synchronicity\pythonProject2\3.0\result")
    args = base_dir.parse_args()
    base_dir=args.base_dir


    for font in os.listdir("font"):
        # print(font)
        font_manager.fontManager.addfont(os.path.join("font", font))
    plt.rcParams['font.family'] = 'Source Han Serif CN'

    index_code_list=["SH000300","SH000852"]
    
    X_cols=["index"]
    start=dt.datetime(2010,1,5)
    end=dt.datetime(2025,6,30)
    freq_list=["10min","30min","3min","12h"]
    period_list=["1","5","10","30","60","90"]
    method_list=["simple","cross_section"]
    # base_dir=r"D:\科研\Synchronicity\pythonProject2\3.0\result"
    
    params_list=[]

    # for index_code in index_code_list:
    #     for freq in freq_list:
    #         for period in period_list:
    #             for method in method_list:
    #                 params={"index_code":index_code,"X_cols":X_cols,"start":start,"end":end,"freq":freq,"period":period,"method":method,"base_dir":base_dir}
    #                 if not exam_params(params): continue
    #                 params_list.append(params)

    # with mp.Pool(processes=10) as pool:
    #     results = pool.map(single_func, params_list)

    for index_code in index_code_list:
        for period in period_list:
            method="simple"
            params={"index_code":index_code,"X_cols":X_cols,"start":start,"end":end,"freq":"5min","period":period,"method":method,"base_dir":base_dir}
            analyze_diff_freq(params)
    for index_code in index_code_list:
        for freq in freq_list:
            method="simple"
            params={"index_code":index_code,"X_cols":X_cols,"start":start,"end":end,"freq":freq,"period":period,"method":method,"base_dir":base_dir}
            analyze_diff_period(params)
    for period in period_list:
        for freq in freq_list:
            method="simple"
            params={"index_code":index_code,"X_cols":X_cols,"start":start,"end":end,"freq":freq,"period":period,"method":method,"base_dir":base_dir}
            analyze_diff_index(params)




    # index_code="SH000300"
    # X_cols=["index"]
    # start=dt.datetime(2010,1,5)
    # end=dt.datetime(2025,6,30)
    # freq="12h"
    # period="30"
    # method="cross_section"
    # base_dir=r"D:\科研\Synchronicity\pythonProject2\3.0\result"
    # params={"index_code":index_code,"X_cols":X_cols,"start":start,"end":end,"freq":freq,"period":period,"method":method,"base_dir":base_dir}
    # big_df=aggerate_simple_temp_result(params)
    # simple_analyze_temp_result(big_df,params)