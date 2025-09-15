# 导入必要的数据分析和可视化库
from llvmlite.ir import Value
import pandas as pd                       # 用于数据处理和分析
import numpy as np                        # 用于数值计算
import statsmodels.api as sm              # 用于统计建模（回归分析）
# from statsmodels.tools.sm_exceptions import MissingDataError  # 处理缺失数据异常

# import matplotlib.pyplot as plt          # 用于数据可视化
import datetime as dt                     # 用于日期时间处理

from cal_return import get_complete_return

import os

from utils import log_error,parse_timedelta,convert_freq_to_min
from get_cache import get_cache_text

def deal_index_unmatching_error(full_code,df_X,df_stock,base_dir):
    if df_stock.shape[0]<df_X.shape[0]:
        missing_set=set(df_X.index.to_list())-set(df_stock.index.to_list())
        missing_list=list(missing_set)
        missing_dates=[date.date() for date in missing_list]
        # try:
        workday_list,error_list=get_cache_text(full_code,dt.datetime(2010,1,5),dt.datetime(2025,6,30))
        # 将pandas Timestamp转换为date对象，并过滤掉在error_list中的日期
        removed=[]
        for missing_date in missing_dates:
            if missing_date in error_list:
                removed.append(missing_date)
        # missing_set=set(missing_dates)
        # print(missing_dates)
        # print(missing_set)
        # print(removed)
        # print(missing_set)
        # print(error_list)
        # except:
        #     pass
        if len(removed)!=len(missing_dates):
            log_error(f'{full_code} 数据长度不一致,start:{df_X.index[0]},end:{df_X.index[-1]}',full_code, base_dir=base_dir)
            log_error(f'stock_length:{df_stock.shape[0]},index_length:{df_X.shape[0]}',full_code, base_dir=base_dir)
        # if len(missing_set)<3:
            log_error(f'缺少的index为{sorted(missing_list)}',full_code, base_dir=base_dir)
            if len(missing_dates)!=len(missing_set):
                log_error(f"对应有问题的日期为{missing_dates}",full_code, base_dir=base_dir)
        # df_stock.to_csv(f'error_list/return/{full_code}.csv')

        df_X=df_X.loc[df_stock.index].copy()
        # df_industry=df_industry.loc[df_stock.index].copy() 
    elif df_stock.shape[0]>df_X.shape[0]:
        raise ValueError(f'{full_code} 数据长度不一致,stock_length:{df_stock.shape[0]},index_length:{df_X.shape[0]}')
        # df_industry=df_industry.loc[df_stock.index].copy()
    return df_X,df_stock


def simple_cal(df_stock,df_X,full_code, base_dir: str = ""):
    """
    单期间普通最小二乘（OLS）回归分析函数
    
    对单只股票的收益率与市场指数和/或行业指数收益率进行线性回归。
    自动处理数据对齐问题，并记录数据质量问题到日志文件。
    
    回归模型:
    R_stock = α + β1*R_index + β2*R_industry + ε
    其中：R 表示收益率，α 为截距项，β 为回归系数
    
    参数:
    df_stock (pd.Series): 股票收益率时间序列数据（因变量）
                          - index: pd.DatetimeIndex，交易时间戳
                          - values: float，收益率数据
    df_X (pd.DataFrame): 解释变量数据集（自变量）
                         - index: pd.DatetimeIndex，与df_stock对应的时间戳
                         - columns: 变量名称，如['index', 'industry']
                         - values: float，市场指数和行业指数收益率
    full_code (str): 完整的股票代码（如"SH000300"）
                     用于错误日志记录和调试信息
    base_dir (str, optional): 基础目录路径，默认为空字符串
                              错误日志保存到 base_dir/error/{full_code}.txt
    
    返回:
    dict: 回归结果字典，包含以下键值对：
          - "r2" (float or np.nan): R平方值，衡量模型解释能力
            * 范围: [0, 1]，越接近1表示拟合度越好
            * 如果数据不足或无法回归则返回np.nan
          - "betas" (pd.Series): 回归系数向量（不包含截距项）
            * index: df_X的列名，如['index', 'industry']
            * values: 对应的回归系数值
            * 解释: beta_index表示对市场风险的暴露度
    
    数据处理逻辑:
    1. **数据验证**: 检查数据量是否足够进行回归（>1观测）
    2. **索引对齐**: 自动处理Y和X数据的时间索引不匹配问题
    3. **缺失处理**: 取交集，删除不匹配的时间点
    4. **日志记录**: 详细记录数据对齐问题和缺失情况
    5. **OLS回归**: 使用statsmodels进行最小二乘回归
    
    回归模型详情:
    - 自动添加截距项（常数项）
    - 支持多元回归（市场 + 行业因子）
    - 返回的betas不包含截距项，仅包含解释变量系数
    
    异常处理:
    - 数据不足时返回 NaN 值而非抛出异常
    - 数据对齐问题会被记录但不中断计算
    - 回归计算失败时会自动处理
    
    应用场景:
    - 单期间的CAPM模型估计
    - Fama-French三因子模型
    - 风险因子暴露度分析
    - 模型性能评估（R平方值）
    
    性能考虑:
    - 如果数据量较大，建议使用分批处理
    - 回归计算的时间复杂度为O(n*k^2)，n为样本数，k为变量数
    """
    # 无法回归
    # df_index=df_X.iloc[:,0]
    if df_X.shape[0] <= 1 or df_stock.shape[0]<=1:
        if df_X.shape[0] <=1:
            log_error(f'single_cal: df_X数据长度小于1',full_code, base_dir=base_dir)
        if df_stock.shape[0] <=1:
            log_error(f'single_cal: df_stock数据长度小于1',full_code, base_dir=base_dir)
        nan_series = pd.Series(np.nan, index=df_X.columns)
        return {"r2": np.nan, "betas": nan_series}
    # if len(X_names)>1:
        # df_indusstry=df_X.iloc[:,1]
    # 数据对齐
    # print(df_stock.index)
    # print(df_X.index)

    if df_stock.shape[0]!=df_X.shape[0]:
        df_X,df_stock=deal_index_unmatching_error(full_code,df_X,df_stock,base_dir)




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
    """
    横截面回归分析函数
    
    对每个交易日内的不同时点分别进行回归分析，用于研究日内不同时段的
    市场风险因子暴露度变化。每个时点使用该时点在所有交易日的数据进行回归。
    
    分析逻辑:
    - 将每个交易日按时间顺序分为total_num个时点
    - 对每个时点位置(nth)，收集所有交易日该位置的收益率数据
    - 对每个时点的数据进行独立的回归分析
    - 最终得到total_num组回归结果
    
    参数:
    df_stock (pd.Series): 股票收益率时间序列数据
                          - index: pd.DatetimeIndex，包含日期和具体时间
                          - values: float，收益率数据
                          - 要求：每个交易日有相同数量的时点数据
    df_X (pd.DataFrame): 解释变量数据集（市场指数、行业指数等）
                         - index: pd.DatetimeIndex，与df_stock对应
                         - columns: 因子名称，如['index', 'industry']
                         - values: float，各因子的收益率数据
    full_code (str): 完整的股票代码（如"SH000300"）
                     用于错误日志记录和调试追踪
    total_num (int): 每个交易日内的时点总数
                     - 通常由频率决定：240//freq_minutes + 1
                     - 如5分钟频率：240//5 + 1 = 49个时点
                     - 包括开盘、连续交易时段、收盘等关键时点
    base_dir (str, optional): 基础目录路径，默认为空字符串
                               错误日志保存到 base_dir/error/ 目录下
    
    返回:
    dict: 横截面回归结果字典，包含以下键值对：
          - "r2" (pd.Series): 各时点的R平方值
            * index: int，时点序号 [0, 1, 2, ..., total_num-1]
            * values: float，对应时点的模型拟合优度
            * 解释：衡量该时点市场因子的解释能力
          - "betas" (dict): 各因子在各时点的回归系数
            * keys: str，因子名称（df_X的列名）
            * values: pd.Series，该因子在各时点的系数值
            * index: int，时点序号
            * 解释：beta["index"][0]表示第0个时点的市场beta
    
    应用场景:
    1. **日内风险模式研究**: 分析开盘、盘中、收盘等不同时段的风险特征
    2. **流动性模式分析**: 研究市场流动性对风险因子暴露的影响
    3. **交易策略优化**: 识别风险因子暴露度较低的最优交易时点
    4. **市场微观结构**: 分析日内价格发现和信息传播模式
    5. **风险管理**: 动态调整日内风险敞口
    
    技术实现:
    1. **数据分组**: 使用groupby(date).nth(n)选取各日第n个时点
    2. **并行回归**: 对total_num个时点分别进行独立回归
    3. **结果整合**: 将各时点的回归结果整合为时间序列
    4. **异常处理**: 自动处理数据不足或回归失败的情况
    
    数据要求:
    - 每个交易日必须有相同的时点数量
    - 时点编号从0开始，按时间顺序递增
    - 数据质量：充足的观测值数量以支持回归分析
    
    性能考虑:
    - 时间复杂度: O(total_num × n × k²)，n为交易日数，k为因子数
    - 内存需求: 与时点数量和交易日数量成正比
    - 建议：对于长时间序列，考虑分批处理
    
    与simple_cal的区别:
    - simple_cal: 使用全部时点数据进行单次回归
    - simple_cross_section_cal: 对每个时点位置进行独立回归
    - 前者关注整体风险特征，后者关注时点间差异
    
    注意事项:
    - total_num应该与实际数据的时点数量匹配
    - 某些时点可能因为数据不足而返回NaN
    - 建议预先验证数据的时间结构完整性
    """
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
    """
    单股票多期间回归分析主函数
    
    对单只股票进行多个时间窗口的滚动回归分析，支持两种分析方法：
    1. simple方法：每个期间进行整体回归
    2. cross_section方法：每个期间内按时点进行横截面回归
    
    这是量化研究中的核心函数，用于分析股票风险因子暴露度的时变特征。
    
    参数:
    full_code (str): 完整的股票代码（如"SH600000"）
                     用于获取对应的股票收益率数据
    df_X (pd.DataFrame): 解释变量数据（市场指数、行业指数等）
                         - index: pd.DatetimeIndex，时间序列索引
                         - columns: 因子名称，如['index', 'industry']
                         - values: 各因子的收益率数据
    workday_list (list): 工作日列表，包含datetime.date对象
                         用于确定分析的有效交易日范围
    params (dict): 参数配置字典，必须包含以下键：
                   - "X_cols" (list): 解释变量列名
                   - "start" (datetime): 分析开始日期
                   - "end" (datetime): 分析结束日期
                   - "freq" (str): 数据频率，如"5min", "30min", "12h"
                   - "period" (str): 回归窗口，如"1", "5", "20", "full"
                   - "method" (str): 分析方法，"simple"或"cross_section"
                   - "base_dir" (str): 基础目录路径
    
    返回:
    dict or None: 分析结果字典或None（当数据不可用时）
                  成功时包含以下键值对：
                  - "r2" (pd.DataFrame): R平方值矩阵
                    * index: 期间开始日期
                    * columns: 时点编号（simple方法只有第0列）
                    * values: 对应的拟合优度
                  - "betas" (dict): 各因子的回归系数矩阵字典
                    * keys: 因子名称（与df_X列名对应）
                    * values: pd.DataFrame，系数矩阵
                    * 结构与r2相同
    
    分析方法详解:
    
    1. **Simple方法** (method="simple"):
       - 每个期间使用全部数据点进行单次回归
       - 适用于研究整体风险特征的时变性
       - 计算效率高，结果易于解释
       - 输出：每个期间一个回归结果
    
    2. **Cross-section方法** (method="cross_section"):
       - 每个期间内按日内时点位置进行横截面回归
       - 适用于研究日内风险模式的变化
       - 能够捕捉更细粒度的风险特征
       - 输出：每个期间×每个时点的回归结果矩阵
    
    期间设置:
    - period="full": 使用全部数据作为单一期间
    - period="N": 使用N个交易日作为滚动窗口
    - 期间之间无重叠，采用非重叠滑动窗口
    
    处理流程:
    1. **数据获取**: 调用get_complete_return获取股票收益率
    2. **参数解析**: 解析期间长度和分析方法
    3. **时点计算**: 根据频率计算每日时点数量
    4. **滚动分析**: 对每个时间窗口执行回归分析
    5. **结果整合**: 将各期间结果合并为完整矩阵

    
    错误处理:
    - 如果股票数据不可用，返回None
    - 某个期间数据不足时，该期间结果为NaN
    - 所有错误信息记录到相应的日志文件
    
    性能考虑:
    - 时间复杂度取决于期间数量和回归方法
    - Simple方法: O(periods × n × k²)
    - Cross-section方法: O(periods × time_points × n × k²)
    - 其中n为每期间观测数，k为因子数
    
    数据要求:
    - 股票和指数数据必须时间对齐
    - Cross-section方法要求每日时点数量一致
    - 建议每个期间至少有10个以上观测值
    
    注意事项:
    - workday_list会根据start日期进行过滤
    - 如果某期间数据为空会被跳过
    - 返回结果的索引为期间开始日期
    - Cross-section方法仅适用于高频数据（分钟级别）
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
        # print(df_X_period.head(2))
        #对于选取1天的情况，可能出现的问题
        if df_X_period.shape[0]==0: continue
        df_stock_period=df_stock.loc[pd.Timestamp(period_start):pd.Timestamp(period_end)]
        # print(df_stock_period)
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
        # break
    return {"r2":r2s_all,"betas":betas_dict}
    

if __name__=="__main__":
    start=dt.datetime(2010,1,5)
    end=dt.datetime(2025,6,30)
    freq="30min"
    method="cross_section"
    period="30"
    X_cols=["index"]
    params={"start":start,"end":end,"freq":freq,"base_dir":"test","method":method,"period":period,"X_cols":X_cols}
    df_index,workday_list,_=get_complete_return(full_code="SH000300",workday_list=None,is_index=True, params=params)
    # print(workday_list)
    # df_industry,_,error_list=get_complete_return(full_code="SH000070",start=start,end=end,freq=freq,workday_list=workday_list,is_index=True)

    # for period in ["full","10"]:
    # for full_code in [""]
    full_code="SH600000"
    print(full_code)
    # results=single_periodic_cal(full_code=full_code,df_index=df_index,df_industry=df_industry,workday_list=workday_list,params=params)
    # results.to_csv(f'test_results_{period}_{method}.csv')
        
    # method="cross_section"
    # for period in ["full","10d"]:
    
    # df_X=pd.concat([df_index,df_industry],axis=1)
    # df_X.columns=["index","industry"]
    # print(df_X)
    df_X=pd.DataFrame(df_index)
    df_X.columns=X_cols

    results=single_periodic_cal(full_code=full_code,df_X=df_X,workday_list=workday_list,params=params)
    os.makedirs("test",exist_ok=True)
    results["r2"].to_csv(os.path.join("test",f'test_results_{period}_{method}.csv'))
    results["betas"]["index"].to_csv(os.path.join("test",f'test_betas_index_{period}_{method}.csv'))
    # results["betas"]["industry"].to_csv(f'test_betas_industry_{period}_{method}.csv')