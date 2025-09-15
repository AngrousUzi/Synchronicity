from single_cal import single_periodic_cal
from cal_return import get_complete_return

import pandas as pd 
import datetime as dt
import numpy as np
import os
import warnings
# warnings.filterwarnings('default', category=RuntimeWarning)
import multiprocessing as mp
from functools import partial
import argparse

warnings.filterwarnings('ignore', 
                       message='invalid value encountered in scalar divide',
                       category=RuntimeWarning, 
                       module='statsmodels')
warnings.filterwarnings("ignore",category=FutureWarning)

def run_single_calculation(full_code, index_code, composites, df_constant, workday_list, params=None):
    """
    单股票计算工作函数（多进程并行处理适配）
    
    为单一股票执行完整的回归分析计算，包括数据加载、因子匹配、
    回归计算和结果保存。该函数被设计为在多进程环境中调用，
    支持multiprocessing.Pool的并行处理。
    
    参数:
    full_code (str): 完整股票代码（如"SH600000"）
                     用于标识待分析的股票
    index_code (str): 主要市场指数代码（如"SH000300"）
                      作为基准因子用于回归分析
    composites (pd.DataFrame): 指数成分股信息表
                               - index: 股票代码或隐含索引
                               - columns: 包含'full_code', 'index'(行业指数代码)
                               用于查找股票对应的行业分类
    df_constant (pd.DataFrame): 所有解释变量的数据矩阵
                                - index: pd.DatetimeIndex，时间序列
                                - columns: 各类因子代码（市场指数、行业指数等）
                                包含所有可能使用的解释变量数据
    workday_list (list): 有效交易日列表，包含datetime.date对象
                         用于数据的时间范围限制
    params (dict): 计算参数配置字典，必须包含完整的配置信息
                   具体要求参见single_periodic_cal函数说明
    
    返回:
    tuple: 二元组包含股票代码和计算结果
           - full_code (str): 输入的股票代码，用于结果标识
           - result (dict or None): 计算结果或None
             * 成功时: {"r2": pd.DataFrame, "betas": dict}
             * 失败时: None（数据不可用或计算失败）
    
    处理流程:
    1. **参数解析**: 从参数字典中提取所需配置
    2. **因子选择**: 根据X_cols选择市场和/或行业因子
       - 如果包含"industry"：从 composites 中查找对应行业指数
       - 从 df_constant 中提取相应列的数据
    3. **回归计算**: 调用single_periodic_cal执行主计算
    4. **结果处理**: 保存中间结果到文件系统（用于断点续传）
    5. **错误记录**: 记录所有异常情况到日志文件
    
    文件输出:
    - base_dir/temp/r2/{full_code}.csv: R平方值矩阵
    - base_dir/temp/betas/{factor}/{full_code}.csv: 各因子系数矩阵
    - base_dir/error/{full_code}.txt: 错误和警告日志
    
    错误处理:
    1. **数据缺失**: 股票数据不存在或无法加载
    2. **计算失败**: 回归分析过程中的各种异常
    3. **结果为空**: 计算成功但结果为空集
    4. **文件I/O**: 中间结果保存失败（不中断主流程）
    
    注意事项:
    - 确保 composites 中包含目标股票的信息
    - df_constant 必须包含所有需要的因子列
    - params 参数必须完整且格式正确
    - 多进程环境下注意进程间的资源竞争
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
        print(f"all_cal: {full_code} \t No return data")
        error_dir = os.path.join(base_dir, 'error')

        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a+', encoding='utf-8') as f:
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
        with open(os.path.join(error_dir, f'{full_code}.txt'), 'a+', encoding='utf-8') as f:
            f.write('all_cal: No r2 data for this code.\n')
    if betas_dict is not None:
        for key,value in betas_dict.items():
            if not value.empty:
                beta_dir = os.path.join(base_dir, "temp", "betas", key)
                os.makedirs(beta_dir, exist_ok=True)
                value.to_csv(os.path.join(beta_dir, f"{full_code}.csv"))
            else:
                error_dir = os.path.join(base_dir, 'error')
                with open(os.path.join(error_dir, f'{full_code}.txt'), 'a+', encoding='utf-8') as f:
                    f.write('all_cal: No betas data for this code.\n')

    return full_code, {"r2": df_r2, "betas": betas_dict}

def get_composites(index_code):
    """
    获取指数成分股信息的数据处理函数
    
    从 Excel 文件中加载指定指数的成分股信息，并进行数据清洗和格式
    标准化处理。将原始的交易所格式（如"000001.SZ"）转换为统一格式（如"SZ000001"）。
    
    参数:
    index_code (str): 指数代码（如"SH000300"）
                      用于构建 Excel 文件路径: index/{index_code}.xlsx
                      该文件必须包含"证券代码"列
    
    返回:
    pd.DataFrame: 标准化后的成分股信息表，包含以下列：
                  - "full_code" (str): 标准化后的股票代码
                    * 格式: "{exchange}{stock_number}"
                    * 示例: "SH600000", "SZ000001", "SZ300001"
                  - "exg" (str): 交易所代码
                    * "SH": 上海证券交易所
                    * "SZ": 深圳证券交易所  
                  - "num_code" (str): 纯数字股票代码
                    * 示例: "600000", "000001", "300001"
    
    数据处理流程:
    1. **数据加载**: 从 index/{index_code}.xlsx 读取成分股列表
    2. **末尾删除**: 移除最后两行（通常为汇总数据或备注）
    3. **代码解析**: 将"证券代码"列按"."分割为股票代码和交易所
       - 原格式: "000001.SZ", "600000.SH" 
       - 分解为: stock_code="000001", exchange="SZ"
    4. **格式转换**: 重新组合为标准格式
       - 目标格式: "{exchange}{stock_code}"  
       - 转换结果: "SZ000001", "SH600000"
    
    文件格式要求:
    - 文件路径: index/{index_code}.xlsx
    - 必需列: "证券代码" (包含如"000001.SZ"的原始代码)
    - 数据格式: Excel 表格，一行一只股票
    - 特殊处理: 自动删除最后两行的非数据内容
    
    使用场景:
    1. **指数成分股分析**: 为沪深300、中证500等指数加载成分股名单
    2. **批量数据处理**: 为大规模回测分析准备股票清单
    3. **代码标准化**: 将不同数据源的代码格式转换为统一格式
    4. **数据验证**: 核对指数成分股的完整性和准确性
    
    错误处理:
    - FileNotFoundError: 指数Excel文件不存在
    - KeyError: 文件中缺少"证券代码"列
    - IndexError: 数据行数不足（少于2行）
    - ValueError: 股票代码格式不符合预期（不包含"."分隔符）
    
    性能考虑:
    - 文件加载速度取决于 Excel 文件大小
    - 内存占用与成分股数量成正比（通常300-500只）
    - 字符串处理的时间复杂度为O(n)，n为成分股数量
    
    返回数据示例:
    ```
         full_code exg num_code
    0    SH600000  SH   600000
    1    SZ000001  SZ   000001  
    2    SZ300001  SZ   300001
    ...       ...  ..      ...
    ```
    
    注意事项:
    - 确保 index 目录存在且包含目标 Excel 文件
    - 检查 Excel 文件的列名是否为"证券代码"
    - 注意处理特殊字符和空值情况
    - 该函数不设置索引，保持原有数值索引以便后续处理
    """
    composites = pd.DataFrame()
    composites['full_code'] = pd.read_excel(f'index/{index_code}.xlsx')['证券代码']
    composites.drop(index=composites.index[-2:], inplace=True)
    composites['exg'] = composites['full_code'].str.split('.').str[1]      # 提取交易所代码
    composites['num_code'] = composites['full_code'].str.split('.').str[0]     # 提取股票代码
    composites['full_code'] = composites['exg'] + composites['num_code']   
    # composites.set_index('full_code',inplace=True)    # 重新组合为统一格式
    return composites

def prerequisite(index_code,params):
    """
    批量分析的数据预处理和环境准备函数
    
    执行大规模股票分析前的所有准备工作，包括成分股加载、因子数据准备、
    行业分类匹配等。该函数构建分析所需的完整数据基础设施。
    
    参数:
    index_code (str): 主要指数代码（如"SH000300"表示沪深300指数）
                      用于:
                      1. 加载指数成分股列表
                      2. 作为主要市场因子的代理变量
                      3. 构建基准因子数据矩阵的主列
    params (dict): 分析参数配置字典，必须包含:
                   - "start" (datetime): 分析起始日期
                   - "end" (datetime): 分析结束日期  
                   - "freq" (str): 数据频率（"5min", "30min", "12h"等）
                   - "X_cols" (list): 需要的解释变量列表
                     * 必含"index"：市场因子
                     * 可选"industry"：行业因子
                   - "base_dir" (str): 结果输出基础目录
    
    返回:
    tuple: 三元组包含分析所需的完整数据集
           - df_constant (pd.DataFrame): 所有解释变量的时间序列矩阵
             * index: pd.DatetimeIndex，分析期间的所有时间点
             * columns: 因子代码列表（市场指数+各行业指数）
             * values: 各因子在各时间点的收益率数据
             * 特点: 无空值，已通过质量检查
           - composites (pd.DataFrame): 扩展后的成分股信息表
             * 基础列: 来自get_composites()的标准化股票代码信息
             * 扩展列: 如果需要行业因子，增加"index"列（对应行业指数代码）
             * 用途: 股票与行业的映射关系
           - workday_list (list): 实际分析期间的交易日列表
             * 类型: list[datetime.date]
             * 范围: start.date() 到 end.date() 之间的所有交易日
             * 用途: 时间窗口划分和数据对齐
    
    处理流程:
    
    1. **成分股加载**:
       - 调用get_composites()获取指数成分股列表
       - 进行代码格式标准化处理
    
    2. **市场因子准备**:
       - 加载基准指数（固定为SH000300沪深300）的收益率数据
       - 获取完整的交易日历和数据质量报告
       - 构建df_constant矩阵的基础列
    
    3. **行业因子处理**（当X_cols包含"industry"时）:
       - 读取行业-指数映射文件: industry/stock_index_match.csv
       - 加载所有涉及行业指数的收益率数据
       - 将行业因子数据合并到df_constant矩阵
       - 在composites表中添加股票-行业映射关系
    
    4. **数据质量控制**:
       - 保存df_constant到文件以便审查和调试
       - 执行空值检查，发现问题立即中止
       - 确保所有因子数据的时间对齐性
    
    5. **时间范围处理**:
       - 根据start参数过滤workday_list
       - 统计并输出实际分析的交易日数量
       - 验证时间范围的合理性
    
    数据文件要求:
    
    1. **指数成分股文件**: index/{index_code}.xlsx
       - 必需列: "证券代码"
       - 格式: "000001.SZ", "600000.SH"等
    
    2. **行业映射文件**: industry/stock_index_match.csv （如需行业因子）
       - 必需列: "full_code"（股票代码）, "index"（行业指数代码）
       - 格式: CSV文件，标准化的股票-行业对应关系
    
    3. **指数数据**: 通过get_complete_return()加载
       - 基准指数: SH000300（沪深300，固定使用）
       - 行业指数: 根据映射文件动态加载
    
    输出文件:
    - {base_dir}/df_constant.csv: 完整因子矩阵的备份文件
    
    错误处理:
    - FileNotFoundError: 成分股或行业映射文件缺失
    - ValueError: df_constant矩阵存在空值时立即中止
    - 数据加载失败: 各种数据源问题会传播异常
    
    性能考虑:
    - 行业因子数量影响加载时间（通常10-30个行业）
    - 内存占用与时间范围和因子数量成正比
    - 建议对长时间序列使用适当的数据缓存策略
    
    使用场景:
    - 大规模因子分析的数据准备阶段
    - 多因子模型的批量验证
    - 行业轮动策略的数据基础构建
    - 指数增强策略的因子准备
    
    注意事项:
    - 确保所有依赖文件存在且格式正确
    - 基准指数固定为SH000300，不随index_code变化
    - 行业因子为可选项，根据X_cols参数决定是否加载
    - 数据质量检查较严格，发现问题会直接中止程序
    - 建议在正式运行前先进行小规模测试
    """
    start=params["start"]
    end=params["end"]
    freq=params["freq"]
    X_cols=params["X_cols"]
    base_dir=params["base_dir"]
    
    composites=get_composites(index_code)

    
    df_index,workday_list,_=get_complete_return(full_code=index_code,workday_list=None,is_index=True, params=params)
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
    workday_list_path = os.path.join(base_dir, "workday_list.txt")
    with open(workday_list_path, "w") as f:
        f.write(f"workday_list: {workday_list}\n")
    if df_constant.isnull().any().any():
        raise ValueError("df_constant has null values.")
        
    
    return df_constant,composites,workday_list

def all_cal(index_code,df_constant,composites,workday_list,cpu_parallel_num=None, params=None):
    """
    大规模股票数据批量并行计算主函数
    
    管理和协调整个指数成分股组合的大规模回归分析计算。支持单线程
    和多进程并行执行模式，自动处理计算结果的汇总和存储。
    
    该函数是量化研究中的核心执行引擎，能够处理上百只股票的同时分析。
    
    参数:
    index_code (str): 基准指数代码（如"SH000300"）
                      用于标识所分析的指数成分股组合
    df_constant (pd.DataFrame): 所有解释变量的完整数据矩阵
                                来自prerequisite()函数的输出
                                - index: pd.DatetimeIndex，时间序列
                                - columns: 因子代码（市场+行业指数）
                                - values: 各因子的收益率数据
    composites (pd.DataFrame): 成分股信息表，包含股票-行业映射
                               来自prerequisite()函数的输出
                               - 'full_code'列: 所有待分析的股票代码
                               - 'index'列: 对应的行业指数（如需要）
    workday_list (list): 有效交易日列表，包含datetime.date对象
                         用于时间范围限制和窗口划分
    cpu_parallel_num (int, optional): 并行计算的进程数量
                                       - None: 单线程顺序执行
                                       - 0: 使用 (CPU核数-1) 个进程
                                       - 正整数: 指定的进程数量
    params (dict): 完整的分析参数配置，详见single_periodic_cal说明
    
    返回:
    pd.DataFrame: 汇总的R平方值结果矩阵
                  - 通过sum_df()函数对所有股票的r2结果进行加权汇总
                  - 具体结构取决于分析方法和期间设置
                  - 可用于评估整体投资组合的解释能力
    
    执行流程:
    
    1. **参数验证和环境检查**:
       - 验证params参数字典的完整性
       - 检查cross_section方法的频率限制条件
       - 对小期间参数发出性能警告
       - 创建必需的输出目录结构
    
    2. **数据准备**:
       - 提取成分股列表，确定分析目标
       - 为每个因子创建结果存储目录
       - 检查及创建中间文件存储路径
    
    3. **计算执行选择**:
       - **单线程模式** (cpu_parallel_num=None):
         * 顺序遍历每只股票
         * 适合调试和小规模数据
         * 缺点: 计算速度相对较慢
       - **多进程模式** (cpu_parallel_num>0):
         * 使用multiprocessing.Pool并行计算
         * 通过functools.partial预填充函数参数
         * 自动负载均衡和结果收集
    
    4. **结果汇总**:
       - 收集所有子进程的计算结果
       - 分别汇总r2和betas结果字典
       - 处理计算失败的股票（设为None）
       - 调用sum_df()进行最终的数值汇总
    
    5. **中间结果存储**:
       - 每个股票的r2结果: base_dir/temp/r2/{stock_code}.csv
       - 各因子系数结果: base_dir/temp/betas/{factor}/{stock_code}.csv
       - 支持断点续传和结果审查
    
    并行计算优化:
    
    - **进程池管理**: 使用with语句确保资源清理
    - **内存优化**: 通过结果返回而非共享状态
    - **负载均衡**: Pool.map自动分配任务
    - **异常隔离**: 单个股票失败不影响其他计算
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

    if method=="cross_section" :
        if not freq.endswith("min") and not freq.endswith("h"):
            raise ValueError("Cross-section method is only supported for minute or hour frequency.")
        if (period=="1" or period=="2"):
            warnings.warn(f"Cross-section method may result in all NaN or 1 for period={period}", RuntimeWarning)


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
    """
    多股票R平方值结果的数值汇总函数
    
    将多个股票的R平方值矩阵进行加权汇总，用于评估整个投资组合
    的整体解释能力。处理数据缺失情况，自动跳过无效值。
    
    参数:
    r2_result_dict (dict): R平方值结果字典
                           - keys: str，股票代码（如"SH600000"）
                           - values: pd.DataFrame or None，R平方值矩阵
                             * 有效时: DataFrame，包含各期间、各时点的r2值
                             * 无效时: None（数据缺失或计算失败）
    beta_index_result_dict (dict, optional): 市场因子系数结果（当前未使用）
                                              保留接口，用于未来扩展
    beta_industry_result_dict (dict, optional): 行业因子系数结果（当前未使用）
                                                 保留接口，用于未来扩展
    
    返回:
    pd.DataFrame or None: 汇总后的R平方值矩阵
                          - 成功时: 各股票R平方值的算术平均值
                          - 全部失败时: None（所有输入都为None）
                          - index: 期间开始日期
                          - columns: 时点编号（取决于分析方法）
                          - values: 汇总后R平方值，可解释为组合解释能力
    
    汇总算法:
    
    1. **初始化**: 使用第一个有效数据作为基准矩阵
    2. **逐个加权**: 使用pandas.DataFrame.add()进行元素级加法
       - fill_value=0: 缺失值填充0后参与计算
       - 保持原有数据的索引结构和时间对齐
    3. **缺失处理**: 跳过None值，不参与数值计算
    4. **结果验证**: 记录并报告所有数据缺失的股票
    
    数学含义:
    
    - **直接加总**: 算术平均，适用于同等权重的组合分析
    - **缺失填0**: 隐含假设数据缺失时解释能力为0
    - **结果解释**: 汇总值越高表示组合整体的因子解释能力越强
    - **对比标准**: 可与单个股票的R平方值进行比较
    
    使用场景:
    
    1. **组合优化**: 评估不同股票组合的整体解释能力
    2. **指数对比**: 将成分股组合与整个指数表现对比
    3. **策略评估**: 衡量量化策略的整体有效性
    4. **风险分散**: 验证分散化投资的风险降低效果
    5. **数据质量**: 检查整体数据集的完整性
    
    副作用和输出:
    
    - **控制台输出**: 打印数据缺失的股票列表
    - **静默处理**: 不中断程序执行，但记录问题
    - **广播通知**: 提醒用户关注数据质量问题
    
    错误处理:
    
    - **空字典**: 返回None，不抛出异常
    - **类型错误**: pandas操作的异常传播给调用方
    - **维度不匹配**: 自动对齐或填充0处理
    
    性能考虑:
    
    - **时间复杂度**: O(N × M × K)，N为股票数，M为期间数，K为时点数
    - **内存占用**: 与输入数据规模成正比，需临时存储中间结果
    - **数值计算**: pandas底层优化，利用向量化计算
    
    限制和注意事项:
    
    1. **数据类型**: 仅处理R平方值，不适用于其他类型指标
    2. **权重假设**: 默认各股票等权重，未考虑市值权重
    3. **缺失处理**: 填0策略可能不适合所有情况
    4. **统计意义**: 算术平均可能不是最佳的汇总方法
    5. **参数保留**: beta相关参数当前未使用，仅为接口兼容性
    
    改进建议:
    
    - **加权汇总**: 按照市值或流动性进行加权平均
    - **稳健汇总**: 使用中位数或截尾平均抗异常值
    - **分层汇总**: 按行业或风格对股票进行分组汇总
    - **结果验证**: 增加统计检验和置信区间计算
    """
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
    """
    断点续传计算函数
    
    从指定的股票代码开始恢复中断的批量计算任务，利用已存在的中间结果
    文件实现部分恢复能力。适用于长时间任务的意外中断恢复场景。
    
    参数:
    index_code (str): 基准指数代码（如"SH000300"）
                      传递给底层计算函数
    df_constant (pd.DataFrame): 所有解释变量的数据矩阵
                                来自prerequisite()函数的输出
    X_cols (list): 解释变量列表（注意：与params中的字段重复）
                   用于与all_cal()的接口兼容
    composites (pd.DataFrame): 完整的成分股信息表
                               - 必须包含'full_code'列
                               - 用于确定续传起始位置和结果收集范围
    workday_list (list): 有效交易日列表，包含datetime.date对象
                         传递给底层计算函数
    cpu_parallel_num (int): 并行计算进程数，传递给all_cal()
    continue_code (str): 断点续传的起始股票代码
                         - 必须是composites中的有效代码
                         - 从该股票开始及其之后的所有股票都会被重新计算
                         - 必须符合标准化格式（如"SH600000"）
    params (dict): 完整的分析参数配置，详见all_cal说明
                   必须包含base_dir等关键参数
    
    返回:
    pd.DataFrame: 汇总后的R平方值矩阵
                  - 结果来自所有成分股（包括之前已完成和新计算的）
                  - 通过sum_df()进行最终数值汇总
                  - 结构与正常all_cal()的返回值一致
    
    工作流程:
    
    1. **参数验证**: 检查params字典的完整性
    2. **位置定位**: 在composites中找到continue_code的位置索引
    3. **列表裁剪**: 从指定位置开始到结束的子集
    4. **部分计算**: 调用all_cal()仅计算被裁剪的股票子集
    5. **结果收集**: 遍历所有成分股，加载中间结果文件
    6. **数据汇总**: 将所有r2结果进行最终汇总
    
    续传机制:
    
    - **文件检测**: 通过检查temp/r2/目录下的CSV文件来判断完成状态
    - **部分重算**: 只重新计算缺失或指定位置之后的股票
    - **结果合并**: 将旧结果和新结果统一汇总
    - **一致性保证**: 确保续传结果与全新计算结果一致
    
    限制和注意事项:
    
    1. **索引要求**: composites必须使用RangeIndex（数值索引）
    2. **代码匹配**: continue_code必须在composites['full_code']中存在
    3. **文件依赖**: 依赖base_dir/temp/r2/目录下的中间文件
    4. **参数重复**: X_cols参数与params中的X_cols可能重复
    
    错误处理:
    
    - **代码不存在**: 如果continue_code不在列表中会抛出IndexError
    - **文件缺失**: 打印警告但不中断，该股票结果设为None
    - **计算异常**: all_cal()的异常会直接传播
    - **文件格式**: CSV读取异常被捕获并处理
    
    性能考虑:
    
    - **I/O开销**: 需要读取所有成分股的中间结果文件
    - **内存占用**: 同时加载所有股票的r2数据到内存
    - **计算重复**: 部分股票可能被重复计算（如果续传位置较前）
    - **网络存储**: 分布式环境下注意文件同步问题
    
    使用场景:
    
    1. **任务中断恢复**: 系统故障或用户中断后的恢复
    2. **增量更新**: 新增成分股时只计算新股票
    3. **错误重试**: 部分股票计算失败时的重新处理
    4. **参数调优**: 改变计算参数后的部分重计
    5. **资源限制**: 在有限计算资源下的分批处理
    
    实用建议:
    
    - **备份策略**: 重要任务前备份temp/目录
    - **进度监控**: 定期检查中间文件的生成情况
    - **空间清理**: 定期清理临时文件释放磁盘空间
    - **版本控制**: 为不同参数配置使用不同的base_dir
    - **执行日志**: 记录续传的起始位置和完成情况
    
    示例用法:
    ```python
    # 从"SH600100"开始续传计算
    result = continue_cal(
        index_code="SH000300",
        df_constant=constant_data,
        X_cols=["index"],
        composites=stock_list,
        workday_list=trading_days,
        cpu_parallel_num=8,
        continue_code="SH600100",
        params=analysis_params
    )
    ```
    """
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
    all_cal(index_code=index_code,df_constant=df_constant,composites=composites_continue,workday_list=workday_list,cpu_parallel_num=cpu_parallel_num, params=params)
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
    parser.add_argument('--period', type=str, default='1', help='Period for calculation')
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
    os.makedirs(os.path.join(base_dir, "error"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "temp"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "temp", "r2"), exist_ok=True)
    for col in X_cols:
        os.makedirs(os.path.join(base_dir, "temp", "betas", col), exist_ok=True)

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