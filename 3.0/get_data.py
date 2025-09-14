import pandas as pd
import numpy as np
import datetime as dt
import os

def get_high_freq_data(start:dt.datetime=None,end:dt.datetime=None,exg:str=None,full_code:str=None,workday_list:list=None, base_dir: str = ""):
    """
    高频股票数据加载函数
    
    从文件系统加载指定时间范围内的分钟级高频股票交易数据。
    支持两种工作模式：自动发现交易日和按指定交易日加载。
    主要用于日内高频分析和策略回测。
    
    参数:
    start (datetime.datetime, optional): 数据加载的起始日期
                                         仅在workday_list为None时使用
                                         用于自动发现交易日范围
    end (datetime.datetime, optional): 数据加载的结束日期
                                       仅在workday_list为None时使用
                                       与start参数配合使用
    exg (str, optional): 交易所代码，用于构建文件路径
                         - "SH": 上海证券交易所
                         - "SZ": 深圳证券交易所
                         - 从 full_code 的前两位提取
    full_code (str, optional): 完整的股票或指数代码
                               格式: "{exchange}{stock_number}"，如"SH600000"
                               用于构建文件路径和错误日志
    workday_list (list, optional): 指定的交易日列表
                                    - None: 自动发现模式，遍历start到end间所有日期
                                    - list[datetime.date]: 指定模式，仅加载列表中的日期
    base_dir (str, optional): 基础目录路径，默认为空字符串
                              错误日志将保存到 base_dir/error/{full_code}.txt
    
    返回:
    tuple: 三元组包含加载的数据和辅助信息
           - df_stock (pd.DataFrame or None): 加载的股票数据
             * 成功时: 包含'Price'列的DataFrame
               - index: pd.DatetimeIndex，精确到分钟的交易时间戳
               - columns: ['Price']，包含交易价格数据
               - 数据已按时间排序（升序）
             * 失败时: None（无数据文件或所有文件缺失）
           - workday_list (list): 实际找到数据的交易日列表
             * list[datetime.date]，按时间顺序排列
             * 仅包含成功加载数据的日期
           - error_list (list): 数据缺失的日期列表
             * 自动发现模式: 空列表（缺失日期不被记录）
             * 指定模式: 包含数据文件不存在的日期
    
    数据文件结构:
    
    - **文件路径格式**: data/ws{YYYYMMDD}fb/{exchange}/{full_code}.csv
    - **目录结构**: 每个交易日一个目录
    - **文件内容**: CSV格式，必须包含'Time'和'Price'列
    - **时间格式**: 'Time'列必须能被pd.to_datetime()解析
    
    工作模式:
    
    1. **自动发现模式** (workday_list=None):
       - 遍历start到end间的所有自然日
       - 尝试加载每个日期的数据文件
       - 成功加载的日期自动加入workday_list
       - 失败的日期被静默跳过（非交易日或数据缺失）
    
    2. **指定模式** (workday_list不为None):
       - 仅加载指定交易日的数据文件
       - 数据缺失的日期记录到error_list
       - 打印警告信息并记录到错误日志
       - 适用于已知交易日列表的快速加载
    
    数据处理流程:
    
    1. **文件加载**: 根据模式遍历目标日期
    2. **数据合并**: 使用pd.concat()合并每日数据
    3. **索引设置**: 将'Time'列设置为时间索引
    4. **数据排序**: 按时间升序排列
    5. **列选择**: 仅保留'Price'列
    6. **质量检查**: 验证数据的完整性
    
    错误处理和日志:
    
    - **文件缺失**: FileNotFoundError被捕获，跳过该日期
    - **数据为空**: 所有文件缺失时返回None
    - **日志记录**: 所有错误和警告信息记录到错误日志
    - **控制台输出**: 重要问题的实时通知
    
    性能考虑:
    
    - **内存优化**: 逐日加载避免大内存占用
    - **I/O效率**: 文件系统访问次数与日期数量成正比
    - **时间复杂度**: O(D × N)，D为日期数，N为每日数据点数
    - **空间复杂度**: O(D × N)，需存储所有数据点
    
    使用场景:
    
    1. **策略回测**: 加载历史高频数据进行回测测试
    2. **量化分析**: 分钟级别的交易信号生成
    3. **风险建模**: 日内风险因子暴露度分析
    4. **高频研究**: 市场微观结构研究
    5. **实时系统**: 做市和风控系统的数据基础
    
    限制和注意事项:
    
    1. **文件依赖**: 依赖data/目录下的标准文件结构
    2. **内存限制**: 大时间范围可能导致内存不足
    3. **数据质量**: 不验证数据的正确性和完整性
    4. **时区假设**: 默认数据为本地时区
    5. **并发安全**: 多进程同时访问相同文件时需谨慎
    
    建议用法:
    
    - **小规模数据**: 优先使用自动发现模式
    - **大规模数据**: 使用指定模式减少文件系统访问
    - **数据校验**: 先加载小量样本数据验证格式
    - **错误处理**: 定期检查错误日志文件
    - **内存管理**: 长时间范围请考虑分批加载
    """
    
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
    """
    低频股票数据加载函数
    
    从预处理的CSV文件加载日级或周级等低频股票数据。
    与get_high_freq_data()相比，该函数加载速度更快，适合大规模历史分析。
    
    参数:
    start (datetime.datetime, optional): 起始日期（当前未使用）
                                         保留参数，用于与get_high_freq_data接口统一
    end (datetime.datetime, optional): 结束日期（当前未使用）
                                       保留参数，用于与get_high_freq_data接口统一
    exg (str, optional): 交易所代码（当前未使用）
                         保留参数，用于与get_high_freq_data接口统一
    full_code (str, optional): 完整的股票或指数代码
                               用于构建文件路径: {base_dir}/{full_code}.csv
                               必须是有效的文件名前缀
    workday_list (list, optional): 预期的交易日列表
                                    用于计算数据缺失情况
                                    list[datetime.date]
    base_dir (str, optional): 数据文件的基础目录路径
                              默认为当前目录（空字符串）
                              数据文件路径: {base_dir}/{full_code}.csv
    
    返回:
    tuple: 三元组包含加载的数据和辅助信息
           - df (pd.DataFrame): 加载的低频数据
             * index: pd.DatetimeIndex，日期索引（日级或周级）
             * columns: ['Price']，价格数据
             * 数据已经被预处理，包括除权调整等
           - workday_list (list): 输入的交易日列表（原样返回）
             * list[datetime.date]，与输入参数相同
             * 用于下游函数的一致性处理
           - error_list (list): 数据缺失的交易日列表
             * list[datetime.date]，在workday_list中但不在数据文件中
             * 用于识别和记录数据质量问题
    
    数据文件要求:
    
    1. **文件路径**: {base_dir}/{full_code}.csv
    2. **文件格式**: 标准CSV文件，第一列为日期索引
    3. **日期列**: 必须能被pd.to_datetime()正确解析
    4. **价格列**: 必须包含'Price'列
    5. **编码格式**: 建议UTF-8编码以支持中文
    
    使用场景:
    
    1. **长周期分析**: 多年历史数据的趋势分析
    2. **策略回测**: 日级或周级策略的历史验证
    3. **基本面分析**: 结合财务数据的价值投资分析
    4. **宏观研究**: 经济周期和市场风格轮动研究
    5. **组合优化**: 大量股票的长期表现对比
    
    性能优势:
    
    - **加载速度**: 单文件加载，比高频数据快数百倍
    - **内存效率**: 数据量小，内存占用低
    - **磁盘I/O**: 仅一次文件访问，减少磁盘压力
    - **网络传输**: 文件小，适合远程数据库访问
    
    处理流程:
    
    1. **文件加载**: 使用pd.read_csv()加载数据
    2. **索引设置**: 第一列作为日期索引
    3. **日期解析**: 确保索引为DatetimeIndex类型
    4. **列过滤**: 仅保留'Price'列减少内存占用
    5. **缺失检查**: 对比workday_list计算数据缺失
    6. **结果返回**: 封装成统一的返回格式
    
    错误处理:
    
    - **文件不存在**: 抛出FileNotFoundError给调用方处理
    - **文件格式错误**: pandas的解析异常传播给调用方
    - **列缺失**: KeyError异常传播给调用方
    - **日期解析失败**: pd.to_datetime()异常传播
    
    限制和注意事项:
    
    1. **文件依赖**: 必须存在对应的CSV文件
    2. **格式固定**: 对文件结构有严格要求
    3. **参数未用**: start/end/exg参数当前未使用
    4. **数据质量**: 不验证数据的正确性
    5. **时区假设**: 默认数据为本地时区
    
    与get_high_freq_data的对比:
    
    | 特征 | get_low_freq_data | get_high_freq_data |
    |------|------------------|--------------------|
    | 数据频率 | 日级/周级 | 分钟级 |
    | 加载速度 | 快 | 慢 |
    | 内存占用 | 小 | 大 |
    | 数据量 | 小 | 大 |
    | 文件数量 | 1个 | 数百个 |
    | 灵活性 | 低 | 高 |
    | 适用场景 | 长期分析 | 短期交易 |
    
    使用建议:
    
    - **数据验证**: 初次使用时检查数据文件的格式
    - **路径管理**: 使用绝对路径避免路径歧义
    - **错误处理**: 做好文件不存在的异常处理
    - **数据备份**: 重要数据文件请做好备份
    - **版本控制**: 考虑使用版本控制系统管理数据文件
    
    示例用法:
    ```python
    # 加载日级数据
    df, workdays, errors = get_low_freq_data(
        full_code="SH000300",
        workday_list=trading_days,
        base_dir="daily_data"
    )
    ```
    """
    df=pd.read_csv(os.path.join(base_dir,f"{full_code}.csv"),index_col=0,parse_dates=True)
    df.index=pd.to_datetime(df.index)
    df=df[['Price']]
    error_list=list(set(workday_list)-set(df.index.date))
    return df,workday_list,error_list


def get_data(start:dt.datetime=None,end:dt.datetime=None,exg:str=None,full_code:str=None,workday_list:list=None, base_dir: str = ""):
    """
    通用数据加载接口函数（高频数据专用）
    
    这是一个对get_high_freq_data()的简单封装，提供了统一的数据加载接口。
    当前版本中默认加载高频数据，未来版本可能扩展为智能选择。
    
    该函数主要用于保持API的向后兼容性和简化调用方式。
    
    参数:
    start (datetime.datetime, optional): 数据加载的起始日期
                                         传递给底层get_high_freq_data()函数
    end (datetime.datetime, optional): 数据加载的结束日期
                                       传递给底层get_high_freq_data()函数
    exg (str, optional): 交易所代码（"SH"或"SZ"）
                         用于构建数据文件的路径
    full_code (str, optional): 完整的股票或指数代码
                               格式示例: "SH000001", "SZ000001", "SH600000"
                               用于指定具体的数据目标
    workday_list (list, optional): 交易日列表
                                    - None: 自动从 start 到 end 发现交易日
                                    - list[datetime.date]: 指定交易日列表
    base_dir (str, optional): 数据目录的基础路径
                              默认为空字符串（使用当前目录）
                              错误日志保存目录
    
    返回:
    tuple: 三元组，与get_high_freq_data()的返回值完全相同
           - df_stock (pd.DataFrame or None): 加载的股票交易数据
             * 成功时: 包含'Price'列的高频数据
             * 失败时: None
           - workday_list (list): 实际加载数据的交易日列表
             * list[datetime.date]，按时间顺序排列
           - error_list (list): 数据缺失的交易日列表
             * list[datetime.date]
    
    内部实现:
    直接调用 get_high_freq_data() 函数，传递所有参数。
    这种设计保证了:
    1. **功能一致性**: 与get_high_freq_data完全相同的行为
    2. **接口稳定性**: API升级时不影响现有代码
    3. **维护简单性**: 集中在一个地方维护核心逻辑
    
    使用场景:
    
    1. **遗留代码兼容**: 保持旧版API的兼容性
    2. **简化调用**: 隐藏底层实现的复杂性
    3. **统一接口**: 为不同的数据类型提供一致的调用方式
    4. **未来扩展**: 为智能选择数据加载方式预留空间
    5. **调试和测试**: 提供简化的调试入口
    
    设计理念:
    
    - **单一职责**: 仅作为数据加载的统一入口
    - **委托模式**: 将具体实现委托给专门的函数
    - **接口隔离**: 上层调用者无需关心底层实现细节
    - **向后兼容**: 保证API升级时的兼容性
    
    未来可能的扩展:
    
    1. **智能路由**: 根据数据频率自动选择get_high_freq_data或get_low_freq_data
    2. **缓存管理**: 集成数据缓存和失效策略
    3. **并行加载**: 支持多股票数据的并行加载
    4. **数据验证**: 集成数据质量检查和验证机制
    5. **多数据源**: 支持从不同数据源加载数据
    
    性能特性:
    
    - **零开销**: 函数调用的额外开销可忽略不计
    - **直接传递**: 参数直接传递，无中间处理
    - **内存效率**: 与底层函数相同的内存使用模式
    
    使用建议:
    
    - **新项目**: 建议直接使用get_high_freq_data或get_low_freq_data
    - **旧项目**: 可以继续使用这个接口保持兼容性
    - **原型开发**: 适合快速原型验证和小规模测试
    - **生产环境**: 在确定数据类型后使用具体的加载函数
    
    示例用法:
    ```python
    # 简单的数据加载
    df, workdays, errors = get_data(
        start=dt.datetime(2023, 1, 1),
        end=dt.datetime(2023, 12, 31),
        exg="SH",
        full_code="SH000300"
    )
    
    # 指定交易日的加载
    df, workdays, errors = get_data(
        full_code="SH000300",
        workday_list=my_trading_days,
        exg="SH"
    )
    ```
    
    注意事项:
    - 该函数当前仅支持高频数据加载
    - 参数要求和行为与get_high_freq_data完全一致
    - 遇到问题时可以直接查看get_high_freq_data的文档
    - 是一个薄封装，不增加额外的错误处理逻辑
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

