import pandas as pd
import os
import datetime as dt
from all_cal import get_composites
import multiprocessing as mp
from functools import partial
from utils import generate_single_txt



def generate_txt(code,base_path):
    code_list=get_composites(code)["full_code"].tolist()

    index_txt_path=os.path.join(base_path,f"{code}.txt")

    workday_list=[]
    with open(index_txt_path,"r") as f:
        lines=f.readlines()
        for line in lines:
            if line.startswith("workday_list:"):
                import datetime
                workday_list=eval(line.split(":",1)[1].strip())

    process_func = partial(generate_single_txt, base_path=base_path, workday_list=workday_list)
    with mp.Pool(processes=60) as pool:
        pool.map(process_func, code_list)

def generate_index_text(code,base_path):
    csv_path=os.path.join(base_path,f"{code}.csv")
    # if os.path.exists(csv_path):
    df_stock=pd.read_csv(csv_path,index_col=0,parse_dates=True)
    workday_list_stock = [d.date() for d in df_stock.index.normalize().tolist()]   
    workday_list_unique=list(set(workday_list_stock))
    workday_list_unique.sort()
    # error_list=list(set(workday_list)-set(workday_list_stock))
    txt_path=os.path.join(base_path,f"{code}.txt")
    with open(txt_path,"w") as f:
        f.write(f"workday_list: {workday_list_unique}\n")
        f.write(f"error_list: []")
    print(code)

if __name__=="__main__":
    start_date=dt.datetime(2010,1,5)
    end_date=dt.datetime(2025,6,30)
    date_str=f"{start_date.date()}_{end_date.date()}"
    base_path=os.path.join(date_str,"agg_raw")
    # txt_path=os.join(date_str,"agg_raw")
    # generate_txt("SH000300",base_path)

    generate_index_text("SH000852",base_path)


