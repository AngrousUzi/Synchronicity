import pandas as pd
import os 
os.makedirs("index\\return",exist_ok=True)
for code in ["H00070","H00071","H00072","H00073","H00074","H00075","H00076","H00077","H00078","H00079","H00006","H00300","H00852","H00906","H00905"]:
    df=pd.read_excel(f"index\{code}.xlsx",header=[0,1,2])
    df.drop(index=df.index[-2:],inplace=True)
    df.columns=["date","price_index_open","price_index_close","return_index_open","return_index_close"]
    df.set_index('date',inplace=True)
    df.index=pd.to_datetime(df.index)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['intra_return']=df['price_index_close']/df['price_index_open']-1
    df['daily_return']=df['return_index_close'].pct_change()
    df.dropna(inplace=True)
    df['overnight_return']=(df['daily_return']+1)/(df['intra_return']+1)-1
    # df.drop(columns=['price_index_open','price_index_close','return_index_open'],inplace=True)

    df[["intra_return","daily_return","overnight_return"]].to_csv(f"index\\return\\{code}.csv")
    # df.to_csv(f"index\return\{code}.csv")