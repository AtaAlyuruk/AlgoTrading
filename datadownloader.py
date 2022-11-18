# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:08:48 2022

@author: ATA
"""

# In[ ]:


import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn")


  
# In[ ]:
    
api_key = ""
secret_key = ""

client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = False)

# In[ ]:
    
def get_history(symbol, interval, start, end = None):
    bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                        start_str = start, end_str = end, limit = 1000)
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                  "Clos Time", "Quote Asset Volume", "Number of Trades",
                  "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = "coerce")
    
    return df
# In[ ]:
    
df = get_history(symbol = "ETHUSDT", interval = "1h", start = "2022-01-01", end = "2022-11-18")
df.to_csv(r'C:\Users\ATA\Desktop\Python\AlgoTrading\datasets\ETHUSDT1hr2022.csv')




