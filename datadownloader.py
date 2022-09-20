# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:08:48 2022

@author: ATA
"""

# In[ ]:


import pandas as pd
from binance.client import Client
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn")


  
# In[ ]:
    
api_key = "dKeLRrvQJ6bvlCcFoht0XIKIr9KXhqYOrmitejCYq7Ulfii1IqzPKwtpexaVmr8x"
secret_key = "aw7c2FRr0p1VRoSnUBRjY3zgjqtdigmFJ5l6C7X3YiUefefTtJAJnZDhAjijQkIf"
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
    
df = get_history(symbol = "ETHUSDT", interval = "1h", start = "2017-09-01", end = "2022-08-01")
df.to_csv(r'C:\Users\ATA\Desktop\Python\BacktesterTrader\ethusdt1h.csv')




