# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:56:06 2022

@author: Kullanıcı
"""

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn")
from Spot_Futures_Backtesting_EMA import Futures_Backtester as Futures_Backtester

# In[1]:

symbol = "ETHUSDT"
start  = "2022-05-01"
end    = "2022-08-01"
tc     = -0.0005

# In[backtest]:
"""
# LOAD THE CLASSES
 
"""
EMAeth1hr    = Futures_Backtester(filepath = "ethusdt1h.csv", symbol = symbol, start = start, end = end, tc = tc)

# In[run backtest]:

print(EMAeth1hr.test_strategy(EMAs = (5, 13, 50)))
results = EMAeth1hr.results

# In[plotbacktest]:
    
"""
PLOT BACKTEST RESULTS

"""
EMAeth1hr.plot_results()

# In[add leverage]:
    
EMAeth1hr.add_leverage(leverage = 4)
EMAeth1hr.plot_results(leverage = True)

# In[add sessions]:

EMAeth1hr.add_sessions(visualize=True)