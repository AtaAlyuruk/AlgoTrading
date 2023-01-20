import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
from Spot_Futures_Backtesting_EMA import Futures_Backtester as Futures_Backtester


"""
# LOAD THE CLASSES
 
"""
ETH = Futures_Backtester(filepath="ETHUSDT1hr2022.csv", symbol="ETHUSDT", start="2022-01-01", end="2022-11-24", tc=-0.00015)
BTC = Futures_Backtester(filepath="BTCUSDT1hr2022.csv", symbol="BTCUSDT", start="2022-10-01", end="2022-11-24", tc=-0.00015)

# In[run backtest]:

print(BTC.test_strategy(EMAs=(5, 13, 50)))


# In[plotbacktest]:

"""
PLOT BACKTEST RESULTS

"""
BTC.plot_results()
BTC.plot_trades(period = 300)

# In[add leverage]:

BTC.add_leverage(leverage=2)
BTC.plot_results(leverage=True)

# In[add sessions]:

BTC.add_sessions(visualize=True)
BTCresults = BTC.results

# In[run backtest]:

print(ETH.test_strategy(EMAs=(5, 13, 50)))


# In[plotbacktest]:

"""
PLOT BACKTEST RESULTS

"""
ETH.plot_results()
ETH.plot_trades(period = 200)

# In[add leverage]:

ETH.add_leverage(leverage=4)
ETH.plot_results(leverage=True)

# In[add sessions]:

ETH.add_sessions(visualize=True)
ETHresults = ETH.results
