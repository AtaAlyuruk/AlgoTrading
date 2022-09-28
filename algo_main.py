import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
from Spot_Futures_Backtesting_EMA import Futures_Backtester as Futures_Backtester


"""
# LOAD THE CLASSES
 
"""
ETH = Futures_Backtester(filepath="ethusdt1h.csv", symbol="ETHUSDT", start="2021-11-01", end="2022-09-01", tc=-0.0005)
BTC = Futures_Backtester(filepath="btcusdt1h.csv", symbol="BTCUSDT", start="2021-11-01", end="2022-09-01", tc=-0.0005)

# In[run backtest]:

print(ETH.test_strategy(EMAs=(5, 13, 50)))


# In[plotbacktest]:

"""
PLOT BACKTEST RESULTS

"""
ETH.plot_results()
ETH.plot_trades(period = 252)

# In[add leverage]:

ETH.add_leverage(leverage=2)
ETH.plot_results(leverage=True)

# In[add sessions]:

ETH.add_sessions(visualize=True)
ETHresults = ETH.results
