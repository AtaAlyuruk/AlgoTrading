import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
from Spot_Futures_Backtesting_EMA import Futures_Backtester as Futures_Backtester


"""
# LOAD THE CLASSES
 
"""
ETH = Futures_Backtester(filepath="ethusdt1h.csv", symbol="ETHUSDT", start="2021-11-01", end="2022-09-01", tc=-0.0005)
BTC = Futures_Backtester(filepath="btcusdt3oct.csv", symbol="BTCUSDT", start="2022-10-03", end="2022-10-03", tc=-0.0005)

# In[run backtest]:

print(BTC.test_strategy(EMAs=(5, 13, 50)))


# In[plotbacktest]:

"""
PLOT BACKTEST RESULTS

"""
BTC.plot_results()
BTC.plot_trades(period = 100)

# In[add leverage]:

ETH.add_leverage(leverage=2)
ETH.plot_results(leverage=True)

# In[add sessions]:

ETH.add_sessions(visualize=True)
ETHresults = ETH.results
