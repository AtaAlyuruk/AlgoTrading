# # Long_Short_Backtester class: For backtesting spot trading
# # Futures_BackTester: Backtesting perpetual futures with Leverage
# # Assumption of one price between futures and spot market for convenience
# # Trading Strategy: __Triple EMA Crossover__ for illustrative purposes, 
# # more complex strategies will be implemented

# ## The underlying historical data is hourly data of ethereum, 
# ## will be changed with higher frequency data

# In[ ]:


import warnings
from itertools import product

import matplotlib.pyplot as plt
# from binance.client import Client
import numpy as np
import pandas as pd
import scipy.stats as sps

from src.sharpe_ratio_stats import probabilistic_sharpe_ratio

warnings.filterwarnings("ignore")
plt.style.use("seaborn")

class Futures_Backtester():
    """ Class for the vectorized backtesting of (levered) Futures trading strategies.

    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade


    Methods
    =======
    get_data:
        imports the data.

    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).

    prepare_data:
        prepares the data for backtesting.

    run_backtest:
        runs the strategy backtest.

    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.

    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper).

    find_best_strategy:
        finds the optimal strategy (global maximum).

    add_sessions:
        adds/labels trading sessions and their compound returns.

    add_leverage:
        adds leverage to the strategy.

    print_performance:
        calculates and prints various performance metrics.

    """

    def __init__(self, filepath, symbol, start, end, tc):

        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def __repr__(self):
        return "Futures_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        """ Imports the data.
        """
        raw = pd.read_csv(self.filepath, parse_dates = ["Date"], index_col = "Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        #raw["normreturns"] = raw.Close / raw.Close.shift(1)
        self.data = raw
        
    def test_strategy(self, EMAs):
        """
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        EMAs: tuple (EMA_S, EMA_M, EMA_L)
            Exponential Moving Averages to be considered for the strategy.

        """
        
        self.EMA_S = EMAs[0]
        self.EMA_M = EMAs[1]
        self.EMA_L = EMAs[2]
        
        
        self.prepare_data(EMAs = EMAs)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()
    
    def prepare_data(self, EMAs):
        """ Prepares the Data for Backtesting.
        """
        ########################## Strategy-Specific #############################
        
        data = self.data.copy()
        data["EMA_S"] = data.Close.ewm(span = EMAs[0], min_periods = EMAs[0]).mean()
        data["EMA_M"] = data.Close.ewm(span = EMAs[1], min_periods = EMAs[1]).mean()
        data["EMA_L"] = data.Close.ewm(span = EMAs[2], min_periods = EMAs[2]).mean()
        
        
        ########################## Create Signal ###############################
        data["EMA_Signal"] = 0
        signal = 0
        for i in range(len(data)):
            if (data["EMA_S"][i-1] < data["EMA_M"][i-1]) & (data["EMA_S"][i] > data["EMA_M"][i]) & (data["Close"][i] > data["EMA_L"][i]):
                if signal != 1:
                    signal = 1
                    data["EMA_Signal"][i] = signal
                else:
                    data["EMA_Signal"][i] = 0
            elif (data["EMA_S"][i-1] > data["EMA_M"][i-1]) & (data["EMA_S"][i] < data["EMA_M"][i]) & (data["Close"][i] < data["EMA_L"][i]):
                if signal != -1:
                    signal = -1
                    data["EMA_Signal"][i] = signal
                else:
                    data["EMA_Signal"][i] = 0
            elif (data["EMA_S"][i-1] > data["EMA_M"][i-1]) & (data["EMA_S"][i] < data["EMA_M"][i]) & (data["Close"][i] > data["EMA_L"][i]):
                if signal != 2:
                    signal = 2
                    data["EMA_Signal"][i] = signal
                else:
                    data["EMA_Signal"][i] = 0
            elif (data["EMA_S"][i-1] < data["EMA_M"][i-1]) & (data["EMA_S"][i] > data["EMA_M"][i]) & (data["Close"][i] < data["EMA_L"][i]):
                if signal != 3:
                    signal = 3
                    data["EMA_Signal"][i] = signal
                else:
                    data["EMA_Signal"][i] = 0
            else:
                data["EMA_Signal"][i] = 0
  
        ######################### Create Position ############################
        data["position"] = 0
        
        for i in range(len(data["Close"])):
            if data["EMA_Signal"][i] == 1:
                data["position"][i] = 1
            elif data["EMA_Signal"][i] == -1:
                data["position"][i] = -1
            elif data["EMA_Signal"][i] == 2:
                data["position"][i] = 0
            elif data["EMA_Signal"][i] == 3:
                data["position"][i] = 0
            else:
                data["position"][i] = data["position"][i-1]
        ######################################################################
        self.results = data
    
    def run_backtest(self):
        """ Runs the strategy backtest.
        """
        
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
        
        self.results = data
    
    def plot_results(self, leverage = False): #Adj!
        """  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        """
        if self.results is None:
            print("Run test_strategy() first.")
        elif leverage: # NEW!
            title = "{} | TC = {} | Leverage = {}".format(self.symbol, self.tc, self.leverage)
            self.results[["creturns", "cstrategy", "cstrategy_levered"]].plot(title=title, figsize=(12, 8))
            plt.show()
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.show()

    def plot_trades(self, period):
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            plt.rcParams['figure.figsize'] = 12, 6
            plt.grid(True, alpha = 0.3)
            plt.plot(self.results.iloc[-period:]['Close'], label = 'Price')
            plt.plot(self.results.iloc[-period:]['EMA_S'], label = 'EMA_S')
            plt.plot(self.results.iloc[-period:]['EMA_M'], label = 'EMA_M')
            plt.plot(self.results.iloc[-period:]['EMA_L'], label = 'EMA_L')
            plt.plot(self.results[-period:].loc[self.results.EMA_Signal == 1].index,
                     self.results[-period:]['EMA_S'][self.results.EMA_Signal == 1], '^',
                     color = 'g', markersize = 12)
            plt.plot(self.results[-period:].loc[self.results.EMA_Signal == -1].index,
                     self.results[-period:]['EMA_L'][self.results.EMA_Signal == -1], 'v',
                     color='r', markersize=12)
            plt.plot(self.results[-period:].loc[self.results.EMA_Signal == 2].index,
                     self.results[-period:]['EMA_L'][self.results.EMA_Signal == 2], 'v',
                     color='b', markersize=12)
            plt.plot(self.results[-period:].loc[self.results.EMA_Signal == 3].index,
                     self.results[-period:]['EMA_S'][self.results.EMA_Signal == 3], '^',
                     color='b', markersize=12)
            plt.legend(loc=2)
            plt.show()


            
    def optimize_strategy(self, EMA_S_range, EMA_M_range, EMA_L_range, metric = "Multiple"):
        """
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        EMA_S_range: tuple
            tuples of the form (start, end, step size).

        EMA_M_range: tuple
            tuples of the form (start, end, step size).

        EMA_L_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be "Multiple" or "Sharpe")
        """
        
        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        
        EMA_S_range = range(*EMA_S_range)
        EMA_M_range = range(*EMA_M_range)
        EMA_L_range = range(*EMA_L_range)
        
        combinations = list(product(EMA_S_range, EMA_M_range, EMA_L_range))
         
        performance = []
        for comb in combinations:
            self.prepare_data(EMAs = comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview =  pd.DataFrame(data = np.array(combinations), columns = ["EMA_S", "EMA_M", "EMA_L"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
        """ Finds the optimal strategy (global maximum).
        """
        
        best = self.results_overview.nlargest(1, "performance")
        EMA_S = best.EMA_S.iloc[0]
        EMA_M = best.EMA_M.iloc[0]
        EMA_L = best.EMA_L.iloc[0]
        perf = best.performance.iloc[0]
        print("EMA_S: {} | EMA_M: {} | EMA_L : {} | {}: {}".format(EMA_S, EMA_M, EMA_L, self.metric, round(perf, 5)))  
        self.test_strategy(EMAs = (EMA_S, EMA_M, EMA_L))
        
    
    def add_sessions(self, visualize = False):
        """
        Adds/Labels Trading Sessions and their compound returns.

        Parameter
        ============
        visualize: bool, default False
            if True, visualize compound session returns over time
        """
        
        if self.results is None:
            print("Run test_strategy() first.")
            
        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data
        if visualize:
            data["session_compound"].plot(figsize = (12, 8))
            plt.show()  
        
    def add_leverage(self, leverage, report = True):
        """
        Adds Leverage to the Strategy.

        Parameter
        ============
        leverage: float (positive)
            degree of leverage.

        report: bool, default True
            if True, print Performance Report incl. Leverage.
        """
        self.add_sessions()
        self.leverage = leverage
        
        data = self.results.copy()
        data["simple_ret"] = np.exp(data.strategy) - 1
        data["eff_lev"] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data.eff_lev.fillna(leverage, inplace = True)
        data.loc[data.trades !=0, "eff_lev"] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data["strategy_levered"] = levered_returns
        data["cstrategy_levered"] = data.strategy_levered.add(1).cumprod()
        
        self.results = data
            
        if report:
            self.print_performance(leverage = True)
            
    ############################## Performance ######################################
    
    def print_performance(self, leverage = False): # Adj
        """ Calculates and prints various Performance Metrics.
        """
        
        data = self.results.copy()
        
        if leverage: # NEW!
            to_analyze = np.log(data.strategy_levered.add(1))
        else: 
            to_analyze = data.strategy
            to_analyze.dropna(inplace = True)
            
            
        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        ann_mean =          round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std =           round(self.calculate_annualized_std(to_analyze), 6)
        skew =              round(self.calculate_skewness(to_analyze), 6)
        kurtosis =          round(self.calculate_kurtosis(to_analyze), 6)
        sharpe =            round(self.calculate_sharpe(to_analyze), 6)
        psr =               round(self.calculate_psr(to_analyze) , 6)
       
        print(100 * "=")
        print("TRIPLE EMA STRATEGY | INSTRUMENT = {} | EMAs = {}".format(self.symbol, [self.EMA_S, self.EMA_M, self.EMA_L]))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Skewness:                    {}".format(skew))
        print("Kurtosis:                    {}".format(kurtosis))
        print("Sharpe Ratio:                {}".format(sharpe))
        print("Probabilistic Sharpe Ratio:  {}".format(psr))
        
        print(100 * "=")
        
    def calculate_skewness(self, series):
        return sps.skew(series)
    
    def calculate_kurtosis(self, series):
        return sps.kurtosis(series)
    
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_annualized_mean(series) / self.calculate_annualized_std(series)
        
    def calculate_psr(self, series):
        return probabilistic_sharpe_ratio(series, sr_benchmark=0)

    