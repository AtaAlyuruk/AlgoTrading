# In[1]:


from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
pd.options.mode.chained_assignment = None

# In[2]:


class FuturesTrader():
    
    def __init__(self, symbol, bar_length, ema_s, ema_m, ema_l, units, position = 0, leverage = 4):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.leverage = leverage
        self.cum_profits = 0
        #self.trades = 0 
        #self.trade_values = []
        
        #*****************add strategy-specific attributes here******************
        self.EMA_S = ema_s
        self.EMA_M = ema_m
        self.EMA_L = ema_l
        #************************************************************************
    
    def start_trading(self, historical_days):
        
        client.futures_change_leverage(symbol = self.symbol, leverage = self.leverage)
        
        self.twm = ThreadedWebsocketManager(testnet = False) # testnet
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
            self.twm.start_kline_futures_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
       
    
    def get_most_recent(self, symbol, interval, days):
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))
    
        bars = client.futures_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000)
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
    
    def stream_candles(self, msg):
        
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        # print out
        print(".", end = "", flush = True) 
    
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades()
        
    def define_strategy(self):
        
        
        ########################## Strategy-Specific #############################
        
        data = self.data.copy()
        data["EMA_S"] = data.Close.ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()
        data["EMA_M"] = data.Close.ewm(span = self.EMA_M, min_periods = self.EMA_M).mean()
        data["EMA_L"] = data.Close.ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
        
        
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
        
        self.prepared_data = data.copy()
    
    def execute_trades(self):
    
        # GOING LONG
        
        if self.prepared_data["EMA_Signal"].iloc[-1] == 1: 
            
            if self.position == 0:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "BUY",
                    type = "MARKET",
                    quantity = self.units
                    )
                
                self.report_trade(order, "GOING LONG")
                
            elif self.position == -1:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "BUY",
                    type = "MARKET",
                    quantity = 2 * self.units
                    )
                
                self.report_trade(order, "GOING LONG")
                
            self.position = 1
            
        # GOING SHORT
        
        elif self.prepared_data["EMA_Signal"].iloc[-1] == -1: 
            
            if self.position == 0:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "SELL",
                    type = "MARKET",
                    quantity = self.units
                    )
            
                self.report_trade(order, "GOING SHORT") 
                
            elif self.position == 1:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "SELL",
                    type = "MARKET",
                    quantity = 2 * self.units
                    )
            
                self.report_trade(order, "GOING SHORT")
                
            self.position = -1
            
        # GOING NEUTRAL FROM LONG  
        
        elif self.prepared_data["EMA_Signal"].iloc[-1] == 2: 
            
            if self.position == 1:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "SELL",
                    type = "MARKET",
                    quantity = self.units
                    )
                
                self.report_trade(order, "GOING NEUTRAL")
            
            self.position = 0
            
        # GOIN NEUTRAL FROM SHORT
        
        elif self.prepared_data["EMA_Signal"].iloc[-1] == 3: 
            
            if self.position == -1:
                
                order = client.futures_create_order(
                    symbol = self.symbol,
                    side = "BUY",
                    type = "MARKET",
                    quantity = self.units
                    )
                
                self.report_trade(order, "GOING NEUTRAL")
            
            self.position = 0
            
        """
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == -1:
                order = client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
        if self.prepared_data["position"].iloc[-1] == -1: # if position is short -> go/stay short
            if self.position == 0:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
            elif self.position == 1:
                order = client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        """
    def report_trade(self, order, going): 
        
        time.sleep(0.1)
        order_time = order["updateTime"]
        trades = client.futures_account_trades(symbol = self.symbol, startTime = order_time)
        order_time = pd.to_datetime(order_time, unit = "ms")
        
        # extract data from trades object
        df = pd.DataFrame(trades)
        columns = ["qty", "quoteQty", "commission","realizedPnl"]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        base_units = round(df.qty.sum(), 5)
        quote_units = round(df.quoteQty.sum(), 5)
        commission = -round(df.commission.sum(), 5)
        real_profit = round(df.realizedPnl.sum(), 5)
        price = round(quote_units / base_units, 5)
        
        # calculate cumulative trading profits
        self.cum_profits += round((commission + real_profit), 5)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(order_time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(order_time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(order_time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")


# In[3]:


api_key = ""
secret_key = ""


# Futures API document:

# https://binance-docs.github.io/apidocs/futures/en/#change-log

client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = False) # Testnet!!!

# print(client.get_my_trades(symbol = "ETHUSDT"))
# In[4]:


symbol = "ETHUSDT"
bar_length = "1h"
ema_s = 5
ema_m = 13
ema_l = 50
units = 45
position = 0
leverage = 4


# In[5]:


trader = FuturesTrader(symbol = symbol, bar_length = bar_length,
                       ema_s = ema_s, ema_m = ema_m, ema_l = ema_l, 
                       units = units, position = position, leverage = leverage)


# In[6]:


trader.start_trading(historical_days = 2)


# In[7]:


trader.twm.stop()


# In[8]:


report = trader.prepared_data

