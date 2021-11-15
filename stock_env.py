#!/usr/bin/env python3
# This script contains the environment for a stock trading algorithm using deep reinforcement learning.
import numpy as np
import csv
import pandas as pd
import talib

INITIAL_AMOUNT = 1000

class StockEnv:
    def __init__(self, tickers=None, begin_date=None, end_date=None, initial_amount=INITIAL_AMOUNT, initial_stocks=[8]):
        self.df = pd.read_csv('data/coke.csv', header=0)
        self.start_day = 35
        self.day = self.start_day
        self.open_price, self.close_price, self.rsi, self.macd, self.obv, self.cci, self.adx = self._get_data(self.day)

        self.max_step = self._get_length_of_data()
        
        self.max_day = 1000
        
        self.initial_amount = initial_amount
        # Amount of money we have.
        self.balance = initial_amount

        # Reward state.
        # A list of rewards. The total sum should be the total reward.
        self.rewards = []

        # List of stocks.
        # In here should be a balance of each stock.
        # For example, say we have 5 stocks of Coke,
        # so this would be [5].
        self.initial_stocks = initial_stocks
        self.stocks = initial_stocks

        # Total assets
        self.total_assets = sum([x*y for x,y in zip([self.close_price], self.stocks)])

        # Technical Indicators
        # RSI, MACD, OBV, CCI, ADX
        self.technical_indicators = [self.rsi, self.macd, self.obv, self.cci, self.adx]

        # Stuff needed for identification
        self.env_name = "RLStockEnv-v1"
        self.state_dim = len(self.reset())
        self.action_dim = 1
        self.target_return = 12345.0
        self.if_discrete = False

    """
    Resets the environment to default.
    """
    def reset(self):
        self.day = self.start_day
        self.balance = self.initial_amount
        self.rewards = []
        self.stocks = self.initial_stocks
        self.open_price, self.close_price, self.rsi, self.macd, self.obv, self.cci, self.adx = self._get_data(self.day)
        self.total_assets = sum([x*y for x,y in zip([self.close_price], self.stocks)])
        self.technical_indicators = [self.rsi, self.macd, self.obv, self.cci, self.adx]
        output = [self.balance, self.stocks[0], self.open_price, self.close_price]
        for item in self.technical_indicators:
            output.append(item)

        return output


    """
    Iterates our current day to the next one.
    """
    def step(self, action):
        self.day += 1
        self.open_price, self.close_price, self.rsi, self.macd, self.obv, self.cci, self.adx = self._get_data(self.day)
        self.technical_indicators = [self.rsi, self.macd, self.obv, self.cci, self.adx]
        done = False
        if self.day == self.max_day:
            done = True
        
        for i in range(self.action_dim):
            stock_action = action[i]
            close_price = self.close_price
            if stock_action > 0:
                #amount of stocks that can be bought with current balance
                available = self.balance // close_price
                change_in_stock = min(available, stock_action)
                self.balance -= close_price * change_in_stock
                self.stocks[i] += change_in_stock
            elif stock_action < 0:
                #amount of stocks that can be sold
                change_in_stock = min(self.stocks[i], -stock_action)
                self.balance += close_price * change_in_stock
                self.stocks[i] -= change_in_stock
        # compute value of total assets and add difference from yesterday's to rewards
        stocks_value = sum([x*y for x,y in zip([self.close_price], self.stocks)])
        daily_total_assets = stocks_value + self.balance
        daily_reward = daily_total_assets - self.total_assets
        self.rewards.append(daily_reward)

        state = [self.balance, self.stocks[0], self.open_price, self.close_price]
        for item in self.technical_indicators:
            state.append(item)
        return state, daily_reward, done, dict()

    """
    Function for buying a stock. Part of our action space.
    """
    def _buy_stock(self, stock_index, action):
        close_price = self.state[3]
        if close_price > 0:
            available = self.balance // close_price
            num_shares_bought = min(available, action)
            amount_bought = close_price * num_shares_bought
            self.balance -= amount_bought
            self.stocks[stock_index] += num_shares_bought
        else:
            num_shares_bought = 0
        return num_shares_bought

    """
    Function to sell a stock. Part of our action space.
    """
    def _sell_stock(self, stock_index, action):
        close_price = self.state[3]
        if close_price > 0:
            #check that agent has some of that stock
            if self.stocks[stock_index] > 0:
                num_shares_sold = min(self.stocks[stock_index], abs(action))
                amount_sold = close_price * num_shares_sold
                self.balance += amount_sold
                self.stocks[stock_index] -= num_shares_sold
            else:
                num_shares_sold = 0
        else:
            num_shares_sold = 0
        return num_shares_sold

    """
    Function to compute and add technical indicators to dataframe.
    """
    def _compute_technical_indicators(self, df):
        df['rsi'] = talib.RSI(df['prcod'])
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['prccd']) # defaults: fastperiod=12, slowperiod=26, signalperiod=9
        df['obv'] = talib.OBV(df['prccd'], df['cshtrd'])
        df['cci'] = talib.CCI(df['prchd'], df['prcld'], df['prccd']) #default period: 14 days
        df['adx'] = talib.ADX(df['prchd'], df['prcld'], df['prccd']) #default period: 14 days

    """
    Function to get the open price, close price, and technical indicators for given day.
    """
    def _get_data(self, today=35):
        self._compute_technical_indicators(self.df)

        opening = self.df['prcod']
        closing = self.df['prccd']
        rsi, macd, obv, cci, adx = self.df['rsi'], self.df['macd'], self.df['obv'], self.df['cci'], self.df['adx']
        
        out = []
        today_open = opening[today]
        today_close = closing[today]
        today_rsi = rsi[today]
        today_macd = macd[today]
        today_obv = obv[today]
        today_cci = cci[today]
        today_adx = adx[today]

        return today_open, today_close, today_rsi, today_macd, today_obv, today_cci, today_adx
   
    def _get_length_of_data(self):
        return self.df['prccd'].shape[0] - self.start_day
    
    def _get_all_days(self):
        return self.df['prccd'].shape[0]
