#!/usr/bin/env python3
# This script contains the environment for a stock trading algorithm using deep reinforcement learning.
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

INITIAL_AMOUNT = 1e6

class StockEnvMultiple:
    def __init__(self, tickers=None, begin_date=None, end_date=None, initial_amount=INITIAL_AMOUNT, initial_stocks=None):
        df_raw = pd.read_csv('data/filtered_with_ti.csv', header=0)
        self.spy = pd.read_csv('data/SPY.csv', header=0)

        # Filter dataframe by stocks we want.
        self.df = df_raw[df_raw['tic'].isin(tickers)]

        # Initialize datetime range
        self._initialize_datetime_range()

        self.start_day = self.get_index(begin_date)
        print(self.start_day)
        if self.start_day is None:
            print("Bad start day!")
            return

        self.day = self.start_day
        tics, self.open_price, self.close_price, self.rsi, self.macd, self.obv, self.cci, self.adx = self._get_data(self.day)

        print(f'Using stocks: {tics}')

        self.max_step = self.daterange.shape[0]
        self.max_day = self.get_index(end_date) if self.get_index(end_date) is not None else 10000
        self.max_stock = 1e2

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
        self.total_assets = sum([x*y for x,y in zip(self.close_price, self.stocks)])

        # Technical Indicators
        # RSI, MACD, OBV, CCI, ADX
        self.technical_indicators = [self.rsi, self.macd, self.obv, self.cci, self.adx]

        # Stuff needed for identification
        self.env_name = "RLStockEnv-v2"
        self.state_dim = len(self.reset())
        self.action_dim = len(self.stocks)
        self.target_return = 1e7
        self.if_discrete = False


    """
    Get the datetime range (from AAPL since we know it has the full datetime)
    """
    def _initialize_datetime_range(self):
        aapl = self.df[self.df['tic'] == 'AAPL']
        self.daterange = aapl['Date'].to_numpy()

    """
    Convert date to time index.
    """
    def get_index(self, date):
        # First convert date to pandas date.
        pd_date = pd.to_datetime(date, infer_datetime_format=True)

        # get index in numpy array
        out = np.where(self.daterange == str(pd_date.date()))

        # Return nothing if no index found.
        if len(out) == 0:
            print(f'{str(date)} not found!')
            return None

        return out[0][0]

    """
    Convert time index to date.
    """
    def get_date(self, index):
        if index >= self.daterange.shape[0]:
            return None
        return self.daterange[index]

    """
    Get data and technical indicators given a time index.
    """
    def _get_data(self, index):
        # Get state variables
        df_day = self.df[self.df['Date'] == self.get_date(index)]
        tics = df_day['tic'].tolist()
        open_price = df_day['Open'].tolist()
        close_price = df_day['Close'].tolist()
        rsi = df_day['rsi'].tolist()
        macd = df_day['macd'].tolist()
        obv = df_day['obv'].tolist()
        cci = df_day['cci'].tolist()
        adx = df_day['adx'].tolist()

        return [tics, open_price, close_price, rsi, macd, obv, cci, adx]


    """
    Resets the environment to default.
    """
    def reset(self):
        self.day = self.start_day
        self.balance = self.initial_amount
        self.rewards = []
        self.stocks = self.initial_stocks

        _, _, close_price, rsi, macd, obv, cci, adx = self._get_data(self.day)
        self.total_assets = sum([x*y for x,y in zip(close_price, self.stocks)])
        output = [self.balance, *self.stocks, *close_price, *rsi, *macd, *obv, *cci, *adx]
        return output


    """
    Iterates our current day to the next one.
    """
    def step(self, action):
        self.day += 1
        tic, open_price, close_price, rsi, macd, obv, cci, adx = self._get_data(self.day)

        action = action * self.max_stock  # actions initially is scaled between 0 to 1
        action = (action.astype(int))  # convert into integer because we can't by fraction of shares

        done = False
        if self.day == self.max_day:
            done = True

        for i in range(self.action_dim):
            stock_action = action[i]
            close_price_idx = close_price[i]
            if stock_action > 0:
                # amount of stocks that can be bought with current balance
                available = self.balance // close_price_idx
                change_in_stock = min(available, stock_action)
                self.balance -= close_price_idx * change_in_stock
                self.stocks[i] += change_in_stock
            elif stock_action < 0:
                # amount of stocks that can be sold
                change_in_stock = min(self.stocks[i], -stock_action)
                self.balance += close_price_idx * change_in_stock
                self.stocks[i] -= change_in_stock

        # compute value of total assets and add difference from yesterday's to rewards
        stocks_value = sum([x*y for x,y in zip(close_price, self.stocks)])
        daily_total_assets = stocks_value + self.balance
        daily_reward = daily_total_assets - self.total_assets
        self.rewards.append(daily_reward)

        state = [self.balance, *self.stocks, *close_price, *rsi, *macd, *obv, *cci, *adx]
        return state, daily_reward, done, dict()


    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_or_load_agent(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.start_day, self.max_day):
                s_tensor = _torch.as_tensor((state,), device=device).float()
                a_tensor = act(s_tensor)
                action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                _, _, close, _, _, _, _, _ = self._get_data(self.day)

                total_asset = self.balance + sum([x*y for x,y in zip(close, self.stocks)])
                episode_return = total_asset / self.initial_amount
                episode_returns.append(episode_return)

        spy_data = self.spy['Close'][self.start_day:self.max_day].to_numpy()
        spy_data = spy_data / spy_data[0]

        start_date = self.get_date(self.start_day)
        end_date = self.get_date(self.max_day)

        plt.plot(episode_returns, 'b')
        plt.plot(spy_data, 'r')
        plt.grid()
        plt.title('Cumulative return')
        plt.xlabel(f'Day index (between {start_date} and {end_date})')
        plt.ylabel('multiple of initial_account')
        plt.legend(['RL-PPO', 'SPY'])
        return episode_returns
