#!/usr/bin/env python3
# This script contains the environment for a stock trading algorithm using deep reinforcement learning.
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

INITIAL_AMOUNT = 1e6

class StockEnvMultiple:
    def __init__(self, tickers={}, begin_date=None, end_date=None, initial_amount=INITIAL_AMOUNT, reward_scaling=1e-6):
        df_raw = pd.read_csv('data/final_filtered.csv', header=0)
        self.spy = pd.read_csv('data/SPY.csv', header=0)
        
        # Filter dataframe by stocks we want.
        self.df = df_raw[df_raw['tic'].isin(tickers)]

        self.daterange = df_raw.loc[df_raw['tic'] == 'AAPL', 'Date'].to_numpy()
        self.env_num = 1
        
        self.start_day = self.get_index(begin_date)
        if self.start_day is None:
            print("Bad start day!")
            return

        self.day = self.start_day
        init_data = self._get_data_dict(self.start_day)

        self.max_step = self.daterange.shape[0]
        self.max_day = self.get_index(end_date) if self.get_index(end_date) is not None else 10000
        self.max_stock = 100

        self.initial_amount = initial_amount
        # Amount of money we have.
        self.balance = initial_amount
        # Technical Indicators
        # RSI, MACD, OBV, CCI, ADX
        self.state_vars = ['Open', 'Close', 'rsi', 'macd', 'obv', 'cci', 'adx']
        

        # Reward state.
        # A list of rewards. The total sum should be the total reward.
        self.rewards = []
        self.reward_scaling = reward_scaling

        # List of stocks.
        # In here should be a balance of each stock.
        # For example, say we have 5 stocks of Coke,
        # so this would be [5].
        self.initial_stocks = tickers
        self.stocks = tickers
        self.stock_list = list(self.stocks.keys())
        
        print(f'Using stocks: {self.stock_list}')
   
        # Total assets
        self.total_assets = self._get_total_assets(init_data)
        self.state = self.reset()


        # Stuff needed for identification
        self.env_name = "RLStockEnv-v3"
        self.state_dim = len(self.reset())
        self.action_dim = len(self.stocks)
        self.if_discrete = False



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
    Get data and TIs and store them into a dict, given a time index.
    """
    def _get_data_dict(self, index):
        df_day = self.df[self.df['Date'] == self.get_date(index)]
        return df_day.set_index('tic').to_dict('index')

    """
    Compute our total number of assets.
    """
    def _get_total_assets(self, data):
        out = self.balance
        for tic in self.stocks:
            if tic in data:
                
                if np.isnan(data[tic]['Close']):
                    continue
                out += data[tic]['Close'] * self.stocks[tic]
        
        return out
    
    """
    Given a dict from _get_data_dict, build a state vector.
    """
    def _build_state(self, data):
        state = [self.balance]
        for tic in self.stock_list:
            state.append(self.stocks[tic])

        
        # Add stocks.
        for var in self.state_vars:
            for tic in self.stock_list:
                if tic not in data:
                    state.append(0)
                else:
                    if np.isnan(data[tic][var]):
                        state.append(0)
                    else:
                        state.append(data[tic][var])
                                 
        return state
        
    
    """
    Resets the environment to default.
    """
    def reset(self):
        self.day = self.start_day
        self.balance = self.initial_amount
        self.rewards = []
        self.stocks = self.initial_stocks

        day_data = self._get_data_dict(self.day)
        self.total_assets = self._get_total_assets(day_data)
        output = np.array(self._build_state(day_data))
        return output


    """
    Iterates our current day to the next one.
    """
    def step(self, action):
        # print('a', action)
        self.day += 1
        day_data = self._get_data_dict(self.day)
        
        action = action * self.max_stock  # actions initially is scaled between 0 to 1
        action = np.floor(action.astype(int))  # convert into integer because we can't by fraction of shares

        done = False
        if self.day == self.max_day:
            done = True

        for i in range(self.action_dim):
            stock_action = action[i]
            ticker = self.stock_list[i]
            if ticker not in day_data:
                continue
            close_price_idx = day_data[ticker]['Close']
            if np.isnan(close_price_idx):
                continue
            self.stocks[ticker] *= np.round(day_data[ticker]['multiplier'])
            if stock_action > 0:
                # amount of stocks that can be bought with current balance
                available = self.balance // close_price_idx
                change_in_stock = min(available, stock_action)
                self.balance -= close_price_idx * change_in_stock
                self.stocks[ticker] += change_in_stock
            elif stock_action < 0:
                # amount of stocks that can be sold
                change_in_stock = min(self.stocks[ticker], -stock_action)
                self.balance += close_price_idx * change_in_stock
                self.stocks[ticker] -= change_in_stock

        # compute value of total assets and add difference from yesterday's to rewards
        daily_total_assets = self._get_total_assets(day_data)
        daily_reward = self.reward_scaling * (daily_total_assets - self.total_assets)
        self.rewards.append(daily_reward)
        
        # Update assets
        self.total_assets = daily_total_assets
        
        state = np.array(self._build_state(day_data))
        return state, daily_reward, done, dict()


    def draw_cumulative_return(self, args, _torch, model_type='', color='b') -> list:
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
        print(state)
        episode_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.start_day, self.max_day):
                s_tensor = _torch.as_tensor((state,), device=device).float()
                a_tensor = act(s_tensor)
                action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                day_data = self._get_data_dict(self.day)

                total_asset = self._get_total_assets(day_data)
                episode_return = total_asset / self.initial_amount
                episode_returns.append(episode_return)


        
        spy_data = self.spy['Close'][self.start_day:self.max_day].to_numpy()
        spy_data = spy_data / spy_data[0]

        start_date = self.get_date(self.start_day)
        end_date = self.get_date(self.max_day)

        plt.plot(episode_returns, color)
        plt.plot(spy_data, 'black')
        plt.grid()
        plt.title('Cumulative return')
        plt.xlabel(f'Day index (between {start_date} and {end_date})')
        plt.ylabel('multiple of initial_account')
        plt.legend(['RL-'+model_type, 'SPY'])
        return episode_returns, spy_data
