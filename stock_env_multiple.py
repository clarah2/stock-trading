#!/usr/bin/env python3
# This script contains the environment for a stock trading algorithm using deep reinforcement learning.
import numpy as np
import csv
import pandas as pd

INITIAL_AMOUNT = 1e6

class StockEnvMultiple:
    def __init__(self, tickers=None, begin_date=None, end_date=None, initial_amount=INITIAL_AMOUNT, initial_stocks=None):
        df_raw = pd.read_csv('data/filtered_with_ti.csv', header=0)
        self.spy = pd.read_csv('data/SPY.csv', header=0)
        
        # Filter dataframe by stocks we want.
        self.df = df_raw[df_raw['tic'].isin(tickets)]
        
        # Initialize datetime range
        self._initialize_datetime_range()
        
        self.start_day = 0
        self.day = self.start_day
        self.open_price, self.close_price, self.rsi, self.macd, self.obv, self.cci, self.adx = self._get_data(self.day)

        self.max_step = self.daterange.shape[0]
        self.max_day = self.get_index(end_date) if self.get_index(end_date) is not None else 10000
        
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
        out = np.where(self.daterange == pd_date)
        
        # Return nothing if no index found.
        if len(out) == 0:
            return None
        
        return out[0]
    
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
        stock_idxs = self.df['Date'] == self.get_date(index)
        tics = self.df[stock_idxs]['tic'].tolist()
        open_price = self.df[stock_idxs]['Open'].tolist()
        close_price = self.df[stock_idxs]['Close'].tolist()
        rsi = self.df[stock_idxs]['rsi'].tolist()
        macd = self.df[stock_idxs]['macd'].tolist()
        obv = self.df[stock_idxs]['obv'].tolist()
        cci = self.df[stock_idxs]['cci'].tolist()
        adx = self.df[stock_idxs]['adx'].tolist()
        
        return [tic, open_price, close_price, rsi, macd, obv, cci, adx]
        
    
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
                self.balance += close_price * change_in_stock
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
            for i in range(1200, 1300):
                s_tensor = _torch.as_tensor((state,), device=device).float()
                a_tensor = act(s_tensor)
                action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                _, close, _, _, _, _, _ = self._get_data(self.day)

                total_asset = self.balance + (close * self.stocks[0]).sum()
                episode_return = total_asset / self.initial_amount
                episode_returns.append(episode_return)
                if done:
                    break

        spy_data = self.spy['Close'][1200:1300].to_numpy()
        spy_data = spy_data / spy_data[0]
                    
        import matplotlib.pyplot as plt
        plt.plot(episode_returns, 'b')
        plt.plot(spy_data, 'r')
        plt.grid()
        plt.title('Cumulative return')
        plt.xlabel('Day index (between 10/8/2010 and 3/2/2011)')
        plt.ylabel('multiple of initial_account')
        plt.legend(['RL-PPO', 'SPY'])
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        return episode_returns
