#!/usr/bin/env python3
# This script contains the environment for a stock trading algorithm using deep reinforcement learning.



class StockEnv:
    def __init__(self, tickers=None, begin_date=None, end_date=None):
        # Define our state vector.
        # Our state vector is our current balance plus each
        # (stock amount, open price, close price, technical indicators)
        self.state = []

        # Reward state.
        # A list of rewards. The total sum should be the total reward.
        self.rewards = []

        # List of stocks.
        # In here should be a balance of each stock.
        # For example, say we have 5 stocks of Coke,
        # so this would be [5].
        self.stocks = []

        # Amount of money we have.
        self.balance = 0

        # Stuff needed for identification
        self.env_name = "RLStockEnv-v1"
        self.state_dim = 0
        self.action_dime = 0
        self.target_return = 12345.0

    """
    Resets the environment to default.
    """
    def reset():
        pass

    """
    Iterates our current day to the next one.
    """
    def step():
        self.day += 1

    """
    Function for buying a stock. Part of our action space.
    """
    def _buy_stock(self, stock_index, action):
        pass

    """
    Function to sell a stock. Part of our action space.
    """
    def _sell_stock(self, stock_index, action):
        pass
