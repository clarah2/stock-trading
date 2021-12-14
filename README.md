# stock-trading
A deep reinforcement learning approach to stock allocation.

![alt text](https://github.com/clarah2/stock-trading/blob/main/model_comparison.png)

## Usage
Install the ElegantRL framework. https://github.com/AI4Finance-Foundation/ElegantRL, Python 3.6+, and PyTorch 1.6+, pandas, numpy, and matplotlib (any version) 

## stock_env_multiple.py
This script defines our environment.
#### init
Creates the environment, reads in the data, and initializes attributes needed for state vector.
#### get_index
Converts date to time index.
#### get_date
Converts time index to date.
#### _get_data_dict
Gets data and TIs and stores them into a dict, given a time index.
#### _get_total_assets
Computes our total number of assets.
#### _build_state
Given a dict from _get_data_dict, build a state vector.
#### reset
Resets the environment to default.
#### step
Iterates our current day to the next one.
#### draw_cumulative_return
Plots line graph of agent's returns over evaluation period.

## Stock_Trading_Multiple.ipynb
This currently runs PPO for 40 stocks in our dataset.

#### Cell 7
Environment attributes, agent, and hyperparameters are defined here. They can be adjusted as needed.

**used_tickers**: list of tickers from the S&P 500 used for this model

**initial_stocks**: initial shares of each stock (in used_tickers) that agent has (default is 0)

**initial_capital**: agent's starting balance with which to buy stocks

**start_date, end_date, start_eval_date, end_eval_date**: start and end dates of training and testing periods, respectively

**ppo**: agent is defined here; can be replaced with DDPG or A2C

**args.gamma...args.rollout_num**: hyperparameters

#### Cell 10
Training and evaluating the agent

#### Cell 12
Backtesting and drawing the graph

All other ipynb's are variants of this one for specific models or parameters.

