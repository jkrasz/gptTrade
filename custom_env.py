import gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(4)  # For 4 discrete actions
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),))  # Modify according to your data
        self.holding = False  # To keep track of whether we are holding a stock or not
        self.buy_price = 0  # To store the price at which we bought the stock

    def step(self, action):
        if self.current_step + 1 >= len(self.data):
            done = True
            reward = self._calculate_final_reward()
            obs = self.data.iloc[self.current_step]  # Return as a Series instead of array
        else:
            self.current_step += 1
            done = False
            reward = self._calculate_reward(action, self.data.iloc[self.current_step])
            obs = self.data.iloc[self.current_step]  # Return as a Series instead of array
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

# Use more sophisticated strategies and risk management
    def _calculate_reward(self, action, obs):
        close_price = obs['Close']
        reward = 0
        if action == 1:
            if not self.holding:
                self.holding = True
                self.buy_price = close_price
                reward = 1  # Encouragement Reward for Buying
            else:
                reward = -1  # Penalty for invalid action
        elif action == 2:
            if self.holding:
                self.holding = False
                profit = close_price - self.buy_price
                reward = profit
            else:
                reward = -1  # Penalty for invalid action
        # Add sophisticated reward computation here
        return reward


    def _calculate_final_reward(self):
        if self.holding:  # Still holding a stock at the end
            final_profit = self.data.iloc[-1]['Close'] - self.buy_price
            return final_profit  # Could be negative
        return 0  # No profit or loss
