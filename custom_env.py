import gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(4)  # For 4 discrete actions
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),))
        self.holding = False
        self.buy_price = 0

    def step(self, action):
        if self.current_step + 1 >= len(self.data):
            done = True
            reward = self._calculate_final_reward()
            obs = self.data.iloc[self.current_step]
        else:
            self.current_step += 1
            done = False
            reward = self._calculate_reward(action, self.data.iloc[self.current_step])
            obs = self.data.iloc[self.current_step]

        info = {"action": "Buy" if action == 1 else "Sell", "holding": self.holding, "buy_price": self.buy_price}
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.buy_price = 0
        return self.data.iloc[self.current_step].values

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
        # Modify reward computation based on additional strategies
        return reward

    def _calculate_final_reward(self):
        if self.holding:
            final_profit = self.data.iloc[-1]['Close'] - self.buy_price
            return final_profit
        return 0
