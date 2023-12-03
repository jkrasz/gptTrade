
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
        self.leverage = 2  # Increased leverage for more aggressive strategy

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

        # Modified reward function to be more aggressive
        if action == 1:  # Buy
            if not self.holding:
                self.holding = True
                self.buy_price = close_price
                reward = 5  # Increased reward for buying
            else:
                reward = -10  # Increased penalty for invalid buy action

        elif action == 2:  # Sell
            if self.holding:
                self.holding = False
                profit = (close_price - self.buy_price) * self.leverage
                reward = profit
                if profit > 0:
                    reward += 10  # Bonus reward for making a profitable sell
            else:
                reward = -10  # Increased penalty for invalid sell action

        elif action == 0:  # Hold
            if self.holding:
                reward = (close_price - self.buy_price) * self.leverage  # Leverage applied to holding performance

        # Reward shaping based on indicators
        rsi = obs['RSI']
        macd_diff = obs['MACD_Line'] - obs['Signal_Line']

        # Positive reward if RSI suggests oversold and we buy, or overbought and we sell.
        if (rsi < 30 and action == 1) or (rsi > 70 and action == 2):
            reward += 5

        # Positive reward for buying when MACD is above signal line and vice-versa
        if (macd_diff > 0 and action == 1) or (macd_diff < 0 and action == 2):
            reward += 5

        return reward

    def _calculate_final_reward(self):
        if self.holding:
            final_profit = self.data.iloc[-1]['Close'] - self.buy_price
            return final_profit * self.leverage
        return 0
