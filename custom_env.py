import numpy as np
import gym
from gym import spaces

class CustomTradingEnvironment(gym.Env):

    def __init__(self, data, features, targets):
        super(CustomTradingEnvironment, self).__init__()

        self.data = data
        self.features = features
        self.targets = targets
        self.current_step = 0
        self.current_position = None  # 0: No position, 1: Long, 2: Short
        self.current_price = self._get_current_price()

        self.action_space = spaces.Discrete(4)  # 0: Hold, 1: Buy (Go long), 2: Sell (Go short), 3: Close position

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3*self.features.shape[1] + 2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.current_position = 0

        obs = self.features.iloc[self.current_step].values
        sma_5 = np.zeros(self.features.shape[1])
        sma_10 = np.zeros(self.features.shape[1])

        self.state = np.concatenate([obs, sma_5, sma_10, [self.current_position], [self.current_price]])

        return self.state.astype(np.float32)

    def _get_current_price(self):
        if 'Close' in self.data.columns:
            return self.data['Close'].iloc[-1]
        elif '4. close' in self.data.columns:
            return self.data['4. close'].iloc[-1]
        else:
            raise ValueError("The data does not have a recognized close price column.")

    def step(self, action):
        self.current_step += 1
        self.current_price = self._get_current_price()

        obs = self.features.iloc[self.current_step].values

        if self.current_step >= 5:
            sma_5 = self.features.iloc[self.current_step-5:self.current_step].mean().values
        else:
            sma_5 = np.zeros(self.features.shape[1])

        if self.current_step >= 10:
            sma_10 = self.features.iloc[self.current_step-10:self.current_step].mean().values
        else:
            sma_10 = np.zeros(self.features.shape[1])

        obs = np.concatenate([obs, sma_5, sma_10, [self.current_position], [self.current_price]]).astype(np.float32)

        reward = self.calculate_reward(action)
        done = self.current_step >= len(self.features) - 1

        if action == 1:
            self.current_position = 1
        elif action == 2:
            self.current_position = 2
        elif action == 3:
            self.current_position = 0

        return obs, reward, done, {}

    def calculate_reward(self, action):
        if self.current_position == 0:
            return 0.0

        current_price = self.targets[self.current_step]
        next_price = self.targets[self.current_step + 1] if self.current_step < len(self.targets) - 1 else current_price

        if self.current_position == 1:  # Long
            reward = next_price - current_price
        elif self.current_position == 2:  # Short
            reward = current_price - next_price
        else:
            reward = 0.0

        return float(reward)

