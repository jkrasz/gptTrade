import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from custom_env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import smtplib
import ta  # For additional technical indicators
from datetime import datetime, timedelta, time
import pytz
from time import sleep


def is_market_open():
    opening_time = time(9, 30)
    closing_time = time(16, 0)
    now = datetime.now(pytz.timezone('US/Eastern'))
    return opening_time <= now.time() <= closing_time and now.weekday() < 5

def send_email(subject, content):
    sender_email = 'chatGptTrade@gmail.com'
    sender_password = 'bymwlvzzmbzxeeas'
    receiver_email = 'john.kraszewski@gmail.com'
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, sender_password)
    message = f"Subject: {subject}\n\n{content}"
    print(message)
    try:
        smtp_server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print("Error sending email: ", e)
    smtp_server.quit()


# Fetch the data
symbol = 'GPRO'
end_date = datetime.now() - timedelta(days=1)
data = yf.download(symbol, start="2020-01-01", end=end_date.strftime('%Y-%m-%d'))

# Additional Technical Indicators
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['RSI'] = ta.momentum.rsi(data['Close'])
data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

data.dropna(inplace=True)

# Define the environment and model
env = DummyVecEnv([lambda: StockTradingEnv(data)])
model = PPO("MlpPolicy", env, verbose=1)

while True:
    if is_market_open():
    #if True:
        # Train the model
        model.learn(total_timesteps=20000)

        # Simulated Trading
        obs = env.reset()
        for i in range(len(data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

            if action == 1:
                print(f"Buying at step {i}, price: {obs['Close']}")
                send_email("Buy Signal", f"Buying at step {i}, price: {obs['Close']}")
            elif action == 2:
                print(f"Selling at step {i}, price: {obs['Close']}")
                send_email("Sell Signal", f"Selling at step {i}, price: {obs['Close']}")

            if done:
                obs = env.reset()
        print("Simulation Completed!")
    else:
        print("Market is closed. Sleeping...")
        sleep(60 * 15)  # Sleep for 15 minutes before checking again
