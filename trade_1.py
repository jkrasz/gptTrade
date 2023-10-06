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
data['EMA'] = ta.trend.ema_indicator(data['Close'])
data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'])
#data['MFI'] = ta.momentum.MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14).money_flow_index()
threshold_rsi_buy = 30
threshold_rsi_sell = 70
stop_loss_percent = 0.95  # sell if price drops to 95% of buying price
take_profit_percent = 1.1  # sell if price increases to 110% of buying price
# ...
data.dropna(inplace=True)

# Define the environment and model
env = DummyVecEnv([lambda: StockTradingEnv(data)])
model = PPO("MlpPolicy", env, verbose=1)


actions = []
in_position = False
while True:
  if is_market_open():
  #if True:
    model.learn(total_timesteps=20000)
    obs = env.reset()
    for i in range(len(data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        close_price = data.iloc[i]['Close']
        rsi = data.iloc[i]['RSI']

        if not in_position:
            if action == 1 and rsi < threshold_rsi_buy:  # Buy condition
                in_position = True
                buy_price = close_price
                actions.append((i, "Buy", close_price))
                # Set stop_loss and take_profit price
                stop_loss_price = buy_price * stop_loss_percent
                take_profit_price = buy_price * take_profit_percent
        else:
            if ((action == 2 and rsi > threshold_rsi_sell) or
                close_price <= stop_loss_price or
                close_price >= take_profit_price):  # Sell condition
                in_position = False
                actions.append((i, "Sell", close_price))
        #...
        if done:
            obs = env.reset()
    print("Simulation Completed!")

    # Batch Email Notifications
    if actions:  # Only send email if there are actions
        actions_summary = "\n".join([f"{i}: {action} at price {price}" for i, action, price in actions])
        send_email("Summary of actions", actions_summary)
        actions = []  # Clear actions after sending email
  else:
    print("Market is closed. Sleeping...")
    sleep(60 * 15)
