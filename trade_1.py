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


def fetch_data():
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
    macd_indicator = ta.trend.MACD(data['Close'])
    data['MACD_Line'] = macd_indicator.macd()
    data['Signal_Line'] = macd_indicator.macd_signal()
    data['RSI_Avg'] = data['RSI'].rolling(window=5).mean()
    data.dropna(inplace=True)
    return data

threshold_rsi_sell = 70
stop_loss_percent = 0.95  # sell if price drops to 95% of buying price
take_profit_percent = 1.1  # sell if price increases to 110% of buying price
threshold_rsi_buy = 30

data = fetch_data()  # Initial fetch
# Define the environment and model

env = DummyVecEnv([lambda: StockTradingEnv(data)])
model = PPO("MlpPolicy", env, verbose=1)

last_data_point_index = len(data) - 1  # Last index of the data

actions = []
in_position = False

while True:
    if is_market_open():
        data = fetch_data()  # Update data while the market is open
        model.learn(total_timesteps=20000)
        obs = env.reset()
        last_data_point_index = len(data) - 1  # Update the index for latest data

        for i in range(len(data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            close_price = data.iloc[i]['Close']
            rsi = data.iloc[i]['RSI']
            macd = data.iloc[i]['MACD_Line']
            signal = data.iloc[i]['Signal_Line']
            rsi_avg = data.iloc[i]['RSI_Avg']

            atr = data.iloc[i]['ATR']
            stop_loss_price = close_price - (atr * 1.5)
            take_profit_price = close_price + (atr * 1.5)

            if not in_position:
                if rsi < threshold_rsi_buy and macd > signal:
                    in_position = True
                    buy_price = close_price
                    actions.append((i, "Buy", close_price))
                    if i == last_data_point_index:  # If it's the latest data point
                        send_email("Buy Signal", f"Buy at price {close_price}")

            else:
                if (rsi > rsi_avg or macd < signal or
                        close_price <= stop_loss_price or close_price >= take_profit_price):
                    in_position = False
                    actions.append((i, "Sell", close_price))
                    if i == last_data_point_index:  # If it's the latest data point
                        send_email("Sell Signal", f"Sell at price {close_price}")

            if done:
                obs = env.reset()

        print("Simulation Completed! Sleeping...")
        sleep(60 * 15)
    else:
        print("Market is closed. Sleeping...")
        sleep(60 * 15)
