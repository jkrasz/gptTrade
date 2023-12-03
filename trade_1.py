
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from custom_env import StockTradingEnv  # Ensure this is correctly imported
from stable_baselines3.common.vec_env import DummyVecEnv
import smtplib
import ta
from datetime import datetime, timedelta, time
import pytz
from time import sleep

# Function to check if the market is open
def is_market_open():
    opening_time = time(9, 30)
    closing_time = time(16, 0)
    now = datetime.now(pytz.timezone('US/Eastern'))
    return opening_time <= now.time() <= closing_time and now.weekday() < 5

# Function to send email alerts
def send_email(subject, content):
    sender_email = 'chatGptTrade@gmail.com'
    sender_password = 'bymwlvzzmbzxeeas'  # In a real scenario, use a secure method to store and access credentials
    receiver_email = 'john.kraszewski@gmail.com'
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, sender_password)
    message = f"""Subject: {subject}\n\n{content}"""
    print(message)
    try:
        smtp_server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print("Error sending email: ", e)
    smtp_server.quit()

# Function to fetch data with additional technical indicators
def fetch_data():
    symbol = 'GPRO'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=27)  # Adjusted for more recent data
    data = yf.download(symbol, start="2020-01-01", end=end_date.strftime('%Y-%m-%d'))
    # Ensure no zero values in the data that will be used in divisions
    data.replace(0, np.nan, inplace=True)
    # Additional Technical Indicators
    data['RSI'] = ta.momentum.rsi(data['Close']).fillna(50)  # Replace NaN values with a neutral RSI value like 50
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
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    data.dropna(inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

def dynamic_stop_loss_percent(volatility):
    return max(0.90, 1 - volatility / 100)

def dynamic_take_profit_percent(volatility):
    return min(1.15, 1 + volatility / 50)

# Trading thresholds and parameters
threshold_rsi_sell = 70
stop_loss_percent = 0.95
take_profit_percent = 1.1
threshold_rsi_buy = 30
leverage = 1  # Using leverage to amplify potential gains and losses

# Fetch initial data and set up environment and model
data = fetch_data()
env = DummyVecEnv([lambda: StockTradingEnv(data)])
model = PPO("MlpPolicy", env, verbose=1)
actions = []
in_position = False

# Main trading loop
while True:
    #if True:
    if is_market_open():
        data = fetch_data()  # Update data while the market is open
        model.learn(total_timesteps=20000)
        obs = env.reset()
        last_data_point_index = len(data) - 1

        for i in range(len(data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            close_price = data.iloc[i]['Close']
            rsi = data.iloc[i]['momentum_rsi']
            macd = data.iloc[i]['trend_macd']
            signal = data.iloc[i]['trend_macd_signal']
            volatility = data.iloc[i]['Volatility']

            stop_loss_percent = dynamic_stop_loss_percent(volatility)
            take_profit_percent = dynamic_take_profit_percent(volatility)


            if not in_position and rsi < threshold_rsi_buy and macd > signal:
                in_position = True
                buy_price = close_price
                actions.append((i, "Buy", close_price))
                if i == last_data_point_index:
                    send_email("Buy Signal", f"Buy at price {close_price}")
            elif in_position:
                if (rsi > threshold_rsi_sell or macd < signal or
                    close_price <= buy_price * stop_loss_percent or
                    close_price >= buy_price * take_profit_percent):
                    in_position = False
                    actions.append((i, "Sell", close_price))
                    if i == last_data_point_index:
                        send_email("Sell Signal", f"Sell at price {close_price}")

            if done:
                obs = env.reset()

        print("Simulation Completed! Sleeping...")
        sleep(60 * 15)
    else:
        print("Market is closed. Sleeping...")
        sleep(60 * 15)
