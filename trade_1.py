import logging, json, smtplib, pytz, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from datetime import datetime, time
from time import sleep
from sklearn.exceptions import DataConversionWarning
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from custom_env import CustomTradingEnvironment
import yfinance as yf
from dateutil.relativedelta import relativedelta

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
log_file = Path("trade_log.json")

if log_file.exists():
    with open(log_file, "r") as file:
        log_data = json.load(file)
    position = log_data["position"]
    entry_price = log_data["entry_price"]
    max_price = log_data["max_price"]
    min_price = log_data["min_price"]
else:
    log_data = {"position": 0, "entry_price": 0, "max_price": 0, "min_price": 0}

logging.basicConfig(filename="algorithm.log", level=logging.INFO)

def make_decision(model_prediction, rsi_value, ema20, ema50):
    if model_prediction == 1 and rsi_value < 30 and ema20 > ema50: # Over-sold and uptrend
        return 1
    elif model_prediction == 2 and rsi_value > 70 and ema20 < ema50: # Over-bought and downtrend
        return 2
    elif model_prediction == 3:
        return 3
    else:
        return 

def train_model(env, model_save_path='ppo_model.pkl'):
    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, ent_coef=0.005, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=None)
    model.learn(total_timesteps=100000)
    model.save(model_save_path)
    return model

def load_model(model_save_path='ppo_model.pkl'):
    model = PPO.load(model_save_path)
    return model
def prepare_data(data):
    data = data.dropna()
    features = data.drop('Close', axis=1)
    targets = data['Close'].shift(-1).dropna()
    features = features[:-1]
    return features, targets


def send_email(subject, message_body):
    sender_email = 'chatGptTrade@gmail.com'
    sender_password = 'bymwlvzzmbzxeeas'
    receiver_email = 'john.kraszewski@gmail.com'
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, sender_password)
    message = f"Subject: {subject}\n\n{message_body}"
    print(message)
    try:
        smtp_server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print("Error sending email: ", e)
    smtp_server.quit()

def is_market_open():
    opening_time = time(9, 30)
    closing_time = time(16, 0)
    now = datetime.now(pytz.timezone('US/Eastern'))
    return opening_time <= now.time() <= closing_time and now.weekday() < 5

symbol = 'GPRO'
model_save_path = 'ppo_model.pkl'

data = yf.download(symbol, period="5y")
data = data.sort_index(ascending=True)

features, targets = prepare_data(data)
env = CustomTradingEnvironment(data, features, targets)
check_env(env)
vec_env = DummyVecEnv([lambda: env])

today = datetime.today()
start_date = today - relativedelta(months=6)
end_date = today - relativedelta(days=1)
data = yf.download(symbol, start=start_date, end=end_date)
print(data.columns)
features, targets = prepare_data(data)
env = CustomTradingEnvironment(data, features, targets)
vec_env = DummyVecEnv([lambda: env])
print(env.current_price)
train = True
if train:
    model = train_model(vec_env, model_save_path=model_save_path)
else:
    model = load_model(model_save_path=model_save_path)


model_save_interval = 1000
position = 0
max_price = 0
min_price = 0
entry_price = 0
last_action = None
while True:
    if True:
    #if is_market_open():
        obs = env.reset()
        data = yf.download(symbol, period="1d", interval="15m")
        
        ema_indicator_20 = EMAIndicator(data['Close'], window=20)
        ema_indicator_50 = EMAIndicator(data['Close'], window=50)
        data['EMA20'] = ema_indicator_20.ema_indicator()
        data['EMA50'] = ema_indicator_50.ema_indicator()
        rsi_indicator = RSIIndicator(data['Close'])
        data['RSI'] = rsi_indicator.rsi()
        
        # Getting current RSI, EMA20, EMA50
        current_rsi = data['RSI'].iloc[-1]
        current_ema20 = data['EMA20'].iloc[-1]
        current_ema50 = data['EMA50'].iloc[-1]
        
        model_prediction, _ = model.predict(obs, deterministic=True)
        
        # Now we take the action based on our new decision-making function
        action = make_decision(model_prediction, current_rsi, current_ema20, current_ema50)
        
        if action is not None:
            obs, reward, done, info = env.step(action)
        else:
            # Handle the case where action is None.
            # For example, you can skip the step and log a warning.
            logging.warning("Action is None, skipping this step.")
            continue
        
        print(env.current_price)
        if action != last_action:
            last_action = action
            if action == 1:  # Buy
                position = 1
                entry_price = env.current_price
                max_price = env.current_price
                min_price = np.inf
                send_email("Entering LONG position", f"Entry price: {entry_price}")
                logging.info(f"{datetime.now(pytz.timezone('US/Eastern'))} - Entering LONG position at {entry_price}")

            elif action == 2:  # Sell
                position = -1
                entry_price = env.current_price
                max_price = -np.inf
                min_price = env.current_price
                send_email("Entering SHORT position", f"Entry price: {entry_price}")
                logging.info(f"{datetime.now(pytz.timezone('US/Eastern'))} - Entering SHORT position at {entry_price}")

            elif action == 3:  # Close position
                if position == 1:  # Close long position
                    pnl = env.current_price / entry_price - 1
                    send_email("Closing LONG position", f"Close price: {env.current_price}, PnL: {pnl:.2%}")
                    logging.info(f"{datetime.now(pytz.timezone('US/Eastern'))} - Closing LONG position at {env.current_price}, PnL: {pnl:.2%}")

                elif position == -1:  # Close short position
                    pnl = entry_price / env.current_price - 1
                    send_email("Closing SHORT position", f"Close price: {env.current_price}, PnL: {pnl:.2%}")
                    logging.info(f"{datetime.now(pytz.timezone('US/Eastern'))} - Closing SHORT position at {env.current_price}, PnL: {pnl:.2%}")

        log_data["position"] = position
        log_data["entry_price"] = entry_price
        log_data["max_price"] = max_price
        log_data["min_price"] = min_price

        with open(log_file, "w") as file:
            json.dump(log_data, file)

        if env.current_step % model_save_interval == 0:
            model.save(model_save_path)
            print(f"Model saved at step {env.current_step}")

    else:
        print('market is closed')
        sleep(60 * 60)  # Sleep for an hour

    sleep(60 * 15)  # Sleep for 15 minutes


