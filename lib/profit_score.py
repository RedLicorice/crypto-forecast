import numpy as np


def get_profit(close, y_pred, initial_balance=100, position_size=0.1):
    position_amount = initial_balance*position_size
    balance=initial_balance
    coins = 0
    last_price = None
    for price, y in zip(close, y_pred):
        if not price or np.isnan(price):
            continue
        if y not in [0, 1]:
            continue
        if not y: # Sell if y == 0
            amount = position_amount/price
            if coins < amount:
                amount = coins
            coins -= amount
            balance += amount*price
        else:# Buy if y == 1
            amount = position_amount/price
            if balance < position_amount:
                amount = balance/price
            balance -= amount*price
            coins += amount
        last_price = price
    if coins and last_price:
        balance += coins*last_price
        coins = 0
    return balance/initial_balance