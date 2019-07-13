import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

file_name = "EURUSD43200.csv"
data = pd.read_csv(file_name)
data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
print(data)

# view the data pattern in plt
plt.plot(data['date'], data['close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(file_name.split("4")[0] + " close price pattern")
plt.show()

class DecisionPolicy:
    def select_action(self, current_state, step):
        pass
    
    def update_q(self, state, action, reward, next_state):
        pass

class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

        def select_action(self, current_state, step):
            action = self.actions[random.randint(0, len(self.actions) - 1)]

            return action

def run_simulation(policy, initial_budget, initial_entry_number_buy, initial_entry_number_sell, prices, hist, debug=False):
    budget = initial_budget
    num_of_entry_buy = initial_entry_number_buy
    num_of_entry_sell = initial_entry_number_sell
    pair_value = 0
    pips = 0
    sell_price = 0
    buy_price = 0
    current_portfolio = 0
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
            current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_of_entry_buy, num_of_entry_sell)))
            if num_of_entry_buy == 1:
                if sell_price > buy_price:
                    # calculate the profit
                    pip = (sell_price - buy_price) * 1000
                    pip_m = pip * 1
                    current_portfolio = budget + pip_m
                else:
                    #calculate the loss
                    pip = (sell_price - buy_price) * 1000
                    pip_m = pip * 1
                    current_portfolio = budget - pip_m
