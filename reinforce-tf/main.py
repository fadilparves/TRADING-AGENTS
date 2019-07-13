from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf
import random
import os
import pandas as pd
from tensorflow.python.framework import ops
import warnings

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

class QLearning(DecisionPolicy):
    def __init__(self,actions,input_dim):
        self.epsilon = 0.5
        self.gamma = 0.001
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        def select_action(self, current_state, step):
            threshold = min(self.epsilon, step / 1000.)
            if random.random() < threshold:
                action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
                action_idx = np.argmax(action_q_vals)
                action = self.actions[action_idx]
            else:
                action = self.actions[random.randint(0, len(self.actions) - 1)]
            return action

        def update_q(self, state, action, reward, next_state):
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
            next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
            next_action_idx = np.argmax(next_action_q_vals)
            action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
            action_q_vals = np.squeeze(np.asarray(action_q_vals))
            self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

def run_simulation(policy, initial_budget, initial_entry_number_buy, initial_entry_number_sell, prices, hist, debug=False):
    budget = initial_budget
    num_of_entry_buy = initial_entry_number_buy
    num_of_entry_sell = initial_entry_number_sell
    pair_value = 0
    pips = 0
    sell_price = 0
    buy_price = 0
    current_portfolio = 0
    reward = 0
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
            current_state = np.asmatrix(np.hstack((prices.loc[i], budget, num_of_entry_buy, num_of_entry_sell)))

            action = policy.select_action(current_state, i)

            if action == 'Buy' and sell_price != 0:
                buy_price = prices.loc[i]
                if sell_price > buy_price:
                    # calculate the profit
                    pip = (sell_price - buy_price) * 1000
                    pip_m = pip * 1
                    reward = reward + pip_m
                    current_portfolio = budget + pip_m
                    print("Day " + str(i) + " BUY at " + str(buy_price) + " and last trade profit is " + str(pip_m))
                else:
                    #calculate the loss
                    pip = (buy_price - sell_price) * 1000
                    pip_m = pip * 1
                    reward = reward - pip_m
                    current_portfolio = budget - pip_m
                    print("Day " + str(i) + " BUY at " + str(buy_price) + " and last trade loss is -" + str(pip_m))

                num_of_entry_buy = 1
                num_of_entry_sell = 0

            elif action == 'Sell' and buy_price != 0:
                sell_price = prices.loc[i]
                if buy_price < sell_price:
                    pip = (sell_price - buy_price) * 1000
                    pip_m = pip * 1
                    reward = reward + pip_m
                    current_portfolio = budget + pip_m
                    print("Day " + str(i) + " SELL at " + str(sell_price) + " and last trade ptofit is " + str(pip_m))
                else:
                    pip = (buy_price - sell_price) * 1000
                    pip_m = pip * 1
                    reward = reward - pip_m
                    current_portfolio = budget - pip_m
                    print("Day " + str(i) + " SELL at " + str(sell_price) + " and last trade loss is -" + str(pip_m))

                num_of_entry_buy = 0
                num_of_entry_sell = 1
            
            else:
                action = 'Hold'
                next_state = np.asmatrix(np.hstack((prices.loc[i], budget, num_of_entry_buy, num_of_entry_sell)))
                transitions.append((current_state, action, reward, next_state))
                policy.update_q(current_state, action, reward, next_state)

    return current_portfolio

def run_simulations(policy, initial_budget, initial_entry_number_buy, initial_entry_number_sell, prices, hist):
    epochs = 100
    final_rewards = list()
    for i in range(epochs):
        final_reward = run_simulation(policy, initial_budget, initial_entry_number_buy, initial_entry_number_sell, prices, hist)
        final_rewards.append(final_reward)
    
    return final_rewards

if __name__ == '__main__':
    prices = data['close']
    actions = ['Buy', 'Sell', 'Hold']
    hist = 220
    policy = QLearning(actions, hist + 2)
    initial_budget = 100.0
    initial_entry_number_buy = 0
    initial_entry_number_sell = 0
    all_rewards = run_simulations(policy, initial_budget, initial_entry_number_buy, initial_entry_number_sell, prices, hist)
    print(all_rewards)


