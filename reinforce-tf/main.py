import pandas as pd
import matplotlib.pyplot as plt
import random

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

