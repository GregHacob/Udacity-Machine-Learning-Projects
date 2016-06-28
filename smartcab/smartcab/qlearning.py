import random

class QLearning:
    def __init__(self, init_value = 0, epsilon=0.2, alpha=0.9, gamma=0.4):
    
        # Initialize the Q-function as a dictionary state:actions
        self.Q_function = {}
        
        # Initial value of any (state, action) tuple is an arbitrary random number
        self.init_value = init_value
        
        ## Discount factor gamma
        self.gamma = gamma
        
        ## Learning rate alpha: 0 (no learning) vs 1 (consider only most recent information)
        self.alpha = alpha
        
        ## Parameter of the epsilon-greedy action selection strategy
        self.epsilon = epsilon

    def getQvalue(self, state, action):
        return self.Q_function[state][action]
    
    def getQfunction(self):
        return self.Q_function   

    def initQfunction(self, filename, actions):
        import pandas as pd
        q_df = pd.read_csv(filename, sep=',', header=None)
        try:
            for row in xrange(q_df.shape[0]):           
                state = q_df.loc[row][0] 
                action_function = {}
                col = 1
                for action in actions:
                    action_function[action] = q_df.loc[row][col]
                    col += 1
                self.Q_function[state] = action_function
        except IOError: 
            print 'There is no file named', filename  

    def chooseAction(self, state, actions, is_current):
        if state in self.Q_function:
            ## Find the action that has the highest value
            action_function = self.Q_function[state]
            q_action = max(action_function, key = action_function.get)
            if is_current:
                ## Generate a random action
                rand_action = random.choice(actions)
                ## Select action using epsilon-greedy heuristic
                rand_num = random.random()
                action = q_action if rand_num <= (1 - self.epsilon) else rand_action
            else:
                action = q_action
        else:
            ## Initialize <state, action> pairs and select random action
            action_function = {}
            for action in actions:
                action_function[action] = self.init_value
            self.Q_function[state] = action_function
            action = random.choice(actions)        
        
        return action
        
    def learn(self, state1, action1, state2, action2, reward):
         ## Get the new Q_value
        current_q_value = self.getQvalue(state1,action1)
        new_state_q_value = self.getQvalue(state2,action2)
        
        ## Update the Q-function
        #self.Q_function[state1][action1] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * new_state_q_value)
        self.Q_function[state1][action1] = current_q_value + self.alpha*(reward + self.gamma * new_state_q_value - current_q_value)

