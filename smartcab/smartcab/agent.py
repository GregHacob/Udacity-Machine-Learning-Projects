import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from qlearning import QLearning
import sys

class BasicAgent(Agent):
    def __init__(self, env):
        super(BasicAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.color = 'red'  # override color

    def reset(self, destination=None):
        self.planner.route_to(destination)  
        
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        action = random.choice(Environment.valid_actions[1:])
        reward = self.env.act(self, action)
        print "Basic.update(): next move = {}".format(action)  # [debug]


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        self.cumulative_reward = 0
 
        self.ai = QLearning()        
          

    def reset(self, destination=None):
        self.planner.route_to(destination) 
        self.cumulative_reward = 0        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        light = inputs['light']
        oncoming = inputs['oncoming']
        left = inputs['left']
        right = inputs['right']
       
        #Update the current observed state
        self.state = (light, oncoming, left, right, self.next_waypoint)
        current_state = self.state
        
        #Choose an action
        action = self.ai.chooseAction(current_state, self.env.valid_actions, True)

        #Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        
        #Update the state variables after action  and move to a new state             
        new_inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        light = new_inputs['light']
        oncoming = new_inputs['oncoming']
        left = new_inputs['left']
        right = new_inputs['right']
        
        state_prime = (light, oncoming, left, right, self.next_waypoint)
          
        #Get the best action for the new state
        action_prime = self.ai.chooseAction(state_prime, self.env.valid_actions, False)    
        
        #Learn policy based on state, action, reward
        self.ai.learn(current_state, action, state_prime, action_prime, reward )
       

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    
    def getStatetuple(self, state):
        light = state['light']
        oncoming = state['oncoming']
        left = state['left']
        right = state['right']
        return (light, oncoming, left, right, self.next_waypoint)
     
    
    def SaveQTable(self):
       import csv
       out_filename = 'q_table.csv'
       f = open(out_filename, 'wb')
       writer = csv.writer(f)
       for state, action_function in self.ai.getQfunction().items():
           q_row = []
           q_row.append(state)
           for action in self.env.valid_actions:
               q_row.append(action_function[action])
           writer.writerow(q_row)
       f.close()
       print 'Written to file: ' + out_filename
 
    def initQFunction(self, filename):
        import pandas as pd
        q_df = pd.read_csv(filename, sep=',', header=None)

        try:
            for row in xrange(q_df.shape[0]):           
                state = q_df.ix[row][0] 
                action_function = {}
                col = 1
                for action in ['Up','down','none','forward']:
                    action_function[action] = q_df.ix[row][col]
                    col += 1
                self.ai.Q_function[state] = action_function
        except IOError: 
            print 'There is no file named', filename  
     
def run(argv):
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create Smart agent
    #a = e.create_agent(BasicAgent)  # create Basic agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    if (len(argv) == 2):
     a.initQFunction(argv[1])
    # Now simulate it
    sim = Simulator(e, update_delay=.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    if (len(argv) < 2):
     a.SaveQTable()

if __name__ == '__main__':
    run(sys.argv)
