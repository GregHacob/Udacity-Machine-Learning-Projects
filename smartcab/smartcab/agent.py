import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from qlearning import QLearning
import getopt, sys
import itertools
import pandas as pd
import csv
import calendar
import time

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
        action = random.choice(self.env.valid_actions)
        reward = self.env.act(self, action)
        
        light = inputs['light']
        oncoming = inputs['oncoming']
        left = inputs['left']
        right = inputs['right']
       
        #Update the current observed state
        self.state = (light, oncoming, left, right, deadline, self.next_waypoint)
        #self.state = (light, oncoming, left, right, self.next_waypoint)
        
        print "Basic.update(): next move = {}".format(action)  # [debug]


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, params = False):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        #print "************************************", params["init_values"]
        self.cumulative_reward = 0
        
        if params:
         self.ai = QLearning(init_value = params[0], epsilon = params[1], alpha=params[2], gamma =params[3])  
        else:
         self.ai = QLearning()          

        self.num_reached_dest = 0        
        self.total_steps = 0
          

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
        self.state = (light, oncoming, left, right, deadline, self.next_waypoint)
        #self.state = (light, oncoming, left, right, self.next_waypoint)
        current_state = self.state
        
        #Choose an action
        action = self.ai.chooseAction(current_state, self.env.valid_actions, True)

        #Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        if self.env.done:
         self.num_reached_dest+=1
        
        self.total_steps+=self.env.t        
        
        #Update the state variables after action  and move to a new state             
        new_inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        light = new_inputs['light']
        oncoming = new_inputs['oncoming']
        left = new_inputs['left']
        right = new_inputs['right']
        
        deadline = self.env.get_deadline(self)
        
        state_prime = (light, oncoming, left, right, deadline, self.next_waypoint)
        #state_prime = (light, oncoming, left, right, self.next_waypoint)
          
        #Get the best action for the new state
        action_prime = self.ai.chooseAction(state_prime, self.env.valid_actions, False)    
        
        #Learn policy based on state, action, reward
        self.ai.learn(current_state, action, state_prime, action_prime, reward )
       

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    
     
    
    def SaveQTable(self):
       datetime_int = int(calendar.timegm(time.gmtime()))
       out_filename = 'q_table_' + str(datetime_int) + '.csv'
       #out_filename = 'q_table.csv'
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
 

     
def run(argv):

    smartAgent = False
    defult_num_trilas=100
    defult_delay = 1
    delay = 0
    num_trials = 0
    h_params = False
    output = False
    preserve = False
    filename = ''
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "bst:d:pr:ho")
        print opts
        print args
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) 
        sys.exit(2)
          
    for o, a in opts:
        if o == "-b":
           smartAgent = False
        elif o == "-s":
           smartAgent = True 
        if o == "-t":
           num_trials = int(a)          
        if o == "-d":
           delay = float(a)    
        if o == "-p":
           preserve = True
        if o == "-r":
           filename = a     
        if o == "-h":
            h_params = True
        if o == "-o":
            output = True    
    
    if (num_trials == 0):
       num_trials = defult_num_trilas
     
    if (delay == 0):
       delay = defult_delay
       
    if smartAgent == False:
        createBasicAgent(num_trials, delay)
    else: 
        crateSmartAgent(num_trials, delay, filename, h_params, output, preserve)
  
        
def createBasicAgent(total_trials, delay): 
    e = Environment()  # create environment (also adds some dummy traffic)            
    a = e.create_agent(BasicAgent)  # create not very smart Basic agent 
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track    
    sim = Simulator(e, update_delay=delay)  # reduce update_delay to speed up simulation
    sim.run(n_trials = total_trials)  # press Esc or close pygame window to quit     
    
def crateSmartAgent(total_trials, delay, filename, withhyperparams = False, output = False, preserve=False):    
    
    """Run the agent for a finite number of trials."""
     
    if withhyperparams:
    
        hyper_params = {"init_values":[0],
                        "epsilons":[.1,.2,.4],
                        "alphas":[.3,.7,.9],
                        "gammas":[.4,.7,.9]}
                           
        params = list(itertools.product(*[hyper_params["init_values"],
                                          hyper_params["epsilons"],
                                          hyper_params["alphas"],
                                          hyper_params["gammas"]]))
        if output:
          df = pd.DataFrame(columns=['init_value', 'epsilon', 'alpha', 'gamma','Total Steps', 'Reached Dest.'])
          
        for i in range(len(params)):
        
         print "*******************************************"
         print params[i]     
         print "*******************************************"
         # Set up environment and agent
         e = Environment()  # create environment (also adds some dummy traffic) 
         agent = e.create_agent(LearningAgent, params[i])  # create Smart agent    
         e.set_primary_agent(agent, enforce_deadline=True)  # set agent to track
        
         # Now simulate it
         sim = Simulator(e, update_delay=delay)  # reduce update_delay to speed up simulation
         sim.run(n_trials = total_trials)  # press Esc or close pygame window to quit 
         #print "number of successful customer drop offs:", agent.num_reached_dest    
         #print "Total Steps: {} in {} trial".format(agent.total_steps, total_trials)
         if output:
          df.loc[i] = [params[i][0],params[i][1],params[i][2],params[i][3], agent.total_steps,agent.num_reached_dest]
         if preserve:
            agent.SaveQTable()
            
        if output:   
         df.to_csv('Smartcab Perfrmance Report.csv')  
     
    else:
        
            # Set up environment and agent
         e = Environment()  # create environment (also adds some dummy traffic) 
         agent = e.create_agent(LearningAgent)  # create Smart agent    
         e.set_primary_agent(agent, enforce_deadline=True)  # set agent to track
         
         if (len(filename) > 0):
            agent.ai.initQfunction(filename,agent.env.valid_actions)
            
         # Now simulate it
         sim = Simulator(e, update_delay=delay)  # reduce update_delay to speed up simulation
         sim.run(n_trials = total_trials)  # press Esc or close pygame window to quit 
         print "number of successful customer drop offs:", agent.num_reached_dest    
         print "Total Steps: {} in {} trial".format(agent.total_steps, total_trials)
         if preserve:
            agent.SaveQTable()       


if __name__ == '__main__':
    run(sys.argv)
