
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
import random
import time

class AI_Collision_1811110(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        #Set containg all the packets
        self.packet_set=set()
        #Dictionary for taken actions {id_pkd:[(action,timestamp),(action,timestamp)...]}
        self.taken_actions = {}
        #Path of the drone
        self.drone_path=[]
        self.counter=0
        #Set of all the collisions
        self.set_collision=set({})
        #Set of all the visited waypoints
        self.set_waypoint=set({})
        #Boolean if a lap is completed
        self.completed_lap=False
        #Dictionary for the delivery of a packet. When pkd:true, the pkd is delivered
        self.packet_dictio={}
        #List of the currently holded packets, sorted by their generation timestep
        self.packet_generation=[]
        #Variable for the first waypoint
        self.first_waypoint=None
        #Variable for the current lap number
        self.number_lap=0
        #Dictionary for the reward for each action
        self.actions_rewards={}
        #Set of all the taken actions
        self.actions_set=set()
        #Dictionary for the Q-Table
        self.qTable_dictionary={}
        #timestamp for each action
        self.actions_timestamp={}
        #Alpha parameter for the Bellman formula
        self.alpha=0.5
        #Gamma parameter for the Bellman formula
        self.gamma=0.8
        #Dictionary for the convergence test
        self.reward_dictionary={}
    def feedback(self, drone, id_event, delay, outcome, depot_index=None):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome)
        actual_time=self.simulator.cur_step
        #Update of the list of the packet sorted by their generation
        if id_event in self.packet_set and drone==self.drone:
            self.packet_generation=[x for x in self.packet_generation if x[0].event_ref.identifier!=id_event]            
            self.packet_generation=sorted(self.packet_generation,key=lambda x: x[1]) 
        if id_event in self.taken_actions:
            #Packet delivered
            if outcome==0:
                self.packet_dictio[id_event]=True
            #Code executed for each action regarding the pkd
            for action,timestamp in self.taken_actions[id_event]:
                next_state=None
                actual_index=self.taken_actions[id_event].index((action,timestamp))
                #Next state involved after the current action
                if actual_index!=len(self.taken_actions[id_event])-1:
                    next_state=self.taken_actions[id_event][actual_index+1][0][0]
                #State in which the action has been done
                state=action[0]
                #The action
                act=action[1]
                #Reward calculation
                reward=self.reward_function(delay,outcome)
                #Update of the q-table through the bellman equation
                value=self.update_q_table(state,act,next_state,reward)
                self.update_step_reward_dictionary(reward,timestamp)
            del self.taken_actions[id_event]
    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        #Current cell/state
        
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                        width_area=self.simulator.env_width,
                                                        x_pos=self.drone.coords[0],  # e.g. 1500
                                                        y_pos=self.drone.coords[1])[0]  # e.g. 500
        #Update of the next state, for the previous actions
        self.update_next_state(cell_index)
        #Inizialition of the taken_actions dictionary for the current cell
        self.initialize_state_action(pkd,cell_index)
        #Inizialization of the q-table dictionary
        self.initialize_q_table(cell_index)
        #Update of the packet set and the list of current packets sorted by generation
        if pkd.event_ref.identifier not in self.packet_set:
            self.packet_set.add(pkd.event_ref.identifier)
            self.packet_generation.append((pkd,pkd.time_step_creation))
            self.packet_generation=sorted(self.packet_generation,key=lambda x: x[1])
        #The drone is added into the packet hops
        pkd.hops.add(self.drone.identifier)
        #If there are 10 or more packets, come back
        if len(self.packet_generation)>=10:
            #TODO AI MARCO---------------------------------------
            return-1
        #When the simulation is going to end, come back
        #TODO NUOVO IS TIME TO GO BACK---------------------------
        if self.is_time_to_goback():
            return -1
        #Set of the first waypoint
        if self.first_waypoint==None:
            self.first_waypoint=self.drone.next_target()
        #--------------Code used to determine in what point of the mission the drone is
        globalhistory=self.drone.waypoint_history
        localHistory=[]            
        for point in reversed(globalhistory):
            localHistory.insert(0,point) 
            self.set_waypoint.add(point)
            if point[0]==globalhistory[0][0] and point[1]==globalhistory[0][1]:
                if len(localHistory)<len(self.set_waypoint):
                    self.completed_lap=True
                break
        #--------------------------------------------------------------------------------
        #Inizialization of dictionary for the delivery of a packet
        if pkd.event_ref.identifier not in self.packet_dictio:
            self.packet_dictio[pkd.event_ref.identifier]=False
        #CODE OF THE BASE VERSION, to understand when the drone is at the end of the mission, and can come back
        #to the depot
        if self.drone_path == [] and self.drone.waypoint_history != []:
            if self.drone.waypoint_history[self.drone.current_waypoint - 1][1] < self.drone.waypoint_history[self.drone.current_waypoint - 2][1]:
                self.drone_path = self.drone.waypoint_history.copy()
        elif self.drone.next_target() in self.drone_path:
            self.counter+=1
            if self.drone_path.index(self.drone.next_target()) == 0 and self.drone.buffer_length() >= 3:
                return -1
        if self.drone.next_target()==self.simulator.depot.list_of_coords[0]:
            return -1
        #Code used for the AI choice, when there is a collision
        drone=None
       # if self.drone.identifier==0:
        #    print(self.check_near_upper_depot())
        if len(opt_neighbors)>0:
            #If the drone has never received the pkd....
            if self.drone_not_seen([v[1] for v in opt_neighbors],pkd):
                #Check to know if the protocol has already taken a choice for the cell
                if self.already_chosen_check(pkd):
                    drone=self.untaken_drone([v[1] for v in opt_neighbors],pkd)
                else:
                    #Epsilon greedy policy implementation
                    randomChoice=random.choices([True,False],weights=(10,90),k=1)[0]
                    if randomChoice:
                        #Action
                        give_packet_away=random.choices([True,False],weights=(100,0),k=1)[0]
                    else:
                        #Action
                        give_packet_away,q_value=self.perform_greedy_action(cell_index)
                    #Case packet given to a drone of the collision
                    if give_packet_away:
                        drone=self.untaken_drone([v[1] for v in opt_neighbors],pkd)
                        self.make_choice(True,cell_index,pkd)
                        self.packet_generation=[x for x in self.packet_generation if x[0].event_ref.identifier!=pkd.event_ref.identifier]            
                        self.packet_generation=sorted(self.packet_generation,key=lambda x: x[1]) 
                    else:
                        #Case maintained packet
                        self.make_choice(False,cell_index,pkd)
                        drone=None
        else:
            #If the packet has been already given to another drone, but i still have it, there is a trassmission error
            #The function is used to correct the data structure about the packets
            if self.already_chosen_check(pkd):
                self.correct_trasmission_error(pkd,cell_index)
        return drone # here you should return a drone object!
    #Function used to know if from the next target i have enough time to come back
    #TODO CAMBIARE CON VERSIONE NUOVA
    def arrival_time(self, drone):
        tot=(util.euclidean_distance(drone.next_target(), drone.coords) / drone.speed)+(util.euclidean_distance(drone.next_target(), self.simulator.depot.list_of_coords[0])/drone.speed)
        return tot 
    #Function used to knwo when the simulation is going to end , and its time to come back to the depot
    #TODO CAMBIARE CON VERSIONE NUOVA
    def is_time_to_goback(self):
        time_expected=self.arrival_time(self.drone)
        end_expected=self.simulator.len_simulation*self.simulator.time_step_duration-(self.simulator.cur_step*self.simulator.time_step_duration)
        return time_expected>end_expected
    #Function that says if a pkd is expiring
    #TODO ELIMINARE
    def is_packet_expiring(self,pkd):
        time_left=8000*self.simulator.time_step_duration-(self.simulator.cur_step*self.simulator.time_step_duration-pkd.time_step_creation*self.simulator.time_step_duration)
        expected_time=self.arrival_time(self.drone)
        return expected_time>time_left
    #Function for the lap counting
    def lap_counter(self,globalHistory):
        if globalHistory==[]:
            return 1
        count_list=[x for x in globalHistory if x==globalHistory[0]]
        return len(count_list)
    #Function used to perform a random action in the epsilon greedy policy
    def perform_random_action(self,opt_neighbors,cell,pkd):
        opt2=[x[1] for x in opt_neighbors]
        randomChoice=self.untaken_drone(opt2,pkd)
        return randomChoice
    #Function that choose a random drone which has not already received the given pkd, during a collision 
    def untaken_drone(self,opt_neighbors,pkd):
        while True:
            drone=self.simulator.rnd_routing.choice(opt_neighbors)
            if drone.identifier not in pkd.hops:
                return drone 
    def drone_not_seen(self,opt_neighbors,pkd):
        for collision in opt_neighbors:
            if collision.identifier not in pkd.hops:
                return True
        return False
    #Function used to update the taken actions dictionary after a choice
    def make_choice(self,choice,cell_index,pkd):
        if choice:
            self.taken_actions[pkd.event_ref.identifier][len(self.taken_actions[pkd.event_ref.identifier])-1]=((cell_index,True,-1),self.simulator.cur_step)
        else:
            self.taken_actions[pkd.event_ref.identifier][len(self.taken_actions[pkd.event_ref.identifier])-1]=((cell_index,False,None),self.simulator.cur_step)
    #Function which says if for the given packet, is already taken a choice (maintain or give away)
    def already_chosen_check(self,pkd):
        if len(self.taken_actions[pkd.event_ref.identifier])==0:
            return False
        if self.taken_actions[pkd.event_ref.identifier][len(self.taken_actions[pkd.event_ref.identifier])-1][0][1]==True:
            return True
        return False
    #Function used to initialize the taken_action dictionary
    def initialize_state_action(self,pkd,cell):
        if pkd.event_ref.identifier not in self.taken_actions:
            self.taken_actions[pkd.event_ref.identifier]=[]
        if len(self.taken_actions[pkd.event_ref.identifier])==0:
            self.taken_actions[pkd.event_ref.identifier].append(((cell,False,None),self.simulator.cur_step))
            return True
        if self.taken_actions[pkd.event_ref.identifier][len(self.taken_actions[pkd.event_ref.identifier])-1][0][0]==cell:
            return False
        self.taken_actions[pkd.event_ref.identifier].append(((cell,False,None),self.simulator.cur_step))
        return True
    #Function used to set the next state of a tuple (state,action)
    def update_next_state(self,cell_index):
        current_state=cell_index
        for pkd in self.taken_actions.keys():
            past_actions=self.taken_actions[pkd]
            if len(past_actions)==0:
                pass
            if past_actions[len(past_actions)-1][0][1]==False and past_actions[len(past_actions)-1][0][0]!=current_state:
                past_actions[len(past_actions)-1]=((past_actions[len(past_actions)-1][0][0],past_actions[len(past_actions)-1][0][1],current_state),past_actions[len(past_actions)-1][1])
    #Function used when a packet is logically given away, but it is physically still maitained by the drone
    #In this case we have a trasmission error, and we must correct the data structures about that pkd
    def correct_trasmission_error(self,pkd,cell):
        past_actions=self.taken_actions[pkd.event_ref.identifier]
        past_actions[len(past_actions)-1]=((past_actions[len(past_actions)-1][0][0],False,cell),past_actions[len(past_actions)-1][1])
    def print(self):
        pass
    #Function used to initialized the q table
    def initialize_q_table(self,cell):
        for action in [True,False]:
            if (cell,action) not in self.qTable_dictionary.keys():
                self.qTable_dictionary[(cell,action)]=0

    #Reward function
    def reward_function(self,delay,outcome):
        if outcome==-1:
            return -1
        else:
            return 1-(delay/8000) 
    #Function used to update the q_table when a new reward about a (state,action) tuple is given     
    def update_q_table(self,state,action,next_state,reward):
        qSa=self.qTable_dictionary[(state,action)]
        argmax=0
        if next_state!=None:
            argmax=self.argmax_next_state(next_state)
        value_to_update=qSa+self.alpha*(reward+self.gamma*argmax-qSa)
        self.qTable_dictionary[(state,action)]=value_to_update
        return value_to_update
    #Utility function used during the bellman formula implementation
    def argmax_next_state(self,nextState):
        max=None
        all_actions=[True,False]
        random.shuffle(all_actions)
        for action in all_actions:
            if max==None:
                max=self.qTable_dictionary[(nextState,action)]
            else:
                if self.qTable_dictionary[(nextState,action)]>max:
                    max=self.qTable_dictionary[(nextState,action)]
        return max
    #Function used for the epsilon greedy implementation. It choice the past action for the given cell, that has the
    #highest q value
    def perform_greedy_action(self,cell):
        give_away=self.qTable_dictionary[(cell,True)]
        keep_packet=self.qTable_dictionary[(cell,False)]
        return max([(x[0],self.qTable_dictionary[x]) for x in self.qTable_dictionary.keys()if x[0]==cell],key=lambda x:x[1])
    #Function used to update the dictionary for the convergence test
    def update_step_reward_dictionary(self,reward,cur_step):
        aprox_step=cur_step
        if aprox_step not in self.reward_dictionary.keys():
            self.reward_dictionary[aprox_step]=[]
        self.reward_dictionary[aprox_step].append(reward)
    def is_coming_back(self):
        return self.drone.next_target()==self.simulator.depot.list_of_coords[0] or self.drone.next_target()==self.simulator.depot.list_of_coords[1]
    def check_near_upper_depot(self):
        print("----------------",self.drone.next_target()[1],int(self.drone.coords[1]),self.drone.buffer_length())
        return self.drone.next_target()[1]<self.drone.return_coords[1] and self.drone.buffer_length() >= 3
  
