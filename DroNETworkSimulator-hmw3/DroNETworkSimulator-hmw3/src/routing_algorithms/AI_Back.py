import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
import random
import time

class AI_Back_1811110(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.packet_set=set()
        # dictionary for taken actions {id_pkd:[(action,timestamp),(action,timestamp)...]}
        self.taken_actions = {}  #id event : (old_action)
        # path of the drone
        self.drone_path=[]
        self.counter = 0
        # set of all the visited waypoints
        self.set_waypoint=set({})
        # boolean if a lap is completed
        self.completed_lap=False
        # dictionary for the delivery of a packet. When pkd:true, the pkd is delivered
        self.packet_dictio={}
        # set containing all the packets
        self.packet_set=set()
        # list of the currently holded packets, sorted by their generation timestep
        self.packet_generation=[]
        # variable for the first waypoint
        self.first_waypoint=None
        # dictionary for the Q table
        self.q_dict = {None : [0, 0]}
        # gamma parameter for the Bellman formula
        self.gamma = 0.8
        # alpha parameter for the Bellman formula
        self.alpha = 0.6
        # list to store the sequence of (state, action) that must get a reward
        self.state_action_list = []
        # dict to store the list of packages that the drone had in a specific state
        self.state_action_packets = {}
        # final time spent to go and return from depot
        self.final_time_to_depot = 0
        # last choice made by AI
        self.last_choice_index = None
        # dictionary to store evaluation info {timestep : reward}
        self.evaluation_dict = {}

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome)

        # if the last choice was taken by the AI and the buffer length is larger than 0,
        # then update Q table for all (state, action) in the sequence without reward
        if drone.buffer_length() != 0 and self.last_choice_index == 1 and drone == self.drone:
            # set the last choice made to none
            self.last_choice_index = None
            # set the future state for the final (state, action) in the sequence
            future_state = None
            # packets delivered when the drone moved directly to the depot
            delivered_packets = set([x.event_ref.identifier for x in drone.all_packets()])
            # empty the packets buffer
            drone.empty_buffer()

            for state, action_index, step in reversed(self.state_action_list):
                # packets that the drone had in a specific state
                pk_state = set(self.state_action_packets[state])
                # number of packets that the drone had in this state
                pk_state_num = len(pk_state)
                # delivered packets among those the drone had
                pk_state_delivered = pk_state.intersection(delivered_packets)
                # number of packets delivered among those the drone had
                pk_state_delivered_num = len(pk_state_delivered)
                # percentage of packets delivered
                delivered_packets_percent = (pk_state_delivered_num * 100) / pk_state_num
                # maximum time the drone would take to go and return to the depot from the farthest point of the map
                max_time = self.get_max_time_to_depot_and_return(drone)
                # call to the reward function
                reward = self.reward_function(delivered_packets_percent, max_time)
                # formula for updating the Q table
                self.q_dict[state][action_index] = self.q_dict[state][action_index] + self.alpha * (reward + self.gamma * max(self.q_dict[future_state]) - self.q_dict[state][action_index])
                # updating of the evaluation dictionary
                self.evaluation_dict[step] = reward
                # set the next future state
                future_state = state
            # set the final time to 0
            self.final_time_to_depot = 0
            # empty state action
            self.state_action_list = []
            # empty state action packets dict
            self.state_action_packets = {}

        # update of the list of the packet sorted by their generation
        if id_event in self.packet_set and drone == self.drone:
            self.packet_generation = [x for x in self.packet_generation if x[0].event_ref.identifier != id_event]
            self.packet_generation = sorted(self.packet_generation,key=lambda x: x[1])

        if id_event in self.taken_actions:
            # packet delivered
            if outcome == 0:
                self.packet_dictio[id_event]=True

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                        width_area=self.simulator.env_width,
                                                        x_pos=self.drone.coords[0],  # e.g. 1500
                                                        y_pos=self.drone.coords[1])[0]  # e.g. 500

        # update of the packet set and the list of current packets sorted by generation
        if pkd.event_ref.identifier not in self.packet_set:
            self.packet_set.add(pkd.event_ref.identifier)
            self.packet_generation.append((pkd,pkd.time_step_creation))
            self.packet_generation=sorted(self.packet_generation, key=lambda x: x[1])

        # if the simulation is about to end the drone takes its packets to the depot
        if self.is_time_to_goback():
            return -1

        # if a packet is expiring apply the reinforcement learning
        if self.packet_generation != [] and self.is_packet_expiring(self.packet_generation[0][0]) and self.last_choice_index != 1:
            # check if the drone has already taken an action in this sequence for the current state (cell)
            if cell_index not in [x[0] for x in self.state_action_list]:
                if cell_index not in self.q_dict.keys():
                    # choose a random action (action_index => index of the action in Q_table, 0 for None, 1 for -1)
                    action_index = random.choice([0, 1])
                    # add the new state to the dict
                    self.q_dict[cell_index] = [0, 0]
                # if the state already exists choose the best action in q_dict
                else:
                    is_random_choice = random.choices([True, False], weights=(10, 90), k=1)[0]
                    if is_random_choice or self.q_dict[cell_index][0] == self.q_dict[cell_index][1]:
                        action_index = random.choice([0, 1])
                    else:
                        action_index = self.q_dict[cell_index].index(max(self.q_dict[cell_index]))

                # add the new (state, action) tuple to be updated
                self.state_action_list.append((cell_index, action_index, self.simulator.cur_step))

                # store all the packets that the drone has when it takes an action
                self.state_action_packets[cell_index] = [x.event_ref.identifier for x in self.drone.all_packets()]

                if action_index == 1:
                    # store the time needed to go and return to depot from this point
                    self.final_time_to_depot = self.time_to_depot_and_return(self.drone)
                    self.last_choice_index = 1
                    return -1
                else:
                    self.last_choice_index = 0
                    return None

        # set of the first waypoint
        if self.first_waypoint==None:
            self.first_waypoint=self.drone.next_target()
        globalhistory=self.drone.waypoint_history
        localHistory = []
        # -------------- code used for determine at what point of the mission the drone is
        for point in reversed(globalhistory):
            localHistory.insert(0,point)
            self.set_waypoint.add(point)
            if point[0] == globalhistory[0][0] and point[1] == globalhistory[0][1]:
                if len(localHistory)<len(self.set_waypoint):
                    self.completed_lap=True
                break
        # --------------------------------------------------------------------------------
        # initialization of dictionary for the delivery of a packet
        if pkd.event_ref.identifier not in self.packet_dictio:
            self.packet_dictio[pkd.event_ref.identifier]=False

        # CODE OF THE BASE VERSION, to understand when the drone is at the end of the mission, and can come back
        # to the depot
        if self.drone_path == [] and self.drone.waypoint_history != []:
            if self.drone.waypoint_history[self.drone.current_waypoint - 1][1] < self.drone.waypoint_history[self.drone.current_waypoint - 2][1]:
                self.drone_path = self.drone.waypoint_history.copy()
        elif self.drone.next_target() in self.drone_path:
            self.counter+=1
            if self.drone_path.index(self.drone.next_target()) == 0 and self.drone.buffer_length() >= 3:
                return -1

        # code to determine which is the fastest drone and give it the packet
        if len(opt_neighbors) != 0:
            speed_drone_list = [(x[0].speed, x[0].src_drone) for x in opt_neighbors]
            max_speed_drone = max(speed_drone_list, key = lambda x:x[0])
            if max_speed_drone[0] <= self.drone.speed:
                return None
            else:
                return max_speed_drone[1]
        return None
    # reward -> { delivered_packets_percent = percentage of delivered packets,
    #             final_time_to_depot = time spent to go and return from depot,
    #             max_time = maximum time to go and return to depot from the farthest point of the map }
    def reward_function(self, delivered_packets_percent, max_time):
        return (delivered_packets_percent / 100) - self.alpha * (self.final_time_to_depot / max_time)
    # function to get the time that a drone would spend to go and return to depot from the farthest point of the map
    def get_max_time_to_depot_and_return(self, drone):
        return ((util.euclidean_distance(self.simulator.depot.coords, (0, 1500))) / drone.speed) * 2
    # function to get the time that a drone spend to go and return to depot from its current position
    def time_to_depot_and_return(self, drone):
        return (util.euclidean_distance(self.simulator.depot.coords, drone.coords) / drone.speed) * 2
    # function used to know if from the next target i have enough time to come back
    def arrival_time(self, drone):
        tot=(util.euclidean_distance(drone.next_target(), drone.coords) / drone.speed)+(util.euclidean_distance(drone.next_target(), self.simulator.depot.coords)/drone.speed)
        return tot
    # function used to knwo when the simulation is going to end , and its time to come back to the depot
    def is_time_to_goback(self):
        time_expected=self.arrival_time(self.drone)
        end_expected=self.simulator.len_simulation*self.simulator.time_step_duration-(self.simulator.cur_step*self.simulator.time_step_duration)
        return time_expected>end_expected
    # function that says if a pkd is expiring
    def is_packet_expiring(self,pkd):
        time_left=8000*self.simulator.time_step_duration-(self.simulator.cur_step*self.simulator.time_step_duration-pkd.time_step_creation*self.simulator.time_step_duration)
        expected_time=self.arrival_time(self.drone)
        return expected_time>time_left

    def print(self):
        pass