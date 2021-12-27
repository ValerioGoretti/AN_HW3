import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
import random
import time

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        # dictionary for taken actions {id_pkd:[(action,timestamp),(action,timestamp)...]}
        self.taken_actions = {}  # id event : (old_action)
        # set containing all the packets
        self.packet_set = set()
        # list of the currently holded packets, sorted by their generation timestep
        self.packet_generation = []
        # dictionary for the Q table
        self.q_dict = {None: [0, 0, 0]}
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



    def feedback(self, drone, id_event, delay, outcome, depot_index=None):
        """ return a possible feedback, if the destination drone has received the packet """
        '''if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome,
                  " - to depot: ", depot_index)'''

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
                # TODO: set the correct depot (to do outside of for statement) set the farthest point of each depot
                # maximum time the drone would take to go and return to the depot from the farthest point of the map
                max_time = self.get_max_time_to_depot_and_return(drone, self.closest_depot(drone))
                print(self.drone.identifier, " ", max_time)
                print()
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
            print()
            print("drone: ", self.drone.identifier)
            print(self.q_dict)

        # update of the list of the packet sorted by their generation
        if id_event in self.packet_set and drone == self.drone:
            self.packet_generation = [x for x in self.packet_generation if x[0].event_ref.identifier != id_event]
            self.packet_generation = sorted(self.packet_generation,key=lambda x: x[1])



        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action = self.taken_actions[id_event]
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed
        #depot in self.simulator.depot.list_of_coords[0] --> (750, 0)
        #depot in self.simulator.depot.list_of_coords[1] --> (750, 1400)

        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500
        #print("Drone: ", self.drone.identifier, " - i-th cell:",  cell_index, " - center:", self.simulator.cell_to_center_coords[cell_index])

        # update of the packet set and the list of current packets sorted by generation
        if pkd.event_ref.identifier not in self.packet_set:
            self.packet_set.add(pkd.event_ref.identifier)
            self.packet_generation.append((pkd, pkd.time_step_creation))
            self.packet_generation = sorted(self.packet_generation, key=lambda x: x[1])

        # if the simulation is about to end the drone takes its packets to the depot
        if self.is_time_to_goback():
            dest_depot_action = -1 if self.closest_depot(self.drone) == (750, 0) else -2
            return dest_depot_action

        # if a packet is expiring apply the reinforcement learning
        if self.packet_generation != [] and self.is_packet_expiring(self.packet_generation[0][0]) and self.last_choice_index != 1:
            # check if the drone has already taken an action in this sequence for the current state (cell)
            if cell_index not in [x[0] for x in self.state_action_list]:
                if cell_index not in self.q_dict.keys():
                    # choose a random action (action_index => index of the action in Q_table, 0 for None, 1 for -1)
                    action_index = random.choice([0, 1])
                    # add the new state to the dict
                    self.q_dict[cell_index] = [0, 0, 0]
                # if the state already exists choose the best action in q_dict
                else:
                    is_random_choice = random.choices([True, False], weights=(10, 90), k=1)[0]
                    if is_random_choice or self.q_dict[cell_index][0] == self.q_dict[cell_index][1] == self.q_dict[cell_index][1]:
                        # make a random choice, 0 for keeping the packets, 1 for coming back to depot
                        action_index = random.choice([0, 1])
                        if action_index == 1:
                            # randomly chooses which depot to go to
                            action_index = random.choice([1, 2])
                    else:
                        # make the best choice among those available
                        action_index = self.q_dict[cell_index].index(max(self.q_dict[cell_index]))

                # add the new (state, action) tuple to be updated
                self.state_action_list.append((cell_index, action_index, self.simulator.cur_step))

                # store all the packets that the drone has when it takes an action
                self.state_action_packets[cell_index] = [x.event_ref.identifier for x in self.drone.all_packets()]

                if action_index == 1 or action_index == 2:
                    # set the index of the last choice made, 1 for coming back to one depot
                    self.last_choice_index = 1
                    # set the destination depot
                    destination_depot_action = -1 if action_index == 1 else -2
                    # set the destination depot coords
                    destination_depot_coords = (750, 0) if action_index == 1 else (750, 1400)
                    # store the time needed to go and return to depot from the current point
                    self.final_time_to_depot = self.time_to_depot_and_return(self.drone, destination_depot_coords)
                    return destination_depot_action
                else:
                    # set the index of the last choice made, 0 for keeping the packet
                    self.last_choice_index = 0
                    return None


        action = None

        # self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
        # self.drone.residual_energy (that tells us when I'll come back to the depot).
        #  .....
        for hpk, drone_instance in opt_neighbors:
            #print(hpk)
            continue

        # Store your current action --- you can add several stuff if needed to take a reward later
        self.taken_actions[pkd.event_ref.identifier] = (action)

        # return action:
        # None --> no transmission
        # -1 --> move to first depot (self.simulator.depot.list_of_coords[0]
        # -2 --> move to second depot (self.simulator.depot.list_of_coords[1]
        # 0, ... , self.ndrones --> send packet to this drone
        return None  # here you should return a drone object!


    # function to get the avaiable actions index
    def get_actions(self, neighbors):
        action_list = [drone for hpk, drone in neighbors]
        action_list.append(None)
        return action_list

    # function used to know when the simulation is going to end , and its time to come back to the depot
    def is_time_to_goback(self):
        time_expected = self.arrival_time(self.drone, self.closest_depot(self.drone))
        end_expected = self.simulator.len_simulation * self.simulator.time_step_duration - (
                    self.simulator.cur_step * self.simulator.time_step_duration)
        return time_expected > end_expected

    # function used to know if from the next target i have enough time to come back
    def arrival_time(self, drone, depot_coords):
        tot = (util.euclidean_distance(drone.next_target(), drone.coords) / drone.speed) + (
                    util.euclidean_distance(drone.next_target(), depot_coords) / drone.speed)
        return tot

    # function that return the closest depot from the drone position
    def closest_depot(self, drone):
        if (util.euclidean_distance(self.simulator.depot.list_of_coords[0], drone.coords) <
                util.euclidean_distance(self.simulator.depot.list_of_coords[1], drone.coords)):
            return self.simulator.depot.list_of_coords[0]  # which depot to refer, based on my position
        else:
            return self.simulator.depot.list_of_coords[1]  # which depot to refer, based on my position

    # function to get the time that a drone would spend to go and return to depot from the farthest point of the map
    def get_max_time_to_depot_and_return(self, drone, depot_coords):
        return ((util.euclidean_distance(depot_coords, (0, 1500))) / drone.speed) * 2

    # function to get the time that a drone spend to go and return to depot from its current position
    def time_to_depot_and_return(self, drone, depot_coords):
        return (util.euclidean_distance(depot_coords, drone.coords) / drone.speed) * 2

    # function that says if a pkd is expiring
    def is_packet_expiring(self, pkd):
        time_left = 8000 * self.simulator.time_step_duration - (self.simulator.cur_step * self.simulator.time_step_duration - pkd.time_step_creation * self.simulator.time_step_duration)
        expected_time_depot1 = self.arrival_time(self.drone, (750, 0))
        expected_time_depot2 = self.arrival_time(self.drone, (750, 1400))
        return expected_time_depot1 > time_left or expected_time_depot2 > time_left

    # reward -> { delivered_packets_percent = percentage of delivered packets,
    #             final_time_to_depot = time spent to go and return from depot,
    #             max_time = maximum time to go and return to depot from the farthest point of the map }
    def reward_function(self, delivered_packets_percent, max_time):
        return (delivered_packets_percent / 100) - self.alpha * (self.final_time_to_depot / max_time)

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass
