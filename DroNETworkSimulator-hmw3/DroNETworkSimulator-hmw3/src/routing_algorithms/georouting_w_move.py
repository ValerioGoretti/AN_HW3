
import numpy as np
import src.utilities.utilities as util

from src.routing_algorithms.BASE_routing import BASE_routing

class GeoMoveRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        if (util.euclidean_distance(self.simulator.depot.list_of_coords[0], self.drone.coords) <
                util.euclidean_distance(self.simulator.depot.list_of_coords[1], self.drone.coords)):
            depot_coords = self.simulator.depot.list_of_coords[0]  # which depot to refer, based on my position
        else:
            depot_coords = self.simulator.depot.list_of_coords[1]  # which depot to refer, based on my position

        best_drone_distance_from_depot = util.euclidean_distance(depot_coords, self.drone.coords)
        best_drone = None

        # Action MOVE is identified by -1
        if len(opt_neighbors) == 0:
            if depot_coords == self.simulator.depot.list_of_coords[0]:
                return -1
            else:
                return -2

        for hpk, drone_istance in opt_neighbors:
            exp_position = hpk.cur_pos  # without estimation, a simple geographic approach
            exp_distance = util.euclidean_distance(exp_position, depot_coords)
            if exp_distance < best_drone_distance_from_depot:
                best_drone_distance_from_depot = exp_distance
                best_drone = drone_istance

        return best_drone


    def __estimated_neighbor_drone_position(self, hello_message):
        """ estimate the current position of the drone """

        # get known info about the neighbor drone
        hello_message_time = hello_message.time_step_creation
        known_position = hello_message.cur_pos
        known_speed = hello_message.speed
        known_next_target = hello_message.next_target

        # compute the time elapsed since the message sent and now
        # elapsed_time in seconds = elapsed_time in steps * step_duration_in_seconds
        elapsed_time = (self.simulator.cur_step - hello_message_time) * self.simulator.time_step_duration  # seconds

        # distance traveled by drone
        distance_traveled = elapsed_time * known_speed

        # direction vector
        a, b = np.asarray(known_position), np.asarray(known_next_target)
        v_ = (b - a) / np.linalg.norm(b - a)

        # compute the expect position
        c = a + (distance_traveled * v_)

        return tuple(c)