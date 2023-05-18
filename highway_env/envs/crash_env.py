from typing import Dict, Text, Optional

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class CrashEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 2,
            "ego_vehicles": 1,
            "npc_vehicles" : 1,
            "initial_lane_id": None,
            "initial_speed" : 25,
            "duration": 40,  # [s]
            "mean_distance": 20,
            "mean_delta_v": 0,
            "vehicles_density": 1,
            "collision_reward": 1,    # The reward received when colliding with a vehicle.
            "ttc_x_reward": 1,  # The reward range for time to collision in the x direction with the ego vehicle.
            "ttc_y_reward": 1,  # The reward range for time to collision in the y direction with the ego vehicle.
            "normalize_reward": False,
            "offroad_terminal": False,
            "tolerance" : 1e-3,
            "spawn_configs" : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center']
        })
        return config

    def _reset(self) -> None:
        self.ttc_x = 0
        self.ttc_y = 0
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        spawn_configs = self.config["spawn_configs"]
        self.spawn_config = self.np_random.choice(spawn_configs)
        spawn_distance = self.np_random.normal(self.config["mean_distance"], self.config["mean_distance"]/10)
        starting_vel_offset = self.np_random.normal(self.config["mean_delta_v"], 5)
        self.controlled_vehicles = []
        for _ in range(self.config["ego_vehicles"]):
            # Behind Left
            if self.spawn_config == 'behind_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)

                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'behind_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'behind_center':
                lane1 = self.np_random.choice(self.road.network.graph['0']['1'])
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = lane1
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(spawn_distance, 0), \
                                            heading = lane2.heading_at(spawn_distance), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'adjacent_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'adjacent_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(0, 0), \
                                                        heading = lane1.heading_at(0), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'forward_left':
                lane1 = self.road.network.graph['0']['1'][0]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][1]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'forward_right':
                lane1 = self.road.network.graph['0']['1'][1]
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = self.road.network.graph['0']['1'][0]
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)
            elif self.spawn_config == 'forward_center':
                lane1 = self.np_random.choice(self.road.network.graph['0']['1'])
                vehicle = self.action_type.vehicle_class(road = self.road, \
                                                        position = lane1.position(spawn_distance, 0), \
                                                        heading = lane1.heading_at(spawn_distance), \
                                                        speed = self.config["initial_speed"]+starting_vel_offset)
                
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

                lane2 = lane1
                vehicle = other_vehicles_type(road = self.road, \
                                            position = lane2.position(0, 0), \
                                            heading = lane2.heading_at(0), \
                                            speed = self.config["initial_speed"]+starting_vel_offset)
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster colliding with the other vehicle.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [-2,self.config["collision_reward"] + self.config["ttc_x_reward"] + self.config["ttc_y_reward"]], [0, 1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        
        dx = self.road.vehicles[1].position[0] - self.road.vehicles[0].position[0]
        dy = self.road.vehicles[1].position[1] - self.road.vehicles[0].position[1]
        vx0 = (np.cos(self.road.vehicles[0].heading))*self.road.vehicles[0].speed
        vx1 = (np.cos(self.road.vehicles[1].heading))*self.road.vehicles[1].speed
        vy0 = (np.sin(self.road.vehicles[0].heading))*self.road.vehicles[0].speed
        vy1 = (np.sin(self.road.vehicles[1].heading))*self.road.vehicles[1].speed

        dvx = vx1 - vx0
        dvy = vy1 - vy0

        self.ttc_x = dx/dvx
        self.ttc_y = dy/dvy


        # Calculate Rewards
        if abs(dvx) < self.config["tolerance"]:
            if abs(dx) < self.config["tolerance"]:
                r_x = self.config['ttc_x_reward']
            else:
                r_x = 0
        else:
            r_x = 1.0/(1.0 + np.exp(-4-0.1*self.ttc_x)) if self.ttc_x <= 0 else -1.0/(1.0 + np.exp(4-0.1*self.ttc_x))
        
        if abs(dvy) < self.config["tolerance"]:
            if abs(dy) < self.config["tolerance"]:
                r_y = self.config['ttc_y_reward']
            else:
                r_y = 0
        else:
            r_y = 1.0/(1.0 + np.exp(-4-0.1*self.ttc_y)) if self.ttc_y <= 0 else -1.0/(1.0 + np.exp(4-0.1*self.ttc_y))
        
        return {
            "collision_reward": float(self.vehicle.crashed),
            "ttc_x_reward": r_x,
            "ttc_y_reward": r_y
        }
    
    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "ttc_x": self.ttc_x,
            "ttc_y": self.ttc_y
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
