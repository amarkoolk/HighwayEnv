from typing import Dict, Text

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
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 2,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "npc_spacing": 1,
            "ego_speed_range" : [-40, 80],
            "npc_speed_range" : [-40, 80],
            "vehicles_density": 1,
            "collision_reward": 1,    # The reward received when colliding with a vehicle.
            "ttc_reward": [0,1],  # The reward range for time to collision with the ego vehicle.
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                speed_range = self.config["npc_speed_range"]
            )
            vehicle = self.action_type.vehicle_class(road = self.road, \
                position = vehicle.position, \
                heading = vehicle.heading, \
                speed = vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1, speed_range = self.config["npc_speed_range"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster colliding with the other vehicle.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        dx = self.road.vehicles[0].position[0] - self.road.vehicles[1].position[0]
        dy = self.road.vehicles[1].position[1] - self.road.vehicles[1].position[1]
        vx0 = (np.cos(self.road.vehicles[0].heading))*self.road.vehicles[0].speed
        vx1 = (np.cos(self.road.vehicles[1].heading))*self.road.vehicles[1].speed
        vy0 = (np.sin(self.road.vehicles[0].heading))*self.road.vehicles[0].speed
        vy1 = (np.sin(self.road.vehicles[1].heading))*self.road.vehicles[1].speed
        ttc_x = np.abs(dx)/np.abs(vx0 - vx1)
        ttc_y = np.abs(dy)/np.abs(vy0 - vy1)
        r_x = -1.0/(1.0 + np.exp(4-ttc_x)) + 1.0
        r_y = -1.0/(1.0 + np.exp(4-ttc_y)) + 1.0
        print('{} {} {} {}'.format(vx0, vx1, vy0, vy1))
        print('ttcx:{} ttcy:{} rx:{} ry:{}'.format(ttc_x, ttc_y, r_x, r_y))
        # print(bool(self.vehicle.crashed))

        return {
            "collision_reward": float(self.vehicle.crashed)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
