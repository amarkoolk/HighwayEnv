from typing import Dict, Text, Optional

import numpy as np
import copy
import math

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle

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
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "initial_speed" : 20,
            "duration": 40,  # [s]
            "mean_distance": 20,
            "mean_delta_v": 0,
            "ego_spacing": 2,
            "use_spawn_distribution": True,
            "vehicles_density": 1,
            "collision_reward": 1,    # The reward received when colliding with a vehicle.
            "ttc_x_reward": 1,  # The reward range for time to collision in the x direction with the ego vehicle.
            "ttc_y_reward": 1,  # The reward range for time to collision in the y direction with the ego vehicle.
            "normalize_reward": False,
            "offroad_terminal": False,
            "tolerance" : 1e-3,
            "spawn_configs" : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center'],
            "adversarial": False,
            "reward_speed_range": [20, 30],
            "use_mobil": False,
            "ego_vs_mobil" : False,
            "scenarios": None,
            "scenario_vehicles_type": "highway_env.vehicle.behavior.ScenarioVehicle",
            "mobil_politeness": 1.0,
            "trial_episodes": 1,
            "sample_reward_signal": False,
            "reward_speed_lower": 20,
            "reward_speed_upper": 30,
            "only_resample_on_failure": False
        })
        return config
    
    def trial_sample(self) -> None:
        self.config["lanes_count"] = self.np_random.choice(range(2, 5))
        self.config["vehicles_count"] = self.np_random.choice(range(1, 5))
        self.config["mobil_politeness"] = self.np_random.uniform(0.0, 1.0)

        if(self.config["sample_reward_signal"]):
            self.config["reward_speed_lower"] = self.np_random.uniform(15, 25)
            self.config["reward_speed_upper"] = self.np_random.uniform(self.config["reward_speed_lower"], 35)
            self.config["reward_speed_range"] = [self.config["reward_speed_lower"], self.config["reward_speed_upper"]]

        print(f"Resampling with lanes_count={self.config['lanes_count']}, vehicles_count={self.config['vehicles_count']}, mobil_politeness={self.config['mobil_politeness']} reward_speed_range={self.config['reward_speed_range']}")

    def _reset(self) -> None:
        self.dx = 0
        self.dy = 0
        self.dvx = 0
        self.dvy = 0
        self.ttc_x = 0
        self.ttc_y = 0
        self.last_ep_crash = False
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            self.last_ep_crash = self.vehicle.crashed
        if self.config['multi_car']:
            if self.config["only_resample_on_failure"]:
                if self.last_ep_crash:
                    self.trial_sample()
            elif self.episode_num % self.config["trial_episodes"] == 0:
                self.trial_sample()
        self._create_road()
        if self.config["scenarios"]:
            self.scenario_spawn()
        if self.config['controlled_vehicles'] == 1:
            if self.config['multi_car']:
                self.multi_car_spawn()
            else:
                self.single_controlled_vehicle_spawn()
        elif self.config['controlled_vehicles'] == 2:
            self.dual_controlled_vehicle_spawn()

        self.episode_num += 1

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
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def create_vehicle(self, vehicle_class, lane, spawn_distance, starting_vel_offset, color = None, scenario = None, randomize = False):
        if scenario:
            vehicle = vehicle_class(
                road=self.road,
                position=lane.position(spawn_distance, 0),
                heading=lane.heading_at(spawn_distance),
                speed=self.config["initial_speed"] + starting_vel_offset,
                target_speed = self.config["initial_speed"] + starting_vel_offset,
                color = color,
                scenario = scenario,
                min_speed = np.min(self.config['action']['action_config']['target_speeds']),
                max_speed = np.max(self.config['action']['action_config']['target_speeds'])
            )
        else:
            vehicle = vehicle_class(
                road=self.road,
                position=lane.position(spawn_distance, 0),
                heading=lane.heading_at(spawn_distance),
                speed=self.config["initial_speed"] + starting_vel_offset,
                target_speed = self.config["initial_speed"] + starting_vel_offset,
                color = color
            )
            if randomize and isinstance(vehicle_class, IDMVehicle):
                vehicle.set_politeness(self.config["mobil_politeness"])
                vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        try:
            if vehicle_class.func == self.action_type.vehicle_class.func:
                self.controlled_vehicles.append(vehicle)
        except AttributeError:
            pass

    def scenario_spawn(self):
        self.scenario_type = self.np_random.choice(self.config["scenarios"])
        if self.scenario_type == "IdleFaster":
            self.config["spawn_configs"] = ['forward_right']
            self.config["mean_delta_v"] = -5.0
        if self.scenario_type == "IdleSlower":
            self.config["spawn_configs"] = ['behind_left']
            self.config["mean_delta_v"] = 5.0
        if self.scenario_type == "CutIn":
            self.config["spawn_configs"] = ['forward_right']
            self.config["mean_delta_v"] = 0.0
        if self.scenario_type == "CutInSlowDown":
            self.config["spawn_configs"] = ['behind_right']
            self.config["mean_delta_v"] = 0.0

    def dual_controlled_vehicle_spawn(self):
        spawn_configs = self.config["spawn_configs"]
        self.spawn_config = self.np_random.choice(spawn_configs)
        if self.config["use_spawn_distribution"]:
            spawn_distance = self.np_random.normal(self.config["mean_distance"], self.config["mean_distance"] / 10)
            starting_vel_offset = self.np_random.normal(self.config["mean_delta_v"], 5)
        else:
            spawn_distance = self.config["mean_distance"]
            starting_vel_offset = self.config["mean_delta_v"]
        self.controlled_vehicles = []

        lanes = self.road.network.graph['0']['1']
        lane_configurations = {
            'behind_left': [0, 1],
            'behind_right': [1, 0],
            'behind_center': 'random',
            'adjacent_left': [0, 1],
            'adjacent_right': [1, 0],
            'forward_left': [0, 1],
            'forward_right': [1, 0],
            'forward_center': 'random'
        }

        lane_indices = lane_configurations[self.spawn_config]
        if lane_indices == 'random':
            lane1 = self.np_random.choice(lanes)
            lane2 = lane1  # For the center configurations, both vehicles are in the same lane
        else:
            lane1, lane2 = [lanes[idx] for idx in lane_indices]

        spawn_distance1 = 0 if self.spawn_config in ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right'] else spawn_distance
        spawn_distance2 = spawn_distance if self.spawn_config in ['behind_left', 'behind_right', 'behind_center'] else 0
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])


        if self.config["use_mobil"]:
            if self.config["ego_vs_mobil"]:
                # Create Mobil Vehicle
                self.create_vehicle(other_vehicles_type, lane1, spawn_distance1, starting_vel_offset, color = (100, 200, 255))
                # Create Controlled Vehicle
                self.create_vehicle(self.action_type.vehicle_class, lane2, spawn_distance2, 0)
            else:
                # Create Mobil Vehicle
                self.create_vehicle(other_vehicles_type, lane2, spawn_distance2, starting_vel_offset, color = (100, 200, 255))
                # Create Controlled Vehicle
                self.create_vehicle(self.action_type.vehicle_class, lane1, spawn_distance1, 0, color = (255, 0, 0))
        else:
            # Create Ego and NPC Vehicles
            self.create_vehicle(self.action_type.vehicle_class, lane2, spawn_distance2, 0)
            self.create_vehicle(self.action_type.vehicle_class, lane1, spawn_distance1, starting_vel_offset, color = (255, 0, 0))



    def single_controlled_vehicle_spawn(self):
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        spawn_configs = self.config["spawn_configs"]
        self.spawn_config = self.np_random.choice(spawn_configs)
        if self.config["use_spawn_distribution"]:
            spawn_distance = self.np_random.normal(self.config["mean_distance"], self.config["mean_distance"] / 10)
            starting_vel_offset = self.np_random.normal(self.config["mean_delta_v"], 5)
        else:
            spawn_distance = self.config["mean_distance"]
            starting_vel_offset = self.config["mean_delta_v"]
        self.controlled_vehicles = []


        lanes = self.road.network.graph['0']['1']
        lane_configurations = {
            'behind_left': [0, 1],
            'behind_right': [1, 0],
            'behind_center': 'random',
            'adjacent_left': [0, 1],
            'adjacent_right': [1, 0],
            'forward_left': [0, 1],
            'forward_right': [1, 0],
            'forward_center': 'random',
        }

        lane_indices = lane_configurations[self.spawn_config]
        if lane_indices == 'random':
            lane1 = self.np_random.choice(lanes)
            lane2 = lane1  # For the center configurations, both vehicles are in the same lane
        else:
            lane1, lane2 = [lanes[idx] for idx in lane_indices]

        spawn_distance1 = 0 if self.spawn_config in ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right'] else spawn_distance
        spawn_distance2 = spawn_distance if self.spawn_config in ['behind_left', 'behind_right', 'behind_center'] else 0

        # Create controlled vehicle
        self.create_vehicle(self.action_type.vehicle_class, lane1, spawn_distance1, starting_vel_offset)

        # Create other vehicle
        starting_speed = self.np_random.choice(self.config['action']['action_config']['target_speeds']) - self.config["initial_speed"]
        if self.config["scenarios"]:
            scenario_type = utils.class_from_path(self.config["scenario_vehicles_type"])
            self.create_vehicle(scenario_type, lane2, spawn_distance2, 0, color=(100,100,0), scenario = self.scenario_type)
        else:
            self.create_vehicle(other_vehicles_type, lane2, spawn_distance2, 0)

    def multi_car_spawn(self):
        # First Check Lane Count
        num_lanes = self.config['lanes_count']
        if num_lanes < 2:
            raise ValueError("At least 2 lanes are required for multi-car spawn.")

        if self.config["use_spawn_distribution"]:
            ego_dist = self.np_random.normal(100.0, 10.0)
            starting_vel_offset = self.np_random.normal(self.config["mean_delta_v"], 5)
        else:
            ego_dist = 100.0
            starting_vel_offset = self.config["mean_delta_v"]

        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            # Choose Random Lane for Ego Spawn
            ego_lane_idx = self.np_random.choice(range(self.config['lanes_count']))
            ego_lane = self.road.network.get_lane(('0', '1', ego_lane_idx))
            self.create_vehicle(self.action_type.vehicle_class, ego_lane, ego_dist, starting_vel_offset)

        occupancy_grid = np.zeros((self.config["lanes_count"], 3))
        starting_dist = np.zeros((self.config["lanes_count"], 3))
        occupancy_grid[ego_lane_idx, 1] = 1  # Mark ego vehicle's lane as occupied
        starting_dist[ego_lane_idx, 1] = 100.0
        self.spawn_config = ego_lane_idx
        
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for idx in range(self.config["vehicles_count"]):

            # Find Spot in Occupancy Grid - rows where sum is < 3
            free_lanes = np.where(np.sum(occupancy_grid, axis=1) < 3)
            lane_choice = self.np_random.choice(free_lanes[0]) if free_lanes[0].size > 0 else None

            # Pick Empty Column
            empty_columns = np.where(occupancy_grid[lane_choice] == 0)[0]
            column_choice = self.np_random.choice(empty_columns) if empty_columns.size > 0 else None

            occupancy_grid[lane_choice, column_choice] = 1
            
            # Calculate Starting Distance - Offset from row in front by 20 m
            if self.config["use_spawn_distribution"]:
                spawn_distance = self.np_random.normal(self.config["mean_distance"], self.config["mean_distance"] / 10)
                starting_vel_offset = self.np_random.normal(self.config["mean_delta_v"], 5)
            starting_dist[lane_choice, column_choice] = 100.0 + spawn_distance * (column_choice - 1)

            lane = self.road.network.get_lane(('0', '1', lane_choice))
            self.create_vehicle(other_vehicles_type, lane, starting_dist[lane_choice, column_choice], starting_vel_offset, randomize=True)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster colliding with the other vehicle.
        :param action: the last action performed
        :return: the corresponding reward
        """

        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["adversarial"]:

            if self.config["normalize_reward"]:
                reward = utils.lmap(reward, [-(self.config["ttc_x_reward"] + self.config["ttc_y_reward"]),self.config["collision_reward"] + self.config["ttc_x_reward"] + self.config["ttc_y_reward"]], [0, 1])

            return reward
        else:
            if self.config["normalize_reward"]:
                reward = utils.lmap(reward,
                                    [self.config["collision_reward"],
                                    self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                    [0, 1])
            reward *= rewards['on_road_reward']
            return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268

        ego_vehicle = self.road.vehicles[0]
        npc_vehicle = self.road.vehicles[1]

        dx = npc_vehicle.position[0] - ego_vehicle.position[0]
        dy = npc_vehicle.position[1] - ego_vehicle.position[1]

        vx0 = (math.cos(ego_vehicle.heading))*ego_vehicle.speed
        vx1 = (math.cos(npc_vehicle.heading))*npc_vehicle.speed
        vy0 = (math.sin(ego_vehicle.heading))*ego_vehicle.speed
        vy1 = (math.sin(npc_vehicle.heading))*npc_vehicle.speed


        dvx = vx1 - vx0
        dvy = vy1 - vy0

        ttc_x = dx/dvx if abs(dvx) > 1e-6 else dx/1e-6
        ttc_y = dy/dvy if abs(dvy) > 1e-6 else dy/1e-6

        self.ttc_x = ttc_x
        self.ttc_y = ttc_y



        # Calculate Rewards
        if abs(dvx) < self.unwrapped.config["tolerance"]:
            if abs(dx) < self.unwrapped.config["tolerance"]:
                r_x = 1.0
            else:
                r_x = 0
        else:
            try:
                r_x = 1.0/(1.0 + math.exp(-4-0.1*ttc_x)) if ttc_x <= 0 else -1.0/(1.0 + math.exp(4-0.1*ttc_x))
            except OverflowError:
                r_x = 0.0

        if abs(dvy) < self.unwrapped.config["tolerance"]:
            if abs(dy) < self.unwrapped.config["tolerance"]:
                r_y = 1.0
            else:
                r_y = 0
        else:
            try:
                r_y = 1.0/(1.0 + math.exp(-4-0.1*ttc_y)) if ttc_y <= 0 else -1.0/(1.0 + math.exp(4-0.1*ttc_y))
            except OverflowError:
                r_y = 0.0

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        if scaled_speed < 0:
            scaled_speed = 0
        if scaled_speed > 1:
            scaled_speed = 0

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1), # TODO: Don't reward speeding past max of 1
            "on_road_reward": float(self.vehicle.on_road),
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
            "spawn_config": self.spawn_config,
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "ttc_x": self.ttc_x,
            "ttc_y": self.ttc_y,
            'dx': self.dx,
            'dy': self.dy,
            'dvx': self.dvx,
            'dvy': self.dvy
            # "vehicle_states": self.get_vehicle_states()
        }
        try:
            rewards = self._rewards(action)
            for name, reward in rewards.items():
                if "rewards" not in info:
                    info["rewards"] = {}
                info["rewards"][f"{name}"] = reward * self.config.get(name, 0)

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

    def get_vehicle_states(self) -> np.ndarray:

        """
        Get the states of the vehicles in the environment.
        :return: A numpy array of shape (num_vehicles, 6) where each row represents a vehicle's state
                 (x, y, vx, vy, heading, lane_id).
        """
        vehicle_states = []
        for vehicle in self.road.vehicles:
            if isinstance(vehicle, Vehicle):
                state = np.array([vehicle.position[0], vehicle.position[1], vehicle.speed * np.cos(vehicle.heading), vehicle.speed * np.sin(vehicle.heading), vehicle.heading, vehicle.lane_index[1]])
                vehicle_states.append(state)

        return np.array(vehicle_states)
