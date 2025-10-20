from enum import Enum

import numpy as np


class Action(Enum):
    LANE_LEFT = "LANE_LEFT"
    IDLE = "IDLE"
    LANE_RIGHT = "LANE_RIGHT"
    FASTER = "FASTER"
    SLOWER = "SLOWER"


class Scenario:
    def __init__(self, use_spawn_distribution: bool = False):
        self.state = None
        self.end_frames = 0
        self.prev_action = Action.IDLE.value
        self.spawn_configs = []
        self.use_spawn_distribution = use_spawn_distribution

    def set_spawn_distriubtion(self, use_spawn_distribution: bool = False):
        self.use_spawn_distribution = use_spawn_distribution

    def set_state(self, ego_state: np.ndarray, npc_state: np.ndarray):

        self.ego_x = ego_state[0]
        self.ego_y = ego_state[1]
        self.npc_x = npc_state[0]
        self.npc_y = npc_state[1]

    def reset(self, test=False):
        """
        Common reset logic across all scenarios.
        """
        self.state = None
        self.end_frames = 0
        self.prev_action = Action.IDLE.value
        self.set_spawn_distriubtion(use_spawn_distribution=not test)

    def get_action(self):
        """
        Should be overridden by each scenario with scenario-specific logic.
        """
        raise NotImplementedError

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        raise NotImplementedError


#
# Child classes
#


class IdleSlower(Scenario):
    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)

    def get_action(self):
        return Action.IDLE.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = -5.0
        config["spawn_configs"] = ["forward_right"]


class IdleFaster(Scenario):
    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)

    def get_action(self):
        return Action.IDLE.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 5.0
        config["spawn_configs"] = ["behind_left"]


class Slowdown(Scenario):
    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)

    def get_action(self):
        return Action.SLOWER.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 0.0
        config["spawn_configs"] = ["forward_left"]


class SlowdownSameLane(Scenario):
    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)

    def get_action(self):
        return Action.SLOWER.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 0.0
        config["spawn_configs"] = ["forward_center"]


class SpeedUp(Scenario):
    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)

    def get_action(self):
        return Action.FASTER.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 0.0
        config["spawn_configs"] = ["behind_left"]


class CutIn(Scenario):
    class CutInManeuver(Enum):
        ACCELERATE = 0
        CUTIN = 1
        END = 2

    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)
        self.current_maneuver = self.CutInManeuver.ACCELERATE
        self.maneuver_counter = 0
        self.spawn_configs = ["behind_left"]

    def reset(self, test=False):
        """
        Override reset if scenario-specific attributes need re-initialization.
        Still call super() for the common steps.
        """
        super().reset(test)
        self.current_maneuver = self.CutInManeuver.ACCELERATE
        self.maneuver_counter = 0

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 0.0
        config["spawn_configs"] = ["behind_left"]

    def get_action(self):
        action = self.prev_action
        self.maneuver_counter += 1

        # Hard-coded behavior based on maneuver phase and counter
        if self.current_maneuver == self.CutInManeuver.ACCELERATE:
            # Accelerate for a fixed number of frames
            action = Action.FASTER.value
            if self.maneuver_counter >= 10:  # Accelerate for 10 frames
                self.current_maneuver = self.CutInManeuver.CUTIN
                self.maneuver_counter = 0

        elif self.current_maneuver == self.CutInManeuver.CUTIN:
            # Execute the cut-in maneuver
            action = Action.LANE_RIGHT.value  # Cut back into the original lane
            if self.maneuver_counter >= 3:  # Complete cut-in in 3 frames
                self.current_maneuver = self.CutInManeuver.END
                self.maneuver_counter = 0

        elif self.current_maneuver == self.CutInManeuver.END:
            self.end_frames += 1
            action = Action.IDLE.value

        self.prev_action = action
        return action


class CutInSlowDown(Scenario):
    class CutInManeuver(Enum):
        WAIT = 0
        CUTIN = 1
        BRAKE = 2
        END = 3

    def __init__(self, use_spawn_distribution: bool = False):
        super().__init__(use_spawn_distribution)
        self.maneuver_counter = 0
        self.current_maneuver = self.CutInManeuver.WAIT
        self.prev_action = Action.IDLE.value
        self.spawn_configs = ["forward_left"]

    def reset(self, test=False):
        """
        Override reset if scenario-specific attributes need re-initialization.
        Still call super() for the common steps.
        """
        super().reset(test)
        self.maneuver_counter = 0
        self.prev_action = Action.IDLE.value
        self.current_maneuver = self.CutInManeuver.WAIT

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config["adversarial"] = False
        config["use_spawn_distribution"] = self.use_spawn_distribution
        config["mean_delta_v"] = 0.0
        config["mean_distance"] = 20.0
        config["spawn_configs"] = ["forward_left"]

    def get_action(self):
        action = self.prev_action
        self.maneuver_counter += 1

        if self.current_maneuver == self.CutInManeuver.WAIT:
            # Wait for a few frames before starting the maneuver
            action = Action.IDLE.value
            if self.maneuver_counter >= 3:
                self.current_maneuver = self.CutInManeuver.CUTIN
                self.maneuver_counter = 0

        elif self.current_maneuver == self.CutInManeuver.CUTIN:
            # Execute the cut-in maneuver
            action = Action.LANE_RIGHT.value
            if self.maneuver_counter >= 3:
                self.current_maneuver = self.CutInManeuver.BRAKE
                self.maneuver_counter = 0

        elif self.current_maneuver == self.CutInManeuver.BRAKE:
            # Brake for a fixed number of frames
            action = Action.SLOWER.value
            if self.maneuver_counter >= 5:
                self.current_maneuver = self.CutInManeuver.END
                self.maneuver_counter = 0

        elif self.current_maneuver == self.CutInManeuver.END:
            # Maintain idle at the end
            action = Action.IDLE.value

        self.prev_action = action
        return action
