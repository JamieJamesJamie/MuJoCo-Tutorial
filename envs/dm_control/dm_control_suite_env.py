"""
Example of how to make a MuJoCo environment using the structure seen in
DeepMind's control suite.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from dm_control import mujoco
from dm_control.rl.control import Environment
from dm_control.suite.base import Task


def spider_env(
    xml_path: Path = Path("../../mujoco/spider.xml"),
    environment_kwargs: Optional[Dict[str, Any]] = None,
) -> Environment:
    """
    Returns the SpiderEnv task.

    :param:`task_params` should be a dictionary containing the arguments for
    initialising :class:`SpiderEnv`.

    :param xml_path: Path that spider.xml is located
    :param environment_kwargs: Environment key word arguments - See
        `dm_control.rl.control.Environment` for more details
    :return: The SpiderEnv environment
    """
    physics = Physics.from_xml_path(str(xml_path))
    task = SpiderEnv()

    environment_kwargs = environment_kwargs or {}
    return Environment(physics, task, **environment_kwargs)


class Physics(mujoco.Physics):
    """
    MuJoCo Physics object with some extra helper functions for this task.

    This class is completely optional for implementing an environment, but it's good
    to see extra functionality can be implemented.
    """

    def current_button_force(self):
        """
        Returns the force being applied to the button.

        :return: The force being applied to the button
        """
        return self.named.data.sensordata["button_touch_sensor"]

    def set_positions(self, positions):
        """
        Set each joint position in the MuJoCo environment.

        :param positions: The positions to set the joints to
        """
        for i, position in enumerate(positions):
            self.data.qpos[i] = position

    def set_velocities(self, velocities):
        """
        Set each joint velocity in the MuJoCo environment.

        :param velocities: The velocities to set the joints to
        """
        for i, velocity in enumerate(velocities):
            self.data.qvel[i] = velocity


class SpiderEnv(Task):
    """
    Spider environment for RL. The task is for the spider to move to the target button.

    The agent will get a sparse reward of 1.0 for stepping on the button.
    """

    def __init__(self):
        """
        Constructor for :class:`SpiderEnv`.
        """
        super().__init__()
        self._initial_positions = None
        self._initial_velocities = None
        self._is_first_episode = True

    def initialize_episode(self, physics: Physics):
        """
        Sets the state of the environment at the start of each episode.

        :param physics: A `dm_control.mujoco.Physics` instance that encapsulates the
            state of the MuJoCo model
        """

        # Record initial object positions / velocities
        # Useful for resetting future episodes
        if self._is_first_episode:
            self._initial_positions = physics.position()
            self._initial_velocities = physics.velocity()
            self._is_first_episode = False

        # Reset positions
        physics.set_positions(positions=self._initial_positions)

        # Reset velocities
        physics.set_velocities(velocities=self._initial_velocities)

    def get_observation(self, physics: Physics):
        """
        Returns an observation from the environment.

        The OrderedDict can include anything you want to include as an observation.
        E.g. we could have generated goal positions in `initialize_episode` then
        include them in the observation dict.

        :param physics: Instance of `Physics`
        :return: An observation from the environment
        """
        observation = OrderedDict()
        observation["state_positions"] = physics.position()
        observation["state_velocities"] = physics.velocity()

        return observation

    def get_reward(self, physics: Physics):
        """
        Returns a sparse reward from the environment.
        i.e. if the button is being pressed, return 1.0, otherwise return 0.0.

        :param physics: Instance of `Physics`
        :return: A reward from the environment.
        """
        return float(physics.current_button_force() > 0)


# Example of how the environment could be used
if __name__ == "__main__":
    # n_sub_steps = action repeat in MuJoCo between time steps
    env = spider_env(environment_kwargs={"n_sub_steps": 20})

    action_spec = env.action_spec()
    obs_spec = env.observation_spec()

    for episode in range(3):
        time_step = env.reset()

        for t in range(20):
            state_positions = time_step.observation["state_positions"]
            state_velocities = time_step.observation["state_velocities"]

            reward = time_step.reward

            # Image observation
            # camera_id can be a camera name defined in the XML file
            pixels = env.physics.render(
                camera_id=-1, height=240, width=320, depth=False
            )

            # Figure out an action...

            random_action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )

            time_step = env.step(random_action)
