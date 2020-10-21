"""
Example of how to make a MuJoCo environment using the Gym library.
"""

from pathlib import Path

from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle


class SpiderEnv(MujocoEnv, EzPickle):
    """
    Spider environment for RL. The task is for the spider to move to the target button.

    The agent will get a sparse reward of 1.0 for stepping on the button.
    """

    def __init__(self, action_repeat=1):
        """
        Constructor for :class:`SpiderEnv`.

        :param action_repeat: Number of times action should be repeated in MuJoCo
            between each RL time step
        """
        EzPickle.__init__(self)

        self._has_button_been_pressed_before = False

        MujocoEnv.__init__(
            self,
            str(Path("../../mujoco/spider.xml").resolve()),
            frame_skip=action_repeat,
        )

    def reset_model(self):
        """
        Reset the spider's degrees of freedom:

        - qpos (joint positions); and
        - qvel (joint velocities)
        """
        self.set_state(self.init_qpos, self.init_qvel)
        self._has_button_been_pressed_before = False

        return self.state_vector()

    def step(self, _action):
        """
        Accepts an :param:`_action`, advances the environment by a single RL time step,
        and returns a tuple (observation, reward, done, info).

        :param _action: An act provided by the RL agent
        :return: A tuple containing an observation, a reward, whether the episode has
            ended, and auxiliary information
        """

        self.do_simulation(_action, self.frame_skip)

        _observation = self.state_vector()
        _reward = self._reward()

        _done = self._has_button_been_pressed_before or self._is_button_pressed()

        if not self._has_button_been_pressed_before and _done:
            self._has_button_been_pressed_before = True

        return _observation, _reward, _done, {}

    def _is_button_pressed(self):
        """
        Returns whether the button is currently being pressed .

        :return: True if the button is currently pressed, False otherwise
        """
        return self.data.sensordata[0] > 0

    def _reward(self):
        """
        Returns a sparse reward from the environment.
        i.e if the button is being pressed, return 1.0 otherwise return 0.0.

        :return: A reward from the environment
        """
        return float(self._is_button_pressed())


# Example of how the environment could be used
if __name__ == "__main__":
    env = SpiderEnv(action_repeat=20)

    for episode in range(3):
        observation = env.reset()

        for t in range(1000):
            # Image observation
            # See `gym.envs.mujoco.mujoco_env.MujocoEnv` for more info about params
            pixels = env.render()
            print("Observation: ", observation)

            # Figure out an action...

            action = env.action_space.sample()
            print("Action: ", action)

            observation, reward, done, info = env.step(action)

            if done:
                print("Episode {} finished after {} timesteps".format(episode, t + 1))
                break

    env.close()
