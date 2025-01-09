"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses
    
    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2:] = task
        
    #UNIFROM RANDOMIZATION
    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_random_parameters())

    def sample_random_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        num_links = len(self.sim.model.body_mass) - 2
        sampled_masses = np.random.normal(loc=1.0, scale=0.1, size=num_links)
        return sampled_masses
     
    #ADAPTIVE DOMAIN RANDOMIZATION
    def set_adr_parameters(self, step, total_steps):
        self.set_parameters(self.sample_adr_parameters(step, total_steps))
        #print(self.sim.model.body_mass)
        
    def sample_adr_parameters(self, step, total_steps):
        # Calculate the randomization factor
        adr_factor = self.get_adr_factor(step, total_steps)
        #print(f"Applying ADR with factor {adr_factor:.2f}")

        # Scale the randomization based on adr_factor
        num_links = len(self.sim.model.body_mass) - 2
        # Sample masses from a normal distribution consideting the adr_factor
        # The scale parameter is multiplied by adr_factor
        # High values of adr_factor result in a wider spread of sampled masses and introduce more randomness
        sampled_masses = np.random.normal(loc=1.0, scale=0.1 * adr_factor, size=num_links)
        return sampled_masses
            
    def get_adr_factor(self, current_step, total_steps, max_factor=1.0, min_factor=0.2):
        # Considering the current training step, calculate the randomization factor
        # The factor is linearly decreased from max_factor to min_factor
        # The high value at the beginning of training allows for exploration
        
        # Linear decay of the randomization factor
        return max(min_factor, max_factor * (1 - current_step / total_steps))

    #CUSTOM DOMAIN RANDOMIZATION
    
    def set_cdr_parameters(self):
        self.set_parameters(self.sample_parameters_cdr())
        
    def sample_parameters_cdr(self):
        # Define categories for different body parts
        # Categories allows to sample masses for different body parts with different ranges
        categories = {
            "light": (0.5, 0.9),
            "medium": (1.0, 1.4),
            "heavy": (1.5, 2.0),
        }
        
        # Assign categories to thigh, leg, and foot
        body_part_categories = ["medium", "heavy", "light"]  # Can be adjusted
        
        # Sample masses based on category ranges
        sampled_masses = []
        for category in body_part_categories:
            low, high = categories[category]
            sampled_mass = np.random.uniform(low, high)
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)
    
    #DOMAIN RANDOMIZATION WITH ENTROPY MAXIMIZATION
    
    def set_doraemon_parameters(self, step, total_steps):
        # Define the phase transitions based on the training progress
        if step < 0.3 * total_steps:  # Early stage: uniform randomization
            training_phase = "uniform"
        elif step < 0.6 * total_steps:  # Mid stage: adaptive randomization
            training_phase = "adaptive"
        else:  # Later stage: custom randomization
            training_phase = "custom"

        if training_phase == "uniform":
            self.set_parameters(self.sample_random_parameters())
        elif training_phase == "adaptive":
            self.set_parameters(self.sample_adr_parameters(step, total_steps))
        elif training_phase == "custom":
            self.set_parameters(self.sample_parameters_cdr())
        else:
            raise ValueError(f"Unknown training_phase: {training_phase}")
    
        
    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

