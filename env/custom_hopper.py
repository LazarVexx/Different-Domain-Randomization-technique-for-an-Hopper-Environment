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
        
    #UNIFORM RANDOMIZATION
    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_random_parameters())

    def sample_random_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """          
        # Define categories for different body parts with their ranges
        categories = {
            "light": (0.5, 1.0),   # Range for light mass (foot)
            "medium": (1.0, 2.0),  # Range for medium mass (leg)
            "heavy": (1.5, 3.0),   # Range for heavy mass (thigh)
        }
        
        # Assign categories to body parts
        body_part_categories = ["medium", "heavy", "light"]  # Thigh, Leg, Foot
        
        # Sample masses based on the category ranges
        sampled_masses = []
        for category in body_part_categories:
            low, high = categories[category]
            sampled_mass = np.random.uniform(low, high)
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)
       
    #ADAPTIVE DOMAIN RANDOMIZATION
    def set_adr_parameters(self, step, total_steps):
        self.set_parameters(self.sample_adr_parameters(step, total_steps))
        #print(self.sim.model.body_mass)
        
    def sample_adr_parameters(self, step, total_steps):
        # Calculate the ADR factor (start with a high factor for exploration)
        adr_factor = self.get_adr_factor(step, total_steps, max_factor=2.0, min_factor=1.0)
        
        # Define categories for different body parts with large ranges for exploration
        categories = {
            "light": (0.5, 1.5),   # Range for foot 
            "medium": (1.0, 3.0),  # Range for leg 
            "heavy": (1.5, 4.0),   # Range for thigh 
        }
        
        # Sample masses based on categories
        sampled_masses = []
        for category in ["medium", "heavy", "light"]:
            low, high = categories[category]
            
            # Prevent the high value from becoming smaller than the low value
            max_expansion = (high - low)
            expanded_high = high + max_expansion * adr_factor / 2
            
            # Ensure that the high value doesn't go below the low value
            expanded_high = max(expanded_high, low)
            
            # Sample mass within the adjusted range
            sampled_mass = np.random.uniform(low, expanded_high)
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)

            
    def get_adr_factor(self, current_step, total_steps, max_factor=3.0, min_factor=1.0):
        # Calculate the ADR factor based on the current training step
        progress = current_step / total_steps
        
        # Early phase: Large ADR factor for exploration
        if progress < 0.4:
            adr_factor = max_factor
        # Mid phase: Gradual reduction in ADR factor for controlled exploration
        elif progress < 0.8:
            adr_factor = max_factor - (max_factor - min_factor) * (progress - 0.4) / 0.4
        # Late phase: Small ADR factor for minimal randomization
        else:
            adr_factor = min_factor
        
        return adr_factor

    #Continual DOMAIN RANDOMIZATION
    
    def set_cdr_parameters(self, progress):
        self.set_parameters(self.sample_parameters_cdr(progress))
        
    def sample_parameters_cdr(self, training_progress):
        # Define categories for different body parts
        # Categories allows to sample masses for different body parts with different ranges
        categories = {
            "light": (0.5, 0.9),
            "medium": (1.0, 1.4),
            "heavy": (1.5, 2.0),
        }
        
        # Gradually expand the ranges based on training progress (0 to 1)
        def expand_range(base_range, progress):
            low, high = base_range
            range_expansion = progress * 0.2  # Adjust expansion rate as needed
            return (low - range_expansion, high + range_expansion)

        # Adjust ranges dynamically
        categories = {k: expand_range(v, training_progress) for k, v in categories.items()}
    
        # Assign categories to thigh, leg, and foot
        body_part_categories = ["medium", "heavy", "light"]  # Can be adjusted
        
        # Sample masses based on category ranges
        sampled_masses = []
        for category in body_part_categories:
            low, high = categories[category]
            sampled_mass = np.random.uniform(low, high)
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)
    
    #DOMAIN RANDOMIZATION WITH ENTROPY REGULATION
    
    def set_entropy_regulation_parameters(self, step, total_steps,model, model_name):
        # Define the phase transitions based on the training progress
        if step < 0.4 * total_steps:  # Early stage: uniform randomization
            training_phase = "adaptive"
        elif step < 0.8 * total_steps:  # Mid stage: adaptive randomization
            training_phase = "continual"
        else:  # Later stage: continual randomization
            training_phase = "uniform"

        if training_phase == "adaptive":
            #Higher alpha to improve exploration
            if model_name == 'sac':
                model.policy.ent_coef = 0.1 
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.02
                
            total_steps_adaptive = 0.4 * total_steps
            self.set_parameters(self.sample_adr_parameters(step, total_steps_adaptive))
                           
        elif training_phase == "continual":
            #Lower alpha to reduce exploration
            if model_name == 'sac':
                model.policy.ent_coef = 0.05
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.01
            
            self.set_parameters(self.sample_parameters_cdr(training_progress=step/total_steps))
        
        elif training_phase == "uniform":
            if model_name == 'sac':
                model.policy.ent_coef = 'auto'
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.005
            
            self.set_parameters(self.sample_random_parameters())
            
            
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

