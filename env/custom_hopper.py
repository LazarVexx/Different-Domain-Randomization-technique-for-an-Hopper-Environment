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
            "Thigh": (self.original_masses[1] - 0.2, self.original_masses[1] + 0.2),   # Range for light mass (foot)
            "Leg": (self.original_masses[2] -0.2, self.original_masses[2] + 0.2),  # Range for medium mass (leg)
            "Foot": (self.original_masses[3]-0.2, self.original_masses[3] + 0.2),   # Range for heavy mass (thigh)
        }
               
        # Assign categories to body parts
        body_part_categories = ["Thigh", "Leg", "Foot"]
        
        # Sample masses based on the category ranges
        sampled_masses = []
        for category in body_part_categories:
            low, high = categories[category]
            sampled_mass = np.random.uniform(low, high)
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)
       
    #REDUCING RANGES DOMAIN RANDOMIZATION
    def set_rrdr_parameters(self, step, total_steps):
        self.set_parameters(self.sample_rrdr_parameters(step, total_steps))
        #print(self.sim.model.body_mass)
        
    def sample_rrdr_parameters(self, step, total_steps):
        # Calculate the ADR factor (start with a high factor for exploration)
        rrdr_factor = self.get_rrdr_factor(step, total_steps, max_factor=0.5, min_factor=0.0)
        
        # Define categories for different body parts with large ranges for exploration ( max - min distanze = 0.7)
        categories = {
            "Thigh": (self.original_masses[1] - 0.2, self.original_masses[1] + 0.2),   # Range for light mass (foot)
            "Leg": (self.original_masses[2] -0.2, self.original_masses[2] + 0.2),  # Range for medium mass (leg)
            "Foot": (self.original_masses[3]-0.2, self.original_masses[3] + 0.2),   # Range for heavy mass (thigh)
        }
        training_progress = step / total_steps
        # Assign categories to body parts
        body_part_categories = ["Thigh", "Leg", "Foot"]
        # Sample masses based on categories
        sampled_masses = []
        for category in body_part_categories:
            
            low, high = categories[category]
            
            expanded_high = high + rrdr_factor
                        
            mean_value = (low + expanded_high) / 2
            sampled_mass = np.random.normal(loc=mean_value, scale=(1-training_progress))
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)

            
    def get_rrdr_factor(self, current_step, total_steps, max_factor, min_factor):
        # Calculate the ADR factor based on the current training step
        progress = current_step / total_steps
        
        # Early phase: Large ADR factor for exploration
        if progress < 0.2:
            rrdr_factor = max_factor
        # Mid phase: Gradual reduction in ADR factor for controlled exploration
        elif progress < 0.8:
            #rrdr_factor = max_factor - (max_factor - min_factor) * progress
            rrdr_factor = max_factor - (max_factor - min_factor) * np.log1p(progress)
        # Late phase: Small ADR factor 
        else:
            rrdr_factor = min_factor 
        
        return rrdr_factor

    #INCREMENTAL RANGES EXPANSION DOMAIN RANDOMIZATION
    
    def set_ire_parameters(self, progress):
        self.set_parameters(self.sample_parameters_ire(progress))
        
    def sample_parameters_ire(self, training_progress):
        # Define categories for different body parts
        # Categories allows to sample masses for different body parts with different ranges
        categories = {
            "Thigh": (self.original_masses[1] - 0.2, self.original_masses[1] + 0.2),   # Range for light mass (foot)
            "Leg": (self.original_masses[2] -0.2, self.original_masses[2] + 0.2),  # Range for medium mass (leg)
            "Foot": (self.original_masses[3]-0.2, self.original_masses[3] + 0.2),   # Range for heavy mass (thigh)
        }
               
        # Assign categories to body parts
        body_part_categories = ["Thigh", "Leg", "Foot"]
        
        # Gradually expand the ranges based on training progress (0 to 1)
        def expand_range(base_range, progress):
            low, high = base_range
            range_expansion = np.log1p(progress) * 0.5  # Adjust expansion rate as needed
            return (low, high + range_expansion)

        # Adjust ranges dynamically
        categories = {k: expand_range(v, training_progress) for k, v in categories.items()}
            
        # Sample masses based on category ranges
        sampled_masses = []
        for category in body_part_categories:
            low, high = categories[category]
            mean_value = (low + high) / 2
            sampled_mass = np.random.normal(loc=mean_value, scale=(1-training_progress))
            sampled_masses.append(sampled_mass)
        
        return np.array(sampled_masses)
        
    #Exploration-Uniform Domain Randomization
    
    def set_eudr_parameters(self, step, total_steps,model, model_name):
        # Define the phase transitions based on the training progress
        uniform_ratio = 0.3
        increment_ratio = 0.4
        reducing_ratio = 0.3

        # Compute boundaries for each phase
        uniform_steps = uniform_ratio * total_steps
        increment_steps = increment_ratio * total_steps
        reducing_steps = reducing_ratio * total_steps
        
        if step < increment_steps:  # Early stage: increment randomization to explore near the range of uniform
            training_phase = "increment"
            phase_step = step
            phase_total_steps = increment_steps
        elif step < reducing_steps + increment_steps:  # Mid stage: reducing randomization that has larger ranges and allows to explore more
            training_phase = "reducing"
            phase_step = step - increment_steps
            phase_total_steps = reducing_steps
        else:  # Later stage: uniform randomization to exploit the learned policy
            training_phase = "uniform"
            phase_step = step - reducing_steps - increment_steps
            phase_total_steps = uniform_steps

         # Calculate normalized progress within the phase
        phase_progress = phase_step / phase_total_steps

        if training_phase == "increment":
            # Higher alpha to improve exploration
            if model_name == 'sac':
                model.policy.ent_coef = 0.1
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.02

            self.set_parameters(self.sample_parameters_ire(training_progress=phase_progress))

        elif training_phase == "reducing":
            # Medium alpha to reduce exploration
            if model_name == 'sac':
                model.policy.ent_coef = 0.05
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.01

            self.set_parameters(self.sample_rrdr_parameters(phase_step, phase_total_steps))

        elif training_phase == "uniform":
            # Lower alpha to encourage exploitation
            if model_name == 'sac':
                model.policy.ent_coef = 'auto'
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.005

            self.set_parameters(self.sample_random_parameters())

        else:
            raise ValueError(f"Unknown training_phase: {training_phase}")
        
        
    ## Dynamic Range Cycle Domain Randomization
    def set_drc_parameters(self, step, total_steps, model, model_name):
        # Define phase proportions
        uniform_ratio = 0.2
        increment_ratio = 0.4
        reducing_ratio = 0.4

        # Compute boundaries for each phase
        uniform_steps = uniform_ratio * total_steps
        increment_steps = increment_ratio * total_steps
        reducing_steps = reducing_ratio * total_steps
        
        # Determine the current phase and normalize the step
        if step < uniform_steps:  # Early stage: uniform randomization to explore near the range of uniform
            training_phase = "uniform"
            phase_step = step  
            phase_total_steps = uniform_steps

        elif step < uniform_steps + increment_steps:  # Mid stage: increment randomization to explore more
            training_phase = "increment"
            phase_step = step - uniform_steps  
            phase_total_steps = increment_steps
        else:  # Later stage: reducing randomization to reduce ranges to a size that is a less bigger than the initial ranges
            training_phase = "reducing"
            phase_step = step - uniform_steps - increment_steps  
            phase_total_steps = reducing_steps

        # Calculate normalized progress within the phase
        phase_progress = phase_step / phase_total_steps

        # Set parameters based on the phase
        if training_phase == "increment":
            # Higher alpha to improve exploration
            if model_name == 'sac':
                model.policy.ent_coef = 0.15
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.03

            self.set_parameters(self.sample_parameters_ire(training_progress=phase_progress))

        elif training_phase == "reducing":
            # Medium alpha to reduce exploration
            if model_name == 'sac':
                model.policy.ent_coef = 'auto'
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.005

            self.set_parameters(self.sample_rrdr_parameters(phase_step, phase_total_steps))

        elif training_phase == "uniform":
            # Lower alpha to encourage exploitation
            if model_name == 'sac':
                model.policy.ent_coef = 0.1
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.02

            self.set_parameters(self.sample_random_parameters())

        else:
            raise ValueError(f"Unknown training_phase: {training_phase}")
        
        
    ## Dynamic Exploration Domain Randomization
    def set_dedr_parameters(self, step, total_steps, model, model_name):
        # Define phase proportions
        uniform_ratio = 0.3
        increment_ratio = 0.3
        reducing_ratio = 0.4

        # Compute boundaries for each phase
        uniform_steps = uniform_ratio * total_steps
        increment_steps = increment_ratio * total_steps
        reducing_steps = reducing_ratio * total_steps
        
        # Determine the current phase and normalize the step
        if step < reducing_steps:  # Early stage: reducing randomization to have larger ranges at the beginning and improve exploration
            training_phase = "reducing"
            phase_step = step  
            phase_total_steps = reducing_steps
        elif step < uniform_steps + reducing_steps:  # Mid stage: Uniform randomization to reduce ranges to a size that is a less bigger than the initial ranges
            training_phase = "uniform"
            phase_step = step - reducing_steps
            phase_total_steps = uniform_steps
        else:  # Later stage: increment randomization to explore more in a range near to the previous uniform range
            training_phase = "increment"
            phase_step = step - uniform_steps - reducing_steps 
            phase_total_steps = increment_steps

        # Calculate normalized progress within the phase
        phase_progress = phase_step / phase_total_steps

        # Set parameters based on the phase
        if training_phase == "increment":
            # Medium alpha to improve exploration at the end
            if model_name == 'sac':
                model.policy.ent_coef = 0.05
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.01

            self.set_parameters(self.sample_parameters_ire(training_progress=phase_progress))

        elif training_phase == "reducing":
            # Higher alpha to improve exploration at the start
            if model_name == 'sac':
                model.policy.ent_coef = 0.1
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.02

            self.set_parameters(self.sample_rrdr_parameters(phase_step, phase_total_steps))

        elif training_phase == "uniform":
            # Lower alpha to encourage exploitation
            if model_name == 'sac':
                model.policy.ent_coef = 0.05
            elif model_name == 'ppo':
                model.policy.ent_coef = 0.01

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

