"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
import matplotlib.pyplot as plt
import argparse
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def create_model_and_train(env, env_name, model_name, n_timesteps, randomization, randomization_type, label):
    log_dir = "./tmp/gym/" + env_name + "/"    
       
    if model_name == 'ppo':
        model = PPO('MlpPolicy', env, verbose=1)
    elif model_name == 'sac':
        model = SAC('MlpPolicy', env, verbose=1)
    else:
        raise ValueError('Unknown model name')
    
    if randomization:
        print("Training with " + randomization_type + " domain randomization")
        batch_size = model.n_steps if model_name=='ppo' else 1 ## For PPO with Uniform Domain Randomization
        for step in range(n_timesteps // batch_size):
            if randomization_type == 'uniform':
                #Uniform Domain Randomization
                env.set_random_parameters()
            elif randomization_type == 'reducing':
                #REDUCING RANGES DOMAIN RANDOMIZATION
                env.set_rrdr_parameters(step, n_timesteps // batch_size)
            elif randomization_type == 'incremental':
                #INCREMENTAL RANGES EXPANSION DOMAIN RANDOMIZATION
                env.set_ire_parameters(progress=step/n_timesteps)
            elif randomization_type == 'exploration-uniform':
                #Exploration-Uniform Domain Randomization
                env.set_eudr_parameters(step, n_timesteps // batch_size, model, model_name)
            elif randomization_type == 'dynamic range cycle':
                 ## Dynamic Range Cycle Domain Randomization
                env.set_drc_parameters(step, n_timesteps // batch_size, model, model_name)
            elif randomization_type == 'dynamic exploration':
                ## Dynamic Exploration Domain Randomization
                env.set_dedr_parameters(step, n_timesteps // batch_size, model, model_name)
            else:
                raise ValueError('Unknown randomization type')               
            
            model.learn(total_timesteps=1, reset_num_timesteps=False)
    else:    
        print("Training without domain randomization")
        model.learn(total_timesteps=n_timesteps)
        
    model.save(os.path.join(log_dir, "trained_model"))
    plot_results(log_dir, model_name, randomization_type, label, randomization)
    return model

def evaluate_model(model, env, n_eval_episodes, render):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render)
    return mean_reward, std_reward

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, model_name, randomization_type, label, randomization,  title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    if randomization:
        plt.title(title + " " + model_name + " "+ randomization_type + " " + label )
    else:
        plt.title(title + " without randomization")
    plt.show()

def main():
    source_env_name = 'CustomHopper-source-v0'
    target_env_name = 'CustomHopper-target-v0'
    algorithm = 'sac'
    total_timesteps = 250000
    test_episodes = 100
    randomization = True
    randomization_type = 'uniform'
    
    log_dir = "./tmp/gym/" + source_env_name + "/"
    os.makedirs(log_dir, exist_ok=True)
       
    source_env = gym.make(source_env_name)
    source_env = Monitor(source_env, log_dir)
    
    print(source_env_name+' State space:', source_env.observation_space)  # state-space
    print(source_env_name+' Action space:', source_env.action_space)  # action-space
    print(source_env_name+' Dynamics parameters:', source_env.get_parameters())  # masses of each link of the Hopper
    
    log_dir = "./tmp/gym/" + target_env_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    
    target_env = gym.make(target_env_name)
    target_env = Monitor(target_env, log_dir)
    
    print(target_env_name+' State space:', target_env.observation_space)  # state-space
    print(target_env_name+' Action space:', target_env.action_space)  # action-space
    print(target_env_name+' Dynamics parameters:', target_env.get_parameters())  # masses of each link of the Hopper
    
    print("Training on source environment...")
    source_model = create_model_and_train(source_env,source_env_name, algorithm, total_timesteps, randomization, randomization_type, label='source')

    print("Training on target environment...")
    target_model = create_model_and_train(target_env,target_env_name, algorithm, total_timesteps, randomization, randomization_type, label='target')

    print("Evaluating models...")
    results = {}
    results['source->source'] = evaluate_model(source_model, source_env, test_episodes, render=True)
    results['source->target'] = evaluate_model(source_model, target_env, test_episodes, render=True)
    results['target->target'] = evaluate_model(target_model, target_env, test_episodes, render=False)

    print("\nResults:")
    for config, (mean, std) in results.items():
        print(f"{config}: Mean Reward = {mean:.2f}, Std Dev = {std:.2f}" )
        
    source_env.close()
    target_env.close()


if __name__ == '__main__':
    main()