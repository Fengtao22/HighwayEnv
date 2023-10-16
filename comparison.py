from stable_baselines3 import PPO, DQN #A2C
import torch, time
import numpy as np
torch.cuda.empty_cache()
import warnings
#warnings.filterwarnings("ignore")
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import gymnasium as gym
os.environ["OFFSCREEN_RENDERING"] = "1"

def make_env(env_id, rank, seed=10):
    def _init():
        env = gym.make(env_id, render_mode='rgb_array')
        env.configure({
        "right_lane_reward": 0,   
        "high_speed_reward": 0, 
        "lanes_count": 4
        "vehicles_count": 100,
        "vehicles_density": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 40],
        "ego_spacing": 1,
        "policy_frequency": 2    
        })  ### I changed the road speed limit to 40 and set the other NPC starting vel to be [15,25]
        '''    
        env.config["observation"] = {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        #env.config["offscreen_rendering"] = True
        #env.config["show_trajectories"] = False
        '''
        env = gym.wrappers.RecordVideo(env, './Testvideos')
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

env_id = 'highway-v0'
env = make_env(env_id, 0)


model = PPO.load("new_logs/best_model", device='cpu')
for i in range(10):
    obs, info = env.reset()
    print('policy_frequency: ', env.configure["policy_frequency"])
    done = truncated = False
    steps = 0
    total_rewards = 0
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        img = env.render()
        print(type(img))
        pyplot.imshow(img)
        pyplot.show()
        clear_output(wait=True)
        
        steps += 1
        total_rewards += reward
        if done or truncated:
            print('{}th exp steps: {} and rewards are: {}\n\n'.format(i+1, steps, total_rewards))
            break


import numpy as np
data=np.load('./new_logs/evaluations.npz')
exp_num = data['results'].shape[1]  ### timesteps, results,  ep_lengths
import pandas as pd
new_times = []
exp_name =  []
scores  = []
for index, ele in enumerate(data['timesteps']):
    for rep in range(exp_num):
        new_times.append(ele)
        exp_name.append('Exp'+str(rep))  
    scores += list(data['results'][index])  
    ep_len += list(data['ep_lengths'][index])  
new_times = np.array(new_times)
exp_name = np.array(exp_name)
print(len(new_times))
print(len(exp_name))
print(len(scores))
print(len(ep_len))
dada = {'timesteps': new_times,
        'exp': exp_name,
       'score': scores,
       'eplen': ep_len}

df = pd.DataFrame(dada)
import seaborn as sns
sns.lineplot(data=df, x="timesteps", y="score")
sns.lineplot(data=df, x="timesteps", y="eplen")
