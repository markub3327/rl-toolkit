import gym
import wandb
import pybulletgym

import tensorflow as tf

from sac import Actor as ActorSAC
from td3 import Actor as ActorTD3

# Main testing function
def main(env_name: str,
         model_a_path: str,
         alg: str,
         max_steps: int,
         logging_wandb: bool):

    # Herne prostredie
    env = gym.make(env_name)
    env.render()    # init pybullet env

    # inicializuj prostredie Weights & Biases
    if logging_wandb == True:
        wandb.init(project="stable-baselines")

    # load actor model
    if (alg == 'sac'):
        actor = ActorSAC(model_path=model_a_path)
    else:
        actor = ActorTD3(model_path=model_a_path)

    # hlavny cyklus hry
    total_steps, total_episodes = 0, 0
    while total_steps < max_steps:
        done = False
        episode_reward, episode_timesteps = 0.0, 0

        obs = env.reset()

        # collect rollout
        while not done:
            env.render()
            
            if (alg == 'sac'):           
                action, _ = actor.model(tf.expand_dims(obs, axis=0))
            elif (alg == 'td3'):
                action = actor.model(tf.expand_dims(obs, axis=0))

            # perform action
            new_obs, reward, done, _ = env.step(action[0])

            episode_reward += reward
            episode_timesteps += 1
            total_steps += 1

            # super critical !!!
            obs = new_obs

        # after each episode
        total_episodes += 1

        print(f'Epoch: {total_episodes}')
        print(f'EpsReward: {episode_reward}')
        print(f'EpsSteps: {episode_timesteps}')
        print(f'TotalInteractions: {total_steps}')
        if logging_wandb == True:
            wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps})
        
    
    env.close()