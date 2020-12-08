import gym
import wandb
import pybulletgym

import numpy as np

# policy
from td3 import TD3
from sac import SAC

# utilities
from utils.replaybuffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


# Main training function
def main(env_name: str,
         alg: str,
         learning_rate: float,
         gamma: float,
         batch_size: int,
         tau: float,
         replay_size: int,
         learning_starts: int,
         update_after: int,
         max_steps: int,
         noise_type: str,
         action_noise: float,
         target_noise: float,
         noise_clip: float,
         policy_delay: int,
         logging_wandb: bool,
         save_path: str,
         model_a_path: str,
         model_c1_path: str,
         model_c2_path: str):

    # Herne prostredie
    env = gym.make(env_name)

    # select noise generator
    if (noise_type == 'normal'):
        noise = NormalActionNoise(mean=0.0, sigma=action_noise, size=env.action_space.shape)
    elif (noise_type == 'ornstein-uhlenbeck'):
        noise = OrnsteinUhlenbeckActionNoise(mean=0.0, sigma=action_noise, size=env.action_space.shape)
    else:
        raise NameError(f"'{noise_type}' noise is not defined")

    # inicializuj prostredie Weights & Biases
    if logging_wandb == True:
        wandb.init(project="stable-baselines")

        ###
        ### Settings
        ###
        wandb.config.gamma                  =  gamma
        wandb.config.batch_size             =  batch_size
        wandb.config.tau                    =  tau
        wandb.config.learning_rate          =  learning_rate
        wandb.config.replay_size            =  replay_size
        wandb.config.learning_starts        =  learning_starts
        wandb.config.update_after           =  update_after
        wandb.config.max_steps              =  max_steps
        wandb.config.action_noise           =  action_noise
        wandb.config.target_noise           =  target_noise
        wandb.config.noise_clip             =  noise_clip
        wandb.config.policy_delay           =  policy_delay

    # policy
    if (alg == 'td3'): 
        agent = TD3(env.observation_space.shape, 
                    env.action_space.shape, 
                    learning_rate=learning_rate,
                    tau=tau, 
                    gamma=gamma,
                    target_noise=target_noise,
                    noise_clip=noise_clip,
                    policy_delay=policy_delay,
                    model_a_path=model_a_path,
                    model_c1_path=model_c1_path,
                    model_c2_path=model_c2_path)
    elif (alg == 'sac'): 
        agent = SAC(env.observation_space.shape, 
                    env.action_space.shape,
                    actor_learning_rate=learning_rate,
                    critic_learning_rate=learning_rate,
                    alpha_learning_rate=learning_rate,
                    tau=tau,
                    gamma=gamma,
                    policy_delay=policy_delay,
                    model_a_path=model_a_path,
                    model_c1_path=model_c1_path,
                    model_c2_path=model_c2_path)
    else:
        raise NameError(f"algorithm '{alg}' is not defined")
 
    # plot model to png
    #agent.actor.save()
    #agent.critic_1.save()

    # replay buffer
    rpm = ReplayBuffer(env.observation_space.shape, env.action_space.shape, replay_size)

    print(env.action_space.low, env.action_space.high)

    # hlavny cyklus hry
    total_steps, total_episodes = 0, 0
    while total_steps < max_steps:
        done = False
        episode_reward, episode_timesteps = 0.0, 0
        log_entropy = []

        obs = env.reset()

        # collect rollout
        while not done:
            # select action randomly or using policy network
            if total_steps < learning_starts:
                # warmup
                action = env.action_space.sample()
            else:
                if (alg == 'td3'):
                    action = agent.get_action(obs)
                    action = np.clip(action + noise(), env.action_space.low, env.action_space.high)
                else:
                    action, logp = agent.get_action(obs)
                    log_entropy.append(logp)

            # perform action
            new_obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_timesteps += 1
            total_steps += 1

            # store interaction
            rpm.store(obs, action, reward, new_obs, done)

            # super critical !!!
            obs = new_obs

        # after each episode
        total_episodes += 1

        if (alg == 'td3'):
            noise.reset()

        print(f'Epoch: {total_episodes}')
        print(f'EpsReward: {episode_reward}')
        print(f'EpsSteps: {episode_timesteps}')
        print(f'TotalInteractions: {total_steps}')
        if (alg == 'td3' and logging_wandb == True):
            wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps, "replayBuffer": len(rpm)})
        elif (alg == 'sac'):
            print(f'Entropy: {-np.mean(log_entropy)}\n')
            if logging_wandb == True:
                wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps, "replayBuffer": len(rpm), "entropy": -np.mean(log_entropy)})
            
        # update models after episode
        if total_steps > update_after and len(rpm) > batch_size:
            losses_a, losses_c1, losses_c2, losses_alpha, alphas = [], [], [], [], []

            for gradient_step in range(1, episode_timesteps+1):         # the first one must be critic network, the second one is actor network
                batch = rpm.sample(batch_size)                
                losses = agent.train(batch, t=gradient_step)

                if (losses[0] is not None):
                    if (alg == 'sac'):
                        losses_alpha.append(losses[3])
                        alphas.append(losses[4])

                    losses_a.append(losses[0])
                    losses_c1.append(losses[1])
                    losses_c2.append(losses[2])

            # logging of epoch's mean loss
            if (logging_wandb == True):
                if (alg == 'td3'):
                    wandb.log({"loss_a": np.mean(losses_a), "loss_c1": np.mean(losses_c1), "loss_c2": np.mean(losses_c2)})
                else:
                    wandb.log({"loss_a": np.mean(losses_a), "loss_c1": np.mean(losses_c1), "loss_c2": np.mean(losses_c2), "loss_alpha": np.mean(losses_alpha), "alpha": np.mean(alphas)})

    # Save model to local drive
    if (type(save_path) == str):
        agent.actor.model.save(f"{save_path}model_A.h5")
        agent.critic_1.model.save(f"{save_path}model_C1.h5")
        agent.critic_2.model.save(f"{save_path}model_C2.h5")

    # zatvor prostredie
    env.close()