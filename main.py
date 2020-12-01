# main libraries
import gym
import numpy as np
import wandb
import argparse

# policy
from td3 import TD3
from sac import SAC

# utilities
from utils.replaybuffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import pybulletgym

# Main function
def main(env_name: str,
         alg: str,
         learning_rate: float,
         gamma: float,
         batch_size: int,
         tau: float,
         replay_size: int,
         learning_starts: int,
         max_steps: int,
         noise_type: str,
         action_noise: float,
         target_noise: float,
         noise_clip: float,
         policy_delay: int,
         logging_wandb: bool):

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
                    policy_delay=policy_delay)
    elif (alg == 'sac'): 
        agent = SAC(env.observation_space.shape, 
                    env.action_space.shape, 
                    learning_rate=learning_rate,
                    tau=tau,
                    gamma=gamma)
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

        obs = env.reset()

        # collect rollout
        while not done:
            # select action randomly or using policy network
            if total_steps < learning_starts:
                # warmup
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)
                if (alg == 'td3'):
                    action = np.clip(action + noise(), env.action_space.low, env.action_space.high)

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
        print(f'TotalInteractions: {total_steps}\n')
        if logging_wandb == True:
            wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps, "replayBuffer": len(rpm)})

        # update models after episode
        if total_steps > learning_starts:
            for gradient_step in range(episode_timesteps):
                batch = rpm.sample(batch_size)
                if (alg == 'td3'):
                    loss_a, loss_c = agent.train(batch, t=gradient_step)
                    if (logging_wandb == True and loss_a is not None):
                        wandb.log({"loss_a": loss_a, "loss_c": loss_c})
                else:
                    loss_a, loss_c, loss_alpha, alpha = agent.train(batch)
                    if (logging_wandb == True):
                        wandb.log({"loss_a": loss_a, "loss_c": loss_c, "loss_alpha": loss_alpha, "alpha": alpha})

    # Save model to local drive
    agent.actor.model.save("save/model_A.h5")
    agent.critic_1.model.save("save/model_C1.h5")
    agent.critic_2.model.save("save/model_C2.h5")

    # zatvor prostredie
    env.close()

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(prog='python3 main.py', description='stable-baselines', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # init args
    my_parser.add_argument('-alg', '--algorithm', type=str, help='Only OpenAI Gym environments are available!', default='sac')
    my_parser.add_argument('-env', '--environment', type=str, help='Only OpenAI Gym environments are available!', default='MountainCarContinuous-v0')
    my_parser.add_argument('-t', '--max_steps', type=int, help='Maximum number of interactions doing in environment', default=int(3e5))
    my_parser.add_argument('--noise_type', type=str, help='Type of used noise generator (only for TD3)', default='ornstein-uhlenbeck')
    my_parser.add_argument('--action_noise', type=float, help='Standard deviation of action noise (only for TD3)', default=0.5)
    my_parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)
    my_parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=3e-4)
    my_parser.add_argument('--tau', type=float, help='Soft update learning rate', default=0.005)
    my_parser.add_argument('--batch_size', type=int, help='Size of the batch', default=256)
    my_parser.add_argument('--replay_size', type=int, help='Size of the replay buffer', default=int(1e6))
    my_parser.add_argument('--learning_starts', type=int, help='Number of steps before training', default=100)
    my_parser.add_argument('--target_noise', type=float, help='Standard deviation of target noise (only for TD3)', default=0.2)
    my_parser.add_argument('--noise_clip', type=float, help='Limit for target noise (only for TD3)', default=0.5)
    my_parser.add_argument('--policy_delay', type=int, help='Delay between critic and policy update (only for TD3)', default=2)
    my_parser.add_argument('--wandb', action='store_true', help='Logging to wanDB')

    # get args
    args = my_parser.parse_args()
    print(args)

    main(env_name=args.environment,
         alg=args.algorithm,
         learning_rate=args.learning_rate,
         gamma=args.gamma, 
         batch_size=args.batch_size, 
         tau=args.tau,         
         replay_size=args.replay_size, 
         learning_starts=args.learning_starts, 
         max_steps=args.max_steps,
         noise_type=args.noise_type,
         action_noise=args.action_noise,
         target_noise=args.target_noise,
         noise_clip=args.noise_clip,
         policy_delay=args.policy_delay,
         logging_wandb=args.wandb)