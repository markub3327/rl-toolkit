# main libraries
import gym
import numpy as np
import wandb
import argparse

# policy
from td3 import TD3

# register PyBullet enviroments with OpenAI Gym
import pybulletgym

# utilities
from utils.replaybuffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

# Main function
def main(env_name: str,
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
         policy_delay: int):

    # Herne prostredie
    env = gym.make(env_name)
    env.render() # call this before env.reset, if you want a window showing the environment

    # select noise generator
    if (noise_type == 'normal'):
        noise = NormalActionNoise(mean=0.0, sigma=action_noise, size=env.action_space.shape)
    elif (noise_type == 'ornstein-uhlenbeck'):
        noise = OrnsteinUhlenbeckActionNoise(mean=0.0, sigma=action_noise, size=env.action_space.shape)
    else:
        raise ValueError()

    # inicializuj prostredie Weights & Biases
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

    log_interval = 1000     # print interval

    # policy
    agent = TD3(env.observation_space.shape, 
                 env.action_space.shape, 
                 learning_rate=learning_rate,
                 tau=tau, 
                 gamma=gamma,
                 target_noise=target_noise,
                 noise_clip=noise_clip,
                 policy_delay=policy_delay)

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
            # prekresli okno hry
            env.render()

            # select action randomly or using policy network
            if total_steps < learning_starts:
                # warmup
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)
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
        noise.reset()

        print(f'Epoch: {total_episodes}')
        print(f'EpsReward: {episode_reward}')
        print(f'EpsSteps: {episode_timesteps}')
        print(f'TotalInteractions: {total_steps}\n')
        wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps, "replayBuffer": len(rpm)})

        # update models after episode
        if total_steps > learning_starts:
            for gradient_step in range(episode_timesteps):
                batch = rpm.sample(batch_size)
                loss_a, loss_c = agent.train(batch, t=gradient_step)
                if (loss_a is not None):
                    wandb.log({"loss_a": loss_a, "loss_c": loss_c})

    # Save model to local drive
    agent.actor.model.save("save/model_A.h5")
    agent.critic_1.model.save("save/model_C1.h5")
    agent.critic_2.model.save("save/model_C2.h5")

    # zatvor prostredie
    env.close()

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(prog='python3 main.py', description='stable-baselines', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # init args
    my_parser.add_argument('-env', '--environment', type=str, help='Only OpenAI Gym environments are available!', default='BipedalWalker-v3')
    my_parser.add_argument('-t', '--max_steps', type=int, help='Maximum number of interactions doing in environment', default=int(1e6))
    my_parser.add_argument('--noise_type', type=str, help='Type of used noise generator', default='normal')
    my_parser.add_argument('--action_noise', type=float, help='Standard deviation of action noise', default=0.1)
    my_parser.add_argument('--gamma', type=float, help='Discount factor', default=0.98)
    my_parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=1e-3)
    my_parser.add_argument('--tau', type=float, help='Soft update learning rate', default=0.005)
    my_parser.add_argument('--batch_size', type=int, help='Size of the batch', default=100)
    my_parser.add_argument('--replay_size', type=int, help='Size of the replay buffer', default=int(2e5))
    my_parser.add_argument('--learning_starts', type=int, help='Number of steps before training', default=10000)
    my_parser.add_argument('--target_noise', type=float, help='Standard deviation of target noise', default=0.2)
    my_parser.add_argument('--noise_clip', type=float, help='Limit for target noise', default=0.5)
    my_parser.add_argument('--policy_delay', type=int, help='Delay between critic and policy update', default=2)

    # get args
    args = my_parser.parse_args()

    main(env_name=args.environment,
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
         policy_delay=args.policy_delay)