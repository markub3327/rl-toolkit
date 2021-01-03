import gym
import pybullet_envs
import wandb
import argparse
import numpy as np

from sac import Actor as ActorSAC
from td3 import Actor as ActorTD3

from utils import ReplayBuffer

if __name__ == "__main__":
    # init argparse
    my_parser = argparse.ArgumentParser(prog='python3 main.py', description='RL training toolkit', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # init arguments
    my_parser.add_argument('-alg', '--algorithm', type=str, help='Select the algorithm, that making a decisions', default='sac')
    my_parser.add_argument('-env', '--environment', type=str, help='Only OpenAI Gym & PyBullet environments are available!', default='BipedalWalker-v3')
    my_parser.add_argument('-t', '--max_steps', type=int, help='Maximum number of interactions doing in environment', default=int(1e6))
    my_parser.add_argument('--noise_type', type=str, help='Type of noise generator [normal, ornstein-uhlenbeck] (only for TD3)', default='ornstein-uhlenbeck')
    my_parser.add_argument('--action_noise', type=float, help='Standard deviation of action noise (only for TD3)', default=0.5)
    my_parser.add_argument('--warm_up_steps', type=int, help='Number of steps before using policy network (warming up)', default=10000)
    my_parser.add_argument('--wandb', action='store_true', help='Logging to wanDB')
    my_parser.add_argument('--test', action='store_true', help='Run test mode')
    my_parser.add_argument('-f', '--model', type=str, help="Actor's model file")
    my_parser.add_argument('-s', '--save', type=str, help='Path for saving model files')

    # get args
    args = my_parser.parse_args()
    print(args)

    # Herne prostredie
    env = gym.make(args.environment)
    
    # inicializuj prostredie Weights & Biases
    if args.wandb == True:
        wandb.init(project="rl-baselines")

        ###
        ### Settings
        ###
        wandb.config.max_steps              =  args.max_steps
        wandb.config.noise_type             =  args.noise_type
        wandb.config.action_noise           =  args.action_noise
        wandb.config.warm_up_steps          =  args.warm_up_steps

    # policy
    if (args.algorithm == 'td3'):
        agent = ActorTD3(
            model_path=args.model, 
            noise_type=args.noise_type,
            action_noise=args.action_noise
        )
    elif (args.algorithm == 'sac'):
        agent = ActorSAC(model_path=args.model)
    else:
        raise NameError(f"Algorithm '{args.algorithm}' is not defined")
 
    # plot model to png
    #agent.save()

    # replay buffer
    rpm = ReplayBuffer(
        env_name=args.environment, 
        server_name="192.168.1.2"
    )

    print(env.action_space.low, env.action_space.high)

    # hlavny cyklus hry
    total_steps, total_episodes = 0, 0
    while total_steps < args.max_steps:
        done = False
        episode_reward, episode_timesteps = 0.0, 0

        obs = env.reset()

        # reset noise
        if (args.algorithm == 'td3'):
            agent.noise.reset()
        # re-new noise matrix
        elif (args.algorithm == 'sac'):
            agent.sample_weights()

        # collect rollout
        while not done:
            # select action randomly or using policy network
            if total_steps < args.warm_up_steps:
                # warmup
                action = env.action_space.sample()
            else:
                action = agent.predict(np.expand_dims(obs, axis=0))[0].numpy()

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

        print(f'Epoch: {total_episodes}')
        print(f'EpsReward: {episode_reward}')
        print(f'EpsSteps: {episode_timesteps}')
        print(f'TotalInteractions: {total_steps}')
        print(f"ReplayBuffer: {len(rpm)}")
        if (args.wandb == True):
            wandb.log({"epoch": total_episodes, "score": episode_reward, "steps": episode_timesteps, "replayBuffer": len(rpm)})

        # sync with db
        rpm.sync()

    # Save model to local drive
    #if (save_path is not None):
    #    agent.model.save(f"{save_path}model_A_{env_name}.h5")
    
    # zatvor prostredie
    env.close()