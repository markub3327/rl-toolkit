import gym
import argparse
import wandb
import numpy as np
import pybullet_envs

# policy
from td3 import TD3
from sac import SAC

# utilities
from utils.replaybuffer import ReplayBuffer


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 training.py",
        description="RL training toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # init args
    my_parser.add_argument(
        "-alg",
        "--algorithm",
        type=str,
        help="Only OpenAI Gym environments are available!",
        default="sac",
    )
    my_parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Only OpenAI Gym environments are available!",
        default="BipedalWalker-v3",
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Maximum number of interactions doing in environment",
        default=int(1e6),
    )
    my_parser.add_argument(
        "--noise_type",
        type=str,
        help="Type of used noise generator [normal, ornstein-uhlenbeck] (only for TD3)",
        default="ornstein-uhlenbeck",
    )
    my_parser.add_argument(
        "--action_noise",
        type=float,
        help="Standard deviation of action noise (only for TD3)",
        default=0.5,
    )
    my_parser.add_argument("--gamma", type=float, help="Discount factor", default=0.98)
    my_parser.add_argument(
        "-lr", "--learning_rate", type=float, help="Learning rate", default=7.3e-4
    )
    my_parser.add_argument("--tau", type=float, help="Soft update rate", default=0.02)
    my_parser.add_argument(
        "--batch_size", type=int, help="Size of the batch", default=256
    )
    my_parser.add_argument(
        "--replay_size", type=int, help="Size of the replay buffer", default=int(1e6)
    )
    my_parser.add_argument(
        "--learning_starts",
        type=int,
        help="Number of steps before using policy network",
        default=10000,
    )
    my_parser.add_argument(
        "--update_after",
        type=int,
        help="Number of steps before training",
        default=10000,
    )
    my_parser.add_argument(
        "--update_every", type=int, help="Train frequency", default=64
    )
    my_parser.add_argument(
        "--target_noise",
        type=float,
        help="Standard deviation of target noise (only for TD3)",
        default=0.2,
    )
    my_parser.add_argument(
        "--noise_clip",
        type=float,
        help="Limit for target noise (only for TD3)",
        default=0.5,
    )
    my_parser.add_argument(
        "--policy_delay",
        type=int,
        help="Delay between critic and policy update",
        default=2,
    )
    my_parser.add_argument("--wandb", action="store_true", help="Logging to wanDB")
    my_parser.add_argument("-s", "--save", type=str, help="Path for saving model files")
    my_parser.add_argument("--model_a", type=str, help="Actor's model file")
    my_parser.add_argument("--model_c1", type=str, help="Critic 1's model file")
    my_parser.add_argument("--model_c2", type=str, help="Critic 2's model file")

    # nacitaj zadane argumenty programu
    args = my_parser.parse_args()

    # Herne prostredie
    env = gym.make(args.environment)

    # inicializuj prostredie Weights & Biases
    if args.wandb == True:
        wandb.init(project="rl-toolkit")

        ###
        ### Settings
        ###
        wandb.config.gamma = args.gamma
        wandb.config.batch_size = args.batch_size
        wandb.config.tau = args.tau
        wandb.config.learning_rate = args.learning_rate
        wandb.config.replay_size = args.replay_size
        wandb.config.learning_starts = args.learning_starts
        wandb.config.update_after = args.update_after
        wandb.config.update_every = args.update_every
        wandb.config.max_steps = args.max_steps
        wandb.config.action_noise = args.action_noise
        wandb.config.target_noise = args.target_noise
        wandb.config.noise_clip = args.noise_clip
        wandb.config.policy_delay = args.policy_delay

    # policy
    if args.algorithm == "td3":
        agent = TD3(
            env.observation_space.shape,
            env.action_space.shape,
            learning_rate=args.learning_rate,
            tau=args.tau,
            gamma=args.gamma,
            noise_type=args.noise_type,
            action_noise=args.action_noise,
            target_noise=args.target_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            model_a_path=args.model_a,
            model_c1_path=args.model_c1,
            model_c2_path=args.model_c2,
        )
    elif args.algorithm == "sac":
        agent = SAC(
            env.observation_space.shape,
            env.action_space.shape,
            actor_learning_rate=args.learning_rate,
            critic_learning_rate=args.learning_rate,
            alpha_learning_rate=args.learning_rate,
            tau=args.tau,
            gamma=args.gamma,
            policy_delay=args.policy_delay,
            model_a_path=args.model_a,
            model_c1_path=args.model_c1,
            model_c2_path=args.model_c2,
        )
    else:
        raise NameError(f"Algorithm '{args.algorithm}' is not defined")

    # replay buffer
    rpm = ReplayBuffer(
        env.observation_space.shape,
        env.action_space.shape,
        args.replay_size,
    )

    print(env.action_space.low, env.action_space.high)

    # hlavny cyklus hry
    total_steps, total_episodes = 0, 0
    while total_steps < args.max_steps:
        done = False
        episode_reward, episode_timesteps = 0.0, 0

        # init env
        obs = env.reset()

        # reset noise
        agent.actor.reset_noise()

        # collect rollout
        while not done:
            # select action randomly or using policy network
            if total_steps < args.learning_starts:
                # warmup
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs).numpy()

            # perform action
            new_obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_timesteps += 1
            total_steps += 1

            # store interaction
            rpm.store(obs, action, reward, new_obs, done)

            # super critical !!!
            obs = new_obs

            # update models after episode
            if (
                total_steps > args.update_after
                and len(rpm) > args.batch_size
                and (total_steps % args.update_every) == 0
            ):
                agent.update(
                    rpm, args.batch_size, args.update_every, logging_wandb=args.wandb
                )

        # after each episode
        total_episodes += 1

        print(f"Epoch: {total_episodes}")
        print(f"EpsReward: {episode_reward}")
        print(f"EpsSteps: {episode_timesteps}")
        print(f"TotalInteractions: {total_steps}")
        if args.wandb == True:
            wandb.log(
                {
                    "epoch": total_episodes,
                    "score": episode_reward,
                    "steps": episode_timesteps,
                    "replayBuffer": len(rpm),
                }
            )

    # Save model to local drive
    if args.save is not None:
        agent.actor.model.save(f"{args.save}model_A_{args.environment}.h5")
        agent.critic_1.model.save(f"{args.save}model_C1_{args.environment}.h5")
        agent.critic_2.model.save(f"{args.save}model_C2_{args.environment}.h5")

    # zatvor prostredie
    env.close()
