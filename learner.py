import argparse
import gym
import pybullet_envs
import reverb

# policy
from policy import SAC

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 learner.py",
        description="RL Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    my_parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Only OpenAI Gym/PyBullet environments are available!",
        default="BipedalWalker-v3",
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Maximum number of interactions doing in environment",
        default=int(1e6),
    )
    my_parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    my_parser.add_argument(
        "-lr", "--learning_rate", type=float, help="Learning rate", default=7.3e-4
    )
    my_parser.add_argument("--tau", type=float, help="Soft update rate", default=0.01)
    my_parser.add_argument(
        "--batch_size", type=int, help="Size of the batch", default=256
    )
    my_parser.add_argument(
        "--buffer_size", type=int, help="Size of the replay buffer", default=int(1e6)
    )
    my_parser.add_argument(
        "--learning_starts",
        type=int,
        help="Number of steps before using policy network",
        default=10000,
    )
    my_parser.add_argument(
        "--env_steps", type=int, help="Num. of environment steps", default=64
    )
    my_parser.add_argument(
        "--gradient_steps", type=int, help="Num. of gradient steps", default=64
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

    print("Action space:")
    print(env.action_space)
    print(env.action_space.low, env.action_space.high)
    print()
    print("Observation space:")
    print(env.observation_space)
    print(env.observation_space.low, env.observation_space.high)
    print()

    # Initialize the Reverb server
    db_server = reverb.Server(
        tables=[
            reverb.Table(
                name='tab1',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=args.buffer_size,
                rate_limiter=reverb.rate_limiters.MinSize(args.learning_starts)
            )
        ],
        port=8000
    )

    # init policy
    agent = SAC(
        env=env,
        db=db_server,
        max_steps=args.max_steps,
        env_steps=args.env_steps,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        actor_learning_rate=args.learning_rate,
        critic_learning_rate=args.learning_rate,
        alpha_learning_rate=args.learning_rate,
        tau=args.tau,
        gamma=args.gamma,
        model_a_path=args.model_a,
        model_c1_path=args.model_c1,
        model_c2_path=args.model_c2,
        logging_wandb=args.wandb,
    )

    try:
        # run training process
        agent.train()
    except KeyboardInterrupt:
        print("Terminated by user ... Bay bay")
    finally:
        # zatvor prostredie
        env.close()
        # save models
        if args.save is not None:
            agent.save(args.save)
