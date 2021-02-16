from rl_training import RLTraining

import argparse


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 training.py",
        description="RL training toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # init args
    my_parser.add_argument(
        "-alg",
        "--policy",
        type=str,
        help="The policy of agent (SAC/TD3)",
        default="sac",
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
        "--env_steps", type=int, help="Num. of environment steps", default=64
    )
    my_parser.add_argument(
        "--gradient_steps", type=int, help="Num. of gradient steps", default=64
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

    rl_training = RLTraining(
        env_name=args.environment,
        max_steps=args.max_steps,
        env_steps=args.env_steps,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        policy=args.policy,
        update_after=args.update_after,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
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
        logging_wandb=args.wandb
    )

    try:
        # run training process
        rl_training.train()
    except KeyboardInterrupt:
        print("Terminated by user ... Bay bay")
    finally:
        # zatvor prostredie
        rl_training.close()

        # save models
        if args.save is not None:
            rl_training.save(args.save)