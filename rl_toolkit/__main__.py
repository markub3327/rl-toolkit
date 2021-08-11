import argparse

from rl_toolkit import Agent

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(
        prog="python3 -m rl_toolkit",
        description="The RL-Toolkit: A toolkit for developing and comparing your reinforcement learning agents in various games (OpenAI Gym or Pybullet).",  # noqa
    )
    my_parser.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Only OpenAI Gym/PyBullet environments are available!",
        required=True,
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    my_parser.add_argument(
        "--env_steps",
        type=int,
        help="Number of steps per rollout",
        default=8,
    )
    my_parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of steps before using policy network",
        default=int(1e4),
    )
    my_parser.add_argument(
        "--gamma",
        type=float,
        help="Discount rate (gamma) for future rewards",
        default=0.99,
    )
    my_parser.add_argument("--tau", type=float, help="Soft update rate", default=0.01)
    my_parser.add_argument(
        "--init_alpha",
        type=float,
        help="Initialization value of alpha (entropy coeff)",
        default=1.0,
    )
    my_parser.add_argument(
        "--actor_learning_rate",
        type=float,
        help="Learning rate for actor network",
        default=7.3e-4,
    )
    my_parser.add_argument(
        "--critic_learning_rate",
        type=float,
        help="Learning rate for critic network",
        default=7.3e-4,
    )
    my_parser.add_argument(
        "--alpha_learning_rate",
        type=float,
        help="Learning rate for alpha parameter",
        default=7.3e-4,
    )
    my_parser.add_argument(
        "--buffer_capacity",
        type=int,
        help="Maximal capacity of replay buffer",
        default=int(1e6),
    )
    my_parser.add_argument(
        "-bs", "--batch_size", type=int, help="Size of the mini-batch", default=256
    )
    my_parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="Path for saving model files",
        default="./save/model",
    )
    my_parser.add_argument("-f", "--model_path", type=str, help="Path to saved model")
    my_parser.add_argument(
        "--db_path", type=str, help="DB's checkpoints path", default="./save/db"
    )
    my_parser.add_argument("--wandb", action="store_true", help="Log into WanDB cloud")

    # nacitaj zadane argumenty
    args = my_parser.parse_args()

    agent = Agent(
        env_name=args.environment,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        env_steps=args.env_steps,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        init_alpha=args.init_alpha,
        model_path=args.model_path,
        save_path=args.save_path,
        log_wandb=args.wandb,
    )

    try:
        agent.run()
    except KeyboardInterrupt:
        print("Terminated by user 👋👋👋")
    finally:
        agent.save()
        agent.close()
