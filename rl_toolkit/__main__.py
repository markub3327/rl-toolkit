import argparse

from rl_toolkit.training import Training

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

    # create sub-parser for selecting mode
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode",
        dest="mode",
        required=True,
    )

    # create the parser for the "training" sub-command
    parser_training = sub_parsers.add_parser(
        "training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Training mode",
    )
    parser_training.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_training.add_argument(
        "--gamma",
        type=float,
        help="Discount rate (gamma) for future rewards",
        default=0.99,
    )
    parser_training.add_argument(
        "--tau", type=float, help="Soft update rate", default=0.01
    )
    parser_training.add_argument(
        "--init_alpha",
        type=float,
        help="Initialization value of alpha (entropy coeff)",
        default=1.0,
    )
    parser_training.add_argument(
        "--actor_learning_rate",
        type=float,
        help="Learning rate for actor network",
        default=7.3e-4,
    )
    parser_training.add_argument(
        "--critic_learning_rate",
        type=float,
        help="Learning rate for critic network",
        default=7.3e-4,
    )
    parser_training.add_argument(
        "--alpha_learning_rate",
        type=float,
        help="Learning rate for alpha parameter",
        default=7.3e-4,
    )
    parser_training.add_argument(
        "-bs", "--batch_size", type=int, help="Size of the mini-batch", default=256
    )
    parser_training.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="Path for saving model files",
        default="./save/model",
    )
    parser_training.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
    )
    parser_training.add_argument(
        "--db_path", type=str, help="DB's checkpoints path", default="./save/db"
    )
    parser_training.add_argument(
        "--wandb", action="store_true", help="Log into WanDB cloud"
    )

    # create the parser for the "tester" sub-command
    parser_tester = sub_parsers.add_parser(
        "tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tester mode",
    )
    parser_tester.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_tester.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    parser_tester.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
    )
    parser_tester.add_argument(
        "--wandb", action="store_true", help="Log into WanDB cloud"
    )

    # nacitaj zadane argumenty
    args = my_parser.parse_args()

    # training mode
    if args.mode == "training":
        agent = Training(
            env_name=args.environment,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            env_steps=args.env_steps,
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

    # Tester mode
    elif args.mode == "tester":
        agent = Tester(
            env_name=args.environment,
            max_steps=args.max_steps,
            render=args.render,
            model_path=args.model_path,
            log_wandb=args.wandb,
        )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()
