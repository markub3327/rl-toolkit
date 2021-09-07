import argparse

from rl_toolkit.core import Agent, Learner, Server, Tester

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

    # create the parser for the "server" sub-command
    parser_server = sub_parsers.add_parser(
        "server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Server process",
    )
    parser_server.add_argument(
        "--min_replay_size",
        type=int,
        help="Minimum number of samples in memory before learning starts",
        default=int(1e4),
    )
    parser_server.add_argument(
        "--max_replay_size",
        type=int,
        help="Maximal capacity of memory",
        default=int(1e6),
    )
    parser_server.add_argument(
        "--samples_per_insert",
        type=float,
        help="Samples per insert ratio (SPI)",
        default=32.0,
    )
    parser_server.add_argument(
        "--db_path", type=str, help="DB's checkpoints path", default="./save/db"
    )

    # create the parser for the "agent" sub-command
    parser_agent = sub_parsers.add_parser(
        "agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Agent process",
    )
    parser_agent.add_argument(
        "--db_server", type=str, help="DB server name", default="localhost"
    )
    parser_agent.add_argument(
        "--env_steps",
        type=int,
        help="Number of steps per rollout",
        default=8,
    )
    parser_agent.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of steps before using policy network",
        default=int(1e4),
    )
    parser_agent.add_argument(
        "--render", action="store_true", help="Render the environment"
    )

    # create the parser for the "learner" sub-command
    parser_learner = sub_parsers.add_parser(
        "learner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Learner process",
    )
    parser_learner.add_argument(
        "--db_server", type=str, help="DB server name", default="localhost"
    )
    parser_learner.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_learner.add_argument(
        "--gamma",
        type=float,
        help="Discount rate (gamma) for future rewards",
        default=0.99,
    )
    parser_learner.add_argument(
        "--tau", type=float, help="Soft update rate", default=0.01
    )
    parser_learner.add_argument(
        "--init_alpha",
        type=float,
        help="Initialization value of alpha (entropy coeff)",
        default=1.0,
    )
    parser_learner.add_argument(
        "--actor_learning_rate",
        type=float,
        help="Learning rate for actor network",
        default=7.3e-4,
    )
    parser_learner.add_argument(
        "--critic_learning_rate",
        type=float,
        help="Learning rate for critic network",
        default=7.3e-4,
    )
    parser_learner.add_argument(
        "--alpha_learning_rate",
        type=float,
        help="Learning rate for alpha parameter",
        default=7.3e-4,
    )
    parser_learner.add_argument(
        "--curiosity_learning_rate",
        type=float,
        help="Learning rate for curiosity network",
        default=1e-3,
    )
    parser_learner.add_argument(
        "--batch_size", type=int, help="Size of the mini-batch", default=256
    )
    parser_learner.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="Path for saving model files",
        default="./save/model",
    )
    parser_learner.add_argument(
        "--actor_critic_model_path", type=str, help="Path to saved actor-critic model"
    )
    parser_learner.add_argument(
        "--curiosity_model_path", type=str, help="Path to saved curiosity model"
    )

    # create the parser for the "tester" sub-command
    parser_tester = sub_parsers.add_parser(
        "tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tester process",
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

    # nacitaj zadane argumenty
    args = my_parser.parse_args()

    # Server mode
    if args.mode == "server":
        agent = Server(
            env_name=args.environment,
            min_replay_size=args.min_replay_size,
            samples_per_insert=args.samples_per_insert,
            max_replay_size=args.max_replay_size,
            db_path=args.db_path,
        )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()

    # Agent mode
    elif args.mode == "agent":
        agent = Agent(
            env_name=args.environment,
            render=args.render,
            db_server=args.db_server,
            warmup_steps=args.warmup_steps,
            env_steps=args.env_steps,
        )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()

    # Learner mode
    elif args.mode == "learner":
        agent = Learner(
            env_name=args.environment,
            db_server=args.db_server,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            alpha_learning_rate=args.alpha_learning_rate,
            curiosity_learning_rate=args.curiosity_learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            init_alpha=args.init_alpha,
            save_path=args.save_path,
            actor_critic_model_path=args.actor_critic_model_path,
            curiosity_model_path=args.curiosity_model_path,
            log_interval=1000,
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
        )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()
