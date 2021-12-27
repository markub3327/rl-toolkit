import argparse

import yaml

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
    my_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path Configuration File",
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

    # create the parser for the "agent" sub-command
    parser_agent = sub_parsers.add_parser(
        "agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Agent process",
    )
    parser_agent.add_argument(
        "--db_server",
        type=str,
        help="Database server name or IP address (e.g. localhost or 192.168.1.1)",
        default="localhost",
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
        "--db_server",
        type=str,
        help="Database server name or IP address (e.g. localhost or 192.168.1.1)",
        default="localhost",
    )
    parser_learner.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
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

    # load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Server mode
    if args.mode == "server":
        agent = Server(
            env_name=args.environment,
            port=config["port"],
            actor_units=config["Actor"]["units"],
            clip_mean_min=config["Actor"]["clip_mean_min"],
            clip_mean_max=config["Actor"]["clip_mean_max"],
            init_noise=config["Actor"]["init_noise"],
            min_replay_size=config["min_replay_size"],
            max_replay_size=config["max_replay_size"],
            samples_per_insert=config["samples_per_insert"],
            db_path=config["db_path"],
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
            db_server=f"{args.db_server}:{config['port']}",
            actor_units=config["Actor"]["units"],
            clip_mean_min=config["Actor"]["clip_mean_min"],
            clip_mean_max=config["Actor"]["clip_mean_max"],
            init_noise=config["Actor"]["init_noise"],
            warmup_steps=config["warmup_steps"],
            env_steps=config["env_steps"],
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
            db_server=f"{args.db_server}:{config['port']}",
            train_steps=config["train_steps"],
            batch_size=config["batch_size"],
            actor_units=config["Actor"]["units"],
            critic_units=config["Critic"]["units"],
            actor_learning_rate=config["Actor"]["learning_rate"],
            critic_learning_rate=config["Critic"]["learning_rate"],
            alpha_learning_rate=config["Alpha"]["learning_rate"],
            n_quantiles=config["Critic"]["n_quantiles"],
            top_quantiles_to_drop=config["Critic"]["top_quantiles_to_drop"],
            n_critics=config["Critic"]["count"],
            clip_mean_min=config["Actor"]["clip_mean_min"],
            clip_mean_max=config["Actor"]["clip_mean_max"],
            gamma=config["gamma"],
            tau=config["tau"],
            init_alpha=config["Alpha"]["init"],
            init_noise=config["Actor"]["init_noise"],
            save_path=config["save_path"],
            model_path=args.model_path,
            log_interval=config["log_interval"],
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
            render=args.render,
            max_steps=args.max_steps,
            actor_units=config["Actor"]["units"],
            clip_mean_min=config["Actor"]["clip_mean_min"],
            clip_mean_max=config["Actor"]["clip_mean_max"],
            init_noise=config["Actor"]["init_noise"],
            model_path=args.model_path,
        )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()
