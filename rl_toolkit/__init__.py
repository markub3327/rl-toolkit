import argparse

import yaml


def main():
    my_parser = argparse.ArgumentParser(
        prog="python3 -m rl_toolkit",
        description="The RL-Toolkit: A toolkit for developing and comparing your reinforcement learning agents in various environments.",  # noqa
    )
    my_parser.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Name of the environment",
        required=True,
    )
    my_parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path Configuration File",
        required=True,
    )
    my_parser.add_argument(
        "-a",
        "--agent",
        type=str,
        help="Method (SAC, DQN, etc.)",
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
    parser_server.add_argument("--model_path", type=str, help="Path to saved model")

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
        help="Maximum number of agent's steps",
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

    # select method
    if args.agent == "sac":
        from rl_toolkit.agents.sac import Agent, Learner, Server, Tester
    elif args.agent == "dqn":
        from rl_toolkit.agents.dueling_dqn import Agent, Learner, Server, Tester
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Server mode
    if args.mode == "server":
        if args.agent == "sac":
            agent = Server(
                env_name=args.environment,
                port=config["Server"]["port"],
                actor_units=config["Actor"]["units"],
                critic_units=config["Critic"]["units"],
                clip_mean_min=config["Actor"]["clip_mean_min"],
                clip_mean_max=config["Actor"]["clip_mean_max"],
                n_quantiles=config["Critic"]["n_quantiles"],
                merge_index=config["Critic"]["merge_index"],
                top_quantiles_to_drop=config["Critic"]["top_quantiles_to_drop"],
                n_critics=config["Critic"]["count"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                init_alpha=config["Alpha"]["init"],
                init_noise=config["Actor"]["init_noise"],
                min_replay_size=config["Agent"]["warmup_steps"],
                max_replay_size=config["Server"]["max_replay_size"],
                samples_per_insert=config["Server"]["samples_per_insert"],
                actor_critic_path=args.model_path,
                db_path=config["db_path"],
            )
        elif args.agent == "dqn":
            agent = Server(
                env_name=args.environment,
                port=config["Server"]["port"],
                num_layers=config["Model"]["num_layers"],
                embed_dim=config["Model"]["embed_dim"],
                ff_mult=config["Model"]["ff_mult"],
                num_heads=config["Model"]["num_heads"],
                dropout_rate=config["Model"]["dropout_rate"],
                attention_dropout_rate=config["Model"]["attention_dropout_rate"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                timesteps=config["Model"]["timesteps"],
                min_replay_size=config["Agent"]["warmup_steps"],
                max_replay_size=config["Server"]["max_replay_size"],
                samples_per_insert=config["Server"]["samples_per_insert"],
                model_path=args.model_path,
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
        if args.agent == "sac":
            agent = Agent(
                env_name=args.environment,
                db_server=f"{args.db_server}:{config['Server']['port']}",
                actor_units=config["Actor"]["units"],
                clip_mean_min=config["Actor"]["clip_mean_min"],
                clip_mean_max=config["Actor"]["clip_mean_max"],
                init_noise=config["Actor"]["init_noise"],
                warmup_steps=config["Agent"]["warmup_steps"],
                env_steps=config["Agent"]["env_steps"],
                save_path=config["save_path"],
            )
        elif args.agent == "dqn":
            agent = Agent(
                env_name=args.environment,
                db_server=f"{args.db_server}:{config['Server']['port']}",
                num_layers=config["Model"]["num_layers"],
                embed_dim=config["Model"]["embed_dim"],
                ff_mult=config["Model"]["ff_mult"],
                num_heads=config["Model"]["num_heads"],
                dropout_rate=config["Model"]["dropout_rate"],
                attention_dropout_rate=config["Model"]["attention_dropout_rate"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                temp_init=config["Agent"]["temp_init"],
                temp_min=config["Agent"]["temp_min"],
                temp_decay=config["Agent"]["temp_decay"],
                warmup_steps=config["Agent"]["warmup_steps"],
                save_path=config["save_path"],
            )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            # Save model
            agent.save()
            agent.close()

    # Learner mode
    elif args.mode == "learner":
        if args.agent == "sac":
            agent = Learner(
                env_name=args.environment,
                db_server=f"{args.db_server}:{config['Server']['port']}",
                train_steps=config["Learner"]["train_steps"],
                batch_size=config["Learner"]["batch_size"],
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
                actor_global_clipnorm=config["Actor"]["global_clipnorm"],
                critic_global_clipnorm=config["Critic"]["global_clipnorm"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                init_alpha=config["Alpha"]["init"],
                init_noise=config["Actor"]["init_noise"],
                merge_index=config["Critic"]["merge_index"],
                save_path=config["save_path"],
            )
        elif args.agent == "dqn":
            agent = Learner(
                env_name=args.environment,
                db_server=f"{args.db_server}:{config['Server']['port']}",
                train_steps=config["Learner"]["train_steps"],
                batch_size=config["Learner"]["batch_size"],
                num_layers=config["Model"]["num_layers"],
                embed_dim=config["Model"]["embed_dim"],
                ff_mult=config["Model"]["ff_mult"],
                num_heads=config["Model"]["num_heads"],
                dropout_rate=config["Model"]["dropout_rate"],
                attention_dropout_rate=config["Model"]["attention_dropout_rate"],
                learning_rate=config["Model"]["learning_rate"],
                global_clipnorm=config["Model"]["global_clipnorm"],
                weight_decay=config["Model"]["weight_decay"],
                warmup_steps=config["Learner"]["warmup_steps"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                save_path=config["save_path"],
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
        if args.agent == "sac":
            agent = Tester(
                env_name=args.environment,
                render=args.render,
                max_steps=args.max_steps,
                actor_units=config["Actor"]["units"],
                clip_mean_min=config["Actor"]["clip_mean_min"],
                clip_mean_max=config["Actor"]["clip_mean_max"],
                init_noise=config["Actor"]["init_noise"],
                model_path=args.model_path,
                enable_wandb=True,
            )
        elif args.agent == "dqn":
            agent = Tester(
                env_name=args.environment,
                render=args.render,
                max_steps=args.max_steps,
                num_layers=config["Model"]["num_layers"],
                embed_dim=config["Model"]["embed_dim"],
                ff_mult=config["Model"]["ff_mult"],
                num_heads=config["Model"]["num_heads"],
                dropout_rate=config["Model"]["dropout_rate"],
                attention_dropout_rate=config["Model"]["attention_dropout_rate"],
                gamma=config["Learner"]["gamma"],
                tau=config["Learner"]["tau"],
                model_path=args.model_path,
                enable_wandb=True,
            )

        try:
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            agent.close()
