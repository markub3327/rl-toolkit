import argparse
import gym
import pybullet_envs
import reverb

# policy
from policy import SAC

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 actor.py",
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
        "-db",
        "--db_address",
        type=str,
        help="IP address or hostname of the Reverb server",
        default="localhost",
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Maximum number of interactions doing in environment",
        default=int(1e6),
    )
    my_parser.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    my_parser.add_argument("--wandb", action="store_true", help="Logging to wanDB")
    my_parser.add_argument("-f", "--model", type=str, help="Actor's model file")

    # nacitaj zadane argumenty programu
    args = my_parser.parse_args()

    # Herne prostredie
    env = gym.make(args.environment)

    # Initializes the reverb client
    db_client = reverb.Client(f'{args.db_address}:8000')
    print(db_client.server_info())

    # init policy
    if args.policy == "sac":
        agent = SAC(
            env=env,
            db=db_client,
            max_steps=args.max_steps,
            model_a_path=args.model,
            logging_wandb=args.wandb,
        )
    else:
        raise NameError(f"Algorithm '{args.policy}' is not defined")

    try:
        # run testing process
        agent.test(render=args.render)
    except KeyboardInterrupt:
        print("Terminated by user ... Bay bay")
    finally:
        # zatvor prostredie
        env.close()
