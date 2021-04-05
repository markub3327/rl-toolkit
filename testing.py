import argparse
import gym
import pybullet_envs

# policy
from policy import SAC

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
    my_parser.add_argument("--wandb", action="store_true", help="Logging to wanDB")
    my_parser.add_argument("-f", "--model", type=str, help="Actor's model file")

    # nacitaj zadane argumenty programu
    args = my_parser.parse_args()

    # Herne prostredie
    env = gym.make(args.environment)

    # init policy
    if args.policy == "sac":
        agent = SAC(
            env=env,
            max_steps=args.max_steps,
            model_a_path=args.model,
            logging_wandb=args.wandb,
        )
    else:
        raise NameError(f"Algorithm '{args.policy}' is not defined")

    try:
        # run testing process
        agent.test()
    except KeyboardInterrupt:
        print("Terminated by user ... Bay bay")
    finally:
        # zatvor prostredie
        env.close()
