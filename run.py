import argparse
import gym

from policy import Agent


my_parser = argparse.ArgumentParser(
    prog="python3 training.py",
    description="RL training toolkit",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

my_parser.add_argument(
    "-env",
    "--environment",
    type=str,
    help="Only OpenAI Gym/PyBullet environments are available!",
    default="BipedalWalker-v3",
)

# nacitaj zadane argumenty programu
args = my_parser.parse_args()

# Herne prostredie
env = gym.make(args.environment)


agent = Agent(env=env, max_steps=1000000)

try:
    # run training process
    agent.run()
except KeyboardInterrupt:
    print("Terminated by user ... Bay bay")
finally:
    # zatvor prostredie
    env.close()