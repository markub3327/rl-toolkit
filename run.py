import argparse
import gym

from policy import Agent
from policy import Learner


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

learner = Learner(
    env=env,
    max_steps=1000000
)

#learner.run()
agent = Agent()

agent.run(1000000)