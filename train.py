import argparse
import gym

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
my_parser.add_argument(
    "-t",
    "--max_steps",
    type=int,
    help="Maximum number of interactions doing in environment",
    default=int(1e6),
)
my_parser.add_argument("-s", "--save", type=str, help="Path for saving model files")
my_parser.add_argument("--model_a", type=str, help="Actor's model file")
my_parser.add_argument("--model_c1", type=str, help="Critic 1's model file")
my_parser.add_argument("--model_c2", type=str, help="Critic 2's model file")

# nacitaj zadane argumenty programu
args = my_parser.parse_args()

# Herne prostredie
env = gym.make(args.environment)

# init learner
learner = Learner(
    env=env,
    max_steps=args.max_steps,
    model_a_path=args.model_a,
    model_c1_path=args.model_c1,
    model_c2_path=args.model_c2,
)

try:
    # run training process
    learner.run()
except KeyboardInterrupt:
    print("Terminated by user ... Bay bay")
finally:
    # zatvor prostredie
    env.close()
    # save models
    if args.save is not None:
        learner.save(args.save)
