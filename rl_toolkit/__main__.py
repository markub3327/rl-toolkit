import argparse
import gym
import pybullet_envs  # noqa

# policy
from rl_toolkit.policy import Learner, Agent, Tester

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 -m rl_toolkit",
        description="The RL-Toolkit: A toolkit for developing and comparing your reinforcement learning agents in various games (OpenAI Gym or Pybullet).",  # noqa
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    my_parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Only OpenAI Gym/PyBullet environments are available!",
        required=True,
    )
    my_parser.add_argument(
        "--mode",
        choices=["agent", "learner", "tester"],
        help="Choose operating mode",
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
        "--gamma",
        type=float,
        help="Discount rate (gamma) for future rewards",
        default=0.99,
    )
    my_parser.add_argument(
        "--tau",
        type=float,
        help="Parameter for soft target network updates",
        default=0.01,
    )
    my_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate for policy & critic networks",
        default=7.3e-4,
    )
    my_parser.add_argument(
        "--batch_size", type=int, help="Size of the mini-batch", default=256
    )
    my_parser.add_argument(
        "--buffer_capacity",
        type=int,
        help="Maximum capacity of replay memory",
        default=int(1e6),
    )
    my_parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of steps before using policy network",
        default=10000,
    )
    my_parser.add_argument(
        "--update_interval", type=int, help="Interval of updating policy parameters", default=64
    )
    my_parser.add_argument(
        "--log_interval", type=int, help="Log into console interval", default=64
    )
    my_parser.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    my_parser.add_argument("--wandb", action="store_true", help="Log into WanDB cloud")
    my_parser.add_argument(
        "-s", "--save_path", type=str, help="Path for saving model files"
    )
    my_parser.add_argument("-a", "--actor_path", type=str, help="Actor's model file")
    my_parser.add_argument("-c", "--critic_path", type=str, help="Critic's model file")
    my_parser.add_argument("--db_path", type=str, help="DB's checkpoints path")
    my_parser.add_argument(
        "--db_server", type=str, help="DB server name", default="localhost"
    )

    # nacitaj zadane argumenty programu
    args = my_parser.parse_args()

    # Herne prostredie
    env = gym.make(args.environment)

    print("Action space:")
    print(env.action_space)
    print(env.action_space.low, env.action_space.high)
    print()
    print("Observation space:")
    print(env.observation_space)
    print(env.observation_space.low, env.observation_space.high)
    print()

    # Agent mode
    if args.mode == "agent":
        agent = Agent(
            db_server=args.db_server,
            env=env,
            update_interval=args.update_interval,
            warmup_steps=args.warmup_steps,
            log_wandb=args.wandb,
        )

        try:
            # run actor process
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            # zatvor herne prostredie
            env.close()

    # Learner mode
    if args.mode == "learner":
        agent = Learner(
            env=env,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            actor_learning_rate=args.learning_rate,
            critic_learning_rate=args.learning_rate,
            alpha_learning_rate=args.learning_rate,
            tau=args.tau,
            gamma=args.gamma,
            actor_path=args.actor_path,
            critic_path=args.critic_path,
            db_path=args.db_path,
            save_path=args.save_path,
            log_wandb=args.wandb,
            log_interval=args.log_interval,
        )

        try:
            # run training process
            agent.run()
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            # zatvor herne prostredie
            env.close()

            # save models and snapshot of the database
            agent.save()

    # Test mode
    elif args.mode == "tester":
        agent = Tester(
            env=env,
            max_steps=args.max_steps,
            model_a_path=args.model_a,
            log_wandb=args.wandb,
        )

        try:
            # run actor process
            agent.run(render=args.render)
        except KeyboardInterrupt:
            print("Terminated by user 👋👋👋")
        finally:
            # zatvor herne prostredie
            env.close()
