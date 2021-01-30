import sys
import gym
import pybullet_envs
import wandb
import argparse

# policy
from sac import SAC
from td3 import TD3

# utilities
from utils import ReplayBufferReader

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 main.py",
        description="RL training toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # init args
    my_parser.add_argument(
        "-alg",
        "--algorithm",
        type=str,
        help="Select the algorithm, that making a decisions",
        default="sac",
    )
    my_parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Only OpenAI Gym & PyBullet environments are available!",
        default="BipedalWalker-v3",
    )
    my_parser.add_argument(
        "-db", "--database", type=str, help="Database name", default="rl-agents"
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Maximum number of interactions doing in environment",
        default=int(1e6),
    )
    my_parser.add_argument("--gamma", type=float, help="Discount factor", default=0.98)
    my_parser.add_argument(
        "-lr", "--learning_rate", type=float, help="Learning rate", default=7.3e-4
    )
    my_parser.add_argument("--tau", type=float, help="Soft update rate", default=0.02)
    my_parser.add_argument(
        "--batch_size", type=int, help="Size of the batch", default=256
    )
    my_parser.add_argument(
        "--target_noise",
        type=float,
        help="Standard deviation of target noise (only for TD3)",
        default=0.2,
    )
    my_parser.add_argument(
        "--noise_clip",
        type=float,
        help="Limit for target noise (only for TD3)",
        default=0.5,
    )
    my_parser.add_argument(
        "--policy_delay",
        type=int,
        help="Delay between critic and policy update",
        default=2,
    )
    my_parser.add_argument("--wandb", action="store_true", help="Logging to wanDB")
    my_parser.add_argument("-s", "--save", type=str, help="Path for saving model files")
    my_parser.add_argument("--model_a", type=str, help="Actor's model file")
    my_parser.add_argument("--model_c1", type=str, help="Critic 1's model file")
    my_parser.add_argument("--model_c2", type=str, help="Critic 2's model file")

    # Get args
    args = my_parser.parse_args()
    print(args)

    # Herne prostredie
    env = gym.make(args.environment)

    # Init Weights & Biases
    if args.wandb:
        wandb.init(project="rl-toolkit")

        # Settings
        wandb.config.gamma = args.gamma
        wandb.config.batch_size = args.batch_size
        wandb.config.tau = args.tau
        wandb.config.learning_rate = args.learning_rate
        wandb.config.max_steps = args.max_steps
        wandb.config.target_noise = args.target_noise
        wandb.config.noise_clip = args.noise_clip
        wandb.config.policy_delay = args.policy_delay

    # Policy
    if args.algorithm == "td3":
        agent = TD3(
            env.observation_space.shape,
            env.action_space.shape,
            learning_rate=args.learning_rate,
            tau=args.tau,
            gamma=args.gamma,
            target_noise=args.target_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            model_a_path=args.model_a,
            model_c1_path=args.model_c1,
            model_c2_path=args.model_c2,
        )
    elif args.algorithm == "sac":
        agent = SAC(
            env.observation_space.shape,
            env.action_space.shape,
            actor_learning_rate=args.learning_rate,
            critic_learning_rate=args.learning_rate,
            alpha_learning_rate=args.learning_rate,
            tau=args.tau,
            gamma=args.gamma,
            policy_delay=args.policy_delay,
            model_a_path=args.model_a,
            model_c1_path=args.model_c1,
            model_c2_path=args.model_c2,
        )
    else:
        raise NameError(f"Algorithm '{args.algorithm}' is not defined")

    # uloz inicializacny model
    agent.actor.model.save(f"{args.save}model_A_{args.environment}.h5")
    print("Actor's init model saved successful 😊")

    # replay buffer
    rpm = ReplayBufferReader(
        size=args.batch_size * 64,
        obs_dim=env.observation_space.shape,
        act_dim=env.action_space.shape,
        env_name=args.environment,
        db_name=args.database,
        server_name="192.168.1.2",
    )

    print(env.action_space.low, env.action_space.high)

    # hlavny cyklus ucenia
    try:
        total_steps = 0
        while total_steps < args.max_steps or len(rpm) < args.max_steps:
            if len(rpm) >= 10000:
                # synchronizuj s DB
                rpm.sync()

                # pozastav
                agent.actor.create_lock(
                    f"{args.save}model_A_{args.environment}_weights.h5"
                )

                # aktualizuj model
                agent.update(rpm, args.batch_size, 64, logging_wandb=args.wandb)

                # uloz novy model
                agent.actor.save_weights(
                    f"{args.save}model_A_{args.environment}_weights.h5"
                )

                # uvolni
                agent.actor.release_lock()

                print(f"Epoch: {total_steps}")
                print(f"ReplayBuffer: {len(rpm)}")

                total_steps += 64  # zapocitavaj iba iteracie ucenia
            else:
                sys.stdout.write("\rPopulating database up to 10000 samples...")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("Terminated by user! 👋")
    finally:
        # uloz Actor siet po treningu
        agent.actor.model.save(f"{args.save}model_A_{args.environment}.h5")
        print("Actor's model saved successful 😊")

        # uloz Critic siete po treningu
        agent.critic_1.model.save(f"{args.save}model_C1_{args.environment}.h5")
        print("Critic 1's model saved successful 😊")
        agent.critic_2.model.save(f"{args.save}model_C2_{args.environment}.h5")
        print("Critic 2's model saved successful 😊")

        # zatvor prostredie
        env.close()

        # uvolni zamok ak existoval
        agent.actor.release_lock()
