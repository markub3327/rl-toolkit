# main libraries
import argparse

# windows
from testing import main as main_testing
from training import main as main_training


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(prog='python3 main.py', description='stable-baselines', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # init args
    my_parser.add_argument('--model_a', type=str, help="Actor's model file")
    my_parser.add_argument('--model_c1', type=str, help="Critic 1's model file")
    my_parser.add_argument('--model_c2', type=str, help="Critic 2's model file")
    my_parser.add_argument('-alg', '--algorithm', type=str, help='Only OpenAI Gym environments are available!', default='sac')
    my_parser.add_argument('-env', '--environment', type=str, help='Only OpenAI Gym environments are available!', default='BipedalWalker-v3')
    my_parser.add_argument('-t', '--max_steps', type=int, help='Maximum number of interactions doing in environment', default=int(1e6))
    my_parser.add_argument('--noise_type', type=str, help='Type of used noise generator [normal, ornstein-uhlenbeck] (only for TD3)', default='ornstein-uhlenbeck')
    my_parser.add_argument('--action_noise', type=float, help='Standard deviation of action noise (only for TD3)', default=0.5)
    my_parser.add_argument('--gamma', type=float, help='Discount factor', default=0.99)
    my_parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=3e-4)
    my_parser.add_argument('--tau', type=float, help='Soft update learning rate', default=0.005)
    my_parser.add_argument('--batch_size', type=int, help='Size of the batch', default=256)
    my_parser.add_argument('--replay_size', type=int, help='Size of the replay buffer', default=int(1e6))
    my_parser.add_argument('--learning_starts', type=int, help='Number of steps before using policy network', default=10000)
    my_parser.add_argument('--update_after', type=int, help='Number of steps before training', default=10000)
    my_parser.add_argument('--target_noise', type=float, help='Standard deviation of target noise (only for TD3)', default=0.2)
    my_parser.add_argument('--noise_clip', type=float, help='Limit for target noise (only for TD3)', default=0.5)
    my_parser.add_argument('--policy_delay', type=int, help='Delay between critic and policy update', default=2)
    my_parser.add_argument('--wandb', action='store_true', help='Logging to wanDB')
    my_parser.add_argument('--test', action='store_true', help='Run test mode')
    my_parser.add_argument('-s', '--save', type=str, help='Path for saving model files')

    # get args
    args = my_parser.parse_args()
    print(args)

    # run testing mode
    if (args.test and args.model_a != None):
        main_testing(env_name=args.environment,
                     alg=args.algorithm,
                     max_steps=args.max_steps,
                     logging_wandb=args.wandb,
                     model_a_path=args.model_a)
    # run training mode
    elif (args.test == False):
        main_training(env_name=args.environment,
                      alg=args.algorithm,
                      learning_rate=args.learning_rate,
                      gamma=args.gamma,
                      batch_size=args.batch_size,
                      tau=args.tau,
                      replay_size=args.replay_size,
                      learning_starts=args.learning_starts,
                      update_after=args.update_after,
                      max_steps=args.max_steps,
                      noise_type=args.noise_type,
                      action_noise=args.action_noise,
                      target_noise=args.target_noise,
                      noise_clip=args.noise_clip,
                      policy_delay=args.policy_delay,
                      logging_wandb=args.wandb,
                      save_path=args.save,
                      model_a_path=args.model_a,
                      model_c1_path=args.model_c1,
                      model_c2_path=args.model_c2)