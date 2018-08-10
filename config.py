
import argparse

def get_args():
  """
      The argument parser
  """
  parser = argparse.ArgumentParser()

  parser.add_argument('--random_seed', type=int, default=827, help='Random seed')

  parser.add_argument('--data_dir', type=str, default='data', help='data path')
  parser.add_argument('--data_type', type=str, default='citeseer', help='dataset')
  parser.add_argument('--log_dir', type=str, default='save/logs', help='log path')
  parser.add_argument('--save_dir', type=str, default='save/saves', help='save path')

  parser.add_argument('--hidden_size', type=int, default=16, help="number of units in hidden layer")

  parser.add_argument('--batch_size', type=int, default=100, help='Train batch size')
  parser.add_argument('--anneal', type=bool, default=True, help='whether to anneal')
  parser.add_argument('--anneal_start', type=int, default=50, help='anneal start epoch')
  parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay rate')
  parser.add_argument('--nb_epoch', type=int, default=20, help='The number of epoch') #mni 70
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--print_step', type=int, default=100, help='number step to print')
  parser.add_argument('--summary_step', type=int, default=100, help='number step to summar')
  parser.add_argument('--save_epoch', type=int, default=2, help='number epoch to save')

  return parser.parse_args()