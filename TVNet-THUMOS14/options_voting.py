import argparse

parser = argparse.ArgumentParser(description='TVNet')
parser.add_argument('--batch-size', type=int, default=256, help='number of instances in a batch of data (default: 256)')
parser.add_argument('--voting_type', type=str, default='end', choices=['start','end'], help='voting for start or end point')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--window_length',type=int, default=10, help='window length during training (default: 10)')
parser.add_argument('--window_stride', type=int, default=5, help='window stride during training (default: 5)')
