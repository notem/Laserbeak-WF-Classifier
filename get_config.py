import json
import torch
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
args = parser.parse_args()


pwd = args.ckpt


print(json.dumps(torch.load(pwd)['config'], indent=4))
