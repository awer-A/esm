import argparse
import time
import random
import torch
from torch import nn

import deepspeed
from deepspeed.pipe import PipelineModule
parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int, default=-1, help="")

in_features = 32
hidden_dim = 32
out_features = 32

deepspeed.init_distributed(dist_backend='nccl')

net = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Linear(hidden_dim, out_features)
)
net = PipelineModule(layers=net, num_stages=2)
ds_config = {
    "train_batch_size": 1,
}


def random_data():
    for _ in range(4):
        yield (torch.randn(1, in_features), 0)

engine, _, _, _ = deepspeed.initialize(model=net, config=ds_config)
output = engine.eval_batch(random_data(), compute_loss = False)

time.sleep(random.uniform(0, 5))
print("stage_id: ", engine.stage_id, "output:", output, "grad_fn:", output.grad_fn)