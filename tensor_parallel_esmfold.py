import os
import torch
import transformers
import deepspeed
from transformers.models.t5.modeling_t5 import T5Block
import torch.distributed as dist
'''
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12355
'''
dist.init_process_group(backend='nccl')
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "4"))

# create the model pipeline
pipe = transformers.pipeline("text-classification", model="facebook/esmfold_v1", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
)

output = pipe(['MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVLGYNIVATPRGYVLAGG', 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVLGYNIVATPRGYVLAGG'])
print(output)