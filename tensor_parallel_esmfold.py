import os
import torch
import transformers
import deepspeed
from transformers.models.t5.modeling_t5 import T5Block
import torch.distributed as dist
from transformers import AutoTokenizer, EsmForProteinFolding
'''
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12355
'''
dist.init_process_group(backend='nccl')
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "4"))
# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

# Initialize the DeepSpeed-Inference engine
model.esm = deepspeed.init_inference(
    model.esm,
    mp_size=world_size,
    dtype=torch.float,
)

sequence = 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVLGYNIVATPRGYVLAGG'

inputs = tokenizer(sequence , return_tensor="pt", add_special_tokens=False)
device = torch.device('cuda' if  torch.cuda.is_avaible() else 'cpu')
model = model.to(device)
inputs = {k:v.to(device) for k,v in inputs.items()}

output = model(**inputs)

print(output)