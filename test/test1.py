import torch as th
import torch.nn.functional as F
import torch.distributed as dist
import sys
import os


rank = os.environ.get("LOCAL_RANK")
rank = 0 if rank is None else int(rank)
dist.init_process_group(backend='gloo')

s_list = th.tensor([1., 2.])
s_list = [item.reshape(1, 1) for item in s_list]
s = th.zeros(1)
if rank == 0:
    dist.scatter(s, s_list)
else:
    dist.scatter(s)

print(s)
