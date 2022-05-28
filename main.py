import torch as th
import torch.distributed as dist
import argparse
import time
import sys
import os

from grid import condition
from multigrid import MultiGrid, FullMultiGrid
from input.func import origin_func, target_func
from smooth import smooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', '-n', type=int, default=16)
    parser.add_argument('--p', '-p', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    n, p = args.n // args.p, args.p
    device = th.device(args.device)
    rank = os.environ.get("LOCAL_RANK")
    rank = 0 if rank is None else int(rank)
    index = th.tensor([rank % p, rank // p], device=device)  # x, y

    if p > 1:
        assert p ** 2 == int(os.environ.get("WORLD_SIZE"))
        dist.init_process_group(backend='gloo')
    
    MG_method = MultiGrid(index, p, device)
    cond_method = condition(n, index, p, device)
    b = cond_method(origin_func, target_func)  # condition(origin_func)

    w = n - 1 if p == 1 else n
    u = th.zeros((w)**2, device=device)
    
    u_list = []
    for _ in range(10):
        u = MG_method(u, b, n=n)
        u_list.append(u)

    if rank == 0:
        print(u.view(w, w)[:8, :8])

    # e_list = [th.norm(u_list[i] - u_t) for i in range(len(u_list))]
    # print(e_list)
    # o_list = [th.log2(e_list[i]) - th.log2(e_list[i+1]) for i in range(len(e_list)-1)]
    # print(o_list)

    # print(u_c.view(n-1, n-1))

    # method = FullMultiGrid(n)
    # u = method(b, n=n)


    # u = th.rand((n-1)**2, device=device)
    # u_list = []
    # G_method = smooth(A, w=2.0/3)
    # for _ in range(200):
    #     u = G_method(u, b, m=1)
    #     u_list.append(u)

    # # print(u_list[-1].view(n-1, n-1))
    # print(*[th.norm(u_list[i] - u_t).item() for i in range(0, len(u_list), 10)])

  
