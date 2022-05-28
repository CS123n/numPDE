import torch as th
import argparse
import time

from grid import laplace, condition
from multigrid import MultiGrid, FullMultiGrid
from input.func import origin_func, target_func
from smooth import smooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n', '-n', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = th.device(args.device)
    e_list = []
    for n in (8, 16, 32, 64, 128):
    
        A = laplace(n, device)[n]
        MG_method = MultiGrid(n, device)
        b_method = condition(n, device)
        b, u_t = b_method(origin_func, target_func)  # condition(origin_func)

        u = th.zeros((n-1)**2, device=device)
        for _ in range(n):
            u = MG_method(u, b, n=n)
        e_list.append(th.norm(u - u_t) / n)

    o_list = [th.log2(e_list[i]) - th.log2(e_list[i+1]) for i in range(len(e_list)-1)]
    print(e_list)
    print(o_list)