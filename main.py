import torch as th
import argparse
import time

from grid import laplace, condition
from multigrid import MultiGrid, FullMultiGrid
from input.func import origin_func, target_func
from smooth import smooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', '-n', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    n = args.n
    device = th.device(args.device)
    
    start = time.time()
    A = laplace(n, device)[n]
    MG_method = MultiGrid(n, device)
    b_method = condition(n, device)
    b, u_t = b_method(origin_func, target_func)  # condition(origin_func)
    print(time.time() - start)

    # start = time.time()
    # # u_c = th.inverse(A) @ b
    # u_c = th.linalg.solve(A, b)
    # print(time.time() - start)
    
    # print(th.norm(u_c - u_t).item())

    start = time.time()
    u = th.zeros((n-1)**2, device=device)
    u_list = []
    for _ in range(20):
        u = MG_method(u, b, n=n)
        u_list.append(u)
    print(time.time() - start)

    # print(u_t.view(n-1, n-1))
    e_list = [th.norm(u_list[i] - u_t) for i in range(len(u_list))]
    o_list = [th.log2(e_list[i]) - th.log2(e_list[i+1]) for i in range(len(e_list)-1)]
    print(e_list)
    print(o_list)

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

  
