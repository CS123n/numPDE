import torch as th
import torch.nn.functional as F
import torch.distributed as dist
# from functorch import vmap
import time
import sys
import os


def condition(n, index, p, device):  # n-1*n-1
    if p == 1:
        x = th.linspace(0, 1, n+1, device=device)
        gridx, gridy = th.meshgrid(x, x, indexing='ij')

        @ th.no_grad()
        def method(origin_func, target_func):
            b = origin_func(gridx[1:-1, 1:-1], gridy[1:-1, 1:-1])
            # u = target_func(gridx[1:-1, 1:-1], gridy[1:-1, 1:-1])
            # b = vmap(origin_func)(grid[:, 1:-1, 1:-1]).view(n-1, n-1)

            b[0, :] += target_func(gridx[0, 1:-1], gridy[0, 1:-1]) * n ** 2
            b[:, 0] += target_func(gridx[1:-1, 0], gridy[1:-1, 0]) * n ** 2
            b[-1, :] += target_func(gridx[-1, 1:-1], gridy[-1, 1:-1]) * n ** 2
            b[:, -1] += target_func(gridx[1:-1, -1], gridy[1:-1, -1]) * n ** 2
    
            return b.view(-1)
        return method
    else:
        direction = index < p // 2
        x_ = th.linspace(0, 1, n*p+1, device=device)
        gridx, gridy = th.meshgrid(x_[direction[0]+index[0]*n:direction[0]+index[0]*n+n], 
                                   x_[direction[1]+index[1]*n:direction[1]+index[1]*n+n], indexing='ij')
        
        @ th.no_grad()
        def method(origin_func, target_func):
            b = origin_func(gridx, gridy)

            if index[0] == 0:
                b[0, :] += target_func(th.zeros(n, device=device), gridy[0, :]) * (n*p) ** 2
            elif index[0] == p-1:
                b[-1, :] += target_func(th.ones(n, device=device), gridy[-1, :]) * (n*p) ** 2

            if index[1] == 0:
                b[:, 0] += target_func(gridx[:, 0], th.zeros(n, device=device)) * (n*p) ** 2
            elif index[1] == p-1:
                b[:, -1] += target_func(gridx[:, -1], th.ones(n, device=device)) * (n*p) ** 2

            return b.view(-1)
        return method


class Transform():
    def __init__(self, device):
        self.cov_l = th.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, device=device)
        weight = th.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=th.float, device=device)
        self.cov_l.weight = th.nn.Parameter(weight)

        self.w = 2.0 / 3
        self.diag = weight[..., 1, 1]
        self.cov_s = th.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, device=device)
        weight, weight[..., 1, 1] = - weight / self.diag, 0
        self.cov_s.weight = th.nn.Parameter(weight)

        self.cov_r = th.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False, device=device)
        weight = 0.125 * th.tensor([[[[0, 1, 0], [1, 4, 1], [0, 1, 0]]]], device=device)  # attention!
        self.cov_r.weight = th.nn.Parameter(weight)  

        self.cov_i = th.nn.Conv2d(1, 1, kernel_size=2, padding=1, bias=False, device=device)
        self.cov_i.weight = th.nn.Parameter(0.25 * th.ones(1, 1, 2, 2, device=device))

        self.ones = th.ones(2, 2, device=device)

    @ th.no_grad()
    def laplace(self, u, n):  # n-1*n-1 -> n-1*n-1
        return self.cov_l(u.view(1, 1, n-1, n-1)).view(-1) * n ** 2

    @ th.no_grad()
    def smooth(self, u, b, m, n):  # n-1*n-1 -> n-1*n-1
        for _ in range(m):
            u_ = self.cov_s(u.view(1, 1, n-1, n-1)).view(-1)
            u_ = u_ + b / (n ** 2 * self.diag)
            u = self.w * u_ + (1 - self.w) * u.view(-1)
        return u

    @ th.no_grad()
    def restriction(self, u, n):  # n-1*n-1 -> n/2-1*n/2-1
        return self.cov_r(u.view(1, 1, n-1, n-1)).view(-1)

    @ th.no_grad()
    def interpolation(self, u, n):  # n/2-1*n/2-1 -> n-1*n-1
        u = th.kron(u.view(n//2-1, n//2-1), self.ones)
        # u = F.pad(u, (1, 1, 1, 1), 'constant', 0)  # mpi
        return self.cov_i(u.view(1, 1, n-2, n-2)).view(-1)


def tridiag(a, b, c, n, device):  # n^2*n^2
    return th.diag_embed(th.ones(n-1, device=device), -1) * a + th.diag_embed(th.ones(n, device=device), 0) * b + \
        th.diag_embed(th.ones(n-1, device=device), 1) * c


def laplace(n, device):  # {h: n-1^2*n-1^2}
    A_dict, i = {}, 4
    while i <= n:
        T, I, B = tridiag(-1, 4, -1, i-1, device), th.eye(i-1, device=device), tridiag(-1, 0, -1, i-1, device)
        A = th.kron(I, T) + th.kron(B, I)
        A_dict[i] = A * i ** 2  # attention!
        i = i * 2
    return A_dict


# # cov_c = th.nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
# # cov_c.weight = th.nn.Parameter(tridiag(-1, 4, -1, 3).view(1, 1, 3, 3) * n ** 2)
# # print(cov_c.weight)
# T, I, B = tridiag(-1, 4, -1, n+1), th.eye(n+1), tridiag(-1, 0, -1, n+1)
# A = th.kron(I, T) + th.kron(B, I) 
# A = A * n ** 2
# # print(A.shape)

# T, I, B = tridiag(-1, 4, -1, n-1), th.eye(n-1), tridiag(-1, 0, -1, n-1)
# A1 = th.kron(I, T) + th.kron(B, I) 
# A1 = A1 * n ** 2


if __name__ == "__main__":
    n, p = 8, 1
    device = th.device('cpu')

    
    A = laplace(n, device)[n]
    transform = Transform(device)

    v = th.rand((n-1)**2)
    v1 = A @ v
    print(v1.view(n-1, n-1))

    v2 = transform.laplace(v, n)
    print(v2.view(n-1, n-1))

    print((v1 - v2).view(n-1, n-1))

    print('------------------------------------------------')

    from smooth import separate, smooth

    smooth_method = smooth(A)
    b = th.rand((n-1)**2)

    D, L, U = separate(A)
    T = (1 / D) * (L + U)
    print(T)
    print((1 / D), b, D)

    v1 = T @ v.view(-1) + (1 / D) * b
    v1 = 2.0 / 3 * v1 + 1.0 / 3 * v
    print(v1.view(n-1, n-1))

    v2 = smooth_method(v, b, 1)
    print(v2.view(n-1, n-1))

    v3 = transform.smooth(v, b, 1, n)
    print(v3.view(n-1, n-1))

    # print((v1 - v2).view(n-1, n-1))

    print('------------------------------------------------')

    v = th.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=th.float)
    print(v, v.shape)

    v_ = transform.laplace(v, n)
    print(v_, v_.shape)

    v = transform.interpolation(v, 2*n)
    print(v.view(2*n-1, 2*n-1), v.shape)

    v = transform.restriction(v, 2*n)
    print(v, v.shape)

    # v = interpolation(v, n)
    # print(v, v.shape)
    # print('------------------------------------------------')

    # from input.func import origin_func, target_func

    # cond = condition(n)
    # b, u = cond(origin_func, target_func)
    # print(b)
    # print(u)