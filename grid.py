import torch as th
import torch.nn.functional as F
# from functorch import vmap
import sys


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


def condition(n, device):  # n-1*n-1
    x = th.linspace(0, 1, n+1, device=device)
    gridx, gridy = th.meshgrid(x, x, indexing='ij')

    @ th.no_grad()
    def method(origin_func, target_func):
        b = origin_func(gridx[1:-1, 1:-1], gridy[1:-1, 1:-1])
        u = target_func(gridx[1:-1, 1:-1], gridy[1:-1, 1:-1])
        # b = vmap(origin_func)(grid[:, 1:-1, 1:-1]).view(n-1, n-1)

        b[0, :] += target_func(gridx[0, 1:-1], gridy[0, 1:-1]) * n ** 2
        b[:, 0] += target_func(gridx[1:-1, 0], gridy[1:-1, 0]) * n ** 2
        b[-1, :] += target_func(gridx[-1, 1:-1], gridy[-1, 1:-1]) * n ** 2
        b[:, -1] += target_func(gridx[1:-1, -1], gridy[1:-1, -1]) * n ** 2
 
        return b.view(-1), u.view(-1)
    return method


class Transform():
    def __init__(self, device):
        self.cov_r = th.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False, device=device)
        weight = 0.125 * th.tensor([[[[0, 1, 0], [1, 4, 1], [0, 1, 0]]]], device=device)
        self.cov_r.weight = th.nn.Parameter(weight)  # attention!

        self.cov_i = th.nn.Conv2d(1, 1, kernel_size=2, bias=False, device=device)
        self.cov_i.weight = th.nn.Parameter(0.25 * th.ones(1, 1, 2, 2, device=device))

        self.ones = th.ones(2, 2, device=device)

    @ th.no_grad()
    def restriction(self, u, n):  # n-1*n-1 -> n/2-1*n/2-1
        # n = u.shape[0] + 1
        return self.cov_r(u.view(1, 1, n-1, n-1)).view(-1)

    @ th.no_grad()
    def interpolation(self, u, n):  # n/2-1*n/2-1 -> n-1*n-1
        # n = (u.shape[0] + 1) * 2
        u = th.kron(u.view(n//2-1, n//2-1), self.ones)
        u = F.pad(u, (1, 1, 1, 1), 'constant', 0)  # mpi
        return self.cov_i(u.view(1, 1, n, n)).view(-1)


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
    n = 4

    # print(laplace(n))
    # print('------------------------------------------------')

    v = th.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    print(v, v.shape)

    v = interpolation(v, 2*n)
    print(v.view(2*n-1, 2*n-1), v.shape)

    v = restriction(v, 2*n)
    print(v, v.shape)

    # v = interpolation(v, n)
    # print(v, v.shape)
    # print('------------------------------------------------')

    # from input.func import origin_func, target_func

    # cond = condition(n)
    # b, u = cond(origin_func, target_func)
    # print(b)
    # print(u)