import torch as th
import torch.nn.functional as F
import torch.distributed as dist
# from functorch import vmap
import time
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
        gridx, gridy = th.meshgrid(x_[direction[0]+index[0]*n:direction[0]+(index[0]+1)*n], 
                                   x_[direction[1]+index[1]*n:direction[1]+(index[1]+1)*n], indexing='ij')
        
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
            u_ = self.cov_s(u.view(1, 1, n-1, n-1)).view(-1) + b / (n ** 2 * self.diag)
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


class Transform_v2(Transform):
    def __init__(self, index, p, device):
        super().__init__(device)
        self.p = p
        self.index = index
        self.direction = (index < p // 2).int()
        d_list = [th.tensor([-1, 0]), th.tensor([1, 0]), th.tensor([0, -1]), th.tensor([0, 1])]
        self.d_list = [((index + item) < p // 2).int() for item in d_list]

    @ th.no_grad()
    def match(self, u_, u, a, m_type='id'):
        index = self.index
        direction = self.direction
        p = self.p

        correction1, correction2 = th.zeros_like(u_[0, :]), th.zeros_like(u_[0, :])  
        if m_type == 'id':
            task1 = dist.isend(u[self.d_list[0][0] - direction[0], :], (index[0] + index[1] * p - 1) % p ** 2)  # attention!
            task2 = dist.isend(u[self.d_list[1][0] - direction[0] - 1, :], (index[0] + index[1] * p + 1) % p ** 2)
        elif m_type == 're':
            task1 = dist.isend(u[self.d_list[0][0] - direction[0], direction[1]::2].contiguous(), (index[0] + index[1] * p - 1) % p ** 2)
            task2 = dist.isend(u[self.d_list[1][0] - direction[0] - 1, direction[1]::2].contiguous(), (index[0] + index[1] * p + 1) % p ** 2)
        task3 = dist.irecv(correction1, (index[0] + index[1] * p - 1) % p**2)
        task4 = dist.irecv(correction2, (index[0] + index[1] * p + 1) % p**2)
        task1.wait()
        task2.wait()
        task3.wait()
        task4.wait()

        if index[0] != 0:
            u_[0, :] += correction1 * a
        if index[0] != p-1:
            u_[-1, :] += correction2 * a

        correction3, correction4 = th.zeros_like(u_[0, :]), th.zeros_like(u_[0, :])  
        if m_type == 'id':
            task1 = dist.isend(u[:, self.d_list[2][1] - direction[1]].contiguous(), (index[0] + index[1] * p - p) % p ** 2)
            task2 = dist.isend(u[:, self.d_list[3][1] - direction[1] - 1].contiguous(), (index[0] + index[1] * p + p) % p ** 2)
        elif m_type == 're':
            task1 = dist.isend(u[direction[0]::2, self.d_list[2][1] - direction[1]].contiguous(), (index[0] + index[1] * p - p) % p ** 2)
            task2 = dist.isend(u[direction[0]::2, self.d_list[3][1] - direction[1] - 1].contiguous(), (index[0] + index[1] * p + p) % p ** 2)
        task3 = dist.irecv(correction4, (index[0] + index[1] * p - p) % p ** 2)  # attention!
        task4 = dist.irecv(correction3, (index[0] + index[1] * p + p) % p ** 2)
        task1.wait()
        task2.wait()
        task3.wait()
        task4.wait()

        if index[1] != 0:
            u_[:, 0] += correction3 * a
        if index[1] != p-1:
            u_[:, -1] += correction4 * a

        return u_

    @ th.no_grad()
    def laplace(self, u, n):  # n*n -> n*n
        u_ = self.cov_l(u.view(1, 1, n, n)) * (n*self.p) ** 2
        u = self.match(u_.view(n, n), u.view(n, n), - (n * self.p) ** 2)
        return u.view(-1)

    @ th.no_grad()
    def smooth(self, u, b, m, n):  # n*n -> n*n
        for _ in range(m):
            u_ = self.cov_s(u.view(1, 1, n, n)).view(-1) 
            u_ = self.match(u_.view(n, n), u.view(n, n), 0.25).view(-1)
            u_ = u_ + b / ((n*self.p) ** 2 * self.diag)
            u = self.w * u_ + (1 - self.w) * u.view(-1)
        return u.view(-1)

    @ th.no_grad()
    def restriction(self, u, n):  # n*n -> n/2*n/2
        direction = self.direction
        pad = (1 - direction[0].item(), direction[0].item(), 1 - direction[1].item(), direction[1].item())
        u_ = F.pad(u.view(n, n), pad=pad)  # better?
        u_ = self.cov_r(u_.view(1, 1, n+1, n+1))
        u = self.match(u_.view(n//2, n//2), u.view(n, n), 0.125, m_type='re')
        return u.view(-1)

    @ th.no_grad()
    def interpolation(self, u, n):  # n/2*n/2 -> n*n
        direction = self.direction
        pad = (1 - direction[0].item(), direction[0].item(), 1 - direction[1].item(), direction[1].item())
        u_ = th.kron(u.view(n//2, n//2), self.ones)
        u_ = self.cov_i(u_.view(1, 1, n, n))
        u = u_.view(n+1, n+1)[0 + pad[0]:n+1-pad[1], 0+pad[2]:n+1-pad[3]]  # attention!
        # u = F.pad(u, (1, 1, 1, 1), 'constant', 0)  # mpi
        return u.reshape(-1)


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
    from multigrid import MultiGrid
    from input.func import origin_func, target_func
    import os

    n, p = 8, 1
    device = th.device('cpu')
    dist.init_process_group(backend='gloo')

    device = th.device(device)
    rank = os.environ.get("LOCAL_RANK")
    rank = 0 if rank is None else int(rank)
    index = th.tensor([rank % p, rank // p], device=device)
    
    # MG_method = MultiGrid(index, p, device)
    cond_method = condition(n, index, p, device)
    b = cond_method(origin_func, target_func)  # condition(origin_func)
    if p > 1:
        transform = Transform_v2(index, p, device)
    else:
        transform = Transform(device)
    b = transform.interpolation(b.view(-1), 2*n)
    w = 2*n - 1 if p == 1 else 2*n
    if rank == 0:
        print(b.view(w, w))
    
    
    # A = laplace(n, device)[n]
    # transform = Transform(device)

    # v = th.rand((n-1)**2)
    # v1 = A @ v
    # print(v1.view(n-1, n-1))

    # v2 = transform.laplace(v, n)
    # print(v2.view(n-1, n-1))

    # print((v1 - v2).view(n-1, n-1))

    # print('------------------------------------------------')

    # from smooth import separate, smooth

    # smooth_method = smooth(A)
    # b = th.rand((n-1)**2)

    # D, L, U = separate(A)
    # T = (1 / D) * (L + U)
    # print(T)
    # print((1 / D), b, D)

    # v1 = T @ v.view(-1) + (1 / D) * b
    # v1 = 2.0 / 3 * v1 + 1.0 / 3 * v
    # print(v1.view(n-1, n-1))

    # v2 = smooth_method(v, b, 1)
    # print(v2.view(n-1, n-1))

    # v3 = transform.smooth(v, b, 1, n)
    # print(v3.view(n-1, n-1))

    # # print((v1 - v2).view(n-1, n-1))

    # print('------------------------------------------------')

    # v = th.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=th.float)
    # print(v, v.shape)

    # v_ = transform.laplace(v, n)
    # print(v_, v_.shape)

    # v = transform.interpolation(v, 2*n)
    # print(v.view(2*n-1, 2*n-1), v.shape)

    # v = transform.restriction(v, 2*n)
    # print(v, v.shape)

    # v = interpolation(v, n)
    # print(v, v.shape)
    # print('------------------------------------------------')

    # from input.func import origin_func, target_func

    # cond = condition(n)
    # b, u = cond(origin_func, target_func)
    # print(b)
    # print(u)