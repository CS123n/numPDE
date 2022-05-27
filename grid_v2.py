import torch as th
import torch.nn.functional as F
import torch.distributed as dist

from grid import Transform


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
            task1 = dist.isend(u[self.d_list[0][0] - direction[0], :], (index[0] + index[1] * p - 1) % p ** 2, tag=0)  # attention!
            task2 = dist.isend(u[self.d_list[1][0] - direction[0] - 1, :], (index[0] + index[1] * p + 1) % p ** 2, tag=1)
        elif m_type == 're':
            task1 = dist.isend(u[self.d_list[0][0] - direction[0], direction[1]::2].contiguous(), (index[0] + index[1] * p - 1) % p ** 2, tag=0)
            task2 = dist.isend(u[self.d_list[1][0] - direction[0] - 1, direction[1]::2].contiguous(), (index[0] + index[1] * p + 1) % p ** 2, tag=1)
        task3 = dist.irecv(correction1, (index[0] + index[1] * p - 1) % p**2, tag=1)
        task4 = dist.irecv(correction2, (index[0] + index[1] * p + 1) % p**2, tag=0)
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
            task1 = dist.isend(u[:, self.d_list[2][1] - direction[1]].contiguous(), (index[0] + index[1] * p - p) % p ** 2, tag=0)
            task2 = dist.isend(u[:, self.d_list[3][1] - direction[1] - 1].contiguous(), (index[0] + index[1] * p + p) % p ** 2, tag=1)
        elif m_type == 're':
            task1 = dist.isend(u[direction[0]::2, self.d_list[2][1] - direction[1]].contiguous(), (index[0] + index[1] * p - p) % p ** 2, tag=0)
            task2 = dist.isend(u[direction[0]::2, self.d_list[3][1] - direction[1] - 1].contiguous(), (index[0] + index[1] * p + p) % p ** 2, tag=1)
        task3 = dist.irecv(correction3, (index[0] + index[1] * p - p) % p ** 2, tag=1)  # attention!
        task4 = dist.irecv(correction4, (index[0] + index[1] * p + p) % p ** 2, tag=0)
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
    def collection():
        pass

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



if __name__ == "__main__":

    from grid import condition
    from input.func import origin_func, target_func
    import sys
    import os


    n, p = 4, 2  # 4, 4
    device = th.device('cpu')
    dist.init_process_group(backend='gloo')

    rank = os.environ.get("LOCAL_RANK")
    rank = 0 if rank is None else int(rank)
    index = th.tensor([rank % p, rank // p], device=device)

    transform = Transform(device)
    transform_v2 = Transform_v2(index, p, device)
    
    cond_method = condition(n*p, index, 1, device)
    b = cond_method(origin_func, target_func)

    cond_method_v2 = condition(n, index, p, device)
    b_v2 = cond_method_v2(origin_func, target_func)

    result = transform.laplace(b, n*p)
    result_v2 = transform_v2.laplace(b_v2, n)

    if rank == 0:
        print('rank 0:')
        print(result.view(n*p-1, n*p-1)[:n, :n])
        print(result_v2.view(n, n))

    dist.barrier()

    if rank == p*p-1:
        print('rank p*p-1:')
        print(result.view(n*p-1, n*p-1)[-n:, -n:])
        print(result_v2.view(n, n))

    dist.barrier()
    
    result = transform.restriction(b, n*p)
    result_v2 = transform_v2.restriction(b_v2, n)

    if rank == 0:
        print('rank 0:')
        print(result.view(n*p//2-1, n*p//2-1)[:n//2, :n//2])
        print(result_v2.view(n//2, n//2))

    dist.barrier()

    if rank == p*p-1:
        print('rank p*p-1:')
        print(result.view(n*p//2-1, n*p//2-1)[-n//2:, -n//2:])
        print(result_v2.view(n//2, n//2))

    dist.barrier()

    result = transform.interpolation(result, n*p)
    result_v2 = transform_v2.interpolation(result_v2, n)

    if rank == 0:
        print('rank 0:')
        print(result.view(n*p-1, n*p-1)[:n, :n])
        print(result_v2.view(n, n))

    dist.barrier()

    if rank == p*p-1:
        print('rank p*p-1:')
        print(result.view(n*p-1, n*p-1)[-n:, -n:])
        print(result_v2.view(n, n))