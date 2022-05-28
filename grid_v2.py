import torch as th
import torch.nn.functional as F
import torch.distributed as dist
import sys

from grid import Transform


class Transform_v2(Transform):
    def __init__(self, index, p, device):
        super().__init__(device)

        self.cov_r = th.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, device=device)
        weight = 0.125 * th.tensor([[[[0, 1, 0], [1, 4, 1], [0, 1, 0]]]], device=device)  # attention!
        self.cov_r.weight = th.nn.Parameter(weight)  

        self.index = index
        self.rank, self.p = index[0] + index[1] * p, p
        self.adjustment = (index < p // 2).int()
        ad_list = [th.tensor([-1, 0]), th.tensor([1, 0]), th.tensor([0, -1]), th.tensor([0, 1])]
        self.ad_list = [((index + item) < p // 2).int() for item in ad_list]

    @ th.no_grad()
    def match(self, u_, u, a):
        index = self.index
        rank, p = self.rank, self.p
        adjustment, ad_list = self.adjustment, self.ad_list

        correction1, correction2 = th.zeros_like(u[0, :]), th.zeros_like(u[0, :])  
        task1 = dist.isend(u[ad_list[0][0] - adjustment[0], :], (rank - 1) % p ** 2, tag=0)  # attention!
        task2 = dist.isend(u[ad_list[1][0] - adjustment[0] - 1, :], (rank + 1) % p ** 2, tag=1)
        task3 = dist.irecv(correction1, (rank - 1) % p**2, tag=1)
        task4 = dist.irecv(correction2, (rank + 1) % p**2, tag=0)

        dist.barrier()

        if index[0] != 0:
            u_[0, :] += correction1 * a
        if index[0] != p-1:
            u_[-1, :] += correction2 * a

        dist.barrier()

        correction3, correction4 = th.zeros_like(u[0, :]), th.zeros_like(u[0, :])
        task1 = dist.isend(u[:, ad_list[2][1] - adjustment[1]].contiguous(), (rank - p) % p ** 2, tag=0)
        task2 = dist.isend(u[:, ad_list[3][1] - adjustment[1] - 1].contiguous(), (rank + p) % p ** 2, tag=1)
        task3 = dist.irecv(correction3, (rank - p) % p ** 2, tag=1)  # attention!
        task4 = dist.irecv(correction4, (rank + p) % p ** 2, tag=0)
        
        dist.barrier()

        if index[1] != 0:
            u_[:, 0] += correction3 * a
        if index[1] != p-1:
            u_[:, -1] += correction4 * a

        dist.barrier()

        return u_

    @ th.no_grad()
    def collection(self, u_, n):  # 1 -> p*p
        rank, p = self.rank, self.p

        if rank == 0:
            u_list = [th.zeros_like(u_) for _ in range(p**2)]
            dist.gather(u_, u_list)
            u_list = th.stack(u_list, dim=0)  # attention!
            u = th.zeros(n*p, n*p)
            for i in range(p**2):
                index = (i % p, i // p)
                u[index[0]*n:index[0]*n+n, index[1]*n:index[1]*n+n] = u_list[i].view(n, n)
            u = u[th.arange(n*p)!=n*p//2][:, th.arange(n*p)!=n*p//2]

            return u.view(-1)
        else:
            dist.gather(u_)

            return None

    @ th.no_grad()
    def distribution(self, u, n):
        rank, p = self.rank, self.p

        u_ = th.zeros(n, n)
        if rank == 0:
            ex_list = list(range(n*p//2)) + list(range(n*p//2-1, n*p-1))
            u = u.view(n*p-1, n*p-1)[ex_list][:, ex_list].view(n*p, n*p)
            
            u_list = []
            for i in range(p**2):
                index = (i % p, i // p)
                u_list.append(u[index[0]*n:index[0]*n+n, index[1]*n:index[1]*n+n].contiguous())
            dist.scatter(u_, u_list)
        else:
            dist.scatter(u_)
            
        return u_

    @ th.no_grad()
    def correction(self, u, n):
        index = self.index
        rank, p = self.rank, self.p
        u = u.view(n, n)
        if index[0] == p//2:
            dist.send(u[0, :], rank - 1)
        elif index[0] == p//2-1:
            u_ = th.zeros_like(u[-1, :])
            dist.recv(u_, rank + 1)
            u[-1, :] = u_
        
        if index[1] == p//2:
            dist.send(u[:, 0].contiguous(), rank - p)
        elif index[1] == p//2-1:
            u_ = th.zeros_like(u[:, -1])
            dist.recv(u_, rank + p)
            u[:, -1] = u_

        return u.view(-1)

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
    def restriction(self, u, n):  # n*n -> n/2*n/2  # problem?
        adjustment = self.adjustment
        u_ = self.cov_r(u.view(1, 1, n, n))
        u = self.match(u_.view(n, n), u.view(n, n), 0.125)
        u = u[adjustment[0]::2, adjustment[1]::2]

        return u.reshape(-1)

    @ th.no_grad()
    def interpolation(self, u, n):  # n/2*n/2 -> n*n
        adjustment = self.adjustment

        pad = (1 - adjustment[0].item(), adjustment[0].item(), 1 - adjustment[1].item(), adjustment[1].item())
        u_ = th.kron(u.view(n//2, n//2), self.ones)
        u_ = self.cov_i(u_.view(1, 1, n, n))
        u_ = self.match(u_.view(n+1, n+1), u_.view(n+1, n+1), 1)  # attention!
        u = u_.view(n+1, n+1)[0 + pad[0]:n+1-pad[1], 0+pad[2]:n+1-pad[3]]  # attention!

        return u.reshape(-1)



if __name__ == "__main__":

    from grid import condition
    from input.func import origin_func, target_func
    import sys
    import os


    n, p = 2, 4  # 4, 4
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

    if rank == 0:
        print(b.view(n*p-1, n*p-1))

    for _ in range(1):
        b_v2_ = transform_v2.collection(b_v2, n)
        b_v2 = transform_v2.distribution(b_v2_, n)

    # if rank == 0:
    #     print(b_v2.view(n, n))

    # u = th.zeros_like(b)
    # u_v2 = th.zeros_like(b_v2)

    # u = transform.smooth(u, b, 1, n*p)
    # u_v2 = transform_v2.smooth(u_v2, b_v2, 1, n)

    # # r = b - transform.laplace(u, n*p)
    # # r_v2 = b_v2 - transform_v2.laplace(u_v2, n)

    # for _ in range(20):

    #     r = transform.restriction(b, n*p)
    #     r_v2 = transform_v2.restriction(b_v2, n)

    #     # r = transform.restriction(r, n*p//2)
    #     # r_v2 = transform_v2.restriction(r_v2, n//2)

    #     # r = transform.interpolation(r, n*p//2)
    #     # r_v2 = transform_v2.interpolation(r_v2, n//2)

    #     r = transform.interpolation(r, n*p)
    #     r_v2 = transform_v2.interpolation(r_v2, n)


    # r2 = transform.smooth(r, r, 2, n*p//2)
    # r2_v2 = transform_v2.smooth(r_v2, r_v2, 2, n//2)

    # r = transform.interpolation(r, n*p)
    # r_v2 = transform_v2.interpolation(r_v2, n)

    # r2 = transform.interpolation(r2, n*p)
    # r2_v2 = transform_v2.interpolation(r2_v2, n)

    # if rank == 0:
    #     print('rank 0:')
    #     print(u.view(n*p-1, n*p-1)[:n, :n])
    #     print(u_v2.view(n, n))
    #     print(u.view(n*p-1, n*p-1)[:n, :n] - u_v2.view(n, n))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(u.view(n*p-1, n*p-1)[-n:, -n:])
    #     print(u_v2.view(n, n))
    #     print(u.view(n*p-1, n*p-1)[-n:, -n:] - u_v2.view(n, n))

    # dist.barrier()

    # ww = 1
    # if rank == 0:
    #     print('rank 0:')
    #     print(r.view(n*p//ww-1, n*p//ww-1)[:n//ww, :n//ww])
    #     print(r_v2.view(n//ww, n//ww))
    #     print(r.view(n*p//ww-1, n*p//ww-1)[:n//ww, :n//ww] - r_v2.view(n//ww, n//ww))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(r.view(n*p//ww-1, n*p//ww-1)[-n//ww:, -n//ww:])
    #     print(r_v2.view(n//ww, n//ww))
    #     print(r.view(n*p//ww-1, n*p//ww-1)[-n//ww:, -n//ww:] - r_v2.view(n//ww, n//ww))

    # dist.barrier()

    # if rank == 0:
    #     print('rank 0:')
    #     print(r2.view(n*p//2-1, n*p//2-1)[:n//2, :n//2])
    #     print(r2_v2.view(n//2, n//2))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(r2.view(n*p//2-1, n*p//2-1)[-n//2:, -n//2:])
    #     print(r2_v2.view(n//2, n//2))

    # dist.barrier()

    # result = transform.laplace(b, n*p)
    # result_v2 = transform_v2.laplace(b_v2, n)

    # if rank == 0:
    #     print('rank 0:')
    #     print(result.view(n*p-1, n*p-1)[:n, :n])
    #     print(result_v2.view(n, n))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(result.view(n*p-1, n*p-1)[-n:, -n:])
    #     print(result_v2.view(n, n))

    # dist.barrier()


    # result = transform.smooth(result, b, 10, n*p)
    # result_v2 = transform_v2.smooth(result_v2, b_v2, 10, n)

    # if rank == 0:
    #     print('rank 0:')
    #     print(result.view(n*p-1, n*p-1)[:n, :n])
    #     print(result_v2.view(n, n))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(result.view(n*p-1, n*p-1)[-n:, -n:])
    #     print(result_v2.view(n, n))

    # dist.barrier()
    
    # result = transform.restriction(b, n*p)
    # result_v2 = transform_v2.restriction(b_v2, n)

    # if rank == 0:
    #     print('rank 0:')
    #     print(result.view(n*p//2-1, n*p//2-1)[:n//2, :n//2])
    #     print(result_v2.view(n//2, n//2))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(result.view(n*p//2-1, n*p//2-1)[-n//2:, -n//2:])
    #     print(result_v2.view(n//2, n//2))

    # dist.barrier()

    # result = transform.interpolation(result, n*p)
    # result_v2 = transform_v2.interpolation(result_v2, n)

    # if rank == 0:
    #     print('rank 0:')
    #     print(result.view(n*p-1, n*p-1)[:n, :n])
    #     print(result_v2.view(n, n))

    # dist.barrier()

    # if rank == p*p-1:
    #     print('rank p*p-1:')
    #     print(result.view(n*p-1, n*p-1)[-n:, -n:])
    #     print(result_v2.view(n, n))