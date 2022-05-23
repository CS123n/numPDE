import torch as th

# from smooth import smooth
from grid import Transform


class MultiGrid():
    def __init__(self, device):
        # self.A, self.n = laplace(n, device), n
        # self.smooth_method = {item: smooth(self.A[item]) for item in self.A}
        self.transform = Transform(device)

    def __call__(self, u, f, m1=1, m2=1, n=None):
        return self.compute(u, f, m1=m1, m2=m2, n=n)

    def compute(self, u, f, m1, m2, n):
        tran = self.transform
        n = n if n else self.n

        u = tran.smooth(u, f, m1, n)
        if n == 4:
            # return th.inverse(A) @ f  # better?
            u = tran.smooth(u, f, m2, n)
            return u

        r = f - tran.laplace(u, n)
        r2 = tran.restriction(r, n)
        e2 = self.compute(th.zeros_like(r2), r2, m1=m1, m2=m2, n=n//2)  # attention!
        # print(n, e2.shape, u.shape)
        u = u + tran.interpolation(e2, n)
        u = tran.smooth(u, f, m2, n)

        return u


class FullMultiGrid(MultiGrid):
    def __call__(self, f, m1=1, m2=1, n=None, r=20):
        return self.compute_full(f, m1, m2, n=n, r=r)

    def compute_full(self, f, m1, m2, n, r=20):
        n = n if n else self.n
        if n == 4:
            return self.compute(th.zeros_like(f), f, m1=m1, m2=m2, n=n)

        f2 = self.transform.restriction(f, n)
        u2 = self.compute_full(f2, m1=m1, m2=m2, n=n//2)

        u = self.transform.interpolation(u2, n)
        for _ in range(r):
            u = self.compute(u, f, m1=m1, m2=m2, n=n)

        return u



# class TwoGrid():
#     def __init__(self, n):
#         self.A, self.n = laplace(n), n
#         self.smooth_method = {item: smooth(self.A[item]) for item in self.A}

#     def __call__(self, u, f, m1, m2):
#         self.compute(u, f, m1, m2)

#     def compute(self, u, f, m1, m2):
#         A, n = self.A, self.n
#         u = self.smooth_method(u, f, m1)
#         r = f - A[n] @ u
#         r2 = restriction(r)
#         e2 = th.inverse(A[n // 2]) @ r2
#         u = u + interpolation(e2)
#         u = self.smooth_method(u, f, m2)

#         return u
    


if __name__ == "__main__":
    
    n = 8
    method = MultiGrid(8)
    

