import torch as th
import math


def separate(A):
    return th.diag(A), -th.tril(A, -1), -th.triu(A, 1)

def smooth(A, w=2.0/3, type='jacobi'):
    D, L, U = separate(A)
    if type == 'jacobi':
        Di = (1 / D)
        T, c_ = Di * (L + U), lambda b: Di * b
    # elif type == 'gauss_seidel':  # todo
    #     DLi = th.inverse(th.diag_embed(D) - L)
    #     T, c_ = DLi @ (L + U), lambda b: DLi @ b
    else:
        T, c_ = None, None

    def method(x, b, m):
        c = c_(b)
        for _ in range(m):
            x = w * (T @ x + c) + (1 - w) * x 
        return x
    return method


if __name__ == "__main__":
    from grid import tridiag

    n = 16
    A, b = tridiag(-1, 2, -1, n, device='cpu'), th.randn(n)
    x = th.inverse(A) @ b

    smooth_method = smooth(A, w=2.0/3)
    x_p = smooth_method(th.zeros(n), b, 10)
    # print(x)


