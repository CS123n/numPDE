import torch as th

def origin_func(x, y):
    # return - 2 * th.exp(x + y) * (th.cos(x) * th.cos(y) - th.sin(x) * th.sin(y))
    return - 2 * th.exp(x + y) * th.cos(x + y)

def target_func(x, y):
    return th.exp(x + y) * th.cos(x) * th.sin(y)
